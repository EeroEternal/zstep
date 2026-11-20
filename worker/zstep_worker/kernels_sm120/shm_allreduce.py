# Copyright (c) 2026 zstep Contributors
#
# Permission is hereby granted, free of charge, to any person obtaining a copy
# of this software and associated documentation files (the "Software"), to deal
# in the Software without restriction, including without limitation the rights
# to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
# copies of the Software, and to permit persons to whom the Software is
# furnished to do so, subject to the following conditions:
#
# The above copyright notice and this permission notice shall be included in all
# copies or substantial portions of the Software.
#
# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
# IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
# FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
# AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
# LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
# OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
# SOFTWARE.

"""CUDA-Graph-safe host-mapped BF16 AllReduce & AllGather (`shm_v5`) for non-P2P (`CNS`) PCIe.

In multi-process TP=8 mode (`exchange_group is not None`), each rank allocates and first-touches
its own NUMA-local `/dev/shm/{name}_r{rank}` segment with 64-byte cacheline-isolated `(block, peer)`
arrival flags. Producers push flags via posted PCIe writes (`st.global.release.sys.b64`) into each
peer's local DRAM so consumers spin exclusively on their own local NUMA node (`ld.global.acquire.sys.b64`)
and read their own shard directly from local VRAM.
"""

from __future__ import annotations

import ctypes
import math
import mmap
import os
from pathlib import Path
import struct
import weakref

import torch
import torch.distributed as dist
from torch.utils.cpp_extension import load_inline

from zstep_worker.numa_pin import DeviceNumaTopology, pin_process_to_device_numa

NUM_RING_SLOTS = 4
NUM_PXB_RING_SLOTS = 2
MAX_RANKS = 8
MAX_BLOCKS = 64
SHM_V5_HEADER_BYTES = 131072
SHM_V5_ABORT_OFFSET = 98304
ARGMAX_SHM_ABORT_OFFSET = 128
DEFAULT_ONE_SHOT_MAX_VECS = 4096  # Up to 64 KiB (e.g. M <= 16 at H=2048 or M <= 6 at H=5120 BF16)
DEFAULT_PXB_MAX_ELEMENTS = 4096 * 5120  # Up to 40.0 MiB (~163.5 MiB 2-slot dual-regime CUDA-IPC HBM ring)

_CSRC_DIR = Path(__file__).resolve().parent / "csrc"
_CPP_SOURCE: str = (_CSRC_DIR / "shm_allreduce.cpp").read_text(encoding="utf-8")
_CUDA_SOURCE: str = (_CSRC_DIR / "shm_allreduce_kernels.cu").read_text(encoding="utf-8")

_EXTENSION = None


def _align_up(value: int, alignment: int) -> int:
    return (value + alignment - 1) & ~(alignment - 1)


def get_host_mapped_allreduce_extension():
    """Build or load the inline CUDA extension without initializing a collective."""
    global _EXTENSION
    if _EXTENSION is None:
        os.environ.setdefault("TORCH_CUDA_ARCH_LIST", "12.0")
        _EXTENSION = load_inline(
            name="zstep_shm_allreduce_sm120_v28_ext",
            cpp_sources=[_CPP_SOURCE],
            cuda_sources=[_CUDA_SOURCE],
            functions=[
                "register_mapped_host",
                "unregister_mapped_host",
                "alloc_ipc_buffer",
                "open_ipc_handle",
                "close_ipc_handle",
                "free_ipc_buffer",
                "launch_host_mapped_two_shot",
                "launch_shm_v5_allreduce",
                "launch_shm_v5_pxb_allreduce",
                "launch_shm_v5_allreduce_add_rmsnorm",
                "launch_shm_v5_pxb_allreduce_add_rmsnorm",
                "launch_shm_v5_allgather",
                "launch_mhc_post_collapse_rmsnorm",
                "launch_mhc_sinkhorn_stats",
                "launch_ar_mhc_cluster8",
                "launch_local_top2_bf16",
                "launch_tp8_argmax_gather_shm",
            ],
            extra_cflags=["-O3"],
            extra_cuda_cflags=[
                "-O3",
                "-gencode=arch=compute_120,code=sm_120",
            ],
            with_cuda=True,
            verbose=False,
        )
    return _EXTENSION


class _PerRankShmRingBase:
    """Shared lifecycle helper for per-rank NUMA-local `/dev/shm/{name}_r{r}` rings."""

    def _init_per_rank_shm(
        self,
        *,
        name: str,
        rank: int,
        world_size: int,
        total_bytes: int,
        exchange_group: dist.ProcessGroup,
        device: torch.device,
        extension,
    ) -> None:
        self._per_rank_paths: list[str] = [
            f"/dev/shm/{name}_r{peer}" for peer in range(world_size)
        ]
        self._per_rank_mappings: list[mmap.mmap] = []
        self._per_rank_host_addrs: list[int] = []
        self._own_shm_path: str = self._per_rank_paths[rank]
        self._own_mm: mmap.mmap | None = None

        own_fd = os.open(
            self._own_shm_path, os.O_CREAT | os.O_RDWR | os.O_TRUNC, 0o600
        )
        try:
            os.ftruncate(own_fd, total_bytes)
            own_mm = mmap.mmap(
                own_fd,
                total_bytes,
                flags=mmap.MAP_SHARED,
                prot=mmap.PROT_READ | mmap.PROT_WRITE,
            )
            self._own_mm = own_mm
        finally:
            os.close(own_fd)
        # First-touch every 4 KiB page from this rank's NUMA-pinned thread.
        mv = memoryview(own_mm)
        for off in range(0, total_bytes, 4096):
            mv[off] = 0
        mv.release()
        ctypes.memset(
            ctypes.addressof(ctypes.c_char.from_buffer(own_mm)),
            0,
            min(SHM_V5_HEADER_BYTES, total_bytes),
        )
        dist.barrier(group=exchange_group)

        dev_ptrs: list[int] = []
        for peer in range(world_size):
            if peer == rank:
                mm = own_mm
            else:
                pfd = os.open(self._per_rank_paths[peer], os.O_RDWR)
                try:
                    mm = mmap.mmap(
                        pfd,
                        total_bytes,
                        flags=mmap.MAP_SHARED,
                        prot=mmap.PROT_READ | mmap.PROT_WRITE,
                    )
                finally:
                    os.close(pfd)
            try:
                host_addr = ctypes.addressof(ctypes.c_char.from_buffer(mm))
                dev_ptr = extension.register_mapped_host(host_addr, total_bytes)
            except BaseException:
                if peer != rank and not mm.closed:
                    mm.close()
                raise
            self._per_rank_mappings.append(mm)
            self._per_rank_host_addrs.append(host_addr)
            dev_ptrs.append(dev_ptr)

        dist.barrier(group=exchange_group)
        # All peers have opened and mmapped every `/dev/shm/<name>_r<peer>`; unlink our
        # directory entry immediately so kernel tmpfs reclaims pages even on SIGKILL.
        try:
            os.unlink(self._own_shm_path)
        except FileNotFoundError:
            pass
        dist.barrier(group=exchange_group)
        self._rank_shm_ptrs: torch.Tensor = torch.tensor(
            dev_ptrs, dtype=torch.int64, device=device
        )

    def signal_abort(self) -> None:
        """Set the cacheline-isolated abort flag across all mapped per-rank SHM segments."""
        for mm in getattr(self, "_per_rank_mappings", ()):
            if not mm.closed and len(mm) >= SHM_V5_ABORT_OFFSET + 8:
                struct.pack_into("<Q", mm, SHM_V5_ABORT_OFFSET, 1)

    def _close_per_rank_shm(self, extension) -> None:
        host_addrs: list[int] = getattr(self, "_per_rank_host_addrs", [])
        for host_addr in host_addrs:
            extension.unregister_mapped_host(host_addr)
        host_addrs.clear()
        mappings: list[mmap.mmap] = getattr(self, "_per_rank_mappings", [])
        for mm in mappings:
            if not mm.closed:
                mm.close()
        mappings.clear()
        own_mm: mmap.mmap | None = getattr(self, "_own_mm", None)
        if own_mm is not None and not own_mm.closed:
            own_mm.close()
        self._own_mm = None
        own_shm_path: str | None = getattr(self, "_own_shm_path", None)
        if own_shm_path is not None:
            try:
                os.unlink(own_shm_path)
            except FileNotFoundError:
                pass


class HostMappedShmAllReduce(_PerRankShmRingBase):
    """Ordered host-mapped 4-slot ring BF16 AllReduce (`shm_v5` + optional CX8 `pxb` IPC)."""

    def __init__(
        self,
        *,
        name: str,
        rank: int,
        world_size: int,
        capacity_elements: int,
        blocks: int,
        threads: int,
        exchange_group: dist.ProcessGroup | None,
        create_shm: bool = False,
        one_shot_max_vecs: int = DEFAULT_ONE_SHOT_MAX_VECS,
        enable_pxb_p2p: bool = False,
        pxb_max_elements: int = DEFAULT_PXB_MAX_ELEMENTS,
        enable_pxb_bfp: int | bool = 0,
    ) -> None:
        if world_size < 2 or world_size > MAX_RANKS or not 0 <= rank < world_size:
            raise ValueError(f"rank={rank} / world_size={world_size} invalid (max {MAX_RANKS})")
        if capacity_elements <= 0 or capacity_elements % 8 != 0:
            raise ValueError("capacity_elements must be positive and divisible by 8")
        if not 1 <= blocks <= MAX_BLOCKS:
            raise ValueError(f"blocks must be in [1, {MAX_BLOCKS}]")
        if threads <= 0 or threads > 1024 or threads % 32 != 0:
            raise ValueError("threads must be a warp multiple in [32, 1024]")
        pxb_bfp_mode = int(enable_pxb_bfp)
        if pxb_bfp_mode not in (0, 1, 2):
            raise ValueError(f"enable_pxb_bfp must be 0, 1, or 2 (got {enable_pxb_bfp!r})")

        device = torch.device("cuda", torch.cuda.current_device())
        self.numa_topology: DeviceNumaTopology = pin_process_to_device_numa(device.index)
        self.name: str = name
        self.rank: int = rank
        self.world_size: int = world_size
        self.capacity_elements: int = capacity_elements
        self.blocks: int = blocks
        self.threads: int = threads
        self.one_shot_max_vecs: int = one_shot_max_vecs
        self.device: torch.device = device
        self._use_shm_v5: bool = exchange_group is not None
        self._exchange_group: dist.ProcessGroup | None = exchange_group
        self._closed: bool = False
        self._registered: bool = False
        self._owns_unlink: bool = False
        self._mapping: mmap.mmap | None = None
        self._shm_path: str | None = None
        self._pxb_enabled: bool = False
        self._my_hbm_ptr: int = 0
        self._pt_hbm_ptr: int = 0
        self.pxb_slot_stride_bytes: int = 0
        self.pxb_max_vecs: int = 0
        self._pxb_bfp: int = pxb_bfp_mode
        self._extension = get_host_mapped_allreduce_extension()

        try:
            if self._use_shm_v5:
                assert exchange_group is not None
                self.rank_stride_bytes = _align_up(capacity_elements * 2, 4096)
                self.data_base_offset = SHM_V5_HEADER_BYTES
                self.slot_stride_bytes = 2 * self.rank_stride_bytes
                self.total_bytes = _align_up(
                    self.data_base_offset + NUM_RING_SLOTS * self.slot_stride_bytes, 4096
                )
                self._init_per_rank_shm(
                    name=name,
                    rank=rank,
                    world_size=world_size,
                    total_bytes=self.total_bytes,
                    exchange_group=exchange_group,
                    device=device,
                    extension=self._extension,
                )
                self._registered = True
                if world_size == 8:
                    self._init_pxb_ipc(
                        exchange_group=exchange_group,
                        want_pxb=enable_pxb_p2p,
                        pxb_max_elements=min(capacity_elements, max(8, pxb_max_elements)),
                    )
            else:
                self.rank_stride_bytes = _align_up(capacity_elements * 2, 4096)
                self.data_base_offset = 16384
                self.slot_stride_bytes = _align_up(
                    2 * world_size * self.rank_stride_bytes, 4096
                )
                self.total_bytes = _align_up(
                    self.data_base_offset + NUM_RING_SLOTS * self.slot_stride_bytes, 4096
                )
                shm_path = f"/dev/shm/{name}"
                flags = (os.O_CREAT | os.O_RDWR | os.O_TRUNC) if create_shm else os.O_RDWR
                fd = os.open(shm_path, flags, 0o600)
                try:
                    if create_shm:
                        os.ftruncate(fd, self.total_bytes)
                    self._mapping = mmap.mmap(
                        fd,
                        self.total_bytes,
                        flags=mmap.MAP_SHARED,
                        prot=mmap.PROT_READ | mmap.PROT_WRITE,
                    )
                finally:
                    os.close(fd)
                self._owns_unlink = create_shm
                self._shm_path = shm_path
                if create_shm:
                    ctypes.memset(
                        ctypes.addressof(ctypes.c_char.from_buffer(self._mapping)),
                        0,
                        self.data_base_offset,
                    )
                self._host_address = ctypes.addressof(ctypes.c_char.from_buffer(self._mapping))
                self._first_touch_legacy_pages()
                self._device_address = self._extension.register_mapped_host(
                    self._host_address, self.total_bytes
                )
                self._registered = True

            self.control: torch.Tensor = torch.zeros(
                MAX_BLOCKS, dtype=torch.int64, device=device
            )
        except BaseException:
            self.close(abort=True)
            raise

    @property
    def pxb_enabled(self) -> bool:
        return self._pxb_enabled

    def _init_pxb_ipc(
        self,
        *,
        exchange_group: dist.ProcessGroup,
        want_pxb: bool,
        pxb_max_elements: int,
    ) -> None:
        aligned_elems = (pxb_max_elements // 8) * 8
        pxb_rank_stride = _align_up(aligned_elems * 2, 4096)
        pxb_slot_stride = 2 * pxb_rank_stride
        small_pxb_stride = 2 * _align_up(40960 * 16, 4096)
        pxb_total_bytes = _align_up(
            SHM_V5_HEADER_BYTES
            + NUM_PXB_RING_SLOTS * (pxb_slot_stride + small_pxb_stride),
            4096,
        )
        own_ptr_t = torch.zeros(1, dtype=torch.int64)
        handle_u8 = torch.zeros(64, dtype=torch.uint8)
        alloc_ok = 0
        if want_pxb:
            try:
                h = self._extension.alloc_ipc_buffer(pxb_total_bytes, own_ptr_t)
                if h.numel() == 64:
                    handle_u8.copy_(h)
                    self._my_hbm_ptr = int(own_ptr_t.item())
                    alloc_ok = 1
            except RuntimeError:
                alloc_ok = 0

        status_t = torch.tensor([alloc_ok], dtype=torch.int32)
        status_list = [torch.zeros(1, dtype=torch.int32) for _ in range(self.world_size)]
        dist.all_gather(status_list, status_t, group=exchange_group)
        if not all(int(s.item()) == 1 for s in status_list):
            self._close_pxb_ipc(sync_peers=True)
            return

        gathered = [torch.empty_like(handle_u8) for _ in range(self.world_size)]
        dist.all_gather(gathered, handle_u8, group=exchange_group)

        open_ok = 0
        try:
            partner = self.rank ^ 1
            self._pt_hbm_ptr = int(self._extension.open_ipc_handle(gathered[partner]))
            open_ok = 1 if self._pt_hbm_ptr != 0 else 0
        except RuntimeError:
            open_ok = 0

        open_t = torch.tensor([open_ok], dtype=torch.int32)
        open_list = [torch.zeros(1, dtype=torch.int32) for _ in range(self.world_size)]
        dist.all_gather(open_list, open_t, group=exchange_group)
        if not all(int(s.item()) == 1 for s in open_list):
            self._close_pxb_ipc(sync_peers=True)
            return

        dist.barrier(group=exchange_group)
        self.pxb_slot_stride_bytes = pxb_slot_stride
        self.pxb_max_vecs = aligned_elems // 8
        self._pxb_enabled = True

    def _close_pxb_ipc(self, *, sync_peers: bool = False) -> None:
        if self._pt_hbm_ptr != 0:
            self._extension.close_ipc_handle(self._pt_hbm_ptr)
            self._pt_hbm_ptr = 0
        if (
            sync_peers
            and self._exchange_group is not None
            and dist.is_initialized()
        ):
            dist.barrier(group=self._exchange_group)
        if self._my_hbm_ptr != 0:
            self._extension.free_ipc_buffer(self._my_hbm_ptr)
            self._my_hbm_ptr = 0
        self._pxb_enabled = False
        self.pxb_slot_stride_bytes = 0
        self.pxb_max_vecs = 0

    def _first_touch_legacy_pages(self) -> None:
        mv = memoryview(self._mapping)
        for slot in range(NUM_RING_SLOTS):
            slot_base = self.data_base_offset + slot * self.slot_stride_bytes
            in_base = slot_base + self.rank * self.rank_stride_bytes
            red_base = slot_base + (self.world_size + self.rank) * self.rank_stride_bytes
            for page_off in range(0, self.rank_stride_bytes, 4096):
                mv[in_base + page_off] = 0
                mv[red_base + page_off] = 0
        mv.release()

    def all_reduce(self, input: torch.Tensor, *, output: torch.Tensor) -> torch.Tensor:
        """Reduce all rank inputs into `output` in a single CUDA-Graph-safe launch."""
        if self._closed:
            raise RuntimeError("AllReduce channel is closed")
        if input.shape != output.shape:
            raise ValueError("input/output shapes must match")
        if input.numel() > self.capacity_elements:
            raise ValueError("input exceeds channel capacity")
        if input.numel() == 0:
            return output
        stream_handle = torch.cuda.current_stream(self.device).cuda_stream
        if self._use_shm_v5:
            # Keep `gridDim.x == self.blocks` invariant across every payload size so all
            # per-block ring step counters `control[0..blocks)` stay strictly in lockstep.
            self._extension.launch_shm_v5_pxb_allreduce(
                input,
                output,
                self.control,
                self._rank_shm_ptrs,
                self._my_hbm_ptr if self._pxb_enabled else 0,
                self._pt_hbm_ptr if self._pxb_enabled else 0,
                self.slot_stride_bytes,
                self.pxb_slot_stride_bytes if self._pxb_enabled else 0,
                self.rank,
                self.world_size,
                self.blocks,
                self.threads,
                self.one_shot_max_vecs,
                self.pxb_max_vecs if self._pxb_enabled else 0,
                self._pxb_bfp if self._pxb_enabled else 0,
                0,
                stream_handle,
            )
        else:
            self._extension.launch_host_mapped_two_shot(
                input,
                output,
                self.control,
                self._device_address,
                self.data_base_offset,
                self.slot_stride_bytes,
                self.rank_stride_bytes,
                self.rank,
                self.world_size,
                self.blocks,
                self.threads,
                stream_handle,
            )
        return output

    def can_fuse_add_rmsnorm(self, input: torch.Tensor) -> bool:
        """Return True when `input` (`[rows, hidden]`) can run the 1-op `shm_v5` epilogue."""
        if not self._use_shm_v5 or input.ndim != 2:
            return False
        rows, hidden = int(input.shape[0]), int(input.shape[1])
        if rows <= 0 or hidden <= 0 or hidden % 8 != 0:
            return False
        vecs_per_row = hidden // 8
        num_vecs = rows * vecs_per_row
        small_pxb = self._pxb_enabled and 32 <= self.blocks <= 56 and num_vecs <= 40960
        eff_blocks = 8 if small_pxb else self.blocks
        capped_threads = 512 if self.threads > 512 else self.threads
        eff_threads = 256 if small_pxb else capped_threads
        vecs_per_cta = (num_vecs + eff_blocks - 1) // eff_blocks
        launch_threads = (
            512
            if (eff_threads == 256 and (vecs_per_cta > 256 or vecs_per_row > 256))
            else eff_threads
        )
        if vecs_per_row > 3 * launch_threads or vecs_per_row > 1536:
            return False
        if rows < eff_blocks and (eff_blocks % rows) != 0:
            return False
        if num_vecs > self.one_shot_max_vecs and (vecs_per_row % self.world_size) != 0:
            return False
        return True

    def all_reduce_add_rmsnorm(
        self,
        input: torch.Tensor,
        residual: torch.Tensor,
        weight: torch.Tensor,
        *,
        normed_out: torch.Tensor,
        eps: float,
        zero_centered: bool = True,
        scratch_out: torch.Tensor | None = None,
    ) -> torch.Tensor:
        """Fused `AllReduce + Residual Add + [Zero-Centered] RMSNorm` with lockstep `control`."""
        if self._closed:
            raise RuntimeError("AllReduce channel is closed")
        if input.ndim != 2:
            raise ValueError(f"Expected 2D [rows, hidden] input, got shape={tuple(input.shape)}")
        if residual.shape != input.shape or normed_out.shape != input.shape:
            raise ValueError("residual and normed_out shapes must match input")
        if weight.shape != (input.shape[1],):
            raise ValueError(f"weight shape {tuple(weight.shape)} != ({input.shape[1]},)")
        if input.numel() > self.capacity_elements:
            raise ValueError("input exceeds channel capacity")
        if input.numel() == 0:
            return normed_out

        if (
            self.can_fuse_add_rmsnorm(input)
            and input.is_contiguous()
            and residual.is_contiguous()
            and weight.is_contiguous()
            and normed_out.is_contiguous()
        ):
            rows, hidden = int(input.shape[0]), int(input.shape[1])
            vecs_per_row = hidden // 8
            num_vecs = rows * vecs_per_row
            eff_blocks = (
                8
                if (self._pxb_enabled and 32 <= self.blocks <= 56 and num_vecs <= 40960)
                else self.blocks
            )
            use_cluster = (
                1
                if (
                    rows < eff_blocks
                    and eff_blocks == 8
                    and (eff_blocks % rows) == 0
                    and (vecs_per_row // (eff_blocks // rows)) <= 3 * 256
                )
                else 0
            )
            stream_handle = torch.cuda.current_stream(self.device).cuda_stream
            self._extension.launch_shm_v5_pxb_allreduce_add_rmsnorm(
                input,
                residual,
                weight,
                normed_out,
                self.control,
                self._rank_shm_ptrs,
                self._my_hbm_ptr if self._pxb_enabled else 0,
                self._pt_hbm_ptr if self._pxb_enabled else 0,
                self.slot_stride_bytes,
                self.pxb_slot_stride_bytes if self._pxb_enabled else 0,
                float(eps),
                1.0 if zero_centered else 0.0,
                self.rank,
                self.world_size,
                self.blocks,
                self.threads,
                self.one_shot_max_vecs,
                self.pxb_max_vecs if self._pxb_enabled else 0,
                use_cluster,
                stream_handle,
            )
            return normed_out

        scratch = scratch_out if scratch_out is not None else torch.empty_like(input)
        self.all_reduce(input, output=scratch)
        residual.copy_((residual.float() + scratch.float()).to(torch.bfloat16))
        res_f32 = residual.float()
        inv_rms = torch.rsqrt(res_f32.pow(2).mean(dim=-1, keepdim=True) + float(eps))
        w_scale = (1.0 + weight.float()) if zero_centered else weight.float()
        normed_out.copy_((res_f32 * inv_rms * w_scale).to(torch.bfloat16))
        return normed_out

    def close(self, *, abort: bool = False) -> None:
        """Release CUDA host/IPC mapping(s) and POSIX shared memory handle(s)."""
        if self._closed:
            return
        if abort:
            self.signal_abort()
        torch.cuda.synchronize(self.device)
        self._close_pxb_ipc(sync_peers=not abort)
        if self._use_shm_v5:
            if self._registered or hasattr(self, "_own_shm_path"):
                self._close_per_rank_shm(self._extension)
                self._registered = False
        else:
            if self._registered:
                self._extension.unregister_mapped_host(self._host_address)
                self._registered = False
            if self._mapping is not None:
                if not self._mapping.closed:
                    self._mapping.close()
                self._mapping = None
            if self._owns_unlink and self._shm_path is not None:
                try:
                    os.unlink(self._shm_path)
                except FileNotFoundError:
                    pass
                self._owns_unlink = False
        self._closed = True

    def __enter__(self) -> HostMappedShmAllReduce:
        return self

    def __exit__(self, exc_type: object, *_args: object) -> None:
        self.close(abort=exc_type is not None)


class HostMappedShmAllGather(_PerRankShmRingBase):
    """CUDA-Graph-safe 1-shot NUMA-local SHM BF16 AllGather across `dim`."""

    def __init__(
        self,
        *,
        name: str,
        rank: int,
        world_size: int,
        capacity_input_elements: int,
        blocks: int,
        threads: int,
        exchange_group: dist.ProcessGroup,
    ) -> None:
        if world_size < 2 or world_size > MAX_RANKS or not 0 <= rank < world_size:
            raise ValueError(f"rank={rank} / world_size={world_size} invalid (max {MAX_RANKS})")
        if capacity_input_elements <= 0 or capacity_input_elements % 8 != 0:
            raise ValueError("capacity_input_elements must be positive and divisible by 8")
        if not 1 <= blocks <= MAX_BLOCKS:
            raise ValueError(f"blocks must be in [1, {MAX_BLOCKS}]")
        if threads <= 0 or threads > 1024 or threads % 32 != 0:
            raise ValueError("threads must be a warp multiple in [32, 1024]")

        device = torch.device("cuda", torch.cuda.current_device())
        self.numa_topology: DeviceNumaTopology = pin_process_to_device_numa(device.index)
        self.name: str = name
        self.rank: int = rank
        self.world_size: int = world_size
        self.capacity_input_elements: int = capacity_input_elements
        self.blocks: int = blocks
        self.threads: int = threads
        self.device: torch.device = device
        self.slot_stride_bytes: int = _align_up(capacity_input_elements * 2, 4096)
        self.total_bytes: int = _align_up(
            SHM_V5_HEADER_BYTES + NUM_RING_SLOTS * self.slot_stride_bytes, 4096
        )
        self._closed: bool = False
        self._registered: bool = False
        self._pxb_ar_ref: weakref.ReferenceType[HostMappedShmAllReduce] | None = None
        self._extension = get_host_mapped_allreduce_extension()
        try:
            self._init_per_rank_shm(
                name=name,
                rank=rank,
                world_size=world_size,
                total_bytes=self.total_bytes,
                exchange_group=exchange_group,
                device=device,
                extension=self._extension,
            )
            self._registered = True
            self.control: torch.Tensor = torch.zeros(MAX_BLOCKS, dtype=torch.int64, device=device)
        except BaseException:
            self.close(abort=True)
            raise

    @property
    def _my_hbm_ptr(self) -> int:
        ar = self._pxb_ar_ref() if self._pxb_ar_ref is not None else None
        if ar is not None and not self._closed and not ar._closed and ar.pxb_enabled:
            return ar._my_hbm_ptr
        return 0

    @property
    def _pt_hbm_ptr(self) -> int:
        ar = self._pxb_ar_ref() if self._pxb_ar_ref is not None else None
        if ar is not None and not self._closed and not ar._closed and ar.pxb_enabled:
            return ar._pt_hbm_ptr
        return 0

    @property
    def _pxb_cap_bytes(self) -> int:
        ar = self._pxb_ar_ref() if self._pxb_ar_ref is not None else None
        if ar is not None and not self._closed and not ar._closed and ar.pxb_enabled:
            return NUM_PXB_RING_SLOTS * ar.pxb_slot_stride_bytes
        return 0

    def attach_pxb_ipc(self, shm_ar: HostMappedShmAllReduce) -> None:
        """Attach CX8 PXB CUDA-IPC partner ring from `shm_ar` (zero extra HBM, owner-liveness tracked)."""
        if self._closed:
            raise RuntimeError("AllGather channel is closed")
        if (
            shm_ar.rank != self.rank
            or shm_ar.world_size != self.world_size
            or shm_ar.device != self.device
        ):
            raise ValueError(
                f"attach_pxb_ipc rank/world/device mismatch: "
                f"ag=({self.rank}, {self.world_size}, {self.device}) vs "
                f"ar=({shm_ar.rank}, {shm_ar.world_size}, {shm_ar.device})"
            )
        if shm_ar.pxb_enabled and not shm_ar._closed and self.world_size == 8:
            self._pxb_ar_ref = weakref.ref(shm_ar)
        else:
            self._pxb_ar_ref = None

    def all_gather(
        self,
        input: torch.Tensor,
        *,
        dim: int = -1,
        output: torch.Tensor | None = None,
    ) -> torch.Tensor:
        """AllGather `input` along `dim` across `world_size` ranks."""
        if self._closed:
            raise RuntimeError("AllGather channel is closed")
        ndim = input.ndim
        if dim < 0:
            dim += ndim
        if not 0 <= dim < ndim:
            raise ValueError(f"dim {dim} out of range for {ndim}D tensor")
        if not input.is_contiguous():
            input = input.contiguous()
        out_shape = list(input.shape)
        out_shape[dim] *= self.world_size
        if output is None:
            output = torch.empty(out_shape, dtype=input.dtype, device=input.device)
        elif list(output.shape) != out_shape:
            raise ValueError(f"output shape {list(output.shape)} != expected {out_shape}")
        if input.numel() == 0:
            return output
        if input.numel() > self.capacity_input_elements:
            raise ValueError("input exceeds AllGather channel capacity")
        outer_rows = math.prod(input.shape[:dim]) if dim > 0 else 1
        inner_elems = math.prod(input.shape[dim:])
        if inner_elems % 8 != 0:
            raise ValueError(f"inner_elems={inner_elems} must be divisible by 8")
        vecs_per_row = inner_elems // 8
        stream_handle = torch.cuda.current_stream(self.device).cuda_stream
        my_ptr = self._my_hbm_ptr
        pt_ptr = self._pt_hbm_ptr
        use_pxb = my_ptr != 0 and pt_ptr != 0 and 3 * input.numel() * 2 <= self._pxb_cap_bytes
        self._extension.launch_shm_v5_allgather(
            input,
            output,
            self.control,
            self._rank_shm_ptrs,
            my_ptr if use_pxb else 0,
            pt_ptr if use_pxb else 0,
            self.slot_stride_bytes,
            outer_rows,
            vecs_per_row,
            self.rank,
            self.world_size,
            self.blocks,
            self.threads,
            stream_handle,
        )
        return output

    def close(self, *, abort: bool = False) -> None:
        if self._closed:
            return
        if abort:
            self.signal_abort()
        torch.cuda.synchronize(self.device)
        self._pxb_ar_ref = None
        if self._registered or hasattr(self, "_own_shm_path"):
            self._close_per_rank_shm(self._extension)
            self._registered = False
        self._closed = True

    def __enter__(self) -> HostMappedShmAllGather:
        return self

    def __exit__(self, exc_type: object, *_args: object) -> None:
        self.close(abort=exc_type is not None)


class HostMappedShmArgmax:
    """Host-mapped SHM ring for lock-free TP=8 deterministic Top-2 `(logit, token_id)` gather."""

    def __init__(
        self,
        *,
        name: str,
        rank: int,
        world_size: int,
        device: torch.device,
        exchange_group: dist.ProcessGroup,
        max_rows_per_step: int,
    ) -> None:
        if max_rows_per_step <= 0:
            raise ValueError("max_rows_per_step must be positive")
        self.name: str = name
        self.rank: int = rank
        self.world_size: int = world_size
        self.device: torch.device = device
        self.max_rows_per_step: int = max_rows_per_step
        unaligned_bytes = 256 + NUM_RING_SLOTS * MAX_RANKS * (max_rows_per_step * 2) * 8
        self.total_bytes: int = _align_up(unaligned_bytes, 4096)
        self._closed: bool = False
        self._owns_unlink: bool = False
        self._shm_path: str = f"/dev/shm/{name}"
        self._extension = get_host_mapped_allreduce_extension()

        if rank == 0:
            fd = os.open(self._shm_path, os.O_CREAT | os.O_RDWR | os.O_TRUNC, 0o600)
            try:
                os.ftruncate(fd, self.total_bytes)
                self._mapping = mmap.mmap(
                    fd,
                    self.total_bytes,
                    flags=mmap.MAP_SHARED,
                    prot=mmap.PROT_READ | mmap.PROT_WRITE,
                )
            finally:
                os.close(fd)
            self._owns_unlink = True
            ctypes.memset(
                ctypes.addressof(ctypes.c_char.from_buffer(self._mapping)),
                0,
                self.total_bytes,
            )
        dist.barrier(group=exchange_group)
        if rank != 0:
            fd = os.open(self._shm_path, os.O_RDWR)
            try:
                self._mapping = mmap.mmap(
                    fd,
                    self.total_bytes,
                    flags=mmap.MAP_SHARED,
                    prot=mmap.PROT_READ | mmap.PROT_WRITE,
                )
            finally:
                os.close(fd)
        dist.barrier(group=exchange_group)
        if rank == 0 and self._owns_unlink:
            try:
                os.unlink(self._shm_path)
            except FileNotFoundError:
                pass
            self._owns_unlink = False
        self._host_addr: int = ctypes.addressof(ctypes.c_char.from_buffer(self._mapping))
        self._dev_addr: int = self._extension.register_mapped_host(
            self._host_addr, self.total_bytes
        )
        self._registered: bool = True
        self.step_counter: torch.Tensor = torch.zeros(1, dtype=torch.int64, device=device)

    def gather_top2_from_logits(
        self,
        local_logits_bf16: torch.Tensor,
        *,
        local_top2_vals_buf: torch.Tensor,
        local_top2_ids_buf: torch.Tensor,
        out_tokens: torch.Tensor,
        out_logprob_bits: torch.Tensor,
        vocab_per_rank: int,
        all_ranks: bool = False,
    ) -> None:
        num_rows = int(local_logits_bf16.shape[0])
        if num_rows <= 0:
            return
        stream_handle = torch.cuda.current_stream(self.device).cuda_stream
        self._extension.launch_local_top2_bf16(
            local_logits_bf16,
            local_top2_vals_buf,
            local_top2_ids_buf,
            num_rows,
            stream_handle,
        )
        self._extension.launch_tp8_argmax_gather_shm(
            local_top2_vals_buf,
            local_top2_ids_buf,
            out_tokens,
            out_logprob_bits,
            self.step_counter,
            self._dev_addr,
            num_rows,
            self.max_rows_per_step,
            self.rank,
            self.world_size,
            vocab_per_rank,
            1 if all_ranks else 0,
            stream_handle,
        )

    def signal_abort(self) -> None:
        mm = getattr(self, "_mapping", None)
        if mm is not None and not mm.closed and len(mm) >= ARGMAX_SHM_ABORT_OFFSET + 8:
            struct.pack_into("<Q", mm, ARGMAX_SHM_ABORT_OFFSET, 1)

    def close(self, *, abort: bool = False) -> None:
        if self._closed:
            return
        if abort:
            self.signal_abort()
        torch.cuda.synchronize(self.device)
        if self._registered:
            self._extension.unregister_mapped_host(self._host_addr)
            self._registered = False
        self._mapping.close()
        if self._owns_unlink:
            try:
                os.unlink(self._shm_path)
            except FileNotFoundError:
                pass
        self._closed = True
