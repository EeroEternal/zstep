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

"""Zero-copy POSIX shared-memory views matching crates/zs-wire/src/lib.rs."""

from __future__ import annotations

import ctypes
import mmap
import os
import struct
import time
from dataclasses import dataclass
from typing import Any

import numpy as np

ZSTEP_SHM_MAGIC: int = 0x5A53_5445_505F_5631
ZSTEP_PROTOCOL_VERSION: int = 1
SHM_SUPERBLOCK_STRUCT_BYTES: int = 128
SHM_SUPERBLOCK_REGION_BYTES: int = 4096
SUPERBLOCK_ABORT_OFFSET: int = 128
STEP_HEADER_BYTES: int = 64
STEP_RESULT_HEADER_BYTES: int = 64
SEQ_STEP_OUTPUT_BYTES: int = 104
WORKER_STATE_BYTES: int = 64
WORKER_STATES_OFFSET: int = 4096
WORKER_STATES_REGION_BYTES: int = 4096
VEC_ALIGN: int = 256
PAGE_ALIGN: int = 4096
ATOMIC_ACQUIRE: int = 2
ATOMIC_RELEASE: int = 3
STEP_FLAG_DECODE_ONLY: int = 1
DECODE_GRAPH_BUCKETS: tuple[int, ...] = (1, 2, 4, 8, 16, 32, 64, 128)
MAX_DECODE_GRAPH_SEQS: int = 128


def _align_up(offset: int, align: int) -> int:
    return (offset + align - 1) & ~(align - 1)


def compute_result_slot_stride(max_seqs: int) -> int:
    """Compute the page-aligned byte stride of one `StepResultSlot`."""
    if max_seqs <= 0:
        raise ValueError(f"max_seqs must be positive, got {max_seqs}")
    payload = STEP_RESULT_HEADER_BYTES + max_seqs * SEQ_STEP_OUTPUT_BYTES
    return _align_up(payload, PAGE_ALIGN)


def select_decode_bucket(num_seqs: int, max_bucket: int) -> int:
    """Select the smallest decode CUDA Graph bucket `>= num_seqs` without silent truncation."""
    if num_seqs <= 0 or num_seqs > max_bucket:
        raise ValueError(
            f"decode num_seqs={num_seqs} is outside supported CUDA Graph range [1, {max_bucket}]"
        )
    for bucket in DECODE_GRAPH_BUCKETS:
        if bucket >= num_seqs and bucket <= max_bucket:
            return bucket
    if max_bucket >= num_seqs:
        return max_bucket
    raise ValueError(
        f"decode num_seqs={num_seqs} exceeds max_graph_bucket={max_bucket}"
    )


@dataclass(frozen=True)
class ShmGeometry:
    """Geometry parameters mirrored identically from Rust `zs_wire::ShmGeometry`."""

    world_size: int
    max_seqs: int
    max_step_tokens: int
    max_decoder_rows: int
    max_pages_per_seq: int
    page_block_tokens: int
    spec_verify_width: int

    @classmethod
    def from_vram_budget(
        cls,
        *,
        world_size: int,
        per_gpu_vram_mb: int,
        reserved_weights_and_workspace_mb: int,
        kv_bytes_per_token_per_gpu: int,
        max_seqs: int,
        max_step_tokens: int,
        max_decoder_rows: int,
        page_block_tokens: int,
        spec_verify_width: int,
    ) -> ShmGeometry | None:
        if (
            world_size <= 0
            or world_size > 8
            or max_seqs <= 0
            or max_step_tokens <= 0
            or max_decoder_rows <= 0
            or max_decoder_rows > max_step_tokens
            or page_block_tokens <= 0
            or kv_bytes_per_token_per_gpu <= 0
            or per_gpu_vram_mb <= reserved_weights_and_workspace_mb
        ):
            return None
        kv_budget_bytes = (per_gpu_vram_mb - reserved_weights_and_workspace_mb) * 1024 * 1024
        bytes_per_page = kv_bytes_per_token_per_gpu * page_block_tokens
        total_pages = kv_budget_bytes // bytes_per_page
        if total_pages <= 0:
            return None
        return cls(
            world_size=world_size,
            max_seqs=max_seqs,
            max_step_tokens=max_step_tokens,
            max_decoder_rows=max_decoder_rows,
            max_pages_per_seq=min(total_pages, 4096),
            page_block_tokens=page_block_tokens,
            spec_verify_width=spec_verify_width,
        )


@dataclass(frozen=True)
class StepPlanLayout:
    """Byte offsets inside one double-buffered `StepPlanSlot`."""

    total_slot_bytes: int
    header_offset: int
    token_ids_offset: int
    positions_offset: int
    req_slot_ids_offset: int
    replay_mask_offset: int
    engram_prev_window_offset: int
    narrow_keep_indices_offset: int
    sample_row_indices_offset: int
    seq_lens_offset: int
    query_start_loc_offset: int
    global_kv_page_table_offset: int
    swa_page_table_offset: int
    compressor_slot_ids_offset: int
    temperatures_offset: int

    @classmethod
    def compute(cls, g: ShmGeometry) -> StepPlanLayout:
        max_t = g.max_step_tokens
        max_dec = g.max_decoder_rows
        max_b = g.max_seqs
        max_p = g.max_pages_per_seq

        header_offset = 0
        cursor = _align_up(STEP_HEADER_BYTES, VEC_ALIGN)

        token_ids_offset = cursor
        cursor = _align_up(cursor + max_t * 4, VEC_ALIGN)

        positions_offset = cursor
        cursor = _align_up(cursor + max_t * 4, VEC_ALIGN)

        req_slot_ids_offset = cursor
        cursor = _align_up(cursor + max_t * 4, VEC_ALIGN)

        replay_mask_offset = cursor
        cursor = _align_up(cursor + max_t, VEC_ALIGN)

        engram_prev_window_offset = cursor
        cursor = _align_up(cursor + max_t * 3 * 4, VEC_ALIGN)

        narrow_keep_indices_offset = cursor
        cursor = _align_up(cursor + max_dec * 4, VEC_ALIGN)

        sample_row_indices_offset = cursor
        cursor = _align_up(cursor + max_dec * 4, VEC_ALIGN)

        seq_lens_offset = cursor
        cursor = _align_up(cursor + max_b * 4, VEC_ALIGN)

        query_start_loc_offset = cursor
        cursor = _align_up(cursor + (max_b + 1) * 4, VEC_ALIGN)

        global_kv_page_table_offset = cursor
        cursor = _align_up(cursor + max_b * max_p * 4, VEC_ALIGN)

        swa_page_table_offset = cursor
        cursor = _align_up(cursor + max_b * 4 * 4, VEC_ALIGN)

        compressor_slot_ids_offset = cursor
        cursor = _align_up(cursor + max_b * 4, VEC_ALIGN)

        temperatures_offset = cursor
        cursor = _align_up(cursor + max_b * 4, 4096)

        return cls(
            total_slot_bytes=cursor,
            header_offset=header_offset,
            token_ids_offset=token_ids_offset,
            positions_offset=positions_offset,
            req_slot_ids_offset=req_slot_ids_offset,
            replay_mask_offset=replay_mask_offset,
            engram_prev_window_offset=engram_prev_window_offset,
            narrow_keep_indices_offset=narrow_keep_indices_offset,
            sample_row_indices_offset=sample_row_indices_offset,
            seq_lens_offset=seq_lens_offset,
            query_start_loc_offset=query_start_loc_offset,
            global_kv_page_table_offset=global_kv_page_table_offset,
            swa_page_table_offset=swa_page_table_offset,
            compressor_slot_ids_offset=compressor_slot_ids_offset,
            temperatures_offset=temperatures_offset,
        )


@dataclass(frozen=True)
class ShmSuperblock:
    magic: int
    protocol_version: int
    geometry: ShmGeometry
    master_pid: int
    total_shm_bytes: int
    plan_slot_offsets: tuple[int, int]
    plan_slot_stride: int
    result_slot_offsets: tuple[int, int]
    result_slot_stride: int
    layout_hash: int
    max_decode_graph_seqs: int


def compute_layout_hash(layout: StepPlanLayout, result_slot_stride: int) -> int:
    """Match `zs_wire::StepPlanLayout::layout_hash` 64-bit FNV-1a wire ABI hash."""
    fnv_hash = 0xCBF2_9CE4_8422_2325
    fnv_prime = 0x0000_0100_0000_01B3
    mask64 = 0xFFFF_FFFF_FFFF_FFFF
    words = (
        ZSTEP_PROTOCOL_VERSION,
        SHM_SUPERBLOCK_STRUCT_BYTES,
        WORKER_STATE_BYTES,
        STEP_HEADER_BYTES,
        STEP_RESULT_HEADER_BYTES,
        SEQ_STEP_OUTPUT_BYTES,
        layout.total_slot_bytes,
        layout.header_offset,
        layout.token_ids_offset,
        layout.positions_offset,
        layout.req_slot_ids_offset,
        layout.replay_mask_offset,
        layout.engram_prev_window_offset,
        layout.narrow_keep_indices_offset,
        layout.sample_row_indices_offset,
        layout.seq_lens_offset,
        layout.query_start_loc_offset,
        layout.global_kv_page_table_offset,
        layout.swa_page_table_offset,
        layout.compressor_slot_ids_offset,
        layout.temperatures_offset,
        result_slot_stride,
    )
    for word in words:
        for byte in int(word).to_bytes(8, "little", signed=False):
            fnv_hash ^= byte
            fnv_hash = (fnv_hash * fnv_prime) & mask64
    return fnv_hash


@dataclass(frozen=True)
class StepHeader:
    step_id: int
    opcode: int
    num_seqs: int
    num_encoder_tokens: int
    encoder_graph_bucket: int
    num_decoder_rows: int
    decoder_graph_bucket: int
    num_sample_rows: int
    is_dspark_verify: int
    flags: int


@dataclass(frozen=True)
class WorkerState:
    ready_step_id: int
    status_code: int
    error_code: int
    device_sm_arch: int
    free_vram_mb: int
    total_vram_mb: int
    numa_node: int


@dataclass(frozen=True)
class SeqResult:
    req_slot: int
    accepted_token: int
    accepted_logprob_bits: int
    runner_up_token: int = -1
    runner_up_logprob_bits: int = 0


_ATOMIC_OPS: tuple[Any, Any, Any, Any] | None = None


def _get_atomic_ops() -> tuple[Any, Any, Any, Any]:
    global _ATOMIC_OPS
    if _ATOMIC_OPS is not None:
        return _ATOMIC_OPS
    last_err: OSError | None = None
    libatomic: ctypes.CDLL | None = None
    for candidate in ("libatomic.so.1", "libatomic.so"):
        try:
            libatomic = ctypes.CDLL(candidate)
            break
        except OSError as exc:
            last_err = exc
    if libatomic is None:
        raise RuntimeError(f"failed to load libatomic.so.1 for C11 SHM atomics: {last_err}")
    load_8 = libatomic["__atomic_load_8"]
    load_8.argtypes = [ctypes.c_void_p, ctypes.c_int]
    load_8.restype = ctypes.c_uint64
    store_8 = libatomic["__atomic_store_8"]
    store_8.argtypes = [ctypes.c_void_p, ctypes.c_uint64, ctypes.c_int]
    store_8.restype = None
    load_4 = libatomic["__atomic_load_4"]
    load_4.argtypes = [ctypes.c_void_p, ctypes.c_int]
    load_4.restype = ctypes.c_uint32
    store_4 = libatomic["__atomic_store_4"]
    store_4.argtypes = [ctypes.c_void_p, ctypes.c_uint32, ctypes.c_int]
    store_4.restype = None
    _ATOMIC_OPS = (load_8, store_8, load_4, store_4)
    return _ATOMIC_OPS


class ShmBridge:
    """Owned worker attachment with atomic headers and zero-copy payload arrays."""

    def __init__(self, *, name: str, shm_fd: int = -1) -> None:
        if not name or "/" in name or name in {".", ".."}:
            raise ValueError("SHM name must be one nonempty path component")
        self.name = name
        self._closed = False
        (
            self._atomic_load_8,
            self._atomic_store_8,
            self._atomic_load_4,
            self._atomic_store_4,
        ) = _get_atomic_ops()

        if shm_fd >= 0:
            fd = shm_fd
        else:
            path = f"/dev/shm/{name}"
            fd = os.open(path, os.O_RDWR | os.O_CLOEXEC)
        try:
            prefix = os.pread(fd, 4096, 0)
            if len(prefix) != 4096:
                raise RuntimeError("SHM superblock is truncated")
            self.superblock = self._parse_superblock(prefix)
            self._mapping = mmap.mmap(
                fd,
                self.superblock.total_shm_bytes,
                flags=mmap.MAP_SHARED,
                prot=mmap.PROT_READ | mmap.PROT_WRITE,
            )
        finally:
            os.close(fd)
        self.layout = StepPlanLayout.compute(self.superblock.geometry)
        if self.layout.total_slot_bytes != self.superblock.plan_slot_stride:
            raise RuntimeError(
                f"StepPlanLayout stride mismatch: python={self.layout.total_slot_bytes} "
                f"!= rust={self.superblock.plan_slot_stride}"
            )
        expected_result_stride = compute_result_slot_stride(self.superblock.geometry.max_seqs)
        if self.superblock.result_slot_stride != expected_result_stride:
            raise RuntimeError(
                f"StepResultSlot stride mismatch: superblock={self.superblock.result_slot_stride} "
                f"!= expected={expected_result_stride}"
            )
        expected_hash = compute_layout_hash(self.layout, expected_result_stride)
        if self.superblock.layout_hash != expected_hash:
            raise RuntimeError(
                f"SHM layout hash mismatch: superblock=0x{self.superblock.layout_hash:016x} "
                f"!= expected=0x{expected_hash:016x}"
            )
        expected_plans_base = SHM_SUPERBLOCK_REGION_BYTES + WORKER_STATES_REGION_BYTES
        expected_results_base = expected_plans_base + 2 * self.layout.total_slot_bytes
        expected_total_bytes = expected_results_base + 2 * expected_result_stride
        if (
            self.superblock.plan_slot_offsets
            != (expected_plans_base, expected_plans_base + self.layout.total_slot_bytes)
            or self.superblock.result_slot_offsets
            != (expected_results_base, expected_results_base + expected_result_stride)
            or self.superblock.total_shm_bytes != expected_total_bytes
        ):
            raise RuntimeError("SHM region offsets or total_shm_bytes failed validation")

    @staticmethod
    def _parse_superblock(prefix: bytes) -> ShmSuperblock:
        values = struct.unpack_from("<Q9I", prefix, 0)
        magic = values[0]
        protocol_version = values[1]
        if magic != ZSTEP_SHM_MAGIC or protocol_version != ZSTEP_PROTOCOL_VERSION:
            raise RuntimeError("SHM magic or protocol version mismatch")
        geometry = ShmGeometry(
            world_size=values[2],
            max_seqs=values[3],
            max_step_tokens=values[4],
            max_decoder_rows=values[5],
            max_pages_per_seq=values[6],
            page_block_tokens=values[7],
            spec_verify_width=values[8],
        )
        total_shm_bytes = struct.unpack_from("<Q", prefix, 48)[0]
        plan_slot_offsets = struct.unpack_from("<2Q", prefix, 56)
        plan_slot_stride = struct.unpack_from("<Q", prefix, 72)[0]
        result_slot_offsets = struct.unpack_from("<2Q", prefix, 80)
        result_slot_stride = struct.unpack_from("<Q", prefix, 96)[0]
        layout_hash = struct.unpack_from("<Q", prefix, 104)[0]
        max_decode_graph_seqs = struct.unpack_from("<I", prefix, 116)[0]
        return ShmSuperblock(
            magic=magic,
            protocol_version=protocol_version,
            geometry=geometry,
            master_pid=values[9],
            total_shm_bytes=total_shm_bytes,
            plan_slot_offsets=plan_slot_offsets,
            plan_slot_stride=plan_slot_stride,
            result_slot_offsets=result_slot_offsets,
            result_slot_stride=result_slot_stride,
            layout_hash=layout_hash,
            max_decode_graph_seqs=max_decode_graph_seqs,
        )

    def _address(self, offset: int) -> int:
        return ctypes.addressof(ctypes.c_char.from_buffer(self._mapping, offset))

    def load_u64(self, offset: int) -> int:
        return int(self._atomic_load_8(self._address(offset), ATOMIC_ACQUIRE))

    def store_u64(self, offset: int, value: int) -> None:
        self._atomic_store_8(self._address(offset), value, ATOMIC_RELEASE)

    def load_u32(self, offset: int) -> int:
        return int(self._atomic_load_4(self._address(offset), ATOMIC_ACQUIRE))

    def store_u32(self, offset: int, value: int) -> None:
        self._atomic_store_4(self._address(offset), value, ATOMIC_RELEASE)

    def store_i32(self, offset: int, value: int) -> None:
        self._atomic_store_4(self._address(offset), value & 0xFFFF_FFFF, ATOMIC_RELEASE)

    def plan(self, step_id: int) -> StepPlanView:
        slot = step_id & 1
        return StepPlanView(
            bridge=self,
            base=self.superblock.plan_slot_offsets[slot],
            layout=self.layout,
        )

    def result(self, step_id: int) -> StepResultView:
        slot = step_id & 1
        return StepResultView(
            bridge=self,
            base=self.superblock.result_slot_offsets[slot],
            max_seqs=self.superblock.geometry.max_seqs,
        )

    def is_aborted(self) -> bool:
        return self.load_u32(SUPERBLOCK_ABORT_OFFSET) != 0

    def touch_heartbeat(self, rank: int) -> None:
        base = WORKER_STATES_OFFSET + rank * WORKER_STATE_BYTES
        self.store_u64(base + 8, time.monotonic_ns())

    def write_worker_state(
        self,
        *,
        rank: int,
        ready_step_id: int,
        status_code: int,
        error_code: int,
        device_sm_arch: int,
        free_vram_mb: int,
        total_vram_mb: int,
        numa_node: int,
        triton_recompilations: int = 0,
        watched_kernels: int = 0,
    ) -> None:
        base = WORKER_STATES_OFFSET + rank * WORKER_STATE_BYTES
        self.store_u64(base, ready_step_id)
        self.store_u64(base + 8, time.monotonic_ns())
        self.store_u32(base + 16, os.getpid())
        self.store_u32(base + 20, device_sm_arch)
        self.store_u32(base + 28, error_code)
        self.store_u32(base + 32, free_vram_mb)
        self.store_u32(base + 36, total_vram_mb)
        self.store_i32(base + 40, numa_node)
        self.store_u32(base + 44, triton_recompilations)
        self.store_u32(base + 48, watched_kernels)
        self.store_u32(base + 24, status_code)

    def array(self, *, offset: int, count: int, dtype: np.dtype) -> np.ndarray:
        return np.frombuffer(self._mapping, dtype=dtype, count=count, offset=offset)

    def close(self) -> None:
        if not self._closed:
            self._mapping.close()
            self._closed = True


class StepPlanView:
    def __init__(self, *, bridge: ShmBridge, base: int, layout: StepPlanLayout) -> None:
        self.bridge = bridge
        self.base = base
        self.layout = layout

    def header(self) -> StepHeader:
        # C11 Acquire load paired with the Rust master's Release store in `StepPlanSlotMut::publish`.
        step_id = self.bridge.load_u64(self.base)
        fields = struct.unpack_from("<12I", self.bridge._mapping, self.base + 8)
        if self.bridge.load_u64(self.base) != step_id:
            raise RuntimeError("StepHeader slot was overwritten while being acquired")
        return StepHeader(
            step_id=step_id,
            opcode=fields[0],
            num_seqs=fields[1],
            num_encoder_tokens=fields[2],
            encoder_graph_bucket=fields[3],
            num_decoder_rows=fields[4],
            decoder_graph_bucket=fields[5],
            num_sample_rows=fields[6],
            is_dspark_verify=fields[7],
            flags=fields[8],
        )

    def token_ids(self, count: int) -> np.ndarray:
        return self.bridge.array(
            offset=self.base + self.layout.token_ids_offset,
            count=count,
            dtype=np.dtype(np.int32),
        )

    def positions(self, count: int) -> np.ndarray:
        return self.bridge.array(
            offset=self.base + self.layout.positions_offset,
            count=count,
            dtype=np.dtype(np.int32),
        )

    def req_slot_ids(self, count: int) -> np.ndarray:
        return self.bridge.array(
            offset=self.base + self.layout.req_slot_ids_offset,
            count=count,
            dtype=np.dtype(np.int32),
        )

    def replay_mask(self, count: int) -> np.ndarray:
        return self.bridge.array(
            offset=self.base + self.layout.replay_mask_offset,
            count=count,
            dtype=np.dtype(np.uint8),
        )

    def engram_prev_window(self, num_seqs: int) -> np.ndarray:
        flat = self.bridge.array(
            offset=self.base + self.layout.engram_prev_window_offset,
            count=num_seqs * 3,
            dtype=np.dtype(np.int32),
        )
        return flat.reshape(num_seqs, 3)

    def narrow_keep_indices(self, count: int) -> np.ndarray:
        return self.bridge.array(
            offset=self.base + self.layout.narrow_keep_indices_offset,
            count=count,
            dtype=np.dtype(np.int32),
        )

    def sample_row_indices(self, count: int) -> np.ndarray:
        return self.bridge.array(
            offset=self.base + self.layout.sample_row_indices_offset,
            count=count,
            dtype=np.dtype(np.int32),
        )

    def seq_lens(self, count: int) -> np.ndarray:
        return self.bridge.array(
            offset=self.base + self.layout.seq_lens_offset,
            count=count,
            dtype=np.dtype(np.int32),
        )

    def query_start_loc(self, count: int) -> np.ndarray:
        return self.bridge.array(
            offset=self.base + self.layout.query_start_loc_offset,
            count=count,
            dtype=np.dtype(np.int32),
        )

    def global_kv_page_table(self, num_seqs: int) -> np.ndarray:
        max_p = self.bridge.superblock.geometry.max_pages_per_seq
        flat = self.bridge.array(
            offset=self.base + self.layout.global_kv_page_table_offset,
            count=num_seqs * max_p,
            dtype=np.dtype(np.int32),
        )
        return flat.reshape(num_seqs, max_p)

    def swa_page_table(self, num_seqs: int) -> np.ndarray:
        flat = self.bridge.array(
            offset=self.base + self.layout.swa_page_table_offset,
            count=num_seqs * 4,
            dtype=np.dtype(np.int32),
        )
        return flat.reshape(num_seqs, 4)

    def compressor_slot_ids(self, num_seqs: int) -> np.ndarray:
        return self.bridge.array(
            offset=self.base + self.layout.compressor_slot_ids_offset,
            count=num_seqs,
            dtype=np.dtype(np.int32),
        )

    def temperatures(self, num_seqs: int) -> np.ndarray:
        return self.bridge.array(
            offset=self.base + self.layout.temperatures_offset,
            count=num_seqs,
            dtype=np.dtype(np.float32),
        )


class StepResultView:
    def __init__(self, *, bridge: ShmBridge, base: int, max_seqs: int) -> None:
        self.bridge = bridge
        self.base = base
        self.max_seqs = max_seqs

    def write(
        self,
        *,
        step_id: int,
        gpu_exec_us: int,
        status_code: int,
        outputs: list[SeqResult],
    ) -> None:
        if len(outputs) > self.max_seqs:
            raise ValueError("too many sequence outputs for result slot")
        header = struct.pack(
            "<QIII11I",
            0,
            gpu_exec_us,
            status_code,
            len(outputs),
            *([0] * 11),
        )
        self.bridge._mapping[self.base : self.base + STEP_RESULT_HEADER_BYTES] = header
        for index, output in enumerate(outputs):
            accepted = [output.accepted_token, output.runner_up_token] + [-1] * 6
            drafts = [-1] * 8
            logprobs = [
                output.accepted_logprob_bits,
                output.runner_up_logprob_bits,
            ] + [0] * 6
            packed = struct.pack(
                "<II8i8i8I",
                output.req_slot,
                1,
                *accepted,
                *drafts,
                *logprobs,
            )
            offset = self.base + STEP_RESULT_HEADER_BYTES + index * SEQ_STEP_OUTPUT_BYTES
            self.bridge._mapping[offset : offset + SEQ_STEP_OUTPUT_BYTES] = packed
        self.bridge.store_u64(self.base, step_id)
