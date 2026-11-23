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

"""Explicit TP=8 SM120 runner for DeepSeek-V4.1-Flash (`deepseek_v41`).

Integrates zstep's zero-copy POSIX SHM `StepPlan` / `StepResult` protocol, Rust-owned
128-token KV page tables, exact 3-token Engram lookback windows (`engram_prev_window`),
NUMA-local `shm_v5` `HostMappedShmAllReduce` + `HostMappedShmAllGather`, and lock-free
`HostMappedShmArgmax` with `vllm 0.31.0`'s native SM120 DeepGEMM UE8M0 + FlashInfer
sparse MLA/SWA kernels and full-40-layer `CUDAGraph` decode buckets.
"""

from __future__ import annotations

from collections import OrderedDict
from dataclasses import dataclass
import dataclasses
import gc
import logging
import math
import os
from pathlib import Path
from typing import Any

import numpy as np
import torch
import torch.distributed as dist

from zstep_worker.kernels_sm120.b12x_dsv41 import (
    SM120_DSA_CANDIDATE_COVER_TOKENS,
    B12xPrepareSummary,
    attach_sm120_mhc_shm_ar,
    close_b12x_dsv41_model,
    flush_sm120_mhc_before_ar,
    install_b12x_dsv41_quant_patches,
    install_sm120_dsa_indexer_patches,
    is_sm120_dsa_decode_available,
    prepare_b12x_dsv41_model,
)
from zstep_worker.kernels_sm120.candidate_mqa import (
    get_sm120_candidate_mqa_extension,
    install_sm120_candidate_mqa_prefill_patch,
)
from zstep_worker.kernels_sm120.shm_allreduce import (
    HostMappedShmAllGather,
    HostMappedShmAllReduce,
    HostMappedShmArgmax,
    get_host_mapped_allreduce_extension,
)
from zstep_worker.kernels_sm120.sm_h2d import SmH2dCopier
from zstep_worker.kv_dram_tier import HostDramKvTier
from zstep_worker.numa_pin import pin_process_to_device_numa
from zstep_worker.p2p_probe import P2pTopologyReport, configure_adaptive_p2p_env
from zstep_worker.sampler import SlotPenaltyState, sample_tp8_logits
from zstep_worker.spec_decode import seq_results, verify_batch
from zstep_worker.shm_bridge import (
    DECODE_GRAPH_BUCKETS,
    STEP_FLAG_DECODE_ONLY,
    STEP_FLAG_DEVICE_TOKENS,
    STEP_FLAG_PENALTY,
    STEP_FLAG_SAMPLING,
    SeqResult,
    ShmBridge,
    StepHeader,
    StepPlanView,
    select_decode_bucket,
)

_LOG = logging.getLogger(__name__)
PREFILL_PIECEWISE_BUCKETS: tuple[int, ...] = (
    64,
    128,
    256,
    384,
    512,
    640,
    768,
    896,
    1024,
    1152,
)


def _slice_piecewise_arg(x: Any, n: int) -> Any:
    if x is None:
        return None
    if hasattr(x, "data") and hasattr(x, "scale") and hasattr(x, "orig_shape"):
        return dataclasses.replace(
            x, data=x.data[:n], orig_shape=(n,) + tuple(x.orig_shape[1:])
        )
    return x[:n]


@dataclass(frozen=True)
class _SpecVerifyContext:
    num_sample: int
    sample_rows_np: np.ndarray
    slot_np: np.ndarray
    tok_np: np.ndarray
    pos_np: np.ndarray
    q_start_np: np.ndarray
    pt_np: np.ndarray
    aux_hidden_states: list[torch.Tensor] | None


@dataclass(frozen=True)
class _LaunchedStep:
    parity: int
    num_out: int
    sample_slots_out: list[int]
    spec_ctx: _SpecVerifyContext | None = None


def apply_vllm_sm120_dsv41_patches() -> None:
    """Idempotently apply SM120 64-token page & TMA alignment patches to vLLM 0.31.0."""
    import vllm.models.deepseek_v41.attention as dsv41_attn
    import vllm.models.deepseek_v41.compressor as dsv41_comp
    import vllm.models.deepseek_v41.nvidia.flashinfer_sparse as fi_sparse
    import vllm.v1.attention.backends.mla.indexer as mla_indexer
    from vllm.v1.attention.backends.mla.sparse_swa import DeepseekV4SWACache

    if getattr(fi_sparse, "_zstep_sm120_patched", False):
        return
    fi_sparse._zstep_sm120_patched = True

    # 1. DeepseekV4SWACache: 32 -> 64 token page, 512B TMA alignment, zero-init on bind
    orig_swa_init = DeepseekV4SWACache.__init__
    orig_swa_bind = DeepseekV4SWACache.bind_kv_cache

    def _patched_swa_init(self, *args, **kwargs):
        backend_cls = kwargs.get("backend_cls")
        if (
            backend_cls is not None
            and backend_cls.__name__ == "DeepseekSparseSWAFlashInferBackend"
            and kwargs.get("block_size") == 32
        ):
            kwargs["block_size"] = 64
            kwargs["packed_page_alignment"] = 512
        orig_swa_init(self, *args, **kwargs)

    def _patched_swa_bind(self, kv_cache: torch.Tensor) -> None:
        kv_cache.zero_()
        orig_swa_bind(self, kv_cache)

    DeepseekV4SWACache.__init__ = _patched_swa_init
    DeepseekV4SWACache.bind_kv_cache = _patched_swa_bind

    # 2. DeepseekV4FlashInferSM120Attention: 512B alignment + 64-token split on cr=1
    attn_cls = fi_sparse.DeepseekV4FlashInferSM120Attention
    orig_attn_init = attn_cls.__init__

    def _patched_attn_init(self, vllm_config, *args, **kwargs):
        orig_attn_init(self, vllm_config, *args, **kwargs)
        self.kv_page_alignment = 512

    def _patched_attn_bind_kv_cache(self, kv_cache: torch.Tensor) -> None:
        kv_cache.zero_()
        kv = kv_cache.squeeze(1)
        if kv.ndim == 3 and kv.shape[1] == 128:
            if kv.is_contiguous():
                kv = kv.reshape(kv.shape[0] * 2, 64, kv.shape[2])
            else:
                kv = torch.zeros(
                    (kv.shape[0] * 2, 64, kv.shape[2]),
                    dtype=kv.dtype,
                    device=kv.device,
                )
        self.kv_cache = kv

    attn_cls.__init__ = _patched_attn_init
    attn_cls.bind_kv_cache = _patched_attn_bind_kv_cache

    # 3. DeepseekV4IndexerCache + CompressorStateCache + DeepseekV32IndexerMetadataBuilder
    idx_cache_cls = dsv41_attn.DeepseekV4IndexerCache
    orig_idx_get_spec = idx_cache_cls.get_kv_cache_spec

    def _patched_idx_get_spec(self, vllm_config):
        spec = orig_idx_get_spec(self, vllm_config)
        return dataclasses.replace(
            spec,
            alignment=512,
            block_stride_alignment=512,
        )

    def _patched_idx_bind_kv_cache(self, kv_cache: torch.Tensor) -> None:
        kv_cache.zero_()
        kv = kv_cache.squeeze(1)
        if kv.ndim == 3 and kv.shape[1] == 128:
            if kv.is_contiguous():
                kv = kv.reshape(kv.shape[0] * 2, 64, kv.shape[2])
            else:
                kv = torch.zeros(
                    (kv.shape[0] * 2, 64, kv.shape[2]),
                    dtype=kv.dtype,
                    device=kv.device,
                )
        self.kv_cache = kv

    idx_cache_cls.get_kv_cache_spec = _patched_idx_get_spec
    idx_cache_cls.bind_kv_cache = _patched_idx_bind_kv_cache

    comp_cache_cls = dsv41_comp.CompressorStateCache
    orig_comp_bind = comp_cache_cls.bind_kv_cache

    def _patched_comp_bind(self, kv_cache: torch.Tensor) -> None:
        kv_cache.zero_()
        orig_comp_bind(self, kv_cache)

    comp_cache_cls.bind_kv_cache = _patched_comp_bind

    orig_get_paged_meta = mla_indexer.get_paged_mqa_logits_metadata
    skip_paged_meta_state: list[bool] = [False]
    empty_sched_by_dev: dict[torch.device, torch.Tensor] = {}

    def _patched_get_paged_meta(context_lens, block_kv, num_sms, *args, **kwargs):
        if skip_paged_meta_state[0]:
            dev = context_lens.device
            empty = empty_sched_by_dev.get(dev)
            if empty is None:
                empty = torch.empty((0, 2), dtype=torch.int32, device=dev)
                empty_sched_by_dev[dev] = empty
            return empty
        effective_block_kv = 64 if block_kv == 128 else block_kv
        return orig_get_paged_meta(
            context_lens, effective_block_kv, num_sms, *args, **kwargs
        )

    mla_indexer.get_paged_mqa_logits_metadata = _patched_get_paged_meta

    idx_builder_cls = mla_indexer.DeepseekV32IndexerMetadataBuilder
    orig_idx_builder_init = idx_builder_cls.__init__
    orig_idx_builder_build = idx_builder_cls.build

    def _patched_idx_builder_init(self, *args, block_table_width: int, **kwargs):
        orig_idx_builder_init(
            self, *args, block_table_width=block_table_width, **kwargs
        )
        self._sm120_dsa_decode_active = bool(
            is_sm120_dsa_decode_available()
            and not self.use_pcp
            and self.dcp_world_size <= 1
            and not self.supports_varlen
        )
        self._sm120_split_cr1_64 = (
            self.compress_ratio == 1 and self.kv_cache_spec.num_states == 128
        )
        if self._sm120_split_cr1_64:
            max_rows = max(
                self.vllm_config.scheduler_config.max_num_batched_tokens,
                self.vllm_config.scheduler_config.max_num_seqs,
            )
            self._sm120_cr1_bt_buf = torch.zeros(
                (max_rows, block_table_width * 2),
                dtype=torch.int32,
                device=self.device,
            )

    def _patched_idx_builder_build(
        self, common_prefix_len, common_attn_metadata, fast_build: bool = False
    ):
        compressed_max_seq = (
            int(common_attn_metadata.max_seq_len) + self.compress_ratio - 1
        ) // self.compress_ratio
        skip_sched = bool(
            getattr(self, "_sm120_dsa_decode_active", False)
            and (
                self.compress_ratio > 1
                or compressed_max_seq <= SM120_DSA_CANDIDATE_COVER_TOKENS
            )
        )
        prev_skip = skip_paged_meta_state[0]
        skip_paged_meta_state[0] = skip_sched
        try:
            meta = orig_idx_builder_build(
                self, common_prefix_len, common_attn_metadata, fast_build=fast_build
            )
        finally:
            skip_paged_meta_state[0] = prev_skip
        if getattr(self, "_sm120_split_cr1_64", False):
            if meta.prefill is not None:
                for chunk in meta.prefill.chunks:
                    bt = chunk.block_table
                    bt64 = torch.empty(
                        (bt.shape[0], bt.shape[1] * 2),
                        dtype=bt.dtype,
                        device=bt.device,
                    )
                    bt64[:, 0::2] = bt * 2
                    bt64[:, 1::2] = bt * 2 + 1
                    chunk.block_table = bt64
            if meta.decode is not None:
                bt = meta.decode.block_table
                rows, cols2 = bt.shape[0], bt.shape[1] * 2
                if (
                    self._sm120_cr1_bt_buf.shape[0] < rows
                    or self._sm120_cr1_bt_buf.shape[1] < cols2
                ):
                    self._sm120_cr1_bt_buf = torch.zeros(
                        (
                            max(rows, self._sm120_cr1_bt_buf.shape[0]),
                            max(cols2, self._sm120_cr1_bt_buf.shape[1]),
                        ),
                        dtype=torch.int32,
                        device=self.device,
                    )
                buf = self._sm120_cr1_bt_buf[:rows, :cols2]
                buf[:, 0::2] = bt * 2
                buf[:, 1::2] = bt * 2 + 1
                meta.decode.block_table = buf
        return meta

    idx_builder_cls.__init__ = _patched_idx_builder_init
    idx_builder_cls.build = _patched_idx_builder_build
    install_b12x_dsv41_quant_patches()
    install_sm120_dsa_indexer_patches()


@dataclass
class _AttnBuilderGroup:
    kind: str  # "swa" | "mla" | "indexer" | "compressor"
    layer_names: list[str]
    builder: Any


class _SwaPagePool:
    """Deterministic LRU pool mapping global 128-token MLA page IDs (1..N)
    to a smaller physical SWA 128-token page pool (1..M, 0=null, M+1=dummy)."""

    def __init__(self, num_swa_pages: int, mla_dummy_vb: int) -> None:
        self.num_swa_pages: int = int(num_swa_pages)
        self.swa_dummy_vb: int = self.num_swa_pages + 1
        self.mla_dummy_vb: int = int(mla_dummy_vb)
        self._mla_to_swa: OrderedDict[int, int] = OrderedDict()
        self._free: list[int] = list(range(self.num_swa_pages, 0, -1))
        self._lut: np.ndarray = np.zeros(self.mla_dummy_vb + 2, dtype=np.int32)
        self._lut[self.mla_dummy_vb] = self.swa_dummy_vb
        self._slot_tails: dict[int, tuple[int, ...]] = {}
        self.epoch: int = 0

    def lookup(self, mla_pages: np.ndarray) -> np.ndarray:
        return self._lut[mla_pages]

    def prune_inactive_slots(self, pt_np: np.ndarray) -> None:
        if not self._slot_tails:
            return
        n_rows = int(pt_np.shape[0])
        dummy_raw = self.mla_dummy_vb - 1
        dead = [
            slot
            for slot in self._slot_tails
            if slot >= n_rows or int(pt_np[slot, 0]) >= dummy_raw or int(pt_np[slot, 0]) < 0
        ]
        for slot in dead:
            del self._slot_tails[slot]

    def map_active_windows(
        self,
        active_pt: np.ndarray,
        seq_pos_min: np.ndarray,
        seq_pos_max: np.ndarray,
        allow_alloc_from: np.ndarray,
        active_slots: np.ndarray | None = None,
    ) -> tuple[np.ndarray, bool]:
        pinned: set[int] = set()
        to_alloc: list[int] = []
        bt_w = int(active_pt.shape[1])
        for i in range(int(active_pt.shape[0])):
            p_lo = max(0, (int(seq_pos_min[i]) - 127) // 128)
            p_hi = min(bt_w - 1, int(seq_pos_max[i]) // 128)
            w_lo = int(allow_alloc_from[i]) // 128
            for col in range(p_lo, p_hi + 1):
                mp = int(active_pt[i, col])
                if mp <= 0 or mp >= self.mla_dummy_vb:
                    continue
                pinned.add(mp)
                if mp in self._mla_to_swa:
                    self._mla_to_swa.move_to_end(mp)
                else:
                    if col < w_lo:
                        raise RuntimeError(
                            f"SWA pool miss on read-only prefix page mla_page={mp} "
                            f"(col={col} < w_lo={w_lo}); increase ZSTEP_SWA_KV_PAGES "
                            f"(currently {self.num_swa_pages})"
                        )
                    to_alloc.append(mp)
            if active_slots is not None:
                slot = int(active_slots[i])
                tail_hi = max(0, p_hi - 1)
                # Keep at most 2 tail pages on multi-page prefill (>= 128 tokens) to preserve
                # SwaSafeLru's `2 * max_seqs` bound, while allowing straddling decode/verify
                # windows (< 128 tokens) to retain `p_lo` in case all drafts are rejected.
                tail_lo = (
                    min(p_lo, tail_hi)
                    if int(seq_pos_max[i]) - int(seq_pos_min[i]) < 128
                    else tail_hi
                )
                tail = tuple(
                    int(active_pt[i, c])
                    for c in range(tail_lo, p_hi + 1)
                    if 0 < int(active_pt[i, c]) < self.mla_dummy_vb
                )
                self._slot_tails[slot] = tail
        for tail_pages in self._slot_tails.values():
            pinned.update(tail_pages)
        allocated_any = False
        for mp in to_alloc:
            if mp in self._mla_to_swa:
                continue
            allocated_any = True
            if self._free:
                sp = self._free.pop()
            else:
                evicted_mp: int | None = None
                for cand_mp in self._mla_to_swa:
                    if cand_mp not in pinned:
                        evicted_mp = cand_mp
                        break
                if evicted_mp is None:
                    raise RuntimeError(
                        f"SWA pool exhausted: all {self.num_swa_pages} pages are "
                        "pinned by active SWA windows"
                    )
                sp = self._mla_to_swa.pop(evicted_mp)
                self._lut[evicted_mp] = 0
            self._mla_to_swa[mp] = sp
            self._lut[mp] = sp
        if allocated_any:
            self.epoch += 1
        return self._lut[active_pt], allocated_any


class DeepSeekV41StepRunner:
    """Explicit GPU step runner for DeepSeek-V4.1-Flash (`TP=8`, `sm_120`)."""

    def __init__(
        self,
        *,
        model_dir: Path,
        rank: int,
        bridge: ShmBridge,
        dist_init_file: str,
        allreduce_name: str,
        allreduce_blocks: int,
        allreduce_threads: int,
        num_kv_pages: int,
    ) -> None:
        os.environ.setdefault("CUDA_DEVICE_MAX_CONNECTIONS", "1")
        os.environ.setdefault("VLLM_USE_BREAKABLE_CUDAGRAPH", "1")
        torch.set_num_threads(2)

        self._closed: bool = False
        self._vllm_config_ctx: Any = None
        self.shm_ar: HostMappedShmAllReduce | None = None
        self.shm_ar_tbo: HostMappedShmAllReduce | None = None
        self.shm_ag: HostMappedShmAllGather | None = None
        self.argmax_channel: HostMappedShmArgmax | None = None
        self.decode_graphs: dict[int, torch.cuda.CUDAGraph] = {}
        self.piecewise_graphs: dict[int, Any] = {}
        self.piecewise_outputs: dict[int, torch.Tensor] = {}
        self.piecewise_aux_outputs: dict[int, list[torch.Tensor] | None] = {}
        self._pw_live_lookback: torch.Tensor | None = None
        self._pw_live_num_tokens: int = 0
        self.verify_graphs: dict[int, torch.cuda.CUDAGraph] = {}
        self._gpu_bt_cache_key: tuple[Any, ...] | None = None
        self._gpu_bt_raw_pt: np.ndarray | None = None
        self.verify_graph_aux: dict[int, list[torch.Tensor] | None] = {}
        self.dspark_drafter: Any | None = None
        self._last_verify_bt_sig: tuple[int, tuple[int, ...]] | None = None
        self.dram_kv_tier: HostDramKvTier | None = None
        self._builder_groups: list[_AttnBuilderGroup] = []

        try:
            self._init_impl(
                model_dir=model_dir,
                rank=rank,
                bridge=bridge,
                dist_init_file=dist_init_file,
                allreduce_name=allreduce_name,
                allreduce_blocks=allreduce_blocks,
                allreduce_threads=allreduce_threads,
                num_kv_pages=num_kv_pages,
            )
        except BaseException:
            self._abort_close()
            raise

    def _init_impl(
        self,
        *,
        model_dir: Path,
        rank: int,
        bridge: ShmBridge,
        dist_init_file: str,
        allreduce_name: str,
        allreduce_blocks: int,
        allreduce_threads: int,
        num_kv_pages: int,
    ) -> None:

        self.model_dir: Path = model_dir
        self.rank: int = rank
        from zstep_worker.env_config import WorkerRuntimeConfig

        self.runtime_config: WorkerRuntimeConfig = WorkerRuntimeConfig.from_env()
        self.spec_method: str = self.runtime_config.spec_method
        self.num_spec_tokens: int = self.runtime_config.num_spec_tokens
        if self.spec_method not in {"none", "ngram", "dspark"}:
            raise ValueError(f"DeepSeek-V4.1 does not support speculative method {self.spec_method}")
        self.dspark_model: Any = None
        self.bridge: ShmBridge = bridge
        self.geometry = bridge.superblock.geometry
        self.world_size: int = self.geometry.world_size
        if self.world_size != 8:
            raise ValueError(
                f"DeepSeekV41StepRunner requires world_size=8, got world_size={self.world_size}"
            )
        self.pp_size: int = self.runtime_config.pipeline_parallel_size
        if self.pp_size not in (1, 2) or (self.world_size % self.pp_size) != 0:
            raise ValueError(
                f"Unsupported pipeline_parallel_size={self.pp_size} for world_size={self.world_size}"
            )
        self.tp_size: int = self.world_size // self.pp_size
        self.pp_rank: int = self.rank // self.tp_size
        self.tp_rank: int = self.rank % self.tp_size
        self.is_first_pp: bool = self.pp_rank == 0
        self.is_last_pp: bool = self.pp_rank == self.pp_size - 1
        self.is_result_rank: bool = self.is_last_pp and self.tp_rank == 0
        self._pp_peer: int = (
            (self.rank + self.tp_size if self.is_first_pp else self.rank - self.tp_size)
            if self.pp_size > 1
            else -1
        )
        if self.geometry.page_block_tokens != 128:
            raise ValueError(
                f"DeepSeekV41StepRunner requires page_block_tokens=128, got {self.geometry.page_block_tokens}"
            )

        torch.cuda.set_device(rank)
        self.device: torch.device = torch.device("cuda", rank)
        self.p2p_report: P2pTopologyReport = configure_adaptive_p2p_env(
            rank=rank,
            world_size=self.world_size,
            device_index=rank,
            dist_init_file=dist_init_file,
        )
        if self.p2p_report.pxb_shm_enabled:
            os.environ.setdefault("NCCL_SHM_USE_CUDA_MEMCPY", "1")
            os.environ.setdefault("NCCL_SHM_MEMCPY_MODE", "3")

        apply_vllm_sm120_dsv41_patches()

        from vllm.config import set_current_vllm_config
        from vllm.config.engram import EngramConfig
        from vllm.distributed import (
            get_tp_group,
            get_world_group,
            init_distributed_environment,
            initialize_model_parallel,
        )
        from vllm.engine.arg_utils import EngineArgs
        import vllm.ir
        from vllm.model_executor.model_loader import get_model
        from vllm.v1.worker.workspace import init_workspace_manager

        self.max_model_len: int = self.runtime_config.max_model_len
        self.max_seqs: int = max(self.geometry.max_seqs, 128)
        self.max_step_tokens: int = max(self.geometry.max_step_tokens, 4096)
        self.max_pages_per_seq: int = self.geometry.max_pages_per_seq
        self.bt_width: int = min(
            self.max_pages_per_seq, (self.max_model_len + 127) // 128
        )
        self.num_kv_pages: int = num_kv_pages
        # Block 0 = permanent all-zero null_block; 1..num_kv_pages = real pages;
        # num_kv_pages + 1 = dummy overflow page.
        self.total_128_blocks: int = num_kv_pages + 2
        swa_env_pages = self.runtime_config.swa_kv_pages
        if 0 < swa_env_pages < num_kv_pages:
            self.swa_total_128_blocks: int = swa_env_pages + 2
            self._swa_pool: _SwaPagePool | None = _SwaPagePool(
                swa_env_pages, num_kv_pages + 1
            )
        else:
            self.swa_total_128_blocks = self.total_128_blocks
            self._swa_pool = None
        self.dummy_comp_block: int = self.max_seqs + 1

        engine_args = EngineArgs(
            model=str(model_dir),
            trust_remote_code=True,
            dtype="bfloat16",
            kv_cache_dtype="fp8_ds_mla",
            block_size=128,
            max_model_len=self.max_model_len,
            max_num_seqs=self.max_seqs,
            max_num_batched_tokens=(
                8192 if self.pp_size > 1 else max(self.max_step_tokens, 8192)
            ),
            tensor_parallel_size=self.tp_size,
            pipeline_parallel_size=self.pp_size,
            distributed_executor_backend=(
                "mp" if self.pp_size > 1 else "external_launcher"
            ),
            enforce_eager=True,
            enable_prefix_caching=False,
            limit_mm_per_prompt={"image": 0},
            engram_config=EngramConfig(cpu_offload=(self.pp_size > 1)),
            speculative_config=(
                {
                    "method": "dspark",
                    "model": str(model_dir),
                    "num_speculative_tokens": self.num_spec_tokens,
                }
                if self.spec_method == "dspark"
                else (
                    {
                        "method": "ngram",
                        "num_speculative_tokens": self.num_spec_tokens,
                        "prompt_lookup_max": 5,
                        "prompt_lookup_min": 2,
                    }
                    if self.spec_method == "ngram"
                    else None
                )
            ),
        )
        self.vllm_config = engine_args.create_engine_config()
        self.vllm_config.kernel_config.ir_op_priority.set_default()
        vllm.ir.set_default_torch_wrap(
            self.vllm_config.compilation_config.ir_enable_torch_wrap
        )
        self._vllm_config_ctx = set_current_vllm_config(self.vllm_config)
        self._vllm_config_ctx.__enter__()

        init_distributed_environment(
            world_size=self.world_size,
            rank=rank,
            distributed_init_method=f"file://{dist_init_file}",
            local_rank=rank,
            backend="nccl",
        )
        initialize_model_parallel(
            tensor_model_parallel_size=self.tp_size,
            pipeline_model_parallel_size=self.pp_size,
        )
        self.exchange_group: dist.ProcessGroup = get_world_group().cpu_group
        tp_group_init = get_tp_group()
        self.tp_cpu_group: dist.ProcessGroup = tp_group_init.cpu_group
        init_workspace_manager(self.device)

        # In PP>1 mode, keep the first `pp_engram_gpu_layers` Engram layers (e.g. L1,
        # which has only 1 preceding layer to hide PCIe offload latency) resident on
        # Stage 0 GPU while offloading later Engram layers (e.g. L14) to CPU.
        pp_engram_gpu_layers = (
            self.runtime_config.pp_engram_gpu_layers if self.pp_size > 1 else 0
        )
        orig_create_embedding = None
        engram_cls = None
        if pp_engram_gpu_layers > 0:
            from vllm.models.deepseek_v41.nvidia import engram as dsv41_engram

            engram_cls = dsv41_engram.Engram
            orig_create_embedding = engram_cls._create_embedding

            def _hybrid_create_embedding(
                engram_self: Any, layout: Any, layer_hash_index: int
            ) -> Any:
                engram_cfg = self.vllm_config.engram_config
                assert engram_cfg is not None
                use_cpu_offload = (
                    engram_cfg.cpu_offload
                    and layer_hash_index >= pp_engram_gpu_layers
                )
                return dsv41_engram.ParallelEngramEmbedding(
                    layout.num_embeddings[layer_hash_index],
                    layout.head_dim,
                    tuple(
                        size
                        for order in layout.primes[layer_hash_index]
                        for size in order
                    ),
                    cpu_offload=use_cpu_offload,
                    dp_shared_memory=bool(engram_cfg.dp_shared_memory),
                    use_thp=engram_cfg.use_thp,
                )

            engram_cls._create_embedding = _hybrid_create_embedding
        try:
            self.model = get_model(vllm_config=self.vllm_config)
        finally:
            if engram_cls is not None and orig_create_embedding is not None:
                engram_cls._create_embedding = orig_create_embedding
        self.model.eval()
        if self.spec_method == "dspark":
            draft_config = self.vllm_config.speculative_config.draft_model_config.hf_config
            target_layers = tuple(int(layer) for layer in draft_config.dspark_target_layer_ids)
            target_language_model = getattr(self.model, "language_model", self.model)
            set_aux = getattr(target_language_model, "set_aux_hidden_state_layers", None)
            if callable(set_aux):
                set_aux(target_layers)
            else:
                target_inner = getattr(target_language_model, "model", target_language_model)
                target_inner.aux_hidden_state_layers = target_layers
        self.b12x_summary: B12xPrepareSummary = prepare_b12x_dsv41_model(
            self.model,
            rank=rank,
            device=self.device,
            max_tokens=(8192 if self.pp_size > 1 else max(self.max_step_tokens, 8192)),
            decode_buckets=DECODE_GRAPH_BUCKETS,
            exchange_group=self.exchange_group,
        )
        if self.vllm_config.speculative_config is not None:
            self.vllm_config.speculative_config.num_speculative_tokens = self.num_spec_tokens

        # Configure Engram V2 stateless exact lookback path (`lookback_token_ids`).
        lang_model = getattr(self.model, "language_model", self.model)
        inner_model = getattr(lang_model, "model", lang_model)
        engram_hash = getattr(inner_model, "engram_hash", None)
        if engram_hash is not None:
            engram_hash.use_slot_cache = False
            self.engram_depth: int = int(engram_hash.lookback_depth)
        else:
            self.engram_depth = 3

        vocab_size = int(self.vllm_config.model_config.get_vocab_size())
        if vocab_size % self.tp_size != 0:
            raise ValueError(
                f"vocab_size={vocab_size} must be divisible by tp_size={self.tp_size}"
            )
        self.vocab_size: int = vocab_size
        self.vocab_per_rank: int = vocab_size // self.tp_size
        self.hidden_size: int = int(self.vllm_config.model_config.get_hidden_size())

        # Load cached FlashInfer autotune configs if present in ~/.cache/flashinfer
        # (intentionally skips ~/.cache/vllm/.../zstep_dsv41_sm120_autotune.json because
        # FlashInfer save_configs strips runner_hash from ProfilingCacheKey, which causes
        # same-named multi-instance runners to mis-match runners[0] on reload).
        self._load_flashinfer_autotune_cache()

        # Serialize CUDA C++ extension build on rank 0 before ranks 1..7 load it
        if rank == 0:
            get_host_mapped_allreduce_extension()
            get_sm120_candidate_mqa_extension()
        dist.barrier(group=self.exchange_group)
        if rank != 0:
            get_host_mapped_allreduce_extension()
            get_sm120_candidate_mqa_extension()
        install_sm120_candidate_mqa_prefill_patch(self.model)
        dist.barrier(group=self.exchange_group)

        # Initialize NUMA-local shm_v5 AllReduce, AllGather, and lock-free TP Argmax
        ar_max_tokens = (
            8192 if self.pp_size > 1 else max(self.max_step_tokens, 2048)
        )
        use_pxb_ar = self.p2p_report.pxb_shm_enabled
        pxb_ar_blocks = max(allreduce_blocks, 56) if use_pxb_ar else allreduce_blocks
        pxb_ag_blocks = max(allreduce_blocks, 32) if use_pxb_ar else allreduce_blocks
        pxb_ar_threads = max(allreduce_threads, 512) if use_pxb_ar else allreduce_threads
        stage_suffix = f"_pp{self.pp_rank}" if self.pp_size > 1 else ""
        self.shm_ar: HostMappedShmAllReduce = HostMappedShmAllReduce(
            name=f"{allreduce_name}{stage_suffix}",
            rank=self.tp_rank,
            world_size=self.tp_size,
            capacity_elements=ar_max_tokens * 8192,
            blocks=pxb_ar_blocks,
            threads=pxb_ar_threads,
            exchange_group=self.tp_cpu_group,
            enable_pxb_p2p=self.p2p_report.pxb_shm_enabled,
            pxb_max_elements=min(ar_max_tokens, 8192) * 5120,
            enable_pxb_bfp=self.runtime_config.shm_pxb_bfp_mode,
        )
        tbo_min_tok = self.runtime_config.prefill_tbo_min_tokens
        tbo_cap_tokens = min(max(ar_max_tokens // 2, 2048), 4096)
        self._tbo_max_ubatch_tokens: int = tbo_cap_tokens
        self._tbo_comm_stream: torch.cuda.Stream | None = None
        self._tbo_comp_done_ev: list[torch.cuda.Event] = []
        self._tbo_ar_done_ev: list[torch.cuda.Event] = []
        self._tbo_cand_bufs: list[torch.Tensor] | None = None
        self._tbo_steps: int = 0
        if self.pp_size == 1 and use_pxb_ar and 0 < tbo_min_tok <= 2 * tbo_cap_tokens:
            self.shm_ar_tbo = HostMappedShmAllReduce(
                name=f"{allreduce_name}_tbo",
                rank=self.tp_rank,
                world_size=self.tp_size,
                capacity_elements=tbo_cap_tokens * 5120,
                blocks=self.runtime_config.prefill_tbo_ar_blocks,
                threads=pxb_ar_threads,
                exchange_group=self.tp_cpu_group,
                enable_pxb_p2p=True,
                pxb_max_elements=tbo_cap_tokens * 5120,
                enable_pxb_bfp=self.runtime_config.shm_pxb_bfp_mode,
            )
            self._tbo_comm_stream = torch.cuda.Stream(device=self.device, priority=-1)
            self._tbo_comp_done_ev = [
                torch.cuda.Event(enable_timing=False),
                torch.cuda.Event(enable_timing=False),
            ]
            self._tbo_ar_done_ev = [
                torch.cuda.Event(enable_timing=False),
                torch.cuda.Event(enable_timing=False),
            ]
        self.shm_ag: HostMappedShmAllGather = HostMappedShmAllGather(
            name=f"{allreduce_name}_ag{stage_suffix}",
            rank=self.tp_rank,
            world_size=self.tp_size,
            capacity_input_elements=ar_max_tokens * 4096,
            blocks=pxb_ag_blocks,
            threads=pxb_ar_threads,
            exchange_group=self.tp_cpu_group,
        )
        self.shm_ag.attach_pxb_ipc(self.shm_ar)
        self.argmax_channel: HostMappedShmArgmax | None = (
            HostMappedShmArgmax(
                name=f"{allreduce_name}_argmax{stage_suffix}",
                rank=self.tp_rank,
                world_size=self.tp_size,
                device=self.device,
                exchange_group=self.tp_cpu_group,
                max_rows_per_step=max(512, self.max_step_tokens, self.max_seqs),
            )
            if self.is_last_pp
            else None
        )
        self.shm_ar_calls: int = 0
        self.shm_ag_calls: int = 0
        self.nccl_eager_large_calls: int = 0
        self.nccl_fallback_calls: int = 0
        tp_group = get_tp_group()
        self.tp_device_group = tp_group.device_group
        self._install_tp_shm_collectives(tp_group)
        if self.spec_method == "dspark" and self.pp_size > 1:
            # Share Stage 0's VocabParallelEmbedding weights with Stage 1 so
            # DSparkDeepseekV4ForCausalLM can bind target_inner.embed_tokens.
            from vllm.model_executor.layers.vocab_parallel_embedding import (
                VocabParallelEmbedding,
            )

            t_lang = getattr(self.model, "language_model", self.model)
            t_inner = getattr(t_lang, "model", t_lang)
            if self.is_first_pp:
                emb_w = t_inner.embed_tokens.weight.data.contiguous()
                for req in dist.batch_isend_irecv(
                    [dist.P2POp(dist.isend, emb_w, self._pp_peer)]
                ):
                    req.wait()
            else:
                emb_mod = VocabParallelEmbedding(
                    self.vocab_size,
                    self.hidden_size,
                    prefix="language_model.model.embed_tokens",
                ).to(device=self.device, dtype=torch.bfloat16)
                for req in dist.batch_isend_irecv(
                    [dist.P2POp(dist.irecv, emb_mod.weight.data, self._pp_peer)]
                ):
                    req.wait()
                t_inner.embed_tokens = emb_mod
            torch.cuda.synchronize(self.device)
            dist.barrier(group=self.exchange_group)
        if self.spec_method == "dspark" and self.is_last_pp:
            from vllm.v1.worker.gpu.spec_decode.dspark.utils import load_dspark_model

            self.dspark_model = load_dspark_model(self.model, self.vllm_config)
            self.dspark_model.eval()
            max_draft_rows = min(256, self.max_step_tokens)
            dspark_buckets = tuple(
                b
                for b in DECODE_GRAPH_BUCKETS
                if b * max(1, self.num_spec_tokens) <= max_draft_rows
            ) or (1,)
            prepare_b12x_dsv41_model(
                self.dspark_model,
                rank=rank,
                device=self.device,
                max_tokens=(8192 if self.pp_size > 1 else max(self.max_step_tokens, 8192)),
                decode_buckets=dspark_buckets,
                exchange_group=self.tp_cpu_group,
            )
        dist.barrier(group=self.exchange_group)
        attach_sm120_mhc_shm_ar(self.device.index, self.shm_ar, self)

        # Reclaim transient b12x / weight-conversion blocks before large KV allocation
        gc.collect()
        torch.cuda.empty_cache()

        # Allocate & bind 512B-aligned KV / SWA / Indexer / Compressor caches
        self._kv_raw_buffers: list[torch.Tensor] = []
        self._paged_kv_raw_buffers: list[torch.Tensor] = []
        self._builder_groups: list[_AttnBuilderGroup] = []
        self._tbo_builder_groups_u1: list[_AttnBuilderGroup] = []
        self._allocate_and_bind_caches_and_builders()

        # Allocate static execution, staging, and sampling workspaces
        mt = max(self.max_step_tokens, 8192)
        ms = self.max_seqs + 2
        sc = max(512, self.max_step_tokens)

        self.token_ids_gpu: torch.Tensor = torch.zeros(
            mt, dtype=torch.int32, device=self.device
        )
        self.positions_gpu: torch.Tensor = torch.zeros(
            mt, dtype=torch.int64, device=self.device
        )
        self.query_start_loc_gpu: torch.Tensor = torch.zeros(
            ms + 1, dtype=torch.int32, device=self.device
        )
        self.seq_lens_gpu: torch.Tensor = torch.zeros(
            ms, dtype=torch.int32, device=self.device
        )
        self.slot_mapping_gpu: torch.Tensor = torch.full(
            (mt,), -1, dtype=torch.int64, device=self.device
        )
        self.swa_slot_mapping_gpu: torch.Tensor = (
            torch.full((mt,), -1, dtype=torch.int64, device=self.device)
            if self._swa_pool is not None
            else self.slot_mapping_gpu
        )
        self.mla_bt_gpu: torch.Tensor = torch.zeros(
            (ms, self.bt_width), dtype=torch.int32, device=self.device
        )
        self.swa_bt_gpu: torch.Tensor = torch.zeros(
            (ms, self.bt_width * 2), dtype=torch.int32, device=self.device
        )
        self.comp_bt_gpu: torch.Tensor = torch.full(
            (ms, 1), self.dummy_comp_block, dtype=torch.int32, device=self.device
        )
        self.lookback_gpu: torch.Tensor = torch.full(
            (mt, self.engram_depth),
            -1,
            dtype=torch.int32,
            device=self.device,
        )
        self.sample_rows_gpu: torch.Tensor = torch.zeros(
            sc, dtype=torch.int64, device=self.device
        )
        self._verify_real_rows_cpu: torch.Tensor = torch.zeros(
            sc, dtype=torch.int64, pin_memory=True
        )
        self._verify_real_rows_gpu: torch.Tensor = torch.zeros(
            sc, dtype=torch.int64, device=self.device
        )
        if self.pp_size > 1:
            pp_g_rows = max(256, ms * max(1, self.num_spec_tokens + 1))
            self._pp_inter_hidden: torch.Tensor | None = torch.zeros(
                (pp_g_rows, 4, self.hidden_size), dtype=torch.bfloat16, device=self.device
            )
            self._pp_inter_pre_mix: torch.Tensor | None = torch.zeros(
                (pp_g_rows, 4), dtype=torch.float32, device=self.device
            )
            pp_k = max(1, self.runtime_config.pp_prefill_microbatches)
            self._pp_qsl_cpu: list[torch.Tensor] = [
                torch.zeros(ms + 1, dtype=torch.int32, pin_memory=True)
                for _ in range(pp_k)
            ]
            self._pp_qsl_gpu: list[torch.Tensor] = [
                torch.zeros(ms + 1, dtype=torch.int32, device=self.device)
                for _ in range(pp_k)
            ]
            self._pp_seq_lens_cpu: list[torch.Tensor] = [
                torch.zeros(ms, dtype=torch.int32, pin_memory=True)
                for _ in range(pp_k)
            ]
            self._pp_seq_lens_gpu: list[torch.Tensor] = [
                torch.zeros(ms, dtype=torch.int32, device=self.device)
                for _ in range(pp_k)
            ]
            self._pp_is_pref_cpu: list[torch.Tensor] = [
                torch.zeros(ms, dtype=torch.bool) for _ in range(pp_k)
            ]
        else:
            self._pp_inter_hidden = None
            self._pp_inter_pre_mix = None
            self._pp_qsl_cpu = []
            self._pp_qsl_gpu = []
            self._pp_seq_lens_cpu = []
            self._pp_seq_lens_gpu = []
            self._pp_is_pref_cpu = []

        # Decode-only static arange query_start_loc `[0, 1, 2, ..., ms]`
        self.decode_qsl_gpu: torch.Tensor = torch.arange(
            ms + 1, dtype=torch.int32, device=self.device
        )
        self.decode_qsl_cpu: torch.Tensor = torch.arange(ms + 1, dtype=torch.int32)
        verify_width = self.num_spec_tokens + 1
        self.verify_qsl_gpu: torch.Tensor = (
            torch.arange(ms + 1, dtype=torch.int32, device=self.device) * verify_width
        )
        self.verify_qsl_cpu: torch.Tensor = (
            torch.arange(ms + 1, dtype=torch.int32) * verify_width
        )
        self.decode_is_prefilling_cpu: torch.Tensor = torch.zeros(
            ms, dtype=torch.bool
        )
        self.prefill_is_prefilling_cpu: torch.Tensor = torch.ones(
            ms, dtype=torch.bool
        )
        self.mixed_is_prefilling_cpu: torch.Tensor = torch.zeros(
            ms, dtype=torch.bool
        )
        self.req_idx_arange_np: np.ndarray = np.arange(ms, dtype=np.intp)

        # Pinned H2D / D2H staging buffers
        self.token_ids_cpu: torch.Tensor = torch.zeros(
            mt, dtype=torch.int32, pin_memory=True
        )
        self.positions_cpu: torch.Tensor = torch.zeros(
            mt, dtype=torch.int64, pin_memory=True
        )
        self.query_start_loc_cpu: torch.Tensor = torch.zeros(
            ms + 1, dtype=torch.int32, pin_memory=True
        )
        self.seq_lens_cpu: torch.Tensor = torch.zeros(
            ms, dtype=torch.int32, pin_memory=True
        )
        self.slot_mapping_cpu: torch.Tensor = torch.full(
            (mt,), -1, dtype=torch.int64, pin_memory=True
        )
        self.swa_slot_mapping_cpu: torch.Tensor = (
            torch.full((mt,), -1, dtype=torch.int64, pin_memory=True)
            if self._swa_pool is not None
            else self.slot_mapping_cpu
        )
        self.mla_bt_cpu: torch.Tensor = torch.zeros(
            (ms, self.bt_width), dtype=torch.int32, pin_memory=True
        )
        self.swa_bt_cpu: torch.Tensor = torch.zeros(
            (ms, self.bt_width * 2), dtype=torch.int32, pin_memory=True
        )
        self.comp_bt_cpu: torch.Tensor = torch.full(
            (ms, 1), self.dummy_comp_block, dtype=torch.int32, pin_memory=True
        )
        self.lookback_cpu: torch.Tensor = torch.full(
            (mt, self.engram_depth), -1, dtype=torch.int32, pin_memory=True
        )
        self.sample_rows_cpu: torch.Tensor = torch.zeros(
            sc, dtype=torch.int64, pin_memory=True
        )
        # Pre-allocated microbatch metadata buffers for Prefill Two-Batch Overlap (TBO)
        self._tbo_qsl_cpu: list[torch.Tensor] = [
            torch.zeros(ms + 1, dtype=torch.int32, pin_memory=True) for _ in range(2)
        ]
        self._tbo_qsl_gpu: list[torch.Tensor] = [
            torch.zeros(ms + 1, dtype=torch.int32, device=self.device) for _ in range(2)
        ]
        self._tbo_seq_lens_cpu: list[torch.Tensor] = [
            torch.zeros(ms, dtype=torch.int32, pin_memory=True) for _ in range(2)
        ]
        self._tbo_seq_lens_gpu: list[torch.Tensor] = [
            torch.zeros(ms, dtype=torch.int32, device=self.device) for _ in range(2)
        ]
        self._tbo_is_pref_cpu: list[torch.Tensor] = [
            torch.zeros(ms, dtype=torch.bool) for _ in range(2)
        ]

        # Sampling buffers (deterministic global Top-2 `(logit, token_id)`)
        self.local_max_vals_buf: torch.Tensor = torch.empty(
            (sc, 2), dtype=torch.float32, device=self.device
        )
        self.local_max_ids_buf: torch.Tensor = torch.empty(
            (sc, 2), dtype=torch.int32, device=self.device
        )
        self.out_tokens_gpu: torch.Tensor = torch.zeros(
            (sc, 2), dtype=torch.int32, device=self.device
        )
        self.out_logprobs_gpu: torch.Tensor = torch.zeros(
            (sc, 2), dtype=torch.int32, device=self.device
        )
        self.out_tokens_cpu: torch.Tensor = torch.zeros(
            (2, sc, 2), dtype=torch.int32, pin_memory=True
        )
        self.out_logprobs_cpu: torch.Tensor = torch.zeros(
            (2, sc, 2), dtype=torch.int32, pin_memory=True
        )
        self._out_tokens_np: list[np.ndarray] = [
            self.out_tokens_cpu[0].numpy(),
            self.out_tokens_cpu[1].numpy(),
        ]
        self._out_logprobs_np: list[np.ndarray] = [
            self.out_logprobs_cpu[0].numpy(),
            self.out_logprobs_cpu[1].numpy(),
        ]
        self.slots_cpu: torch.Tensor = torch.zeros(
            ms, dtype=torch.int64, pin_memory=True
        )
        self.slots_gpu: torch.Tensor = torch.zeros(
            ms, dtype=torch.int64, device=self.device
        )
        self.slot_last_tok_gpu: torch.Tensor = torch.zeros(
            self.max_seqs, dtype=torch.int32, device=self.device
        )
        self.slot_lb_gpu: torch.Tensor = torch.full(
            (self.max_seqs, self.engram_depth),
            -1,
            dtype=torch.int32,
            device=self.device,
        )
        self.next_lb_gpu: torch.Tensor = torch.full(
            (ms, self.engram_depth), -1, dtype=torch.int32, device=self.device
        )

        self.start_events: list[torch.cuda.Event] = [
            torch.cuda.Event(enable_timing=True) for _ in range(2)
        ]
        self.end_events: list[torch.cuda.Event] = [
            torch.cuda.Event(enable_timing=True) for _ in range(2)
        ]
        self._h2d_done_event: torch.cuda.Event = torch.cuda.Event()
        self.max_graph_bucket: int = min(self.geometry.max_seqs, 128)
        self.decode_hidden_gpu: torch.Tensor = torch.empty(
            (self.max_graph_bucket, self.hidden_size),
            dtype=torch.bfloat16,
            device=self.device,
        )
        self.sample_temp_cpu: torch.Tensor = torch.zeros(
            (2, ms), dtype=torch.float32, pin_memory=True
        )
        self.sample_top_p_cpu: torch.Tensor = torch.ones(
            (2, ms), dtype=torch.float32, pin_memory=True
        )
        self.sample_top_k_cpu: torch.Tensor = torch.zeros(
            (2, ms), dtype=torch.int32, pin_memory=True
        )
        self.sample_seed_cpu: torch.Tensor = torch.zeros(
            (2, ms), dtype=torch.int64, pin_memory=True
        )
        self._sample_temp_np: list[np.ndarray] = [
            self.sample_temp_cpu[0].numpy(),
            self.sample_temp_cpu[1].numpy(),
        ]
        self._sample_top_p_np: list[np.ndarray] = [
            self.sample_top_p_cpu[0].numpy(),
            self.sample_top_p_cpu[1].numpy(),
        ]
        self._sample_top_k_np: list[np.ndarray] = [
            self.sample_top_k_cpu[0].numpy(),
            self.sample_top_k_cpu[1].numpy(),
        ]
        self._sample_seed_np: list[np.ndarray] = [
            self.sample_seed_cpu[0].numpy(),
            self.sample_seed_cpu[1].numpy(),
        ]
        self.sample_temp_gpu: torch.Tensor = torch.zeros(
            ms, dtype=torch.float32, device=self.device
        )
        self.sample_top_p_gpu: torch.Tensor = torch.ones(
            ms, dtype=torch.float32, device=self.device
        )
        self.sample_top_k_gpu: torch.Tensor = torch.zeros(
            ms, dtype=torch.int32, device=self.device
        )
        self.sample_seed_gpu: torch.Tensor = torch.zeros(
            ms, dtype=torch.int64, device=self.device
        )
        self.sample_pres_pen_cpu: torch.Tensor = torch.zeros(
            (2, ms), dtype=torch.float32, pin_memory=True
        )
        self.sample_freq_pen_cpu: torch.Tensor = torch.zeros(
            (2, ms), dtype=torch.float32, pin_memory=True
        )
        self.sample_reset_cpu: torch.Tensor = torch.zeros(
            (2, ms), dtype=torch.uint8, pin_memory=True
        )
        self.sample_slots_cpu: torch.Tensor = torch.zeros(
            (2, ms), dtype=torch.int64, pin_memory=True
        )
        self._sample_pres_pen_np: list[np.ndarray] = [
            self.sample_pres_pen_cpu[0].numpy(),
            self.sample_pres_pen_cpu[1].numpy(),
        ]
        self._sample_freq_pen_np: list[np.ndarray] = [
            self.sample_freq_pen_cpu[0].numpy(),
            self.sample_freq_pen_cpu[1].numpy(),
        ]
        self._sample_reset_np: list[np.ndarray] = [
            self.sample_reset_cpu[0].numpy(),
            self.sample_reset_cpu[1].numpy(),
        ]
        self._sample_slots_np: list[np.ndarray] = [
            self.sample_slots_cpu[0].numpy(),
            self.sample_slots_cpu[1].numpy(),
        ]
        self.sample_pres_pen_gpu: torch.Tensor = torch.zeros(
            ms, dtype=torch.float32, device=self.device
        )
        self.sample_freq_pen_gpu: torch.Tensor = torch.zeros(
            ms, dtype=torch.float32, device=self.device
        )
        self.sample_reset_gpu: torch.Tensor = torch.zeros(
            ms, dtype=torch.uint8, device=self.device
        )
        self.sample_slots_gpu: torch.Tensor = torch.zeros(
            ms, dtype=torch.int64, device=self.device
        )
        self._penalty_state: SlotPenaltyState | None = None
        self.decode_graph_max_seq_len: int = (
            min(self.max_model_len, SM120_DSA_CANDIDATE_COVER_TOKENS)
            if self.b12x_summary.dsa_decode_enabled
            else self.max_model_len
        )
        max_verify_rows = min(256, self.max_step_tokens)
        self.verify_buckets: list[int] = (
            [
                b
                for b in DECODE_GRAPH_BUCKETS
                if b <= self.max_graph_bucket
                and b * (self.num_spec_tokens + 1) <= max_verify_rows
            ]
            if self.spec_method != "none"
            else []
        )

        if self.dspark_model is not None:
            from zstep_worker.models.deepseek_v41_dspark import (
                DeepSeekV41DSparkDrafter,
            )

            self.dspark_drafter = DeepSeekV41DSparkDrafter(self)

        self.dram_kv_tier: HostDramKvTier | None = None
        if self.runtime_config.dram_kv_blocks > 0:
            if self.runtime_config.swa_kv_pages > 0:
                raise ValueError(
                    "ZSTEP_SWA_KV_PAGES cannot be combined with dram_kv_blocks > 0"
                )
            try:
                numa_node = pin_process_to_device_numa(int(self.device.index)).numa_node
            except Exception:
                numa_node = -1
            self.dram_kv_tier = HostDramKvTier(
                self._paged_kv_raw_buffers,
                total_hbm_blocks=self.total_128_blocks,
                num_dram_blocks=self.runtime_config.dram_kv_blocks,
                device=self.device,
                numa_node=numa_node,
                enable_driver_batch_memcpy=self.runtime_config.enable_driver_batch_memcpy,
            )
        self._sm_h2d: SmH2dCopier = SmH2dCopier(
            enabled=self.dram_kv_tier is not None and self.runtime_config.enable_sm_h2d
        )
        self._warmup_and_capture_decode_graphs()

    def _load_flashinfer_autotune_cache(self) -> int:
        loaded_files = 0
        try:
            from flashinfer.autotuner import AutoTuner

            tuner = AutoTuner.get()
            load_fn = getattr(tuner, "load_configs", None)
            if not callable(load_fn):
                return 0
            cache_root = Path.home() / ".cache" / "flashinfer"
            if cache_root.is_dir():
                candidates = sorted(
                    cache_root.rglob("*autotune*.json"),
                    key=lambda x: (x.stat().st_size, x.stat().st_mtime),
                )
                for p in candidates:
                    try:
                        if load_fn(str(p)):
                            loaded_files += 1
                    except (OSError, ValueError, RuntimeError) as exc:
                        _LOG.debug("Skipping FlashInfer autotune cache %s: %s", p, exc)
        except ImportError:
            return 0
        return loaded_files

    def _install_tp_shm_collectives(self, tp_group: Any) -> None:
        dev_comm = tp_group.device_communicator
        orig_ar = dev_comm.all_reduce
        orig_ag = dev_comm.all_gather
        shm_ar = self.shm_ar
        shm_ag = self.shm_ag
        runner = self
        # Route up to `shm_ar`'s allocated PXB capacity (M <= 4096 when max_step_tokens >= 4096,
        # or M <= 384 on non-P2P) plus 100% of CUDAGraph captures through `shm_ar`; route larger
        # eager AR payloads to topology-aware NCCL (`PXB` on P2P switches, `LOC` on non-P2P).
        default_ar_numel = shm_ar.pxb_max_vecs * 8 if shm_ar.pxb_enabled else 384 * 5120
        shm_max_ar_numel = self.runtime_config.effective_shm_max_ar_numel(
            default_ar_numel
        )
        default_ag_elems = (
            min(self.max_step_tokens, 8192) * 3200 if shm_ar.pxb_enabled else 1024 * 4096
        )
        shm_max_ag_numel = self.runtime_config.effective_shm_max_ag_numel(
            default_ag_elems
        )

        tp_group.use_custom_op_call = False

        def _fast_all_reduce(input_: torch.Tensor) -> torch.Tensor:
            flush_sm120_mhc_before_ar(runner.device.index)
            numel = input_.numel()
            capturing = torch.cuda.is_current_stream_capturing()
            if (
                input_.dtype == torch.bfloat16
                and input_.is_cuda
                and 0 < numel <= shm_ar.capacity_elements
                and numel % 8 == 0
                and (capturing or numel <= shm_max_ar_numel)
            ):
                contig = input_ if input_.is_contiguous() else input_.contiguous()
                if contig.data_ptr() % 16 == 0:
                    shm_ar.all_reduce(contig, output=contig)
                    runner.shm_ar_calls += 1
                    return contig
            if not capturing and numel > shm_max_ar_numel:
                runner.nccl_eager_large_calls += 1
            else:
                runner.nccl_fallback_calls += 1
            return orig_ar(input_)

        def _fast_all_gather(input_: torch.Tensor, dim: int = -1) -> torch.Tensor:
            ndim = input_.ndim
            d = dim + ndim if dim < 0 else dim
            numel = input_.numel()
            capturing = torch.cuda.is_current_stream_capturing()
            if (
                input_.dtype == torch.bfloat16
                and input_.is_cuda
                and 0 <= d < ndim
                and 0 < numel <= shm_ag.capacity_input_elements
                and (capturing or numel <= shm_max_ag_numel)
            ):
                inner = math.prod(input_.shape[d:])
                if inner % 8 == 0:
                    contig = input_ if input_.is_contiguous() else input_.contiguous()
                    if contig.data_ptr() % 16 == 0:
                        runner.shm_ag_calls += 1
                        return shm_ag.all_gather(contig, dim=d)
            if not capturing and numel > shm_max_ag_numel:
                runner.nccl_eager_large_calls += 1
            else:
                runner.nccl_fallback_calls += 1
            return orig_ag(input_, dim)

        dev_comm.all_reduce = _fast_all_reduce
        dev_comm.all_gather = _fast_all_gather

    def _allocate_and_bind_caches_and_builders(self) -> None:
        import vllm.models.deepseek_v41.attention as dsv41_attn
        import vllm.models.deepseek_v41.compressor as dsv41_comp
        import vllm.models.deepseek_v41.nvidia.flashinfer_sparse as fi_sparse
        from vllm.v1.attention.backends.mla.sparse_swa import DeepseekV4SWACache

        sfc = self.vllm_config.compilation_config.static_forward_context
        self._draft_layer_names: set[str] = (
            set(self.dspark_model.get_draft_kv_cache_layer_names())
            if self.dspark_model is not None
            else set()
        )
        grouped_layers: dict[tuple[str, type, Any], tuple[Any, list[str]]] = {}

        for name, layer in sfc.items():
            if not hasattr(layer, "get_kv_cache_spec") or not hasattr(
                layer, "bind_kv_cache"
            ):
                continue
            spec = layer.get_kv_cache_spec(self.vllm_config)
            if spec is None:
                continue
            builder_cls = layer.get_attn_backend().get_builder_cls()
            is_draft_layer = name in self._draft_layer_names

            if isinstance(layer, DeepseekV4SWACache):
                kind = "swa_draft" if is_draft_layer else "swa"
                # 64-token pages, 584B row (64 * 584 = 37376 == 73 * 512)
                raw = torch.zeros(
                    (self.swa_total_128_blocks * 2, 1, 64, 584),
                    dtype=torch.uint8,
                    device=self.device,
                )
                self._kv_raw_buffers.append(raw)
                self._paged_kv_raw_buffers.append(raw)
                layer.bind_kv_cache(raw)
                group_key = (kind, builder_cls, spec)
            elif isinstance(layer, fi_sparse.DeepseekV4FlashInferSM120Attention):
                kind = "mla"
                cr = int(layer.compress_ratio)
                num_states = 128 // cr
                raw = torch.zeros(
                    (self.total_128_blocks, 1, num_states, 584),
                    dtype=torch.uint8,
                    device=self.device,
                )
                self._kv_raw_buffers.append(raw)
                self._paged_kv_raw_buffers.append(raw)
                layer.bind_kv_cache(raw)
                group_key = (f"mla_cr{cr}", builder_cls, spec)
            elif isinstance(layer, dsv41_attn.DeepseekV4IndexerCache):
                kind = "indexer"
                cr = int(spec.tokens_per_state)
                eff_blocks = (
                    self.total_128_blocks * 2 if cr == 1 else self.total_128_blocks
                )
                row_bytes = int(layer.head_dim)
                aligned_page = ((64 * row_bytes + 511) // 512) * 512
                raw_2d = torch.zeros(
                    (eff_blocks, aligned_page),
                    dtype=torch.uint8,
                    device=self.device,
                )
                self._kv_raw_buffers.append(raw_2d)
                self._paged_kv_raw_buffers.append(raw_2d)
                view_4d = torch.as_strided(
                    raw_2d,
                    size=(eff_blocks, 1, 64, row_bytes),
                    stride=(aligned_page, aligned_page, row_bytes, 1),
                )
                layer.bind_kv_cache(view_4d)
                group_key = (f"idx_cr{cr}", builder_cls, spec)
            elif isinstance(layer, dsv41_comp.CompressorStateCache):
                kind = "compressor"
                raw_f32 = torch.zeros(
                    (self.max_seqs + 3, 1, spec.block_size, spec.head_size),
                    dtype=torch.float32,
                    device=self.device,
                )
                self._kv_raw_buffers.append(raw_f32)
                layer.bind_kv_cache(raw_f32)
                group_key = (kind, builder_cls, spec)
            else:
                raise RuntimeError(
                    f"Unhandled AttentionLayerBase in static_forward_context: {name} ({type(layer)})"
                )

            if group_key not in grouped_layers:
                grouped_layers[group_key] = (spec, [name])
            else:
                grouped_layers[group_key][1].append(name)

        for (tag, builder_cls, _), (spec, layer_names) in grouped_layers.items():
            if tag.startswith("swa"):
                kind = "swa"
            elif tag.startswith("mla"):
                kind = "mla"
            elif tag.startswith("idx"):
                kind = "indexer"
            else:
                kind = "compressor"

            kwargs: dict[str, Any] = {}
            if getattr(builder_cls, "requires_block_table_width", False):
                kwargs["block_table_width"] = self.bt_width
            builder = builder_cls(
                spec,
                layer_names,
                self.vllm_config,
                self.device,
                **kwargs,
            )
            self._builder_groups.append(
                _AttnBuilderGroup(kind=kind, layer_names=layer_names, builder=builder)
            )
            if self.shm_ar_tbo is not None and (
                not self._draft_layer_names
                or set(layer_names).isdisjoint(self._draft_layer_names)
            ):
                builder_u1 = builder_cls(
                    spec,
                    layer_names,
                    self.vllm_config,
                    self.device,
                    **kwargs,
                )
                self._tbo_builder_groups_u1.append(
                    _AttnBuilderGroup(
                        kind=kind, layer_names=layer_names, builder=builder_u1
                    )
                )

    def _matches_gpu_bt_cache(
        self,
        bt_key: tuple[Any, ...],
        pt_np: np.ndarray,
        slots_np: np.ndarray,
        c_hi: np.ndarray,
    ) -> bool:
        if bt_key != self._gpu_bt_cache_key or self._gpu_bt_raw_pt is None:
            return False
        cached = self._gpu_bt_raw_pt
        return all(
            np.array_equal(pt_np[int(s), : int(ch) + 1], cached[i, : int(ch) + 1])
            for i, (s, ch) in enumerate(zip(slots_np, c_hi, strict=True))
        )

    def _store_gpu_bt_cache(
        self,
        cache_key: tuple[Any, ...],
        pt_np: np.ndarray,
        slots_np: np.ndarray,
    ) -> None:
        self._gpu_bt_cache_key = cache_key
        n_rows = int(slots_np.shape[0])
        if n_rows == 0:
            return
        if (
            self._gpu_bt_raw_pt is None
            or self._gpu_bt_raw_pt.shape[0] < n_rows
            or self._gpu_bt_raw_pt.shape[1] != pt_np.shape[1]
        ):
            self._gpu_bt_raw_pt = np.empty(
                (max(n_rows, int(self.max_seqs)), pt_np.shape[1]), dtype=pt_np.dtype
            )
        self._gpu_bt_raw_pt[:n_rows] = pt_np[slots_np, :]

    def _zero_all_caches(self) -> None:
        self._gpu_bt_cache_key = None
        for buf in self._kv_raw_buffers:
            buf.zero_()
        for mod in self.model.modules():
            hcache = getattr(mod, "_cache", None)
            if (
                isinstance(hcache, torch.Tensor)
                and hcache.dtype == torch.int32
                and hasattr(mod, "lookback_depth")
                and hcache.numel() > 0
            ):
                hcache.zero_()
        if hasattr(self, "slot_last_tok_gpu"):
            self.slot_last_tok_gpu.zero_()
            self.slot_lb_gpu.fill_(-1)
        torch.cuda.synchronize(self.device)

    def _build_step_attn_metadata(
        self,
        *,
        num_reqs: int,
        num_tokens: int,
        max_query_len: int,
        max_seq_len: int,
        qsl_gpu: torch.Tensor,
        qsl_cpu: torch.Tensor,
        seq_lens_gpu: torch.Tensor,
        seq_lens_cpu: torch.Tensor,
        positions_gpu: torch.Tensor,
        slot_mapping_gpu: torch.Tensor,
        swa_slot_mapping_gpu: torch.Tensor | None = None,
        is_prefilling_cpu: torch.Tensor,
        for_capture: bool,
        causal: bool = True,
        only_layer_names: set[str] | None = None,
        mla_bt_gpu: torch.Tensor | None = None,
        swa_bt_gpu: torch.Tensor | None = None,
        comp_bt_gpu: torch.Tensor | None = None,
        builder_groups: list[_AttnBuilderGroup] | None = None,
    ) -> dict[str, Any]:
        from vllm.v1.attention.backend import CommonAttentionMetadata

        mla_bt = self.mla_bt_gpu[:num_reqs] if mla_bt_gpu is None else mla_bt_gpu
        swa_bt = self.swa_bt_gpu[:num_reqs] if swa_bt_gpu is None else swa_bt_gpu
        comp_bt = self.comp_bt_gpu[:num_reqs] if comp_bt_gpu is None else comp_bt_gpu
        req_idx = self.req_idx_arange_np[:num_reqs]
        token_to_req_cache: dict[str, torch.Tensor | None] = {"val": None}

        def _make_cam(
            bt: torch.Tensor, sm: torch.Tensor = slot_mapping_gpu
        ) -> CommonAttentionMetadata:
            cam = CommonAttentionMetadata(
                query_start_loc=qsl_gpu,
                query_start_loc_cpu=qsl_cpu,
                seq_lens=seq_lens_gpu,
                seq_lens_cpu_upper_bound=seq_lens_cpu,
                max_seq_len=max_seq_len,
                num_reqs=num_reqs,
                num_actual_tokens=num_tokens,
                max_query_len=max_query_len,
                block_table_tensor=bt,
                slot_mapping=sm,
                causal=causal,
                positions=positions_gpu,
                is_prefilling=is_prefilling_cpu,
                req_idx=req_idx,
                _token_to_req_indices_cache=token_to_req_cache["val"],
            )
            return cam

        cam_mla = _make_cam(mla_bt)
        cam_swa = _make_cam(
            swa_bt,
            slot_mapping_gpu if swa_slot_mapping_gpu is None else swa_slot_mapping_gpu,
        )
        cam_comp = _make_cam(comp_bt)

        attn_metadata: dict[str, Any] = {}
        active_groups = self._builder_groups if builder_groups is None else builder_groups
        for grp in active_groups:
            if only_layer_names is None:
                if self._draft_layer_names and not set(grp.layer_names).isdisjoint(
                    self._draft_layer_names
                ):
                    continue
            elif set(grp.layer_names).isdisjoint(only_layer_names):
                continue
            if grp.kind == "swa":
                cam = cam_swa
            elif grp.kind == "compressor":
                cam = cam_comp
            else:
                cam = cam_mla
            if for_capture:
                meta = grp.builder.build_for_cudagraph_capture(cam)
            else:
                meta = grp.builder.build(0, cam)
            if token_to_req_cache["val"] is None and getattr(
                cam, "_token_to_req_indices_cache", None
            ) is not None:
                token_to_req_cache["val"] = cam._token_to_req_indices_cache
                cam_mla._token_to_req_indices_cache = token_to_req_cache["val"]
                cam_swa._token_to_req_indices_cache = token_to_req_cache["val"]
                cam_comp._token_to_req_indices_cache = token_to_req_cache["val"]
            for lname in grp.layer_names:
                if only_layer_names is None or lname in only_layer_names:
                    attn_metadata[lname] = meta
        return attn_metadata

    def _pp_send_intermediate(
        self, hidden_states: torch.Tensor, pre_mix: torch.Tensor
    ) -> list[Any]:
        ops = [
            dist.P2POp(dist.isend, hidden_states.contiguous(), self._pp_peer),
            dist.P2POp(dist.isend, pre_mix.contiguous(), self._pp_peer),
        ]
        return dist.batch_isend_irecv(ops)

    def _pp_irecv_intermediate(
        self, hidden_states: torch.Tensor, pre_mix: torch.Tensor
    ) -> list[Any]:
        ops = [
            dist.P2POp(dist.irecv, hidden_states, self._pp_peer),
            dist.P2POp(dist.irecv, pre_mix, self._pp_peer),
        ]
        return dist.batch_isend_irecv(ops)

    def _pp_recv_intermediate(
        self, hidden_states: torch.Tensor, pre_mix: torch.Tensor
    ) -> Any:
        from vllm.sequence import IntermediateTensors

        for req in self._pp_irecv_intermediate(hidden_states, pre_mix):
            req.wait()
        return IntermediateTensors(
            {"hidden_states": hidden_states, "pre_mix": pre_mix}
        )

    @staticmethod
    def _compute_pp_microbatch_slices(
        total_tokens: int,
        num_seqs: int,
        q_start_np: np.ndarray,
        pos_i64: np.ndarray,
        num_microbatches: int,
    ) -> list[tuple[int, int, int, int, list[int], list[int], list[bool], int, int]]:
        """Split a contiguous prefill batch into `K` causal microbatch metadata tuples."""
        k = max(1, min(num_microbatches, total_tokens))
        slices: list[tuple[int, int, int, int, list[int], list[int], list[bool], int, int]] = []
        for u in range(k):
            lo = (total_tokens * u) // k
            hi = (total_tokens * (u + 1)) // k
            if hi <= lo:
                continue
            r_start = -1
            r_end = 0
            u_qsl: list[int] = [0]
            u_seqlens: list[int] = []
            u_is_pref: list[bool] = []
            max_ql = 0
            max_sl = 0
            for r in range(num_seqs):
                s = int(q_start_np[r])
                e = int(q_start_np[r + 1])
                if e <= lo or s >= hi:
                    continue
                if r_start < 0:
                    r_start = r
                r_end = r + 1
                seg_s = max(s, lo)
                seg_e = min(e, hi)
                ql = seg_e - seg_s
                sl = int(pos_i64[seg_e - 1]) + 1
                u_qsl.append(u_qsl[-1] + ql)
                u_seqlens.append(sl)
                u_is_pref.append((ql > 1) or (int(pos_i64[seg_s]) == 0))
                if ql > max_ql:
                    max_ql = ql
                if sl > max_sl:
                    max_sl = sl
            slices.append((lo, hi, r_start, r_end, u_qsl, u_seqlens, u_is_pref, max_ql, max_sl))
        return slices

    def _run_prefill_pp2(
        self,
        *,
        total_tokens: int,
        num_seqs: int,
        q_start_np: np.ndarray,
        pos_i64: np.ndarray,
        need_hidden: bool = True,
        capture_aux: bool = False,
    ) -> tuple[torch.Tensor | None, list[torch.Tensor] | None]:
        """PP=2 x TP=4 intra-step K-microbatch pipelined prefill."""
        from vllm.config import CUDAGraphMode
        from vllm.forward_context import set_forward_context

        assert self._pp_qsl_cpu is not None and self._pp_qsl_gpu is not None
        assert self._pp_seq_lens_cpu is not None and self._pp_seq_lens_gpu is not None
        assert self._pp_is_pref_cpu is not None

        min_k = max(1, (total_tokens + 8191) // 8192)
        k = max(
            min_k,
            min(
                self.runtime_config.pp_prefill_microbatches,
                max(1, total_tokens // 1024),
            ),
        )
        slices = self._compute_pp_microbatch_slices(
            total_tokens, num_seqs, q_start_np, pos_i64, k
        )
        input_ids_full = self.token_ids_gpu[:total_tokens]
        positions_full = self.positions_gpu[:total_tokens]
        slot_map_full = self.slot_mapping_gpu[:total_tokens]
        swa_slot_map_full = self.swa_slot_mapping_gpu[:total_tokens]

        lang_mod = getattr(self.model, "language_model", self.model)
        dsv4_core = getattr(lang_mod, "model", lang_mod)
        saved_aux = (
            getattr(dsv4_core, "aux_hidden_state_layers", None)
            if (self.is_last_pp and not capture_aux)
            else None
        )
        if saved_aux:
            dsv4_core.aux_hidden_state_layers = ()

        hidden_out: torch.Tensor | None = (
            torch.empty(
                (total_tokens, self.hidden_size),
                dtype=torch.bfloat16,
                device=self.device,
            )
            if (self.is_last_pp and need_hidden)
            else None
        )
        aux_out: list[torch.Tensor] | None = None
        in_flight: list[tuple[list[Any], torch.Tensor, torch.Tensor]] = []
        next_recv: tuple[list[Any], torch.Tensor, torch.Tensor] | None = None
        if not self.is_first_pp and slices:
            from vllm.sequence import IntermediateTensors

            u0_toks = slices[0][1] - slices[0][0]
            h0 = torch.empty(
                (u0_toks, 4, self.hidden_size),
                dtype=torch.bfloat16,
                device=self.device,
            )
            p0 = torch.empty((u0_toks, 4), dtype=torch.float32, device=self.device)
            next_recv = (self._pp_irecv_intermediate(h0, p0), h0, p0)
        try:
            for u_idx, (
                lo,
                hi,
                r_start,
                r_end,
                u_qsl,
                u_seqlens,
                u_is_pref,
                max_ql,
                max_sl,
            ) in enumerate(slices):
                u_toks = hi - lo
                u_reqs = r_end - r_start
                qsl_cpu = self._pp_qsl_cpu[u_idx][: u_reqs + 1]
                qsl_cpu.numpy()[:] = u_qsl
                qsl_gpu = self._pp_qsl_gpu[u_idx][: u_reqs + 1]
                qsl_gpu.copy_(qsl_cpu, non_blocking=True)

                sl_cpu = self._pp_seq_lens_cpu[u_idx][:u_reqs]
                sl_cpu.numpy()[:] = u_seqlens
                sl_gpu = self._pp_seq_lens_gpu[u_idx][:u_reqs]
                sl_gpu.copy_(sl_cpu, non_blocking=True)

                is_pref_cpu = self._pp_is_pref_cpu[u_idx][:u_reqs]
                is_pref_cpu.numpy()[:] = u_is_pref

                meta_u = self._build_step_attn_metadata(
                    num_reqs=u_reqs,
                    num_tokens=u_toks,
                    max_query_len=max_ql,
                    max_seq_len=max_sl,
                    qsl_gpu=qsl_gpu,
                    qsl_cpu=qsl_cpu,
                    seq_lens_gpu=sl_gpu,
                    seq_lens_cpu=sl_cpu,
                    positions_gpu=positions_full[lo:hi],
                    slot_mapping_gpu=slot_map_full[lo:hi],
                    swa_slot_mapping_gpu=swa_slot_map_full[lo:hi],
                    is_prefilling_cpu=is_pref_cpu,
                    for_capture=False,
                    mla_bt_gpu=self.mla_bt_gpu[r_start:r_end],
                    swa_bt_gpu=self.swa_bt_gpu[r_start:r_end],
                    comp_bt_gpu=self.comp_bt_gpu[r_start:r_end],
                )
                lb_u = self.lookback_gpu[r_start:r_end]
                if self.is_first_pp:
                    with set_forward_context(
                        meta_u,
                        self.vllm_config,
                        num_tokens=u_toks,
                        cudagraph_runtime_mode=CUDAGraphMode.NONE,
                    ):
                        inter = self.model(
                            input_ids=input_ids_full[lo:hi],
                            positions=positions_full[lo:hi],
                            lookback_token_ids=lb_u,
                        )
                    h_send = inter["hidden_states"].contiguous()
                    p_send = inter["pre_mix"].contiguous()
                    reqs = self._pp_send_intermediate(h_send, p_send)
                    in_flight.append((reqs, h_send, p_send))
                    if len(in_flight) >= 2:
                        old_reqs, _, _ = in_flight.pop(0)
                        for req in old_reqs:
                            req.wait()
                else:
                    assert next_recv is not None
                    cur_reqs, h_recv, p_recv = next_recv
                    next_recv = None
                    for req in cur_reqs:
                        req.wait()
                    if u_idx + 1 < len(slices):
                        un_toks = slices[u_idx + 1][1] - slices[u_idx + 1][0]
                        hn = torch.empty(
                            (un_toks, 4, self.hidden_size),
                            dtype=torch.bfloat16,
                            device=self.device,
                        )
                        pn = torch.empty(
                            (un_toks, 4), dtype=torch.float32, device=self.device
                        )
                        next_recv = (self._pp_irecv_intermediate(hn, pn), hn, pn)
                    inter = IntermediateTensors(
                        {"hidden_states": h_recv, "pre_mix": p_recv}
                    )
                    with set_forward_context(
                        meta_u,
                        self.vllm_config,
                        num_tokens=u_toks,
                        cudagraph_runtime_mode=CUDAGraphMode.NONE,
                    ):
                        out_u = self.model(
                            input_ids=input_ids_full[lo:hi],
                            positions=positions_full[lo:hi],
                            intermediate_tensors=inter,
                            lookback_token_ids=lb_u,
                        )
                    if isinstance(out_u, tuple):
                        h_u, aux_u = out_u
                    else:
                        h_u, aux_u = out_u, None
                    if hidden_out is not None:
                        hidden_out[lo:hi].copy_(h_u)
                    if capture_aux and aux_u is not None:
                        if aux_out is None:
                            aux_out = [
                                torch.empty(
                                    (total_tokens, t.shape[-1]),
                                    dtype=t.dtype,
                                    device=self.device,
                                )
                                for t in aux_u
                            ]
                        for a_dst, a_src in zip(aux_out, aux_u, strict=True):
                            a_dst[lo:hi].copy_(a_src)
        finally:
            if saved_aux:
                dsv4_core.aux_hidden_state_layers = saved_aux
            for reqs, _, _ in in_flight:
                for req in reqs:
                    req.wait()
        return hidden_out, aux_out

    def _can_run_prefill_tbo(
        self,
        total_tokens: int,
        *,
        max_seq_len: int,
        is_decode: bool,
        is_spec_step: bool = False,
    ) -> bool:
        del is_spec_step
        if (
            is_decode
            or self.shm_ar_tbo is None
            or self._tbo_comm_stream is None
            or torch.cuda.is_current_stream_capturing()
        ):
            return False
        min_tok = self.runtime_config.prefill_tbo_min_tokens
        if min_tok <= 0 or total_tokens < min_tok:
            return False
        min_ctx = self.runtime_config.prefill_tbo_min_ctx
        if min_ctx > 0 and max_seq_len <= min_ctx:
            return False
        mid = total_tokens // 2
        if mid <= 0 or max(mid, total_tokens - mid) > self._tbo_max_ubatch_tokens:
            return False
        lang_mod = getattr(self.model, "language_model", self.model)
        dsv4_core = getattr(lang_mod, "model", lang_mod)
        if not getattr(dsv4_core, "fuse_mhc_all_reduce", False):
            return False
        if getattr(dsv4_core, "use_sequence_parallel", False):
            return False
        if (
            getattr(self, "dspark_drafter", None) is None
            and getattr(dsv4_core, "_mtp_hidden_buffer", None) is not None
        ):
            return False
        return hasattr(dsv4_core, "layers") and hasattr(dsv4_core, "embed_input_ids")

    @staticmethod
    def _compute_tbo_microbatch_slices(
        total_tokens: int,
        num_seqs: int,
        q_start_np: np.ndarray,
        pos_i64: np.ndarray,
    ) -> list[tuple[int, int, int, int, list[int], list[int], list[bool], int, int]]:
        """Split a contiguous prefill batch into two causal microbatch metadata tuples."""
        mid = total_tokens // 2
        slices: list[tuple[int, int, int, int, list[int], list[int], list[bool], int, int]] = []
        for lo, hi in ((0, mid), (mid, total_tokens)):
            r_start = -1
            r_end = 0
            u_qsl: list[int] = [0]
            u_seqlens: list[int] = []
            u_is_pref: list[bool] = []
            max_ql = 0
            max_sl = 0
            for r in range(num_seqs):
                s = int(q_start_np[r])
                e = int(q_start_np[r + 1])
                if e <= lo or s >= hi:
                    continue
                if r_start < 0:
                    r_start = r
                r_end = r + 1
                seg_s = max(s, lo)
                seg_e = min(e, hi)
                ql = seg_e - seg_s
                sl = int(pos_i64[seg_e - 1]) + 1
                u_qsl.append(u_qsl[-1] + ql)
                u_seqlens.append(sl)
                u_is_pref.append((ql > 1) or (int(pos_i64[seg_s]) == 0))
                if ql > max_ql:
                    max_ql = ql
                if sl > max_sl:
                    max_sl = sl
            slices.append((lo, hi, r_start, r_end, u_qsl, u_seqlens, u_is_pref, max_ql, max_sl))
        return slices

    def _run_prefill_tbo(
        self,
        *,
        total_tokens: int,
        num_seqs: int,
        q_start_np: np.ndarray,
        pos_i64: np.ndarray,
        full_attn_meta: dict[str, Any],
        need_hidden: bool = True,
        capture_aux: bool = False,
    ) -> tuple[torch.Tensor | None, list[torch.Tensor] | None]:
        """Prefill Two-Batch Overlap (TBO): split a long prefill chunk into two
        causal microbatches (`u0`, `u1`) and overlap each microbatch's `shm_v5`
        AllReduce (`blocks=ZSTEP_PREFILL_TBO_AR_BLOCKS`, high-priority comm stream)
        with the other microbatch's Attn/FFN compute on the main stream."""
        from vllm.forward_context import get_forward_context
        from vllm.models.deepseek_v41.nvidia import model as dsv41_model

        lang_mod = getattr(self.model, "language_model", self.model)
        dsv4_core = getattr(lang_mod, "model", lang_mod)
        shm_ar_tbo = self.shm_ar_tbo
        comm_s = self._tbo_comm_stream
        assert shm_ar_tbo is not None and comm_s is not None
        main_s = torch.cuda.current_stream(self.device)
        dev_idx = self.device.index

        self._tbo_steps += 1
        if self._tbo_steps == 1 and self.rank == 0:
            _LOG.info(
                "Prefill TBO active: total_tokens=%d mid=%d ar_blocks=%d",
                total_tokens,
                total_tokens // 2,
                shm_ar_tbo.blocks,
            )
        mid = total_tokens // 2
        slices = self._compute_tbo_microbatch_slices(total_tokens, num_seqs, q_start_np, pos_i64)
        attn_metas: list[dict[str, Any]] = []
        positions_full = self.positions_gpu[:total_tokens]
        slot_map_full = self.slot_mapping_gpu[:total_tokens]
        swa_slot_map_full = self.swa_slot_mapping_gpu[:total_tokens]

        for u, (lo, hi, r_start, r_end, u_qsl, u_seqlens, u_is_pref, max_ql, max_sl) in enumerate(
            slices
        ):
            u_toks = hi - lo
            u_reqs = r_end - r_start
            qsl_cpu = self._tbo_qsl_cpu[u][: u_reqs + 1]
            qsl_cpu.numpy()[:] = u_qsl
            qsl_gpu = self._tbo_qsl_gpu[u][: u_reqs + 1]
            qsl_gpu.copy_(qsl_cpu, non_blocking=True)

            sl_cpu = self._tbo_seq_lens_cpu[u][:u_reqs]
            sl_cpu.numpy()[:] = u_seqlens
            sl_gpu = self._tbo_seq_lens_gpu[u][:u_reqs]
            sl_gpu.copy_(sl_cpu, non_blocking=True)

            is_pref_cpu = self._tbo_is_pref_cpu[u][:u_reqs]
            is_pref_cpu.numpy()[:] = u_is_pref

            meta_u = self._build_step_attn_metadata(
                num_reqs=u_reqs,
                num_tokens=u_toks,
                max_query_len=max_ql,
                max_seq_len=max_sl,
                qsl_gpu=qsl_gpu,
                qsl_cpu=qsl_cpu,
                seq_lens_gpu=sl_gpu,
                seq_lens_cpu=sl_cpu,
                positions_gpu=positions_full[lo:hi],
                slot_mapping_gpu=slot_map_full[lo:hi],
                swa_slot_mapping_gpu=swa_slot_map_full[lo:hi],
                is_prefilling_cpu=is_pref_cpu,
                for_capture=False,
                mla_bt_gpu=self.mla_bt_gpu[r_start:r_end],
                swa_bt_gpu=self.swa_bt_gpu[r_start:r_end],
                comp_bt_gpu=self.comp_bt_gpu[r_start:r_end],
                builder_groups=(
                    self._builder_groups
                    if u == 0 or not self._tbo_builder_groups_u1
                    else self._tbo_builder_groups_u1
                ),
            )
            attn_metas.append(meta_u)
        self._h2d_done_event.record(main_s)

        input_ids_full = self.token_ids_gpu[:total_tokens]
        hidden_full = dsv4_core.embed_input_ids(input_ids_full)
        if dsv4_core.use_mega_moe:
            input_ids_full = input_ids_full.to(torch.int64)

        # Compute Engram n-gram hashes once on the full contiguous token stream.
        engram_hashes: torch.Tensor | None = None
        engram_mask: torch.Tensor | None = None
        gathered_hashes: torch.Tensor | None = None
        if dsv4_core.engram_hash is not None and dsv4_core.engram_hash.ensure_cache():
            image_mask = dsv41_model.image_sentinel_mask(input_ids_full)
            engram_mask = ~image_mask
            lb_gpu = self.lookback_gpu[:num_seqs]
            engram_hashes = dsv4_core.engram_hash(
                input_ids_full,
                positions_full,
                self.query_start_loc_gpu[: num_seqs + 1],
                image_mask,
                lb_gpu,
                dsv41_model.image_sentinel_mask(lb_gpu),
                swa_slot_map_full,
                self.swa_bt_gpu[:num_seqs],
            )
            gathered_hashes = dsv41_model.gather_engram_hashes(
                engram_hashes, dp_shared_memory=dsv4_core.engram_dp_shared_memory
            )

        xs = [hidden_full[:mid].contiguous(), hidden_full[mid:].contiguous()]
        del hidden_full
        poss = [positions_full[:mid], positions_full[mid:]]
        ids_u = [input_ids_full[:mid], input_ids_full[mid:]]
        if dsv4_core.use_mega_moe:
            img_base = (
                dsv41_model.IMAGE_SENTINEL_BASE_ID
                if getattr(dsv4_core.config, "vision_n_layers", 0) > 0
                else None
            )
            mega_u = [
                dsv41_model.prepare_mega_gate_routing_metadata(
                    ids_u[u], has_hash_routing=False, image_sentinel_base_id=img_base
                )
                for u in (0, 1)
            ]
        else:
            mega_u = [None, None]
        eh_u = (
            [engram_hashes[:mid], engram_hashes[mid:]]
            if engram_hashes is not None
            else [None, None]
        )
        em_u = (
            [engram_mask[:mid], engram_mask[mid:]]
            if engram_mask is not None
            else [None, None]
        )
        gh_u = (
            [gathered_hashes[:mid], gathered_hashes[mid:]]
            if gathered_hashes is not None
            else [None, None]
        )

        res_u: list[torch.Tensor | None] = [None, None]
        post_u: list[torch.Tensor | None] = [None, None]
        resmix_u: list[torch.Tensor | None] = [None, None]
        pre_u: list[torch.Tensor | None] = [None, None]

        mhc_pre = dsv41_model.mhc_pre_delayed_tilelang
        mhc_post = dsv41_model.mhc_post_tilelang
        mhc_shifted = dsv41_model.mhc_shifted_post_pre
        hc_collapse = dsv41_model.hc_collapse_triton

        def _launch_ar(t: torch.Tensor, u_idx: int) -> torch.Tensor:
            t_c = t if t.is_contiguous() else t.contiguous()
            comp_ev = self._tbo_comp_done_ev[u_idx]
            ar_ev = self._tbo_ar_done_ev[u_idx]
            comp_ev.record(main_s)
            with torch.cuda.stream(comm_s):
                comm_s.wait_event(comp_ev)
                shm_ar_tbo.all_reduce(t_c, output=t_c)
                ar_ev.record(comm_s)
            self.shm_ar_calls += 1
            return t_c

        fwd_ctx = get_forward_context()
        # Zero-copy double-buffered candidate_blocks per microbatch
        # (layer 20 -> layers 24, 28, 32, 36 share one buffer on SparseAttnIndexer).
        cand_orig_by_op: dict[int, tuple[Any, torch.Tensor]] = {}
        last_layer_idx = dsv4_core.end_layer - 1
        aux_layers: tuple[int, ...] = (
            tuple(getattr(dsv4_core, "aux_hidden_state_layers", ()) or ())
            if capture_aux
            else ()
        )
        aux_layers_set = set(aux_layers)
        aux_full_by_layer: dict[int, torch.Tensor] = {}

        def _store_aux(lid: int, u_idx: int, tensor: torch.Tensor) -> None:
            buf = aux_full_by_layer.get(lid)
            if buf is None:
                buf = torch.empty(
                    (total_tokens, tensor.shape[-1]),
                    dtype=tensor.dtype,
                    device=tensor.device,
                )
                aux_full_by_layer[lid] = buf
            if u_idx == 0:
                buf[:mid].copy_(tensor)
            else:
                buf[mid:].copy_(tensor)

        need_final_aux = dsv4_core.end_layer in aux_layers_set
        need_final_post = need_hidden or need_final_aux
        try:
            for idx in range(dsv4_core.start_layer, dsv4_core.end_layer):
                layer = dsv4_core.layers[idx]
                is_last_layer = idx == last_layer_idx
                capture_prev_aux = idx in aux_layers_set
                idx_mod = getattr(getattr(layer, "attn", None), "indexer", None)
                sai_op = getattr(idx_mod, "sparse_attn_indexer", None) if idx_mod is not None else None
                sai_cand = getattr(sai_op, "candidate_blocks", None) if sai_op is not None else None
                if sai_op is not None and sai_cand is not None:
                    if self._tbo_cand_bufs is None:
                        self._tbo_cand_bufs = [
                            torch.empty(
                                (self._tbo_max_ubatch_tokens, sai_cand.shape[1]),
                                dtype=sai_cand.dtype,
                                device=sai_cand.device,
                            )
                            for _ in range(2)
                        ]
                    if id(sai_op) not in cand_orig_by_op:
                        cand_orig_by_op[id(sai_op)] = (sai_op, sai_cand)

                # Phase A: Attention (u0 writes KV before u1 reads KV on main_s;
                # u0's Attn AllReduce overlaps with u1's Attn compute on comm_s).
                for u in (0, 1):
                    if idx > dsv4_core.start_layer:
                        main_s.wait_event(self._tbo_ar_done_ev[u])
                    if res_u[u] is None:
                        res_u[u] = (
                            xs[u].unsqueeze(1).expand(-1, layer.hc_mult, -1).contiguous()
                        )
                        post_u[u], resmix_u[u], xs[u], pre_u[u] = mhc_pre(
                            res_u[u],
                            layer.hc_attn_fn_broadcast,
                            layer.hc_attn_scale,
                            layer.hc_attn_base,
                            layer.rms_norm_eps,
                            layer.hc_eps,
                            layer.hc_eps,
                            layer.hc_post_alpha,
                            layer.hc_sinkhorn_iters,
                            x=xs[u],
                            norm_weight=layer.attn_norm.weight,
                            norm_eps=layer.attn_norm.variance_epsilon,
                        )
                    elif layer.engram is not None and eh_u[u] is not None:
                        prev_post = mhc_post(xs[u], res_u[u], post_u[u], resmix_u[u])
                        if capture_prev_aux:
                            _store_aux(idx, u, prev_post.mean(dim=1))
                        assert gh_u[u] is not None
                        layer.engram.prepare_embeddings(
                            gh_u[u][:, layer.engram.layer_hash_index]
                        )
                        res_u[u] = layer.engram(
                            prev_post,
                            eh_u[u][:, layer.engram.layer_hash_index],
                            em_u[u],
                        )
                        post_u[u], resmix_u[u], xs[u], pre_u[u] = mhc_pre(
                            res_u[u],
                            layer.hc_attn_fn,
                            layer.hc_attn_scale,
                            layer.hc_attn_base,
                            layer.rms_norm_eps,
                            layer.hc_eps,
                            layer.hc_eps,
                            layer.hc_post_alpha,
                            layer.hc_sinkhorn_iters,
                            pre_mix=pre_u[u],
                            norm_weight=layer.attn_norm.weight,
                            norm_eps=layer.attn_norm.variance_epsilon,
                        )
                    else:
                        (
                            res_u[u],
                            post_u[u],
                            resmix_u[u],
                            xs[u],
                            pre_u[u],
                            prev_aux,
                        ) = mhc_shifted(
                            xs[u],
                            res_u[u],
                            post_u[u],
                            resmix_u[u],
                            layer.hc_attn_fn,
                            layer.hc_attn_scale,
                            layer.hc_attn_base,
                            layer.rms_norm_eps,
                            layer.hc_eps,
                            layer.hc_eps,
                            layer.hc_post_alpha,
                            layer.hc_sinkhorn_iters,
                            pre_mix=pre_u[u],
                            norm_weight=layer.attn_norm.weight,
                            norm_eps=layer.attn_norm.variance_epsilon,
                            capture_aux=capture_prev_aux,
                            stream=None,
                            reduce_results=False,
                        )
                        if capture_prev_aux and prev_aux is not None:
                            _store_aux(idx, u, prev_aux)
                            del prev_aux
                    flush_sm120_mhc_before_ar(dev_idx)
                    if sai_op is not None and sai_cand is not None and self._tbo_cand_bufs is not None:
                        sai_op.candidate_blocks = self._tbo_cand_bufs[u]
                    fwd_ctx.attn_metadata = attn_metas[u]
                    xs[u] = layer.attn(poss[u], xs[u], None)
                    if not (is_last_layer and not need_final_post):
                        xs[u] = _launch_ar(xs[u], u)

                # On non-final prefill chunks (num_sample == 0 and no final-layer aux
                # capture), layer 39's KV cache is written; skip stateless last-layer
                # FFN + AR + final norm.
                if is_last_layer and not need_final_post:
                    aux_out = [
                        aux_full_by_layer[lid]
                        for lid in aux_layers
                        if lid in aux_full_by_layer
                    ]
                    return None, (aux_out if aux_out else None)

                # Phase B: FFN / MoE (u0's FFN overlaps with u1's Attn AR;
                # u1's FFN overlaps with u0's FFN AR; next layer u0's Attn
                # overlaps with u1's FFN AR).
                for u in (0, 1):
                    main_s.wait_event(self._tbo_ar_done_ev[u])
                    res_u[u], post_u[u], resmix_u[u], xs[u], pre_u[u], _ = mhc_shifted(
                        xs[u],
                        res_u[u],
                        post_u[u],
                        resmix_u[u],
                        layer.hc_ffn_fn,
                        layer.hc_ffn_scale,
                        layer.hc_ffn_base,
                        layer.rms_norm_eps,
                        layer.hc_eps,
                        layer.hc_eps,
                        layer.hc_post_alpha,
                        layer.hc_sinkhorn_iters,
                        pre_mix=pre_u[u],
                        norm_weight=layer.ffn_norm.weight,
                        norm_eps=layer.ffn_norm.variance_epsilon,
                        stream=None,
                        reduce_results=False,
                    )
                    flush_sm120_mhc_before_ar(dev_idx)
                    xs[u] = layer.ffn(xs[u], ids_u[u], mega_u[u])
                    xs[u] = _launch_ar(xs[u], u)
        finally:
            try:
                flush_sm120_mhc_before_ar(dev_idx)
                main_s.wait_stream(comm_s)
            except Exception as exc:
                _LOG.warning("TBO cleanup stream sync failed: %s", exc)
            fwd_ctx.attn_metadata = full_attn_meta
            for op, orig_buf in cand_orig_by_op.values():
                op.candidate_blocks = orig_buf

        hidden_out: torch.Tensor | None = None
        for u in (0, 1):
            main_s.wait_event(self._tbo_ar_done_ev[u])
            h = mhc_post(xs[u], res_u[u], post_u[u], resmix_u[u])
            res_u[u] = None
            post_u[u] = None
            resmix_u[u] = None
            if need_final_aux:
                _store_aux(dsv4_core.end_layer, u, h.mean(dim=1))
            if need_hidden:
                assert pre_u[u] is not None
                h = hc_collapse(h, pre_u[u])
                pre_u[u] = None
                normed = dsv4_core.norm(h)
                if hidden_out is None:
                    hidden_out = torch.empty(
                        (total_tokens, normed.shape[-1]),
                        dtype=normed.dtype,
                        device=normed.device,
                    )
                if u == 0:
                    hidden_out[:mid].copy_(normed)
                else:
                    hidden_out[mid:].copy_(normed)
        aux_out = [
            aux_full_by_layer[lid]
            for lid in aux_layers
            if lid in aux_full_by_layer
        ]
        return hidden_out, (aux_out if aux_out else None)

    def _sample_rows_into_argmax(
        self,
        hidden_states: torch.Tensor,
        sample_rows: torch.Tensor | None,
        count: int,
        *,
        all_ranks: bool = False,
        return_local_logits: bool = False,
    ) -> torch.Tensor | None:
        if count <= 0:
            return None
        if isinstance(hidden_states, tuple):
            hidden_states = hidden_states[0]
        sampled_h = (
            hidden_states[:count]
            if sample_rows is None
            else hidden_states.index_select(0, sample_rows)
        )
        if sample_rows is None and count <= self.max_graph_bucket:
            self.decode_hidden_gpu[:count].copy_(sampled_h)
        local_logits = self.model.compute_logits_local(sampled_h)
        self.argmax_channel.gather_top2_from_logits(
            local_logits,
            local_top2_vals_buf=self.local_max_vals_buf[:count],
            local_top2_ids_buf=self.local_max_ids_buf[:count],
            out_tokens=self.out_tokens_gpu[:count],
            out_logprob_bits=self.out_logprobs_gpu[:count],
            vocab_per_rank=self.vocab_per_rank,
            all_ranks=all_ranks,
        )
        return local_logits if return_local_logits else None

    def _apply_stochastic_sampling(
        self,
        plan: StepPlanView,
        local_logits: torch.Tensor,
        count: int,
        parity: int,
        *,
        use_penalty: bool = False,
        sample_slots_np: np.ndarray | None = None,
    ) -> None:
        np.copyto(self._sample_temp_np[parity][:count], plan.temperatures(count))
        np.copyto(self._sample_top_p_np[parity][:count], plan.top_p(count))
        np.copyto(self._sample_top_k_np[parity][:count], plan.top_k(count))
        np.copyto(self._sample_seed_np[parity][:count], plan.rng_seeds(count).view(np.int64))
        self.sample_temp_gpu[:count].copy_(
            self.sample_temp_cpu[parity, :count], non_blocking=True
        )
        self.sample_top_p_gpu[:count].copy_(
            self.sample_top_p_cpu[parity, :count], non_blocking=True
        )
        self.sample_top_k_gpu[:count].copy_(
            self.sample_top_k_cpu[parity, :count], non_blocking=True
        )
        self.sample_seed_gpu[:count].copy_(
            self.sample_seed_cpu[parity, :count], non_blocking=True
        )
        penalty_state: SlotPenaltyState | None = None
        slots_gpu: torch.Tensor | None = None
        reset_gpu: torch.Tensor | None = None
        pres_gpu: torch.Tensor | None = None
        freq_gpu: torch.Tensor | None = None
        if use_penalty and sample_slots_np is not None:
            if self._penalty_state is None:
                self._penalty_state = SlotPenaltyState(self.max_seqs, device=self.device)
            penalty_state = self._penalty_state
            np.copyto(self._sample_pres_pen_np[parity][:count], plan.presence_penalties(count))
            np.copyto(self._sample_freq_pen_np[parity][:count], plan.frequency_penalties(count))
            np.copyto(self._sample_reset_np[parity][:count], plan.penalty_reset_mask(count))
            self._sample_slots_np[parity][:count] = sample_slots_np
            self.sample_pres_pen_gpu[:count].copy_(
                self.sample_pres_pen_cpu[parity, :count], non_blocking=True
            )
            self.sample_freq_pen_gpu[:count].copy_(
                self.sample_freq_pen_cpu[parity, :count], non_blocking=True
            )
            self.sample_reset_gpu[:count].copy_(
                self.sample_reset_cpu[parity, :count], non_blocking=True
            )
            self.sample_slots_gpu[:count].copy_(
                self.sample_slots_cpu[parity, :count], non_blocking=True
            )
            slots_gpu = self.sample_slots_gpu[:count]
            reset_gpu = self.sample_reset_gpu[:count]
            pres_gpu = self.sample_pres_pen_gpu[:count]
            freq_gpu = self.sample_freq_pen_gpu[:count]
        sampled, sampled_lps, ru_toks, ru_lps = sample_tp8_logits(
            local_logits,
            temperatures=self.sample_temp_gpu[:count],
            top_p=self.sample_top_p_gpu[:count],
            top_k=self.sample_top_k_gpu[:count],
            rng_seeds=self.sample_seed_gpu[:count],
            tp_group=self.tp_device_group,
            world_size=self.tp_size,
            vocab_size=self.vocab_size,
            penalty_state=penalty_state,
            slots=slots_gpu,
            reset_mask=reset_gpu,
            presence_penalties=pres_gpu,
            frequency_penalties=freq_gpu,
            return_top2=True,
        )
        self.out_tokens_gpu[:count, 0].copy_(sampled)
        self.out_tokens_gpu[:count, 1].copy_(ru_toks)
        self.out_logprobs_gpu[:count, 0].copy_(sampled_lps.view(torch.int32))
        self.out_logprobs_gpu[:count, 1].copy_(ru_lps.view(torch.int32))

    @torch.inference_mode()
    def _warmup_and_capture_decode_graphs(self) -> None:
        from contextlib import ExitStack
        from flashinfer.autotuner import AutoTuner, set_autotune_process_group
        from vllm.config import CUDAGraphMode
        from vllm.forward_context import set_forward_context
        from vllm.model_executor.warmup.cutedsl_warmup import cutedsl_warmup
        from vllm.model_executor.warmup.kernel_warmup import _warmup_ll_bf16_router_gemm
        import vllm.utils.flashinfer as fi_utils
        from vllm.v1.worker.workspace import lock_workspace

        # Warm up CuTeDSL & low-latency BF16 router GEMM kernels (`M=1..16`, `(5120, 384)`).
        try:
            _warmup_ll_bf16_router_gemm(self.model)
        except (ImportError, RuntimeError, TypeError, ValueError) as exc:
            _LOG.warning(
                "Optional router BF16 warmup skipped on rank %d: %s", self.rank, exc
            )
        try:
            cutedsl_warmup()
        except (ImportError, RuntimeError, TypeError, ValueError) as exc:
            _LOG.warning(
                "Optional CuTeDSL warmup skipped on rank %d: %s", self.rank, exc
            )
        torch.cuda.empty_cache()

        # Warm up eager prefill/chunked-extend shapes across all Triton specialization
        # classes (`==1`, `%16 != 0`, `%16 == 0`), multi-seq batches, short contexts,
        # and long contexts (`seq_len > topk_tokens * compress_ratio`) so zero Triton
        # JIT compilations ever occur during `RuntimeLoop.run()`, while activating
        # FlashInfer SM120 CutlassMxfp8GemmRunner + Sparse MLA autotuned tactics.
        dummy_vb = self.num_kv_pages + 1  # scratch overflow block
        swa_dummy_vb = self.swa_total_128_blocks - 1
        self.mla_bt_gpu.fill_(dummy_vb)
        self.swa_bt_gpu[:, 0::2] = swa_dummy_vb * 2
        self.swa_bt_gpu[:, 1::2] = swa_dummy_vb * 2 + 1
        self.comp_bt_gpu.fill_(self.dummy_comp_block)
        self.lookback_gpu.fill_(-1)

        warmup_cases: list[tuple[list[int], int]] = [
            ([min(self.max_step_tokens, 8192) if self.pp_size > 1 else self.max_step_tokens], 0),
            ([1], 0),
            ([5], 0),
            ([16], 0),
            ([24], 0),
            ([24], 24),
            ([1, 24], 48),
            ([24, 24, 17], 0),
            ([64], 0),
            ([256], 0),
            ([512], 0),
            ([1024], 0),
            ([2048], 0),
            ([512] * 8, 0),
            ([1024] * 4, 0),
            ([2048, 2048], 0),
            ([1], 2560),
            ([24], 2560),
            ([17, 24, 13], 2560),
            ([64, 64], 2560),
            ([1] * min(15, self.max_seqs - 1) + [512], -1),
            ([1] * min(31, self.max_seqs - 1) + [1024], -1),
            ([1] * min(31, self.max_seqs - 1) + [512, 512], -2),
            ([1] * min(64, self.max_seqs - 1) + [1408], -1),
            ([1] * min(127, self.max_seqs - 1) + [1921], -1),
        ]
        tuning_buckets = fi_utils.flashinfer_get_hybrid_num_tokens_buckets(
            8192 if self.pp_size > 1 else max(self.max_step_tokens, 8192)
        )
        set_autotune_process_group(self.exchange_group)
        try:
            for case_idx, (q_lens, base_ctx) in enumerate(warmup_cases):
                n_reqs = len(q_lens)
                n_toks = sum(q_lens)
                if n_toks > self.token_ids_gpu.numel():
                    continue
                qsl = [0]
                pos_list: list[int] = []
                s_lens: list[int] = []
                is_pref_list: list[bool] = []
                for req_i, ql in enumerate(q_lens):
                    qsl.append(qsl[-1] + ql)
                    if base_ctx == -1:
                        rc = 512 if ql == 1 else 0
                    elif base_ctx == -2:
                        rc = 512 if ql == 1 else (1024 if req_i == n_reqs - 2 else 0)
                    else:
                        rc = base_ctx
                    pos_list.extend(range(rc, rc + ql))
                    s_lens.append(rc + ql)
                    is_pref_list.append(bool(ql > 1 or rc == 0))
                if base_ctx != 0:
                    self.lookback_gpu[:n_reqs].fill_(1)
                else:
                    self.lookback_gpu[:n_reqs].fill_(-1)

                qsl_cpu = torch.tensor(qsl, dtype=torch.int32)
                sl_cpu = torch.tensor(s_lens, dtype=torch.int32)
                self.mixed_is_prefilling_cpu[:n_reqs].copy_(
                    torch.tensor(is_pref_list, dtype=torch.bool)
                )
                self.query_start_loc_gpu[: n_reqs + 1].copy_(qsl_cpu)
                self.seq_lens_gpu[:n_reqs].copy_(sl_cpu)
                self.token_ids_gpu[:n_toks].copy_(
                    (torch.arange(n_toks, device=self.device, dtype=torch.int32) % 4096)
                    + 1
                )
                pos_t = torch.tensor(pos_list, dtype=torch.int64, device=self.device)
                self.positions_gpu[:n_toks].copy_(pos_t)
                self.slot_mapping_gpu[:n_toks].copy_(
                    dummy_vb * 128 + (pos_t % 128)
                )
                if self._swa_pool is not None:
                    self.swa_slot_mapping_gpu[:n_toks].copy_(
                        swa_dummy_vb * 128 + (pos_t % 128)
                    )
                self.sample_rows_gpu[0] = n_toks - 1

                attn_meta = self._build_step_attn_metadata(
                    num_reqs=n_reqs,
                    num_tokens=n_toks,
                    max_query_len=max(q_lens),
                    max_seq_len=max(s_lens),
                    qsl_gpu=self.query_start_loc_gpu[: n_reqs + 1],
                    qsl_cpu=qsl_cpu,
                    seq_lens_gpu=self.seq_lens_gpu[:n_reqs],
                    seq_lens_cpu=sl_cpu,
                    positions_gpu=self.positions_gpu[:n_toks],
                    slot_mapping_gpu=self.slot_mapping_gpu[:n_toks],
                    swa_slot_mapping_gpu=self.swa_slot_mapping_gpu[:n_toks],
                    is_prefilling_cpu=self.mixed_is_prefilling_cpu[:n_reqs],
                    for_capture=False,
                )
                with ExitStack() as stack:
                    if case_idx == 0:
                        stack.enter_context(
                            fi_utils.autotune(tuning_buckets=tuning_buckets)
                        )
                    stack.enter_context(
                        set_forward_context(
                            attn_meta,
                            self.vllm_config,
                            num_tokens=n_toks,
                            cudagraph_runtime_mode=CUDAGraphMode.NONE,
                        )
                    )
                    if self.pp_size > 1 and not self.is_first_pp:
                        lang_m = getattr(self.model, "language_model", self.model)
                        core_m = getattr(lang_m, "model", lang_m)
                        w_inter = core_m.make_empty_intermediate_tensors(
                            n_toks, torch.bfloat16, self.device
                        )
                        hidden = self.model(
                            input_ids=self.token_ids_gpu[:n_toks],
                            positions=self.positions_gpu[:n_toks],
                            intermediate_tensors=w_inter,
                            lookback_token_ids=self.lookback_gpu,
                        )
                    else:
                        hidden = self.model(
                            input_ids=self.token_ids_gpu[:n_toks],
                            positions=self.positions_gpu[:n_toks],
                            lookback_token_ids=self.lookback_gpu,
                        )
                    if self.is_last_pp:
                        self._sample_rows_into_argmax(hidden, self.sample_rows_gpu[:1], 1)
                    del hidden
                torch.cuda.synchronize(self.device)
                if n_toks >= 2048:
                    torch.cuda.empty_cache()
                dist.barrier(group=self.exchange_group)

            # Also warm up + autotune eager decode (`seq_len = 64..2563`) so all
            # FlashInfer SM120 sparse MLA decode tactics & persistent_topk kernels
            # are active before CUDAGraph capture.
            for warm_b, warm_sl in (
                (16, 2560),
                (1, 64),
                (1, 2561),
                (3, 2563),
                (32, 1536),
                (128, 1280),
            ):
                if warm_b > self.max_graph_bucket:
                    continue
                sl_cpu = torch.full((warm_b,), warm_sl, dtype=torch.int32)
                self.seq_lens_gpu[:warm_b].copy_(sl_cpu)
                self.token_ids_gpu[:warm_b].copy_(
                    torch.arange(warm_b, device=self.device, dtype=torch.int32) + 1
                )
                self.positions_gpu[:warm_b].fill_(warm_sl - 1)
                self.slot_mapping_gpu[:warm_b].fill_(
                    dummy_vb * 128 + ((warm_sl - 1) % 128)
                )
                if self._swa_pool is not None:
                    self.swa_slot_mapping_gpu[:warm_b].fill_(
                        swa_dummy_vb * 128 + ((warm_sl - 1) % 128)
                    )
                self.lookback_gpu[:warm_b].fill_(1)
                attn_meta = self._build_step_attn_metadata(
                    num_reqs=warm_b,
                    num_tokens=warm_b,
                    max_query_len=1,
                    max_seq_len=self.decode_graph_max_seq_len,
                    qsl_gpu=self.decode_qsl_gpu[: warm_b + 1],
                    qsl_cpu=self.decode_qsl_cpu[: warm_b + 1],
                    seq_lens_gpu=self.seq_lens_gpu[:warm_b],
                    seq_lens_cpu=sl_cpu,
                    positions_gpu=self.positions_gpu[:warm_b],
                    slot_mapping_gpu=self.slot_mapping_gpu[:warm_b],
                    swa_slot_mapping_gpu=self.swa_slot_mapping_gpu[:warm_b],
                    is_prefilling_cpu=self.decode_is_prefilling_cpu[:warm_b],
                    for_capture=True,
                )
                with ExitStack() as stack:
                    stack.enter_context(fi_utils.autotune(tune_mode=True))
                    stack.enter_context(
                        set_forward_context(
                            attn_meta,
                            self.vllm_config,
                            num_tokens=warm_b,
                            cudagraph_runtime_mode=CUDAGraphMode.NONE,
                        )
                    )
                    if self.pp_size > 1 and not self.is_first_pp:
                        from vllm.sequence import IntermediateTensors

                        assert (
                            self._pp_inter_hidden is not None
                            and self._pp_inter_pre_mix is not None
                        )
                        w_inter = IntermediateTensors(
                            {
                                "hidden_states": self._pp_inter_hidden[:warm_b],
                                "pre_mix": self._pp_inter_pre_mix[:warm_b],
                            }
                        )
                        hidden = self.model(
                            input_ids=self.token_ids_gpu[:warm_b],
                            positions=self.positions_gpu[:warm_b],
                            intermediate_tensors=w_inter,
                            lookback_token_ids=self.lookback_gpu,
                        )
                    else:
                        hidden = self.model(
                            input_ids=self.token_ids_gpu[:warm_b],
                            positions=self.positions_gpu[:warm_b],
                            lookback_token_ids=self.lookback_gpu,
                        )
                    if self.is_last_pp:
                        self._sample_rows_into_argmax(hidden, None, warm_b)
                    del hidden
                torch.cuda.synchronize(self.device)
                dist.barrier(group=self.exchange_group)
        finally:
            set_autotune_process_group(None)

        if self.rank == 0:
            try:
                zstep_cache = (
                    Path.home()
                    / ".cache"
                    / "vllm"
                    / "flashinfer_autotune_cache"
                    / "zstep_dsv41_sm120_autotune.json"
                )
                zstep_cache.parent.mkdir(parents=True, exist_ok=True)
                save_fn = getattr(AutoTuner.get(), "save_configs", None)
                if callable(save_fn):
                    save_fn(str(zstep_cache))
            except (OSError, RuntimeError, TypeError, ValueError) as exc:
                _LOG.warning(
                    "Failed to save FlashInfer autotune cache to %s: %s",
                    zstep_cache,
                    exc,
                )
        dist.barrier(group=self.exchange_group)

        # Zero all caches before capturing decode CUDAGraphs so Block 0 is clean.
        self._zero_all_caches()
        dist.barrier(group=self.exchange_group)

        gc.collect()
        torch.cuda.empty_cache()
        buckets = [b for b in DECODE_GRAPH_BUCKETS if b <= self.max_graph_bucket]
        if not buckets or buckets[-1] != self.max_graph_bucket:
            buckets.append(self.max_graph_bucket)

        self.mla_bt_gpu.zero_()
        self.swa_bt_gpu.zero_()
        self.comp_bt_gpu.fill_(self.dummy_comp_block)
        self.lookback_gpu.fill_(-1)
        self.slot_mapping_gpu.fill_(-1)
        if self._swa_pool is not None:
            self.swa_slot_mapping_gpu.fill_(-1)
        self.seq_lens_gpu.zero_()

        graph_pool = torch.cuda.graph_pool_handle()
        capture_stream = torch.cuda.Stream(device=self.device)

        for b in buckets:
            self.token_ids_gpu[:b].zero_()
            self.positions_gpu[:b].zero_()
            self.seq_lens_gpu[:b].fill_(1)
            self.seq_lens_gpu[b:].zero_()
            self.slot_mapping_gpu[:b].fill_(-1)
            sl_cpu = torch.ones(b, dtype=torch.int32)

            with torch.cuda.stream(capture_stream):
                for w_idx in range(2):
                    attn_meta = self._build_step_attn_metadata(
                        num_reqs=b,
                        num_tokens=b,
                        max_query_len=1,
                        max_seq_len=self.decode_graph_max_seq_len,
                        qsl_gpu=self.decode_qsl_gpu[: b + 1],
                        qsl_cpu=self.decode_qsl_cpu[: b + 1],
                        seq_lens_gpu=self.seq_lens_gpu[:b],
                        seq_lens_cpu=sl_cpu,
                        positions_gpu=self.positions_gpu[:b],
                        slot_mapping_gpu=self.slot_mapping_gpu[:b],
                        swa_slot_mapping_gpu=self.swa_slot_mapping_gpu[:b],
                        is_prefilling_cpu=self.decode_is_prefilling_cpu[:b],
                        for_capture=(w_idx == 0),
                    )
                    with set_forward_context(
                        attn_meta,
                        self.vllm_config,
                        num_tokens=b,
                        cudagraph_runtime_mode=CUDAGraphMode.NONE,
                    ):
                        if self.pp_size > 1 and not self.is_first_pp:
                            from vllm.sequence import IntermediateTensors

                            assert (
                                self._pp_inter_hidden is not None
                                and self._pp_inter_pre_mix is not None
                            )
                            d_inter = IntermediateTensors(
                                {
                                    "hidden_states": self._pp_inter_hidden[:b],
                                    "pre_mix": self._pp_inter_pre_mix[:b],
                                }
                            )
                            hidden = self.model(
                                input_ids=self.token_ids_gpu[:b],
                                positions=self.positions_gpu[:b],
                                intermediate_tensors=d_inter,
                                lookback_token_ids=self.lookback_gpu,
                            )
                        else:
                            hidden = self.model(
                                input_ids=self.token_ids_gpu[:b],
                                positions=self.positions_gpu[:b],
                                lookback_token_ids=self.lookback_gpu,
                            )
                        if self.is_last_pp:
                            self._sample_rows_into_argmax(
                                hidden, None, b, all_ranks=True
                            )
                        elif (
                            self._pp_inter_hidden is not None
                            and self._pp_inter_pre_mix is not None
                        ):
                            self._pp_inter_hidden[:b].copy_(hidden["hidden_states"])
                            self._pp_inter_pre_mix[:b].copy_(hidden["pre_mix"])
            capture_stream.synchronize()
            dist.barrier(group=self.exchange_group)

            with torch.cuda.stream(capture_stream):
                attn_meta = self._build_step_attn_metadata(
                    num_reqs=b,
                    num_tokens=b,
                    max_query_len=1,
                    max_seq_len=self.decode_graph_max_seq_len,
                    qsl_gpu=self.decode_qsl_gpu[: b + 1],
                    qsl_cpu=self.decode_qsl_cpu[: b + 1],
                    seq_lens_gpu=self.seq_lens_gpu[:b],
                    seq_lens_cpu=sl_cpu,
                    positions_gpu=self.positions_gpu[:b],
                    slot_mapping_gpu=self.slot_mapping_gpu[:b],
                    swa_slot_mapping_gpu=self.swa_slot_mapping_gpu[:b],
                    is_prefilling_cpu=self.decode_is_prefilling_cpu[:b],
                    for_capture=True,
                )
            capture_stream.synchronize()
            g = torch.cuda.CUDAGraph()
            with set_forward_context(
                attn_meta,
                self.vllm_config,
                num_tokens=b,
                cudagraph_runtime_mode=CUDAGraphMode.NONE,
            ):
                with torch.cuda.graph(g, pool=graph_pool, stream=capture_stream):
                    self._build_step_attn_metadata(
                        num_reqs=b,
                        num_tokens=b,
                        max_query_len=1,
                        max_seq_len=self.decode_graph_max_seq_len,
                        qsl_gpu=self.decode_qsl_gpu[: b + 1],
                        qsl_cpu=self.decode_qsl_cpu[: b + 1],
                        seq_lens_gpu=self.seq_lens_gpu[:b],
                        seq_lens_cpu=sl_cpu,
                        positions_gpu=self.positions_gpu[:b],
                        slot_mapping_gpu=self.slot_mapping_gpu[:b],
                        swa_slot_mapping_gpu=self.swa_slot_mapping_gpu[:b],
                        is_prefilling_cpu=self.decode_is_prefilling_cpu[:b],
                        for_capture=False,
                    )
                    if self.pp_size > 1 and not self.is_first_pp:
                        from vllm.sequence import IntermediateTensors

                        assert (
                            self._pp_inter_hidden is not None
                            and self._pp_inter_pre_mix is not None
                        )
                        d_inter = IntermediateTensors(
                            {
                                "hidden_states": self._pp_inter_hidden[:b],
                                "pre_mix": self._pp_inter_pre_mix[:b],
                            }
                        )
                        hidden = self.model(
                            input_ids=self.token_ids_gpu[:b],
                            positions=self.positions_gpu[:b],
                            intermediate_tensors=d_inter,
                            lookback_token_ids=self.lookback_gpu,
                        )
                    else:
                        hidden = self.model(
                            input_ids=self.token_ids_gpu[:b],
                            positions=self.positions_gpu[:b],
                            lookback_token_ids=self.lookback_gpu,
                        )
                    if self.is_last_pp:
                        self._sample_rows_into_argmax(
                            hidden, None, b, all_ranks=True
                        )
                    elif (
                        self._pp_inter_hidden is not None
                        and self._pp_inter_pre_mix is not None
                    ):
                        self._pp_inter_hidden[:b].copy_(hidden["hidden_states"])
                        self._pp_inter_pre_mix[:b].copy_(hidden["pre_mix"])
            capture_stream.synchronize()
            dist.barrier(group=self.exchange_group)
            self.decode_graphs[b] = g

        if (
            self.spec_method != "none"
            and not self.runtime_config.disable_verify_graphs
        ):
            self._capture_spec_verify_graphs(graph_pool, capture_stream)
        if self.dspark_drafter is not None:
            self.dspark_drafter.capture_graphs(graph_pool, capture_stream)
        dist.barrier(group=self.exchange_group)
        if self.pp_size == 1:
            self._capture_piecewise_prefill_graphs(graph_pool, capture_stream)
        else:
            # Warm up NCCL P2P communicators between Stage 0 and Stage 1 once at startup.
            assert self._pp_inter_hidden is not None and self._pp_inter_pre_mix is not None
            if self.is_first_pp:
                for req in self._pp_send_intermediate(
                    self._pp_inter_hidden[:4], self._pp_inter_pre_mix[:4]
                ):
                    req.wait()
            else:
                self._pp_recv_intermediate(
                    self._pp_inter_hidden[:4], self._pp_inter_pre_mix[:4]
                )
            torch.cuda.synchronize(self.device)
            dist.barrier(group=self.exchange_group)
        lock_workspace()
        gc.collect()
        torch.cuda.empty_cache()
        self._zero_all_caches()
        dist.barrier(group=self.exchange_group)
        if self.rank == 0:
            b12x_info = (
                f"b12x=on(moe={self.b12x_summary.moe_layers},"
                f"lin={self.b12x_summary.linear_layers}/{self.b12x_summary.linear_shapes},"
                f"prep_ms={self.b12x_summary.prepare_ms:.0f})"
                if self.b12x_summary.enabled
                else "b12x=off"
            )
            mg_info = (
                f"mega_gate=on({self.b12x_summary.mega_gate_layers}/"
                f"{self.b12x_summary.mega_gate_schedules},"
                f"min_m={self.b12x_summary.mega_gate_min_tokens})"
                if self.b12x_summary.mega_gate_layers > 0
                else "mega_gate=off"
            )
            dsa_info = (
                f"dsa_decode=on(cg_max_seq={self.decode_graph_max_seq_len})"
                if self.b12x_summary.dsa_decode_enabled
                else "dsa_decode=off"
            )
            p2p_info = (
                f"p2p={self.p2p_report.nccl_p2p_level}"
                f"(pxb_shm={'on' if self.shm_ar.pxb_enabled else 'off'})"
            )
            if (
                self.b12x_summary.dsa_decode_enabled
                and self.max_model_len > SM120_DSA_CANDIDATE_COVER_TOKENS
            ):
                _LOG.warning(
                    "max_model_len=%d exceeds SM120_DSA_CANDIDATE_COVER_TOKENS=%d; "
                    "decode CUDA graphs are clamped to %d tokens. Set "
                    "ZSTEP_ENABLE_SM120_DSA_DECODE=0 for full-context decode graphs.",
                    self.max_model_len,
                    SM120_DSA_CANDIDATE_COVER_TOKENS,
                    self.decode_graph_max_seq_len,
                )
            print(
                f"[DeepSeekV41StepRunner] ready: max_model_len={self.max_model_len} "
                f"kv_pages={self.num_kv_pages} buckets={buckets} "
                f"{b12x_info} {mg_info} {dsa_info} {p2p_info} "
                f"shm_ar_warm={self.shm_ar_calls} shm_ag_warm={self.shm_ag_calls} "
                f"nccl_large_warm={self.nccl_eager_large_calls} "
                f"nccl_fallback_warm={self.nccl_fallback_calls}",
                flush=True,
            )

    def _select_piecewise_bucket(self, total_tokens: int) -> int | None:
        for b in PREFILL_PIECEWISE_BUCKETS:
            if b in self.piecewise_graphs and total_tokens <= b:
                return b if (b <= 128 or (b - total_tokens) <= 128) else None
        return None

    def _install_piecewise_hooks(self) -> None:
        from vllm.compilation.breakable_cudagraph import BreakableCUDAGraphCapture
        from vllm.forward_context import get_forward_context, in_piecewise_cudagraph
        from vllm.models.deepseek_v41.common.engram import DEAD_ID
        from vllm.models.deepseek_v41.nvidia.model import image_sentinel_mask
        from vllm.utils.torch_utils import weak_ref_tensor

        lang_model = getattr(self.model, "language_model", self.model)
        inner_model = getattr(lang_model, "model", lang_model)
        eh = getattr(inner_model, "engram_hash", None)
        eh_prefix = getattr(inner_model, "engram_swa_prefix", None)
        if eh is not None and eh_prefix is not None:
            orig_eh = eh.forward
            n_lyr, max_ng = eh.multipliers.shape
            h_cols = (max_ng - 1) * eh.primes.shape[-1]

            def _pw_eh(ids, pos, qsl, dead, lb, lb_dead, sm, bt):
                cap = BreakableCUDAGraphCapture.current()
                if cap is None or not cap._capturing or not in_piecewise_cudagraph():
                    return orig_eh(ids, pos, qsl, dead, lb, lb_dead, sm, bt)
                b_toks = int(ids.shape[0])
                out = ids.new_full((b_toks, n_lyr, h_cols), DEAD_ID, dtype=torch.int32)
                w_ids, w_pos, w_dead, w_out = (
                    weak_ref_tensor(ids), weak_ref_tensor(pos),
                    weak_ref_tensor(dead), weak_ref_tensor(out),
                )

                def _run_eh() -> None:
                    if BreakableCUDAGraphCapture.is_active():
                        w_out.fill_(DEAD_ID)
                        return
                    swa = get_forward_context().attn_metadata[eh_prefix]
                    n = self._pw_live_num_tokens or int(swa.num_decode_tokens + swa.num_prefill_tokens)
                    live_lb = self._pw_live_lookback if self._pw_live_lookback is not None else lb
                    res = orig_eh(
                        w_ids[:n], w_pos[:n], swa.query_start_loc, w_dead[:n],
                        live_lb, image_sentinel_mask(live_lb), swa.slot_mapping, swa.block_table,
                    )
                    w_out[:n].copy_(res)
                    if n < b_toks:
                        w_out[n:].fill_(DEAD_ID)

                cap.add_eager(_run_eh)
                return out

            eh.forward = _pw_eh  # type: ignore[method-assign]

        for lyr in inner_model.layers:
            attn = getattr(lyr, "attn", None)
            if attn is None or not hasattr(attn, "_prepare_and_attn"):
                continue
            orig_paa = attn._prepare_and_attn

            def _make_paa(_orig: Any) -> Any:
                def _pw_paa(h, qr, kv, qr_s, kv_s, idx_w, pos, out):
                    if in_piecewise_cudagraph():
                        if BreakableCUDAGraphCapture.is_active():
                            out.zero_()
                            return None
                        n = self._pw_live_num_tokens
                        if 0 < n < int(h.shape[0]):
                            return _orig(
                                h[:n], _slice_piecewise_arg(qr, n), kv[:n],
                                _slice_piecewise_arg(qr_s, n), _slice_piecewise_arg(kv_s, n),
                                _slice_piecewise_arg(idx_w, n), pos[:n], out[:n],
                            )
                    return _orig(h, qr, kv, qr_s, kv_s, idx_w, pos, out)
                return _pw_paa

            attn._prepare_and_attn = _make_paa(orig_paa)
            attn._prepare_and_attn_fn = attn._prepare_and_attn_eager

    @torch.inference_mode()
    def _capture_piecewise_prefill_graphs(
        self, graph_pool: Any, capture_stream: torch.cuda.Stream
    ) -> None:
        from vllm.compilation.breakable_cudagraph import (
            BreakableCUDAGraphCapture, is_breakable_cudagraph_enabled,
        )
        from vllm.config import CUDAGraphMode
        from vllm.forward_context import set_forward_context

        if not is_breakable_cudagraph_enabled():
            return
        self._install_piecewise_hooks()
        torch.cuda.empty_cache()
        for b in reversed(PREFILL_PIECEWISE_BUCKETS):
            n_pages = (b + 127) // 128
            if (
                b > self.max_step_tokens
                or n_pages > self.num_kv_pages
                or n_pages > self.swa_total_128_blocks - 2
            ):
                continue
            self.token_ids_gpu[:b].zero_()
            self.positions_gpu[:b].copy_(torch.arange(b, dtype=torch.int64, device=self.device))
            self.mla_bt_cpu[0].zero_()
            self.swa_bt_cpu[0].zero_()
            for p in range(n_pages):
                pg = 1 + p
                self.mla_bt_cpu[0, p] = pg
                self.swa_bt_cpu[0, 2 * p], self.swa_bt_cpu[0, 2 * p + 1] = pg * 2, pg * 2 + 1
            self.comp_bt_cpu[0, 0] = 1
            self.mla_bt_gpu[:1].copy_(self.mla_bt_cpu[:1])
            self.swa_bt_gpu[:1].copy_(self.swa_bt_cpu[:1])
            self.comp_bt_gpu[:1].copy_(self.comp_bt_cpu[:1])
            self.lookback_gpu[:1].fill_(-1)
            self.slot_mapping_gpu[:b].fill_(-1)
            self.swa_slot_mapping_gpu[:b].fill_(-1)
            self.seq_lens_cpu[0], self.query_start_loc_cpu[0], self.query_start_loc_cpu[1] = b, 0, b
            self.seq_lens_gpu[:1].fill_(b)
            self.query_start_loc_gpu[:2].copy_(self.query_start_loc_cpu[:2])
            self.mixed_is_prefilling_cpu[0] = True
            attn_meta = self._build_step_attn_metadata(
                num_reqs=1, num_tokens=b, max_query_len=b, max_seq_len=b,
                qsl_gpu=self.query_start_loc_gpu[:2], qsl_cpu=self.query_start_loc_cpu[:2],
                seq_lens_gpu=self.seq_lens_gpu[:1], seq_lens_cpu=self.seq_lens_cpu[:1],
                positions_gpu=self.positions_gpu[:b], slot_mapping_gpu=self.slot_mapping_gpu[:b],
                swa_slot_mapping_gpu=self.swa_slot_mapping_gpu[:b],
                is_prefilling_cpu=self.mixed_is_prefilling_cpu[:1], for_capture=False,
            )
            self._pw_live_lookback, self._pw_live_num_tokens = self.lookback_gpu[:1], b
            with torch.cuda.stream(capture_stream):
                for _ in range(2):
                    with set_forward_context(attn_meta, self.vllm_config, num_tokens=b, cudagraph_runtime_mode=CUDAGraphMode.PIECEWISE):
                        self.model(input_ids=self.token_ids_gpu[:b], positions=self.positions_gpu[:b], lookback_token_ids=self.lookback_gpu[:1])
            capture_stream.synchronize()
            dist.barrier(group=self.exchange_group)
            cap = BreakableCUDAGraphCapture(pool=graph_pool)
            with torch.cuda.stream(capture_stream):
                with set_forward_context(attn_meta, self.vllm_config, num_tokens=b, cudagraph_runtime_mode=CUDAGraphMode.PIECEWISE):
                    with cap:
                        out = self.model(input_ids=self.token_ids_gpu[:b], positions=self.positions_gpu[:b], lookback_token_ids=self.lookback_gpu[:1])
            capture_stream.synchronize()
            dist.barrier(group=self.exchange_group)
            if isinstance(out, tuple):
                out_hidden, out_aux = out
                self.piecewise_outputs[b] = out_hidden
                self.piecewise_aux_outputs[b] = out_aux
            else:
                self.piecewise_outputs[b] = out
                self.piecewise_aux_outputs[b] = None
            self.piecewise_graphs[b] = cap
        self._pw_live_lookback, self._pw_live_num_tokens = None, 0

    @torch.inference_mode()
    def _capture_spec_verify_graphs(
        self,
        graph_pool: Any,
        capture_stream: torch.cuda.Stream,
    ) -> None:
        from vllm.config import CUDAGraphMode
        from vllm.forward_context import set_forward_context

        width = self.num_spec_tokens + 1
        self.mla_bt_gpu.zero_()
        self.swa_bt_gpu.zero_()
        self.comp_bt_gpu.fill_(self.dummy_comp_block)
        self.lookback_gpu.fill_(-1)
        self.slot_mapping_gpu.fill_(-1)
        if self._swa_pool is not None:
            self.swa_slot_mapping_gpu.fill_(-1)
        self.seq_lens_gpu.zero_()

        for bucket in self.verify_buckets:
            rows = bucket * width
            sl_cpu = torch.full((bucket,), width, dtype=torch.int32)
            self.token_ids_gpu[:rows].zero_()
            self.positions_gpu[:rows].copy_(
                torch.arange(rows, dtype=torch.int64, device=self.device) % width
            )
            self.lookback_gpu.fill_(-1)
            self.slot_mapping_gpu[:rows].fill_(-1)
            self.seq_lens_gpu[:bucket].fill_(width)
            self.seq_lens_gpu[bucket:].zero_()

            with torch.cuda.stream(capture_stream):
                for w_idx in range(2):
                    attn_meta = self._build_step_attn_metadata(
                        num_reqs=bucket,
                        num_tokens=rows,
                        max_query_len=width,
                        max_seq_len=self.decode_graph_max_seq_len,
                        qsl_gpu=self.verify_qsl_gpu[: bucket + 1],
                        qsl_cpu=self.verify_qsl_cpu[: bucket + 1],
                        seq_lens_gpu=self.seq_lens_gpu[:bucket],
                        seq_lens_cpu=sl_cpu,
                        positions_gpu=self.positions_gpu[:rows],
                        slot_mapping_gpu=self.slot_mapping_gpu[:rows],
                        swa_slot_mapping_gpu=self.swa_slot_mapping_gpu[:rows],
                        is_prefilling_cpu=self.decode_is_prefilling_cpu[:bucket],
                        for_capture=(w_idx == 0),
                    )
                    with set_forward_context(
                        attn_meta,
                        self.vllm_config,
                        num_tokens=rows,
                        cudagraph_runtime_mode=CUDAGraphMode.NONE,
                    ):
                        if self.pp_size > 1 and not self.is_first_pp:
                            from vllm.sequence import IntermediateTensors

                            assert (
                                self._pp_inter_hidden is not None
                                and self._pp_inter_pre_mix is not None
                            )
                            v_inter = IntermediateTensors(
                                {
                                    "hidden_states": self._pp_inter_hidden[:rows],
                                    "pre_mix": self._pp_inter_pre_mix[:rows],
                                }
                            )
                            output = self.model(
                                input_ids=self.token_ids_gpu[:rows],
                                positions=self.positions_gpu[:rows],
                                intermediate_tensors=v_inter,
                                lookback_token_ids=self.lookback_gpu,
                            )
                        else:
                            output = self.model(
                                input_ids=self.token_ids_gpu[:rows],
                                positions=self.positions_gpu[:rows],
                                lookback_token_ids=self.lookback_gpu,
                            )
                        if self.is_last_pp:
                            hidden = output[0] if isinstance(output, tuple) else output
                            self._sample_rows_into_argmax(
                                hidden, None, rows, all_ranks=True
                            )
                        elif (
                            self._pp_inter_hidden is not None
                            and self._pp_inter_pre_mix is not None
                        ):
                            self._pp_inter_hidden[:rows].copy_(output["hidden_states"])
                            self._pp_inter_pre_mix[:rows].copy_(output["pre_mix"])
            capture_stream.synchronize()
            dist.barrier(group=self.exchange_group)

            with torch.cuda.stream(capture_stream):
                attn_meta = self._build_step_attn_metadata(
                    num_reqs=bucket,
                    num_tokens=rows,
                    max_query_len=width,
                    max_seq_len=self.decode_graph_max_seq_len,
                    qsl_gpu=self.verify_qsl_gpu[: bucket + 1],
                    qsl_cpu=self.verify_qsl_cpu[: bucket + 1],
                    seq_lens_gpu=self.seq_lens_gpu[:bucket],
                    seq_lens_cpu=sl_cpu,
                    positions_gpu=self.positions_gpu[:rows],
                    slot_mapping_gpu=self.slot_mapping_gpu[:rows],
                    swa_slot_mapping_gpu=self.swa_slot_mapping_gpu[:rows],
                    is_prefilling_cpu=self.decode_is_prefilling_cpu[:bucket],
                    for_capture=True,
                )
            capture_stream.synchronize()
            graph = torch.cuda.CUDAGraph()
            aux: list[torch.Tensor] | None = None
            with set_forward_context(
                attn_meta,
                self.vllm_config,
                num_tokens=rows,
                cudagraph_runtime_mode=CUDAGraphMode.NONE,
            ):
                with torch.cuda.graph(graph, pool=graph_pool, stream=capture_stream):
                    self._build_step_attn_metadata(
                        num_reqs=bucket,
                        num_tokens=rows,
                        max_query_len=width,
                        max_seq_len=self.decode_graph_max_seq_len,
                        qsl_gpu=self.verify_qsl_gpu[: bucket + 1],
                        qsl_cpu=self.verify_qsl_cpu[: bucket + 1],
                        seq_lens_gpu=self.seq_lens_gpu[:bucket],
                        seq_lens_cpu=sl_cpu,
                        positions_gpu=self.positions_gpu[:rows],
                        slot_mapping_gpu=self.slot_mapping_gpu[:rows],
                        swa_slot_mapping_gpu=self.swa_slot_mapping_gpu[:rows],
                        is_prefilling_cpu=self.decode_is_prefilling_cpu[:bucket],
                        for_capture=False,
                    )
                    if self.pp_size > 1 and not self.is_first_pp:
                        from vllm.sequence import IntermediateTensors

                        assert (
                            self._pp_inter_hidden is not None
                            and self._pp_inter_pre_mix is not None
                        )
                        v_inter = IntermediateTensors(
                            {
                                "hidden_states": self._pp_inter_hidden[:rows],
                                "pre_mix": self._pp_inter_pre_mix[:rows],
                            }
                        )
                        output = self.model(
                            input_ids=self.token_ids_gpu[:rows],
                            positions=self.positions_gpu[:rows],
                            intermediate_tensors=v_inter,
                            lookback_token_ids=self.lookback_gpu,
                        )
                    else:
                        output = self.model(
                            input_ids=self.token_ids_gpu[:rows],
                            positions=self.positions_gpu[:rows],
                            lookback_token_ids=self.lookback_gpu,
                        )
                    if self.is_last_pp:
                        if isinstance(output, tuple):
                            hidden, aux = output
                        else:
                            hidden, aux = output, None
                        self._sample_rows_into_argmax(
                            hidden, None, rows, all_ranks=True
                        )
                    elif (
                        self._pp_inter_hidden is not None
                        and self._pp_inter_pre_mix is not None
                    ):
                        self._pp_inter_hidden[:rows].copy_(output["hidden_states"])
                        self._pp_inter_pre_mix[:rows].copy_(output["pre_mix"])
            capture_stream.synchronize()
            dist.barrier(group=self.exchange_group)
            self.verify_graphs[bucket] = graph
            self.verify_graph_aux[bucket] = aux

    def watched_triton_kernels(self) -> list[tuple[str, object]]:
        import vllm.models.deepseek_v41.attention as dsv41_attn
        import vllm.models.deepseek_v41.common.engram as dsv41_engram
        import vllm.models.deepseek_v41.compressor as dsv41_comp

        candidates: list[tuple[str, object]] = [
            ("dsv41_comp_ring_slot", getattr(dsv41_comp, "_ring_slot_mapping_kernel", None)),
            ("dsv41_engram_hash_ids", getattr(dsv41_engram, "_hash_ids_kernel", None)),
            ("dsv41_idx_short_ctx", getattr(dsv41_attn, "_fill_short_context_topk_indices", None)),
            ("dsv41_idx_mask_cand", getattr(dsv41_attn, "_mask_candidates_kernel", None)),
            ("dsv41_idx_block_scores", getattr(dsv41_attn, "_block_scores_kernel", None)),
        ]
        return [
            (name, k)
            for name, k in candidates
            if k is not None and hasattr(k, "_do_compile")
        ]

    def _speculative_outputs(
        self,
        *,
        parity: int,
        num_sample: int,
        sample_rows_np: np.ndarray,
        slot_np: np.ndarray,
        tok_np: np.ndarray,
        pos_np: np.ndarray,
        q_start_np: np.ndarray,
        pt_np: np.ndarray,
        aux_hidden_states: list[torch.Tensor] | None,
    ) -> list[SeqResult]:
        tokens = self._out_tokens_np[parity][:num_sample, 0]
        logprobs = self._out_logprobs_np[parity][:num_sample, 0]
        verified = verify_batch(
            predictions=tokens,
            logprob_bits=logprobs,
            token_ids=tok_np,
            sample_rows=sample_rows_np,
            query_start_loc=q_start_np,
            req_slot_ids=slot_np,
        )
        drafts: dict[int, list[int]] = {}
        if self.dspark_drafter is not None:
            if aux_hidden_states is None:
                raise RuntimeError("DSpark target forward returned no auxiliary hidden states")
            drafts = self.dspark_drafter.draft(
                verified,
                aux_hidden_states=aux_hidden_states,
                positions=pos_np,
                req_slot_ids=slot_np,
                query_start_loc=q_start_np,
                page_table=pt_np,
            )
        return seq_results(verified, drafts) if self.is_result_rank else []

    def _commit_decode_slot_state(self, n: int) -> None:
        s_idx = self.slots_gpu[:n]
        self.next_lb_gpu[:n, 0].copy_(self.token_ids_gpu[:n])
        self.next_lb_gpu[:n, 1:3].copy_(self.lookback_gpu[:n, 0:2])
        self.slot_lb_gpu.index_copy_(0, s_idx, self.next_lb_gpu[:n])
        self.slot_last_tok_gpu.index_copy_(0, s_idx, self.out_tokens_gpu[:n, 0])

    @staticmethod
    def _can_use_ragged_verify_graph(
        *,
        q_start_np: np.ndarray,
        q_lens: np.ndarray,
        pos_np: np.ndarray,
        seq_lens_np: np.ndarray,
        sample_rows_np: np.ndarray,
        total_tokens: int,
        width: int,
        max_seq_len: int,
    ) -> bool:
        """Return True when a heterogeneous speculative-verify batch (`1 <= q_len <= width`)
        can be safely padded into the uniform `verify_graphs[v_bucket]` layout without
        mutating KV caches (`slot_mapping = -1` on padded slots) or altering causal
        attention windows (`seq_len += width - q_len`)."""
        num_seqs = len(q_lens)
        if num_seqs == 0 or total_tokens == 0:
            return False
        if not bool(np.all((q_lens >= 1) & (q_lens <= width))):
            return False
        if not bool(
            np.array_equal(
                sample_rows_np, np.arange(total_tokens, dtype=sample_rows_np.dtype)
            )
        ):
            return False
        p_min = pos_np[q_start_np[:num_seqs]].astype(np.int64)
        p_max = pos_np[q_start_np[1:] - 1].astype(np.int64)
        if not bool(np.all(p_min > 0)) or not bool(np.all(seq_lens_np == (p_max + 1))):
            return False
        p_max_pad = p_max + (width - q_lens).astype(np.int64)
        return bool(
            np.all(p_max_pad < max_seq_len)
            and np.all((p_max_pad // 128) == (p_max // 128))
        )

    @staticmethod
    def _pack_ragged_verify_buffers(
        *,
        num_seqs: int,
        v_bucket: int,
        width: int,
        q_start_np: np.ndarray,
        q_lens: np.ndarray,
        tok_np: np.ndarray,
        pos_i64: np.ndarray,
        seq_lens_np: np.ndarray,
        sm_packed: np.ndarray,
        swa_sm_packed: np.ndarray | None,
        tok_cpu_np: np.ndarray,
        pos_cpu_np: np.ndarray,
        sl_cpu_np: np.ndarray,
        sm_cpu_np: np.ndarray,
        swa_sm_np: np.ndarray | None,
        real_rows_np: np.ndarray,
    ) -> None:
        """Scatter ragged speculative-verify tokens into fixed-width `[v_bucket, width]`
        CUDAGraph staging buffers while keeping padded tail slots side-effect-free."""
        rows = v_bucket * width
        tok_cpu_np[:] = 0
        pos_cpu_np[:] = np.arange(rows, dtype=np.int64) % width
        sm_cpu_np[:] = -1
        if swa_sm_np is not None:
            swa_sm_np[:] = -1
        write_idx = 0
        for s_i in range(num_seqs):
            src_s = int(q_start_np[s_i])
            src_e = int(q_start_np[s_i + 1])
            ql = src_e - src_s
            dst_s = s_i * width
            tok_cpu_np[dst_s : dst_s + ql] = tok_np[src_s:src_e]
            pos_cpu_np[dst_s : dst_s + ql] = pos_i64[src_s:src_e]
            sm_cpu_np[dst_s : dst_s + ql] = sm_packed[src_s:src_e]
            if swa_sm_np is not None and swa_sm_packed is not None:
                swa_sm_np[dst_s : dst_s + ql] = swa_sm_packed[src_s:src_e]
            if ql < width:
                last_p = int(pos_i64[src_e - 1])
                for p_off in range(1, width - ql + 1):
                    pos_cpu_np[dst_s + ql + p_off - 1] = last_p + p_off
            for j in range(ql):
                real_rows_np[write_idx + j] = dst_s + j
            write_idx += ql
        sl_cpu_np[:num_seqs] = seq_lens_np + (width - q_lens)
        if v_bucket > num_seqs:
            sl_cpu_np[num_seqs:v_bucket] = width

    @torch.inference_mode()
    def execute(
        self, plan: StepPlanView, header: StepHeader
    ) -> tuple[list[SeqResult], int]:
        return self.finish(self.launch(plan, header))

    @torch.inference_mode()
    def launch(self, plan: StepPlanView, header: StepHeader) -> _LaunchedStep | None:
        from vllm.config import CUDAGraphMode
        from vllm.forward_context import set_forward_context

        num_seqs = header.num_seqs
        total_tokens = header.num_encoder_tokens
        num_sample = header.num_sample_rows
        if total_tokens == 0 or num_seqs == 0:
            return None
        parity = header.step_id & 1

        tok_np = plan.token_ids(total_tokens)
        pos_np = plan.positions(total_tokens)
        slot_np = plan.req_slot_ids(total_tokens)
        seq_lens_np = plan.seq_lens(num_seqs)
        q_start_np = plan.query_start_loc(num_seqs + 1)
        prev_win_np = plan.engram_prev_window(num_seqs)
        sample_rows_np = plan.sample_row_indices(num_sample)

        max_slot_end = min(self.max_seqs, int(slot_np.max()) + 1)
        pt_np = plan.global_kv_page_table(max_slot_end)[:, : self.bt_width]
        if self._swa_pool is not None:
            self._swa_pool.prune_inactive_slots(
                plan.global_kv_page_table(self.geometry.max_seqs)
            )

        is_spec_step = header.is_dspark_verify != 0
        is_decode = (
            (header.flags & STEP_FLAG_DECODE_ONLY) != 0 and not is_spec_step
        )
        is_sampling = (header.flags & STEP_FLAG_SAMPLING) != 0
        is_penalty = (header.flags & STEP_FLAG_PENALTY) != 0
        use_dev_tok = (
            self.pp_size == 1
            and is_decode
            and (header.flags & STEP_FLAG_DEVICE_TOKENS) != 0
        )
        width = self.num_spec_tokens + 1
        q_lens = q_start_np[1:] - q_start_np[:-1]
        v_bucket = (
            select_decode_bucket(num_seqs, self.verify_buckets[-1])
            if self.verify_buckets and num_seqs <= self.verify_buckets[-1]
            else 0
        )
        is_uniform_verify = bool(
            total_tokens == num_seqs * width and np.all(q_lens == width)
        )
        is_ragged_verify = (
            is_spec_step
            and not is_uniform_verify
            and v_bucket in self.verify_graphs
            and num_sample == total_tokens
            and self._can_use_ragged_verify_graph(
                q_start_np=q_start_np,
                q_lens=q_lens,
                pos_np=pos_np,
                seq_lens_np=seq_lens_np,
                sample_rows_np=sample_rows_np,
                total_tokens=total_tokens,
                width=width,
                max_seq_len=self.decode_graph_max_seq_len,
            )
        )
        use_verify_graph = (
            is_spec_step
            and v_bucket in self.verify_graphs
            and num_sample == total_tokens
            and int(seq_lens_np.max()) <= self.decode_graph_max_seq_len
            and (is_uniform_verify or is_ragged_verify)
        )
        aux_hidden_states: list[torch.Tensor] | None = None
        self._last_verify_bt_sig = None
        self._last_verify_graph_bucket = None
        self._h2d_done_event.synchronize()

        self.start_events[parity].record()
        if is_decode and int(seq_lens_np.max()) <= self.decode_graph_max_seq_len:
            bucket = select_decode_bucket(num_seqs, self.max_graph_bucket)
            self.slots_cpu[:num_seqs].numpy()[:] = slot_np
            self.slots_gpu[:num_seqs].copy_(
                self.slots_cpu[:num_seqs], non_blocking=True
            )

            pos_i64 = pos_np.astype(np.int64)
            c_lo = np.maximum(0, (pos_i64 - 127) // 128)
            c_hi = pos_i64 // 128
            swa_epoch = self._swa_pool.epoch if self._swa_pool is not None else -1
            bt_key = (
                bucket,
                swa_epoch,
                tuple(int(s) for s in slot_np),
                tuple(int(c) for c in c_lo),
                tuple(int(c) for c in c_hi),
                tuple(int(pt_np[s, c]) for s, c in zip(slot_np, c_hi, strict=True)),
                tuple(int(pt_np[s, 0]) for s in slot_np),
            )
            bt_hit = self._matches_gpu_bt_cache(bt_key, pt_np, slot_np, c_hi)
            if not bt_hit:
                active_pt = pt_np[slot_np, :] + 1
                mla_bt_np = self.mla_bt_cpu[:bucket].numpy()
                mla_bt_np[:num_seqs] = active_pt
                if bucket > num_seqs:
                    mla_bt_np[num_seqs:bucket] = 0

                if self._swa_pool is not None:
                    swa_pt, _ = self._swa_pool.map_active_windows(
                        active_pt, pos_i64, pos_i64, pos_i64, active_slots=slot_np
                    )
                else:
                    swa_pt = active_pt

                swa_bt_np = self.swa_bt_cpu[:bucket].numpy()
                swa_bt_np[:num_seqs, 0::2] = swa_pt * 2
                swa_bt_np[:num_seqs, 1::2] = swa_pt * 2 + 1
                if bucket > num_seqs:
                    swa_bt_np[num_seqs:bucket] = 0

                comp_bt_np = self.comp_bt_cpu[:bucket, 0].numpy()
                comp_bt_np[:num_seqs] = slot_np + 1
                if bucket > num_seqs:
                    comp_bt_np[num_seqs:bucket] = self.dummy_comp_block

            if not use_dev_tok:
                lb_np = self.lookback_cpu[:bucket].numpy()
                lb_np[:num_seqs, 0] = prev_win_np[:, 2]
                lb_np[:num_seqs, 1] = prev_win_np[:, 1]
                lb_np[:num_seqs, 2] = prev_win_np[:, 0]
                if bucket > num_seqs:
                    lb_np[num_seqs:bucket] = -1

                tok_cpu_np = self.token_ids_cpu[:bucket].numpy()
                tok_cpu_np[:num_seqs] = tok_np
                if bucket > num_seqs:
                    tok_cpu_np[num_seqs:bucket] = 0

            pos_cpu_np = self.positions_cpu[:bucket].numpy()
            pos_cpu_np[:num_seqs] = pos_i64
            if bucket > num_seqs:
                pos_cpu_np[num_seqs:bucket] = 0

            sl_cpu_np = self.seq_lens_cpu[:bucket].numpy()
            sl_cpu_np[:num_seqs] = seq_lens_np
            if bucket > num_seqs:
                sl_cpu_np[num_seqs:bucket] = 0

            sm_cpu_np = self.slot_mapping_cpu[:bucket].numpy()
            vb = (pt_np[slot_np, c_hi] + 1).astype(np.int64)
            sm_cpu_np[:num_seqs] = vb * 128 + (pos_i64 % 128)
            if bucket > num_seqs:
                sm_cpu_np[num_seqs:bucket] = -1
            if self._swa_pool is not None:
                swa_sm_np = self.swa_slot_mapping_cpu[:bucket].numpy()
                swa_vb = self._swa_pool.lookup(vb).astype(np.int64)
                swa_sm_np[:num_seqs] = swa_vb * 128 + (pos_i64 % 128)
                if bucket > num_seqs:
                    swa_sm_np[num_seqs:bucket] = -1

            if not bt_hit:
                self.mla_bt_gpu[:bucket].copy_(self.mla_bt_cpu[:bucket], non_blocking=True)
                self.swa_bt_gpu[:bucket].copy_(self.swa_bt_cpu[:bucket], non_blocking=True)
                self.comp_bt_gpu[:bucket].copy_(
                    self.comp_bt_cpu[:bucket], non_blocking=True
                )
                post_epoch = self._swa_pool.epoch if self._swa_pool is not None else -1
                self._store_gpu_bt_cache(
                    (
                        bucket,
                        post_epoch,
                        bt_key[2],
                        bt_key[3],
                        bt_key[4],
                        bt_key[5],
                        bt_key[6],
                    ),
                    pt_np,
                    slot_np,
                )
            if use_dev_tok:
                s_idx = self.slots_gpu[:num_seqs]
                torch.index_select(
                    self.slot_last_tok_gpu, 0, s_idx, out=self.token_ids_gpu[:num_seqs]
                )
                torch.index_select(
                    self.slot_lb_gpu, 0, s_idx, out=self.lookback_gpu[:num_seqs]
                )
                if bucket > num_seqs:
                    self.token_ids_gpu[num_seqs:bucket].zero_()
                    self.lookback_gpu[num_seqs:bucket].fill_(-1)
            if self._sm_h2d.enabled:
                if not use_dev_tok:
                    self._sm_h2d.copy_async(
                        self.lookback_gpu[:bucket], self.lookback_cpu[:bucket]
                    )
                    self._sm_h2d.copy_async(
                        self.token_ids_gpu[:bucket], self.token_ids_cpu[:bucket]
                    )
                self._sm_h2d.copy_async(
                    self.positions_gpu[:bucket], self.positions_cpu[:bucket]
                )
                self._sm_h2d.copy_async(
                    self.seq_lens_gpu[:bucket], self.seq_lens_cpu[:bucket]
                )
                self._sm_h2d.copy_async(
                    self.slot_mapping_gpu[:bucket], self.slot_mapping_cpu[:bucket]
                )
            else:
                if not use_dev_tok:
                    self.lookback_gpu[:bucket].copy_(
                        self.lookback_cpu[:bucket], non_blocking=True
                    )
                    self.token_ids_gpu[:bucket].copy_(
                        self.token_ids_cpu[:bucket], non_blocking=True
                    )
                self.positions_gpu[:bucket].copy_(
                    self.positions_cpu[:bucket], non_blocking=True
                )
                self.seq_lens_gpu[:bucket].copy_(
                    self.seq_lens_cpu[:bucket], non_blocking=True
                )
                self.slot_mapping_gpu[:bucket].copy_(
                    self.slot_mapping_cpu[:bucket], non_blocking=True
                )
            if self._swa_pool is not None:
                self.swa_slot_mapping_gpu[:bucket].copy_(
                    self.swa_slot_mapping_cpu[:bucket], non_blocking=True
                )

            self._h2d_done_event.record()
            if self.pp_size > 1 and not self.is_first_pp:
                assert (
                    self._pp_inter_hidden is not None
                    and self._pp_inter_pre_mix is not None
                )
                self._pp_recv_intermediate(
                    self._pp_inter_hidden[:bucket], self._pp_inter_pre_mix[:bucket]
                )
            self.decode_graphs[bucket].replay()
            if self.pp_size > 1 and self.is_first_pp:
                assert (
                    self._pp_inter_hidden is not None
                    and self._pp_inter_pre_mix is not None
                )
                for req in self._pp_send_intermediate(
                    self._pp_inter_hidden[:bucket], self._pp_inter_pre_mix[:bucket]
                ):
                    req.wait()
            if self.is_last_pp and is_sampling:
                local_logits = self.model.compute_logits_local(
                    self.decode_hidden_gpu[:num_seqs]
                )
                self._apply_stochastic_sampling(
                    plan,
                    local_logits,
                    num_seqs,
                    parity,
                    use_penalty=is_penalty,
                    sample_slots_np=slot_np,
                )
            if self.pp_size == 1:
                self._commit_decode_slot_state(num_seqs)

            if self.is_result_rank:
                self.out_tokens_cpu[parity, :num_seqs].copy_(
                    self.out_tokens_gpu[:num_seqs], non_blocking=True
                )
                self.out_logprobs_cpu[parity, :num_seqs].copy_(
                    self.out_logprobs_gpu[:num_seqs], non_blocking=True
                )
            sample_slots_out = [int(slot_np[i]) for i in range(num_seqs)]
            num_out = num_seqs if self.is_last_pp else 0
        elif use_verify_graph:
            rows = v_bucket * width
            pos_i64 = pos_np.astype(np.int64)
            if is_ragged_verify:
                seq_slots_np = slot_np[q_start_np[:num_seqs]]
                p_min = pos_i64[q_start_np[:num_seqs]]
                p_max = pos_i64[q_start_np[1:] - 1]
            else:
                seq_slots_np = slot_np[0:total_tokens:width]
                p_min = pos_i64[0:total_tokens:width]
                p_max = pos_i64[width - 1 : total_tokens : width]
            c_lo = np.maximum(0, (p_min - 127) // 128)
            c_hi = np.minimum(pt_np.shape[1] - 1, p_max // 128)
            swa_epoch = self._swa_pool.epoch if self._swa_pool is not None else -1
            bt_key = (
                v_bucket,
                swa_epoch,
                tuple(int(s) for s in seq_slots_np),
                tuple(int(c) for c in c_lo),
                tuple(int(c) for c in c_hi),
                tuple(int(pt_np[s, c]) for s, c in zip(seq_slots_np, c_hi, strict=True)),
                tuple(int(pt_np[s, 0]) for s in seq_slots_np),
            )
            bt_hit = self._matches_gpu_bt_cache(bt_key, pt_np, seq_slots_np, c_hi)
            if not bt_hit:
                active_pt = pt_np[seq_slots_np, :] + 1
                mla_bt_np = self.mla_bt_cpu[:v_bucket].numpy()
                mla_bt_np[:num_seqs] = active_pt
                if v_bucket > num_seqs:
                    mla_bt_np[num_seqs:v_bucket] = 0

                if self._swa_pool is not None:
                    swa_pt, _ = self._swa_pool.map_active_windows(
                        active_pt,
                        p_min,
                        p_max,
                        p_min,
                        active_slots=seq_slots_np,
                    )
                else:
                    swa_pt = active_pt

                swa_bt_np = self.swa_bt_cpu[:v_bucket].numpy()
                swa_bt_np[:num_seqs, 0::2] = swa_pt * 2
                swa_bt_np[:num_seqs, 1::2] = swa_pt * 2 + 1
                if v_bucket > num_seqs:
                    swa_bt_np[num_seqs:v_bucket] = 0

                comp_bt_np = self.comp_bt_cpu[:v_bucket, 0].numpy()
                comp_bt_np[:num_seqs] = seq_slots_np + 1
                if v_bucket > num_seqs:
                    comp_bt_np[num_seqs:v_bucket] = self.dummy_comp_block

            lb_np = self.lookback_cpu[:v_bucket].numpy()
            lb_np[:num_seqs, 0] = prev_win_np[:, 2]
            lb_np[:num_seqs, 1] = prev_win_np[:, 1]
            lb_np[:num_seqs, 2] = prev_win_np[:, 0]
            if v_bucket > num_seqs:
                lb_np[num_seqs:v_bucket] = -1

            tok_cpu_np = self.token_ids_cpu[:rows].numpy()
            pos_cpu_np = self.positions_cpu[:rows].numpy()
            sl_cpu_np = self.seq_lens_cpu[:v_bucket].numpy()
            sm_cpu_np = self.slot_mapping_cpu[:rows].numpy()
            vb_tok = (pt_np[slot_np, pos_i64 // 128] + 1).astype(np.int64)
            if is_ragged_verify:
                sm_packed = vb_tok * 128 + (pos_i64 % 128)
                swa_sm_np = (
                    self.swa_slot_mapping_cpu[:rows].numpy()
                    if self._swa_pool is not None
                    else None
                )
                swa_sm_packed = (
                    self._swa_pool.lookup(vb_tok).astype(np.int64) * 128
                    + (pos_i64 % 128)
                    if self._swa_pool is not None
                    else None
                )
                self._pack_ragged_verify_buffers(
                    num_seqs=num_seqs,
                    v_bucket=v_bucket,
                    width=width,
                    q_start_np=q_start_np,
                    q_lens=q_lens,
                    tok_np=tok_np,
                    pos_i64=pos_i64,
                    seq_lens_np=seq_lens_np,
                    sm_packed=sm_packed,
                    swa_sm_packed=swa_sm_packed,
                    tok_cpu_np=tok_cpu_np,
                    pos_cpu_np=pos_cpu_np,
                    sl_cpu_np=sl_cpu_np,
                    sm_cpu_np=sm_cpu_np,
                    swa_sm_np=swa_sm_np,
                    real_rows_np=self._verify_real_rows_cpu[:total_tokens].numpy(),
                )
            else:
                tok_cpu_np[:total_tokens] = tok_np
                if rows > total_tokens:
                    tok_cpu_np[total_tokens:rows] = 0

                pos_cpu_np[:total_tokens] = pos_i64
                if rows > total_tokens:
                    pos_cpu_np[total_tokens:rows] = (
                        np.arange(rows - total_tokens, dtype=np.int64) % width
                    )

                sl_cpu_np[:num_seqs] = seq_lens_np
                if v_bucket > num_seqs:
                    sl_cpu_np[num_seqs:v_bucket] = width

                sm_cpu_np[:total_tokens] = vb_tok * 128 + (pos_i64 % 128)
                if rows > total_tokens:
                    sm_cpu_np[total_tokens:rows] = -1
                if self._swa_pool is not None:
                    swa_sm_np = self.swa_slot_mapping_cpu[:rows].numpy()
                    swa_vb_tok = self._swa_pool.lookup(vb_tok).astype(np.int64)
                    swa_sm_np[:total_tokens] = swa_vb_tok * 128 + (pos_i64 % 128)
                    if rows > total_tokens:
                        swa_sm_np[total_tokens:rows] = -1

            if not bt_hit:
                self.mla_bt_gpu[:v_bucket].copy_(
                    self.mla_bt_cpu[:v_bucket], non_blocking=True
                )
                self.swa_bt_gpu[:v_bucket].copy_(
                    self.swa_bt_cpu[:v_bucket], non_blocking=True
                )
                self.comp_bt_gpu[:v_bucket].copy_(
                    self.comp_bt_cpu[:v_bucket], non_blocking=True
                )
                post_epoch = self._swa_pool.epoch if self._swa_pool is not None else -1
                self._store_gpu_bt_cache(
                    (
                        v_bucket,
                        post_epoch,
                        bt_key[2],
                        bt_key[3],
                        bt_key[4],
                        bt_key[5],
                        bt_key[6],
                    ),
                    pt_np,
                    seq_slots_np,
                )
            self._last_verify_bt_sig = (v_bucket, tuple(int(s) for s in seq_slots_np))
            self.lookback_gpu[:v_bucket].copy_(
                self.lookback_cpu[:v_bucket], non_blocking=True
            )
            self.token_ids_gpu[:rows].copy_(
                self.token_ids_cpu[:rows], non_blocking=True
            )
            self.positions_gpu[:rows].copy_(
                self.positions_cpu[:rows], non_blocking=True
            )
            self.seq_lens_gpu[:v_bucket].copy_(
                self.seq_lens_cpu[:v_bucket], non_blocking=True
            )
            self.slot_mapping_gpu[:rows].copy_(
                self.slot_mapping_cpu[:rows], non_blocking=True
            )
            if self._swa_pool is not None:
                self.swa_slot_mapping_gpu[:rows].copy_(
                    self.swa_slot_mapping_cpu[:rows], non_blocking=True
                )

            if is_ragged_verify:
                self._verify_real_rows_gpu[:total_tokens].copy_(
                    self._verify_real_rows_cpu[:total_tokens], non_blocking=True
                )
            self._h2d_done_event.record()
            if self.pp_size > 1 and not self.is_first_pp:
                assert (
                    self._pp_inter_hidden is not None
                    and self._pp_inter_pre_mix is not None
                )
                self._pp_recv_intermediate(
                    self._pp_inter_hidden[:rows], self._pp_inter_pre_mix[:rows]
                )
            self.verify_graphs[v_bucket].replay()
            if self.pp_size > 1 and self.is_first_pp:
                assert (
                    self._pp_inter_hidden is not None
                    and self._pp_inter_pre_mix is not None
                )
                for req in self._pp_send_intermediate(
                    self._pp_inter_hidden[:rows], self._pp_inter_pre_mix[:rows]
                ):
                    req.wait()
                sample_slots_out = []
                num_out = 0
            else:
                raw_aux = self.verify_graph_aux[v_bucket]
                if is_ragged_verify:
                    real_rows_gpu = self._verify_real_rows_gpu[:total_tokens]
                    self._last_verify_graph_bucket = None
                    aux_hidden_states = (
                        [a.index_select(0, real_rows_gpu) for a in raw_aux]
                        if raw_aux is not None
                        else None
                    )
                    self.out_tokens_cpu[parity, :num_sample].copy_(
                        self.out_tokens_gpu.index_select(0, real_rows_gpu),
                        non_blocking=True,
                    )
                    self.out_logprobs_cpu[parity, :num_sample].copy_(
                        self.out_logprobs_gpu.index_select(0, real_rows_gpu),
                        non_blocking=True,
                    )
                else:
                    self._last_verify_graph_bucket = v_bucket
                    aux_hidden_states = (
                        [a[:total_tokens] for a in raw_aux]
                        if raw_aux is not None
                        else None
                    )
                    self.out_tokens_cpu[parity, :num_sample].copy_(
                        self.out_tokens_gpu[:num_sample], non_blocking=True
                    )
                    self.out_logprobs_cpu[parity, :num_sample].copy_(
                        self.out_logprobs_gpu[:num_sample], non_blocking=True
                    )
                sample_slots_out = []
                num_out = num_sample
        else:
            self._gpu_bt_cache_key = None
            first_pos = pos_np[q_start_np[:num_seqs]]
            # Enforce vLLM's strict 4-region mixed-batch ordering:
            # 0: decode (q_len==1, pos>0) -> 1: 1-tok prefill (q_len==1, pos==0)
            # -> 2: long_extend (q_len>1, pos>0) -> 3: prefill (q_len>1, pos==0).
            req_regions = np.where(
                q_lens <= 1,
                np.where(first_pos > 0, 0, 1),
                np.where(first_pos > 0, 2, 3),
            )
            if num_seqs > 1 and bool(np.any(req_regions[:-1] > req_regions[1:])):
                order = np.argsort(req_regions, kind="stable")
                new_tok = np.empty_like(tok_np)
                new_pos = np.empty_like(pos_np)
                new_slot = np.empty_like(slot_np)
                old_to_new = np.empty(total_tokens, dtype=np.int32)
                new_qsl = np.empty(num_seqs + 1, dtype=np.int32)
                new_qsl[0] = 0
                cur = 0
                for new_r, old_r in enumerate(order):
                    s = int(q_start_np[old_r])
                    e = int(q_start_np[old_r + 1])
                    ln = e - s
                    new_tok[cur : cur + ln] = tok_np[s:e]
                    new_pos[cur : cur + ln] = pos_np[s:e]
                    new_slot[cur : cur + ln] = slot_np[s:e]
                    old_to_new[s:e] = np.arange(cur, cur + ln, dtype=np.int32)
                    cur += ln
                    new_qsl[new_r + 1] = cur
                tok_np = new_tok
                pos_np = new_pos
                slot_np = new_slot
                q_start_np = new_qsl
                q_lens = q_lens[order]
                first_pos = first_pos[order]
                seq_lens_np = seq_lens_np[order]
                prev_win_np = prev_win_np[order]
                if num_sample > 0:
                    sample_rows_np = old_to_new[sample_rows_np]

            sample_slots_out = [
                int(slot_np[int(sample_rows_np[i])]) for i in range(num_sample)
            ]
            self.mixed_is_prefilling_cpu[:num_seqs].numpy()[:] = (q_lens > 1) | (
                first_pos == 0
            )

            seq_slots_np = slot_np[q_start_np[:num_seqs]]
            active_pt = pt_np[seq_slots_np, :] + 1

            pos_i64 = pos_np.astype(np.int64)
            if self._swa_pool is not None:
                p_min = pos_i64[q_start_np[:num_seqs]]
                p_max = pos_i64[q_start_np[1 : num_seqs + 1] - 1]
                swa_pt, _ = self._swa_pool.map_active_windows(
                    active_pt, p_min, p_max, p_min, active_slots=seq_slots_np
                )
            else:
                swa_pt = active_pt

            self.mla_bt_cpu[:num_seqs].numpy()[:] = active_pt
            swa_bt_np = self.swa_bt_cpu[:num_seqs].numpy()
            swa_bt_np[:, 0::2] = swa_pt * 2
            swa_bt_np[:, 1::2] = swa_pt * 2 + 1
            self.comp_bt_cpu[:num_seqs, 0].numpy()[:] = seq_slots_np + 1

            if not use_dev_tok:
                lb_np = self.lookback_cpu[:num_seqs].numpy()
                lb_np[:, 0] = prev_win_np[:, 2]
                lb_np[:, 1] = prev_win_np[:, 1]
                lb_np[:, 2] = prev_win_np[:, 0]
                np.copyto(self.token_ids_cpu[:total_tokens].numpy(), tok_np)

            np.copyto(self.positions_cpu[:total_tokens].numpy(), pos_i64)
            np.copyto(self.seq_lens_cpu[:num_seqs].numpy(), seq_lens_np)
            np.copyto(self.query_start_loc_cpu[: num_seqs + 1].numpy(), q_start_np)

            vb_tok = (pt_np[slot_np, pos_i64 // 128] + 1).astype(np.int64)
            self.slot_mapping_cpu[:total_tokens].numpy()[:] = vb_tok * 128 + (
                pos_i64 % 128
            )
            if self._swa_pool is not None:
                swa_vb_tok = self._swa_pool.lookup(vb_tok).astype(np.int64)
                self.swa_slot_mapping_cpu[:total_tokens].numpy()[:] = (
                    swa_vb_tok * 128 + (pos_i64 % 128)
                )

            if num_sample > 0:
                self.sample_rows_cpu[:num_sample].numpy()[:] = sample_rows_np.astype(
                    np.int64
                )
                self.sample_rows_gpu[:num_sample].copy_(
                    self.sample_rows_cpu[:num_sample], non_blocking=True
                )

            self.mla_bt_gpu[:num_seqs].copy_(
                self.mla_bt_cpu[:num_seqs], non_blocking=True
            )
            self.swa_bt_gpu[:num_seqs].copy_(
                self.swa_bt_cpu[:num_seqs], non_blocking=True
            )
            self.comp_bt_gpu[:num_seqs].copy_(
                self.comp_bt_cpu[:num_seqs], non_blocking=True
            )
            if not use_dev_tok:
                self.lookback_gpu[:num_seqs].copy_(
                    self.lookback_cpu[:num_seqs], non_blocking=True
                )
                self.token_ids_gpu[:total_tokens].copy_(
                    self.token_ids_cpu[:total_tokens], non_blocking=True
                )
            self.positions_gpu[:total_tokens].copy_(
                self.positions_cpu[:total_tokens], non_blocking=True
            )
            self.seq_lens_gpu[:num_seqs].copy_(
                self.seq_lens_cpu[:num_seqs], non_blocking=True
            )
            self.query_start_loc_gpu[: num_seqs + 1].copy_(
                self.query_start_loc_cpu[: num_seqs + 1], non_blocking=True
            )
            self.slot_mapping_gpu[:total_tokens].copy_(
                self.slot_mapping_cpu[:total_tokens], non_blocking=True
            )
            if self._swa_pool is not None:
                self.swa_slot_mapping_gpu[:total_tokens].copy_(
                    self.swa_slot_mapping_cpu[:total_tokens], non_blocking=True
                )

            max_q_len = int(q_lens.max())
            max_s_len = int(seq_lens_np.max())
            pw_b = None if is_spec_step else self._select_piecewise_bucket(total_tokens)
            use_pp2_prefill = (
                pw_b is None
                and self.pp_size > 1
                and total_tokens >= 2048
                and not is_decode
            )
            use_tbo = (
                pw_b is None
                and not use_pp2_prefill
                and self._can_run_prefill_tbo(
                    total_tokens,
                    max_seq_len=max_s_len,
                    is_decode=is_decode,
                    is_spec_step=is_spec_step,
                )
            )
            attn_meta = (
                {}
                if (use_tbo or use_pp2_prefill)
                else self._build_step_attn_metadata(
                    num_reqs=num_seqs,
                    num_tokens=total_tokens,
                    max_query_len=max_q_len,
                    max_seq_len=max_s_len,
                    qsl_gpu=self.query_start_loc_gpu[: num_seqs + 1],
                    qsl_cpu=self.query_start_loc_cpu[: num_seqs + 1],
                    seq_lens_gpu=self.seq_lens_gpu[:num_seqs],
                    seq_lens_cpu=self.seq_lens_cpu[:num_seqs],
                    positions_gpu=self.positions_gpu[:total_tokens],
                    slot_mapping_gpu=self.slot_mapping_gpu[:total_tokens],
                    swa_slot_mapping_gpu=self.swa_slot_mapping_gpu[:total_tokens],
                    is_prefilling_cpu=self.mixed_is_prefilling_cpu[:num_seqs],
                    for_capture=False,
                )
            )
            if is_decode:
                self.slots_cpu[:num_seqs].numpy()[:] = seq_slots_np
                self.slots_gpu[:num_seqs].copy_(
                    self.slots_cpu[:num_seqs], non_blocking=True
                )
                if use_dev_tok:
                    s_idx = self.slots_gpu[:num_seqs]
                    torch.index_select(
                        self.slot_last_tok_gpu,
                        0,
                        s_idx,
                        out=self.token_ids_gpu[:num_seqs],
                    )
                    torch.index_select(
                        self.slot_lb_gpu,
                        0,
                        s_idx,
                        out=self.lookback_gpu[:num_seqs],
                    )
            self._h2d_done_event.record()
            if pw_b is not None:
                if total_tokens < pw_b:
                    self.token_ids_gpu[total_tokens:pw_b].zero_()
                    self.positions_gpu[total_tokens:pw_b].zero_()
                self._pw_live_lookback = self.lookback_gpu[:num_seqs]
                self._pw_live_num_tokens = total_tokens
            with set_forward_context(
                attn_meta,
                self.vllm_config,
                num_tokens=pw_b if pw_b is not None else total_tokens,
                cudagraph_runtime_mode=(
                    CUDAGraphMode.PIECEWISE if pw_b is not None else CUDAGraphMode.NONE
                ),
            ):
                if pw_b is not None:
                    self.piecewise_graphs[pw_b].replay()
                    self._pw_live_lookback, self._pw_live_num_tokens = None, 0
                    hidden = self.piecewise_outputs[pw_b][:total_tokens]
                    pw_aux = self.piecewise_aux_outputs.get(pw_b)
                    aux_hidden_states = (
                        [t[:total_tokens] for t in pw_aux]
                        if pw_aux is not None
                        else None
                    )
                elif use_pp2_prefill:
                    hidden, aux_hidden_states = self._run_prefill_pp2(
                        total_tokens=total_tokens,
                        num_seqs=num_seqs,
                        q_start_np=q_start_np,
                        pos_i64=pos_i64,
                        need_hidden=(num_sample > 0),
                        capture_aux=is_spec_step,
                    )
                elif use_tbo:
                    hidden, aux_hidden_states = self._run_prefill_tbo(
                        total_tokens=total_tokens,
                        num_seqs=num_seqs,
                        q_start_np=q_start_np,
                        pos_i64=pos_i64,
                        full_attn_meta=attn_meta,
                        need_hidden=(num_sample > 0),
                        capture_aux=is_spec_step,
                    )
                else:
                    lang_mod = getattr(self.model, "language_model", self.model)
                    dsv4_core = getattr(lang_mod, "model", lang_mod)
                    saved_aux = (
                        getattr(dsv4_core, "aux_hidden_state_layers", None)
                        if (self.is_last_pp and not is_spec_step)
                        else None
                    )
                    if saved_aux:
                        dsv4_core.aux_hidden_state_layers = ()
                    try:
                        if self.pp_size > 1 and not self.is_first_pp:
                            h_recv = torch.empty(
                                (total_tokens, 4, self.hidden_size),
                                dtype=torch.bfloat16,
                                device=self.device,
                            )
                            p_recv = torch.empty(
                                (total_tokens, 4),
                                dtype=torch.float32,
                                device=self.device,
                            )
                            pp_inter = self._pp_recv_intermediate(h_recv, p_recv)
                            model_output = self.model(
                                input_ids=self.token_ids_gpu[:total_tokens],
                                positions=self.positions_gpu[:total_tokens],
                                intermediate_tensors=pp_inter,
                                lookback_token_ids=self.lookback_gpu[:num_seqs],
                            )
                        else:
                            model_output = self.model(
                                input_ids=self.token_ids_gpu[:total_tokens],
                                positions=self.positions_gpu[:total_tokens],
                                lookback_token_ids=self.lookback_gpu[:num_seqs],
                            )
                            if self.pp_size > 1 and self.is_first_pp:
                                for req in self._pp_send_intermediate(
                                    model_output["hidden_states"],
                                    model_output["pre_mix"],
                                ):
                                    req.wait()
                    finally:
                        if saved_aux:
                            dsv4_core.aux_hidden_state_layers = saved_aux
                    if self.is_last_pp:
                        if isinstance(model_output, tuple):
                            hidden, aux_hidden_states = model_output
                        else:
                            hidden = model_output
                    else:
                        hidden = None
                if self.is_last_pp and num_sample > 0:
                    local_logits = self._sample_rows_into_argmax(
                        hidden,
                        self.sample_rows_gpu[:num_sample],
                        num_sample,
                        all_ranks=(is_spec_step or is_decode),
                        return_local_logits=is_sampling,
                    )
                    if is_sampling and local_logits is not None:
                        self._apply_stochastic_sampling(
                            plan,
                            local_logits,
                            num_sample,
                            parity,
                            use_penalty=is_penalty,
                            sample_slots_np=slot_np[sample_rows_np],
                        )
            if self.pp_size == 1 and is_decode:
                self._commit_decode_slot_state(num_seqs)

            if self.is_last_pp and (self.is_result_rank or is_spec_step) and num_sample > 0:
                self.out_tokens_cpu[parity, :num_sample].copy_(
                    self.out_tokens_gpu[:num_sample], non_blocking=True
                )
                self.out_logprobs_cpu[parity, :num_sample].copy_(
                    self.out_logprobs_gpu[:num_sample], non_blocking=True
                )
            num_out = num_sample if self.is_last_pp else 0

        self.end_events[parity].record()
        spec_ctx = (
            _SpecVerifyContext(
                num_sample=num_sample,
                sample_rows_np=sample_rows_np.copy(),
                slot_np=slot_np.copy(),
                tok_np=tok_np.copy(),
                pos_np=pos_np.copy(),
                q_start_np=q_start_np.copy(),
                pt_np=pt_np.copy(),
                aux_hidden_states=aux_hidden_states,
            )
            if (is_spec_step and self.is_last_pp)
            else None
        )
        return _LaunchedStep(
            parity=parity,
            num_out=num_out,
            sample_slots_out=sample_slots_out,
            spec_ctx=spec_ctx,
        )

    @torch.inference_mode()
    def finish(self, step: _LaunchedStep | None) -> tuple[list[SeqResult], int]:
        if step is None:
            return [], 1
        p = step.parity
        self.end_events[p].synchronize()
        outputs: list[SeqResult] = []
        if step.spec_ctx is not None:
            ctx = step.spec_ctx
            outputs = self._speculative_outputs(
                parity=p,
                num_sample=ctx.num_sample,
                sample_rows_np=ctx.sample_rows_np,
                slot_np=ctx.slot_np,
                tok_np=ctx.tok_np,
                pos_np=ctx.pos_np,
                q_start_np=ctx.q_start_np,
                pt_np=ctx.pt_np,
                aux_hidden_states=ctx.aux_hidden_states,
            )
            self.end_events[p].record()
            self.end_events[p].synchronize()
        elif self.is_result_rank and step.num_out > 0:
            toks_np = self._out_tokens_np[p]
            lps_np = self._out_logprobs_np[p]
            for i in range(step.num_out):
                outputs.append(
                    SeqResult(
                        req_slot=step.sample_slots_out[i],
                        accepted_token=int(toks_np[i, 0]),
                        accepted_logprob_bits=int(lps_np[i, 0]) & 0xFFFF_FFFF,
                        runner_up_token=int(toks_np[i, 1]),
                        runner_up_logprob_bits=int(lps_np[i, 1]) & 0xFFFF_FFFF,
                    )
                )
        gpu_exec_us = max(
            1, int(self.start_events[p].elapsed_time(self.end_events[p]) * 1000.0)
        )
        return outputs, gpu_exec_us

    @torch.inference_mode()
    def execute_kv_transfer(self, plan: StepPlanView, header: StepHeader) -> int:
        if self.dram_kv_tier is None:
            raise RuntimeError(
                "Received StepOpcode::KvTransfer but HostDramKvTier is disabled "
                "(set --dram-kv-blocks > 0)"
            )
        count = int(header.num_seqs)
        if count <= 0:
            return 1
        hbm_ids = plan.token_ids(count)
        dram_ids = plan.positions(count)
        directions = plan.req_slot_ids(count)
        self.start_events[0].record()
        self.dram_kv_tier.execute_transfers(hbm_ids, dram_ids, directions, count)
        self.end_events[0].record()
        self.end_events[0].synchronize()
        return max(
            1, int(self.start_events[0].elapsed_time(self.end_events[0]) * 1000.0)
        )

    def signal_abort(self) -> None:
        if self.argmax_channel is not None:
            self.argmax_channel.signal_abort()
        if self.shm_ag is not None:
            self.shm_ag.signal_abort()
        if self.shm_ar is not None:
            self.shm_ar.signal_abort()

    def _abort_close(self) -> None:
        if self._closed:
            return
        self._closed = True
        self.signal_abort()
        self.decode_graphs.clear()
        self.piecewise_graphs.clear()
        self.piecewise_outputs.clear()
        self.piecewise_aux_outputs.clear()
        self.verify_graphs.clear()
        self.verify_graph_aux.clear()
        if self.dspark_drafter is not None:
            self.dspark_drafter.close()
        if self.dram_kv_tier is not None:
            self.dram_kv_tier.close()
        close_b12x_dsv41_model(getattr(self, "dspark_model", None))
        close_b12x_dsv41_model(getattr(self, "model", None))
        self.dspark_model = None
        self.model = None
        builder_groups = getattr(self, "_builder_groups", None)
        if builder_groups is not None:
            builder_groups.clear()
        tbo_u1_groups = getattr(self, "_tbo_builder_groups_u1", None)
        if tbo_u1_groups is not None:
            tbo_u1_groups.clear()
        if self.argmax_channel is not None:
            self.argmax_channel.close(abort=True)
        if self.shm_ag is not None:
            self.shm_ag.close(abort=True)
        if self.shm_ar is not None:
            self.shm_ar.close(abort=True)
        if self._vllm_config_ctx is not None:
            self._vllm_config_ctx.__exit__(None, None, None)
            self._vllm_config_ctx = None

    def close(self, *, abort: bool = False) -> None:
        if abort:
            self._abort_close()
            return
        if self._closed:
            return
        self._closed = True
        from vllm.distributed import (
            destroy_distributed_environment,
            destroy_model_parallel,
        )

        torch.cuda.synchronize(self.device)
        self.decode_graphs.clear()
        self.piecewise_graphs.clear()
        self.piecewise_outputs.clear()
        self.piecewise_aux_outputs.clear()
        self.verify_graphs.clear()
        self.verify_graph_aux.clear()
        if self.dspark_drafter is not None:
            self.dspark_drafter.close()
        if self.dram_kv_tier is not None:
            self.dram_kv_tier.close()
        close_b12x_dsv41_model(getattr(self, "dspark_model", None))
        close_b12x_dsv41_model(getattr(self, "model", None))
        self.dspark_model = None
        self.model = None
        builder_groups = getattr(self, "_builder_groups", None)
        if builder_groups is not None:
            builder_groups.clear()
        tbo_u1_groups = getattr(self, "_tbo_builder_groups_u1", None)
        if tbo_u1_groups is not None:
            tbo_u1_groups.clear()
        gc.collect()
        torch.cuda.empty_cache()
        if dist.is_initialized() and hasattr(self, "exchange_group"):
            dist.barrier(group=self.exchange_group)
        if self.argmax_channel is not None:
            self.argmax_channel.close()
        if self.shm_ag is not None:
            self.shm_ag.close()
        if self.shm_ar_tbo is not None:
            self.shm_ar_tbo.close()
        if self.shm_ar is not None:
            self.shm_ar.close()
        if dist.is_initialized() and hasattr(self, "exchange_group"):
            dist.barrier(group=self.exchange_group)
            destroy_model_parallel()
            destroy_distributed_environment()
        if self._vllm_config_ctx is not None:
            self._vllm_config_ctx.__exit__(None, None, None)
            self._vllm_config_ctx = None
