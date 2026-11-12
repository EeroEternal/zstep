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

"""Explicit Mode B TP=8 SM120 runner for DeepSeek-V4.1-Flash (`deepseek_v41`).

Integrates zstep's zero-copy POSIX SHM `StepPlan` / `StepResult` protocol, Rust-owned
128-token KV page tables, exact 3-token Engram lookback windows (`engram_prev_window`),
NUMA-local `shm_v5` `HostMappedShmAllReduce` + `HostMappedShmAllGather`, and lock-free
`HostMappedShmArgmax` with `vllm 0.31.0`'s native SM120 DeepGEMM UE8M0 + FlashInfer
sparse MLA/SWA kernels and full-40-layer `CUDAGraph` decode buckets.
"""

from __future__ import annotations

from dataclasses import dataclass
import dataclasses
import math
import os
from pathlib import Path
from typing import Any

import numpy as np
import torch
import torch.distributed as dist

from zstep_worker.kernels_sm120.b12x_dsv41 import (
    B12xPrepareSummary,
    install_b12x_dsv41_quant_patches,
    prepare_b12x_dsv41_model,
)
from zstep_worker.kernels_sm120.shm_allreduce import (
    HostMappedShmAllGather,
    HostMappedShmAllReduce,
    HostMappedShmArgmax,
    get_host_mapped_allreduce_extension,
)
from zstep_worker.shm_bridge import SeqResult, ShmBridge, StepHeader, StepPlanView

DECODE_GRAPH_BUCKETS: tuple[int, ...] = (1, 2, 4, 8, 16, 32, 64, 128)
STEP_FLAG_DECODE_ONLY: int = 1


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

    def _patched_get_paged_meta(context_lens, block_kv, num_sms, *args, **kwargs):
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
        meta = orig_idx_builder_build(
            self, common_prefix_len, common_attn_metadata, fast_build=fast_build
        )
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


def _select_decode_bucket(num_seqs: int, max_bucket: int) -> int:
    for b in DECODE_GRAPH_BUCKETS:
        if b >= num_seqs and b <= max_bucket:
            return b
    return max_bucket


@dataclass
class _AttnBuilderGroup:
    kind: str  # "swa" | "mla" | "indexer" | "compressor"
    layer_names: list[str]
    builder: Any


class DeepSeekV41StepRunner:
    """Explicit Mode B GPU step runner for DeepSeek-V4.1-Flash (`TP=8`, `sm_120`)."""

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
        os.environ.setdefault("NCCL_P2P_DISABLE", "1")
        os.environ.setdefault("NCCL_P2P_LEVEL", "LOC")
        os.environ.setdefault("CUDA_DEVICE_MAX_CONNECTIONS", "1")
        torch.set_num_threads(2)

        self._closed: bool = False
        self._vllm_config_ctx: Any = None
        self.shm_ar: HostMappedShmAllReduce | None = None
        self.shm_ag: HostMappedShmAllGather | None = None
        self.argmax_channel: HostMappedShmArgmax | None = None
        self.decode_graphs: dict[int, torch.cuda.CUDAGraph] = {}

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
        self.bridge: ShmBridge = bridge
        self.geometry = bridge.superblock.geometry
        self.world_size: int = self.geometry.world_size
        if self.world_size != 8:
            raise ValueError(
                f"DeepSeekV41StepRunner requires TP=8, got world_size={self.world_size}"
            )
        if self.geometry.page_block_tokens != 128:
            raise ValueError(
                f"DeepSeekV41StepRunner requires page_block_tokens=128, got {self.geometry.page_block_tokens}"
            )

        torch.cuda.set_device(rank)
        self.device: torch.device = torch.device("cuda", rank)

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

        self.max_model_len: int = int(os.environ.get("ZSTEP_MAX_MODEL_LEN", "32768"))
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
        self.dummy_comp_block: int = self.max_seqs + 1

        engine_args = EngineArgs(
            model=str(model_dir),
            trust_remote_code=True,
            dtype="bfloat16",
            kv_cache_dtype="fp8_ds_mla",
            block_size=128,
            max_model_len=self.max_model_len,
            max_num_seqs=self.max_seqs,
            max_num_batched_tokens=max(self.max_step_tokens, 8192),
            tensor_parallel_size=self.world_size,
            distributed_executor_backend="external_launcher",
            enforce_eager=True,
            enable_prefix_caching=False,
            limit_mm_per_prompt={"image": 0},
            engram_config=EngramConfig(cpu_offload=False),
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
        initialize_model_parallel(tensor_model_parallel_size=self.world_size)
        self.exchange_group: dist.ProcessGroup = get_world_group().cpu_group
        init_workspace_manager(self.device)

        self.model = get_model(vllm_config=self.vllm_config)
        self.model.eval()
        self.b12x_summary: B12xPrepareSummary = prepare_b12x_dsv41_model(
            self.model,
            rank=rank,
            device=self.device,
            max_tokens=max(self.max_step_tokens, 8192),
            decode_buckets=DECODE_GRAPH_BUCKETS,
            exchange_group=self.exchange_group,
        )

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
        if vocab_size % self.world_size != 0:
            raise ValueError(
                f"vocab_size={vocab_size} must be divisible by world_size={self.world_size}"
            )
        self.vocab_per_rank: int = vocab_size // self.world_size

        # Load cached FlashInfer autotune configs if present in ~/.cache/flashinfer
        self._autotune_loaded_files: int = self._load_flashinfer_autotune_cache()

        # Serialize CUDA C++ extension build on rank 0 before ranks 1..7 load it
        if rank == 0:
            get_host_mapped_allreduce_extension()
        dist.barrier(group=self.exchange_group)
        if rank != 0:
            get_host_mapped_allreduce_extension()
        dist.barrier(group=self.exchange_group)

        # Initialize NUMA-local shm_v5 AllReduce, AllGather, and lock-free TP=8 Argmax
        ar_max_tokens = max(self.max_step_tokens, 2048)
        self.shm_ar: HostMappedShmAllReduce = HostMappedShmAllReduce(
            name=allreduce_name,
            rank=rank,
            world_size=self.world_size,
            capacity_elements=ar_max_tokens * 8192,
            blocks=allreduce_blocks,
            threads=allreduce_threads,
            exchange_group=self.exchange_group,
        )
        self.shm_ag: HostMappedShmAllGather = HostMappedShmAllGather(
            name=f"{allreduce_name}_ag",
            rank=rank,
            world_size=self.world_size,
            capacity_input_elements=ar_max_tokens * 4096,
            blocks=allreduce_blocks,
            threads=allreduce_threads,
            exchange_group=self.exchange_group,
        )
        self.argmax_channel: HostMappedShmArgmax = HostMappedShmArgmax(
            name=f"{allreduce_name}_argmax",
            rank=rank,
            world_size=self.world_size,
            device=self.device,
            exchange_group=self.exchange_group,
            max_rows_per_step=max(512, self.max_seqs),
        )
        self.shm_ar_calls: int = 0
        self.shm_ag_calls: int = 0
        self.nccl_eager_large_calls: int = 0
        self.nccl_fallback_calls: int = 0
        self._install_tp_shm_collectives(get_tp_group())

        # Allocate & bind 512B-aligned KV / SWA / Indexer / Compressor caches
        self._kv_raw_buffers: list[torch.Tensor] = []
        self._builder_groups: list[_AttnBuilderGroup] = []
        self._allocate_and_bind_caches_and_builders()

        # Allocate static execution, staging, and sampling workspaces
        mt = max(self.max_step_tokens, 8192)
        ms = self.max_seqs + 2

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
            (self.max_seqs, self.engram_depth),
            -1,
            dtype=torch.int32,
            device=self.device,
        )
        self.sample_rows_gpu: torch.Tensor = torch.zeros(
            ms, dtype=torch.int64, device=self.device
        )

        # Decode-only static arange query_start_loc `[0, 1, 2, ..., ms]`
        self.decode_qsl_gpu: torch.Tensor = torch.arange(
            ms + 1, dtype=torch.int32, device=self.device
        )
        self.decode_qsl_cpu: torch.Tensor = torch.arange(ms + 1, dtype=torch.int32)
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
            (self.max_seqs, self.engram_depth), -1, dtype=torch.int32, pin_memory=True
        )
        self.sample_rows_cpu: torch.Tensor = torch.zeros(
            ms, dtype=torch.int64, pin_memory=True
        )

        # Sampling buffers (deterministic global Top-2 `(logit, token_id)`)
        self.local_max_vals_buf: torch.Tensor = torch.empty(
            (ms, 2), dtype=torch.float32, device=self.device
        )
        self.local_max_ids_buf: torch.Tensor = torch.empty(
            (ms, 2), dtype=torch.int32, device=self.device
        )
        self.out_tokens_gpu: torch.Tensor = torch.zeros(
            (ms, 2), dtype=torch.int32, device=self.device
        )
        self.out_logprobs_gpu: torch.Tensor = torch.zeros(
            (ms, 2), dtype=torch.int32, device=self.device
        )
        self.out_tokens_cpu: torch.Tensor = torch.zeros(
            (ms, 2), dtype=torch.int32, pin_memory=True
        )
        self.out_logprobs_cpu: torch.Tensor = torch.zeros(
            (ms, 2), dtype=torch.int32, pin_memory=True
        )

        self.start_event: torch.cuda.Event = torch.cuda.Event(enable_timing=True)
        self.end_event: torch.cuda.Event = torch.cuda.Event(enable_timing=True)
        self.max_graph_bucket: int = min(self.geometry.max_seqs, 128)

        self._warmup_and_capture_decode_graphs()

    def _load_flashinfer_autotune_cache(self) -> int:
        loaded_files = 0
        try:
            from flashinfer.autotuner import AutoTuner

            tuner = AutoTuner.get()
            load_fn = getattr(tuner, "load_configs", None)
            if not callable(load_fn):
                return 0
            cache_root = Path.home() / ".cache"
            if cache_root.is_dir():
                candidates = sorted(
                    cache_root.rglob("*autotune*.json"),
                    key=lambda x: (x.stat().st_size, x.stat().st_mtime),
                )
                for p in candidates:
                    try:
                        if load_fn(str(p)):
                            loaded_files += 1
                    except Exception:
                        pass
        except Exception:
            pass
        return loaded_files

    def _install_tp_shm_collectives(self, tp_group: Any) -> None:
        dev_comm = tp_group.device_communicator
        orig_ar = dev_comm.all_reduce
        orig_ag = dev_comm.all_gather
        shm_ar = self.shm_ar
        shm_ag = self.shm_ag
        runner = self
        # Route M <= 256 (<= 4 MiB AR / <= 2 MiB AG) and 100% of CUDAGraph captures
        # through `shm_v5`; route larger eager prefill/mixed payloads (M > 256) to
        # multi-SM `NCCL_LOC` (`orig_ar` / `orig_ag`) to saturate dual-socket DRAM/UPI.
        shm_max_ar_numel = int(
            os.environ.get("ZSTEP_SHM_MAX_AR_NUMEL", str(256 * 8192))
        )
        shm_max_ag_numel = int(
            os.environ.get("ZSTEP_SHM_MAX_AG_NUMEL", str(256 * 4096))
        )

        tp_group.use_custom_op_call = False

        def _fast_all_reduce(input_: torch.Tensor) -> torch.Tensor:
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
                    out = torch.empty_like(contig)
                    shm_ar.all_reduce(contig, output=out)
                    runner.shm_ar_calls += 1
                    return out
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

            if isinstance(layer, DeepseekV4SWACache):
                kind = "swa"
                # 64-token pages, 584B row (64 * 584 = 37376 == 73 * 512)
                raw = torch.zeros(
                    (self.total_128_blocks * 2, 1, 64, 584),
                    dtype=torch.uint8,
                    device=self.device,
                )
                self._kv_raw_buffers.append(raw)
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
            if tag == "swa":
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

    def _zero_all_caches(self) -> None:
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
        is_prefilling_cpu: torch.Tensor,
        for_capture: bool,
    ) -> dict[str, Any]:
        from vllm.v1.attention.backend import CommonAttentionMetadata

        mla_bt = self.mla_bt_gpu[:num_reqs]
        swa_bt = self.swa_bt_gpu[:num_reqs]
        comp_bt = self.comp_bt_gpu[:num_reqs]
        req_idx = self.req_idx_arange_np[:num_reqs]
        token_to_req_cache: dict[str, torch.Tensor | None] = {"val": None}

        def _make_cam(bt: torch.Tensor) -> CommonAttentionMetadata:
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
                slot_mapping=slot_mapping_gpu,
                causal=True,
                positions=positions_gpu,
                is_prefilling=is_prefilling_cpu,
                req_idx=req_idx,
                _token_to_req_indices_cache=token_to_req_cache["val"],
            )
            return cam

        cam_mla = _make_cam(mla_bt)
        cam_swa = _make_cam(swa_bt)
        cam_comp = _make_cam(comp_bt)

        attn_metadata: dict[str, Any] = {}
        for grp in self._builder_groups:
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
                attn_metadata[lname] = meta
        return attn_metadata

    def _sample_rows_into_argmax(
        self, hidden_states: torch.Tensor, sample_rows: torch.Tensor | None, count: int
    ) -> None:
        if count <= 0:
            return
        sampled_h = (
            hidden_states[:count]
            if sample_rows is None
            else hidden_states.index_select(0, sample_rows)
        )
        local_logits = self.model.compute_logits_local(sampled_h)
        self.argmax_channel.gather_top2_from_logits(
            local_logits,
            local_top2_vals_buf=self.local_max_vals_buf[:count],
            local_top2_ids_buf=self.local_max_ids_buf[:count],
            out_tokens=self.out_tokens_gpu[:count],
            out_logprob_bits=self.out_logprobs_gpu[:count],
            vocab_per_rank=self.vocab_per_rank,
        )

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
        except Exception:
            pass
        try:
            cutedsl_warmup()
        except Exception:
            pass

        # Warm up eager prefill/chunked-extend shapes across all Triton specialization
        # classes (`==1`, `%16 != 0`, `%16 == 0`), multi-seq batches, short contexts,
        # and long contexts (`seq_len > topk_tokens * compress_ratio`) so zero Triton
        # JIT compilations ever occur during `RuntimeLoop.run()`, while activating
        # FlashInfer SM120 CutlassMxfp8GemmRunner + Sparse MLA autotuned tactics.
        dummy_vb = self.num_kv_pages + 1  # scratch overflow block
        self.mla_bt_gpu.fill_(dummy_vb)
        self.swa_bt_gpu[:, 0::2] = dummy_vb * 2
        self.swa_bt_gpu[:, 1::2] = dummy_vb * 2 + 1
        self.comp_bt_gpu.fill_(self.dummy_comp_block)
        self.lookback_gpu.fill_(-1)

        warmup_cases: list[tuple[list[int], int]] = [
            ([self.max_step_tokens], 0),
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
            max(self.max_step_tokens, 8192)
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
                    is_prefilling_cpu=self.mixed_is_prefilling_cpu[:n_reqs],
                    for_capture=False,
                )
                with ExitStack() as stack:
                    if case_idx == 0:
                        stack.enter_context(
                            fi_utils.autotune(
                                tune_mode=(self._autotune_loaded_files == 0)
                            )
                        )
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
                    hidden = self.model(
                        input_ids=self.token_ids_gpu[:n_toks],
                        positions=self.positions_gpu[:n_toks],
                        lookback_token_ids=self.lookback_gpu,
                    )
                    self._sample_rows_into_argmax(hidden, self.sample_rows_gpu[:1], 1)
                torch.cuda.synchronize(self.device)
                dist.barrier(group=self.exchange_group)

            # Also warm up + autotune eager decode (`seq_len = 64..2563`) so all
            # FlashInfer SM120 sparse MLA decode tactics & persistent_topk kernels
            # are active before CUDAGraph capture.
            for warm_idx, (warm_b, warm_sl) in enumerate(
                (
                    (16, 2560),
                    (1, 64),
                    (1, 2561),
                    (3, 2563),
                    (32, 1536),
                    (128, 1280),
                )
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
                self.lookback_gpu[:warm_b].fill_(1)
                attn_meta = self._build_step_attn_metadata(
                    num_reqs=warm_b,
                    num_tokens=warm_b,
                    max_query_len=1,
                    max_seq_len=self.max_model_len,
                    qsl_gpu=self.decode_qsl_gpu[: warm_b + 1],
                    qsl_cpu=self.decode_qsl_cpu[: warm_b + 1],
                    seq_lens_gpu=self.seq_lens_gpu[:warm_b],
                    seq_lens_cpu=sl_cpu,
                    positions_gpu=self.positions_gpu[:warm_b],
                    slot_mapping_gpu=self.slot_mapping_gpu[:warm_b],
                    is_prefilling_cpu=self.decode_is_prefilling_cpu[:warm_b],
                    for_capture=True,
                )
                with ExitStack() as stack:
                    if warm_idx == 0:
                        stack.enter_context(
                            fi_utils.autotune(
                                tune_mode=(self._autotune_loaded_files == 0)
                            )
                        )
                    stack.enter_context(
                        set_forward_context(
                            attn_meta,
                            self.vllm_config,
                            num_tokens=warm_b,
                            cudagraph_runtime_mode=CUDAGraphMode.NONE,
                        )
                    )
                    hidden = self.model(
                        input_ids=self.token_ids_gpu[:warm_b],
                        positions=self.positions_gpu[:warm_b],
                        lookback_token_ids=self.lookback_gpu,
                    )
                    self._sample_rows_into_argmax(hidden, None, warm_b)
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
            except Exception:
                pass
        dist.barrier(group=self.exchange_group)

        # Zero all caches before capturing decode CUDAGraphs so Block 0 is clean.
        self._zero_all_caches()
        dist.barrier(group=self.exchange_group)

        buckets = [b for b in DECODE_GRAPH_BUCKETS if b <= self.max_graph_bucket]
        if not buckets or buckets[-1] != self.max_graph_bucket:
            buckets.append(self.max_graph_bucket)

        self.mla_bt_gpu.zero_()
        self.swa_bt_gpu.zero_()
        self.comp_bt_gpu.fill_(self.dummy_comp_block)
        self.lookback_gpu.fill_(-1)
        self.slot_mapping_gpu.fill_(-1)
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
                for _ in range(2):
                    attn_meta = self._build_step_attn_metadata(
                        num_reqs=b,
                        num_tokens=b,
                        max_query_len=1,
                        max_seq_len=self.max_model_len,
                        qsl_gpu=self.decode_qsl_gpu[: b + 1],
                        qsl_cpu=self.decode_qsl_cpu[: b + 1],
                        seq_lens_gpu=self.seq_lens_gpu[:b],
                        seq_lens_cpu=sl_cpu,
                        positions_gpu=self.positions_gpu[:b],
                        slot_mapping_gpu=self.slot_mapping_gpu[:b],
                        is_prefilling_cpu=self.decode_is_prefilling_cpu[:b],
                        for_capture=True,
                    )
                    with set_forward_context(
                        attn_meta,
                        self.vllm_config,
                        num_tokens=b,
                        cudagraph_runtime_mode=CUDAGraphMode.NONE,
                    ):
                        hidden = self.model(
                            input_ids=self.token_ids_gpu[:b],
                            positions=self.positions_gpu[:b],
                            lookback_token_ids=self.lookback_gpu,
                        )
                        self._sample_rows_into_argmax(hidden, None, b)
            capture_stream.synchronize()
            dist.barrier(group=self.exchange_group)

            with torch.cuda.stream(capture_stream):
                attn_meta = self._build_step_attn_metadata(
                    num_reqs=b,
                    num_tokens=b,
                    max_query_len=1,
                    max_seq_len=self.max_model_len,
                    qsl_gpu=self.decode_qsl_gpu[: b + 1],
                    qsl_cpu=self.decode_qsl_cpu[: b + 1],
                    seq_lens_gpu=self.seq_lens_gpu[:b],
                    seq_lens_cpu=sl_cpu,
                    positions_gpu=self.positions_gpu[:b],
                    slot_mapping_gpu=self.slot_mapping_gpu[:b],
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
                    hidden = self.model(
                        input_ids=self.token_ids_gpu[:b],
                        positions=self.positions_gpu[:b],
                        lookback_token_ids=self.lookback_gpu,
                    )
                    self._sample_rows_into_argmax(hidden, None, b)
            capture_stream.synchronize()
            dist.barrier(group=self.exchange_group)
            self.decode_graphs[b] = g

        lock_workspace()
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
            print(
                f"[DeepSeekV41StepRunner] ready: max_model_len={self.max_model_len} "
                f"kv_pages={self.num_kv_pages} buckets={buckets} {b12x_info} "
                f"shm_ar_warm={self.shm_ar_calls} shm_ag_warm={self.shm_ag_calls} "
                f"nccl_large_warm={self.nccl_eager_large_calls} "
                f"nccl_fallback_warm={self.nccl_fallback_calls}",
                flush=True,
            )

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

    @torch.inference_mode()
    def execute(
        self, plan: StepPlanView, header: StepHeader
    ) -> tuple[list[SeqResult], int]:
        from vllm.config import CUDAGraphMode
        from vllm.forward_context import set_forward_context

        num_seqs = header.num_seqs
        total_tokens = header.num_encoder_tokens
        num_sample = header.num_sample_rows
        if total_tokens == 0 or num_seqs == 0:
            return [], 1

        tok_np = plan.token_ids(total_tokens)
        pos_np = plan.positions(total_tokens)
        slot_np = plan.req_slot_ids(total_tokens)
        seq_lens_np = plan.seq_lens(num_seqs)
        q_start_np = plan.query_start_loc(num_seqs + 1)
        prev_win_np = plan.engram_prev_window(num_seqs)
        sample_rows_np = plan.sample_row_indices(num_sample)

        max_slot_end = min(self.max_seqs, int(slot_np.max()) + 1)
        pt_np = plan.global_kv_page_table(max_slot_end)[:, : self.bt_width]

        is_decode = (header.flags & STEP_FLAG_DECODE_ONLY) != 0

        self.start_event.record()
        if is_decode:
            bucket = _select_decode_bucket(num_seqs, self.max_graph_bucket)
            active_pt = pt_np[slot_np, :] + 1

            mla_bt_np = self.mla_bt_cpu[:bucket].numpy()
            mla_bt_np[:num_seqs] = active_pt
            if bucket > num_seqs:
                mla_bt_np[num_seqs:bucket] = 0

            swa_bt_np = self.swa_bt_cpu[:bucket].numpy()
            swa_bt_np[:num_seqs, 0::2] = active_pt * 2
            swa_bt_np[:num_seqs, 1::2] = active_pt * 2 + 1
            if bucket > num_seqs:
                swa_bt_np[num_seqs:bucket] = 0

            comp_bt_np = self.comp_bt_cpu[:bucket, 0].numpy()
            comp_bt_np[:num_seqs] = slot_np + 1
            if bucket > num_seqs:
                comp_bt_np[num_seqs:bucket] = self.dummy_comp_block

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
            pos_i64 = pos_np.astype(np.int64)
            pos_cpu_np[:num_seqs] = pos_i64
            if bucket > num_seqs:
                pos_cpu_np[num_seqs:bucket] = 0

            sl_cpu_np = self.seq_lens_cpu[:bucket].numpy()
            sl_cpu_np[:num_seqs] = seq_lens_np
            if bucket > num_seqs:
                sl_cpu_np[num_seqs:bucket] = 0

            sm_cpu_np = self.slot_mapping_cpu[:bucket].numpy()
            vb = active_pt[np.arange(num_seqs), pos_i64 // 128].astype(np.int64)
            sm_cpu_np[:num_seqs] = vb * 128 + (pos_i64 % 128)
            if bucket > num_seqs:
                sm_cpu_np[num_seqs:bucket] = -1

            self.mla_bt_gpu[:bucket].copy_(self.mla_bt_cpu[:bucket], non_blocking=True)
            self.swa_bt_gpu[:bucket].copy_(self.swa_bt_cpu[:bucket], non_blocking=True)
            self.comp_bt_gpu[:bucket].copy_(
                self.comp_bt_cpu[:bucket], non_blocking=True
            )
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

            self._build_step_attn_metadata(
                num_reqs=bucket,
                num_tokens=bucket,
                max_query_len=1,
                max_seq_len=self.max_model_len,
                qsl_gpu=self.decode_qsl_gpu[: bucket + 1],
                qsl_cpu=self.decode_qsl_cpu[: bucket + 1],
                seq_lens_gpu=self.seq_lens_gpu[:bucket],
                seq_lens_cpu=self.seq_lens_cpu[:bucket],
                positions_gpu=self.positions_gpu[:bucket],
                slot_mapping_gpu=self.slot_mapping_gpu[:bucket],
                is_prefilling_cpu=self.decode_is_prefilling_cpu[:bucket],
                for_capture=False,
            )
            self.decode_graphs[bucket].replay()

            if self.rank == 0:
                self.out_tokens_cpu[:num_seqs].copy_(
                    self.out_tokens_gpu[:num_seqs], non_blocking=True
                )
                self.out_logprobs_cpu[:num_seqs].copy_(
                    self.out_logprobs_gpu[:num_seqs], non_blocking=True
                )
            sample_slots_out = [int(slot_np[i]) for i in range(num_seqs)]
        else:
            q_lens = q_start_np[1:] - q_start_np[:-1]
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

            self.mla_bt_cpu[:num_seqs].numpy()[:] = active_pt
            swa_bt_np = self.swa_bt_cpu[:num_seqs].numpy()
            swa_bt_np[:, 0::2] = active_pt * 2
            swa_bt_np[:, 1::2] = active_pt * 2 + 1
            self.comp_bt_cpu[:num_seqs, 0].numpy()[:] = seq_slots_np + 1

            lb_np = self.lookback_cpu[:num_seqs].numpy()
            lb_np[:, 0] = prev_win_np[:, 2]
            lb_np[:, 1] = prev_win_np[:, 1]
            lb_np[:, 2] = prev_win_np[:, 0]

            np.copyto(self.token_ids_cpu[:total_tokens].numpy(), tok_np)
            pos_i64 = pos_np.astype(np.int64)
            np.copyto(self.positions_cpu[:total_tokens].numpy(), pos_i64)
            np.copyto(self.seq_lens_cpu[:num_seqs].numpy(), seq_lens_np)
            np.copyto(self.query_start_loc_cpu[: num_seqs + 1].numpy(), q_start_np)

            vb_tok = (pt_np[slot_np, pos_i64 // 128] + 1).astype(np.int64)
            self.slot_mapping_cpu[:total_tokens].numpy()[:] = vb_tok * 128 + (
                pos_i64 % 128
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

            max_q_len = int(q_lens.max())
            max_s_len = int(seq_lens_np.max())
            attn_meta = self._build_step_attn_metadata(
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
                is_prefilling_cpu=self.mixed_is_prefilling_cpu[:num_seqs],
                for_capture=False,
            )
            with set_forward_context(
                attn_meta,
                self.vllm_config,
                num_tokens=total_tokens,
                cudagraph_runtime_mode=CUDAGraphMode.NONE,
            ):
                hidden = self.model(
                    input_ids=self.token_ids_gpu[:total_tokens],
                    positions=self.positions_gpu[:total_tokens],
                    lookback_token_ids=self.lookback_gpu,
                )
                if num_sample > 0:
                    self._sample_rows_into_argmax(
                        hidden, self.sample_rows_gpu[:num_sample], num_sample
                    )

            if self.rank == 0 and num_sample > 0:
                self.out_tokens_cpu[:num_sample].copy_(
                    self.out_tokens_gpu[:num_sample], non_blocking=True
                )
                self.out_logprobs_cpu[:num_sample].copy_(
                    self.out_logprobs_gpu[:num_sample], non_blocking=True
                )

        self.end_event.record()
        self.end_event.synchronize()
        gpu_exec_us = max(1, int(self.start_event.elapsed_time(self.end_event) * 1000.0))

        outputs: list[SeqResult] = []
        if self.rank == 0 and num_sample > 0:
            toks_out = self.out_tokens_cpu[:num_sample].tolist()
            lps_out = self.out_logprobs_cpu[:num_sample].tolist()
            for i in range(num_sample):
                outputs.append(
                    SeqResult(
                        req_slot=sample_slots_out[i],
                        accepted_token=int(toks_out[i][0]),
                        accepted_logprob_bits=int(lps_out[i][0]) & 0xFFFF_FFFF,
                        runner_up_token=int(toks_out[i][1]),
                        runner_up_logprob_bits=int(lps_out[i][1]) & 0xFFFF_FFFF,
                    )
                )
        return outputs, gpu_exec_us

    def _abort_close(self) -> None:
        if self._closed:
            return
        self._closed = True
        self.decode_graphs.clear()
        if self.argmax_channel is not None:
            self.argmax_channel.close()
        if self.shm_ag is not None:
            self.shm_ag.close()
        if self.shm_ar is not None:
            self.shm_ar.close()
        if self._vllm_config_ctx is not None:
            self._vllm_config_ctx.__exit__(None, None, None)
            self._vllm_config_ctx = None

    def close(self) -> None:
        if self._closed:
            return
        self._closed = True
        from vllm.distributed import (
            destroy_distributed_environment,
            destroy_model_parallel,
        )

        self.decode_graphs.clear()
        if dist.is_initialized() and hasattr(self, "exchange_group"):
            dist.barrier(group=self.exchange_group)
        if self.argmax_channel is not None:
            self.argmax_channel.close()
        if self.shm_ag is not None:
            self.shm_ag.close()
        if self.shm_ar is not None:
            self.shm_ar.close()
        if dist.is_initialized() and hasattr(self, "exchange_group"):
            dist.barrier(group=self.exchange_group)
            destroy_model_parallel()
            destroy_distributed_environment()
        if self._vllm_config_ctx is not None:
            self._vllm_config_ctx.__exit__(None, None, None)
            self._vllm_config_ctx = None
