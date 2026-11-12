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

"""Explicit TP=8 BF16 runner for Qwen/Qwen3.5-35B-A3B in zstep Mode B.

Loads per-rank tensor-parallel weight slices directly from safetensors memory maps,
manages fixed-slot GatedDeltaNet recurrent/conv states plus Rust-planned 16-token paged
KV caches for the 10 full-attention layers, and captures full-40-layer CUDA Graphs across
power-of-two decode batch buckets.
"""

from __future__ import annotations

import ctypes
from dataclasses import dataclass
from datetime import timedelta
import json
import mmap
import os
from pathlib import Path
from typing import Any

import numpy as np
import torch
import torch.distributed as dist
from safetensors import safe_open

from zstep_worker.kernels_sm120.qwen35_ops import (
    fused_add_zero_centered_rmsnorm,
    get_qwen35_cuda_ext,
    linear_attn_decode_step,
    linear_attn_prefill_varlen,
    max_moe_tiles_for_rows,
    paged_gqa_decode_gated,
    paged_gqa_prefill_varlen_gated,
    qwen35_attn_prologue,
    sparse_moe_tp8_forward,
    zero_centered_rmsnorm,
)
from zstep_worker.kernels_sm120.shm_allreduce import (
    HostMappedShmAllReduce,
    get_host_mapped_allreduce_extension,
)
from zstep_worker.shm_bridge import SeqResult, ShmBridge, StepHeader, StepPlanView

DECODE_GRAPH_BUCKETS: tuple[int, ...] = (1, 2, 4, 8, 16, 32, 64, 128)


@dataclass(frozen=True)
class Qwen35MoeTextConfig:
    """Validated architectural constants for Qwen3.5-35B-A3B text backbone."""

    vocab_size: int
    hidden_size: int
    num_hidden_layers: int
    layer_types: tuple[str, ...]
    rms_norm_eps: float
    num_attention_heads: int
    num_key_value_heads: int
    head_dim: int
    rope_theta: float
    partial_rotary_factor: float
    linear_num_key_heads: int
    linear_key_head_dim: int
    linear_num_value_heads: int
    linear_value_head_dim: int
    linear_conv_kernel_dim: int
    num_experts: int
    num_experts_per_tok: int
    moe_intermediate_size: int
    shared_expert_intermediate_size: int

    @classmethod
    def from_model_dir(cls, model_dir: Path) -> Qwen35MoeTextConfig:
        raw = json.loads((model_dir / "config.json").read_text(encoding="utf-8"))
        tc: dict[str, Any] = raw.get("text_config", raw)
        rope = tc["rope_parameters"]
        layer_types = tuple(tc["layer_types"])
        cfg = cls(
            vocab_size=int(tc["vocab_size"]),
            hidden_size=int(tc["hidden_size"]),
            num_hidden_layers=int(tc["num_hidden_layers"]),
            layer_types=layer_types,
            rms_norm_eps=float(tc["rms_norm_eps"]),
            num_attention_heads=int(tc["num_attention_heads"]),
            num_key_value_heads=int(tc["num_key_value_heads"]),
            head_dim=int(tc["head_dim"]),
            rope_theta=float(rope["rope_theta"]),
            partial_rotary_factor=float(rope["partial_rotary_factor"]),
            linear_num_key_heads=int(tc["linear_num_key_heads"]),
            linear_key_head_dim=int(tc["linear_key_head_dim"]),
            linear_num_value_heads=int(tc["linear_num_value_heads"]),
            linear_value_head_dim=int(tc["linear_value_head_dim"]),
            linear_conv_kernel_dim=int(tc["linear_conv_kernel_dim"]),
            num_experts=int(tc["num_experts"]),
            num_experts_per_tok=int(tc["num_experts_per_tok"]),
            moe_intermediate_size=int(tc["moe_intermediate_size"]),
            shared_expert_intermediate_size=int(tc["shared_expert_intermediate_size"]),
        )
        if (
            cfg.hidden_size != 2048
            or cfg.num_hidden_layers != 40
            or cfg.num_attention_heads != 16
            or cfg.num_key_value_heads != 2
            or cfg.head_dim != 256
            or cfg.linear_num_key_heads != 16
            or cfg.linear_key_head_dim != 128
            or cfg.linear_num_value_heads != 32
            or cfg.linear_value_head_dim != 128
            or cfg.linear_conv_kernel_dim != 4
            or cfg.num_experts != 256
            or cfg.num_experts_per_tok != 8
            or cfg.moe_intermediate_size != 512
            or cfg.shared_expert_intermediate_size != 512
            or cfg.vocab_size % 8 != 0
        ):
            raise ValueError(f"Unsupported Qwen3.5 MoE TP=8 shape profile: {cfg}")
        return cfg


@dataclass
class LinearAttnLayerWeights:
    input_norm_w: torch.Tensor
    w_qkvzba: torch.Tensor
    conv_w: torch.Tensor
    a_log: torch.Tensor
    dt_bias: torch.Tensor
    gated_norm_w: torch.Tensor
    w_out_proj: torch.Tensor
    post_attn_norm_w: torch.Tensor
    w_router_shared_gu: torch.Tensor
    w_shared_down: torch.Tensor
    w_experts_gate_up: torch.Tensor
    w_experts_down: torch.Tensor
    state_layer_idx: int


@dataclass
class FullAttnLayerWeights:
    input_norm_w: torch.Tensor
    w_qkv: torch.Tensor
    q_norm_w: torch.Tensor
    k_norm_w: torch.Tensor
    w_o_proj: torch.Tensor
    post_attn_norm_w: torch.Tensor
    w_router_shared_gu: torch.Tensor
    w_shared_down: torch.Tensor
    w_experts_gate_up: torch.Tensor
    w_experts_down: torch.Tensor
    kv_layer_idx: int


class _Tp8ArgmaxShmChannel:
    """Host-mapped SHM ring for lock-free TP=8 Top-2 `(logit, token_id)` gather."""

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
        unaligned_bytes = 256 + 4 * 8 * (max_rows_per_step * 2) * 8
        self.total_bytes: int = (unaligned_bytes + 4095) & ~4095
        self._closed: bool = False
        self._owns_unlink: bool = False
        self._shm_path: str = f"/dev/shm/{name}"
        self._ar_ext = get_host_mapped_allreduce_extension()
        self._qwen_ext = get_qwen35_cuda_ext()

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

        self._host_addr: int = ctypes.addressof(ctypes.c_char.from_buffer(self._mapping))
        self._dev_addr: int = self._ar_ext.register_mapped_host(
            self._host_addr, self.total_bytes
        )
        self._registered: bool = True
        self.step_counter: torch.Tensor = torch.zeros(1, dtype=torch.int64, device=device)

    def gather_argmax(
        self,
        logits: torch.Tensor,
        *,
        local_top2_vals: torch.Tensor,
        local_top2_ids: torch.Tensor,
        out_tokens: torch.Tensor,
        out_logprob_bits: torch.Tensor,
        vocab_per_rank: int,
    ) -> None:
        rows = logits.shape[0]
        stream_handle = torch.cuda.current_stream(self.device).cuda_stream
        self._qwen_ext.launch_local_top2_bf16(
            logits,
            local_top2_vals,
            local_top2_ids,
            rows,
            stream_handle,
        )
        self._qwen_ext.launch_tp8_argmax_gather_shm(
            local_top2_vals,
            local_top2_ids,
            out_tokens,
            out_logprob_bits,
            self.step_counter,
            self._dev_addr,
            rows,
            self.max_rows_per_step,
            self.rank,
            self.world_size,
            vocab_per_rank,
            stream_handle,
        )

    def close(self) -> None:
        if self._closed:
            return
        torch.cuda.synchronize(self.device)
        if self._registered:
            self._ar_ext.unregister_mapped_host(self._host_addr)
            self._registered = False
        self._mapping.close()
        if self._owns_unlink:
            try:
                os.unlink(self._shm_path)
            except FileNotFoundError:
                pass
        self._closed = True


class _SafetensorShardReader:
    """Lazy safetensors slice reader so each rank only loads its 1/8 shard from mmap."""

    def __init__(self, model_dir: Path) -> None:
        self.model_dir: Path = model_dir
        index_path = model_dir / "model.safetensors.index.json"
        raw_map: dict[str, str] = json.loads(index_path.read_text(encoding="utf-8"))[
            "weight_map"
        ]
        self.weight_map: dict[str, str] = raw_map
        self.prefix: str = (
            "model.language_model."
            if "model.language_model.norm.weight" in raw_map
            else "model."
        )
        self._handles: dict[str, Any] = {}

    def get_slice(self, key: str):
        shard_name = self.weight_map[key]
        handle = self._handles.get(shard_name)
        if handle is None:
            handle = safe_open(str(self.model_dir / shard_name), framework="pt", device="cpu")
            self._handles[shard_name] = handle
        return handle.get_slice(key)

    def close(self) -> None:
        self._handles.clear()


def _build_rope_caches(
    *,
    max_positions: int,
    rotary_dim: int,
    rope_theta: float,
    device: torch.device,
) -> tuple[torch.Tensor, torch.Tensor]:
    inv_freq = 1.0 / (
        rope_theta ** (torch.arange(0, rotary_dim, 2, dtype=torch.float32, device=device) / rotary_dim)
    )
    positions = torch.arange(max_positions, dtype=torch.float32, device=device)
    freqs = torch.outer(positions, inv_freq)
    # Match Qwen3_5MoeTextRotaryEmbedding: cast cos/sin to BF16 before rotation.
    return freqs.cos().to(torch.bfloat16).contiguous(), freqs.sin().to(torch.bfloat16).contiguous()


def _select_decode_bucket(num_seqs: int, max_bucket: int) -> int:
    for b in DECODE_GRAPH_BUCKETS:
        if b >= num_seqs and b <= max_bucket:
            return b
    return max_bucket


class Qwen35MoeStepRunner:
    """Explicit Mode B GPU step runner for Qwen/Qwen3.5-35B-A3B (`TP=8`, `bfloat16`)."""

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
        self.model_dir: Path = model_dir
        self.rank: int = rank
        self.bridge: ShmBridge = bridge
        self.geometry = bridge.superblock.geometry
        self.world_size: int = self.geometry.world_size
        if self.world_size != 8:
            raise ValueError(f"Qwen35MoeStepRunner requires TP=8, got world_size={self.world_size}")
        if self.geometry.page_block_tokens != 16:
            raise ValueError(
                f"Qwen35MoeStepRunner requires page_block_tokens=16, got {self.geometry.page_block_tokens}"
            )

        torch.cuda.set_device(rank)
        self.device: torch.device = torch.device("cuda", rank)
        self.config: Qwen35MoeTextConfig = Qwen35MoeTextConfig.from_model_dir(model_dir)
        self.vocab_per_rank: int = self.config.vocab_size // 8
        self.max_seqs: int = self.geometry.max_seqs
        self.dummy_slot: int = self.max_seqs
        self.max_step_tokens: int = self.geometry.max_step_tokens
        self.max_pages_per_seq: int = self.geometry.max_pages_per_seq
        self.num_kv_pages: int = num_kv_pages
        self.dummy_page: int = num_kv_pages

        dist.init_process_group(
            backend="gloo",
            init_method=f"file://{dist_init_file}",
            rank=rank,
            world_size=self.world_size,
            timeout=timedelta(seconds=300),
        )
        self.exchange_group = dist.group.WORLD
        self.communicator: HostMappedShmAllReduce = HostMappedShmAllReduce(
            name=allreduce_name,
            rank=rank,
            world_size=self.world_size,
            capacity_elements=self.max_step_tokens * self.config.hidden_size,
            blocks=allreduce_blocks,
            threads=allreduce_threads,
            exchange_group=self.exchange_group,
        )
        self.argmax_channel: _Tp8ArgmaxShmChannel = _Tp8ArgmaxShmChannel(
            name=f"{allreduce_name}_argmax",
            rank=rank,
            world_size=self.world_size,
            device=self.device,
            exchange_group=self.exchange_group,
            max_rows_per_step=max(512, self.max_seqs),
        )

        # Load TP=8 weight slices directly from safetensors mmap
        self.embed_weight: torch.Tensor
        self.final_norm_weight: torch.Tensor
        self.lm_head_weight: torch.Tensor
        self.layers: list[LinearAttnLayerWeights | FullAttnLayerWeights] = []
        self._load_tp8_weights()

        max_pos = max(65536, self.max_pages_per_seq * 16 + 16)
        self.cos_cache, self.sin_cache = _build_rope_caches(
            max_positions=max_pos,
            rotary_dim=64,
            rope_theta=self.config.rope_theta,
            device=self.device,
        )

        # Persistent per-slot state (+1 dummy slot so padded CUDA Graph rows never touch real slots)
        total_slots = self.max_seqs + 1
        self.conv_states: torch.Tensor = torch.zeros(
            (30, total_slots, 1024, 3), dtype=torch.bfloat16, device=self.device
        )
        self.ssm_states: torch.Tensor = torch.zeros(
            (30, total_slots, 4, 128, 128), dtype=torch.float32, device=self.device
        )
        # 10 full-attention layers paged KV cache (+1 scratch page for dummy slot)
        self.k_caches: torch.Tensor = torch.zeros(
            (10, self.num_kv_pages + 1, 16, 256), dtype=torch.bfloat16, device=self.device
        )
        self.v_caches: torch.Tensor = torch.zeros(
            (10, self.num_kv_pages + 1, 16, 256), dtype=torch.bfloat16, device=self.device
        )

        # Static execution workspaces sized for max_step_tokens
        mt = self.max_step_tokens
        ms = total_slots
        max_tiles = max_moe_tiles_for_rows(mt)

        self.token_ids_gpu: torch.Tensor = torch.zeros(mt, dtype=torch.int32, device=self.device)
        self.positions_gpu: torch.Tensor = torch.zeros(mt, dtype=torch.int32, device=self.device)
        self.req_slot_ids_gpu: torch.Tensor = torch.full(
            (mt,), self.dummy_slot, dtype=torch.int32, device=self.device
        )
        self.seq_lens_gpu: torch.Tensor = torch.zeros(ms, dtype=torch.int32, device=self.device)
        self.query_start_loc_gpu: torch.Tensor = torch.zeros(
            ms + 1, dtype=torch.int32, device=self.device
        )
        self.sample_rows_gpu: torch.Tensor = torch.zeros(ms, dtype=torch.int64, device=self.device)
        self.page_table_gpu: torch.Tensor = torch.full(
            (ms, self.max_pages_per_seq),
            self.dummy_page,
            dtype=torch.int32,
            device=self.device,
        )

        # Pinned H2D / D2H staging buffers
        self.token_ids_cpu: torch.Tensor = torch.zeros(mt, dtype=torch.int32, pin_memory=True)
        self.positions_cpu: torch.Tensor = torch.zeros(mt, dtype=torch.int32, pin_memory=True)
        self.req_slot_ids_cpu: torch.Tensor = torch.full(
            (mt,), self.dummy_slot, dtype=torch.int32, pin_memory=True
        )
        self.seq_lens_cpu: torch.Tensor = torch.zeros(ms, dtype=torch.int32, pin_memory=True)
        self.query_start_loc_cpu: torch.Tensor = torch.zeros(
            ms + 1, dtype=torch.int32, pin_memory=True
        )
        self.sample_rows_cpu: torch.Tensor = torch.zeros(ms, dtype=torch.int64, pin_memory=True)
        self.page_table_cpu: torch.Tensor = torch.full(
            (ms, self.max_pages_per_seq),
            self.dummy_page,
            dtype=torch.int32,
            pin_memory=True,
        )
        self.out_tokens_cpu: torch.Tensor = torch.zeros((ms, 2), dtype=torch.int32, pin_memory=True)
        self.out_logprobs_cpu: torch.Tensor = torch.zeros((ms, 2), dtype=torch.int32, pin_memory=True)
        self.req_slots_sample_cpu: list[int] = [0] * ms

        # Activation & scratch workspaces
        self.hidden_buf: torch.Tensor = torch.empty(
            (mt, 2048), dtype=torch.bfloat16, device=self.device
        )
        self.normed_buf: torch.Tensor = torch.empty(
            (mt, 2048), dtype=torch.bfloat16, device=self.device
        )
        self.qkvzba_buf: torch.Tensor = torch.empty(
            (mt, 1544), dtype=torch.bfloat16, device=self.device
        )
        self.qkv_conv_buf: torch.Tensor = torch.empty(
            (mt, 1024), dtype=torch.bfloat16, device=self.device
        )
        self.lin_raw_out_buf: torch.Tensor = torch.empty(
            (mt, 512), dtype=torch.bfloat16, device=self.device
        )
        self.lin_gated_out_buf: torch.Tensor = torch.empty(
            (mt, 512), dtype=torch.bfloat16, device=self.device
        )
        self.qkv_attn_buf: torch.Tensor = torch.empty(
            (mt, 1536), dtype=torch.bfloat16, device=self.device
        )
        self.q_attn_buf: torch.Tensor = torch.empty(
            (mt, 512), dtype=torch.bfloat16, device=self.device
        )
        self.gate_attn_buf: torch.Tensor = torch.empty(
            (mt, 512), dtype=torch.bfloat16, device=self.device
        )
        self.attn_out_buf: torch.Tensor = torch.empty(
            (mt, 512), dtype=torch.bfloat16, device=self.device
        )
        self.partial_acc_buf: torch.Tensor = torch.empty(
            (ms, 16, 512), dtype=torch.float32, device=self.device
        )
        self.partial_m_buf: torch.Tensor = torch.empty(
            (ms, 16, 2), dtype=torch.float32, device=self.device
        )
        self.partial_l_buf: torch.Tensor = torch.empty(
            (ms, 16, 2), dtype=torch.float32, device=self.device
        )
        self.proj_local_buf: torch.Tensor = torch.empty(
            (mt, 2048), dtype=torch.bfloat16, device=self.device
        )
        self.reduced_buf: torch.Tensor = torch.empty(
            (mt, 2048), dtype=torch.bfloat16, device=self.device
        )

        # MoE scratch
        self.router_shared_buf: torch.Tensor = torch.empty(
            (mt, 392), dtype=torch.bfloat16, device=self.device
        )
        self.topk_ids_buf: torch.Tensor = torch.empty(
            mt * 8, dtype=torch.int32, device=self.device
        )
        self.topk_weights_buf: torch.Tensor = torch.empty(
            mt * 8, dtype=torch.bfloat16, device=self.device
        )
        self.shared_gate_scale_buf: torch.Tensor = torch.empty(
            mt, dtype=torch.float32, device=self.device
        )
        self.shared_inter_buf: torch.Tensor = torch.empty(
            (mt, 64), dtype=torch.bfloat16, device=self.device
        )
        self.shared_out_buf: torch.Tensor = torch.empty(
            (mt, 2048), dtype=torch.bfloat16, device=self.device
        )
        self.sorted_pair_ids_buf: torch.Tensor = torch.empty(
            max_tiles * 16, dtype=torch.int32, device=self.device
        )
        self.expert_ids_per_tile_buf: torch.Tensor = torch.empty(
            max_tiles, dtype=torch.int32, device=self.device
        )
        self.num_valid_tiles_buf: torch.Tensor = torch.zeros(
            1, dtype=torch.int32, device=self.device
        )
        self.moe_inter_buf: torch.Tensor = torch.empty(
            (mt * 8, 64), dtype=torch.bfloat16, device=self.device
        )
        self.pair_out_buf: torch.Tensor = torch.empty(
            (mt * 8, 2048), dtype=torch.bfloat16, device=self.device
        )
        self.moe_out_buf: torch.Tensor = torch.empty(
            (mt, 2048), dtype=torch.bfloat16, device=self.device
        )

        # LM head + sampling buffers
        self.sample_hidden_buf: torch.Tensor = torch.empty(
            (ms, 2048), dtype=torch.bfloat16, device=self.device
        )
        self.logits_local_buf: torch.Tensor = torch.empty(
            (ms, self.vocab_per_rank), dtype=torch.bfloat16, device=self.device
        )
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

        self.start_event: torch.cuda.Event = torch.cuda.Event(enable_timing=True)
        self.end_event: torch.cuda.Event = torch.cuda.Event(enable_timing=True)
        self.decode_graphs: dict[int, torch.cuda.CUDAGraph] = {}
        self.max_graph_bucket: int = min(self.max_seqs, 128)
        self._warmup_and_capture_decode_graphs()

    def _load_tp8_weights(self) -> None:
        r = self.rank
        reader = _SafetensorShardReader(self.model_dir)
        pfx = reader.prefix

        self.embed_weight = reader.get_slice(f"{pfx}embed_tokens.weight")[:].to(
            device=self.device, dtype=torch.bfloat16, non_blocking=True
        ).contiguous()
        self.final_norm_weight = reader.get_slice(f"{pfx}norm.weight")[:].to(
            device=self.device, dtype=torch.bfloat16, non_blocking=True
        ).contiguous()
        lm_slice = reader.get_slice("lm_head.weight")[
            r * self.vocab_per_rank : (r + 1) * self.vocab_per_rank, :
        ]
        self.lm_head_weight = lm_slice.to(
            device=self.device, dtype=torch.bfloat16, non_blocking=True
        ).contiguous()

        lin_idx = 0
        kv_idx = 0
        for layer_i in range(self.config.num_hidden_layers):
            lp = f"{pfx}layers.{layer_i}."
            in_norm = reader.get_slice(f"{lp}input_layernorm.weight")[:].to(
                device=self.device, dtype=torch.bfloat16, non_blocking=True
            ).contiguous()
            post_norm = reader.get_slice(f"{lp}post_attention_layernorm.weight")[:].to(
                device=self.device, dtype=torch.bfloat16, non_blocking=True
            ).contiguous()

            # MoE weights (common across all 40 layers)
            router_w = reader.get_slice(f"{lp}mlp.gate.weight")[:].to(
                device=self.device, dtype=torch.bfloat16, non_blocking=True
            )
            sh_gate_w = reader.get_slice(f"{lp}mlp.shared_expert_gate.weight")[:].to(
                device=self.device, dtype=torch.bfloat16, non_blocking=True
            )
            sh_gate_pad = torch.zeros((8, 2048), dtype=torch.bfloat16, device=self.device)
            sh_gate_pad[0:1].copy_(sh_gate_w)
            sh_gp = reader.get_slice(f"{lp}mlp.shared_expert.gate_proj.weight")[
                r * 64 : (r + 1) * 64, :
            ].to(device=self.device, dtype=torch.bfloat16, non_blocking=True)
            sh_up = reader.get_slice(f"{lp}mlp.shared_expert.up_proj.weight")[
                r * 64 : (r + 1) * 64, :
            ].to(device=self.device, dtype=torch.bfloat16, non_blocking=True)
            w_router_shared_gu = torch.cat([router_w, sh_gate_pad, sh_gp, sh_up], dim=0).contiguous()

            w_shared_down = reader.get_slice(f"{lp}mlp.shared_expert.down_proj.weight")[
                :, r * 64 : (r + 1) * 64
            ].to(device=self.device, dtype=torch.bfloat16, non_blocking=True).contiguous()

            gu_sl = reader.get_slice(f"{lp}mlp.experts.gate_up_proj")
            eg = gu_sl[:, r * 64 : (r + 1) * 64, :].to(
                device=self.device, dtype=torch.bfloat16, non_blocking=True
            )
            eu = gu_sl[:, 512 + r * 64 : 512 + (r + 1) * 64, :].to(
                device=self.device, dtype=torch.bfloat16, non_blocking=True
            )
            w_experts_gate_up = torch.cat([eg, eu], dim=1).contiguous()
            del eg, eu

            w_experts_down = reader.get_slice(f"{lp}mlp.experts.down_proj")[
                :, :, r * 64 : (r + 1) * 64
            ].to(device=self.device, dtype=torch.bfloat16, non_blocking=True).contiguous()

            ltype = self.config.layer_types[layer_i]
            if ltype == "linear_attention":
                qkv_sl = reader.get_slice(f"{lp}linear_attn.in_proj_qkv.weight")
                lq = qkv_sl[r * 256 : (r + 1) * 256, :].to(self.device, dtype=torch.bfloat16)
                lk = qkv_sl[2048 + r * 256 : 2048 + (r + 1) * 256, :].to(
                    self.device, dtype=torch.bfloat16
                )
                lv = qkv_sl[4096 + r * 512 : 4096 + (r + 1) * 512, :].to(
                    self.device, dtype=torch.bfloat16
                )
                lz = reader.get_slice(f"{lp}linear_attn.in_proj_z.weight")[
                    r * 512 : (r + 1) * 512, :
                ].to(self.device, dtype=torch.bfloat16)
                lb = reader.get_slice(f"{lp}linear_attn.in_proj_b.weight")[
                    r * 4 : (r + 1) * 4, :
                ].to(self.device, dtype=torch.bfloat16)
                la = reader.get_slice(f"{lp}linear_attn.in_proj_a.weight")[
                    r * 4 : (r + 1) * 4, :
                ].to(self.device, dtype=torch.bfloat16)
                w_qkvzba = torch.cat([lq, lk, lv, lz, lb, la], dim=0).contiguous()

                cw_sl = reader.get_slice(f"{lp}linear_attn.conv1d.weight")
                cq = cw_sl[r * 256 : (r + 1) * 256, 0, :].to(self.device, dtype=torch.bfloat16)
                ck = cw_sl[2048 + r * 256 : 2048 + (r + 1) * 256, 0, :].to(
                    self.device, dtype=torch.bfloat16
                )
                cv = cw_sl[4096 + r * 512 : 4096 + (r + 1) * 512, 0, :].to(
                    self.device, dtype=torch.bfloat16
                )
                conv_w = torch.cat([cq, ck, cv], dim=0).contiguous()

                a_log = reader.get_slice(f"{lp}linear_attn.A_log")[r * 4 : (r + 1) * 4].to(
                    self.device, dtype=torch.float32
                ).contiguous()
                dt_bias = reader.get_slice(f"{lp}linear_attn.dt_bias")[r * 4 : (r + 1) * 4].to(
                    self.device, dtype=torch.bfloat16
                ).contiguous()
                gated_norm_w = reader.get_slice(f"{lp}linear_attn.norm.weight")[:].to(
                    self.device, dtype=torch.float32
                ).contiguous()
                w_out = reader.get_slice(f"{lp}linear_attn.out_proj.weight")[
                    :, r * 512 : (r + 1) * 512
                ].to(self.device, dtype=torch.bfloat16).contiguous()

                self.layers.append(
                    LinearAttnLayerWeights(
                        input_norm_w=in_norm,
                        w_qkvzba=w_qkvzba,
                        conv_w=conv_w,
                        a_log=a_log,
                        dt_bias=dt_bias,
                        gated_norm_w=gated_norm_w,
                        w_out_proj=w_out,
                        post_attn_norm_w=post_norm,
                        w_router_shared_gu=w_router_shared_gu,
                        w_shared_down=w_shared_down,
                        w_experts_gate_up=w_experts_gate_up,
                        w_experts_down=w_experts_down,
                        state_layer_idx=lin_idx,
                    )
                )
                lin_idx += 1
            else:
                kv_head = r // 4
                qp = reader.get_slice(f"{lp}self_attn.q_proj.weight")[
                    r * 1024 : (r + 1) * 1024, :
                ].to(self.device, dtype=torch.bfloat16)
                kp = reader.get_slice(f"{lp}self_attn.k_proj.weight")[
                    kv_head * 256 : (kv_head + 1) * 256, :
                ].to(self.device, dtype=torch.bfloat16)
                vp = reader.get_slice(f"{lp}self_attn.v_proj.weight")[
                    kv_head * 256 : (kv_head + 1) * 256, :
                ].to(self.device, dtype=torch.bfloat16)
                w_qkv = torch.cat([qp, kp, vp], dim=0).contiguous()

                qn = reader.get_slice(f"{lp}self_attn.q_norm.weight")[:].to(
                    self.device, dtype=torch.bfloat16
                ).contiguous()
                kn = reader.get_slice(f"{lp}self_attn.k_norm.weight")[:].to(
                    self.device, dtype=torch.bfloat16
                ).contiguous()
                wo = reader.get_slice(f"{lp}self_attn.o_proj.weight")[
                    :, r * 512 : (r + 1) * 512
                ].to(self.device, dtype=torch.bfloat16).contiguous()

                self.layers.append(
                    FullAttnLayerWeights(
                        input_norm_w=in_norm,
                        w_qkv=w_qkv,
                        q_norm_w=qn,
                        k_norm_w=kn,
                        w_o_proj=wo,
                        post_attn_norm_w=post_norm,
                        w_router_shared_gu=w_router_shared_gu,
                        w_shared_down=w_shared_down,
                        w_experts_gate_up=w_experts_gate_up,
                        w_experts_down=w_experts_down,
                        kv_layer_idx=kv_idx,
                    )
                )
                kv_idx += 1

        reader.close()
        torch.cuda.synchronize(self.device)

    def _run_decode_bucket_forward(self, bucket: int) -> None:
        eps = self.config.rms_norm_eps
        sm_scale = self.config.head_dim**-0.5

        hidden = self.hidden_buf[:bucket]
        normed = self.normed_buf[:bucket]
        proj_local = self.proj_local_buf[:bucket]
        reduced = self.reduced_buf[:bucket]

        torch.index_select(
            self.embed_weight, 0, self.token_ids_gpu[:bucket], out=hidden
        )
        zero_centered_rmsnorm(
            hidden, self.layers[0].input_norm_w, out=normed, eps=eps
        )

        for idx, lw in enumerate(self.layers):
            if isinstance(lw, LinearAttnLayerWeights):
                qkvzba = self.qkvzba_buf[:bucket]
                torch.matmul(normed, lw.w_qkvzba.t(), out=qkvzba)
                gated = linear_attn_decode_step(
                    qkvzba,
                    self.conv_states[lw.state_layer_idx],
                    lw.conv_w,
                    lw.a_log,
                    lw.dt_bias,
                    self.ssm_states[lw.state_layer_idx],
                    lw.gated_norm_w,
                    self.positions_gpu[:bucket],
                    self.req_slot_ids_gpu[:bucket],
                    qkv_conv_buf=self.qkv_conv_buf[:bucket],
                    raw_out_buf=self.lin_raw_out_buf[:bucket],
                    gated_out_buf=self.lin_gated_out_buf[:bucket],
                    num_active_rows=bucket,
                    eps=eps,
                )
                torch.matmul(gated, lw.w_out_proj.t(), out=proj_local)
            else:
                qkv = self.qkv_attn_buf[:bucket]
                q_out = self.q_attn_buf[:bucket]
                gate_out = self.gate_attn_buf[:bucket]
                attn_out = self.attn_out_buf[:bucket]
                torch.matmul(normed, lw.w_qkv.t(), out=qkv)
                qwen35_attn_prologue(
                    qkv,
                    self.positions_gpu[:bucket],
                    self.req_slot_ids_gpu[:bucket],
                    self.page_table_gpu,
                    lw.q_norm_w,
                    lw.k_norm_w,
                    self.cos_cache,
                    self.sin_cache,
                    q_out=q_out,
                    gate_out=gate_out,
                    k_cache=self.k_caches[lw.kv_layer_idx],
                    v_cache=self.v_caches[lw.kv_layer_idx],
                    eps=eps,
                )
                paged_gqa_decode_gated(
                    q_out,
                    gate_out,
                    self.k_caches[lw.kv_layer_idx],
                    self.v_caches[lw.kv_layer_idx],
                    self.page_table_gpu,
                    self.seq_lens_gpu[:bucket],
                    self.req_slot_ids_gpu[:bucket],
                    partial_acc=self.partial_acc_buf[:bucket],
                    partial_m=self.partial_m_buf[:bucket],
                    partial_l=self.partial_l_buf[:bucket],
                    out=attn_out,
                    num_active_seqs=bucket,
                    sm_scale=sm_scale,
                )
                torch.matmul(attn_out, lw.w_o_proj.t(), out=proj_local)

            self.communicator.all_reduce(proj_local, output=reduced)
            fused_add_zero_centered_rmsnorm(
                reduced, hidden, lw.post_attn_norm_w, normed_out=normed, eps=eps
            )

            moe_local = sparse_moe_tp8_forward(
                normed,
                lw.w_router_shared_gu,
                lw.w_shared_down,
                lw.w_experts_gate_up,
                lw.w_experts_down,
                router_shared_buf=self.router_shared_buf,
                topk_ids_buf=self.topk_ids_buf,
                topk_weights_buf=self.topk_weights_buf,
                shared_gate_scale_buf=self.shared_gate_scale_buf,
                shared_inter_buf=self.shared_inter_buf,
                shared_out_buf=self.shared_out_buf,
                sorted_pair_ids_buf=self.sorted_pair_ids_buf,
                expert_ids_per_tile_buf=self.expert_ids_per_tile_buf,
                num_valid_tiles_buf=self.num_valid_tiles_buf,
                moe_inter_buf=self.moe_inter_buf,
                pair_out_buf=self.pair_out_buf,
                moe_out_buf=self.moe_out_buf,
                num_active_rows=bucket,
            )
            self.communicator.all_reduce(moe_local, output=reduced)
            if idx + 1 < len(self.layers):
                fused_add_zero_centered_rmsnorm(
                    reduced,
                    hidden,
                    self.layers[idx + 1].input_norm_w,
                    normed_out=normed,
                    eps=eps,
                )
            else:
                fused_add_zero_centered_rmsnorm(
                    reduced, hidden, self.final_norm_weight, normed_out=normed, eps=eps
                )

        logits = self.logits_local_buf[:bucket]
        torch.matmul(normed, self.lm_head_weight.t(), out=logits)
        self.argmax_channel.gather_argmax(
            logits,
            local_top2_vals=self.local_max_vals_buf[:bucket],
            local_top2_ids=self.local_max_ids_buf[:bucket],
            out_tokens=self.out_tokens_gpu[:bucket],
            out_logprob_bits=self.out_logprobs_gpu[:bucket],
            vocab_per_rank=self.vocab_per_rank,
        )

    def _run_prefill_varlen_forward(
        self,
        *,
        total_tokens: int,
        num_seqs: int,
        num_sample_rows: int,
        max_q_len: int,
    ) -> None:
        eps = self.config.rms_norm_eps
        sm_scale = self.config.head_dim**-0.5

        hidden = self.hidden_buf[:total_tokens]
        normed = self.normed_buf[:total_tokens]
        proj_local = self.proj_local_buf[:total_tokens]
        reduced = self.reduced_buf[:total_tokens]

        torch.index_select(
            self.embed_weight, 0, self.token_ids_gpu[:total_tokens], out=hidden
        )
        zero_centered_rmsnorm(
            hidden, self.layers[0].input_norm_w, out=normed, eps=eps
        )

        for idx, lw in enumerate(self.layers):
            if isinstance(lw, LinearAttnLayerWeights):
                qkvzba = self.qkvzba_buf[:total_tokens]
                torch.matmul(normed, lw.w_qkvzba.t(), out=qkvzba)
                gated = linear_attn_prefill_varlen(
                    qkvzba,
                    self.conv_states[lw.state_layer_idx],
                    lw.conv_w,
                    lw.a_log,
                    lw.dt_bias,
                    self.ssm_states[lw.state_layer_idx],
                    lw.gated_norm_w,
                    self.query_start_loc_gpu[: num_seqs + 1],
                    self.positions_gpu[:total_tokens],
                    self.req_slot_ids_gpu[:total_tokens],
                    qkv_conv_buf=self.qkv_conv_buf[:total_tokens],
                    raw_out_buf=self.lin_raw_out_buf[:total_tokens],
                    gated_out_buf=self.lin_gated_out_buf[:total_tokens],
                    num_seqs=num_seqs,
                    eps=eps,
                )
                torch.matmul(gated, lw.w_out_proj.t(), out=proj_local)
            else:
                qkv = self.qkv_attn_buf[:total_tokens]
                q_out = self.q_attn_buf[:total_tokens]
                gate_out = self.gate_attn_buf[:total_tokens]
                attn_out = self.attn_out_buf[:total_tokens]
                torch.matmul(normed, lw.w_qkv.t(), out=qkv)
                qwen35_attn_prologue(
                    qkv,
                    self.positions_gpu[:total_tokens],
                    self.req_slot_ids_gpu[:total_tokens],
                    self.page_table_gpu,
                    lw.q_norm_w,
                    lw.k_norm_w,
                    self.cos_cache,
                    self.sin_cache,
                    q_out=q_out,
                    gate_out=gate_out,
                    k_cache=self.k_caches[lw.kv_layer_idx],
                    v_cache=self.v_caches[lw.kv_layer_idx],
                    eps=eps,
                )
                paged_gqa_prefill_varlen_gated(
                    q_out,
                    gate_out,
                    self.k_caches[lw.kv_layer_idx],
                    self.v_caches[lw.kv_layer_idx],
                    self.page_table_gpu,
                    self.query_start_loc_gpu[: num_seqs + 1],
                    self.seq_lens_gpu[:num_seqs],
                    self.req_slot_ids_gpu[:total_tokens],
                    out=attn_out,
                    num_seqs=num_seqs,
                    max_q_len=max_q_len,
                    sm_scale=sm_scale,
                )
                torch.matmul(attn_out, lw.w_o_proj.t(), out=proj_local)

            self.communicator.all_reduce(proj_local, output=reduced)
            fused_add_zero_centered_rmsnorm(
                reduced, hidden, lw.post_attn_norm_w, normed_out=normed, eps=eps
            )

            moe_local = sparse_moe_tp8_forward(
                normed,
                lw.w_router_shared_gu,
                lw.w_shared_down,
                lw.w_experts_gate_up,
                lw.w_experts_down,
                router_shared_buf=self.router_shared_buf,
                topk_ids_buf=self.topk_ids_buf,
                topk_weights_buf=self.topk_weights_buf,
                shared_gate_scale_buf=self.shared_gate_scale_buf,
                shared_inter_buf=self.shared_inter_buf,
                shared_out_buf=self.shared_out_buf,
                sorted_pair_ids_buf=self.sorted_pair_ids_buf,
                expert_ids_per_tile_buf=self.expert_ids_per_tile_buf,
                num_valid_tiles_buf=self.num_valid_tiles_buf,
                moe_inter_buf=self.moe_inter_buf,
                pair_out_buf=self.pair_out_buf,
                moe_out_buf=self.moe_out_buf,
                num_active_rows=total_tokens,
            )
            self.communicator.all_reduce(moe_local, output=reduced)
            if idx + 1 < len(self.layers):
                fused_add_zero_centered_rmsnorm(
                    reduced,
                    hidden,
                    self.layers[idx + 1].input_norm_w,
                    normed_out=normed,
                    eps=eps,
                )
            else:
                fused_add_zero_centered_rmsnorm(
                    reduced, hidden, self.final_norm_weight, normed_out=normed, eps=eps
                )

        if num_sample_rows > 0:
            sampled_normed = self.sample_hidden_buf[:num_sample_rows]
            torch.index_select(
                normed, 0, self.sample_rows_gpu[:num_sample_rows], out=sampled_normed
            )
            logits = self.logits_local_buf[:num_sample_rows]
            torch.matmul(sampled_normed, self.lm_head_weight.t(), out=logits)
            self.argmax_channel.gather_argmax(
                logits,
                local_top2_vals=self.local_max_vals_buf[:num_sample_rows],
                local_top2_ids=self.local_max_ids_buf[:num_sample_rows],
                out_tokens=self.out_tokens_gpu[:num_sample_rows],
                out_logprob_bits=self.out_logprobs_gpu[:num_sample_rows],
                vocab_per_rank=self.vocab_per_rank,
            )

    def _warmup_and_capture_decode_graphs(self) -> None:
        # Warm up all 3 Triton integer-specialization buckets (==1, >1 & %16!=0, %16==0)
        # for prefill before RuntimeLoop activates zero-recompilation enforcement.
        self.page_table_gpu.fill_(self.dummy_page)
        for warm_tokens in (1, 5, 16):
            self.token_ids_gpu[:warm_tokens].zero_()
            self.positions_gpu[:warm_tokens].copy_(
                torch.arange(warm_tokens, dtype=torch.int32, device=self.device)
            )
            self.req_slot_ids_gpu[:warm_tokens].fill_(self.dummy_slot)
            self.seq_lens_gpu[0] = warm_tokens
            self.query_start_loc_gpu[0] = 0
            self.query_start_loc_gpu[1] = warm_tokens
            self.sample_rows_gpu[0] = warm_tokens - 1
            self._run_prefill_varlen_forward(
                total_tokens=warm_tokens,
                num_seqs=1,
                num_sample_rows=1,
                max_q_len=warm_tokens,
            )
            torch.cuda.synchronize(self.device)
            dist.barrier(group=self.exchange_group)

        buckets = [b for b in DECODE_GRAPH_BUCKETS if b <= self.max_graph_bucket]
        if not buckets or buckets[-1] != self.max_graph_bucket:
            buckets.append(self.max_graph_bucket)

        graph_pool = torch.cuda.graph_pool_handle()
        capture_stream = torch.cuda.Stream(device=self.device)
        for b in buckets:
            self.token_ids_gpu[:b].zero_()
            self.positions_gpu[:b].zero_()
            self.req_slot_ids_gpu[:b].fill_(self.dummy_slot)
            self.seq_lens_gpu[:b].zero_()
            with torch.cuda.stream(capture_stream):
                for _ in range(2):
                    self._run_decode_bucket_forward(b)
            capture_stream.synchronize()
            dist.barrier(group=self.exchange_group)

            g = torch.cuda.CUDAGraph()
            with torch.cuda.graph(g, pool=graph_pool, stream=capture_stream):
                self._run_decode_bucket_forward(b)
            capture_stream.synchronize()
            dist.barrier(group=self.exchange_group)
            self.decode_graphs[b] = g

        # Clear dummy state after capture
        self.conv_states.zero_()
        self.ssm_states.zero_()
        torch.cuda.synchronize(self.device)
        dist.barrier(group=self.exchange_group)

    def execute(self, plan: StepPlanView, header: StepHeader) -> tuple[list[SeqResult], int]:
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
        max_slot_end = min(self.max_seqs, int(slot_np.max()) + 1)
        pt_np = plan.global_kv_page_table(max_slot_end)
        sample_rows_np = plan.sample_row_indices(num_sample)

        # Synchronize only active slot range into pinned host staging buffer
        np.copyto(self.page_table_cpu[:max_slot_end].numpy(), pt_np)
        self.page_table_gpu[:max_slot_end].copy_(
            self.page_table_cpu[:max_slot_end], non_blocking=True
        )

        is_decode = header.flags == 1

        self.start_event.record()
        if is_decode:
            bucket = _select_decode_bucket(num_seqs, self.max_graph_bucket)
            np.copyto(self.token_ids_cpu[:num_seqs].numpy(), tok_np)
            np.copyto(self.positions_cpu[:num_seqs].numpy(), pos_np)
            np.copyto(self.req_slot_ids_cpu[:num_seqs].numpy(), slot_np)
            np.copyto(self.seq_lens_cpu[:num_seqs].numpy(), seq_lens_np)
            if bucket > num_seqs:
                self.token_ids_cpu[num_seqs:bucket].zero_()
                self.positions_cpu[num_seqs:bucket].zero_()
                self.req_slot_ids_cpu[num_seqs:bucket].fill_(self.dummy_slot)
                self.seq_lens_cpu[num_seqs:bucket].zero_()

            self.token_ids_gpu[:bucket].copy_(self.token_ids_cpu[:bucket], non_blocking=True)
            self.positions_gpu[:bucket].copy_(self.positions_cpu[:bucket], non_blocking=True)
            self.req_slot_ids_gpu[:bucket].copy_(self.req_slot_ids_cpu[:bucket], non_blocking=True)
            self.seq_lens_gpu[:bucket].copy_(self.seq_lens_cpu[:bucket], non_blocking=True)

            self.decode_graphs[bucket].replay()
            if self.rank == 0:
                self.out_tokens_cpu[:num_seqs].copy_(
                    self.out_tokens_gpu[:num_seqs], non_blocking=True
                )
                self.out_logprobs_cpu[:num_seqs].copy_(
                    self.out_logprobs_gpu[:num_seqs], non_blocking=True
                )
        else:
            np.copyto(self.token_ids_cpu[:total_tokens].numpy(), tok_np)
            np.copyto(self.positions_cpu[:total_tokens].numpy(), pos_np)
            np.copyto(self.req_slot_ids_cpu[:total_tokens].numpy(), slot_np)
            np.copyto(self.seq_lens_cpu[:num_seqs].numpy(), seq_lens_np)
            np.copyto(self.query_start_loc_cpu[: num_seqs + 1].numpy(), q_start_np)
            if num_sample > 0:
                self.sample_rows_cpu[:num_sample].copy_(torch.from_numpy(sample_rows_np.astype(np.int64)))

            self.token_ids_gpu[:total_tokens].copy_(
                self.token_ids_cpu[:total_tokens], non_blocking=True
            )
            self.positions_gpu[:total_tokens].copy_(
                self.positions_cpu[:total_tokens], non_blocking=True
            )
            self.req_slot_ids_gpu[:total_tokens].copy_(
                self.req_slot_ids_cpu[:total_tokens], non_blocking=True
            )
            self.seq_lens_gpu[:num_seqs].copy_(self.seq_lens_cpu[:num_seqs], non_blocking=True)
            self.query_start_loc_gpu[: num_seqs + 1].copy_(
                self.query_start_loc_cpu[: num_seqs + 1], non_blocking=True
            )
            if num_sample > 0:
                self.sample_rows_gpu[:num_sample].copy_(
                    self.sample_rows_cpu[:num_sample], non_blocking=True
                )

            q_lens = q_start_np[1:] - q_start_np[:-1]
            max_q_len = int(q_lens.max()) if num_seqs > 0 else 1
            self._run_prefill_varlen_forward(
                total_tokens=total_tokens,
                num_seqs=num_seqs,
                num_sample_rows=num_sample,
                max_q_len=max_q_len,
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
                row_idx = int(sample_rows_np[i])
                req_slot = int(slot_np[row_idx])
                outputs.append(
                    SeqResult(
                        req_slot=req_slot,
                        accepted_token=int(toks_out[i][0]),
                        accepted_logprob_bits=int(lps_out[i][0]) & 0xFFFF_FFFF,
                        runner_up_token=int(toks_out[i][1]),
                        runner_up_logprob_bits=int(lps_out[i][1]) & 0xFFFF_FFFF,
                    )
                )
        return outputs, gpu_exec_us

    def close(self) -> None:
        self.argmax_channel.close()
        self.communicator.close()
        if dist.is_initialized():
            dist.destroy_process_group()
