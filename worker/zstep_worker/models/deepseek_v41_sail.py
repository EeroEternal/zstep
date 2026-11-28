# Copyright (c) 2026 zstep Contributors
# SPDX-License-Identifier: MIT

"""Native TP=8 PPU/SAIL (`ZW810E`) runner for DeepSeek-V4.1-Flash (`deepseek_v41`).

Implements the complete 40-layer DeepSeek-V4.1-Flash backbone + 3-layer DSpark MTP
drafter directly over HGGC C++/CUDA kernels, `triton-for-sail` AIU Tensor-Core kernels,
and CUDA-Graph-safe `PpuIcnCollectives` (`ICNLink` P2P IPC + `PCCL` hybrid), with:
- Parallel threaded zero-copy `safetensors` TP=8 weight loading + GPU FP8*UE8M0 dequant
- Exact stateless Engram V2 4-gram normalized lookup (`engram_compressed_vocab_size=99092`)
- Unified `kv_arena` (`[total_slots, 1, 512]` BF16) + `_SwaPagePool` + paged `indexer_k_cache`
- Rollback-free `cr=2` Compressor state (`comp_even_state` indexed by `(pos // 2) & 7`)
- Full-40-layer `CUDAGraph` decode buckets, speculative verify `CUDAGraph`s, and native
  DSpark (`K=5`) `CUDAGraph` drafting (`TRITON_RECOMPILATIONS=0`)
- Optional lightweight CUDA-event operator-family observer (`ZSTEP_SAIL_OBSERVE_PATH`)
  for `evolith` measurement-driven autonomous evolution
"""

from __future__ import annotations

from collections import OrderedDict
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass
from datetime import timedelta
import gc
import json
import math
import os
from pathlib import Path
import time
from typing import Any

import numpy as np
import torch
import torch.distributed as dist
import torch.nn.functional as F

from zstep_worker.env_config import WorkerRuntimeConfig
from zstep_worker.kernels_sail import dsv41_sail_ops as sail_ops
from zstep_worker.kernels_sail.ppu_collectives import (
    PpuIcnCollectives,
    get_sail_ops_extension,
)
from zstep_worker.sampler import SlotPenaltyState, sample_tp8_logits
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
from zstep_worker.spec_decode import VerifiedSequence, seq_results, verify_batch


# ---------------------------------------------------------------------------
# Pure Python / NumPy Initialization Helpers (Zero `.item()` / `.cpu()`)
# ---------------------------------------------------------------------------


def _is_prime_int(n: int) -> bool:
    if n < 2:
        return False
    if n % 2 == 0:
        return n == 2
    if n % 3 == 0:
        return n == 3
    d = 5
    while d * d <= n:
        if n % d == 0 or n % (d + 2) == 0:
            return False
        d += 6
    return True


def _build_engram_tables(
    model_dir: Path,
    *,
    layer_ids: tuple[int, ...],
    max_ngram_size: int,
    engram_vocab_size: int,
    n_heads: int,
    pad_token_id: int,
    expected_compressed_vocab: int,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, int]:
    """Build exact Engram V2 `token_map`, `multipliers`, `primes`, `offsets`, and `pad_id`."""
    from tokenizers import Regex, Tokenizer, normalizers

    tok_json = model_dir / "tokenizer.json"
    if tok_json.is_file():
        backend = Tokenizer.from_file(str(tok_json))
        total_vocab = backend.get_vocab_size(with_added_tokens=True)
    else:
        from transformers import AutoTokenizer

        hf_tok = AutoTokenizer.from_pretrained(str(model_dir), trust_remote_code=True)
        backend = hf_tok.backend_tokenizer
        total_vocab = len(hf_tok)

    sentinel = "\ue000"
    normalizer = normalizers.Sequence(
        [
            normalizers.NFKC(),
            normalizers.NFD(),
            normalizers.StripAccents(),
            normalizers.Lowercase(),
            normalizers.Replace(Regex(r"[ \t\r\n]+"), " "),
            normalizers.Replace(Regex(r"^ $"), sentinel),
            normalizers.Strip(),
            normalizers.Replace(sentinel, " "),
        ]
    )

    key_to_new: dict[str, int] = {}
    lookup = np.zeros(total_vocab, dtype=np.int64)
    for token_id in range(total_vocab):
        text = backend.decode([token_id], skip_special_tokens=False)
        if "\ufffd" in text:
            key = backend.id_to_token(token_id)
        else:
            normalized = normalizer.normalize_str(text)
            key = normalized if normalized else text
        new_id = key_to_new.get(key)
        if new_id is None:
            new_id = len(key_to_new)
            key_to_new[key] = new_id
        lookup[token_id] = new_id

    comp_vocab_size = len(key_to_new)
    if expected_compressed_vocab > 0 and comp_vocab_size != expected_compressed_vocab:
        from transformers import AutoTokenizer

        hf_tok = AutoTokenizer.from_pretrained(str(model_dir), trust_remote_code=True)
        backend = hf_tok.backend_tokenizer
        total_vocab = len(hf_tok)
        key_to_new.clear()
        lookup = np.zeros(total_vocab, dtype=np.int64)
        for token_id in range(total_vocab):
            text = backend.decode([token_id], skip_special_tokens=False)
            if "\ufffd" in text:
                key = backend.id_to_token(token_id)
            else:
                normalized = normalizer.normalize_str(text)
                key = normalized if normalized else text
            new_id = key_to_new.get(key)
            if new_id is None:
                new_id = len(key_to_new)
                key_to_new[key] = new_id
            lookup[token_id] = new_id
        comp_vocab_size = len(key_to_new)
        if comp_vocab_size != expected_compressed_vocab:
            raise RuntimeError(
                f"Engram compressed vocab mismatch: got {comp_vocab_size}, "
                f"expected {expected_compressed_vocab}"
            )

    seen_primes: set[int] = set()
    primes_list: list[list[int]] = []
    offsets_list: list[list[int]] = []
    for _ in layer_ids:
        flat_layer: list[int] = []
        for _ in range(max_ngram_size - 1):
            cur = engram_vocab_size - 1
            for _ in range(n_heads):
                cand = cur + 1
                while not _is_prime_int(cand) or cand in seen_primes:
                    cand += 1
                seen_primes.add(cand)
                flat_layer.append(cand)
                cur = cand
        primes_list.append(flat_layer)
        offsets_list.append(list(np.cumsum([0, *flat_layer[:-1]], dtype=np.int64)))

    max_long = int(np.iinfo(np.int64).max)
    mult_bound = max(1, (max_long // comp_vocab_size) // 2)
    mult_rows: list[np.ndarray] = []
    for lid in layer_ids:
        rng = np.random.default_rng(10007 * lid)
        vals = rng.integers(low=0, high=mult_bound, size=(max_ngram_size,), dtype=np.int64)
        mult_rows.append(vals * 2 + 1)

    pad_comp_id = int(lookup[pad_token_id])
    return (
        lookup,
        np.stack(mult_rows, axis=0).astype(np.int64),
        np.asarray(primes_list, dtype=np.int64),
        np.asarray(offsets_list, dtype=np.int64),
        pad_comp_id,
    )


def _precompute_cos_sin_np(
    dim: int,
    max_seq_len: int,
    *,
    original_seq_len: int,
    base: float,
    factor: float,
    beta_fast: int,
    beta_slow: int,
) -> np.ndarray:
    """Build interleaved `[max_seq_len, dim]` FP32 `[cos_0, sin_0, ...]` matching `model.py` YaRN."""
    half = dim // 2
    idx = np.arange(half, dtype=np.float32)
    freqs = 1.0 / (np.float32(base) ** ((2.0 * idx) / np.float32(dim)))
    if original_seq_len > 0:

        def _corr_dim(rotations: float) -> float:
            return dim * math.log(original_seq_len / (rotations * 2.0 * math.pi)) / (
                2.0 * math.log(base)
            )

        low = max(math.floor(_corr_dim(float(beta_fast))), 0)
        high = min(math.ceil(_corr_dim(float(beta_slow))), dim - 1)
        ramp = np.clip((idx - float(low)) / max(float(high - low), 1.0e-3), 0.0, 1.0).astype(
            np.float32
        )
        smooth = np.float32(1.0) - ramp
        freqs = (freqs / np.float32(factor)) * (np.float32(1.0) - smooth) + freqs * smooth

    angles = np.outer(np.arange(max_seq_len, dtype=np.float32), freqs).astype(np.float32)
    out = np.empty((max_seq_len, dim), dtype=np.float32)
    out[:, 0::2] = np.cos(angles)
    out[:, 1::2] = np.sin(angles)
    return out


class _SwaPagePool:
    """Deterministic LRU pool mapping global 128-token MLA page IDs (`1..N`)
    to physical SWA 128-token pages (`1..M`, `0`=null, `M+1`=dummy)."""

    def __init__(self, num_swa_pages: int, mla_dummy_vb: int) -> None:
        self.num_swa_pages: int = int(num_swa_pages)
        self.swa_dummy_vb: int = self.num_swa_pages + 1
        self.mla_dummy_vb: int = int(mla_dummy_vb)
        self._fwd: np.ndarray = np.zeros(self.mla_dummy_vb + 2, dtype=np.int32)
        self._fwd[self.mla_dummy_vb] = self.swa_dummy_vb
        self._rev: np.ndarray = np.zeros(self.num_swa_pages + 2, dtype=np.int32)
        self._free: list[int] = list(range(self.num_swa_pages, 0, -1))
        self._lru: OrderedDict[int, int] = OrderedDict()
        self.epoch: int = 0

    def lookup(self, mla_vbs: np.ndarray) -> np.ndarray:
        safe = np.clip(mla_vbs, 0, len(self._fwd) - 1)
        return self._fwd[safe]

    def map_active_windows(
        self,
        active_mla_pt: np.ndarray,
        p_min: np.ndarray,
        p_max: np.ndarray,
        *,
        window_tokens: int = 128,
    ) -> np.ndarray:
        b, width = active_mla_pt.shape
        needed: set[int] = set()
        c_lo_arr = np.maximum(0, (p_min - (window_tokens - 1)) // 128)
        c_hi_arr = np.minimum(width - 1, p_max // 128)
        for i in range(b):
            lo = int(c_lo_arr[i])
            hi = int(c_hi_arr[i])
            for col in range(lo, hi + 1):
                vb = int(active_mla_pt[i, col])
                if 0 < vb < self.mla_dummy_vb:
                    needed.add(vb)
                    if vb in self._lru:
                        self._lru.move_to_end(vb)

        changed = False
        for vb in sorted(needed):
            if self._fwd[vb] > 0:
                continue
            if self._free:
                phys = self._free.pop()
            else:
                evict_vb = None
                for cand_vb in self._lru:
                    if cand_vb not in needed:
                        evict_vb = cand_vb
                        break
                if evict_vb is None:
                    raise RuntimeError("SWA page pool exhausted across active 128-token windows")
                phys = self._lru.pop(evict_vb)
                self._fwd[evict_vb] = 0
            self._fwd[vb] = phys
            self._rev[phys] = vb
            self._lru[vb] = phys
            changed = True
        if changed:
            self.epoch += 1

        out = np.zeros_like(active_mla_pt, dtype=np.int32)
        for i in range(b):
            lo = int(c_lo_arr[i])
            hi = int(c_hi_arr[i])
            if hi >= lo:
                row_vbs = active_mla_pt[i, lo : hi + 1]
                out[i, lo : hi + 1] = self.lookup(row_vbs)
        return out


# ---------------------------------------------------------------------------
# Lightweight CUDA-Event Operator-Family Observer (`ZSTEP_SAIL_OBSERVE_PATH`)
# ---------------------------------------------------------------------------


class _SailOperatorObserver:
    """Records unbiased real device CUDA-event durations per operator family when enabled.

    Recycles completed CUDA event pairs across steps (`drain_completed()`) so 100% of
    prefill and decode calls are measured without sampling bias or unbounded event memory.
    """

    FAMILIES: tuple[str, ...] = (
        "icn_allreduce",
        "mxfp4_moe_decode",
        "mxfp4_moe_prefill",
        "sparse_attn_tp8",
        "mhc_mixes",
        "attn_qkv_o_proj",
        "engram_compressor_indexer",
    )

    def __init__(self, out_path: str, rank: int, world_size: int) -> None:
        self.out_path = out_path
        self.rank = rank
        self.world_size = world_size
        self.enabled = bool(out_path)
        self.active = False
        self._pending: dict[str, list[tuple[torch.cuda.Event, torch.cuda.Event]]] = {
            fam: [] for fam in self.FAMILIES
        }
        self._free_pool: list[tuple[torch.cuda.Event, torch.cuda.Event]] = []
        self._accum_us: dict[str, float] = {fam: 0.0 for fam in self.FAMILIES}
        self._sampled_calls: dict[str, int] = {fam: 0 for fam in self.FAMILIES}
        self._call_counts: dict[str, int] = {fam: 0 for fam in self.FAMILIES}
        self._max_inflight_pairs = 8192

    def activate(self, device: torch.device) -> None:
        if not self.enabled:
            return
        torch.cuda.synchronize(device)
        for fam in self.FAMILIES:
            self._pending[fam].clear()
            self._accum_us[fam] = 0.0
            self._sampled_calls[fam] = 0
            self._call_counts[fam] = 0
        self.active = True

    def drain_completed(self) -> None:
        if not self.active:
            return
        for fam in self.FAMILIES:
            pend = self._pending[fam]
            if not pend:
                continue
            keep_idx = 0
            for idx, (e0, e1) in enumerate(pend):
                if e1.query():
                    self._accum_us[fam] += max(0.0, float(e0.elapsed_time(e1)) * 1000.0)
                    self._sampled_calls[fam] += 1
                    self._free_pool.append((e0, e1))
                    keep_idx = idx + 1
                else:
                    break
            if keep_idx > 0:
                del pend[:keep_idx]

    def begin(self, family: str) -> tuple[torch.cuda.Event, torch.cuda.Event] | None:
        if not self.active or torch.cuda.is_current_stream_capturing():
            return None
        self._call_counts[family] = self._call_counts.get(family, 0) + 1
        pend = self._pending[family]
        if self._free_pool:
            ev0, ev1 = self._free_pool.pop()
        elif len(pend) < self._max_inflight_pairs:
            ev0 = torch.cuda.Event(enable_timing=True)
            ev1 = torch.cuda.Event(enable_timing=True)
        else:
            return None
        ev0.record()
        pend.append((ev0, ev1))
        return (ev0, ev1)

    def end(self, pair: tuple[torch.cuda.Event, torch.cuda.Event] | None) -> None:
        if pair is not None:
            pair[1].record()

    def flush_json(self, device: torch.device) -> None:
        if not self.enabled:
            return
        torch.cuda.synchronize(device)
        self.drain_completed()
        summary: dict[str, Any] = {
            "rank": self.rank,
            "world_size": self.world_size,
            "families": {},
        }
        total_us = 0.0
        for fam in self.FAMILIES:
            sampled = int(self._sampled_calls.get(fam, 0))
            fam_sum = float(self._accum_us.get(fam, 0.0))
            calls = int(self._call_counts.get(fam, 0))
            scale = (calls / sampled) if sampled > 0 else 1.0
            est_total_us = fam_sum * scale
            total_us += est_total_us
            summary["families"][fam] = {
                "calls": calls,
                "sampled_calls": sampled,
                "sampled_total_us": round(fam_sum, 3),
                "estimated_total_us": round(est_total_us, 3),
                "mean_us": round(fam_sum / sampled, 3) if sampled > 0 else 0.0,
            }
        summary["total_observed_us"] = round(total_us, 3)
        for fam in self.FAMILIES:
            est = float(summary["families"][fam]["estimated_total_us"])
            summary["families"][fam]["share"] = (
                round(est / total_us, 6) if total_us > 0.0 else 0.0
            )
        target = Path(f"{self.out_path}.rank{self.rank}.json")
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")


# ---------------------------------------------------------------------------
# Per-Layer Weight Containers
# ---------------------------------------------------------------------------


@dataclass
class _CompressorWeights:
    compress_ratio: int
    comp_idx: int  # 0..2 for cr=2 layers (2, 8, 14), -1 for cr=1 (layer 20)
    norm_weight: torch.Tensor
    wkv_weight: torch.Tensor  # FP32 for cr=2, BF16 for cr=1
    wgate_weight: torch.Tensor | None  # FP32 for cr=2, None for cr=1


@dataclass
class _IndexerWeights:
    owns_k: bool
    kv_src_idx: int
    compress_ratio: int
    is_candidate_source: bool
    uses_candidates: bool
    wq_b_bf16: torch.Tensor  # [4 * 128, 1280] BF16
    weights_proj_bf16: torch.Tensor  # [4, 5120] BF16
    wk_bf16: torch.Tensor | None  # [128, 512] BF16
    k_norm_weight: torch.Tensor | None  # [128] BF16


@dataclass
class _EngramWeights:
    layer_hash_idx: int
    vocab_start_idx: int
    vocab_end_idx: int
    embed_weight_u8: torch.Tensor  # [part_rows, 256] uint8 (FP8 E4M3 bytes)
    embed_scale_u8: torch.Tensor  # [part_rows, 8] uint8 (UE8M0 bytes)
    qk_prod_f32: torch.Tensor  # [4, 5120] FP32 (`q_weight.float() * k_weight.float()`)
    wkv_bf16: torch.Tensor  # [25600, 6144] BF16


@dataclass
class _LayerWeights:
    layer_id: int
    arena_swa_layer_idx: int
    compress_ratio: int
    kv_src_idx: int  # 0..3 if compress_ratio > 0 else -1
    attn_norm_weight: torch.Tensor
    ffn_norm_weight: torch.Tensor
    hc_attn_fn_f32: torch.Tensor
    hc_attn_fn_bf16: torch.Tensor
    hc_attn_base_f32: torch.Tensor
    hc_attn_scale_f32: torch.Tensor
    hc_ffn_fn_f32: torch.Tensor
    hc_ffn_fn_bf16: torch.Tensor
    hc_ffn_base_f32: torch.Tensor
    hc_ffn_scale_f32: torch.Tensor
    attn_sink_f32: torch.Tensor  # [8] FP32
    wq_a_bf16: torch.Tensor  # [1280, 5120] BF16
    q_norm_weight: torch.Tensor  # [1280] BF16
    wq_b_bf16: torch.Tensor  # [4096, 1280] BF16
    wkv_bf16: torch.Tensor  # [512, 5120] BF16
    kv_norm_weight: torch.Tensor  # [512] BF16
    wo_a_bf16: torch.Tensor  # [1024, 4096] BF16
    wo_b_bf16: torch.Tensor  # [5120, 1024] BF16
    compressor: _CompressorWeights | None
    indexer: _IndexerWeights | None
    engram: _EngramWeights | None
    gate_weight_f32: torch.Tensor  # [E, 5120] FP32
    gate_bias_f32: torch.Tensor  # [E] FP32
    shared_w13_bf16: torch.Tensor  # [576, 5120] BF16
    shared_w2_bf16: torch.Tensor  # [5120, 288] BF16
    experts_w13_packed: torch.Tensor  # [E, 576, 2560] uint8
    experts_w13_scale: torch.Tensor  # [E, 576, 160] uint8
    experts_w2_packed: torch.Tensor  # [E, 5120, 144] uint8
    experts_w2_scale: torch.Tensor  # [E, 5120, 9] uint8
    top_k: int


@dataclass(frozen=True)
class _SailSpecVerifyContext:
    num_sample: int
    sample_rows_np: np.ndarray
    slot_np: np.ndarray
    tok_np: np.ndarray
    pos_np: np.ndarray
    q_start_np: np.ndarray
    pt_np: np.ndarray
    main_x_out: torch.Tensor | None


@dataclass(frozen=True)
class _SailLaunchedStep:
    parity: int
    num_out: int
    sample_slots_out: list[int]
    spec_ctx: _SailSpecVerifyContext | None = None


# ---------------------------------------------------------------------------
# Native DSpark (`mtp.0..2` + `markov_head`) Drafter on SAIL / ICNLink
# ---------------------------------------------------------------------------


class DeepSeekV41SailDSparkDrafter:
    """Native 3-layer DSpark MTP + Markov head drafter (`K <= 5`, zero NCCL, CUDA-Graphed)."""

    def __init__(self, runner: DeepSeekV41SailStepRunner) -> None:
        self.runner = runner
        self.num_draft_tokens: int = runner.num_spec_tokens
        self.noise_token_id: int = runner.dspark_noise_token_id
        self.vpr: int = runner.vocab_per_rank
        k = self.num_draft_tokens

        max_draft_rows = min(512, runner.max_step_tokens)
        self.draft_buckets: list[int] = [
            b
            for b in DECODE_GRAPH_BUCKETS
            if b <= runner.max_graph_bucket and b * k <= max_draft_rows
        ]
        max_b = self.draft_buckets[-1] if self.draft_buckets else max(1, runner.max_seqs)
        max_rows = max_b * k
        dev = runner.device

        self.draft_tokens_cpu = torch.full(
            (max_rows,), self.noise_token_id, dtype=torch.int32, pin_memory=True
        )
        self.draft_tokens_gpu = torch.full(
            (max_rows,), self.noise_token_id, dtype=torch.int32, device=dev
        )
        self.draft_positions_cpu = torch.zeros(max_rows, dtype=torch.int32, pin_memory=True)
        self.draft_positions_gpu = torch.zeros(max_rows, dtype=torch.int32, device=dev)
        self.draft_last_ctx_pos_cpu = torch.zeros(max_b, dtype=torch.int32, pin_memory=True)
        self.draft_last_ctx_pos_gpu = torch.zeros(max_b, dtype=torch.int32, device=dev)
        self.draft_swa_bt_cpu = torch.zeros(
            (max_b, runner.bt_width), dtype=torch.int32, pin_memory=True
        )
        self.draft_swa_bt_gpu = torch.zeros(
            (max_b, runner.bt_width), dtype=torch.int32, device=dev
        )

        self.ctx_positions_cpu = torch.zeros(
            runner.max_step_tokens, dtype=torch.int32, pin_memory=True
        )
        self.ctx_positions_gpu = torch.zeros(runner.max_step_tokens, dtype=torch.int32, device=dev)
        self.ctx_swa_slots_cpu = torch.zeros(
            runner.max_step_tokens, dtype=torch.int64, pin_memory=True
        )
        self.ctx_swa_slots_gpu = torch.zeros(runner.max_step_tokens, dtype=torch.int64, device=dev)
        self.ctx_rows_cpu = torch.zeros(
            runner.max_step_tokens, dtype=torch.int64, pin_memory=True
        )
        self.ctx_rows_gpu = torch.zeros(runner.max_step_tokens, dtype=torch.int64, device=dev)

        self.markov_vec_buf = torch.zeros(
            (max_b, runner.dspark_markov_rank), dtype=torch.float32, device=dev
        )
        self.step_logits_buf = torch.zeros((max_b, self.vpr), dtype=torch.float32, device=dev)
        self.step_logits_bf16_buf = torch.zeros(
            (max_b, self.vpr), dtype=torch.bfloat16, device=dev
        )
        self.draft_out_tokens_gpu = torch.zeros((k, max_b, 2), dtype=torch.int32, device=dev)
        self.draft_out_logprobs_gpu = torch.zeros((k, max_b, 2), dtype=torch.int32, device=dev)
        self.draft_out_tokens_cpu = torch.zeros((k, max_b), dtype=torch.int32, pin_memory=True)
        self._draft_out_tokens_np = self.draft_out_tokens_cpu.numpy()

        self._w_offs_128 = torch.arange(128, dtype=torch.int32, device=dev)
        self._k_offs = torch.arange(k, dtype=torch.int32, device=dev)
        self.draft_topk_buf = torch.full((max_rows, 256), -1, dtype=torch.int32, device=dev)
        self.draft_graphs: dict[int, torch.cuda.CUDAGraph] = {}

    def store_context_kv(
        self,
        main_x: torch.Tensor,
        *,
        ctx_rows_np: np.ndarray,
        ctx_slots_np: np.ndarray,
        positions_np: np.ndarray,
        page_table_np: np.ndarray,
    ) -> None:
        n_ctx = int(ctx_rows_np.shape[0])
        if n_ctx == 0:
            return
        runner = self.runner
        max_ctx = min(runner.max_model_len, page_table_np.shape[1] * 128)
        pos_sel = positions_np[ctx_rows_np].astype(np.int64)
        valid = (pos_sel >= 0) & (pos_sel < max_ctx) & ((pos_sel // 128) < page_table_np.shape[1])
        if not bool(np.all(valid)):
            ctx_rows_np = ctx_rows_np[valid]
            ctx_slots_np = ctx_slots_np[valid]
            pos_sel = pos_sel[valid]
            n_ctx = int(ctx_rows_np.shape[0])
            if n_ctx == 0:
                return

        mla_vb = (page_table_np[ctx_slots_np, pos_sel // 128] + 1).astype(np.int32)
        if runner._swa_pool is not None:
            phys_vb = runner._swa_pool.lookup(mla_vb).astype(np.int64)
        else:
            phys_vb = mla_vb.astype(np.int64)
        base_slots = np.where(phys_vb > 0, phys_vb * 128 + (pos_sel % 128), 0).astype(np.int64)

        np.copyto(self.ctx_positions_cpu[:n_ctx].numpy(), pos_sel.astype(np.int32))
        np.copyto(self.ctx_swa_slots_cpu[:n_ctx].numpy(), base_slots)
        self.ctx_positions_gpu[:n_ctx].copy_(self.ctx_positions_cpu[:n_ctx], non_blocking=True)
        self.ctx_swa_slots_gpu[:n_ctx].copy_(self.ctx_swa_slots_cpu[:n_ctx], non_blocking=True)

        if n_ctx == main_x.shape[0] and bool(np.array_equal(ctx_rows_np, np.arange(n_ctx))):
            mx = main_x
        else:
            np.copyto(self.ctx_rows_cpu[:n_ctx].numpy(), ctx_rows_np.astype(np.int64))
            self.ctx_rows_gpu[:n_ctx].copy_(self.ctx_rows_cpu[:n_ctx], non_blocking=True)
            mx = main_x.index_select(0, self.ctx_rows_gpu[:n_ctx])

        q_in = sail_ops.act_quant_ue8m0(mx, out=runner.act_quant_buf[:n_ctx])
        pos_i32 = self.ctx_positions_gpu[:n_ctx]
        base_s = self.ctx_swa_slots_gpu[:n_ctx]
        for stage_idx, stage in enumerate(runner.mtp_layers):
            kv = runner.kv_buf[:n_ctx]
            torch.matmul(q_in, stage.wkv_bf16.t(), out=kv)
            sail_ops.rmsnorm_bf16(kv, stage.kv_norm_weight, eps=runner.norm_eps, out=kv)
            sail_ops.apply_rope_inplace(
                kv.unsqueeze(1), runner.swa_cos_sin_gpu, pos_i32, rope_dim=64, inverse=False
            )
            sail_ops.inplace_act_quant_ue8m0(kv)
            dst_slots = runner.swa_arena_base[stage.arena_swa_layer_idx] + base_s
            runner.kv_arena.view(-1, 512).index_copy_(0, dst_slots, kv)

    def _run_draft_forward_and_markov(self, b: int) -> None:
        runner = self.runner
        k = self.num_draft_tokens
        rows = b * k

        toks_i64 = self.draft_tokens_gpu[:rows].to(torch.int64).clamp(
            min=0, max=runner.vocab_size - 1
        )
        emb = F.embedding(toks_i64, runner.embed_weight)
        h = runner.h_stream_buf[0][:rows]
        h.copy_(emb.unsqueeze(1).expand(rows, 4, runner.hidden_size))
        pre_mix = runner.pre_mix_buf[0][:rows]
        pre_mix.zero_()
        pre_mix[:, 0] = 1.0

        # Build `[b, 128]` base SWA context slots for `last_ctx_pos - 127 .. last_ctx_pos`
        last_p = self.draft_last_ctx_pos_gpu[:b]
        win_p = last_p[:, None] - 127 + self._w_offs_128[None, :]
        valid_win = win_p >= 0
        safe_wp = win_p.clamp(min=0)
        b_arange = runner.arange_seqs_i64[:b, None]
        phys_p = self.draft_swa_bt_gpu[b_arange, (safe_wp // 128).to(torch.int64)]
        valid_ctx = valid_win & (phys_p > 0)
        base_ctx_slots = phys_p * 128 + (safe_wp % 128)

        pos_i32 = self.draft_positions_gpu[:rows]
        cur_idx = 0
        for stage_idx, stage in enumerate(runner.mtp_layers):
            h_cur = runner.h_stream_buf[cur_idx][:rows]
            h_nxt = runner.h_stream_buf[1 - cur_idx][:rows]
            pm_cur = runner.pre_mix_buf[cur_idx][:rows]
            pm_nxt = runner.pre_mix_buf[1 - cur_idx][:rows]

            attn_pre = runner.tmp_pre_buf[:rows]
            attn_post = runner.tmp_post_buf[:rows]
            attn_comb = runner.tmp_comb_buf[:rows]
            sail_ops.mhc_mixes_sinkhorn(
                h_cur,
                stage.hc_attn_fn_f32,
                stage.hc_attn_fn_bf16,
                stage.hc_attn_scale_f32,
                stage.hc_attn_base_f32,
                pre_out=attn_pre,
                post_out=attn_post,
                comb_out=attn_comb,
                raw_proj_buf=runner.mhc_proj_buf[:rows],
                split_k_part_buf=runner.mhc_split_k_part_buf,
                split_k=runner.mhc_split_k,
                fused_max_rows=runner.mhc_fused_max_rows,
                norm_eps=runner.norm_eps,
                hc_eps=runner.hc_eps,
                sinkhorn_iters=runner.hc_sinkhorn_iters,
            )
            x_norm = runner.normed_buf[:rows]
            sail_ops.mhc_pre_rmsnorm(
                h_cur,
                pm_cur,
                stage.attn_norm_weight,
                normed_out=x_norm,
                norm_eps=runner.norm_eps,
            )

            # DSpark bidirectional draft attention (`128` context slots + `k` scratch slots -> `256`)
            q_in = sail_ops.act_quant_ue8m0(x_norm, out=runner.act_quant_buf[:rows])
            qr = runner.qr_buf[:rows]
            torch.matmul(q_in, stage.wq_a_bf16.t(), out=qr)
            sail_ops.rmsnorm_bf16(qr, stage.q_norm_weight, eps=runner.norm_eps, out=qr)
            sail_ops.inplace_act_quant_ue8m0(qr)
            q_flat = runner.q_buf[:rows].view(rows, -1)
            torch.matmul(qr, stage.wq_b_bf16.t(), out=q_flat)
            q = q_flat.view(rows, 8, 512)
            sail_ops.apply_rope_inplace(
                q, runner.swa_cos_sin_gpu, pos_i32, rope_dim=64, inverse=False
            )

            dkv = runner.kv_buf[:rows]
            torch.matmul(q_in, stage.wkv_bf16.t(), out=dkv)
            sail_ops.rmsnorm_bf16(dkv, stage.kv_norm_weight, eps=runner.norm_eps, out=dkv)
            sail_ops.apply_rope_inplace(
                dkv.unsqueeze(1), runner.swa_cos_sin_gpu, pos_i32, rope_dim=64, inverse=False
            )
            sail_ops.inplace_act_quant_ue8m0(dkv)

            scratch_base = runner.dspark_scratch_arena_base[stage_idx]
            scratch_slots = (
                scratch_base
                + runner.arange_seqs_i64[:b, None] * 8
                + self._k_offs[None, :].to(torch.int64)
            ).reshape(-1)
            runner.kv_arena.view(-1, 512).index_copy_(0, scratch_slots, dkv)

            topk_view = self.draft_topk_buf[:rows]
            topk_3d = topk_view.view(b, k, 256)
            ctx_slots = torch.where(
                valid_ctx,
                runner.swa_arena_base[stage.arena_swa_layer_idx] + base_ctx_slots,
                -1,
            )
            topk_3d[:, :, :128] = ctx_slots[:, None, :].expand(b, k, 128)
            draft_s2d = (
                scratch_base
                + runner.arange_seqs_i64[:b, None].to(torch.int32) * 8
                + self._k_offs[None, :]
            )
            topk_3d[:, :, 128 : 128 + k] = draft_s2d[:, None, :].expand(b, k, k)
            topk_3d[:, :, 128 + k :] = -1

            attn_o = runner.attn_out_buf[:rows]
            sail_ops.sparse_attn_tp8_forward(
                q,
                runner.kv_arena,
                stage.attn_sink_f32,
                topk_view,
                out=attn_o,
                partial_acc=runner.attn_partial_acc[: rows * 64],
                partial_m=runner.attn_partial_m[: rows * 64],
                partial_l=runner.attn_partial_l[: rows * 64],
                q_pad64_buf=runner.attn_q_pad64_buf[:rows],
                sm_scale=runner.sm_scale,
                backend=runner.sparse_attn_backend,
            )
            sail_ops.apply_rope_inplace(
                attn_o, runner.swa_cos_sin_gpu, pos_i32, rope_dim=64, inverse=True
            )
            o_flat = attn_o.view(rows, 4096)
            sail_ops.inplace_act_quant_ue8m0(o_flat)
            o_lora = runner.o_lora_buf[:rows]
            torch.matmul(o_flat, stage.wo_a_bf16.t(), out=o_lora)
            sail_ops.inplace_act_quant_ue8m0(o_lora)
            attn_local = runner.sub_local_buf[:rows]
            torch.matmul(o_lora, stage.wo_b_bf16.t(), out=attn_local)
            attn_red = runner.sub_red_buf[:rows]
            runner.icn_comm.all_reduce_bf16(attn_local, output=attn_red)
            sail_ops.mhc_post(attn_red, h_cur, attn_post, attn_comb, out=h_nxt)

            ffn_post = runner.tmp_post_buf[:rows]
            ffn_comb = runner.tmp_comb_buf[:rows]
            sail_ops.mhc_mixes_sinkhorn(
                h_nxt,
                stage.hc_ffn_fn_f32,
                stage.hc_ffn_fn_bf16,
                stage.hc_ffn_scale_f32,
                stage.hc_ffn_base_f32,
                pre_out=pm_nxt,
                post_out=ffn_post,
                comb_out=ffn_comb,
                raw_proj_buf=runner.mhc_proj_buf[:rows],
                split_k_part_buf=runner.mhc_split_k_part_buf,
                split_k=runner.mhc_split_k,
                fused_max_rows=runner.mhc_fused_max_rows,
                norm_eps=runner.norm_eps,
                hc_eps=runner.hc_eps,
                sinkhorn_iters=runner.hc_sinkhorn_iters,
            )
            sail_ops.mhc_pre_rmsnorm(
                h_nxt,
                attn_pre,
                stage.ffn_norm_weight,
                normed_out=x_norm,
                norm_eps=runner.norm_eps,
            )
            moe_local = sail_ops.dsv41_moe_tp8_forward(
                x_norm,
                gate_weight=stage.gate_weight_f32,
                gate_bias=stage.gate_bias_f32,
                shared_w13_bf16=stage.shared_w13_bf16,
                shared_w2_bf16=stage.shared_w2_bf16,
                experts_w13_packed=stage.experts_w13_packed,
                experts_w13_scale=stage.experts_w13_scale,
                experts_w2_packed=stage.experts_w2_packed,
                experts_w2_scale=stage.experts_w2_scale,
                top_k=stage.top_k,
                route_scale=runner.routed_scaling_factor,
                swiglu_limit=runner.swiglu_limit,
                router_logits_buf=runner.router_logits_buf,
                topk_weights_buf=runner.topk_weights_buf,
                topk_ids_buf=runner.topk_ids_buf,
                shared_gu_buf=runner.shared_gu_buf,
                shared_out_buf=runner.shared_out_buf,
                moe_inter_buf=runner.moe_inter_buf,
                sorted_pair_ids_buf=runner.sorted_pair_ids_buf,
                expert_ids_per_tile_buf=runner.expert_ids_per_tile_buf,
                num_valid_tiles_buf=runner.num_valid_tiles_buf,
                pair_out_buf=runner.pair_out_buf,
                moe_out_buf=runner.sub_local_buf,
                act_quant_x_buf=runner.act_quant_buf,
                gemv_max_rows=runner.moe_gemv_max_rows,
                gemv_threads=runner.moe_gemv_threads,
                block_m=runner.moe_block_m,
                num_warps=runner.moe_num_warps,
            )
            moe_red = runner.sub_red_buf[:rows]
            runner.icn_comm.all_reduce_bf16(moe_local, output=moe_red)
            sail_ops.mhc_post(moe_red, h_nxt, ffn_post, ffn_comb, out=h_cur)

        # Collapse final DSpark stream + norm + base logits + 5-step Markov chain
        head_hidden = runner.normed_buf[:rows]
        sail_ops.mhc_pre_rmsnorm(
            runner.h_stream_buf[0][:rows],
            runner.pre_mix_buf[1][:rows],
            runner.dspark_final_norm_weight,
            normed_out=head_hidden,
            norm_eps=runner.norm_eps,
        )
        base_logits = runner.logits_f32_buf[:rows]
        torch.matmul(head_hidden.float(), runner.head_weight_f32.t(), out=base_logits)
        base_3d = base_logits.view(b, k, self.vpr)

        step_f32 = self.step_logits_buf[:b]
        step_bf16 = self.step_logits_bf16_buf[:b]
        m_vec = self.markov_vec_buf[:b]
        w2_t = runner.dspark_markov_head_f32_t
        local_vals = runner.local_top2_vals_buf[:b]
        local_ids = runner.local_top2_ids_buf[:b]
        prev_tok = self.draft_tokens_gpu[0:rows:k].to(torch.int64).clamp(
            min=0, max=runner.vocab_size - 1
        )
        for s_idx in range(k):
            torch.index_select(runner.dspark_markov_embed_f32, 0, prev_tok, out=m_vec)
            torch.addmm(base_3d[:, s_idx, :], m_vec, w2_t, out=step_f32)
            step_bf16.copy_(step_f32.to(torch.bfloat16))
            runner.icn_comm.gather_top2_from_logits(
                step_bf16,
                local_top2_vals_buf=local_vals,
                local_top2_ids_buf=local_ids,
                out_tokens=self.draft_out_tokens_gpu[s_idx, :b],
                out_logprob_bits=self.draft_out_logprobs_gpu[s_idx, :b],
                vocab_per_rank=self.vpr,
            )
            prev_tok = self.draft_out_tokens_gpu[s_idx, :b, 0].to(torch.int64).clamp(
                min=0, max=runner.vocab_size - 1
            )

    @torch.inference_mode()
    def capture_graphs(self, graph_pool: Any, capture_stream: torch.cuda.Stream) -> None:
        if self.runner.runtime_config.disable_draft_graphs:
            return
        runner = self.runner
        k = self.num_draft_tokens
        for b in self.draft_buckets:
            rows = b * k
            self.draft_tokens_gpu[:rows].fill_(self.noise_token_id)
            self.draft_tokens_gpu[0:rows:k].zero_()
            self.draft_positions_gpu[:rows].copy_(
                torch.arange(rows, dtype=torch.int32, device=runner.device) % k + 1
            )
            self.draft_last_ctx_pos_gpu[:b].zero_()
            self.draft_swa_bt_gpu[:b].zero_()

            with torch.cuda.stream(capture_stream):
                for _ in range(2):
                    self._run_draft_forward_and_markov(b)
            capture_stream.synchronize()
            dist.barrier(group=runner.exchange_group)

            g = torch.cuda.CUDAGraph()
            with torch.cuda.graph(g, pool=graph_pool, stream=capture_stream):
                self._run_draft_forward_and_markov(b)
            capture_stream.synchronize()
            dist.barrier(group=runner.exchange_group)
            self.draft_graphs[b] = g

    def close(self) -> None:
        self.draft_graphs.clear()

    def _draft_batch(
        self,
        sequences: list[VerifiedSequence],
        *,
        positions_np: np.ndarray,
        page_table_np: np.ndarray,
    ) -> dict[int, list[int]]:
        n = len(sequences)
        if n == 0:
            return {}
        runner = self.runner
        k = self.num_draft_tokens
        max_ctx = min(runner.max_model_len, page_table_np.shape[1] * 128)
        sequences = [
            seq
            for seq in sequences
            if int(positions_np[seq.rows[len(seq.accepted_tokens) - 1]]) + 1 < max_ctx
        ]
        n = len(sequences)
        if n == 0:
            return {}

        req_slots = np.array([seq.req_slot for seq in sequences], dtype=np.intp)
        anchors = np.array([seq.accepted_tokens[-1] for seq in sequences], dtype=np.int32)
        last_ctx_pos = np.array(
            [int(positions_np[seq.rows[len(seq.accepted_tokens) - 1]]) for seq in sequences],
            dtype=np.int64,
        )
        pos_next = last_ctx_pos + 1

        bucket = (
            select_decode_bucket(n, self.draft_buckets[-1])
            if self.draft_buckets and n <= self.draft_buckets[-1]
            else 0
        )
        use_graph = bucket in self.draft_graphs
        eff_b = bucket if use_graph else n
        rows = eff_b * k
        actual_rows = n * k

        active_mla_pt = (page_table_np[req_slots, :] + 1).astype(np.int32)
        if runner._swa_pool is not None:
            swa_pt = runner._swa_pool.map_active_windows(
                active_mla_pt, last_ctx_pos, last_ctx_pos
            )
        else:
            swa_pt = active_mla_pt

        swa_bt_np = self.draft_swa_bt_cpu[:eff_b].numpy()
        swa_bt_np[:n] = swa_pt
        if eff_b > n:
            swa_bt_np[n:eff_b] = 0

        lcp_np = self.draft_last_ctx_pos_cpu[:eff_b].numpy()
        lcp_np[:n] = last_ctx_pos.astype(np.int32)
        if eff_b > n:
            lcp_np[n:eff_b] = -1

        tok_np = self.draft_tokens_cpu[:rows].numpy()
        tok_np[:rows] = self.noise_token_id
        tok_np[0:actual_rows:k] = anchors
        if eff_b > n:
            tok_np[actual_rows:rows:k] = 0

        q_pos_2d = np.minimum(
            pos_next[:, None] + np.arange(k, dtype=np.int64)[None, :], max_ctx - 1
        ).astype(np.int32)
        pos_cpu_np = self.draft_positions_cpu[:rows].numpy()
        pos_cpu_np[:actual_rows] = q_pos_2d.reshape(-1)
        if eff_b > n:
            pos_cpu_np[actual_rows:rows] = 1

        self.draft_swa_bt_gpu[:eff_b].copy_(self.draft_swa_bt_cpu[:eff_b], non_blocking=True)
        self.draft_last_ctx_pos_gpu[:eff_b].copy_(
            self.draft_last_ctx_pos_cpu[:eff_b], non_blocking=True
        )
        self.draft_tokens_gpu[:rows].copy_(self.draft_tokens_cpu[:rows], non_blocking=True)
        self.draft_positions_gpu[:rows].copy_(self.draft_positions_cpu[:rows], non_blocking=True)

        if use_graph:
            self.draft_graphs[eff_b].replay()
        else:
            self._run_draft_forward_and_markov(eff_b)

        if not runner.is_result_rank:
            return {}
        self.draft_out_tokens_cpu[:, :n].copy_(
            self.draft_out_tokens_gpu[:, :n, 0], non_blocking=True
        )
        done_ev = runner.draft_copy_done_event
        done_ev.record()
        done_ev.synchronize()
        mat = self._draft_out_tokens_np[:, :n]
        out_map: dict[int, list[int]] = {}
        for i in range(n):
            lim = min(k, max_ctx - int(pos_next[i]))
            out_map[sequences[i].req_slot] = [int(mat[s, i]) for s in range(lim)]
        return out_map

    def draft(
        self,
        verified: list[VerifiedSequence],
        *,
        main_x_out: torch.Tensor | None,
        positions_np: np.ndarray,
        req_slot_ids_np: np.ndarray,
        query_start_loc_np: np.ndarray,
        page_table_np: np.ndarray,
    ) -> dict[int, list[int]]:
        if main_x_out is None:
            raise RuntimeError("DSpark target forward returned no main_x context tensor")

        by_seq = {seq.batch_index: seq for seq in verified}
        ctx_row_parts: list[np.ndarray] = []
        ctx_slot_parts: list[np.ndarray] = []
        for seq_idx in range(len(query_start_loc_np) - 1):
            begin = int(query_start_loc_np[seq_idx])
            end = int(query_start_loc_np[seq_idx + 1])
            if end <= begin:
                continue
            seq = by_seq.get(seq_idx)
            if seq is None or len(seq.rows) == 1:
                rows_arr = np.arange(begin, end, dtype=np.intp)
                req_slot = int(req_slot_ids_np[begin]) if seq is None else seq.req_slot
            else:
                acc_cnt = len(seq.accepted_tokens)
                rows_arr = np.asarray(seq.rows[:acc_cnt], dtype=np.intp)
                req_slot = seq.req_slot
            if rows_arr.size > 0:
                ctx_row_parts.append(rows_arr)
                ctx_slot_parts.append(np.full(rows_arr.size, req_slot, dtype=np.intp))

        if ctx_row_parts:
            self.store_context_kv(
                main_x_out,
                ctx_rows_np=np.concatenate(ctx_row_parts),
                ctx_slots_np=np.concatenate(ctx_slot_parts),
                positions_np=positions_np,
                page_table_np=page_table_np,
            )

        return self._draft_batch(
            verified,
            positions_np=positions_np,
            page_table_np=page_table_np,
        )


# ---------------------------------------------------------------------------
# Main TP=8 DeepSeek-V4.1-Flash StepRunner on T-Head PPU-ZW810E
# ---------------------------------------------------------------------------


class DeepSeekV41SailStepRunner:
    """Full TP=8 PPU/SAIL execution engine for DeepSeek-V4.1-Flash."""

    backend_name: str = "sail"

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
        runtime_config: WorkerRuntimeConfig | None = None,
    ) -> None:
        t0 = time.monotonic()
        self.model_dir = Path(model_dir)
        self.rank = int(rank)
        self.bridge = bridge
        self.geometry = bridge.superblock.geometry
        self.world_size = int(self.geometry.world_size)
        if self.world_size != 8:
            raise ValueError(
                f"DeepSeekV41SailStepRunner requires TP=8 world_size=8, got {self.world_size}"
            )
        self.pp_size: int = 1
        self.is_first_pp: bool = True
        self.is_last_pp: bool = True
        self.is_result_rank: bool = self.rank == 0
        self._closed: bool = False

        self.runtime_config = (
            WorkerRuntimeConfig.from_env() if runtime_config is None else runtime_config
        )
        torch.cuda.set_device(self.rank)
        self.device = torch.device("cuda", self.rank)
        # Warm up SAIL 1.7 context, PCIe pinned DMA engine, and acBLAS before any H2D upload
        self._init_stage_u8 = torch.zeros(64 * 1024 * 1024, dtype=torch.uint8, pin_memory=True)
        _warm_gpu = torch.zeros(4 * 1024 * 1024, dtype=torch.uint8, device=self.device)
        _warm_gpu.copy_(self._init_stage_u8[: 4 * 1024 * 1024], non_blocking=False)
        _warm_mm = torch.ones((16, 16), dtype=torch.bfloat16, device=self.device)
        _ = _warm_mm @ _warm_mm
        torch.cuda.synchronize(self.device)
        del _warm_gpu, _warm_mm

        # Trigger HGGC extension compile/load early (rank 0 compiles, others wait at Gloo barrier)
        if self.rank == 0:
            get_sail_ops_extension()
            sail_ops.is_flash_mla_available()

        # Keep all SAIL 1.7 / UKI driver calls strictly single-threaded (disable c10d NCCL watchdog thread)
        os.environ.setdefault("TORCH_NCCL_ASYNC_ERROR_HANDLING", "0")
        os.environ.setdefault("TORCH_NCCL_ENABLE_MONITORING", "0")
        os.environ.setdefault("NCCL_SOCKET_IFNAME", "lo")
        os.environ.setdefault("PCCL_SOCKET_IFNAME", "lo")

        if not dist.is_initialized():
            dist.init_process_group(
                backend="gloo",
                init_method=f"file://{dist_init_file}",
                rank=self.rank,
                world_size=self.world_size,
                timeout=timedelta(seconds=300),
            )
        self.exchange_group = dist.group.WORLD
        dist.barrier(group=self.exchange_group)
        if self.rank != 0:
            get_sail_ops_extension()
            sail_ops.is_flash_mla_available()
        dist.barrier(group=self.exchange_group)

        # Parse `config.json` (`text_config` nested dict)
        raw_cfg = json.loads((self.model_dir / "config.json").read_text(encoding="utf-8"))
        tcfg = raw_cfg.get("text_config", raw_cfg)
        self.vocab_size = int(tcfg["vocab_size"])
        self.vocab_per_rank = self.vocab_size // self.world_size
        self.hidden_size = int(tcfg["hidden_size"])
        self.moe_inter_size = int(tcfg["moe_intermediate_size"])
        self.inter_per_rank = self.moe_inter_size // self.world_size
        self.num_layers = int(tcfg["num_hidden_layers"])
        self.num_mtp_layers = int(tcfg.get("num_nextn_predict_layers", 0))
        self.num_heads = int(tcfg["num_attention_heads"])
        self.local_heads = self.num_heads // self.world_size
        self.head_dim = int(tcfg["head_dim"])
        self.rope_dim = int(tcfg["qk_rope_head_dim"])
        self.q_lora_rank = int(tcfg["q_lora_rank"])
        self.o_groups = int(tcfg["o_groups"])
        self.o_lora_rank = int(tcfg["o_lora_rank"])
        self.window_size = int(tcfg["sliding_window"])
        self.norm_eps = float(tcfg.get("rms_norm_eps", 1.0e-20))
        self.sm_scale = float(self.head_dim**-0.5)

        self.n_routed_experts = int(tcfg["n_routed_experts"])
        self.num_experts_per_tok = int(tcfg["num_experts_per_tok"])
        self.routed_scaling_factor = float(tcfg.get("routed_scaling_factor", 1.5))
        self.swiglu_limit = float(tcfg.get("swiglu_limit", 10.0))

        self.kv_source_layers = tuple(int(x) for x in tcfg["kv_source_layer_ids"])
        self.index_source_layers = tuple(int(x) for x in tcfg["index_source_layer_ids"])
        self.index_n_heads = int(tcfg["index_n_heads"])
        self.index_local_heads = self.index_n_heads // self.world_size
        self.index_head_dim = int(tcfg["index_head_dim"])
        self.index_topk = int(tcfg["index_topk"])
        self.index_weight_scale = float((self.index_head_dim**-0.5) * (self.index_n_heads**-0.5))
        self.candidate_source_layer = int(tcfg.get("candidate_source_layer_id", -1))
        self.candidate_topk_blocks = int(tcfg.get("candidate_topk_blocks", 2048))
        self.candidate_block_size = int(tcfg.get("candidate_block_size", 8))
        self.compress_ratios = tuple(int(x) for x in tcfg["compress_ratios"])

        self.hc_mult = int(tcfg.get("hc_mult", 4))
        self.hc_sinkhorn_iters = int(tcfg.get("hc_sinkhorn_iters", 20))
        self.hc_eps = float(tcfg.get("hc_eps", 1.0e-6))

        self.engram_layer_ids = tuple(int(x) for x in tcfg.get("engram_layer_ids", ()))
        self.engram_num_embeddings = tuple(int(x) for x in tcfg.get("engram_num_embeddings", ()))
        self.engram_max_ngram = int(tcfg.get("engram_max_ngram_size", 4))
        self.engram_n_heads = int(tcfg.get("engram_n_heads", 8))
        self.engram_head_dim = int(tcfg.get("engram_head_dim", 256))
        self.engram_pad_id = int(tcfg.get("engram_pad_token_id", 2))
        self.engram_vocab_size = int(tcfg.get("engram_vocab_size", 16000000))
        self.engram_compressed_vocab = int(tcfg.get("engram_compressed_vocab_size", 99092))

        self.spec_method = self.runtime_config.spec_method
        self.num_spec_tokens = int(self.runtime_config.num_spec_tokens)
        self.dspark_enabled = self.spec_method == "dspark" and self.num_spec_tokens > 0
        self.dspark_noise_token_id = int(tcfg.get("dspark_noise_token_id", 128799))
        self.dspark_target_layers = tuple(int(x) for x in tcfg.get("dspark_target_layer_ids", (37, 38, 39)))
        self.dspark_markov_rank = int(tcfg.get("dspark_markov_rank", 256))
        self.dspark_n_experts = int(tcfg.get("dspark_n_routed_experts", 128))
        self.dspark_top_k = int(tcfg.get("dspark_num_experts_per_tok", 3))

        # Approved SAIL / Evolith runtime policy knobs
        self.moe_gemv_max_rows = int(self.runtime_config.sail_moe_gemv_max_rows)
        self.moe_gemv_threads = int(self.runtime_config.sail_moe_gemv_threads)
        self.moe_block_m = int(self.runtime_config.sail_moe_block_m)
        self.moe_num_warps = int(self.runtime_config.sail_moe_num_warps)
        self.sparse_attn_backend = self.runtime_config.sail_sparse_attn_backend
        self.mhc_split_k = int(self.runtime_config.sail_mhc_split_k)
        self.mhc_fused_max_rows = int(self.runtime_config.sail_mhc_fused_max_rows)
        ar_blocks = int(self.runtime_config.sail_ar_blocks or allreduce_blocks)
        ar_threads = int(self.runtime_config.sail_ar_threads or allreduce_threads)

        self.observer = _SailOperatorObserver(
            self.runtime_config.sail_observe_path, self.rank, self.world_size
        )

        # Geometry & KV / RoPE sizing
        self.max_seqs = int(self.geometry.max_seqs)
        self.max_step_tokens = int(self.geometry.max_step_tokens)
        self.num_kv_pages = int(num_kv_pages)
        self.max_model_len = min(
            int(self.runtime_config.max_model_len),
            int(self.geometry.max_pages_per_seq) * 128,
        )
        self.bt_width = min(
            int(self.geometry.max_pages_per_seq),
            (self.max_model_len + 127) // 128,
        )
        self.decode_graph_max_seq_len = min(self.max_model_len, 2048)
        self.max_graph_bucket = min(
            self.max_seqs, int(bridge.superblock.max_decode_graph_seqs)
        )

        # Initialize ICNLink P2P IPC + optional PCCL hybrid communicator
        pccl_min_b = int(self.runtime_config.sail_ar_pccl_min_bytes)
        self.pccl_group: dist.ProcessGroup | None = None
        if pccl_min_b < 67108864:
            os.environ.setdefault("NCCL_SOCKET_IFNAME", "lo")
            os.environ.setdefault("PCCL_SOCKET_IFNAME", "lo")
            self.pccl_group = dist.new_group(
                backend="nccl", timeout=timedelta(seconds=300)
            )
            _warm_pccl = torch.zeros(1024, dtype=torch.bfloat16, device=self.device)
            dist.all_reduce(_warm_pccl, group=self.pccl_group)
            torch.cuda.synchronize(self.device)
            del _warm_pccl

        self.icn_comm = PpuIcnCollectives(
            name=allreduce_name,
            rank=self.rank,
            world_size=self.world_size,
            device=self.device,
            exchange_group=self.exchange_group,
            pccl_group=self.pccl_group,
            blocks=ar_blocks,
            threads=ar_threads,
            oneshot_max_bytes=int(self.runtime_config.sail_ar_oneshot_max_bytes),
            pccl_min_bytes=pccl_min_b,
            max_bf16_ar_elements=max(self.max_step_tokens, 512) * 6144,
            max_f32_ar_elements=max(self.max_seqs, 256) * self.decode_graph_max_seq_len,
            max_bf16_ag_input_elements=max(self.max_seqs, 128) * self.vocab_per_rank,
            max_top2_rows=max(self.max_step_tokens, 512),
        )

        # Build RoPE tables (`swa` base=10000 no YaRN; `compress` base=160000 YaRN factor=16)
        rope_cfg = tcfg.get("rope_scaling") or {}
        swa_cs_np = _precompute_cos_sin_np(
            self.rope_dim,
            self.max_model_len + 32,
            original_seq_len=0,
            base=float(tcfg.get("rope_theta", 10000.0)),
            factor=1.0,
            beta_fast=32,
            beta_slow=1,
        )
        comp_cs_np = _precompute_cos_sin_np(
            self.rope_dim,
            self.max_model_len + 32,
            original_seq_len=int(rope_cfg.get("original_max_position_embeddings", 65536)),
            base=float(tcfg.get("compress_rope_theta", 160000.0)),
            factor=float(rope_cfg.get("factor", 16.0)),
            beta_fast=int(rope_cfg.get("beta_fast", 32)),
            beta_slow=int(rope_cfg.get("beta_slow", 1)),
        )
        self.swa_cos_sin_gpu = self._pinned_to_gpu(torch.from_numpy(swa_cs_np))
        self.comp_cos_sin_gpu = self._pinned_to_gpu(torch.from_numpy(comp_cs_np))

        # Build Engram V2 compressed token map & prime tables
        (
            tok_map_np,
            mult_np,
            primes_np,
            offsets_np,
            self.engram_pad_comp_id,
        ) = _build_engram_tables(
            self.model_dir,
            layer_ids=self.engram_layer_ids,
            max_ngram_size=self.engram_max_ngram,
            engram_vocab_size=self.engram_vocab_size,
            n_heads=self.engram_n_heads,
            pad_token_id=self.engram_pad_id,
            expected_compressed_vocab=self.engram_compressed_vocab,
        )
        self.engram_token_map_gpu = self._pinned_to_gpu(torch.from_numpy(tok_map_np))
        self.engram_mult_gpu = self._pinned_to_gpu(torch.from_numpy(mult_np))
        self.engram_primes_gpu = self._pinned_to_gpu(torch.from_numpy(primes_np))
        self.engram_offsets_gpu = self._pinned_to_gpu(torch.from_numpy(offsets_np))

        # Load all TP=8 sharded & replicated weights in parallel
        self._load_all_weights()
        del self._init_stage_u8
        gc.collect()
        torch.cuda.empty_cache()

        # Allocate unified `kv_arena` + `_SwaPagePool` + `indexer_k_cache` + workspaces
        self._allocate_kv_and_workspaces()

        # Warm up watched Triton kernels across representative shapes BEFORE `RuntimeLoop` spy attach
        self._warmup_watched_triton_kernels()

        # Capture full-40-layer `CUDAGraph` decode buckets (+ verify & DSpark draft graphs)
        self.decode_graphs: dict[int, torch.cuda.CUDAGraph] = {}
        self.verify_graphs: dict[int, torch.cuda.CUDAGraph] = {}
        self.verify_graph_main_x: dict[int, torch.Tensor | None] = {}
        self.verify_buckets: list[int] = []
        self.dspark_drafter: DeepSeekV41SailDSparkDrafter | None = None
        if self.dspark_enabled:
            self.dspark_drafter = DeepSeekV41SailDSparkDrafter(self)
        self._capture_all_graphs()

        self._penalty_state: SlotPenaltyState | None = None
        self.start_events = [torch.cuda.Event(enable_timing=True) for _ in range(2)]
        self.end_events = [torch.cuda.Event(enable_timing=True) for _ in range(2)]
        self._h2d_done_event = torch.cuda.Event()
        self.draft_copy_done_event = torch.cuda.Event()

        torch.cuda.synchronize(self.device)
        self.observer.activate(self.device)
        dist.barrier(group=self.exchange_group)
        if self.rank == 0:
            free_b, tot_b = torch.cuda.mem_get_info(self.device)
            print(
                f"[zstep-sail] DeepSeekV41SailStepRunner TP=8 ready in "
                f"{time.monotonic() - t0:.2f}s | decode_graphs={sorted(self.decode_graphs)} | "
                f"verify_graphs={sorted(self.verify_graphs)} | "
                f"dspark={self.dspark_enabled} | "
                f"free_hbm={free_b / (1024**3):.2f}/{tot_b / (1024**3):.2f} GiB",
                flush=True,
            )

    # -----------------------------------------------------------------------
    # Parallel Zero-Copy Safetensors Loader (`ThreadPoolExecutor`)
    # -----------------------------------------------------------------------

    def _pinned_copy_into_gpu(self, dst_gpu: torch.Tensor, src_cpu: torch.Tensor) -> None:
        """Copy contiguous CPU tensor into `dst_gpu` strictly via `pin_memory=True` chunks."""
        src_c = src_cpu.contiguous()
        src_b = src_c.view(torch.uint8).reshape(-1)
        dst_b = dst_gpu.view(torch.uint8).reshape(-1)
        cap = self._init_stage_u8.numel()
        total = src_b.numel()
        for off in range(0, total, cap):
            nb = min(cap, total - off)
            self._init_stage_u8[:nb].copy_(src_b[off : off + nb])
            dst_b[off : off + nb].copy_(self._init_stage_u8[:nb], non_blocking=False)

    def _pinned_to_gpu(
        self, src_cpu: torch.Tensor, *, dtype: torch.dtype | None = None
    ) -> torch.Tensor:
        t_c = (
            src_cpu.to(dtype=dtype).contiguous()
            if (dtype is not None and src_cpu.dtype != dtype)
            else src_cpu.contiguous()
        )
        dst = torch.zeros(t_c.shape, dtype=t_c.dtype, device=self.device)
        self._pinned_copy_into_gpu(dst, t_c)
        return dst

    def _dequant_fp8_block32_gpu(
        self, w_fp8_cpu: torch.Tensor, s_e8_cpu: torch.Tensor
    ) -> torch.Tensor:
        out_f, in_f = w_fp8_cpu.shape
        w_u8_gpu = self._pinned_to_gpu(w_fp8_cpu.contiguous().view(torch.uint8))
        w_gpu = w_u8_gpu.view(w_fp8_cpu.dtype).float()
        s_u8_gpu = self._pinned_to_gpu(s_e8_cpu.contiguous().view(torch.uint8))
        s_gpu = s_u8_gpu.view(s_e8_cpu.dtype).float()
        deq = (
            w_gpu.view(out_f // 32, 32, in_f // 32, 32)
            * s_gpu.unsqueeze(1).unsqueeze(3)
        ).reshape(out_f, in_f)
        return deq.to(torch.bfloat16).contiguous()

    def _load_all_weights(self) -> None:
        from safetensors import safe_open

        t_load = time.monotonic()
        idx_path = self.model_dir / "model.safetensors.index.json"
        weight_map: dict[str, str] = json.loads(idx_path.read_text(encoding="utf-8"))[
            "weight_map"
        ]

        r = self.rank
        vpr = self.vocab_per_rank
        ipr = self.inter_per_rank

        # Preallocate expert packed weight & scale buffers directly on GPU for zero intermediate copies
        total_layer_defs: list[tuple[str, int, int, int]] = [
            (f"layers.{lid}", lid, self.n_routed_experts, self.num_experts_per_tok)
            for lid in range(self.num_layers)
        ]
        if self.dspark_enabled:
            for stg in range(self.num_mtp_layers):
                total_layer_defs.append(
                    (f"mtp.{stg}", self.num_layers + stg, self.dspark_n_experts, self.dspark_top_k)
                )

        exp_buffers: dict[str, tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]] = {}
        for prefix, _lid, n_exp, _tk in total_layer_defs:
            w13_p = torch.zeros(
                (n_exp, 2 * ipr, self.hidden_size // 2), dtype=torch.uint8, device=self.device
            )
            w13_s = torch.zeros(
                (n_exp, 2 * ipr, self.hidden_size // 32), dtype=torch.uint8, device=self.device
            )
            w2_p = torch.zeros(
                (n_exp, self.hidden_size, ipr // 2), dtype=torch.uint8, device=self.device
            )
            w2_s = torch.zeros(
                (n_exp, self.hidden_size, ipr // 32), dtype=torch.uint8, device=self.device
            )
            exp_buffers[prefix] = (w13_p, w13_s, w2_p, w2_s)

        # Group non-expert & expert load tasks by shard filename so each shard opens once per worker thread
        tasks_by_shard: dict[str, list[tuple[str, Any]]] = {}

        def _add_task(key: str, spec: Any) -> None:
            sh = weight_map[key]
            tasks_by_shard.setdefault(sh, []).append((key, spec))

        _add_task("embed.weight", ("full",))
        _add_task("norm.weight", ("full",))
        _add_task("head.weight", ("row_slice", r * vpr, (r + 1) * vpr))

        kv_src_map = {lid: i for i, lid in enumerate(self.kv_source_layers)}
        cr2_comp_idx = 0
        comp_idx_map: dict[int, int] = {}
        for lid in self.kv_source_layers:
            if self.compress_ratios[lid] == 2:
                comp_idx_map[lid] = cr2_comp_idx
                cr2_comp_idx += 1
            else:
                comp_idx_map[lid] = -1

        for prefix, lid, n_exp, _tk in total_layer_defs:
            is_backbone = lid < self.num_layers
            _add_task(f"{prefix}.attn_norm.weight", ("full",))
            _add_task(f"{prefix}.ffn_norm.weight", ("full",))
            for sub in ("attn", "ffn"):
                _add_task(f"{prefix}.hc_{sub}_fn", ("full",))
                _add_task(f"{prefix}.hc_{sub}_base", ("full",))
                _add_task(f"{prefix}.hc_{sub}_scale", ("full",))

            _add_task(f"{prefix}.attn.attn_sink", ("row_slice", r * 8, (r + 1) * 8))
            _add_task(f"{prefix}.attn.wq_a.weight", ("full",))
            _add_task(f"{prefix}.attn.wq_a.scale", ("full",))
            _add_task(f"{prefix}.attn.q_norm.weight", ("full",))
            _add_task(f"{prefix}.attn.wq_b.weight", ("row_slice", r * 4096, (r + 1) * 4096))
            _add_task(f"{prefix}.attn.wq_b.scale", ("row_slice", r * 128, (r + 1) * 128))
            _add_task(f"{prefix}.attn.wkv.weight", ("full",))
            _add_task(f"{prefix}.attn.wkv.scale", ("full",))
            _add_task(f"{prefix}.attn.kv_norm.weight", ("full",))
            _add_task(f"{prefix}.attn.wo_a.weight", ("row_slice", r * 1024, (r + 1) * 1024))
            _add_task(f"{prefix}.attn.wo_a.scale", ("row_slice", r * 32, (r + 1) * 32))
            _add_task(f"{prefix}.attn.wo_b.weight", ("col_slice", r * 1024, (r + 1) * 1024))
            _add_task(f"{prefix}.attn.wo_b.scale", ("col_slice", r * 32, (r + 1) * 32))

            if is_backbone and lid in kv_src_map:
                _add_task(f"{prefix}.attn.compressor.norm.weight", ("full",))
                _add_task(f"{prefix}.attn.compressor.wkv.weight", ("full",))
                if self.compress_ratios[lid] > 1:
                    _add_task(f"{prefix}.attn.compressor.wgate.weight", ("full",))

            if is_backbone and lid in self.index_source_layers:
                _add_task(f"{prefix}.attn.indexer.wq_b.weight", ("row_slice", r * 512, (r + 1) * 512))
                _add_task(f"{prefix}.attn.indexer.wq_b.scale", ("row_slice", r * 16, (r + 1) * 16))
                _add_task(f"{prefix}.attn.indexer.weights_proj.weight", ("row_slice", r * 4, (r + 1) * 4))
                if lid in kv_src_map:
                    _add_task(f"{prefix}.attn.indexer.wk.weight", ("full",))
                    _add_task(f"{prefix}.attn.indexer.k_norm.weight", ("full",))

            if is_backbone and lid in self.engram_layer_ids:
                eng_idx = self.engram_layer_ids.index(lid)
                tot_rows = self.engram_num_embeddings[eng_idx]
                part_rows = (tot_rows + self.world_size - 1) // self.world_size
                lo_e = min(r * part_rows, tot_rows)
                hi_e = min((r + 1) * part_rows, tot_rows)
                _add_task(f"{prefix}.engram.embed.weight", ("engram_slice", lo_e, hi_e, part_rows, 256))
                _add_task(f"{prefix}.engram.embed.scale", ("engram_slice", lo_e, hi_e, part_rows, 8))
                _add_task(f"{prefix}.engram.q_weight", ("full",))
                _add_task(f"{prefix}.engram.k_weight", ("full",))
                _add_task(f"{prefix}.engram.wkv.weight", ("full",))
                _add_task(f"{prefix}.engram.wkv.scale", ("full",))

            _add_task(f"{prefix}.ffn.gate.weight", ("full",))
            _add_task(f"{prefix}.ffn.gate.bias", ("full",))
            for sw in ("w1", "w3"):
                _add_task(f"{prefix}.ffn.shared_experts.{sw}.weight", ("row_slice", r * ipr, (r + 1) * ipr))
                _add_task(
                    f"{prefix}.ffn.shared_experts.{sw}.scale",
                    ("row_slice", r * (ipr // 32), (r + 1) * (ipr // 32)),
                )
            _add_task(f"{prefix}.ffn.shared_experts.w2.weight", ("col_slice", r * ipr, (r + 1) * ipr))
            _add_task(
                f"{prefix}.ffn.shared_experts.w2.scale",
                ("col_slice", r * (ipr // 32), (r + 1) * (ipr // 32)),
            )

            for e_id in range(n_exp):
                _add_task(f"{prefix}.ffn.experts.{e_id}.w1.weight", ("exp_w13_p", prefix, e_id, 0))
                _add_task(f"{prefix}.ffn.experts.{e_id}.w1.scale", ("exp_w13_s", prefix, e_id, 0))
                _add_task(f"{prefix}.ffn.experts.{e_id}.w3.weight", ("exp_w13_p", prefix, e_id, ipr))
                _add_task(f"{prefix}.ffn.experts.{e_id}.w3.scale", ("exp_w13_s", prefix, e_id, ipr))
                _add_task(f"{prefix}.ffn.experts.{e_id}.w2.weight", ("exp_w2_p", prefix, e_id))
                _add_task(f"{prefix}.ffn.experts.{e_id}.w2.scale", ("exp_w2_s", prefix, e_id))

        if self.dspark_enabled:
            _add_task("mtp.0.main_proj.weight", ("full",))
            _add_task("mtp.0.main_proj.scale", ("full",))
            _add_task("mtp.0.main_norm.weight", ("full",))
            last_m = f"mtp.{self.num_mtp_layers - 1}"
            _add_task(f"{last_m}.norm.weight", ("full",))
            _add_task(f"{last_m}.markov_head.embed.weight", ("full",))
            _add_task(f"{last_m}.markov_head.head.weight", ("row_slice", r * vpr, (r + 1) * vpr))

        loaded_cpu: dict[str, torch.Tensor] = {}

        def _process_shard_cpu(
            shard_item: tuple[str, list[tuple[str, Any]]]
        ) -> list[tuple[str, Any, torch.Tensor | None]]:
            """Pure-CPU safetensors slicing (`zero` CUDA/SAIL driver calls in worker threads!)."""
            shard_file, entries = shard_item
            out_items: list[tuple[str, Any, torch.Tensor | None]] = []
            with safe_open(str(self.model_dir / shard_file), framework="pt", device="cpu") as f:
                for key, spec in entries:
                    kind = spec[0]
                    sl = f.get_slice(key)
                    if kind == "full":
                        out_items.append((key, spec, sl[:].contiguous()))
                    elif kind == "row_slice":
                        out_items.append((key, spec, sl[spec[1] : spec[2]].contiguous()))
                    elif kind == "col_slice":
                        out_items.append((key, spec, sl[:, spec[1] : spec[2]].contiguous()))
                    elif kind == "engram_slice":
                        _, lo_e, hi_e, _part_rows, _cols = spec
                        chunk_u8 = (
                            sl[lo_e:hi_e].contiguous().view(torch.uint8)
                            if hi_e > lo_e
                            else None
                        )
                        out_items.append((key, spec, chunk_u8))
                    elif kind in ("exp_w13_p", "exp_w13_s"):
                        sub_t = sl[r * ipr : (r + 1) * ipr].contiguous().view(torch.uint8)
                        out_items.append((key, spec, sub_t))
                    elif kind == "exp_w2_p":
                        sub_t = (
                            sl[:, r * (ipr // 2) : (r + 1) * (ipr // 2)]
                            .contiguous()
                            .view(torch.uint8)
                        )
                        out_items.append((key, spec, sub_t))
                    elif kind == "exp_w2_s":
                        sub_t = (
                            sl[:, r * (ipr // 32) : (r + 1) * (ipr // 32)]
                            .contiguous()
                            .view(torch.uint8)
                        )
                        out_items.append((key, spec, sub_t))
            return out_items

        shard_items = sorted(tasks_by_shard.items())
        with ThreadPoolExecutor(max_workers=8) as pool:
            for b_start in range(0, len(shard_items), 8):
                batch_shards = shard_items[b_start : b_start + 8]
                for shard_out in pool.map(_process_shard_cpu, batch_shards):
                    for key, spec, cpu_t in shard_out:
                        kind = spec[0]
                        if kind in ("full", "row_slice", "col_slice"):
                            assert cpu_t is not None
                            loaded_cpu[key] = cpu_t
                        elif kind == "engram_slice":
                            _, lo_e, hi_e, part_rows, cols = spec
                            buf_u8 = torch.zeros(
                                (part_rows, cols), dtype=torch.uint8, device=self.device
                            )
                            if cpu_t is not None and hi_e > lo_e:
                                self._pinned_copy_into_gpu(buf_u8[: hi_e - lo_e], cpu_t)
                            loaded_cpu[key] = buf_u8
                        elif kind == "exp_w13_p":
                            assert cpu_t is not None
                            _, pfx, e_id, row_off = spec
                            self._pinned_copy_into_gpu(
                                exp_buffers[pfx][0][e_id, row_off : row_off + ipr], cpu_t
                            )
                        elif kind == "exp_w13_s":
                            assert cpu_t is not None
                            _, pfx, e_id, row_off = spec
                            self._pinned_copy_into_gpu(
                                exp_buffers[pfx][1][e_id, row_off : row_off + ipr], cpu_t
                            )
                        elif kind == "exp_w2_p":
                            assert cpu_t is not None
                            _, pfx, e_id = spec
                            self._pinned_copy_into_gpu(exp_buffers[pfx][2][e_id], cpu_t)
                        elif kind == "exp_w2_s":
                            assert cpu_t is not None
                            _, pfx, e_id = spec
                            self._pinned_copy_into_gpu(exp_buffers[pfx][3][e_id], cpu_t)
                torch.cuda.synchronize(self.device)

        self.embed_weight = self._pinned_to_gpu(
            loaded_cpu.pop("embed.weight"), dtype=torch.bfloat16
        )
        self.final_norm_weight = self._pinned_to_gpu(
            loaded_cpu.pop("norm.weight"), dtype=torch.bfloat16
        )
        self.head_weight_f32 = self._pinned_to_gpu(
            loaded_cpu.pop("head.weight"), dtype=torch.float32
        )

        def _build_layer(prefix: str, lid: int, arena_idx: int, top_k: int) -> _LayerWeights:
            is_backbone = lid < self.num_layers
            cr = self.compress_ratios[lid]
            cur_kv_src = -1
            if cr > 0:
                for src_i, src_lid in enumerate(self.kv_source_layers):
                    if src_lid <= lid and self.compress_ratios[src_lid] == cr:
                        cur_kv_src = src_i

            comp_w: _CompressorWeights | None = None
            if is_backbone and lid in kv_src_map:
                cn = self._pinned_to_gpu(
                    loaded_cpu.pop(f"{prefix}.attn.compressor.norm.weight"),
                    dtype=torch.bfloat16,
                )
                wkv_dt = torch.float32 if cr > 1 else torch.bfloat16
                cwkv = self._pinned_to_gpu(
                    loaded_cpu.pop(f"{prefix}.attn.compressor.wkv.weight"),
                    dtype=wkv_dt,
                )
                cwgate = (
                    self._pinned_to_gpu(
                        loaded_cpu.pop(f"{prefix}.attn.compressor.wgate.weight"),
                        dtype=torch.float32,
                    )
                    if cr > 1
                    else None
                )
                comp_w = _CompressorWeights(
                    compress_ratio=cr,
                    comp_idx=comp_idx_map[lid],
                    norm_weight=cn,
                    wkv_weight=cwkv,
                    wgate_weight=cwgate,
                )

            idx_w: _IndexerWeights | None = None
            if is_backbone and lid in self.index_source_layers:
                owns_k = lid in kv_src_map
                iwq = self._dequant_fp8_block32_gpu(
                    loaded_cpu.pop(f"{prefix}.attn.indexer.wq_b.weight"),
                    loaded_cpu.pop(f"{prefix}.attn.indexer.wq_b.scale"),
                )
                iwp = self._pinned_to_gpu(
                    loaded_cpu.pop(f"{prefix}.attn.indexer.weights_proj.weight"),
                    dtype=torch.bfloat16,
                )
                iwk = (
                    self._pinned_to_gpu(
                        loaded_cpu.pop(f"{prefix}.attn.indexer.wk.weight"),
                        dtype=torch.bfloat16,
                    )
                    if owns_k
                    else None
                )
                ikn = (
                    self._pinned_to_gpu(
                        loaded_cpu.pop(f"{prefix}.attn.indexer.k_norm.weight"),
                        dtype=torch.bfloat16,
                    )
                    if owns_k
                    else None
                )
                idx_w = _IndexerWeights(
                    owns_k=owns_k,
                    kv_src_idx=cur_kv_src,
                    compress_ratio=cr,
                    is_candidate_source=(lid == self.candidate_source_layer),
                    uses_candidates=(0 <= self.candidate_source_layer < lid),
                    wq_b_bf16=iwq,
                    weights_proj_bf16=iwp,
                    wk_bf16=iwk,
                    k_norm_weight=ikn,
                )

            eng_w: _EngramWeights | None = None
            if is_backbone and lid in self.engram_layer_ids:
                eh_idx = self.engram_layer_ids.index(lid)
                tot_rows = self.engram_num_embeddings[eh_idx]
                part_rows = (tot_rows + self.world_size - 1) // self.world_size
                qw = self._pinned_to_gpu(
                    loaded_cpu.pop(f"{prefix}.engram.q_weight"), dtype=torch.float32
                )
                kw = self._pinned_to_gpu(
                    loaded_cpu.pop(f"{prefix}.engram.k_weight"), dtype=torch.float32
                )
                ewkv = self._dequant_fp8_block32_gpu(
                    loaded_cpu.pop(f"{prefix}.engram.wkv.weight"),
                    loaded_cpu.pop(f"{prefix}.engram.wkv.scale"),
                )
                eng_w = _EngramWeights(
                    layer_hash_idx=eh_idx,
                    vocab_start_idx=r * part_rows,
                    vocab_end_idx=min((r + 1) * part_rows, tot_rows),
                    embed_weight_u8=loaded_cpu.pop(f"{prefix}.engram.embed.weight"),
                    embed_scale_u8=loaded_cpu.pop(f"{prefix}.engram.embed.scale"),
                    qk_prod_f32=(qw * kw).contiguous(),
                    wkv_bf16=ewkv,
                )

            sh_w1 = self._dequant_fp8_block32_gpu(
                loaded_cpu.pop(f"{prefix}.ffn.shared_experts.w1.weight"),
                loaded_cpu.pop(f"{prefix}.ffn.shared_experts.w1.scale"),
            )
            sh_w3 = self._dequant_fp8_block32_gpu(
                loaded_cpu.pop(f"{prefix}.ffn.shared_experts.w3.weight"),
                loaded_cpu.pop(f"{prefix}.ffn.shared_experts.w3.scale"),
            )
            sh_w13 = torch.cat([sh_w1, sh_w3], dim=0).contiguous()
            sh_w2 = self._dequant_fp8_block32_gpu(
                loaded_cpu.pop(f"{prefix}.ffn.shared_experts.w2.weight"),
                loaded_cpu.pop(f"{prefix}.ffn.shared_experts.w2.scale"),
            )

            hc_attn_f32 = self._pinned_to_gpu(
                loaded_cpu.pop(f"{prefix}.hc_attn_fn"), dtype=torch.float32
            )
            hc_ffn_f32 = self._pinned_to_gpu(
                loaded_cpu.pop(f"{prefix}.hc_ffn_fn"), dtype=torch.float32
            )
            w13_p, w13_s, w2_p, w2_s = exp_buffers[prefix]

            return _LayerWeights(
                layer_id=lid,
                arena_swa_layer_idx=arena_idx,
                compress_ratio=cr,
                kv_src_idx=cur_kv_src,
                attn_norm_weight=self._pinned_to_gpu(
                    loaded_cpu.pop(f"{prefix}.attn_norm.weight"), dtype=torch.bfloat16
                ),
                ffn_norm_weight=self._pinned_to_gpu(
                    loaded_cpu.pop(f"{prefix}.ffn_norm.weight"), dtype=torch.bfloat16
                ),
                hc_attn_fn_f32=hc_attn_f32,
                hc_attn_fn_bf16=hc_attn_f32.to(torch.bfloat16).contiguous(),
                hc_attn_base_f32=self._pinned_to_gpu(
                    loaded_cpu.pop(f"{prefix}.hc_attn_base"), dtype=torch.float32
                ),
                hc_attn_scale_f32=self._pinned_to_gpu(
                    loaded_cpu.pop(f"{prefix}.hc_attn_scale"), dtype=torch.float32
                ),
                hc_ffn_fn_f32=hc_ffn_f32,
                hc_ffn_fn_bf16=hc_ffn_f32.to(torch.bfloat16).contiguous(),
                hc_ffn_base_f32=self._pinned_to_gpu(
                    loaded_cpu.pop(f"{prefix}.hc_ffn_base"), dtype=torch.float32
                ),
                hc_ffn_scale_f32=self._pinned_to_gpu(
                    loaded_cpu.pop(f"{prefix}.hc_ffn_scale"), dtype=torch.float32
                ),
                attn_sink_f32=self._pinned_to_gpu(
                    loaded_cpu.pop(f"{prefix}.attn.attn_sink"), dtype=torch.float32
                ),
                wq_a_bf16=self._dequant_fp8_block32_gpu(
                    loaded_cpu.pop(f"{prefix}.attn.wq_a.weight"),
                    loaded_cpu.pop(f"{prefix}.attn.wq_a.scale"),
                ),
                q_norm_weight=self._pinned_to_gpu(
                    loaded_cpu.pop(f"{prefix}.attn.q_norm.weight"), dtype=torch.bfloat16
                ),
                wq_b_bf16=self._dequant_fp8_block32_gpu(
                    loaded_cpu.pop(f"{prefix}.attn.wq_b.weight"),
                    loaded_cpu.pop(f"{prefix}.attn.wq_b.scale"),
                ),
                wkv_bf16=self._dequant_fp8_block32_gpu(
                    loaded_cpu.pop(f"{prefix}.attn.wkv.weight"),
                    loaded_cpu.pop(f"{prefix}.attn.wkv.scale"),
                ),
                kv_norm_weight=self._pinned_to_gpu(
                    loaded_cpu.pop(f"{prefix}.attn.kv_norm.weight"), dtype=torch.bfloat16
                ),
                wo_a_bf16=self._dequant_fp8_block32_gpu(
                    loaded_cpu.pop(f"{prefix}.attn.wo_a.weight"),
                    loaded_cpu.pop(f"{prefix}.attn.wo_a.scale"),
                ),
                wo_b_bf16=self._dequant_fp8_block32_gpu(
                    loaded_cpu.pop(f"{prefix}.attn.wo_b.weight"),
                    loaded_cpu.pop(f"{prefix}.attn.wo_b.scale"),
                ),
                compressor=comp_w,
                indexer=idx_w,
                engram=eng_w,
                gate_weight_f32=self._pinned_to_gpu(
                    loaded_cpu.pop(f"{prefix}.ffn.gate.weight"), dtype=torch.float32
                ),
                gate_bias_f32=self._pinned_to_gpu(
                    loaded_cpu.pop(f"{prefix}.ffn.gate.bias"), dtype=torch.float32
                ),
                shared_w13_bf16=sh_w13,
                shared_w2_bf16=sh_w2,
                experts_w13_packed=w13_p,
                experts_w13_scale=w13_s,
                experts_w2_packed=w2_p,
                experts_w2_scale=w2_s,
                top_k=top_k,
            )

        self.layers: list[_LayerWeights] = [
            _build_layer(f"layers.{lid}", lid, lid, self.num_experts_per_tok)
            for lid in range(self.num_layers)
        ]
        self.mtp_layers: list[_LayerWeights] = []
        if self.dspark_enabled:
            for stg in range(self.num_mtp_layers):
                self.mtp_layers.append(
                    _build_layer(
                        f"mtp.{stg}",
                        self.num_layers + stg,
                        self.num_layers + stg,
                        self.dspark_top_k,
                    )
                )
            self.dspark_main_proj_bf16 = self._dequant_fp8_block32_gpu(
                loaded_cpu.pop("mtp.0.main_proj.weight"),
                loaded_cpu.pop("mtp.0.main_proj.scale"),
            )
            self.dspark_main_norm_weight = self._pinned_to_gpu(
                loaded_cpu.pop("mtp.0.main_norm.weight"), dtype=torch.bfloat16
            )
            last_m = f"mtp.{self.num_mtp_layers - 1}"
            self.dspark_final_norm_weight = self._pinned_to_gpu(
                loaded_cpu.pop(f"{last_m}.norm.weight"), dtype=torch.bfloat16
            )
            self.dspark_markov_embed_f32 = self._pinned_to_gpu(
                loaded_cpu.pop(f"{last_m}.markov_head.embed.weight"), dtype=torch.float32
            )
            mk_head = self._pinned_to_gpu(
                loaded_cpu.pop(f"{last_m}.markov_head.head.weight"), dtype=torch.float32
            )
            self.dspark_markov_head_f32_t = mk_head.t().contiguous()
        torch.cuda.synchronize(self.device)

        loaded_cpu.clear()
        if self.rank == 0:
            print(
                f"[zstep-sail] Loaded {len(self.layers)} backbone + {len(self.mtp_layers)} DSpark "
                f"layers in {time.monotonic() - t_load:.2f}s",
                flush=True,
            )

    # -----------------------------------------------------------------------
    # Unified KV Arena + Indexer Cache + Static Staging Workspaces
    # -----------------------------------------------------------------------

    def _allocate_kv_and_workspaces(self) -> None:
        dev = self.device
        num_mla_phys_pages = self.num_kv_pages + 2  # 0=null/trash, 1..N=live, N+1=dummy
        cfg_swa_pages = int(self.runtime_config.swa_kv_pages)
        if cfg_swa_pages > 0 and cfg_swa_pages < self.num_kv_pages:
            self.num_swa_pages = cfg_swa_pages
            self._swa_pool: _SwaPagePool | None = _SwaPagePool(
                self.num_swa_pages, self.num_kv_pages + 1
            )
            num_swa_phys_pages = self.num_swa_pages + 2
        elif self.num_kv_pages > 2048:
            # Cap SWA physical pool at 2048 pages (`262,144` SWA tokens) so `kv_arena` stays ~13 GB
            self.num_swa_pages = 2048
            self._swa_pool = _SwaPagePool(self.num_swa_pages, self.num_kv_pages + 1)
            num_swa_phys_pages = self.num_swa_pages + 2
        else:
            self.num_swa_pages = self.num_kv_pages
            self._swa_pool = None
            num_swa_phys_pages = num_mla_phys_pages

        num_swa_layers = self.num_layers + len(self.mtp_layers)
        cursor = 0
        self.swa_arena_base: list[int] = []
        swa_slots_per_layer = num_swa_phys_pages * 128
        for _ in range(num_swa_layers):
            self.swa_arena_base.append(cursor)
            cursor += swa_slots_per_layer

        self.mla_arena_base: list[int] = []
        self.mla_lat_per_page: list[int] = []
        self.indexer_k_cache: list[torch.Tensor] = []
        for lid in self.kv_source_layers:
            cr = self.compress_ratios[lid]
            lat_pp = 128 // cr
            self.mla_arena_base.append(cursor)
            self.mla_lat_per_page.append(lat_pp)
            cursor += num_mla_phys_pages * lat_pp
            self.indexer_k_cache.append(
                torch.zeros(
                    (num_mla_phys_pages, lat_pp, self.index_head_dim),
                    dtype=torch.bfloat16,
                    device=dev,
                )
            )

        self.dspark_scratch_arena_base: list[int] = []
        for _ in range(len(self.mtp_layers)):
            self.dspark_scratch_arena_base.append(cursor)
            cursor += (self.max_seqs + 2) * 8

        self.kv_arena = torch.zeros((cursor, 1, self.head_dim), dtype=torch.bfloat16, device=dev)

        # Rollback-free `cr=2` Compressor state: `[3, max_seqs + 2, 8, 2, 512]` FP32
        # (`slot=0` is trash/padding; `1..max_seqs` are live sequence slots; `ring = (pos // 2) & 7`)
        num_cr2 = sum(1 for lid in self.kv_source_layers if self.compress_ratios[lid] == 2)
        self.comp_even_state = torch.zeros(
            (max(1, num_cr2), self.max_seqs + 2, 8, 2, self.head_dim),
            dtype=torch.float32,
            device=dev,
        )
        self.comp_even_state[:, :, :, 1, :].fill_(-1.0e30)

        max_t = self.max_step_tokens
        max_b = self.max_seqs

        # Host pinned <-> GPU staging buffers
        self.token_ids_cpu = torch.zeros(max_t, dtype=torch.int32, pin_memory=True)
        self.token_ids_gpu = torch.zeros(max_t, dtype=torch.int32, device=dev)
        self.positions_cpu = torch.zeros(max_t, dtype=torch.int32, pin_memory=True)
        self.positions_gpu = torch.zeros(max_t, dtype=torch.int32, device=dev)
        self.row_seq_ids_cpu = torch.zeros(max_t, dtype=torch.int64, pin_memory=True)
        self.row_seq_ids_gpu = torch.zeros(max_t, dtype=torch.int64, device=dev)
        self.row_comp_slots_cpu = torch.zeros(max_t, dtype=torch.int64, pin_memory=True)
        self.row_comp_slots_gpu = torch.zeros(max_t, dtype=torch.int64, device=dev)
        self.swa_write_slots_cpu = torch.zeros(max_t, dtype=torch.int64, pin_memory=True)
        self.swa_write_slots_gpu = torch.zeros(max_t, dtype=torch.int64, device=dev)
        self.mla_write_pages_cpu = torch.zeros(max_t, dtype=torch.int64, pin_memory=True)
        self.mla_write_pages_gpu = torch.zeros(max_t, dtype=torch.int64, device=dev)

        self.lookback_cpu = torch.full((max_b, 3), -1, dtype=torch.int32, pin_memory=True)
        self.lookback_gpu = torch.full((max_b, 3), -1, dtype=torch.int32, device=dev)
        self.query_start_loc_cpu = torch.zeros(max_b + 1, dtype=torch.int32, pin_memory=True)
        self.query_start_loc_gpu = torch.zeros(max_b + 1, dtype=torch.int32, device=dev)
        self.decode_qsl_gpu = torch.arange(max_b + 1, dtype=torch.int32, device=dev)
        width = max(1, self.num_spec_tokens) + 1
        self.verify_qsl_gpu = torch.arange(max_b + 1, dtype=torch.int32, device=dev) * width

        self.mla_bt_cpu = torch.zeros((max_b, self.bt_width), dtype=torch.int32, pin_memory=True)
        self.mla_bt_gpu = torch.zeros((max_b, self.bt_width), dtype=torch.int32, device=dev)
        self.swa_bt_cpu = torch.zeros((max_b, self.bt_width), dtype=torch.int32, pin_memory=True)
        self.swa_bt_gpu = torch.zeros((max_b, self.bt_width), dtype=torch.int32, device=dev)

        self.sample_rows_cpu = torch.zeros(max_t, dtype=torch.int64, pin_memory=True)
        self.sample_rows_gpu = torch.zeros(max_t, dtype=torch.int64, device=dev)
        self.slots_cpu = torch.zeros(max_b, dtype=torch.int64, pin_memory=True)
        self.slots_gpu = torch.zeros(max_b, dtype=torch.int64, device=dev)
        self._verify_real_rows_cpu = torch.zeros(max_t, dtype=torch.int64, pin_memory=True)
        self._verify_real_rows_gpu = torch.zeros(max_t, dtype=torch.int64, device=dev)

        # Device-resident decode token & chronological 3-token lookback state (`STEP_FLAG_DEVICE_TOKENS`)
        self.slot_last_tok_gpu = torch.zeros(max_b, dtype=torch.int32, device=dev)
        self.slot_lb_gpu = torch.full((max_b, 3), -1, dtype=torch.int32, device=dev)
        self.next_lb_gpu = torch.full((max_b, 3), -1, dtype=torch.int32, device=dev)

        # Double-buffered Top-2 output staging
        self.local_top2_vals_buf = torch.zeros((max_t, 2), dtype=torch.float32, device=dev)
        self.local_top2_ids_buf = torch.zeros((max_t, 2), dtype=torch.int32, device=dev)
        self.out_tokens_gpu = torch.zeros((max_t, 2), dtype=torch.int32, device=dev)
        self.out_logprobs_gpu = torch.zeros((max_t, 2), dtype=torch.int32, device=dev)
        self.out_tokens_cpu = torch.zeros((2, max_t, 2), dtype=torch.int32, pin_memory=True)
        self.out_logprobs_cpu = torch.zeros((2, max_t, 2), dtype=torch.int32, pin_memory=True)
        self._out_tokens_np = self.out_tokens_cpu.numpy()
        self._out_logprobs_np = self.out_logprobs_cpu.numpy()

        # Stochastic sampling staging buffers
        self.samp_temps_cpu = torch.zeros(max_b, dtype=torch.float32, pin_memory=True)
        self.samp_temps_gpu = torch.zeros(max_b, dtype=torch.float32, device=dev)
        self.samp_top_p_cpu = torch.ones(max_b, dtype=torch.float32, pin_memory=True)
        self.samp_top_p_gpu = torch.ones(max_b, dtype=torch.float32, device=dev)
        self.samp_top_k_cpu = torch.full((max_b,), -1, dtype=torch.int32, pin_memory=True)
        self.samp_top_k_gpu = torch.full((max_b,), -1, dtype=torch.int32, device=dev)
        self.samp_seeds_cpu = torch.zeros(max_b, dtype=torch.int64, pin_memory=True)
        self.samp_seeds_gpu = torch.zeros(max_b, dtype=torch.int64, device=dev)
        self.samp_pres_cpu = torch.zeros(max_b, dtype=torch.float32, pin_memory=True)
        self.samp_pres_gpu = torch.zeros(max_b, dtype=torch.float32, device=dev)
        self.samp_freq_cpu = torch.zeros(max_b, dtype=torch.float32, pin_memory=True)
        self.samp_freq_gpu = torch.zeros(max_b, dtype=torch.float32, device=dev)
        self.samp_reset_cpu = torch.zeros(max_b, dtype=torch.uint8, pin_memory=True)
        self.samp_reset_gpu = torch.zeros(max_b, dtype=torch.uint8, device=dev)
        self.samp_slots_cpu = torch.zeros(max_b, dtype=torch.int64, pin_memory=True)
        self.samp_slots_gpu = torch.zeros(max_b, dtype=torch.int64, device=dev)
        self.full_logits_ag_buf = torch.zeros(
            (max_b, self.world_size * self.vocab_per_rank), dtype=torch.bfloat16, device=dev
        )

        # Forward execution workspaces (zero-initialized to prevent recycled-HBM NaN hazards)
        self.h_stream_buf = [
            torch.zeros((max_t, 4, self.hidden_size), dtype=torch.bfloat16, device=dev),
            torch.zeros((max_t, 4, self.hidden_size), dtype=torch.bfloat16, device=dev),
        ]
        self.pre_mix_buf = [
            torch.zeros((max_t, 4), dtype=torch.float32, device=dev),
            torch.zeros((max_t, 4), dtype=torch.float32, device=dev),
        ]
        self.tmp_pre_buf = torch.zeros((max_t, 4), dtype=torch.float32, device=dev)
        self.tmp_post_buf = torch.zeros((max_t, 4), dtype=torch.float32, device=dev)
        self.tmp_comb_buf = torch.zeros((max_t, 4, 4), dtype=torch.float32, device=dev)
        self.mhc_proj_buf = torch.zeros((max_t, 24), dtype=torch.float32, device=dev)
        self.mhc_split_k_part_buf = torch.zeros(
            (max_t * 16 * 25,), dtype=torch.float32, device=dev
        )
        self.normed_buf = torch.zeros((max_t, self.hidden_size), dtype=torch.bfloat16, device=dev)
        self.act_quant_buf = torch.zeros(
            (max_t, self.hidden_size), dtype=torch.bfloat16, device=dev
        )
        self.qr_buf = torch.zeros((max_t, self.q_lora_rank), dtype=torch.bfloat16, device=dev)
        self.qr_quant_buf = torch.zeros(
            (max_t, self.q_lora_rank), dtype=torch.bfloat16, device=dev
        )
        self.q_buf = torch.zeros((max_t, 8, self.head_dim), dtype=torch.bfloat16, device=dev)
        self.kv_buf = torch.zeros((max_t, self.head_dim), dtype=torch.bfloat16, device=dev)
        self.attn_out_buf = torch.zeros(
            (max_t, 8, self.head_dim), dtype=torch.bfloat16, device=dev
        )
        self.attn_partial_acc = torch.zeros(
            (max_t * 64, self.head_dim), dtype=torch.float32, device=dev
        )
        self.attn_partial_m = torch.zeros((max_t * 64,), dtype=torch.float32, device=dev)
        self.attn_partial_l = torch.zeros((max_t * 64,), dtype=torch.float32, device=dev)
        self.attn_q_pad64_buf = torch.zeros(
            (max_t, 64, self.head_dim), dtype=torch.bfloat16, device=dev
        )
        self.o_lora_buf = torch.zeros((max_t, self.o_lora_rank), dtype=torch.bfloat16, device=dev)
        self.sub_local_buf = torch.zeros(
            (max_t, self.hidden_size), dtype=torch.bfloat16, device=dev
        )
        self.sub_red_buf = torch.zeros(
            (max_t, self.hidden_size), dtype=torch.bfloat16, device=dev
        )

        # Engram workspaces
        self.engram_local_buf = torch.zeros((max_t, 6144), dtype=torch.bfloat16, device=dev)
        self.engram_red_buf = torch.zeros((max_t, 6144), dtype=torch.bfloat16, device=dev)
        self.engram_kv_buf = torch.zeros((max_t, 25600), dtype=torch.bfloat16, device=dev)

        # MoE workspaces
        max_pairs = max_t * self.num_experts_per_tok
        max_tiles = max_pairs // 16 + self.n_routed_experts + 16
        self.router_logits_buf = torch.zeros(
            (max_t, self.n_routed_experts), dtype=torch.float32, device=dev
        )
        self.topk_weights_buf = torch.zeros(
            (max_t, self.num_experts_per_tok), dtype=torch.float32, device=dev
        )
        self.topk_ids_buf = torch.zeros(
            (max_t, self.num_experts_per_tok), dtype=torch.int32, device=dev
        )
        self.shared_gu_buf = torch.zeros(
            (max_t, 2 * self.inter_per_rank), dtype=torch.bfloat16, device=dev
        )
        self.shared_out_buf = torch.zeros(
            (max_t, self.hidden_size), dtype=torch.bfloat16, device=dev
        )
        self.moe_inter_buf = torch.zeros(
            (max_pairs, self.inter_per_rank), dtype=torch.bfloat16, device=dev
        )
        self.sorted_pair_ids_buf = torch.full(
            (max_tiles * 64,), -1, dtype=torch.int32, device=dev
        )
        self.expert_ids_per_tile_buf = torch.zeros((max_tiles,), dtype=torch.int32, device=dev)
        self.num_valid_tiles_buf = torch.zeros((1,), dtype=torch.int32, device=dev)
        self.pair_out_buf = torch.zeros(
            (max_pairs, self.hidden_size), dtype=torch.bfloat16, device=dev
        )

        # Sparse attention & Indexer workspaces
        self.swa_only_topk_buf = torch.full((max_t, 128), -1, dtype=torch.int32, device=dev)
        self.shared_full_topk_buf = torch.full((max_t, 640), -1, dtype=torch.int32, device=dev)
        self.idx_q_buf = torch.zeros(
            (max_t, self.index_local_heads, self.index_head_dim), dtype=torch.bfloat16, device=dev
        )
        self.idx_weights_buf = torch.zeros(
            (max_t, self.index_local_heads), dtype=torch.bfloat16, device=dev
        )
        self.idx_scores_local_buf = torch.zeros(
            (max_t, self.decode_graph_max_seq_len), dtype=torch.float32, device=dev
        )
        self.idx_scores_red_buf = torch.zeros(
            (max_t, self.decode_graph_max_seq_len), dtype=torch.float32, device=dev
        )

        # Final head & DSpark auxiliary hidden workspaces
        self.decode_hidden_gpu = torch.zeros(
            (max_t, self.hidden_size), dtype=torch.bfloat16, device=dev
        )
        self.logits_f32_buf = torch.zeros(
            (max_t, self.vocab_per_rank), dtype=torch.float32, device=dev
        )
        self.logits_bf16_buf = torch.zeros(
            (max_t, self.vocab_per_rank), dtype=torch.bfloat16, device=dev
        )
        if self.dspark_enabled:
            self.dspark_aux_cat_buf = torch.zeros(
                (max_t, 3 * self.hidden_size), dtype=torch.bfloat16, device=dev
            )
            self.dspark_main_x_buf = torch.zeros(
                (max_t, self.hidden_size), dtype=torch.bfloat16, device=dev
            )

        self._w_offs_128 = torch.arange(128, dtype=torch.int32, device=dev)
        self._comp_arange_gpu = torch.arange(self.max_model_len + 16, dtype=torch.int32, device=dev)
        self.arange_rows_i64 = torch.arange(max_t, dtype=torch.int64, device=dev)
        self.arange_seqs_i64 = torch.arange(max_b + 2, dtype=torch.int64, device=dev)

    # -----------------------------------------------------------------------
    # Watched Triton Kernel Pre-Warmup (`TRITON_RECOMPILATIONS == 0`)
    # -----------------------------------------------------------------------

    def watched_triton_kernels(self) -> list[tuple[str, object]]:
        return sail_ops.watched_sail_triton_kernels()

    def _warmup_watched_triton_kernels(self) -> None:
        layer0 = self.layers[0]
        for warm_rows in (24, 64):
            if warm_rows > self.max_step_tokens:
                continue
            x_w = self.normed_buf[:warm_rows].zero_()
            sail_ops.dsv41_moe_tp8_forward(
                x_w,
                gate_weight=layer0.gate_weight_f32,
                gate_bias=layer0.gate_bias_f32,
                shared_w13_bf16=layer0.shared_w13_bf16,
                shared_w2_bf16=layer0.shared_w2_bf16,
                experts_w13_packed=layer0.experts_w13_packed,
                experts_w13_scale=layer0.experts_w13_scale,
                experts_w2_packed=layer0.experts_w2_packed,
                experts_w2_scale=layer0.experts_w2_scale,
                top_k=layer0.top_k,
                route_scale=self.routed_scaling_factor,
                swiglu_limit=self.swiglu_limit,
                router_logits_buf=self.router_logits_buf,
                topk_weights_buf=self.topk_weights_buf,
                topk_ids_buf=self.topk_ids_buf,
                shared_gu_buf=self.shared_gu_buf,
                shared_out_buf=self.shared_out_buf,
                moe_inter_buf=self.moe_inter_buf,
                sorted_pair_ids_buf=self.sorted_pair_ids_buf,
                expert_ids_per_tile_buf=self.expert_ids_per_tile_buf,
                num_valid_tiles_buf=self.num_valid_tiles_buf,
                pair_out_buf=self.pair_out_buf,
                moe_out_buf=self.sub_local_buf,
                act_quant_x_buf=self.act_quant_buf,
                gemv_max_rows=0,
                gemv_threads=self.moe_gemv_threads,
                block_m=self.moe_block_m,
                num_warps=self.moe_num_warps,
            )
            if self.dspark_enabled and self.mtp_layers:
                ml0 = self.mtp_layers[0]
                sail_ops.dsv41_moe_tp8_forward(
                    x_w,
                    gate_weight=ml0.gate_weight_f32,
                    gate_bias=ml0.gate_bias_f32,
                    shared_w13_bf16=ml0.shared_w13_bf16,
                    shared_w2_bf16=ml0.shared_w2_bf16,
                    experts_w13_packed=ml0.experts_w13_packed,
                    experts_w13_scale=ml0.experts_w13_scale,
                    experts_w2_packed=ml0.experts_w2_packed,
                    experts_w2_scale=ml0.experts_w2_scale,
                    top_k=ml0.top_k,
                    route_scale=self.routed_scaling_factor,
                    swiglu_limit=self.swiglu_limit,
                    router_logits_buf=self.router_logits_buf,
                    topk_weights_buf=self.topk_weights_buf,
                    topk_ids_buf=self.topk_ids_buf,
                    shared_gu_buf=self.shared_gu_buf,
                    shared_out_buf=self.shared_out_buf,
                    moe_inter_buf=self.moe_inter_buf,
                    sorted_pair_ids_buf=self.sorted_pair_ids_buf,
                    expert_ids_per_tile_buf=self.expert_ids_per_tile_buf,
                    num_valid_tiles_buf=self.num_valid_tiles_buf,
                    pair_out_buf=self.pair_out_buf,
                    moe_out_buf=self.sub_local_buf,
                    act_quant_x_buf=self.act_quant_buf,
                    gemv_max_rows=0,
                    gemv_threads=self.moe_gemv_threads,
                    block_m=self.moe_block_m,
                    num_warps=self.moe_num_warps,
                )

        # Warm up `_triton_sparse_attn_tp8_kernel` across topk widths (128, 256, 640)
        for tk_w in (128, 256, 640):
            tk_buf = torch.full((4, tk_w), -1, dtype=torch.int32, device=self.device)
            sail_ops.sparse_attn_tp8_forward(
                self.q_buf[:4].zero_(),
                self.kv_arena,
                layer0.attn_sink_f32,
                tk_buf,
                out=self.attn_out_buf[:4],
                partial_acc=self.attn_partial_acc[:256],
                partial_m=self.attn_partial_m[:256],
                partial_l=self.attn_partial_l[:256],
                q_pad64_buf=self.attn_q_pad64_buf[:4],
                sm_scale=self.sm_scale,
                backend="triton_tp8",
            )
        torch.cuda.synchronize(self.device)

    # -----------------------------------------------------------------------
    # Core 40-Layer Backbone Forward (Shared by Varlen, Decode & Verify Graphs)
    # -----------------------------------------------------------------------

    def _run_engram_layer(
        self,
        h_stream: torch.Tensor,
        eng: _EngramWeights,
        *,
        rows: int,
        num_seqs: int,
        qsl_gpu: torch.Tensor,
    ) -> None:
        ev_e = self.observer.begin("engram_compressor_indexer")
        ext = get_sail_ops_extension()
        stream_handle = torch.cuda.current_stream(self.device).cuda_stream
        loc_emb = self.engram_local_buf[:rows]
        ext.launch_engram_lookup_dequant(
            self.token_ids_gpu[:rows],
            self.lookback_gpu[:num_seqs],
            qsl_gpu[: num_seqs + 1],
            self.engram_token_map_gpu,
            self.engram_mult_gpu[eng.layer_hash_idx],
            self.engram_primes_gpu[eng.layer_hash_idx],
            self.engram_offsets_gpu[eng.layer_hash_idx],
            eng.embed_weight_u8,
            eng.embed_scale_u8,
            loc_emb,
            num_seqs,
            eng.vocab_start_idx,
            eng.vocab_end_idx,
            self.engram_pad_comp_id,
            stream_handle,
        )
        self.observer.end(ev_e)

        ev_ar = self.observer.begin("icn_allreduce")
        red_emb = self.engram_red_buf[:rows]
        self.icn_comm.all_reduce_bf16(loc_emb, output=red_emb)
        self.observer.end(ev_ar)

        ev_e2 = self.observer.begin("engram_compressor_indexer")
        sail_ops.inplace_act_quant_ue8m0(red_emb)
        kv_proj = self.engram_kv_buf[:rows]
        torch.matmul(red_emb, eng.wkv_bf16.t(), out=kv_proj)
        key_f32 = kv_proj[:, : 4 * self.hidden_size].float().view(rows, 4, self.hidden_size)
        val_f32 = kv_proj[:, 4 * self.hidden_size :].float().unsqueeze(1)
        h_f32 = h_stream.float()
        rstd = torch.rsqrt(h_f32.square().mean(-1) + self.norm_eps) * torch.rsqrt(
            key_f32.square().mean(-1) + self.norm_eps
        )
        dot = (
            (h_f32 * eng.qk_prod_f32.unsqueeze(0) * key_f32).sum(-1)
            * rstd
            * (self.hidden_size**-0.5)
        )
        gate = torch.sigmoid(torch.copysign(dot.abs().clamp_min(1.0e-6).sqrt(), dot))
        h_stream.copy_((h_f32 + gate.unsqueeze(-1) * val_f32).to(torch.bfloat16))
        self.observer.end(ev_e2)

    def _run_compressor_and_indexer(
        self,
        lw: _LayerWeights,
        x_norm: torch.Tensor,
        qr_unquant: torch.Tensor,
        *,
        rows: int,
        num_seqs: int,
        max_seq_len_bound: int,
        uniform_width: int,
        qsl_np: np.ndarray | None,
        seq_lens_np: np.ndarray | None,
        candidates_mask: torch.Tensor | None,
    ) -> torch.Tensor | None:
        """Run optional `Compressor` (`kv_source`) + `Indexer` (`index_source`), updating
        `self.shared_full_topk_buf[:rows, 128:640]` and returning updated `candidates_mask`."""
        cr = lw.compress_ratio
        kv_src = lw.kv_src_idx
        lat_pp = self.mla_lat_per_page[kv_src]
        pos_i32 = self.positions_gpu[:rows]
        mla_pages = self.mla_write_pages_gpu[:rows]
        valid_write = mla_pages > 0

        # 1. Compressor (`layers 2, 8, 14` have `cr=2`; `layer 20` has `cr=1`)
        comp = lw.compressor
        latent: torch.Tensor | None = None
        group_pos_i32: torch.Tensor | None = None
        mla_in_page_slots: torch.Tensor | None = None

        if comp is not None:
            if cr == 1:
                lat_buf = self.kv_buf[:rows]
                torch.matmul(x_norm, comp.wkv_weight.t(), out=lat_buf)
                sail_ops.rmsnorm_bf16(lat_buf, comp.norm_weight, eps=self.norm_eps, out=lat_buf)
                latent = lat_buf
                group_pos_i32 = pos_i32
                mla_in_page_slots = torch.where(
                    valid_write,
                    mla_pages * 128 + (pos_i32 % 128).to(torch.int64),
                    0,
                )
            else:
                assert comp.wgate_weight is not None
                x_f32 = x_norm.float()
                c_kv = torch.matmul(x_f32, comp.wkv_weight.t())
                c_sc = torch.matmul(x_f32, comp.wgate_weight.t())
                c_slots = self.row_comp_slots_gpu[:rows]
                ring_idx = ((pos_i32 // 2) & 7).to(torch.int64)
                is_even = ((pos_i32 & 1) == 0) & valid_write
                is_odd = ((pos_i32 & 1) == 1) & valid_write
                state = self.comp_even_state[comp.comp_idx]

                if uniform_width == 1:
                    # Pure 1-token-per-seq decode: even writes state, odd pools with saved even!
                    even_cs = torch.where(is_even, c_slots, 0)
                    state[even_cs, ring_idx, 0] = c_kv
                    state[even_cs, ring_idx, 1] = c_sc

                    odd_cs = torch.where(is_odd, c_slots, 0)
                    prev_kv = state[odd_cs, ring_idx, 0]
                    prev_sc = state[odd_cs, ring_idx, 1]
                    m_max = torch.maximum(prev_sc, c_sc)
                    e0 = torch.exp(prev_sc - m_max)
                    e1 = torch.exp(c_sc - m_max)
                    denom = (e0 + e1).clamp_min(1.0e-30)
                    pooled = ((prev_kv * e0 + c_kv * e1) / denom).to(torch.bfloat16)
                    lat_buf = self.kv_buf[:rows]
                    sail_ops.rmsnorm_bf16(pooled, comp.norm_weight, eps=self.norm_eps, out=lat_buf)
                    latent = lat_buf
                    group_pos_i32 = (pos_i32 - 1).clamp(min=0)
                    mla_in_page_slots = torch.where(
                        is_odd,
                        mla_pages * lat_pp + ((group_pos_i32 % 128) // 2).to(torch.int64),
                        0,
                    )
                elif uniform_width > 1:
                    # Uniform verify graph (`rows = num_seqs * width`, `width <= 8` so all even fit in 8-slot ring!)
                    c_kv_3d = c_kv.view(num_seqs, uniform_width, self.head_dim)
                    c_sc_3d = c_sc.view(num_seqs, uniform_width, self.head_dim)
                    pos_2d = pos_i32.view(num_seqs, uniform_width)
                    cs_2d = c_slots.view(num_seqs, uniform_width)
                    vw_2d = valid_write.view(num_seqs, uniform_width)
                    for col in range(uniform_width):
                        p_col = pos_2d[:, col]
                        ev_col = ((p_col & 1) == 0) & vw_2d[:, col]
                        dst_cs = torch.where(ev_col, cs_2d[:, col], 0)
                        r_col = ((p_col // 2) & 7).to(torch.int64)
                        state[dst_cs, r_col, 0] = c_kv_3d[:, col]
                        state[dst_cs, r_col, 1] = c_sc_3d[:, col]

                    odd_cs = torch.where(is_odd, c_slots, 0)
                    prev_kv = state[odd_cs, ring_idx, 0]
                    prev_sc = state[odd_cs, ring_idx, 1]
                    m_max = torch.maximum(prev_sc, c_sc)
                    e0 = torch.exp(prev_sc - m_max)
                    e1 = torch.exp(c_sc - m_max)
                    denom = (e0 + e1).clamp_min(1.0e-30)
                    pooled = ((prev_kv * e0 + c_kv * e1) / denom).to(torch.bfloat16)
                    lat_buf = self.kv_buf[:rows]
                    sail_ops.rmsnorm_bf16(pooled, comp.norm_weight, eps=self.norm_eps, out=lat_buf)
                    latent = lat_buf
                    group_pos_i32 = (pos_i32 - 1).clamp(min=0)
                    mla_in_page_slots = torch.where(
                        is_odd,
                        mla_pages * lat_pp + ((group_pos_i32 % 128) // 2).to(torch.int64),
                        0,
                    )
                else:
                    # Varlen prefill / mixed step: pair `r` with `r - 1` when `r > seq_begin`,
                    # and read/write `comp_even_state` at sequence boundaries!
                    assert qsl_np is not None
                    prev_rows = (self.arange_rows_i64[:rows] - 1).clamp(min=0)
                    prev_kv = c_kv.index_select(0, prev_rows)
                    prev_sc = c_sc.index_select(0, prev_rows)
                    for s_i in range(num_seqs):
                        s_beg = int(qsl_np[s_i])
                        s_end = int(qsl_np[s_i + 1])
                        if s_end <= s_beg:
                            continue
                        # If `s_beg` is odd, its even partner came from the previous chunk/step
                        prev_kv[s_beg] = state[c_slots[s_beg], ring_idx[s_beg], 0]
                        prev_sc[s_beg] = state[c_slots[s_beg], ring_idx[s_beg], 1]
                        # Save trailing `<= 16` tokens' even positions into the 8-slot ring
                        tail_lo = max(s_beg, s_end - 16)
                        for tr in range(tail_lo, s_end):
                            p_tr = pos_i32[tr]
                            ev_tr = ((p_tr & 1) == 0) & valid_write[tr]
                            d_cs = torch.where(ev_tr, c_slots[tr], 0)
                            state[d_cs, ring_idx[tr], 0] = c_kv[tr]
                            state[d_cs, ring_idx[tr], 1] = c_sc[tr]

                    m_max = torch.maximum(prev_sc, c_sc)
                    e0 = torch.exp(prev_sc - m_max)
                    e1 = torch.exp(c_sc - m_max)
                    denom = (e0 + e1).clamp_min(1.0e-30)
                    pooled = ((prev_kv * e0 + c_kv * e1) / denom).to(torch.bfloat16)
                    lat_buf = self.kv_buf[:rows]
                    sail_ops.rmsnorm_bf16(pooled, comp.norm_weight, eps=self.norm_eps, out=lat_buf)
                    latent = lat_buf
                    group_pos_i32 = (pos_i32 - 1).clamp(min=0)
                    mla_in_page_slots = torch.where(
                        is_odd,
                        mla_pages * lat_pp + ((group_pos_i32 % 128) // 2).to(torch.int64),
                        0,
                    )

        # 2. Indexer key store (`owns_k` MUST run BEFORE `latent` is rotated/quantized for MLA!)
        idx = lw.indexer
        if idx is not None and idx.owns_k and latent is not None:
            assert idx.wk_bf16 is not None and idx.k_norm_weight is not None
            assert group_pos_i32 is not None and mla_in_page_slots is not None
            k_lat = torch.matmul(latent, idx.wk_bf16.t())
            sail_ops.rmsnorm_bf16(k_lat, idx.k_norm_weight, eps=self.norm_eps, out=k_lat)
            sail_ops.apply_rope_inplace(
                k_lat.unsqueeze(1),
                self.comp_cos_sin_gpu,
                group_pos_i32,
                rope_dim=64,
                inverse=False,
            )
            sail_ops.inplace_fp4_quant(k_lat, block_size=32, is_e4m3_scale=False)
            self.indexer_k_cache[kv_src].view(-1, self.index_head_dim).index_copy_(
                0, mla_in_page_slots, k_lat
            )

        # 3. Rotate + FP4(block=16, E4M3) quantize `latent` and store into `kv_arena`!
        if comp is not None and latent is not None:
            assert group_pos_i32 is not None and mla_in_page_slots is not None
            sail_ops.apply_rope_inplace(
                latent.unsqueeze(1),
                self.comp_cos_sin_gpu,
                group_pos_i32,
                rope_dim=64,
                inverse=False,
            )
            sail_ops.inplace_fp4_quant(latent, block_size=16, is_e4m3_scale=True)
            arena_mla_slots = self.mla_arena_base[kv_src] + mla_in_page_slots
            self.kv_arena.view(-1, self.head_dim).index_copy_(0, arena_mla_slots, latent)

        # 4. Indexer Top-K scoring (`layers 2, 8, 14, 20, 24, 28, 32, 36`)
        if idx is None:
            return candidates_mask

        max_c = max_seq_len_bound // cr
        if max_c <= 0:
            self.shared_full_topk_buf[:rows, 128:] = -1
            return None if idx.is_candidate_source else candidates_mask

        qr_q = self.qr_quant_buf[:rows]
        qr_q.copy_(qr_unquant)
        sail_ops.inplace_act_quant_ue8m0(qr_q)
        q_idx_flat = self.idx_q_buf[:rows].view(rows, -1)
        torch.matmul(qr_q, idx.wq_b_bf16.t(), out=q_idx_flat)
        q_idx = q_idx_flat.view(rows, self.index_local_heads, self.index_head_dim)
        sail_ops.apply_rope_inplace(
            q_idx, self.comp_cos_sin_gpu, pos_i32, rope_dim=64, inverse=False
        )
        sail_ops.inplace_fp4_quant(q_idx, block_size=32, is_e4m3_scale=False)

        w_idx = self.idx_weights_buf[:rows]
        torch.matmul(x_norm, idx.weights_proj_bf16.t(), out=w_idx)
        w_scaled_f32 = w_idx.float() * self.index_weight_scale

        pages_needed = min(self.bt_width, (max_c + lat_pp - 1) // lat_pp)
        eff_max_c = min(max_c, pages_needed * lat_pp)
        k_pages = self.indexer_k_cache[kv_src].index_select(
            0, self.mla_bt_gpu[:num_seqs, :pages_needed].reshape(-1).to(torch.int64)
        )
        k_seq = k_pages.view(num_seqs, pages_needed * lat_pp, self.index_head_dim)[
            :, :eff_max_c, :
        ]

        max_score_elems = self.idx_scores_local_buf.numel()
        if rows * eff_max_c <= max_score_elems:
            scores_local = self.idx_scores_local_buf.view(-1)[: rows * eff_max_c].view(
                rows, eff_max_c
            )
            scores_red = self.idx_scores_red_buf.view(-1)[: rows * eff_max_c].view(
                rows, eff_max_c
            )
        else:
            scores_local = torch.empty((rows, eff_max_c), dtype=torch.float32, device=self.device)
            scores_red = torch.empty((rows, eff_max_c), dtype=torch.float32, device=self.device)

        if uniform_width >= 1:
            q_bmm = q_idx.view(
                num_seqs, uniform_width * self.index_local_heads, self.index_head_dim
            )
            raw_sc = torch.bmm(q_bmm, k_seq.transpose(1, 2)).view(
                rows, self.index_local_heads, eff_max_c
            )
            torch.sum(
                raw_sc.float().relu_() * w_scaled_f32.unsqueeze(-1),
                dim=1,
                out=scores_local,
            )
        else:
            assert qsl_np is not None and seq_lens_np is not None
            scores_local.zero_()
            for s_i in range(num_seqs):
                s_beg = int(qsl_np[s_i])
                s_end = int(qsl_np[s_i + 1])
                if s_end <= s_beg:
                    continue
                s_c = min(eff_max_c, int(seq_lens_np[s_i]) // cr)
                if s_c <= 0:
                    continue
                q_sub = q_idx[s_beg:s_end].reshape(-1, self.index_head_dim)
                raw_sub = torch.matmul(q_sub, k_seq[s_i, :s_c].t()).view(
                    s_end - s_beg, self.index_local_heads, s_c
                )
                scores_local[s_beg:s_end, :s_c] = (
                    raw_sub.float().relu_() * w_scaled_f32[s_beg:s_end].unsqueeze(-1)
                ).sum(dim=1)

        ev_ar = self.observer.begin("icn_allreduce")
        self.icn_comm.all_reduce_f32(scores_local, output=scores_red)
        self.observer.end(ev_ar)

        comp_lens = (pos_i32 + 1) // cr
        c_offs = self._comp_arange_gpu[:eff_max_c]
        reachable = c_offs[None, :] < comp_lens[:, None]
        scores_masked = torch.where(reachable, scores_red, -1.0e30)

        if idx.is_candidate_source:
            num_blk = (eff_max_c + self.candidate_block_size - 1) // self.candidate_block_size
            if num_blk <= self.candidate_topk_blocks:
                candidates_mask = reachable
            else:
                pad_c = (-eff_max_c) % self.candidate_block_size
                sc_pad = (
                    F.pad(scores_masked, (0, pad_c), value=-1.0e30)
                    if pad_c > 0
                    else scores_masked
                )
                blk_sc = sc_pad.unflatten(-1, (-1, self.candidate_block_size)).amax(dim=-1)
                last_blk = ((comp_lens - 1).clamp(min=0)) // self.candidate_block_size
                b_idx = self._comp_arange_gpu[: blk_sc.shape[1]]
                blk_sc = torch.where(b_idx[None, :] == last_blk[:, None], 1.0e30, blk_sc)
                top_b = blk_sc.topk(self.candidate_topk_blocks, dim=-1)
                keep_b = torch.zeros_like(blk_sc, dtype=torch.bool).scatter_(
                    -1, top_b.indices, top_b.values > -1.0e29
                )
                candidates_mask = keep_b.repeat_interleave(self.candidate_block_size, dim=-1)[
                    :, :eff_max_c
                ]
        elif idx.uses_candidates and candidates_mask is not None:
            scores_masked = torch.where(
                candidates_mask[:, :eff_max_c], scores_masked, -1.0e20
            )

        k_pick = min(self.index_topk, eff_max_c)
        top_vals, top_c = scores_masked.topk(k_pick, dim=-1, sorted=False)
        top_c_sorted, perm = top_c.sort(dim=-1)
        top_vals_sorted = top_vals.gather(1, perm)
        valid_pick = (top_c_sorted < comp_lens[:, None].to(torch.int64)) & (
            top_vals_sorted > -1.0e19
        )

        seq_ids_2d = self.row_seq_ids_gpu[:rows, None]
        phys_mla_p = self.mla_bt_gpu[seq_ids_2d, top_c_sorted // lat_pp]
        mla_slots = (
            self.mla_arena_base[kv_src]
            + phys_mla_p * lat_pp
            + (top_c_sorted % lat_pp).to(torch.int32)
        )
        picked_slots = torch.where(valid_pick & (phys_mla_p > 0), mla_slots, -1)
        self.shared_full_topk_buf[:rows, 128 : 128 + k_pick] = picked_slots
        if k_pick < 512:
            self.shared_full_topk_buf[:rows, 128 + k_pick : 640] = -1
        return candidates_mask

    def _forward_backbone(
        self,
        *,
        rows: int,
        num_seqs: int,
        qsl_gpu: torch.Tensor,
        max_seq_len_bound: int,
        uniform_width: int,
        qsl_np: np.ndarray | None = None,
        seq_lens_np: np.ndarray | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor | None]:
        """Run all 40 DeepSeek-V4.1-Flash backbone layers and return `(normed_hidden, main_x)`."""
        toks_i64 = self.token_ids_gpu[:rows].to(torch.int64).clamp(
            min=0, max=self.vocab_size - 1
        )
        emb = F.embedding(toks_i64, self.embed_weight)
        self.h_stream_buf[0][:rows].copy_(emb.unsqueeze(1).expand(rows, 4, self.hidden_size))
        self.pre_mix_buf[0][:rows].zero_()
        self.pre_mix_buf[0][:rows, 0] = 1.0

        # Precompute `[rows, 128]` base SWA window slots (`pos - 127 .. pos`) ONCE for all 40 layers!
        pos_i32 = self.positions_gpu[:rows]
        win_p = pos_i32[:, None] - 127 + self._w_offs_128[None, :]
        valid_win = win_p >= 0
        safe_wp = win_p.clamp(min=0)
        seq_ids_2d = self.row_seq_ids_gpu[:rows, None]
        phys_swa_p = self.swa_bt_gpu[seq_ids_2d, (safe_wp // 128).to(torch.int64)]
        valid_swa = valid_win & (phys_swa_p > 0)
        base_swa_win_slots = phys_swa_p * 128 + (safe_wp % 128)
        swa_write_base = self.swa_write_slots_gpu[:rows]

        candidates_mask: torch.Tensor | None = None
        dspark_slot_idx = 0

        for lid, lw in enumerate(self.layers):
            h_cur = self.h_stream_buf[0][:rows]
            h_mid = self.h_stream_buf[1][:rows]
            pm_cur = self.pre_mix_buf[0][:rows]
            pm_nxt = self.pre_mix_buf[1][:rows]

            if lw.engram is not None:
                self._run_engram_layer(
                    h_cur, lw.engram, rows=rows, num_seqs=num_seqs, qsl_gpu=qsl_gpu
                )

            if self.dspark_enabled and lid in self.dspark_target_layers:
                s_lo = dspark_slot_idx * self.hidden_size
                s_hi = s_lo + self.hidden_size
                self.dspark_aux_cat_buf[:rows, s_lo:s_hi].copy_(
                    h_cur.float().mean(dim=1).to(torch.bfloat16)
                )
                dspark_slot_idx += 1

            # --- Attention Sub-Block ---
            ev_m = self.observer.begin("mhc_mixes")
            attn_pre = self.tmp_pre_buf[:rows]
            attn_post = self.tmp_post_buf[:rows]
            attn_comb = self.tmp_comb_buf[:rows]
            sail_ops.mhc_mixes_sinkhorn(
                h_cur,
                lw.hc_attn_fn_f32,
                lw.hc_attn_fn_bf16,
                lw.hc_attn_scale_f32,
                lw.hc_attn_base_f32,
                pre_out=attn_pre,
                post_out=attn_post,
                comb_out=attn_comb,
                raw_proj_buf=self.mhc_proj_buf[:rows],
                split_k_part_buf=self.mhc_split_k_part_buf,
                split_k=self.mhc_split_k,
                fused_max_rows=self.mhc_fused_max_rows,
                norm_eps=self.norm_eps,
                hc_eps=self.hc_eps,
                sinkhorn_iters=self.hc_sinkhorn_iters,
            )
            x_norm = self.normed_buf[:rows]
            sail_ops.mhc_pre_rmsnorm(
                h_cur,
                pm_cur,
                lw.attn_norm_weight,
                normed_out=x_norm,
                norm_eps=self.norm_eps,
            )
            self.observer.end(ev_m)

            ev_qkv = self.observer.begin("attn_qkv_o_proj")
            cos_sin = self.comp_cos_sin_gpu if lw.compress_ratio > 0 else self.swa_cos_sin_gpu
            q_in = sail_ops.act_quant_ue8m0(x_norm, out=self.act_quant_buf[:rows])

            qr = self.qr_buf[:rows]
            torch.matmul(q_in, lw.wq_a_bf16.t(), out=qr)
            sail_ops.rmsnorm_bf16(qr, lw.q_norm_weight, eps=self.norm_eps, out=qr)

            # Write SWA KV into `kv_arena` BEFORE `qr` is quantized in-place (`Indexer` uses unquantized `qr`!)
            wkv_out = self.kv_buf[:rows]
            torch.matmul(q_in, lw.wkv_bf16.t(), out=wkv_out)
            sail_ops.rmsnorm_bf16(wkv_out, lw.kv_norm_weight, eps=self.norm_eps, out=wkv_out)
            sail_ops.apply_rope_inplace(
                wkv_out.unsqueeze(1), cos_sin, pos_i32, rope_dim=64, inverse=False
            )
            sail_ops.inplace_act_quant_ue8m0(wkv_out)
            layer_swa_base = self.swa_arena_base[lw.arena_swa_layer_idx]
            self.kv_arena.view(-1, self.head_dim).index_copy_(
                0, layer_swa_base + swa_write_base, wkv_out
            )

            # Build Query (`[rows, 8, 512]`) using `qr_quant_buf` so `qr` stays clean for `Indexer`!
            qr_for_q = sail_ops.act_quant_ue8m0(qr, out=self.qr_quant_buf[:rows])
            q_flat = self.q_buf[:rows].view(rows, -1)
            torch.matmul(qr_for_q, lw.wq_b_bf16.t(), out=q_flat)
            q = q_flat.view(rows, 8, self.head_dim)
            sail_ops.apply_rope_inplace(q, cos_sin, pos_i32, rope_dim=64, inverse=False)
            self.observer.end(ev_qkv)

            layer_swa_topk = torch.where(valid_swa, layer_swa_base + base_swa_win_slots, -1)
            if lw.compress_ratio == 0:
                self.swa_only_topk_buf[:rows].copy_(layer_swa_topk)
                topk_idxs = self.swa_only_topk_buf[:rows]
            else:
                if lw.compressor is not None or lw.indexer is not None:
                    ev_ci = self.observer.begin("engram_compressor_indexer")
                    candidates_mask = self._run_compressor_and_indexer(
                        lw,
                        x_norm,
                        qr,
                        rows=rows,
                        num_seqs=num_seqs,
                        max_seq_len_bound=max_seq_len_bound,
                        uniform_width=uniform_width,
                        qsl_np=qsl_np,
                        seq_lens_np=seq_lens_np,
                        candidates_mask=candidates_mask,
                    )
                    self.observer.end(ev_ci)
                self.shared_full_topk_buf[:rows, :128].copy_(layer_swa_topk)
                topk_idxs = self.shared_full_topk_buf[:rows]

            ev_sa = self.observer.begin("sparse_attn_tp8")
            attn_o = self.attn_out_buf[:rows]
            sail_ops.sparse_attn_tp8_forward(
                q,
                self.kv_arena,
                lw.attn_sink_f32,
                topk_idxs,
                out=attn_o,
                partial_acc=self.attn_partial_acc[: rows * 64],
                partial_m=self.attn_partial_m[: rows * 64],
                partial_l=self.attn_partial_l[: rows * 64],
                q_pad64_buf=self.attn_q_pad64_buf[:rows],
                sm_scale=self.sm_scale,
                backend=self.sparse_attn_backend,
            )
            self.observer.end(ev_sa)

            ev_o = self.observer.begin("attn_qkv_o_proj")
            sail_ops.apply_rope_inplace(attn_o, cos_sin, pos_i32, rope_dim=64, inverse=True)
            o_flat = attn_o.view(rows, 4096)
            sail_ops.inplace_act_quant_ue8m0(o_flat)
            o_lora = self.o_lora_buf[:rows]
            torch.matmul(o_flat, lw.wo_a_bf16.t(), out=o_lora)
            sail_ops.inplace_act_quant_ue8m0(o_lora)
            attn_local = self.sub_local_buf[:rows]
            torch.matmul(o_lora, lw.wo_b_bf16.t(), out=attn_local)
            self.observer.end(ev_o)

            ev_ar1 = self.observer.begin("icn_allreduce")
            attn_red = self.sub_red_buf[:rows]
            self.icn_comm.all_reduce_bf16(attn_local, output=attn_red)
            self.observer.end(ev_ar1)

            ev_mp = self.observer.begin("mhc_mixes")
            sail_ops.mhc_post(attn_red, h_cur, attn_post, attn_comb, out=h_mid)

            # --- FFN / MoE Sub-Block ---
            ffn_post = self.tmp_post_buf[:rows]
            ffn_comb = self.tmp_comb_buf[:rows]
            sail_ops.mhc_mixes_sinkhorn(
                h_mid,
                lw.hc_ffn_fn_f32,
                lw.hc_ffn_fn_bf16,
                lw.hc_ffn_scale_f32,
                lw.hc_ffn_base_f32,
                pre_out=pm_nxt,
                post_out=ffn_post,
                comb_out=ffn_comb,
                raw_proj_buf=self.mhc_proj_buf[:rows],
                split_k_part_buf=self.mhc_split_k_part_buf,
                split_k=self.mhc_split_k,
                fused_max_rows=self.mhc_fused_max_rows,
                norm_eps=self.norm_eps,
                hc_eps=self.hc_eps,
                sinkhorn_iters=self.hc_sinkhorn_iters,
            )
            sail_ops.mhc_pre_rmsnorm(
                h_mid,
                attn_pre,
                lw.ffn_norm_weight,
                normed_out=x_norm,
                norm_eps=self.norm_eps,
            )
            self.observer.end(ev_mp)

            # `rows <= 32` is the decode / verify / batch-decode regime governed by `mxfp4_moe_decode`
            # (`moe_gemv_max_rows` up to 32 + `moe_gemv_threads`); `rows > 32` is `mxfp4_moe_prefill`.
            moe_fam = "mxfp4_moe_decode" if rows <= 32 else "mxfp4_moe_prefill"
            ev_moe = self.observer.begin(moe_fam)
            moe_local = sail_ops.dsv41_moe_tp8_forward(
                x_norm,
                gate_weight=lw.gate_weight_f32,
                gate_bias=lw.gate_bias_f32,
                shared_w13_bf16=lw.shared_w13_bf16,
                shared_w2_bf16=lw.shared_w2_bf16,
                experts_w13_packed=lw.experts_w13_packed,
                experts_w13_scale=lw.experts_w13_scale,
                experts_w2_packed=lw.experts_w2_packed,
                experts_w2_scale=lw.experts_w2_scale,
                top_k=lw.top_k,
                route_scale=self.routed_scaling_factor,
                swiglu_limit=self.swiglu_limit,
                router_logits_buf=self.router_logits_buf,
                topk_weights_buf=self.topk_weights_buf,
                topk_ids_buf=self.topk_ids_buf,
                shared_gu_buf=self.shared_gu_buf,
                shared_out_buf=self.shared_out_buf,
                moe_inter_buf=self.moe_inter_buf,
                sorted_pair_ids_buf=self.sorted_pair_ids_buf,
                expert_ids_per_tile_buf=self.expert_ids_per_tile_buf,
                num_valid_tiles_buf=self.num_valid_tiles_buf,
                pair_out_buf=self.pair_out_buf,
                moe_out_buf=self.sub_local_buf,
                act_quant_x_buf=self.act_quant_buf,
                gemv_max_rows=self.moe_gemv_max_rows,
                gemv_threads=self.moe_gemv_threads,
                block_m=self.moe_block_m,
                num_warps=self.moe_num_warps,
            )
            self.observer.end(ev_moe)

            ev_ar2 = self.observer.begin("icn_allreduce")
            moe_red = self.sub_red_buf[:rows]
            self.icn_comm.all_reduce_bf16(moe_local, output=moe_red)
            self.observer.end(ev_ar2)

            ev_mp2 = self.observer.begin("mhc_mixes")
            sail_ops.mhc_post(moe_red, h_mid, ffn_post, ffn_comb, out=h_cur)
            pm_cur.copy_(pm_nxt)
            self.observer.end(ev_mp2)

        final_h = self.decode_hidden_gpu[:rows]
        sail_ops.mhc_pre_rmsnorm(
            self.h_stream_buf[0][:rows],
            self.pre_mix_buf[0][:rows],
            self.final_norm_weight,
            normed_out=final_h,
            norm_eps=self.norm_eps,
        )

        main_x_out: torch.Tensor | None = None
        if self.dspark_enabled:
            aux_cat = self.dspark_aux_cat_buf[:rows]
            sail_ops.inplace_act_quant_ue8m0(aux_cat)
            mx = self.dspark_main_x_buf[:rows]
            torch.matmul(aux_cat, self.dspark_main_proj_bf16.t(), out=mx)
            sail_ops.rmsnorm_bf16(mx, self.dspark_main_norm_weight, eps=self.norm_eps, out=mx)
            main_x_out = mx
        return final_h, main_x_out

    def _sample_rows_into_argmax(
        self,
        hidden: torch.Tensor,
        sample_rows: torch.Tensor | None,
        num_sample: int,
        *,
        all_ranks: bool,
        return_local_logits: bool = False,
    ) -> torch.Tensor | None:
        sel = hidden if sample_rows is None else hidden.index_select(0, sample_rows)
        logits_f32 = self.logits_f32_buf[:num_sample]
        torch.matmul(sel.float(), self.head_weight_f32.t(), out=logits_f32)
        logits_bf16 = self.logits_bf16_buf[:num_sample]
        logits_bf16.copy_(logits_f32.to(torch.bfloat16))
        self.icn_comm.gather_top2_from_logits(
            logits_bf16,
            local_top2_vals_buf=self.local_top2_vals_buf[:num_sample],
            local_top2_ids_buf=self.local_top2_ids_buf[:num_sample],
            out_tokens=self.out_tokens_gpu[:num_sample],
            out_logprob_bits=self.out_logprobs_gpu[:num_sample],
            vocab_per_rank=self.vocab_per_rank,
        )
        return logits_bf16 if return_local_logits else None

    def _apply_stochastic_sampling(
        self,
        plan: StepPlanView,
        local_logits: torch.Tensor,
        num_sample: int,
        *,
        use_penalty: bool,
        sample_slots_np: np.ndarray,
    ) -> None:
        np.copyto(self.samp_temps_cpu[:num_sample].numpy(), plan.temperatures(num_sample))
        np.copyto(self.samp_top_p_cpu[:num_sample].numpy(), plan.top_p(num_sample))
        np.copyto(self.samp_top_k_cpu[:num_sample].numpy(), plan.top_k(num_sample))
        np.copyto(
            self.samp_seeds_cpu[:num_sample].numpy(),
            plan.rng_seeds(num_sample).view(np.int64),
        )
        self.samp_temps_gpu[:num_sample].copy_(self.samp_temps_cpu[:num_sample], non_blocking=True)
        self.samp_top_p_gpu[:num_sample].copy_(self.samp_top_p_cpu[:num_sample], non_blocking=True)
        self.samp_top_k_gpu[:num_sample].copy_(self.samp_top_k_cpu[:num_sample], non_blocking=True)
        self.samp_seeds_gpu[:num_sample].copy_(self.samp_seeds_cpu[:num_sample], non_blocking=True)

        penalty_state: SlotPenaltyState | None = None
        slots_t: torch.Tensor | None = None
        reset_t: torch.Tensor | None = None
        pres_t: torch.Tensor | None = None
        freq_t: torch.Tensor | None = None
        if use_penalty:
            if self._penalty_state is None:
                self._penalty_state = SlotPenaltyState(self.max_seqs, device=self.device)
            penalty_state = self._penalty_state
            np.copyto(
                self.samp_slots_cpu[:num_sample].numpy(),
                sample_slots_np.astype(np.int64),
            )
            np.copyto(
                self.samp_pres_cpu[:num_sample].numpy(),
                plan.presence_penalties(num_sample),
            )
            np.copyto(
                self.samp_freq_cpu[:num_sample].numpy(),
                plan.frequency_penalties(num_sample),
            )
            np.copyto(
                self.samp_reset_cpu[:num_sample].numpy(),
                plan.penalty_reset_mask(num_sample),
            )
            slots_t = self.samp_slots_gpu[:num_sample]
            pres_t = self.samp_pres_gpu[:num_sample]
            freq_t = self.samp_freq_gpu[:num_sample]
            reset_t = self.samp_reset_gpu[:num_sample]
            slots_t.copy_(self.samp_slots_cpu[:num_sample], non_blocking=True)
            pres_t.copy_(self.samp_pres_cpu[:num_sample], non_blocking=True)
            freq_t.copy_(self.samp_freq_cpu[:num_sample], non_blocking=True)
            reset_t.copy_(self.samp_reset_cpu[:num_sample], non_blocking=True)

        full_ag = self.full_logits_ag_buf[:num_sample]
        self.icn_comm.all_gather_bf16(local_logits, output=full_ag)
        toks, s_lps, ru_toks, ru_lps = sample_tp8_logits(
            full_ag[:, : self.vocab_size],
            temperatures=self.samp_temps_gpu[:num_sample],
            top_p=self.samp_top_p_gpu[:num_sample],
            top_k=self.samp_top_k_gpu[:num_sample],
            rng_seeds=self.samp_seeds_gpu[:num_sample],
            world_size=1,
            vocab_size=self.vocab_size,
            penalty_state=penalty_state,
            slots=slots_t,
            reset_mask=reset_t,
            presence_penalties=pres_t,
            frequency_penalties=freq_t,
            return_top2=True,
        )
        self.out_tokens_gpu[:num_sample, 0].copy_(toks)
        self.out_tokens_gpu[:num_sample, 1].copy_(ru_toks)
        self.out_logprobs_gpu[:num_sample, 0].copy_(s_lps.view(torch.int32))
        self.out_logprobs_gpu[:num_sample, 1].copy_(ru_lps.view(torch.int32))

    # -----------------------------------------------------------------------
    # Full-40-Layer Decode & Verify `CUDAGraph` Capture
    # -----------------------------------------------------------------------

    def _run_decode_bucket_forward(self, bucket: int) -> None:
        hidden, _ = self._forward_backbone(
            rows=bucket,
            num_seqs=bucket,
            qsl_gpu=self.decode_qsl_gpu[: bucket + 1],
            max_seq_len_bound=self.decode_graph_max_seq_len,
            uniform_width=1,
        )
        self._sample_rows_into_argmax(hidden, None, bucket, all_ranks=True)

    @torch.inference_mode()
    def _capture_all_graphs(self) -> None:
        if self.observer.enabled:
            # When Evolith's per-operator CUDA-event observer is active, run eager so
            # every operator family's CUDA events record real per-call device timings!
            return

        graph_pool = torch.cuda.graph_pool_handle()
        capture_stream = torch.cuda.Stream(device=self.device)
        self._capture_stream = capture_stream
        decode_buckets = [b for b in DECODE_GRAPH_BUCKETS if b <= self.max_graph_bucket]

        for b in decode_buckets:
            self.token_ids_gpu[:b].zero_()
            self.positions_gpu[:b].zero_()
            self.lookback_gpu[:b].fill_(-1)
            self.row_seq_ids_gpu[:b].copy_(self.arange_seqs_i64[:b])
            self.row_comp_slots_gpu[:b].zero_()
            self.swa_write_slots_gpu[:b].zero_()
            self.mla_write_pages_gpu[:b].zero_()
            self.mla_bt_gpu[:b].zero_()
            self.swa_bt_gpu[:b].zero_()

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

        if (
            self.spec_method != "none"
            and self.num_spec_tokens > 0
            and not self.runtime_config.disable_verify_graphs
        ):
            width = self.num_spec_tokens + 1
            max_v_rows = min(256, self.max_step_tokens)
            self.verify_buckets = [
                b for b in decode_buckets if b * width <= max_v_rows
            ]
            for vb in self.verify_buckets:
                rows = vb * width
                self.token_ids_gpu[:rows].zero_()
                self.positions_gpu[:rows].copy_(
                    torch.arange(rows, dtype=torch.int32, device=self.device) % width
                )
                self.lookback_gpu[:vb].fill_(-1)
                self.row_seq_ids_gpu[:rows].copy_(
                    self.arange_seqs_i64[:vb, None].expand(vb, width).reshape(-1)
                )
                self.row_comp_slots_gpu[:rows].zero_()
                self.swa_write_slots_gpu[:rows].zero_()
                self.mla_write_pages_gpu[:rows].zero_()
                self.mla_bt_gpu[:vb].zero_()
                self.swa_bt_gpu[:vb].zero_()

                with torch.cuda.stream(capture_stream):
                    for _ in range(2):
                        vh, _ = self._forward_backbone(
                            rows=rows,
                            num_seqs=vb,
                            qsl_gpu=self.verify_qsl_gpu[: vb + 1],
                            max_seq_len_bound=self.decode_graph_max_seq_len,
                            uniform_width=width,
                        )
                        self._sample_rows_into_argmax(vh, None, rows, all_ranks=True)
                capture_stream.synchronize()
                dist.barrier(group=self.exchange_group)

                vg = torch.cuda.CUDAGraph()
                v_mx: torch.Tensor | None = None
                with torch.cuda.graph(vg, pool=graph_pool, stream=capture_stream):
                    vh, v_mx = self._forward_backbone(
                        rows=rows,
                        num_seqs=vb,
                        qsl_gpu=self.verify_qsl_gpu[: vb + 1],
                        max_seq_len_bound=self.decode_graph_max_seq_len,
                        uniform_width=width,
                    )
                    self._sample_rows_into_argmax(vh, None, rows, all_ranks=True)
                capture_stream.synchronize()
                dist.barrier(group=self.exchange_group)
                self.verify_graphs[vb] = vg
                self.verify_graph_main_x[vb] = v_mx

        if self.dspark_drafter is not None:
            self.dspark_drafter.capture_graphs(graph_pool, capture_stream)

    # -----------------------------------------------------------------------
    # Hot-Path Step Execution (`launch` / `finish` / `execute`)
    # -----------------------------------------------------------------------

    def _commit_decode_slot_state(self, n: int) -> None:
        s_idx = self.slots_gpu[:n]
        self.next_lb_gpu[:n, 0:2].copy_(self.lookback_gpu[:n, 1:3])
        self.next_lb_gpu[:n, 2].copy_(self.token_ids_gpu[:n])
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

    @torch.inference_mode()
    def execute(
        self, plan: StepPlanView, header: StepHeader
    ) -> tuple[list[SeqResult], int]:
        return self.finish(self.launch(plan, header))

    @torch.inference_mode()
    def launch(
        self, plan: StepPlanView, header: StepHeader
    ) -> _SailLaunchedStep | None:
        num_seqs = int(header.num_seqs)
        total_tokens = int(header.num_encoder_tokens)
        num_sample = int(header.num_sample_rows)
        if total_tokens == 0 or num_seqs == 0:
            return None
        parity = int(header.step_id) & 1

        tok_np = plan.token_ids(total_tokens)
        pos_np = plan.positions(total_tokens)
        slot_np = plan.req_slot_ids(total_tokens)
        seq_lens_np = plan.seq_lens(num_seqs)
        q_start_np = plan.query_start_loc(num_seqs + 1)
        prev_win_np = plan.engram_prev_window(num_seqs)
        sample_rows_np = plan.sample_row_indices(num_sample)

        max_slot_end = min(self.max_seqs, int(slot_np.max()) + 1)
        pt_np = plan.global_kv_page_table(max_slot_end)[:, : self.bt_width]

        is_spec_step = header.is_dspark_verify != 0 or (
            self.spec_method == "ngram"
            and self.num_spec_tokens > 0
            and (header.flags & STEP_FLAG_DECODE_ONLY) != 0
        )
        is_decode = (header.flags & STEP_FLAG_DECODE_ONLY) != 0 and not is_spec_step
        is_sampling = (header.flags & STEP_FLAG_SAMPLING) != 0
        is_penalty = (header.flags & STEP_FLAG_PENALTY) != 0
        use_dev_tok = is_decode and (header.flags & STEP_FLAG_DEVICE_TOKENS) != 0

        width = self.num_spec_tokens + 1
        q_lens = q_start_np[1:] - q_start_np[:-1]
        max_s_len = int(seq_lens_np.max())

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
            and max_s_len <= self.decode_graph_max_seq_len
            and (is_uniform_verify or is_ragged_verify)
            and not is_sampling
        )
        use_decode_graph = (
            is_decode
            and max_s_len <= self.decode_graph_max_seq_len
            and bool(self.decode_graphs)
            and num_seqs <= self.max_graph_bucket
        )

        self._h2d_done_event.synchronize()
        self.observer.drain_completed()
        self.start_events[parity].record()
        main_x_out: torch.Tensor | None = None

        if use_decode_graph:
            bucket = select_decode_bucket(num_seqs, self.max_graph_bucket)
            self.slots_cpu[:num_seqs].numpy()[:] = slot_np.astype(np.int64)
            self.slots_gpu[:num_seqs].copy_(self.slots_cpu[:num_seqs], non_blocking=True)

            pos_i64 = pos_np.astype(np.int64)
            active_mla_pt = (pt_np[slot_np, :] + 1).astype(np.int32)
            if self._swa_pool is not None:
                swa_pt = self._swa_pool.map_active_windows(active_mla_pt, pos_i64, pos_i64)
            else:
                swa_pt = active_mla_pt

            mla_bt_np = self.mla_bt_cpu[:bucket].numpy()
            mla_bt_np[:num_seqs] = active_mla_pt
            if bucket > num_seqs:
                mla_bt_np[num_seqs:bucket] = 0

            swa_bt_np = self.swa_bt_cpu[:bucket].numpy()
            swa_bt_np[:num_seqs] = swa_pt
            if bucket > num_seqs:
                swa_bt_np[num_seqs:bucket] = 0

            if not use_dev_tok:
                lb_np = self.lookback_cpu[:bucket].numpy()
                lb_np[:num_seqs] = prev_win_np
                if bucket > num_seqs:
                    lb_np[num_seqs:bucket] = -1
                tok_c = self.token_ids_cpu[:bucket].numpy()
                tok_c[:num_seqs] = tok_np
                if bucket > num_seqs:
                    tok_c[num_seqs:bucket] = 0

            pos_c = self.positions_cpu[:bucket].numpy()
            pos_c[:num_seqs] = pos_np
            if bucket > num_seqs:
                pos_c[num_seqs:bucket] = 0

            rs_c = self.row_seq_ids_cpu[:bucket].numpy()
            rs_c[:bucket] = np.arange(bucket, dtype=np.int64)

            cs_c = self.row_comp_slots_cpu[:bucket].numpy()
            cs_c[:num_seqs] = slot_np.astype(np.int64) + 1
            if bucket > num_seqs:
                cs_c[num_seqs:bucket] = 0

            c_hi = np.clip(pos_i64 // 128, 0, self.bt_width - 1)
            mla_vb_tok = active_mla_pt[np.arange(num_seqs), c_hi].astype(np.int64)
            swa_vb_tok = swa_pt[np.arange(num_seqs), c_hi].astype(np.int64)
            sw_c = self.swa_write_slots_cpu[:bucket].numpy()
            sw_c[:num_seqs] = np.where(swa_vb_tok > 0, swa_vb_tok * 128 + (pos_i64 % 128), 0)
            if bucket > num_seqs:
                sw_c[num_seqs:bucket] = 0

            mw_c = self.mla_write_pages_cpu[:bucket].numpy()
            mw_c[:num_seqs] = mla_vb_tok
            if bucket > num_seqs:
                mw_c[num_seqs:bucket] = 0

            self.mla_bt_gpu[:bucket].copy_(self.mla_bt_cpu[:bucket], non_blocking=True)
            self.swa_bt_gpu[:bucket].copy_(self.swa_bt_cpu[:bucket], non_blocking=True)
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
            else:
                self.lookback_gpu[:bucket].copy_(self.lookback_cpu[:bucket], non_blocking=True)
                self.token_ids_gpu[:bucket].copy_(
                    self.token_ids_cpu[:bucket], non_blocking=True
                )
            self.positions_gpu[:bucket].copy_(self.positions_cpu[:bucket], non_blocking=True)
            self.row_seq_ids_gpu[:bucket].copy_(self.row_seq_ids_cpu[:bucket], non_blocking=True)
            self.row_comp_slots_gpu[:bucket].copy_(
                self.row_comp_slots_cpu[:bucket], non_blocking=True
            )
            self.swa_write_slots_gpu[:bucket].copy_(
                self.swa_write_slots_cpu[:bucket], non_blocking=True
            )
            self.mla_write_pages_gpu[:bucket].copy_(
                self.mla_write_pages_cpu[:bucket], non_blocking=True
            )
            self._h2d_done_event.record()

            self.decode_graphs[bucket].replay()
            if is_sampling:
                local_f32 = self.logits_f32_buf[:num_seqs]
                torch.matmul(
                    self.decode_hidden_gpu[:num_seqs].float(),
                    self.head_weight_f32.t(),
                    out=local_f32,
                )
                local_bf16 = self.logits_bf16_buf[:num_seqs]
                local_bf16.copy_(local_f32.to(torch.bfloat16))
                self._apply_stochastic_sampling(
                    plan,
                    local_bf16,
                    num_seqs,
                    use_penalty=is_penalty,
                    sample_slots_np=slot_np,
                )
            self._commit_decode_slot_state(num_seqs)
            if self.is_result_rank:
                self.out_tokens_cpu[parity, :num_seqs].copy_(
                    self.out_tokens_gpu[:num_seqs], non_blocking=True
                )
                self.out_logprobs_cpu[parity, :num_seqs].copy_(
                    self.out_logprobs_gpu[:num_seqs], non_blocking=True
                )
            sample_slots_out = [int(slot_np[i]) for i in range(num_seqs)]
            num_out = num_seqs

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

            active_mla_pt = (pt_np[seq_slots_np, :] + 1).astype(np.int32)
            if self._swa_pool is not None:
                swa_pt = self._swa_pool.map_active_windows(active_mla_pt, p_min, p_max)
            else:
                swa_pt = active_mla_pt

            mla_bt_np = self.mla_bt_cpu[:v_bucket].numpy()
            mla_bt_np[:num_seqs] = active_mla_pt
            if v_bucket > num_seqs:
                mla_bt_np[num_seqs:v_bucket] = 0

            swa_bt_np = self.swa_bt_cpu[:v_bucket].numpy()
            swa_bt_np[:num_seqs] = swa_pt
            if v_bucket > num_seqs:
                swa_bt_np[num_seqs:v_bucket] = 0

            lb_np = self.lookback_cpu[:v_bucket].numpy()
            lb_np[:num_seqs] = prev_win_np
            if v_bucket > num_seqs:
                lb_np[num_seqs:v_bucket] = -1

            tok_c = self.token_ids_cpu[:rows].numpy()
            pos_c = self.positions_cpu[:rows].numpy()
            rs_c = self.row_seq_ids_cpu[:rows].numpy()
            cs_c = self.row_comp_slots_cpu[:rows].numpy()
            sw_c = self.swa_write_slots_cpu[:rows].numpy()
            mw_c = self.mla_write_pages_cpu[:rows].numpy()

            tok_c[:] = 0
            pos_c[:] = 0
            rs_c[:] = np.repeat(np.arange(v_bucket, dtype=np.int64), width)
            cs_c[:] = 0
            sw_c[:] = 0
            mw_c[:] = 0

            c_cols_tok = np.clip(pos_i64 // 128, 0, self.bt_width - 1)
            mla_vb_all = (pt_np[slot_np, c_cols_tok] + 1).astype(np.int64)
            if self._swa_pool is not None:
                swa_vb_all = self._swa_pool.lookup(mla_vb_all.astype(np.int32)).astype(np.int64)
            else:
                swa_vb_all = mla_vb_all
            sw_all = np.where(swa_vb_all > 0, swa_vb_all * 128 + (pos_i64 % 128), 0)

            if is_ragged_verify:
                real_rows_np = self._verify_real_rows_cpu[:total_tokens].numpy()
                w_cur = 0
                for s_i in range(num_seqs):
                    src_s = int(q_start_np[s_i])
                    src_e = int(q_start_np[s_i + 1])
                    ql = src_e - src_s
                    dst_s = s_i * width
                    tok_c[dst_s : dst_s + ql] = tok_np[src_s:src_e]
                    pos_c[dst_s : dst_s + ql] = pos_np[src_s:src_e]
                    cs_c[dst_s : dst_s + ql] = int(seq_slots_np[s_i]) + 1
                    sw_c[dst_s : dst_s + ql] = sw_all[src_s:src_e]
                    mw_c[dst_s : dst_s + ql] = mla_vb_all[src_s:src_e]
                    if ql < width:
                        last_p = int(pos_np[src_e - 1])
                        for p_off in range(1, width - ql + 1):
                            pos_c[dst_s + ql + p_off - 1] = last_p + p_off
                    for j in range(ql):
                        real_rows_np[w_cur + j] = dst_s + j
                    w_cur += ql
                self._verify_real_rows_gpu[:total_tokens].copy_(
                    self._verify_real_rows_cpu[:total_tokens], non_blocking=True
                )
            else:
                tok_c[:total_tokens] = tok_np
                pos_c[:total_tokens] = pos_np
                cs_c[:total_tokens] = slot_np.astype(np.int64) + 1
                sw_c[:total_tokens] = sw_all
                mw_c[:total_tokens] = mla_vb_all

            self.mla_bt_gpu[:v_bucket].copy_(self.mla_bt_cpu[:v_bucket], non_blocking=True)
            self.swa_bt_gpu[:v_bucket].copy_(self.swa_bt_cpu[:v_bucket], non_blocking=True)
            self.lookback_gpu[:v_bucket].copy_(self.lookback_cpu[:v_bucket], non_blocking=True)
            self.token_ids_gpu[:rows].copy_(self.token_ids_cpu[:rows], non_blocking=True)
            self.positions_gpu[:rows].copy_(self.positions_cpu[:rows], non_blocking=True)
            self.row_seq_ids_gpu[:rows].copy_(self.row_seq_ids_cpu[:rows], non_blocking=True)
            self.row_comp_slots_gpu[:rows].copy_(
                self.row_comp_slots_cpu[:rows], non_blocking=True
            )
            self.swa_write_slots_gpu[:rows].copy_(
                self.swa_write_slots_cpu[:rows], non_blocking=True
            )
            self.mla_write_pages_gpu[:rows].copy_(
                self.mla_write_pages_cpu[:rows], non_blocking=True
            )
            self._h2d_done_event.record()

            self.verify_graphs[v_bucket].replay()
            raw_mx = self.verify_graph_main_x.get(v_bucket)
            if is_ragged_verify:
                real_r = self._verify_real_rows_gpu[:total_tokens]
                main_x_out = raw_mx.index_select(0, real_r) if raw_mx is not None else None
                self.out_tokens_cpu[parity, :num_sample].copy_(
                    self.out_tokens_gpu.index_select(0, real_r), non_blocking=True
                )
                self.out_logprobs_cpu[parity, :num_sample].copy_(
                    self.out_logprobs_gpu.index_select(0, real_r), non_blocking=True
                )
            else:
                main_x_out = raw_mx[:total_tokens] if raw_mx is not None else None
                self.out_tokens_cpu[parity, :num_sample].copy_(
                    self.out_tokens_gpu[:num_sample], non_blocking=True
                )
                self.out_logprobs_cpu[parity, :num_sample].copy_(
                    self.out_logprobs_gpu[:num_sample], non_blocking=True
                )
            sample_slots_out = []
            num_out = num_sample

        else:
            # General varlen forward (prefill, chunked prefill, APC, mixed step, long-context decode/verify)
            seq_slots_np = slot_np[q_start_np[:num_seqs]]
            pos_i64 = pos_np.astype(np.int64)
            p_min = pos_i64[q_start_np[:num_seqs]]
            p_max = pos_i64[q_start_np[1 : num_seqs + 1] - 1]

            active_mla_pt = (pt_np[seq_slots_np, :] + 1).astype(np.int32)
            if self._swa_pool is not None:
                swa_pt = self._swa_pool.map_active_windows(
                    active_mla_pt, p_min, p_max, window_tokens=max(128, int(q_lens.max()) + 128)
                )
            else:
                swa_pt = active_mla_pt

            self.mla_bt_cpu[:num_seqs].numpy()[:] = active_mla_pt
            self.swa_bt_cpu[:num_seqs].numpy()[:] = swa_pt
            if not use_dev_tok:
                self.lookback_cpu[:num_seqs].numpy()[:] = prev_win_np
                np.copyto(self.token_ids_cpu[:total_tokens].numpy(), tok_np)

            np.copyto(self.positions_cpu[:total_tokens].numpy(), pos_np)
            np.copyto(self.query_start_loc_cpu[: num_seqs + 1].numpy(), q_start_np)

            rs_np = self.row_seq_ids_cpu[:total_tokens].numpy()
            for s_i in range(num_seqs):
                rs_np[int(q_start_np[s_i]) : int(q_start_np[s_i + 1])] = s_i
            self.row_comp_slots_cpu[:total_tokens].numpy()[:] = slot_np.astype(np.int64) + 1

            c_cols_tok = np.clip(pos_i64 // 128, 0, self.bt_width - 1)
            mla_vb_tok = (pt_np[slot_np, c_cols_tok] + 1).astype(np.int64)
            if self._swa_pool is not None:
                swa_vb_tok = self._swa_pool.lookup(mla_vb_tok.astype(np.int32)).astype(np.int64)
            else:
                swa_vb_tok = mla_vb_tok
            self.swa_write_slots_cpu[:total_tokens].numpy()[:] = np.where(
                swa_vb_tok > 0, swa_vb_tok * 128 + (pos_i64 % 128), 0
            )
            self.mla_write_pages_cpu[:total_tokens].numpy()[:] = mla_vb_tok

            if num_sample > 0 and not is_decode:
                self.sample_rows_cpu[:num_sample].numpy()[:] = sample_rows_np.astype(np.int64)
                self.sample_rows_gpu[:num_sample].copy_(
                    self.sample_rows_cpu[:num_sample], non_blocking=True
                )

            self.mla_bt_gpu[:num_seqs].copy_(self.mla_bt_cpu[:num_seqs], non_blocking=True)
            self.swa_bt_gpu[:num_seqs].copy_(self.swa_bt_cpu[:num_seqs], non_blocking=True)
            if is_decode:
                self.slots_cpu[:num_seqs].numpy()[:] = seq_slots_np.astype(np.int64)
                self.slots_gpu[:num_seqs].copy_(self.slots_cpu[:num_seqs], non_blocking=True)
            if use_dev_tok:
                s_idx = self.slots_gpu[:num_seqs]
                torch.index_select(
                    self.slot_last_tok_gpu, 0, s_idx, out=self.token_ids_gpu[:num_seqs]
                )
                torch.index_select(
                    self.slot_lb_gpu, 0, s_idx, out=self.lookback_gpu[:num_seqs]
                )
            else:
                self.lookback_gpu[:num_seqs].copy_(
                    self.lookback_cpu[:num_seqs], non_blocking=True
                )
                self.token_ids_gpu[:total_tokens].copy_(
                    self.token_ids_cpu[:total_tokens], non_blocking=True
                )
            self.positions_gpu[:total_tokens].copy_(
                self.positions_cpu[:total_tokens], non_blocking=True
            )
            self.query_start_loc_gpu[: num_seqs + 1].copy_(
                self.query_start_loc_cpu[: num_seqs + 1], non_blocking=True
            )
            self.row_seq_ids_gpu[:total_tokens].copy_(
                self.row_seq_ids_cpu[:total_tokens], non_blocking=True
            )
            self.row_comp_slots_gpu[:total_tokens].copy_(
                self.row_comp_slots_cpu[:total_tokens], non_blocking=True
            )
            self.swa_write_slots_gpu[:total_tokens].copy_(
                self.swa_write_slots_cpu[:total_tokens], non_blocking=True
            )
            self.mla_write_pages_gpu[:total_tokens].copy_(
                self.mla_write_pages_cpu[:total_tokens], non_blocking=True
            )
            self._h2d_done_event.record()

            hidden, main_x_out = self._forward_backbone(
                rows=total_tokens,
                num_seqs=num_seqs,
                qsl_gpu=self.query_start_loc_gpu[: num_seqs + 1],
                max_seq_len_bound=max_s_len,
                uniform_width=1 if (total_tokens == num_seqs) else 0,
                qsl_np=q_start_np,
                seq_lens_np=seq_lens_np,
            )

            # Seed DSpark context KV on non-speculative prefill/mixed steps too!
            if (
                self.dspark_drafter is not None
                and main_x_out is not None
                and not is_spec_step
            ):
                self.dspark_drafter.store_context_kv(
                    main_x_out,
                    ctx_rows_np=np.arange(total_tokens, dtype=np.intp),
                    ctx_slots_np=slot_np.astype(np.intp),
                    positions_np=pos_np,
                    page_table_np=pt_np,
                )

            if num_sample > 0:
                local_bf16 = self._sample_rows_into_argmax(
                    hidden,
                    None if is_decode else self.sample_rows_gpu[:num_sample],
                    num_sample,
                    all_ranks=(is_spec_step or is_decode),
                    return_local_logits=is_sampling,
                )
                if is_sampling and local_bf16 is not None:
                    self._apply_stochastic_sampling(
                        plan,
                        local_bf16,
                        num_sample,
                        use_penalty=is_penalty,
                        sample_slots_np=seq_slots_np if is_decode else slot_np[sample_rows_np],
                    )
            if is_decode:
                self._commit_decode_slot_state(num_seqs)

            if (self.is_result_rank or is_spec_step) and num_sample > 0:
                self.out_tokens_cpu[parity, :num_sample].copy_(
                    self.out_tokens_gpu[:num_sample], non_blocking=True
                )
                self.out_logprobs_cpu[parity, :num_sample].copy_(
                    self.out_logprobs_gpu[:num_sample], non_blocking=True
                )
            sample_slots_out = (
                [int(seq_slots_np[i]) for i in range(num_seqs)]
                if is_decode
                else [int(slot_np[int(sample_rows_np[i])]) for i in range(num_sample)]
            )
            num_out = num_sample

        self.end_events[parity].record()
        spec_ctx = (
            _SailSpecVerifyContext(
                num_sample=num_sample,
                sample_rows_np=sample_rows_np.copy(),
                slot_np=slot_np.copy(),
                tok_np=tok_np.copy(),
                pos_np=pos_np.copy(),
                q_start_np=q_start_np.copy(),
                pt_np=pt_np.copy(),
                main_x_out=main_x_out,
            )
            if is_spec_step
            else None
        )
        return _SailLaunchedStep(
            parity=parity,
            num_out=num_out,
            sample_slots_out=sample_slots_out,
            spec_ctx=spec_ctx,
        )

    @torch.inference_mode()
    def finish(
        self, step: _SailLaunchedStep | None
    ) -> tuple[list[SeqResult], int]:
        if step is None:
            return [], 1
        p = step.parity
        self.end_events[p].synchronize()
        outputs: list[SeqResult] = []
        if step.spec_ctx is not None:
            ctx = step.spec_ctx
            verified = verify_batch(
                predictions=self._out_tokens_np[p][: ctx.num_sample, 0],
                logprob_bits=self._out_logprobs_np[p][: ctx.num_sample, 0],
                token_ids=ctx.tok_np,
                sample_rows=ctx.sample_rows_np,
                query_start_loc=ctx.q_start_np,
                req_slot_ids=ctx.slot_np,
            )
            drafts: dict[int, list[int]] = {}
            if self.dspark_drafter is not None:
                drafts = self.dspark_drafter.draft(
                    verified,
                    main_x_out=ctx.main_x_out,
                    positions_np=ctx.pos_np,
                    req_slot_ids_np=ctx.slot_np,
                    query_start_loc_np=ctx.q_start_np,
                    page_table_np=ctx.pt_np,
                )
            outputs = seq_results(verified, drafts) if self.is_result_rank else []
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

    def signal_abort(self) -> None:
        if self.icn_comm is not None:
            self.icn_comm.signal_abort()

    def close(self, *, abort: bool = False) -> None:
        if self._closed:
            return
        self._closed = True
        if abort:
            self.signal_abort()
        else:
            self.observer.flush_json(self.device)
        self.decode_graphs.clear()
        self.verify_graphs.clear()
        self.verify_graph_main_x.clear()
        if self.dspark_drafter is not None:
            self.dspark_drafter.close()
        if not abort and dist.is_initialized():
            dist.barrier(group=self.exchange_group)
        if self.icn_comm is not None:
            self.icn_comm.close(abort=abort)
        if dist.is_initialized():
            if not abort:
                dist.barrier(group=self.exchange_group)
            dist.destroy_process_group()
