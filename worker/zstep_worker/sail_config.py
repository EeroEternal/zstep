# Copyright (c) 2026 zstep Contributors
# SPDX-License-Identifier: MIT

"""Typed T-Head PPU-ZW810E (SAIL) runtime policy configuration.

Single source of truth for ``ZSTEP_SAIL_*`` environment variables and
Evolith-approved SAIL policy JSON keys. Both ``WorkerRuntimeConfig`` and
``scripts/export_nebula_bundle.py`` derive their mappings from this module.
"""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass, fields
import json
import os
from pathlib import Path

VALID_SAIL_BACKENDS = frozenset({"auto", "sail", "sm120"})
VALID_SAIL_SPARSE_ATTN_BACKENDS = frozenset(
    {"auto", "flash_mla", "triton_tp8", "triton_splitk", "hggc_tp8"}
)
VALID_SAIL_MOE_PREFILL_BACKENDS = frozenset({"triton_grouped", "dequant_acblas"})
VALID_SAIL_QKV_FUSION_MODES = frozenset({"separate", "fused_qkv_epilogue"})
VALID_SAIL_ENGRAM_INDEXER_BACKENDS = frozenset({"eager_py", "fused_hggc", "fused_hggc_ozaki"})
VALID_SAIL_LM_HEAD_DTYPES = frozenset({"fp32", "bf16"})
VALID_SAIL_MOE_ROUTER_MODES = frozenset({"fp32", "ozaki_bf16", "ozaki_fast_fp4"})
VALID_SAIL_MHC_BACKENDS = frozenset({"legacy_2pass", "vec_1pass_ozaki"})

_NON_POLICY_FIELDS = frozenset({"policy_path", "observe_path"})


def _parse_int(name: str, default: int, *, environ: Mapping[str, str]) -> int:
    raw = environ.get(name, "").strip()
    if not raw:
        return default
    try:
        return int(raw)
    except ValueError as exc:
        raise ValueError(f"Invalid integer for {name}: {raw!r}") from exc


@dataclass(frozen=True)
class SailRuntimeConfig:
    """Validated SAIL backend runtime policy parsed from ``ZSTEP_SAIL_*`` / policy JSON."""

    policy_path: str = ""
    ar_blocks: int = 8
    ar_threads: int = 256
    ar_oneshot_max_bytes: int = 131072
    ar_pccl_min_bytes: int = 67108864
    moe_gemv_max_rows: int = 16
    moe_gemv_threads: int = 256
    moe_block_m: int = 32
    moe_num_warps: int = 4
    moe_prefill_backend: str = "triton_grouped"
    sparse_attn_backend: str = "auto"
    mhc_split_k: int = 1
    mhc_fused_max_rows: int = 16
    qkv_fusion_mode: str = "separate"
    engram_indexer_backend: str = "eager_py"
    lm_head_dtype: str = "fp32"
    ar_fuse_mhc_post: int = 0
    moe_router_mode: str = "fp32"
    mhc_backend: str = "legacy_2pass"
    observe_path: str = ""

    @classmethod
    def policy_env_map(cls) -> dict[str, str]:
        """Return ``{policy_json_key: ZSTEP_SAIL_<KEY>}`` for all Evolith policy knobs."""
        return {
            f.name: f"ZSTEP_SAIL_{f.name.upper()}"
            for f in fields(cls)
            if f.name not in _NON_POLICY_FIELDS
        }

    @classmethod
    def from_env(cls, environ: Mapping[str, str] | None = None) -> SailRuntimeConfig:
        env = os.environ if environ is None else environ

        policy_path = env.get("ZSTEP_SAIL_POLICY_PATH", "").strip()
        policy_defaults: dict[str, object] = {}
        if policy_path:
            raw_policy = json.loads(Path(policy_path).read_text(encoding="utf-8"))
            if not isinstance(raw_policy, dict):
                raise ValueError("ZSTEP_SAIL_POLICY_PATH must contain a JSON object")
            policy_defaults = raw_policy

        def _sail_int(env_key: str, policy_key: str, fallback: int) -> int:
            if env.get(env_key, "").strip():
                return _parse_int(env_key, fallback, environ=env)
            if policy_key in policy_defaults:
                val = policy_defaults[policy_key]
                if isinstance(val, bool) or not isinstance(val, int):
                    raise ValueError(f"Policy field {policy_key!r} must be an integer")
                return val
            return fallback

        def _sail_str(env_key: str, policy_key: str, fallback: str) -> str:
            raw_val = env.get(env_key, "").strip()
            if raw_val:
                return raw_val
            if policy_key in policy_defaults:
                val = policy_defaults[policy_key]
                if not isinstance(val, str):
                    raise ValueError(f"Policy field {policy_key!r} must be a string")
                return val.strip() or fallback
            return fallback

        ar_blocks = _sail_int("ZSTEP_SAIL_AR_BLOCKS", "ar_blocks", 8)
        if not 1 <= ar_blocks <= 64:
            raise ValueError(f"sail_ar_blocks must be in 1..=64 (got {ar_blocks})")
        ar_threads = _sail_int("ZSTEP_SAIL_AR_THREADS", "ar_threads", 256)
        if ar_threads not in (64, 128, 256, 512):
            raise ValueError(f"sail_ar_threads must be 64, 128, 256, or 512 (got {ar_threads})")
        ar_oneshot_max_bytes = _sail_int(
            "ZSTEP_SAIL_AR_ONESHOT_MAX_BYTES", "ar_oneshot_max_bytes", 131072
        )
        if ar_oneshot_max_bytes < 0:
            raise ValueError("sail_ar_oneshot_max_bytes must be >= 0")
        ar_pccl_min_bytes = _sail_int(
            "ZSTEP_SAIL_AR_PCCL_MIN_BYTES", "ar_pccl_min_bytes", 67108864
        )
        if ar_pccl_min_bytes < 0:
            raise ValueError("sail_ar_pccl_min_bytes must be >= 0")
        moe_gemv_max_rows = _sail_int("ZSTEP_SAIL_MOE_GEMV_MAX_ROWS", "moe_gemv_max_rows", 16)
        if not 0 <= moe_gemv_max_rows <= 256:
            raise ValueError("sail_moe_gemv_max_rows must be in 0..=256")
        moe_gemv_threads = _sail_int("ZSTEP_SAIL_MOE_GEMV_THREADS", "moe_gemv_threads", 256)
        if moe_gemv_threads not in (128, 256, 512):
            raise ValueError("sail_moe_gemv_threads must be 128, 256, or 512")
        moe_block_m = _sail_int("ZSTEP_SAIL_MOE_BLOCK_M", "moe_block_m", 32)
        if moe_block_m not in (16, 32, 64):
            raise ValueError("sail_moe_block_m must be 16, 32, or 64")
        moe_num_warps = _sail_int("ZSTEP_SAIL_MOE_NUM_WARPS", "moe_num_warps", 4)
        if moe_num_warps not in (2, 4, 8):
            raise ValueError("sail_moe_num_warps must be 2, 4, or 8")
        sparse_attn_backend = _sail_str(
            "ZSTEP_SAIL_SPARSE_ATTN_BACKEND", "sparse_attn_backend", "auto"
        ).lower()
        if sparse_attn_backend not in VALID_SAIL_SPARSE_ATTN_BACKENDS:
            raise ValueError(
                f"Unsupported ZSTEP_SAIL_SPARSE_ATTN_BACKEND={sparse_attn_backend!r}; "
                f"expected one of {sorted(VALID_SAIL_SPARSE_ATTN_BACKENDS)}"
            )
        mhc_split_k = _sail_int("ZSTEP_SAIL_MHC_SPLIT_K", "mhc_split_k", 1)
        if mhc_split_k not in (1, 2, 4, 8, 16):
            raise ValueError("sail_mhc_split_k must be 1, 2, 4, 8, or 16")
        mhc_fused_max_rows = _sail_int(
            "ZSTEP_SAIL_MHC_FUSED_MAX_ROWS", "mhc_fused_max_rows", 16
        )
        if not 0 <= mhc_fused_max_rows <= 4096:
            raise ValueError("sail_mhc_fused_max_rows must be in 0..=4096")
        moe_prefill_backend = _sail_str(
            "ZSTEP_SAIL_MOE_PREFILL_BACKEND", "moe_prefill_backend", "triton_grouped"
        ).lower()
        if moe_prefill_backend not in VALID_SAIL_MOE_PREFILL_BACKENDS:
            raise ValueError(
                f"Unsupported ZSTEP_SAIL_MOE_PREFILL_BACKEND={moe_prefill_backend!r}; "
                f"expected one of {sorted(VALID_SAIL_MOE_PREFILL_BACKENDS)}"
            )
        qkv_fusion_mode = _sail_str(
            "ZSTEP_SAIL_QKV_FUSION_MODE", "qkv_fusion_mode", "separate"
        ).lower()
        if qkv_fusion_mode not in VALID_SAIL_QKV_FUSION_MODES:
            raise ValueError(
                f"Unsupported ZSTEP_SAIL_QKV_FUSION_MODE={qkv_fusion_mode!r}; "
                f"expected one of {sorted(VALID_SAIL_QKV_FUSION_MODES)}"
            )
        engram_indexer_backend = _sail_str(
            "ZSTEP_SAIL_ENGRAM_INDEXER_BACKEND", "engram_indexer_backend", "eager_py"
        ).lower()
        if engram_indexer_backend not in VALID_SAIL_ENGRAM_INDEXER_BACKENDS:
            raise ValueError(
                f"Unsupported ZSTEP_SAIL_ENGRAM_INDEXER_BACKEND={engram_indexer_backend!r}; "
                f"expected one of {sorted(VALID_SAIL_ENGRAM_INDEXER_BACKENDS)}"
            )
        lm_head_dtype = _sail_str("ZSTEP_SAIL_LM_HEAD_DTYPE", "lm_head_dtype", "fp32").lower()
        if lm_head_dtype not in VALID_SAIL_LM_HEAD_DTYPES:
            raise ValueError(
                f"Unsupported ZSTEP_SAIL_LM_HEAD_DTYPE={lm_head_dtype!r}; "
                f"expected one of {sorted(VALID_SAIL_LM_HEAD_DTYPES)}"
            )
        ar_fuse_mhc_post = _sail_int("ZSTEP_SAIL_AR_FUSE_MHC_POST", "ar_fuse_mhc_post", 0)
        if ar_fuse_mhc_post not in (0, 1):
            raise ValueError("sail_ar_fuse_mhc_post must be 0 or 1")
        moe_router_mode = _sail_str("ZSTEP_SAIL_MOE_ROUTER_MODE", "moe_router_mode", "fp32").lower()
        if moe_router_mode not in VALID_SAIL_MOE_ROUTER_MODES:
            raise ValueError(
                f"Unsupported ZSTEP_SAIL_MOE_ROUTER_MODE={moe_router_mode!r}; "
                f"expected one of {sorted(VALID_SAIL_MOE_ROUTER_MODES)}"
            )
        mhc_backend = _sail_str("ZSTEP_SAIL_MHC_BACKEND", "mhc_backend", "legacy_2pass").lower()
        if mhc_backend not in VALID_SAIL_MHC_BACKENDS:
            raise ValueError(
                f"Unsupported ZSTEP_SAIL_MHC_BACKEND={mhc_backend!r}; "
                f"expected one of {sorted(VALID_SAIL_MHC_BACKENDS)}"
            )
        observe_path = env.get("ZSTEP_SAIL_OBSERVE_PATH", "").strip()

        return cls(
            policy_path=policy_path,
            ar_blocks=ar_blocks,
            ar_threads=ar_threads,
            ar_oneshot_max_bytes=ar_oneshot_max_bytes,
            ar_pccl_min_bytes=ar_pccl_min_bytes,
            moe_gemv_max_rows=moe_gemv_max_rows,
            moe_gemv_threads=moe_gemv_threads,
            moe_block_m=moe_block_m,
            moe_num_warps=moe_num_warps,
            moe_prefill_backend=moe_prefill_backend,
            sparse_attn_backend=sparse_attn_backend,
            mhc_split_k=mhc_split_k,
            mhc_fused_max_rows=mhc_fused_max_rows,
            qkv_fusion_mode=qkv_fusion_mode,
            engram_indexer_backend=engram_indexer_backend,
            lm_head_dtype=lm_head_dtype,
            ar_fuse_mhc_post=ar_fuse_mhc_post,
            moe_router_mode=moe_router_mode,
            mhc_backend=mhc_backend,
            observe_path=observe_path,
        )

    def format_log_fragment(self, backend: str) -> str:
        return (
            f"sail={backend}(ar={self.ar_blocks}x{self.ar_threads},"
            f"os={self.ar_oneshot_max_bytes},pccl={self.ar_pccl_min_bytes},"
            f"moe_gemv<={self.moe_gemv_max_rows}@{self.moe_gemv_threads},"
            f"bm={self.moe_block_m}w{self.moe_num_warps}:{self.moe_prefill_backend},"
            f"attn={self.sparse_attn_backend},"
            f"mhc={self.mhc_split_k}r{self.mhc_fused_max_rows},"
            f"qkv={self.qkv_fusion_mode},ei={self.engram_indexer_backend},"
            f"head={self.lm_head_dtype}) "
            f"sail_v4=(fmp={self.ar_fuse_mhc_post},"
            f"rtr={self.moe_router_mode},"
            f"mhc_bk={self.mhc_backend})"
        )
