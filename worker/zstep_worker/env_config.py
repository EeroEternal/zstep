# Copyright (c) 2026 zstep Contributors
# SPDX-License-Identifier: MIT

"""Centralized, typed parser for `ZSTEP_*` worker runtime environment variables.

Every runtime knob read during worker startup, P2P probing, or SM120 kernel setup
is declared and validated here so effective values can be logged once at startup
and audited without grepping scattered `os.environ.get(...)` calls.
"""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass
import os

VALID_SPEC_METHODS: frozenset[str] = frozenset({"none", "ngram", "dspark", "mtp"})
VALID_P2P_MODES: frozenset[str] = frozenset(
    {"auto", "0", "off", "false", "disable", "disabled", "force"}
)
VALID_QWEN_GDN_DECODE_MODES: tuple[str, ...] = (
    "auto",
    "cluster2",
    "split_2op",
    "split_3op",
)
DEFAULT_MIN_PXB_BI_GBPS: float = 25.0


def env_flag(
    name: str,
    *,
    default: bool = True,
    environ: Mapping[str, str] | None = None,
) -> bool:
    """Parse a boolean `ZSTEP_*` flag (`0`/`false`/`no`/`off` -> `False`)."""
    env = os.environ if environ is None else environ
    raw = env.get(name)
    if raw is None:
        return default
    return raw.strip().lower() not in ("0", "false", "no", "off")


def env_int(
    name: str,
    default: int,
    *,
    environ: Mapping[str, str] | None = None,
) -> int:
    """Parse an integer `ZSTEP_*` variable, falling back to `default` if unset/blank."""
    env = os.environ if environ is None else environ
    raw = env.get(name)
    if raw is None or not raw.strip():
        return default
    return int(raw.strip())


def env_optional_int(
    name: str,
    *,
    environ: Mapping[str, str] | None = None,
) -> int | None:
    """Parse an optional integer `ZSTEP_*` override (`None` if unset/blank)."""
    env = os.environ if environ is None else environ
    raw = env.get(name)
    if raw is None or not raw.strip():
        return None
    return int(raw.strip())


def env_float(
    name: str,
    default: float,
    *,
    environ: Mapping[str, str] | None = None,
) -> float:
    """Parse a float `ZSTEP_*` variable, falling back to `default` if unset/blank."""
    env = os.environ if environ is None else environ
    raw = env.get(name)
    if raw is None or not raw.strip():
        return default
    return float(raw.strip())


def parse_qwen_fused_gdn_enabled(environ: Mapping[str, str] | None = None) -> bool:
    """Return whether `ZSTEP_QWEN_ENABLE_FUSED_GDN` is enabled (default `True`)."""
    return env_flag("ZSTEP_QWEN_ENABLE_FUSED_GDN", default=True, environ=environ)


def parse_qwen_gdn_decode_mode(environ: Mapping[str, str] | None = None) -> str:
    """Parse and validate `ZSTEP_QWEN_GDN_DECODE_MODE`."""
    env = os.environ if environ is None else environ
    mode_cfg = env.get("ZSTEP_QWEN_GDN_DECODE_MODE", "auto").strip() or "auto"
    if mode_cfg not in VALID_QWEN_GDN_DECODE_MODES:
        raise ValueError(
            f"Unsupported ZSTEP_QWEN_GDN_DECODE_MODE={mode_cfg!r}; "
            f"expected one of {VALID_QWEN_GDN_DECODE_MODES!r}"
        )
    return mode_cfg


def parse_pxb_bfp_mode(environ: Mapping[str, str] | None = None) -> int:
    """Parse `ZSTEP_SHM_ENABLE_PXB_BFP` (`0`/`off`, `1`/`on` hybrid, or `2` Stage-1-only)."""
    env = os.environ if environ is None else environ
    raw = env.get("ZSTEP_SHM_ENABLE_PXB_BFP")
    if raw is None or not raw.strip():
        return 1
    token = raw.strip().lower()
    if token in ("0", "false", "no", "off"):
        return 0
    if token in ("1", "true", "yes", "on"):
        return 1
    if token == "2":
        return 2
    raise ValueError(
        f"Unsupported ZSTEP_SHM_ENABLE_PXB_BFP={raw!r}; expected 0/false/off, 1/true/on, or 2"
    )


@dataclass(frozen=True)
class WorkerRuntimeConfig:
    """Typed snapshot of effective `ZSTEP_*` worker runtime configuration."""

    # Serving / Speculative Decoding / Context / APC
    spec_method: str = "none"
    num_spec_tokens: int = 0
    max_model_len: int = 32768
    num_recurrent_checkpoints: int = 0
    dram_kv_blocks: int = 0
    swa_kv_pages: int = 0
    enable_sm_h2d: bool = True
    enable_driver_batch_memcpy: bool = True
    prefill_runahead_depth: int = 2
    disable_verify_graphs: bool = False
    disable_draft_graphs: bool = False

    # Collectives & PCIe Topology
    p2p_mode: str = "auto"
    p2p_min_pxb_bi_gbps: float = DEFAULT_MIN_PXB_BI_GBPS
    shm_enable_pxb_p2p: bool = True
    shm_pxb_bfp_mode: int = 1
    shm_max_ar_numel: int | None = None
    shm_max_ag_numel: int | None = None

    # SM120 / `b12x` / DeepSeek-V4.1
    enable_b12x: bool = True
    explicit_enable_b12x: bool = False
    require_b12x: bool = False
    require_sm120_exp: bool = False
    enable_sm120_dsa_decode: bool = True
    enable_topk_canon: bool = True
    enable_sm12x_mega_gate: bool = True
    mega_gate_min_tokens: int = 1
    mega_gate_enable_pdl: bool = False
    enable_sm120_mhc_overlap: bool = True
    enable_sm120_fuse_mhc_ar: bool = True
    b12x_compat_dir: str = ""
    b12x_cache_dir: str | None = None
    b12x_autotune: bool = True

    # Qwen3.5 Gated DeltaNet
    qwen_enable_fused_gdn: bool = True
    qwen_gdn_decode_mode: str = "auto"

    @classmethod
    def from_env(cls, environ: Mapping[str, str] | None = None) -> WorkerRuntimeConfig:
        env = os.environ if environ is None else environ

        spec_method = env.get("ZSTEP_SPEC_METHOD", "none").strip() or "none"
        if spec_method not in VALID_SPEC_METHODS:
            raise ValueError(
                f"Unsupported ZSTEP_SPEC_METHOD={spec_method!r}; "
                f"expected one of {sorted(VALID_SPEC_METHODS)}"
            )
        num_spec_tokens = env_int("ZSTEP_NUM_SPEC_TOKENS", 0, environ=env)
        if spec_method != "none" and not 1 <= num_spec_tokens <= 7:
            raise ValueError("ZSTEP_NUM_SPEC_TOKENS must be in 1..=7")
        if spec_method == "none" and not 0 <= num_spec_tokens <= 7:
            raise ValueError("ZSTEP_NUM_SPEC_TOKENS must be in 0..=7")

        max_model_len = env_int("ZSTEP_MAX_MODEL_LEN", 32768, environ=env)
        if max_model_len <= 0:
            raise ValueError(f"ZSTEP_MAX_MODEL_LEN must be > 0 (got {max_model_len})")

        num_recurrent_checkpoints = env_int(
            "ZSTEP_NUM_RECURRENT_CHECKPOINTS", 0, environ=env
        )
        if not 0 <= num_recurrent_checkpoints <= 255:
            raise ValueError(
                "ZSTEP_NUM_RECURRENT_CHECKPOINTS must be in 0..=255 "
                f"(got {num_recurrent_checkpoints})"
            )

        dram_kv_blocks = env_int("ZSTEP_DRAM_KV_BLOCKS", 0, environ=env)
        if dram_kv_blocks < 0:
            raise ValueError(f"ZSTEP_DRAM_KV_BLOCKS must be >= 0 (got {dram_kv_blocks})")
        swa_kv_pages = env_int("ZSTEP_SWA_KV_PAGES", 0, environ=env)
        if swa_kv_pages < 0:
            raise ValueError(f"ZSTEP_SWA_KV_PAGES must be >= 0 (got {swa_kv_pages})")
        enable_sm_h2d = env_flag("ZSTEP_ENABLE_SM_H2D", default=True, environ=env)
        enable_driver_batch_memcpy = env_flag(
            "ZSTEP_ENABLE_DRIVER_BATCH_MEMCPY", default=True, environ=env
        )
        prefill_runahead_depth = env_int("ZSTEP_PREFILL_RUNAHEAD_DEPTH", 2, environ=env)
        if prefill_runahead_depth < 0:
            raise ValueError(
                f"ZSTEP_PREFILL_RUNAHEAD_DEPTH must be >= 0 (got {prefill_runahead_depth})"
            )

        p2p_mode = env.get("ZSTEP_P2P_MODE", "auto").strip().lower() or "auto"
        if p2p_mode not in VALID_P2P_MODES:
            raise ValueError(
                f"Unsupported ZSTEP_P2P_MODE={p2p_mode!r}; "
                f"expected one of {sorted(VALID_P2P_MODES)}"
            )
        p2p_min_pxb_bi_gbps = env_float(
            "ZSTEP_P2P_MIN_PXB_BI_GBPS", DEFAULT_MIN_PXB_BI_GBPS, environ=env
        )
        if p2p_min_pxb_bi_gbps < 0.0:
            raise ValueError(
                "ZSTEP_P2P_MIN_PXB_BI_GBPS must be >= 0.0 "
                f"(got {p2p_min_pxb_bi_gbps})"
            )

        shm_max_ar_numel = env_optional_int("ZSTEP_SHM_MAX_AR_NUMEL", environ=env)
        if shm_max_ar_numel is not None and shm_max_ar_numel < 0:
            raise ValueError(
                f"ZSTEP_SHM_MAX_AR_NUMEL must be >= 0 (got {shm_max_ar_numel})"
            )
        shm_max_ag_numel = env_optional_int("ZSTEP_SHM_MAX_AG_NUMEL", environ=env)
        if shm_max_ag_numel is not None and shm_max_ag_numel < 0:
            raise ValueError(
                f"ZSTEP_SHM_MAX_AG_NUMEL must be >= 0 (got {shm_max_ag_numel})"
            )

        raw_cache_dir = env.get("ZSTEP_B12X_CACHE_DIR")
        b12x_cache_dir = (
            raw_cache_dir.strip()
            if raw_cache_dir is not None and raw_cache_dir.strip()
            else None
        )

        return cls(
            spec_method=spec_method,
            num_spec_tokens=num_spec_tokens,
            max_model_len=max_model_len,
            num_recurrent_checkpoints=num_recurrent_checkpoints,
            dram_kv_blocks=dram_kv_blocks,
            swa_kv_pages=swa_kv_pages,
            enable_sm_h2d=enable_sm_h2d,
            enable_driver_batch_memcpy=enable_driver_batch_memcpy,
            prefill_runahead_depth=prefill_runahead_depth,
            disable_verify_graphs=env_flag(
                "ZSTEP_DISABLE_VERIFY_GRAPHS", default=False, environ=env
            ),
            disable_draft_graphs=env_flag(
                "ZSTEP_DISABLE_DRAFT_GRAPHS", default=False, environ=env
            ),
            p2p_mode=p2p_mode,
            p2p_min_pxb_bi_gbps=p2p_min_pxb_bi_gbps,
            shm_enable_pxb_p2p=env_flag(
                "ZSTEP_SHM_ENABLE_PXB_P2P", default=True, environ=env
            ),
            shm_pxb_bfp_mode=parse_pxb_bfp_mode(env),
            shm_max_ar_numel=shm_max_ar_numel,
            shm_max_ag_numel=shm_max_ag_numel,
            enable_b12x=env_flag("ZSTEP_ENABLE_B12X", default=True, environ=env),
            explicit_enable_b12x="ZSTEP_ENABLE_B12X" in env,
            require_b12x=env_flag("ZSTEP_REQUIRE_B12X", default=False, environ=env),
            require_sm120_exp=env_flag(
                "ZSTEP_REQUIRE_SM120_EXP", default=False, environ=env
            ),
            enable_sm120_dsa_decode=env_flag(
                "ZSTEP_ENABLE_SM120_DSA_DECODE", default=True, environ=env
            ),
            enable_topk_canon=env_flag(
                "ZSTEP_ENABLE_TOPK_CANON", default=True, environ=env
            ),
            enable_sm12x_mega_gate=env_flag(
                "ZSTEP_ENABLE_SM12X_MEGA_GATE", default=True, environ=env
            ),
            mega_gate_min_tokens=env_int(
                "ZSTEP_MEGA_GATE_MIN_TOKENS", 1, environ=env
            ),
            mega_gate_enable_pdl=env_flag(
                "ZSTEP_MEGA_GATE_ENABLE_PDL", default=False, environ=env
            ),
            enable_sm120_mhc_overlap=env_flag(
                "ZSTEP_ENABLE_SM120_MHC_OVERLAP", default=True, environ=env
            ),
            enable_sm120_fuse_mhc_ar=env_flag(
                "ZSTEP_ENABLE_SM120_FUSE_MHC_AR", default=True, environ=env
            ),
            b12x_compat_dir=env.get("ZSTEP_B12X_COMPAT_DIR", "").strip(),
            b12x_cache_dir=b12x_cache_dir,
            b12x_autotune=env_flag("ZSTEP_B12X_AUTOTUNE", default=True, environ=env),
            qwen_enable_fused_gdn=parse_qwen_fused_gdn_enabled(env),
            qwen_gdn_decode_mode=parse_qwen_gdn_decode_mode(env),
        )

    def effective_shm_max_ar_numel(self, default_numel: int) -> int:
        return default_numel if self.shm_max_ar_numel is None else self.shm_max_ar_numel

    def effective_shm_max_ag_numel(self, default_numel: int) -> int:
        return default_numel if self.shm_max_ag_numel is None else self.shm_max_ag_numel

    def format_log_line(self) -> str:
        return (
            f"zstep_worker_config spec={self.spec_method}:{self.num_spec_tokens} "
            f"max_model_len={self.max_model_len} "
            f"recurrent_ckpts={self.num_recurrent_checkpoints} "
            f"p2p_mode={self.p2p_mode} pxb_shm={int(self.shm_enable_pxb_p2p)} "
            f"pxb_bfp={self.shm_pxb_bfp_mode} "
            f"min_pxb_gbps={self.p2p_min_pxb_bi_gbps:.1f} "
            f"shm_max_ar={self.shm_max_ar_numel if self.shm_max_ar_numel is not None else 'auto'} "
            f"shm_max_ag={self.shm_max_ag_numel if self.shm_max_ag_numel is not None else 'auto'} "
            f"b12x={int(self.enable_b12x)}(req={int(self.require_b12x)},autotune={int(self.b12x_autotune)}) "
            f"sm120_exp_req={int(self.require_sm120_exp)} "
            f"dsa_decode={int(self.enable_sm120_dsa_decode)} "
            f"mega_gate={int(self.enable_sm12x_mega_gate)}(min_m={self.mega_gate_min_tokens},pdl={int(self.mega_gate_enable_pdl)}) "
            f"mhc_overlap={int(self.enable_sm120_mhc_overlap)}(fuse_ar={int(self.enable_sm120_fuse_mhc_ar)}) "
            f"qwen_gdn={int(self.qwen_enable_fused_gdn)}:{self.qwen_gdn_decode_mode} "
            f"verify_graphs={int(not self.disable_verify_graphs)} "
            f"draft_graphs={int(not self.disable_draft_graphs)} "
            f"prefill_runahead={self.prefill_runahead_depth} "
            f"driver_batch_memcpy={int(self.enable_driver_batch_memcpy)}"
        )
