# Copyright (c) 2026 zstep Contributors
# SPDX-License-Identifier: MIT

from __future__ import annotations

import pytest

from zstep_worker.env_config import (
    DEFAULT_MIN_PXB_BI_GBPS,
    WorkerRuntimeConfig,
    parse_pxb_bfp_mode,
    parse_qwen_fused_gdn_enabled,
    parse_qwen_gdn_decode_mode,
)


def test_worker_runtime_config_defaults_and_summary() -> None:
    cfg = WorkerRuntimeConfig.from_env({})
    assert cfg.spec_method == "none"
    assert cfg.num_spec_tokens == 0
    assert cfg.max_model_len == 32768
    assert cfg.num_recurrent_checkpoints == 0
    assert cfg.disable_verify_graphs is False
    assert cfg.disable_draft_graphs is False
    assert cfg.p2p_mode == "auto"
    assert cfg.p2p_min_pxb_bi_gbps == DEFAULT_MIN_PXB_BI_GBPS
    assert cfg.shm_enable_pxb_p2p is True
    assert cfg.shm_pxb_bfp_mode == 1
    assert cfg.effective_shm_max_ar_numel(1234) == 1234
    assert cfg.effective_shm_max_ag_numel(5678) == 5678
    assert cfg.enable_b12x is True
    assert cfg.explicit_enable_b12x is False
    assert cfg.require_b12x is False
    assert cfg.require_sm120_exp is False
    assert cfg.enable_sm120_dsa_decode is True
    assert cfg.enable_sm120_candidate_mqa is True
    assert cfg.enable_sm12x_mega_gate is True
    assert cfg.mega_gate_min_tokens == 1
    assert cfg.mega_gate_enable_pdl is False
    assert cfg.enable_sm120_mhc_overlap is True
    assert cfg.enable_sm120_fuse_mhc_ar is True
    assert cfg.b12x_compat_dir == ""
    assert cfg.b12x_cache_dir is None
    assert cfg.b12x_autotune is True
    assert cfg.qwen_enable_fused_gdn is True
    assert cfg.qwen_gdn_decode_mode == "auto"
    assert cfg.enable_driver_batch_memcpy is True
    assert cfg.prefill_runahead_depth == 2
    assert cfg.swa_kv_pages == 0

    line = cfg.format_log_line()
    assert line.startswith("zstep_worker_config ")
    assert "spec=none:0" in line
    assert "pxb_bfp=1" in line
    assert "mhc_overlap=1(fuse_ar=1)" in line
    assert "shm_max_ar=auto" in line
    assert "prefill_runahead=2" in line
    assert "driver_batch_memcpy=1" in line


def test_worker_runtime_config_overrides_and_helpers() -> None:
    env = {
        "ZSTEP_SPEC_METHOD": "dspark",
        "ZSTEP_NUM_SPEC_TOKENS": "3",
        "ZSTEP_MAX_MODEL_LEN": "16384",
        "ZSTEP_NUM_RECURRENT_CHECKPOINTS": "64",
        "ZSTEP_DISABLE_VERIFY_GRAPHS": "1",
        "ZSTEP_DISABLE_DRAFT_GRAPHS": "true",
        "ZSTEP_P2P_MODE": "OFF",
        "ZSTEP_P2P_MIN_PXB_BI_GBPS": "55.5",
        "ZSTEP_SHM_ENABLE_PXB_P2P": "0",
        "ZSTEP_SHM_ENABLE_PXB_BFP": "2",
        "ZSTEP_SHM_MAX_AR_NUMEL": "65536",
        "ZSTEP_SHM_MAX_AG_NUMEL": "131072",
        "ZSTEP_ENABLE_B12X": "off",
        "ZSTEP_REQUIRE_B12X": "1",
        "ZSTEP_REQUIRE_SM120_EXP": "1",
        "ZSTEP_ENABLE_SM120_DSA_DECODE": "false",
        "ZSTEP_ENABLE_SM120_CANDIDATE_MQA": "0",
        "ZSTEP_ENABLE_SM12X_MEGA_GATE": "no",
        "ZSTEP_MEGA_GATE_MIN_TOKENS": "8",
        "ZSTEP_MEGA_GATE_ENABLE_PDL": "1",
        "ZSTEP_ENABLE_SM120_MHC_OVERLAP": "0",
        "ZSTEP_ENABLE_SM120_FUSE_MHC_AR": "0",
        "ZSTEP_B12X_COMPAT_DIR": " /opt/b12x ",
        "ZSTEP_B12X_CACHE_DIR": " /tmp/b12x_cache ",
        "ZSTEP_B12X_AUTOTUNE": "0",
        "ZSTEP_QWEN_ENABLE_FUSED_GDN": "0",
        "ZSTEP_QWEN_GDN_DECODE_MODE": "cluster2",
        "ZSTEP_ENABLE_DRIVER_BATCH_MEMCPY": "0",
        "ZSTEP_PREFILL_RUNAHEAD_DEPTH": "4",
        "ZSTEP_SWA_KV_PAGES": "2048",
    }
    cfg = WorkerRuntimeConfig.from_env(env)
    assert cfg.spec_method == "dspark"
    assert cfg.num_spec_tokens == 3
    assert cfg.max_model_len == 16384
    assert cfg.num_recurrent_checkpoints == 64
    assert cfg.disable_verify_graphs is True
    assert cfg.disable_draft_graphs is True
    assert cfg.p2p_mode == "off"
    assert cfg.p2p_min_pxb_bi_gbps == 55.5
    assert cfg.shm_enable_pxb_p2p is False
    assert cfg.shm_pxb_bfp_mode == 2
    assert parse_pxb_bfp_mode({"ZSTEP_SHM_ENABLE_PXB_BFP": "false"}) == 0
    assert parse_pxb_bfp_mode({"ZSTEP_SHM_ENABLE_PXB_BFP": "true"}) == 1
    assert cfg.effective_shm_max_ar_numel(1234) == 65536
    assert cfg.effective_shm_max_ag_numel(5678) == 131072
    assert cfg.enable_b12x is False
    assert cfg.explicit_enable_b12x is True
    assert cfg.require_b12x is True
    assert cfg.require_sm120_exp is True
    assert cfg.enable_sm120_dsa_decode is False
    assert cfg.enable_sm120_candidate_mqa is False
    assert cfg.enable_sm12x_mega_gate is False
    assert cfg.mega_gate_min_tokens == 8
    assert cfg.mega_gate_enable_pdl is True
    assert cfg.enable_sm120_mhc_overlap is False
    assert cfg.enable_sm120_fuse_mhc_ar is False
    assert cfg.b12x_compat_dir == "/opt/b12x"
    assert cfg.b12x_cache_dir == "/tmp/b12x_cache"
    assert cfg.b12x_autotune is False
    assert parse_qwen_fused_gdn_enabled(env) is False
    assert parse_qwen_gdn_decode_mode(env) == "cluster2"
    assert cfg.enable_driver_batch_memcpy is False
    assert cfg.prefill_runahead_depth == 4
    assert cfg.swa_kv_pages == 2048
    assert cfg.prefill_tbo_min_tokens == 6144
    assert cfg.prefill_tbo_min_ctx == 0
    assert cfg.prefill_tbo_ar_blocks == 24

    tbo_cfg = WorkerRuntimeConfig.from_env(
        {
            "ZSTEP_PREFILL_TBO_MIN_TOKENS": "0",
            "ZSTEP_PREFILL_TBO_MIN_CTX": "65536",
            "ZSTEP_PREFILL_TBO_AR_BLOCKS": "16",
            "ZSTEP_PIPELINE_PARALLEL_SIZE": "2",
            "ZSTEP_PP_PREFILL_MICROBATCHES": "32",
            "ZSTEP_PP_ENGRAM_GPU_LAYERS": "1",
        }
    )
    assert tbo_cfg.prefill_tbo_min_tokens == 0
    assert tbo_cfg.prefill_tbo_min_ctx == 65536
    assert tbo_cfg.prefill_tbo_ar_blocks == 16
    assert tbo_cfg.pipeline_parallel_size == 2
    assert tbo_cfg.pp_prefill_microbatches == 32
    assert tbo_cfg.pp_engram_gpu_layers == 1


@pytest.mark.parametrize(
    ("env", "match"),
    [
        ({"ZSTEP_SPEC_METHOD": "medusa"}, "ZSTEP_SPEC_METHOD"),
        ({"ZSTEP_PREFILL_RUNAHEAD_DEPTH": "-1"}, "ZSTEP_PREFILL_RUNAHEAD_DEPTH"),
        ({"ZSTEP_SPEC_METHOD": "ngram", "ZSTEP_NUM_SPEC_TOKENS": "0"}, "1..=7"),
        ({"ZSTEP_SPEC_METHOD": "mtp", "ZSTEP_NUM_SPEC_TOKENS": "8"}, "1..=7"),
        ({"ZSTEP_MAX_MODEL_LEN": "0"}, "ZSTEP_MAX_MODEL_LEN"),
        ({"ZSTEP_NUM_RECURRENT_CHECKPOINTS": "256"}, "ZSTEP_NUM_RECURRENT_CHECKPOINTS"),
        ({"ZSTEP_P2P_MODE": "bogus"}, "ZSTEP_P2P_MODE"),
        ({"ZSTEP_P2P_MIN_PXB_BI_GBPS": "-1.0"}, "ZSTEP_P2P_MIN_PXB_BI_GBPS"),
        ({"ZSTEP_SHM_ENABLE_PXB_BFP": "3"}, "ZSTEP_SHM_ENABLE_PXB_BFP"),
        ({"ZSTEP_SHM_ENABLE_PXB_BFP": "bogus"}, "ZSTEP_SHM_ENABLE_PXB_BFP"),
        ({"ZSTEP_SHM_MAX_AR_NUMEL": "-4"}, "ZSTEP_SHM_MAX_AR_NUMEL"),
        ({"ZSTEP_QWEN_GDN_DECODE_MODE": "invalid_mode"}, "ZSTEP_QWEN_GDN_DECODE_MODE"),
        ({"ZSTEP_SWA_KV_PAGES": "-1"}, "ZSTEP_SWA_KV_PAGES"),
        ({"ZSTEP_PREFILL_TBO_MIN_TOKENS": "-1"}, "ZSTEP_PREFILL_TBO_MIN_TOKENS"),
        ({"ZSTEP_PREFILL_TBO_MIN_CTX": "-1"}, "ZSTEP_PREFILL_TBO_MIN_CTX"),
        ({"ZSTEP_PREFILL_TBO_AR_BLOCKS": "0"}, "ZSTEP_PREFILL_TBO_AR_BLOCKS"),
        ({"ZSTEP_PREFILL_TBO_AR_BLOCKS": "65"}, "ZSTEP_PREFILL_TBO_AR_BLOCKS"),
        ({"ZSTEP_PIPELINE_PARALLEL_SIZE": "0"}, "ZSTEP_PIPELINE_PARALLEL_SIZE"),
        ({"ZSTEP_PP_PREFILL_MICROBATCHES": "0"}, "ZSTEP_PP_PREFILL_MICROBATCHES"),
        ({"ZSTEP_PP_ENGRAM_GPU_LAYERS": "-1"}, "ZSTEP_PP_ENGRAM_GPU_LAYERS"),
    ],
)
def test_worker_runtime_config_rejects_invalid_values(
    env: dict[str, str], match: str
) -> None:
    with pytest.raises(ValueError, match=match):
        WorkerRuntimeConfig.from_env(env)
