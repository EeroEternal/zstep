# Copyright (c) 2026 zstep Contributors
# SPDX-License-Identifier: MIT

"""Unit tests for Host DRAM KV tier, NUMA mbind helper, and `sm_h2d` copier."""

from __future__ import annotations

import numpy as np
import pytest
import torch

from zstep_worker.env_config import WorkerRuntimeConfig
from zstep_worker.kernels_sm120.sm_h2d import SmH2dCopier
from zstep_worker.kv_dram_tier import (
    KV_TRANSFER_DIR_D2H,
    KV_TRANSFER_DIR_H2D,
    HostDramKvTier,
)
from zstep_worker.numa_pin import bind_memory_to_numa_node


def test_host_dram_kv_tier_d2h_and_h2d_roundtrip_multi_layer() -> None:
    total_hbm_blocks = 8
    num_dram_blocks = 4
    # Heterogeneous layer shapes mirroring DeepSeek-V4.1 (2x SWA, 1x MLA, 2x Indexer)
    swa = torch.randint(0, 256, (total_hbm_blocks * 2, 1, 8, 16), dtype=torch.uint8)
    mla = torch.randint(0, 256, (total_hbm_blocks, 1, 16, 16), dtype=torch.uint8)
    idx = torch.randint(0, 256, (total_hbm_blocks * 2, 64), dtype=torch.uint8)

    expected_swa_b2 = swa[4:6].clone()
    expected_mla_b2 = mla[2:3].clone()
    expected_idx_b2 = idx[4:6].clone()

    tier = HostDramKvTier(
        [swa, mla, idx],
        total_hbm_blocks=total_hbm_blocks,
        num_dram_blocks=num_dram_blocks,
        device=torch.device("cpu"),
        numa_node=-1,
    )
    assert tier.bytes_per_block == (2 * 8 * 16) + (16 * 16) + (2 * 64)

    # 1. Offload HBM logical block 2 -> DRAM slot 1
    tier.execute_transfers(
        np.array([2], dtype=np.int32),
        np.array([1], dtype=np.int32),
        np.array([KV_TRANSFER_DIR_D2H], dtype=np.int32),
        1,
    )
    assert tier.offload_ops == 1

    # Overwrite HBM block 2 and target block 5 with zeros to prove reload restores from DRAM
    swa[4:6].zero_()
    mla[2:3].zero_()
    idx[4:6].zero_()
    swa[10:12].zero_()
    mla[5:6].zero_()
    idx[10:12].zero_()

    # 2. Reload DRAM slot 1 -> HBM logical block 5 (simulating 10% Loading Area reload)
    tier.execute_transfers(
        np.array([5], dtype=np.int32),
        np.array([1], dtype=np.int32),
        np.array([KV_TRANSFER_DIR_H2D], dtype=np.int32),
        1,
    )
    assert tier.reload_ops == 1
    assert torch.equal(swa[10:12], expected_swa_b2)
    assert torch.equal(mla[5:6], expected_mla_b2)
    assert torch.equal(idx[10:12], expected_idx_b2)


def test_host_dram_kv_tier_rejects_out_of_bounds_and_invalid_direction() -> None:
    buf = torch.zeros((4, 32), dtype=torch.uint8)
    tier = HostDramKvTier(
        [buf],
        total_hbm_blocks=4,
        num_dram_blocks=2,
        device=torch.device("cpu"),
    )
    with pytest.raises(ValueError, match="hbm_page_id"):
        tier.execute_transfers(
            np.array([4], dtype=np.int32),
            np.array([0], dtype=np.int32),
            np.array([KV_TRANSFER_DIR_D2H], dtype=np.int32),
            1,
        )
    with pytest.raises(ValueError, match="dram_slot_id"):
        tier.execute_transfers(
            np.array([0], dtype=np.int32),
            np.array([2], dtype=np.int32),
            np.array([KV_TRANSFER_DIR_H2D], dtype=np.int32),
            1,
        )
    with pytest.raises(ValueError, match="direction"):
        tier.execute_transfers(
            np.array([0], dtype=np.int32),
            np.array([0], dtype=np.int32),
            np.array([9], dtype=np.int32),
            1,
        )


def test_sm_h2d_copier_int32_and_int64_fallback_on_host() -> None:
    copier = SmH2dCopier(enabled=True)
    src32 = torch.tensor([11, 22, 33, 44], dtype=torch.int32)
    dst32 = torch.zeros_like(src32)
    copier.copy_async(dst32, src32)
    assert torch.equal(dst32, src32)

    src64 = torch.tensor([100, 200, 300], dtype=torch.int64)
    dst64 = torch.zeros_like(src64)
    copier.copy_async(dst64, src64)
    assert torch.equal(dst64, src64)
    assert copier.fallback_copies == 2


def test_env_config_dram_kv_and_sm_h2d_flags() -> None:
    cfg = WorkerRuntimeConfig.from_env(
        {"ZSTEP_DRAM_KV_BLOCKS": "512", "ZSTEP_ENABLE_SM_H2D": "0"}
    )
    assert cfg.dram_kv_blocks == 512
    assert cfg.enable_sm_h2d is False

    with pytest.raises(ValueError, match="ZSTEP_DRAM_KV_BLOCKS"):
        WorkerRuntimeConfig.from_env({"ZSTEP_DRAM_KV_BLOCKS": "-1"})


def test_bind_memory_to_numa_node_handles_invalid_inputs_safely() -> None:
    assert bind_memory_to_numa_node(0, 4096, 0) is False
    assert bind_memory_to_numa_node(4096, 4096, -1) is False


def test_host_dram_kv_tier_rejects_non_contiguous_buffer() -> None:
    non_contig = torch.zeros((4, 32), dtype=torch.uint8).t()
    with pytest.raises(ValueError, match="contiguous"):
        HostDramKvTier(
            [non_contig],
            total_hbm_blocks=4,
            num_dram_blocks=2,
            device=torch.device("cpu"),
        )


def test_host_dram_kv_tier_lossless_multi_plane_codec_0ulp_and_crc32_guard() -> None:
    from zstep_worker.kernels_sm120.kv_rans_codec import (
        decode_lossless_kv_packet,
        encode_lossless_kv_packet,
    )

    # Realistic 128-row DSv41 584B compound KV page: structured FP8 latent + BF16 RoPE + scales
    torch.manual_seed(20261128)
    page = torch.zeros((4, 128, 584), dtype=torch.uint8)
    # Simulated FP8 E4M3 Gaussian latent (clustered exponents around 120..128)
    fp8_vals = (torch.randn((4, 128, 512), dtype=torch.bfloat16) * 0.25).to(
        torch.float8_e4m3fn
    )
    page[:, :, :512] = fp8_vals.view(torch.uint8)
    page[:, :, 512:520] = 125  # UE8M0 / E4M3 block scales
    rope_bf16 = (torch.randn((4, 128, 32), dtype=torch.bfloat16) * 0.5).view(torch.uint8)
    page[:, :, 520:584] = rope_bf16
    expected_b1 = page[1].clone()

    tier = HostDramKvTier(
        [page],
        total_hbm_blocks=4,
        num_dram_blocks=2,
        device=torch.device("cpu"),
        codec="lossless",
    )
    tier.execute_transfers(
        np.array([1], dtype=np.int32),
        np.array([0], dtype=np.int32),
        np.array([KV_TRANSFER_DIR_D2H], dtype=np.int32),
        1,
    )
    assert tier.offload_ops == 1
    assert tier.compression_ratio > 1.25
    assert tier.last_batch_mode == "lossless_zkv1"

    page[1].zero_()
    page[3].zero_()
    tier.execute_transfers(
        np.array([3], dtype=np.int32),
        np.array([0], dtype=np.int32),
        np.array([KV_TRANSFER_DIR_H2D], dtype=np.int32),
        1,
    )
    assert tier.reload_ops == 1
    assert tier.crc32_verified_blocks == 1
    assert torch.equal(page[3], expected_b1)  # Exact 0-ULP bit-for-bit equality

    # Verify IEEE 802.3 CRC32 rejects single-bit corruption
    raw = bytes(expected_b1.numpy().tobytes())
    pkt, stats = encode_lossless_kv_packet(raw, rows=128, row_stride=584)
    assert stats.compression_ratio > 1.25
    assert decode_lossless_kv_packet(pkt, rows=128, row_stride=584) == raw
    corrupted = bytearray(pkt)
    corrupted[-1] ^= 0x01
    with pytest.raises(ValueError, match="CRC32|Error"):
        decode_lossless_kv_packet(bytes(corrupted), rows=128, row_stride=584)
