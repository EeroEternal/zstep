# Copyright (c) 2026 zstep Contributors
# SPDX-License-Identifier: MIT

from __future__ import annotations

import os
from pathlib import Path
from unittest.mock import patch

import pytest
import torch
import torch.distributed as dist
import torch.multiprocessing as mp

from zstep_worker.p2p_probe import (
    _CudaIpcMemHandle,
    classify_pci_pair_relation,
    configure_adaptive_p2p_env,
)


def test_cuda_ipc_mem_handle_preserves_embedded_null_bytes() -> None:
    raw = b"\x00\x01\x00\xff" * 16
    h = _CudaIpcMemHandle.from_buffer_copy(raw)
    assert len(bytes(h.reserved)) == 64
    assert bytes(h.reserved) == raw


def test_shm_allreduce_partial_init_cleanup_preserves_exception(tmp_path: Path) -> None:
    from zstep_worker.kernels_sm120 import shm_allreduce as sa

    fake_ext = object()
    with (
        patch.object(torch.cuda, "current_device", return_value=0),
        patch.object(torch.cuda, "synchronize", return_value=None),
        patch.object(sa, "pin_process_to_device_numa", return_value=object()),
        patch.object(sa, "get_host_mapped_allreduce_extension", return_value=fake_ext),
    ):
        # Legacy path (`exchange_group=None`, `create_shm=False`) missing SHM must raise
        # `FileNotFoundError`, not `AttributeError` on `self._mapping`.
        with pytest.raises(FileNotFoundError):
            sa.HostMappedShmAllReduce(
                name=f"nonexistent_legacy_{os.getpid()}",
                rank=0,
                world_size=2,
                capacity_elements=64,
                blocks=1,
                threads=32,
                exchange_group=None,
                create_shm=False,
            )

        # `shm_v5` path failing midway through `_init_per_rank_shm` must clean up `_own_shm_path`.
        leaked_file = tmp_path / "partial_r0"

        def _failing_init_per_rank_shm(self, **_kwargs) -> None:
            leaked_file.write_bytes(b"0" * 64)
            self._own_shm_path = str(leaked_file)
            self._per_rank_mappings = []
            self._per_rank_host_addrs = []
            raise RuntimeError("simulated mid-init barrier failure")

        with (
            patch.object(sa.HostMappedShmAllReduce, "_init_per_rank_shm", _failing_init_per_rank_shm),
            pytest.raises(RuntimeError, match="simulated mid-init barrier failure"),
        ):
            sa.HostMappedShmAllReduce(
                name="sim_v5",
                rank=0,
                world_size=2,
                capacity_elements=64,
                blocks=1,
                threads=32,
                exchange_group=object(),  # type: ignore[arg-type]
            )
        assert not leaked_file.exists()


def test_classify_pci_pair_relation_matches_cx8_6000d_hierarchy() -> None:
    gpu0 = "/sys/devices/pci0000:16/0000:16:01.0/0000:17:00.0/0000:18:00.0/0000:19:00.0/0000:1a:00.0/0000:1b:00.0"
    gpu1 = "/sys/devices/pci0000:16/0000:16:01.0/0000:17:00.0/0000:18:00.0/0000:19:00.0/0000:1d:00.0/0000:1e:00.0"
    gpu2 = "/sys/devices/pci0000:28/0000:28:01.0/0000:29:00.0/0000:2a:00.0/0000:2b:00.0/0000:2c:00.0"
    gpu_phb = "/sys/devices/pci0000:16/0000:16:02.0/0000:20:00.0"
    gpu4 = "/sys/devices/pci0000:98/0000:98:01.0/0000:99:00.0/0000:9c:00.0/0000:9d:00.0"
    nic0 = "/sys/devices/pci0000:16/0000:16:01.0/0000:17:00.0/0000:18:01.0/0000:1f:00.0"
    nic1 = "/sys/devices/pci0000:16/0000:16:01.0/0000:17:00.0/0000:18:01.0/0000:1f:00.1"

    assert classify_pci_pair_relation(gpu0, 0, gpu1, 0) == "PXB"
    assert classify_pci_pair_relation(gpu0, 0, gpu2, 0) == "NODE"
    assert classify_pci_pair_relation(gpu0, 0, gpu_phb, 0) == "PHB"
    assert classify_pci_pair_relation(gpu0, 0, gpu4, 1) == "SYS"
    assert classify_pci_pair_relation(nic0, 0, nic1, 0) == "PIX"


def test_configure_adaptive_p2p_env_respects_off_and_alert_file(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    for key in list(os.environ.keys()):
        if key.startswith("NCCL_") or key.startswith("ZSTEP_P2P"):
            monkeypatch.delenv(key, raising=False)

    monkeypatch.setenv("ZSTEP_P2P_MODE", "off")
    rep_off = configure_adaptive_p2p_env(rank=0, world_size=8, device_index=0)
    assert not rep_off.p2p_enabled
    assert not rep_off.pxb_shm_enabled
    assert os.environ["NCCL_P2P_DISABLE"] == "1"
    assert os.environ["NCCL_P2P_LEVEL"] == "LOC"
    assert os.environ["NCCL_MIN_NCHANNELS"] == "6"

    for key in list(os.environ.keys()):
        if key.startswith("NCCL_") or key.startswith("ZSTEP_P2P"):
            monkeypatch.delenv(key, raising=False)
    alert = tmp_path / "6000d-acs-off.ALERT"
    alert.write_text("ACS failed", encoding="utf-8")
    rep_alert = configure_adaptive_p2p_env(
        rank=0, world_size=8, device_index=0, alert_file=alert
    )
    assert not rep_alert.p2p_enabled
    assert rep_alert.reason == "acs_alert_file_present"
    assert os.environ["NCCL_P2P_DISABLE"] == "1"


def test_configure_adaptive_p2p_env_selects_pxb_phb_and_rejects_slow_acs(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    fake_sysfs = [
        ("/sys/devices/pci0000:16/0000:16:01.0/0000:19:00.0/0000:1a:00.0/0000:1b:00.0", 0),
        ("/sys/devices/pci0000:16/0000:16:01.0/0000:19:00.0/0000:1d:00.0/0000:1e:00.0", 0),
        ("/sys/devices/pci0000:28/0000:28:01.0/0000:2a:00.0/0000:2b:00.0/0000:2c:00.0", 0),
        ("/sys/devices/pci0000:28/0000:28:01.0/0000:2a:00.0/0000:2e:00.0/0000:2f:00.0", 0),
        ("/sys/devices/pci0000:98/0000:98:01.0/0000:9a:00.0/0000:9c:00.0/0000:9d:00.0", 1),
        ("/sys/devices/pci0000:98/0000:98:01.0/0000:9a:00.0/0000:9f:00.0/0000:a0:00.0", 1),
        ("/sys/devices/pci0000:a8/0000:a8:01.0/0000:aa:00.0/0000:ac:00.0/0000:ad:00.0", 1),
        ("/sys/devices/pci0000:a8/0000:a8:01.0/0000:aa:00.0/0000:af:00.0/0000:b0:00.0", 1),
    ]

    def reset_env() -> None:
        for key in list(os.environ.keys()):
            if key.startswith("NCCL_") or key.startswith("ZSTEP_P2P") or key.startswith("ZSTEP_SHM_ENABLE_PXB"):
                monkeypatch.delenv(key, raising=False)

    # 1) Slow PXB link (broken ACS ReqRedir+ ~8.09 GB/s) -> safe fallback to non-P2P SHM
    reset_env()
    with (
        patch("zstep_worker.p2p_probe._load_cudart", return_value=object()),
        patch(
            "zstep_worker.p2p_probe._query_peer_access_matrix",
            return_value=[[True] * 8 for _ in range(8)],
        ),
        patch("zstep_worker.p2p_probe._query_cuda_pci_bus_id", side_effect=lambda i: f"0000:0{i}:00.0"),
        patch("zstep_worker.p2p_probe._read_sysfs_pci_info", side_effect=lambda bdf: fake_sysfs[int(bdf[6])]),
        patch("zstep_worker.p2p_probe._probe_pxb_ipc_bandwidth_all_pairs", return_value=8.09),
    ):
        rep_slow = configure_adaptive_p2p_env(
            rank=0,
            world_size=8,
            device_index=0,
            dist_init_file=str(tmp_path / "init_slow"),
            alert_file=tmp_path / "no_alert",
        )
    assert not rep_slow.p2p_enabled
    assert not rep_slow.pxb_shm_enabled
    assert rep_slow.measured_pxb_bi_gbps == 8.09
    assert os.environ["NCCL_P2P_DISABLE"] == "1"
    assert os.environ["NCCL_P2P_LEVEL"] == "LOC"

    # 2) Healthy 2-NUMA CX8 TP=8 (~103.2 GB/s) -> PXB + pxb_shm_enabled=True, no 6-channel cap
    reset_env()
    with (
        patch("zstep_worker.p2p_probe._load_cudart", return_value=object()),
        patch(
            "zstep_worker.p2p_probe._query_peer_access_matrix",
            return_value=[[True] * 8 for _ in range(8)],
        ),
        patch("zstep_worker.p2p_probe._query_cuda_pci_bus_id", side_effect=lambda i: f"0000:0{i}:00.0"),
        patch("zstep_worker.p2p_probe._read_sysfs_pci_info", side_effect=lambda bdf: fake_sysfs[int(bdf[6])]),
        patch("zstep_worker.p2p_probe._probe_pxb_ipc_bandwidth_all_pairs", return_value=103.2),
    ):
        rep_ok = configure_adaptive_p2p_env(
            rank=0,
            world_size=8,
            device_index=0,
            dist_init_file=str(tmp_path / "init_ok"),
            alert_file=tmp_path / "no_alert",
        )
    assert rep_ok.p2p_enabled
    assert rep_ok.pxb_shm_enabled
    assert rep_ok.nccl_p2p_level == "PXB"
    assert os.environ["NCCL_P2P_DISABLE"] == "0"
    assert os.environ["NCCL_P2P_LEVEL"] == "PXB"
    assert "NCCL_MIN_NCHANNELS" not in os.environ

    # 3) Single-NUMA TP=4 -> PHB
    reset_env()
    with (
        patch("zstep_worker.p2p_probe._load_cudart", return_value=object()),
        patch(
            "zstep_worker.p2p_probe._query_peer_access_matrix",
            return_value=[[True] * 4 for _ in range(4)],
        ),
        patch("zstep_worker.p2p_probe._query_cuda_pci_bus_id", side_effect=lambda i: f"0000:0{i}:00.0"),
        patch("zstep_worker.p2p_probe._read_sysfs_pci_info", side_effect=lambda bdf: fake_sysfs[int(bdf[6])]),
        patch("zstep_worker.p2p_probe._probe_pxb_ipc_bandwidth_all_pairs", return_value=103.2),
    ):
        rep_tp4 = configure_adaptive_p2p_env(
            rank=0,
            world_size=4,
            device_index=0,
            dist_init_file=str(tmp_path / "init_tp4"),
            alert_file=tmp_path / "no_alert",
        )
    assert rep_tp4.p2p_enabled
    assert not rep_tp4.pxb_shm_enabled
    assert rep_tp4.nccl_p2p_level == "PHB"

    # 4) Asymmetric switch peer access (Switch 0 OK, Switch 1 blocked) -> unanimous non-P2P on all ranks
    asym_mat = [[True] * 8 for _ in range(8)]
    for i in range(8):
        for j in range(8):
            if i != j and not ({i, j} == {0, 1}):
                asym_mat[i][j] = False
    for r in (0, 2):
        reset_env()
        with (
            patch("zstep_worker.p2p_probe._load_cudart", return_value=object()),
            patch("zstep_worker.p2p_probe._query_peer_access_matrix", return_value=asym_mat),
            patch(
                "zstep_worker.p2p_probe._query_cuda_pci_bus_id",
                side_effect=lambda i: f"0000:0{i}:00.0",
            ),
            patch(
                "zstep_worker.p2p_probe._read_sysfs_pci_info",
                side_effect=lambda bdf: fake_sysfs[int(bdf[6])],
            ),
        ):
            rep_asym = configure_adaptive_p2p_env(
                rank=r,
                world_size=8,
                device_index=r,
                dist_init_file=str(tmp_path / f"init_asym_{r}"),
                alert_file=tmp_path / "no_alert",
            )
        assert not rep_asym.p2p_enabled
        assert not rep_asym.pxb_shm_enabled
        assert os.environ["NCCL_P2P_DISABLE"] == "1"
        assert os.environ["NCCL_P2P_LEVEL"] == "LOC"


def _pxb_8gpu_worker(rank: int, init_file: str, shm_name: str) -> None:
    from zstep_worker.kernels_sm120.shm_allreduce import HostMappedShmAllReduce

    for k in list(os.environ.keys()):
        if k.startswith("NCCL_P2P"):
            os.environ.pop(k, None)

    os.environ.setdefault("GLOO_SOCKET_IFNAME", "lo")
    torch.cuda.set_device(rank)
    device = torch.device(f"cuda:{rank}")
    report = configure_adaptive_p2p_env(
        rank=rank, world_size=8, device_index=rank, dist_init_file=init_file
    )
    assert report.p2p_enabled, f"Expected P2P enabled on 8-GPU P2P host, got {report}"
    assert report.pxb_shm_enabled, f"Expected pxb_shm_enabled on 8-GPU switch host, got {report}"

    dist.init_process_group(
        backend="gloo", init_method=f"file://{init_file}", rank=rank, world_size=8
    )
    try:
        configs = [
            (False, 8, 256),
            (True, 8, 256),
            (True, 32, 512),
        ]
        for cfg_idx, (use_pxb, blk, thr) in enumerate(configs):
            with HostMappedShmAllReduce(
                name=f"{shm_name}_{cfg_idx}",
                rank=rank,
                world_size=8,
                capacity_elements=1024 * 5120,
                blocks=blk,
                threads=thr,
                exchange_group=dist.group.WORLD,
                enable_pxb_p2p=use_pxb,
                pxb_max_elements=512 * 5120,
            ) as comm:
                assert comm.pxb_enabled is use_pxb
                # Rapidly alternate across all internal branches + dual-grid regimes:
                # M=1 (cluster8 / 1s) -> M=4 (pxb_1s / cluster8) -> M=8/24/64 (small_pxb 8x256)
                # -> M=96/256 (large_pxb 32x512) -> M=768 (shm_v5 > pxb_max)
                shapes = [
                    (1, 5120),
                    (4, 5120),
                    (8, 5120),
                    (24, 5120),
                    (64, 5120),
                    (96, 5120),
                    (256, 5120),
                    (768, 5120),
                    (16, 2048),
                    (128, 2048),
                    (256, 2048),
                ]
                large_steps = 0
                small_steps = 0
                for step_i in range(3):
                    for m, h in shapes:
                        num_vecs = (m * h) // 8
                        is_small_split = use_pxb and blk == 32 and num_vecs <= 40960
                        eff_blocks = 8 if is_small_split else blk

                        torch.manual_seed(500 + step_i * 97 + m * 13 + rank)
                        x = torch.randn((m, h), dtype=torch.bfloat16, device=device) * 0.125
                        ref_cpu = x.float().cpu()
                        dist.all_reduce(ref_cpu, group=dist.group.WORLD)
                        out = torch.empty_like(x)
                        comm.all_reduce(x, output=out)
                        if is_small_split:
                            small_steps += 1
                        else:
                            large_steps += 1
                        out_cpu = out.float().cpu()
                        diff = float((out_cpu - ref_cpu).abs().max().item())
                        assert diff < 0.03, (
                            f"all_reduce mismatch (pxb={use_pxb}, blk={blk}) at {(m, h)}: {diff}"
                        )

                        res_in = torch.randn((m, h), dtype=torch.bfloat16, device=device) * 0.1
                        res = res_in.clone()
                        exact_res_ref = (res_in.float().cpu() + out_cpu).to(torch.bfloat16)
                        res_ref = (res_in.float().cpu() + ref_cpu.to(torch.bfloat16).float()).to(torch.bfloat16)
                        w = torch.randn((h,), dtype=torch.bfloat16, device=device) * 0.05
                        norm_out = torch.empty_like(x)
                        comm.all_reduce_add_rmsnorm(
                            x, res, w, normed_out=norm_out, eps=1e-6, zero_centered=True
                        )
                        if is_small_split:
                            small_steps += 1
                        else:
                            large_steps += 1
                        exact_rf = exact_res_ref.float()
                        exact_inv = torch.rsqrt(exact_rf.pow(2).mean(dim=-1, keepdim=True) + 1e-6)
                        exact_norm = (exact_rf * exact_inv * (1.0 + w.float().cpu())).to(torch.bfloat16)
                        rf = res_ref.float()
                        inv = torch.rsqrt(rf.pow(2).mean(dim=-1, keepdim=True) + 1e-6)
                        exp_norm = (rf * inv * (1.0 + w.float().cpu())).to(torch.bfloat16)

                        exact_res_diff = float((res.float().cpu() - exact_res_ref.float()).abs().max().item())
                        exact_norm_diff = float((norm_out.float().cpu() - exact_norm.float()).abs().max().item())
                        res_diff = float((res.float().cpu() - res_ref.float()).abs().max().item())
                        norm_diff = float((norm_out.float().cpu() - exp_norm.float()).abs().max().item())
                        if m >= eff_blocks:
                            assert exact_res_diff == 0.0, (
                                f"exact residual mismatch (pxb={use_pxb}, blk={blk}) at {(m, h)}: {exact_res_diff}"
                            )
                            assert exact_norm_diff <= 0.035, (
                                f"exact rmsnorm mismatch (pxb={use_pxb}, blk={blk}) at {(m, h)}: {exact_norm_diff}"
                            )
                        assert res_diff < 0.03, f"residual mismatch (pxb={use_pxb}, blk={blk}) at {(m, h)}: {res_diff}"
                        assert norm_diff < 0.08, f"rmsnorm mismatch (pxb={use_pxb}, blk={blk}) at {(m, h)}: {norm_diff}"

                torch.cuda.synchronize(device)
                ctrl_main = comm.control[:blk].cpu().tolist()
                assert ctrl_main == [large_steps] * blk, (
                    f"main lockstep broken: {ctrl_main} != {large_steps}"
                )
                if use_pxb and blk == 32:
                    ctrl_small = comm.control[32:40].cpu().tolist()
                    assert ctrl_small == [small_steps] * 8, (
                        f"small_pxb lockstep broken: {ctrl_small} != {small_steps}"
                    )
    finally:
        dist.destroy_process_group()


@pytest.mark.skipif(
    not torch.cuda.is_available()
    or torch.cuda.device_count() < 8
    or not torch.cuda.can_device_access_peer(0, 1),
    reason="Requires 8 P2P-capable GPUs",
)
def test_shm_v5_pxb_8gpu_alternating_parity_and_fused_rmsnorm(tmp_path: Path) -> None:
    init_file = str(tmp_path / "dist_init")
    mp.spawn(
        _pxb_8gpu_worker,
        args=(init_file, f"test_pxb_{os.getpid()}"),
        nprocs=8,
        join=True,
    )
