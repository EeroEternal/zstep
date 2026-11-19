# Copyright (c) 2026 zstep Contributors
# SPDX-License-Identifier: MIT

from __future__ import annotations

import os
from pathlib import Path
from unittest.mock import patch

import pytest

try:
    import torch
    import torch.distributed as dist
    import torch.multiprocessing as mp
except ImportError:  # pragma: no cover - exercised in torch-free CI
    torch = None  # type: ignore[assignment]
    dist = None  # type: ignore[assignment]
    mp = None  # type: ignore[assignment]

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


@pytest.mark.skipif(torch is None, reason="Requires PyTorch")
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


@pytest.mark.skipif(torch is None, reason="Requires PyTorch")
def test_shm_allgather_attach_pxb_ipc_tracks_owner_liveness() -> None:
    from types import SimpleNamespace
    from zstep_worker.kernels_sm120.shm_allreduce import (
        HostMappedShmAllGather,
        HostMappedShmAllReduce,
    )

    ag = object.__new__(HostMappedShmAllGather)
    ag.rank = 0
    ag.world_size = 8
    ag.device = torch.device("cpu")
    ag._closed = False
    ag._registered = False
    ag._pxb_ar_ref = None

    ar = object.__new__(HostMappedShmAllReduce)
    ar.rank = 0
    ar.world_size = 8
    ar.device = torch.device("cpu")
    ar._closed = False
    ar._pxb_enabled = True
    ar._my_hbm_ptr = 0x1000_0000
    ar._pt_hbm_ptr = 0x2000_0000
    ar.pxb_slot_stride_bytes = 4096

    ag.attach_pxb_ipc(ar)
    assert ag._my_hbm_ptr == 0x1000_0000
    assert ag._pt_hbm_ptr == 0x2000_0000
    assert ag._pxb_cap_bytes == 8192

    # Mismatched rank/world/device must fail loudly.
    bad_ar = SimpleNamespace(
        rank=1,
        world_size=8,
        device=torch.device("cpu"),
        pxb_enabled=True,
        _closed=False,
    )
    with pytest.raises(ValueError, match="attach_pxb_ipc rank/world/device mismatch"):
        ag.attach_pxb_ipc(bad_ar)  # type: ignore[arg-type]

    # Closing `ar` before `ag` immediately zeroes IPC pointers (no dangling UAF).
    ar._closed = True
    ar._pxb_enabled = False
    assert ag._my_hbm_ptr == 0
    assert ag._pt_hbm_ptr == 0
    assert ag._pxb_cap_bytes == 0


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
    from zstep_worker.kernels_sm120.shm_allreduce import (
        HostMappedShmAllGather,
        HostMappedShmAllReduce,
    )

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
                    # Verify PXB partner-relayed AllGather (`attach_pxb_ipc`) interleaved
                    # back-to-back with `comm.all_reduce` on the shared 2-slot IPC ring,
                    # including `use_pxb=True -> False (ag_m=1152) -> True (ag_m=24)` transitions.
                    with HostMappedShmAllGather(
                        name=f"{shm_name}_ag_{cfg_idx}",
                        rank=rank,
                        world_size=8,
                        capacity_input_elements=1280 * 3200,
                        blocks=32,
                        threads=512,
                        exchange_group=dist.group.WORLD,
                    ) as ag:
                        ag.attach_pxb_ipc(comm)
                        assert ag._my_hbm_ptr == comm._my_hbm_ptr and ag._my_hbm_ptr != 0
                        for ag_m in (1, 8, 16, 32, 128, 1152, 24):
                            torch.manual_seed(900 + ag_m * 17 + rank)
                            ag_in = torch.randn(
                                (ag_m, 3200), dtype=torch.bfloat16, device=device
                            )
                            ref_list = [
                                torch.empty((ag_m, 3200), dtype=torch.bfloat16)
                                for _ in range(8)
                            ]
                            dist.all_gather(ref_list, ag_in.cpu(), group=dist.group.WORLD)
                            ag_out0 = ag.all_gather(ag_in, dim=0)
                            ref0 = torch.cat(ref_list, dim=0)
                            assert float((ag_out0.cpu().float() - ref0.float()).abs().max()) == 0.0
                            ag_out_last = ag.all_gather(ag_in, dim=-1)
                            ref_last = torch.cat(ref_list, dim=-1)
                            assert (
                                float((ag_out_last.cpu().float() - ref_last.float()).abs().max())
                                == 0.0
                            )

                            # 3D inner-dim AllGather (`dim=1`, matching Engram `[T, G, H]`)
                            ag_in_3d = torch.randn(
                                (ag_m, 3, 256), dtype=torch.bfloat16, device=device
                            )
                            ref_3d_list = [
                                torch.empty((ag_m, 3, 256), dtype=torch.bfloat16)
                                for _ in range(8)
                            ]
                            dist.all_gather(ref_3d_list, ag_in_3d.cpu(), group=dist.group.WORLD)
                            ag_out_3d = ag.all_gather(ag_in_3d, dim=1)
                            ref_3d = torch.cat(ref_3d_list, dim=1)
                            assert (
                                float((ag_out_3d.cpu().float() - ref_3d.float()).abs().max())
                                == 0.0
                            )

                            # Interleave AllReduce immediately after PXB AllGather on same ring
                            ar_m = min(ag_m, 768)
                            ar_x = torch.randn(
                                (ar_m, 5120), dtype=torch.bfloat16, device=device
                            ) * 0.125
                            ar_ref = ar_x.float().cpu()
                            dist.all_reduce(ar_ref, group=dist.group.WORLD)
                            ar_out = torch.empty_like(ar_x)
                            comm.all_reduce(ar_x, output=ar_out)
                            assert float((ar_out.float().cpu() - ar_ref).abs().max()) < 0.03

                # Verify SM120 Cluster-8 DSMEM fused `AR + mHC + collapse + RMSNorm` (`M in {1, 2, 4}`)
                # across both non-PXB (`blk=8, flag_base=0`) and dual-grid PXB (`blk=32, flag_base=32`).
                if torch.cuda.get_device_capability(device) == (12, 0) and (
                    (not use_pxb and blk == 8) or (use_pxb and blk == 32)
                ):
                    from zstep_worker.kernels_sm120.shm_allreduce import (
                        get_host_mapped_allreduce_extension,
                    )

                    ext = get_host_mapped_allreduce_extension()
                    stream = torch.cuda.current_stream(device).cuda_stream
                    flag_base = 32 if (use_pxb and blk == 32) else 0
                    torch.cuda.synchronize(device)
                    base_c8 = comm.control[flag_base : flag_base + 8].cpu().tolist()
                    assert len(set(base_c8)) == 1
                    c8_calls = 0
                    empty_aux = torch.empty((0, 5120), dtype=torch.bfloat16, device=device)
                    for c8_rows in (1, 2, 4):
                        for write_aux in (0, 1):
                            torch.manual_seed(2100 + c8_rows * 19 + write_aux * 7 + rank)
                            x_c8 = (
                                torch.randn((c8_rows, 5120), dtype=torch.bfloat16, device=device)
                                * 0.125
                            )
                            torch.manual_seed(2200 + c8_rows * 19 + write_aux * 7)
                            res0_c8 = (
                                torch.randn(
                                    (c8_rows, 4, 5120), dtype=torch.bfloat16, device=device
                                )
                                * 0.25
                            )
                            post_c8 = (
                                torch.sigmoid(
                                    torch.randn((c8_rows, 4), dtype=torch.float32, device=device)
                                )
                                * 2.0
                            )
                            comb_c8 = torch.softmax(
                                torch.randn(
                                    (c8_rows, 4, 4), dtype=torch.float32, device=device
                                ),
                                dim=-1,
                            ).contiguous()
                            pre_c8 = (
                                torch.sigmoid(
                                    torch.randn((c8_rows, 4), dtype=torch.float32, device=device)
                                )
                                + 1e-4
                            )
                            nw_c8 = (
                                torch.randn((5120,), dtype=torch.bfloat16, device=device) * 0.5
                                + 1.0
                            )

                            ref_ar_c8 = torch.empty_like(x_c8)
                            comm.all_reduce(x_c8, output=ref_ar_c8)
                            c8_calls += 1
                            ref_res_c8 = res0_c8.clone()
                            ref_in_c8 = torch.empty_like(x_c8)
                            ref_aux_c8 = torch.empty_like(x_c8) if write_aux else empty_aux
                            ext.launch_mhc_post_collapse_rmsnorm(
                                ref_ar_c8,
                                ref_res_c8,
                                post_c8,
                                comb_c8,
                                pre_c8,
                                nw_c8,
                                ref_in_c8,
                                ref_aux_c8,
                                1e-6,
                                write_aux,
                                0,
                                stream,
                            )

                            got_res_c8 = res0_c8.clone()
                            got_in_c8 = torch.empty_like(x_c8)
                            got_aux_c8 = torch.empty_like(x_c8) if write_aux else empty_aux
                            ext.launch_ar_mhc_cluster8(
                                x_c8,
                                got_res_c8,
                                post_c8,
                                comb_c8,
                                pre_c8,
                                nw_c8,
                                got_in_c8,
                                got_aux_c8,
                                comm.control,
                                comm._rank_shm_ptrs,
                                comm._my_hbm_ptr if comm._pxb_enabled else 0,
                                comm._pt_hbm_ptr if comm._pxb_enabled else 0,
                                comm.slot_stride_bytes,
                                comm.pxb_slot_stride_bytes if comm._pxb_enabled else 0,
                                1e-6,
                                write_aux,
                                0,
                                rank,
                                flag_base,
                                stream,
                            )
                            c8_calls += 1

                            assert (
                                float((got_res_c8.float() - ref_res_c8.float()).abs().max().item())
                                == 0.0
                            )
                            # `got_in_c8` reduces `sum_sq` via 8-CTA x 8-warp DSMEM tree vs
                            # 1-CTA x 20-warp SMEM tree (`<= 1 BF16 ULP`), while all 8 ranks
                            # remain strictly bit-identical (`torch.equal` below).
                            assert (
                                float((got_in_c8.float() - ref_in_c8.float()).abs().max().item())
                                <= 0.016
                            )
                            if write_aux:
                                assert (
                                    float(
                                        (got_aux_c8.float() - ref_aux_c8.float()).abs().max().item()
                                    )
                                    == 0.0
                                )
                            gathered_in = [
                                torch.empty_like(got_in_c8, device="cpu") for _ in range(8)
                            ]
                            dist.all_gather(
                                gathered_in, got_in_c8.cpu(), group=dist.group.WORLD
                            )
                            for peer_r in range(1, 8):
                                assert torch.equal(gathered_in[0], gathered_in[peer_r])

                    torch.cuda.synchronize(device)
                    after_c8 = comm.control[flag_base : flag_base + 8].cpu().tolist()
                    assert after_c8 == [base_c8[0] + c8_calls] * 8

        # Verify `FP12+BFP7` (`pxb_bfp` modes 1 & 2) accuracy, in-place safety, and
        # 0-ULP cross-rank bit-identity across all 8 ranks.
        for bfp_mode in (1, 2):
            with HostMappedShmAllReduce(
                name=f"{shm_name}_bfp_{bfp_mode}",
                rank=rank,
                world_size=8,
                capacity_elements=1024 * 5120,
                blocks=32,
                threads=512,
                exchange_group=dist.group.WORLD,
                enable_pxb_p2p=True,
                pxb_max_elements=1024 * 5120,
                enable_pxb_bfp=bfp_mode,
            ) as comm_bfp:
                assert comm_bfp._pxb_bfp == bfp_mode
                for m, h in ((8, 5120), (96, 5120), (256, 5120), (512, 5120)):
                    torch.manual_seed(3100 + bfp_mode * 101 + m * 11 + rank)
                    x = torch.randn((m, h), dtype=torch.bfloat16, device=device) * 0.125
                    ref_cpu = x.float().cpu()
                    dist.all_reduce(ref_cpu, group=dist.group.WORLD)

                    out = torch.empty_like(x)
                    comm_bfp.all_reduce(x, output=out)
                    x_ip = x.clone()
                    comm_bfp.all_reduce(x_ip, output=x_ip)
                    assert torch.equal(out.cpu(), x_ip.cpu()), (
                        f"in-place vs out-of-place mismatch (bfp={bfp_mode}) at {(m, h)}"
                    )
                    diff = float((out.float().cpu() - ref_cpu).abs().max().item())
                    assert diff < 0.03, f"bfp={bfp_mode} all_reduce diff at {(m, h)}: {diff}"

                    gathered_out = [torch.empty((m, h), dtype=torch.bfloat16) for _ in range(8)]
                    dist.all_gather(gathered_out, out.cpu(), group=dist.group.WORLD)
                    for peer_r in range(1, 8):
                        assert torch.equal(gathered_out[0], gathered_out[peer_r]), (
                            f"cross-rank drift (bfp={bfp_mode}) at {(m, h)} rank {peer_r}"
                        )

                    torch.manual_seed(3200 + bfp_mode * 101 + m * 11)
                    res_in = torch.randn((m, h), dtype=torch.bfloat16, device=device) * 0.1
                    w = torch.randn((h,), dtype=torch.bfloat16, device=device) * 0.05
                    res = res_in.clone()
                    norm_out = torch.empty_like(x)
                    comm_bfp.all_reduce_add_rmsnorm(
                        x, res, w, normed_out=norm_out, eps=1e-6, zero_centered=True
                    )
                    res_ref = (
                        res_in.float().cpu() + ref_cpu.to(torch.bfloat16).float()
                    ).to(torch.bfloat16)
                    rf = res_ref.float()
                    inv = torch.rsqrt(rf.pow(2).mean(dim=-1, keepdim=True) + 1e-6)
                    exp_norm = (rf * inv * (1.0 + w.float().cpu())).to(torch.bfloat16)
                    assert float((res.float().cpu() - res_ref.float()).abs().max().item()) < 0.03
                    assert (
                        float((norm_out.float().cpu() - exp_norm.float()).abs().max().item()) < 0.08
                    )
                    gathered_norm = [torch.empty((m, h), dtype=torch.bfloat16) for _ in range(8)]
                    dist.all_gather(gathered_norm, norm_out.cpu(), group=dist.group.WORLD)
                    for peer_r in range(1, 8):
                        assert torch.equal(gathered_norm[0], gathered_norm[peer_r])
    finally:
        dist.destroy_process_group()


@pytest.mark.skipif(
    torch is None
    or not torch.cuda.is_available()
    or torch.cuda.get_device_capability() != (12, 0),
    reason="Requires an sm_120 CUDA device",
)
def test_sm120_mhc_fused_kernels_parity() -> None:
    from zstep_worker.kernels_sm120.shm_allreduce import (
        get_host_mapped_allreduce_extension,
    )

    device = torch.device("cuda:0")
    ext = get_host_mapped_allreduce_extension()
    stream = torch.cuda.current_stream(device).cuda_stream

    for num_toks in (1, 17, 128):
        torch.manual_seed(1200 + num_toks)
        x = torch.randn((num_toks, 5120), dtype=torch.bfloat16, device=device) * 0.25
        res0 = torch.randn((num_toks, 4, 5120), dtype=torch.bfloat16, device=device) * 0.25
        post_mix = torch.sigmoid(
            torch.randn((num_toks, 4), dtype=torch.float32, device=device)
        ) * 2.0
        comb_raw = torch.softmax(
            torch.randn((num_toks, 4, 4), dtype=torch.float32, device=device), dim=-1
        )
        pre_mix = torch.sigmoid(
            torch.randn((num_toks, 4), dtype=torch.float32, device=device)
        ) + 1e-4
        norm_w = torch.randn((5120,), dtype=torch.bfloat16, device=device) * 0.5 + 1.0
        eps = 1e-6

        ref_res = (
            torch.einsum("tkc,tkj->tjc", res0.float(), comb_raw)
            + post_mix.unsqueeze(-1) * x.float().unsqueeze(1)
        ).to(torch.bfloat16)
        ref_aux = (ref_res.float().sum(dim=1) * 0.25).to(torch.bfloat16)
        ref_col = (
            (pre_mix.unsqueeze(-1) * ref_res.float()).sum(dim=1).to(torch.bfloat16)
        )
        ref_inv = torch.rsqrt(ref_col.float().pow(2).mean(dim=-1, keepdim=True) + eps)
        ref_in = (ref_col.float() * ref_inv * norm_w.float()).to(torch.bfloat16)

        for write_aux in (0, 1):
            res_cur = res0.clone()
            layer_in = torch.empty_like(x)
            aux_out = (
                torch.empty_like(x)
                if write_aux
                else torch.empty((0, 5120), dtype=torch.bfloat16, device=device)
            )
            ext.launch_mhc_post_collapse_rmsnorm(
                x,
                res_cur,
                post_mix,
                comb_raw.contiguous(),
                pre_mix,
                norm_w,
                layer_in,
                aux_out,
                eps,
                write_aux,
                0,
                stream,
            )
            assert float((res_cur.float() - ref_res.float()).abs().max().item()) <= 0.008
            exact_aux = (res_cur.float().sum(dim=1) * 0.25).to(torch.bfloat16)
            assert float((layer_in.float() - ref_in.float()).abs().max().item()) <= 0.016
            if write_aux:
                assert float((aux_out.float() - exact_aux.float()).abs().max().item()) == 0.0
                assert float((aux_out.float() - ref_aux.float()).abs().max().item()) <= 0.008

        res_post = res0.clone()
        empty_aux = torch.empty((0, 5120), dtype=torch.bfloat16, device=device)
        ext.launch_mhc_post_collapse_rmsnorm(
            x,
            res_post,
            post_mix,
            comb_raw.contiguous(),
            post_mix,
            empty_aux,
            empty_aux,
            empty_aux,
            eps,
            0,
            1,
            stream,
        )
        assert float((res_post.float() - res_cur.float()).abs().max().item()) == 0.0
        assert float((res_post.float() - ref_res.float()).abs().max().item()) <= 0.008

    for n_splits in (1, 4, 16):
        for num_toks in (1, 17, 128):
            torch.manual_seed(3400 + n_splits * 31 + num_toks)
            gemm_mul = torch.randn(
                (n_splits, num_toks, 24), dtype=torch.float32, device=device
            ) * 0.2
            gemm_sqr = (
                torch.rand((n_splits, num_toks), dtype=torch.float32, device=device)
                * 5000.0
                + 100.0
            )
            hc_scale = torch.tensor([0.5, 0.7, 0.3], dtype=torch.float32, device=device)
            hc_base = torch.randn((24,), dtype=torch.float32, device=device) * 0.1
            rms_eps = 1e-20
            pre_eps = 1e-4
            sk_eps = 1e-6
            post_mult = 2.0
            sk_rep = 20

            post_out = torch.empty((num_toks, 4), dtype=torch.float32, device=device)
            comb_out = torch.empty((num_toks, 16), dtype=torch.float32, device=device)
            pre_out = torch.empty((num_toks, 4), dtype=torch.float32, device=device)
            ext.launch_mhc_sinkhorn_stats(
                gemm_mul,
                gemm_sqr,
                hc_scale,
                hc_base,
                post_out,
                comb_out,
                pre_out,
                20480,
                rms_eps,
                pre_eps,
                sk_eps,
                post_mult,
                sk_rep,
                1,
                stream,
            )

            rms = torch.rsqrt(gemm_sqr.sum(dim=0) * (1.0 / 20480.0) + rms_eps).unsqueeze(-1)
            mix = gemm_mul.sum(dim=0) * rms
            ref_pre = torch.sigmoid(mix[:, 0:4] * hc_scale[0] + hc_base[0:4]) + pre_eps
            ref_post = torch.sigmoid(mix[:, 4:8] * hc_scale[1] + hc_base[4:8]) * post_mult
            cm = (mix[:, 8:24] * hc_scale[2] + hc_base[8:24]).view(num_toks, 4, 4)
            cm = torch.exp(cm - cm.amax(dim=-1, keepdim=True))
            cm = cm / cm.sum(dim=-1, keepdim=True) + sk_eps
            cm = cm / (cm.sum(dim=-2, keepdim=True) + sk_eps)
            for _ in range(sk_rep - 1):
                cm = cm / (cm.sum(dim=-1, keepdim=True) + sk_eps)
                cm = cm / (cm.sum(dim=-2, keepdim=True) + sk_eps)

            assert float((pre_out - ref_pre).abs().max().item()) < 1e-5
            assert float((post_out - ref_post).abs().max().item()) < 1e-5
            assert float((comb_out.view(num_toks, 4, 4) - cm).abs().max().item()) < 1e-5


@pytest.mark.skipif(
    torch is None
    or not torch.cuda.is_available()
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
