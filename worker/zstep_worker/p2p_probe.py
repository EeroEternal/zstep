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

"""Adaptive PCIe P2P topology discovery, ACS health verification, and NCCL tuning.

Inspects per-GPU sysfs PCIe bridge paths and `cudaDeviceCanAccessPeer` without
creating peer CUDA contexts, verifies live intra-switch (`PXB`/`PIX`) CUDA-IPC D2D
bandwidth before `ncclCommInitRank` so hosts with unconfigured PCIe ACS redirection
(`ReqRedir+`, where P2P BAR throttles to ~8 GB/s and NCCL P2P rings deadlock) safely
fall back to Host-SHM, and selects topology-optimal NCCL / `shm_v5_pxb` settings.
"""

from __future__ import annotations

import ctypes
from dataclasses import dataclass
import json
import logging
import os
from pathlib import Path
import time

from zstep_worker.numa_pin import _load_cudart, _query_cuda_pci_bus_id

_LOG = logging.getLogger(__name__)

DEFAULT_MIN_PXB_BI_GBPS = 25.0
_CUDA_MEMCPY_DEVICE_TO_DEVICE = 3
_CUDA_IPC_MEM_LAZY_ENABLE_PEER_ACCESS = 1
_CUDA_IPC_HANDLE_BYTES = 64


class _CudaIpcMemHandle(ctypes.Structure):
    _fields_ = [("reserved", ctypes.c_ubyte * _CUDA_IPC_HANDLE_BYTES)]


@dataclass(frozen=True)
class P2pTopologyReport:
    """Resolved P2P capability, PCIe hierarchy, and collective routing policy."""

    p2p_enabled: bool
    pxb_shm_enabled: bool
    nccl_p2p_level: str
    symmetric_pxb_pairs: bool
    unique_numa_nodes: int
    measured_pxb_bi_gbps: float | None
    reason: str


def classify_pci_pair_relation(
    realpath_a: str,
    numa_a: int,
    realpath_b: str,
    numa_b: int,
) -> str:
    """Classify PCIe topology relation (`PIX`, `PXB`, `PHB`, `NODE`, `SYS`) from sysfs paths."""
    if realpath_a == realpath_b and realpath_a:
        return "LOC"

    def _pci_chain(rp: str) -> list[str]:
        marker = "/sys/devices/"
        idx = rp.find(marker)
        tail = rp[idx + len(marker) :] if idx >= 0 else rp.strip("/")
        parts = [p for p in tail.split("/") if p]
        # Keep `pci0000:xx` host bridge + intermediate PCIe bridges + endpoint BDF.
        return parts

    chain_a = _pci_chain(realpath_a)
    chain_b = _pci_chain(realpath_b)
    if len(chain_a) < 2 or len(chain_b) < 2:
        if numa_a >= 0 and numa_a == numa_b:
            return "NODE"
        return "SYS"

    anc_a = chain_a[:-1]
    anc_b = chain_b[:-1]
    lcp_len = 0
    for pa, pb in zip(anc_a, anc_b, strict=False):
        if pa != pb:
            break
        lcp_len += 1

    if lcp_len >= 2:
        hops_a = len(anc_a) - lcp_len
        hops_b = len(anc_b) - lcp_len
        if hops_a == 0 and hops_b == 0:
            return "PIX"
        return "PXB"
    if lcp_len == 1:
        return "PHB"
    if numa_a >= 0 and numa_a == numa_b:
        return "NODE"
    return "SYS"


def _read_sysfs_pci_info(bdf: str) -> tuple[str, int]:
    sysfs_link = Path("/sys/bus/pci/devices") / bdf
    realpath = os.path.realpath(sysfs_link) if sysfs_link.exists() else ""
    numa_node = -1
    numa_file = sysfs_link / "numa_node"
    if numa_file.is_file():
        try:
            numa_node = int(numa_file.read_text(encoding="utf-8").strip())
        except ValueError:
            numa_node = -1
    return realpath, numa_node


def _query_peer_access_matrix(cudart: ctypes.CDLL, world_size: int) -> list[list[bool]]:
    fn = cudart.cudaDeviceCanAccessPeer
    fn.argtypes = [ctypes.POINTER(ctypes.c_int), ctypes.c_int, ctypes.c_int]
    fn.restype = ctypes.c_int
    matrix: list[list[bool]] = []
    for src in range(world_size):
        row: list[bool] = []
        for dst in range(world_size):
            if src == dst:
                row.append(True)
                continue
            can = ctypes.c_int(0)
            rc = fn(ctypes.byref(can), int(src), int(dst))
            row.append(rc == 0 and can.value != 0)
        matrix.append(row)
    return matrix


def _atomic_write_bytes(path: Path, payload: bytes) -> None:
    tmp = path.with_name(f".{path.name}.tmp.{os.getpid()}")
    tmp.write_bytes(payload)
    os.replace(tmp, path)


def _wait_for_file(path: Path, timeout_s: float) -> bytes | None:
    deadline = time.monotonic() + timeout_s
    while time.monotonic() < deadline:
        if path.is_file():
            try:
                return path.read_bytes()
            except OSError:
                pass
        time.sleep(0.002)
    return None


def _probe_pxb_ipc_bandwidth_all_pairs(
    cudart: ctypes.CDLL,
    *,
    rank: int,
    world_size: int,
    device_index: int,
    dist_init_file: str,
    timeout_s: float = 30.0,
) -> float | None:
    """Measure min bidirectional CUDA-IPC D2D GB/s across all `(2k, 2k+1)` switch pairs."""
    ppid = os.getppid()
    base = Path(f"{dist_init_file}.p2p_probe.{ppid}")
    verdict_path = Path(f"{dist_init_file}.p2p_verdict.{ppid}.json")
    h_paths = [Path(f"{base}.h{r}") for r in range(world_size)]
    go_paths = [Path(f"{base}.go{r}") for r in range(world_size)]
    done_paths = [Path(f"{base}.done{r}") for r in range(world_size)]
    partner = rank ^ 1

    set_dev = cudart.cudaSetDevice
    set_dev.argtypes = [ctypes.c_int]
    set_dev.restype = ctypes.c_int

    malloc_fn = cudart.cudaMalloc
    malloc_fn.argtypes = [ctypes.POINTER(ctypes.c_void_p), ctypes.c_size_t]
    malloc_fn.restype = ctypes.c_int

    free_fn = cudart.cudaFree
    free_fn.argtypes = [ctypes.c_void_p]
    free_fn.restype = ctypes.c_int

    memset_fn = cudart.cudaMemset
    memset_fn.argtypes = [ctypes.c_void_p, ctypes.c_int, ctypes.c_size_t]
    memset_fn.restype = ctypes.c_int

    sync_fn = cudart.cudaDeviceSynchronize
    sync_fn.argtypes = []
    sync_fn.restype = ctypes.c_int

    get_ipc = cudart.cudaIpcGetMemHandle
    get_ipc.argtypes = [ctypes.POINTER(_CudaIpcMemHandle), ctypes.c_void_p]
    get_ipc.restype = ctypes.c_int

    open_ipc = cudart.cudaIpcOpenMemHandle
    open_ipc.argtypes = [
        ctypes.POINTER(ctypes.c_void_p),
        _CudaIpcMemHandle,
        ctypes.c_uint,
    ]
    open_ipc.restype = ctypes.c_int

    close_ipc = cudart.cudaIpcCloseMemHandle
    close_ipc.argtypes = [ctypes.c_void_p]
    close_ipc.restype = ctypes.c_int

    memcpy_async = cudart.cudaMemcpyAsync
    memcpy_async.argtypes = [
        ctypes.c_void_p,
        ctypes.c_void_p,
        ctypes.c_size_t,
        ctypes.c_int,
        ctypes.c_void_p,
    ]
    memcpy_async.restype = ctypes.c_int

    size_bytes = 16 * 1024 * 1024  # 16 MiB transient probe buffer
    own_src = ctypes.c_void_p(0)
    own_dst = ctypes.c_void_p(0)
    peer_dst = ctypes.c_void_p(0)
    measured: float | None = None
    try:
        if set_dev(int(device_index)) != 0:
            raise RuntimeError("cudaSetDevice failed")
        if malloc_fn(ctypes.byref(own_src), size_bytes) != 0:
            raise RuntimeError("cudaMalloc src failed")
        if malloc_fn(ctypes.byref(own_dst), size_bytes) != 0:
            raise RuntimeError("cudaMalloc dst failed")
        memset_fn(own_src, 1, size_bytes)
        memset_fn(own_dst, 0, size_bytes)
        sync_fn()

        handle = _CudaIpcMemHandle()
        if get_ipc(ctypes.byref(handle), own_dst) != 0:
            raise RuntimeError("cudaIpcGetMemHandle failed")
        _atomic_write_bytes(h_paths[rank], bytes(handle.reserved))

        peer_raw = _wait_for_file(h_paths[partner], timeout_s)
        if peer_raw is None or len(peer_raw) != _CUDA_IPC_HANDLE_BYTES:
            raise RuntimeError("timed out waiting for peer IPC handle")
        peer_handle = _CudaIpcMemHandle.from_buffer_copy(peer_raw)
        if (
            open_ipc(
                ctypes.byref(peer_dst),
                peer_handle,
                _CUDA_IPC_MEM_LAZY_ENABLE_PEER_ACCESS,
            )
            != 0
        ):
            raise RuntimeError("cudaIpcOpenMemHandle failed")

        for _ in range(2):
            if (
                memcpy_async(
                    peer_dst, own_src, size_bytes, _CUDA_MEMCPY_DEVICE_TO_DEVICE, None
                )
                != 0
            ):
                raise RuntimeError("warmup cudaMemcpyAsync failed")
        sync_fn()
        _atomic_write_bytes(go_paths[rank], b"1")
        if _wait_for_file(go_paths[partner], timeout_s) is None:
            raise RuntimeError("timed out waiting for peer warmup barrier")

        iters = 8
        t0 = time.perf_counter()
        for _ in range(iters):
            if (
                memcpy_async(
                    peer_dst, own_src, size_bytes, _CUDA_MEMCPY_DEVICE_TO_DEVICE, None
                )
                != 0
            ):
                raise RuntimeError("timed cudaMemcpyAsync failed")
        sync_fn()
        elapsed = max(time.perf_counter() - t0, 1e-6)
        bi_gbps = (2.0 * iters * size_bytes / 1e9) / elapsed
        measured = round(bi_gbps, 2)
    except Exception as exc:
        _LOG.warning("Rank %d PXB IPC bandwidth probe failed: %s", rank, exc)
        measured = None
    finally:
        if own_src.value or own_dst.value or peer_dst.value:
            sync_fn()
        if peer_dst.value:
            close_ipc(peer_dst)
        _atomic_write_bytes(
            done_paths[rank],
            json.dumps({"bi_gbps": measured}).encode("utf-8"),
        )
        partner_done = _wait_for_file(done_paths[partner], 5.0)
        if partner_done is None:
            try:
                h_paths[rank].unlink(missing_ok=True)
            except OSError:
                pass
        if own_dst.value:
            free_fn(own_dst)
        if own_src.value:
            free_fn(own_src)

    if rank == 0:
        all_gbps: list[float] = []
        all_ok = True
        for r in range(world_size):
            r_raw = _wait_for_file(done_paths[r], timeout_s)
            if r_raw is None:
                all_ok = False
                break
            try:
                val = json.loads(r_raw.decode("utf-8")).get("bi_gbps")
            except (UnicodeDecodeError, json.JSONDecodeError):
                val = None
            if val is None:
                all_ok = False
                break
            all_gbps.append(float(val))
        final_gbps = round(min(all_gbps), 2) if (all_ok and all_gbps) else None
        _atomic_write_bytes(
            verdict_path,
            json.dumps({"bi_gbps": final_gbps}).encode("utf-8"),
        )

    raw_verdict = _wait_for_file(verdict_path, timeout_s + 5.0)
    seen_path = Path(f"{base}.seen{rank}")
    try:
        _atomic_write_bytes(seen_path, b"1")
    except OSError:
        pass

    if rank == 0:
        deadline = time.monotonic() + 2.0
        while time.monotonic() < deadline:
            if all(Path(f"{base}.seen{r}").is_file() for r in range(world_size)):
                break
            time.sleep(0.002)
        for p in [
            verdict_path,
            *h_paths,
            *go_paths,
            *done_paths,
            *(Path(f"{base}.seen{r}") for r in range(world_size)),
        ]:
            try:
                p.unlink(missing_ok=True)
            except OSError:
                pass

    if raw_verdict is None:
        return None
    try:
        val = json.loads(raw_verdict.decode("utf-8")).get("bi_gbps")
        return float(val) if val is not None else None
    except (UnicodeDecodeError, json.JSONDecodeError, TypeError, ValueError):
        return None


def configure_adaptive_p2p_env(
    *,
    rank: int,
    world_size: int,
    device_index: int,
    dist_init_file: str | None = None,
    alert_file: Path = Path("/var/log/6000d-acs-off.ALERT"),
) -> P2pTopologyReport:
    """Auto-detect PCIe P2P + ACS health and populate optimal `NCCL_*` env vars before init."""
    os.environ.setdefault("NCCL_IB_DISABLE", "1")
    mode = os.environ.get("ZSTEP_P2P_MODE", "auto").strip().lower()
    user_p2p_disable = os.environ.get("NCCL_P2P_DISABLE")

    def _apply_non_p2p_defaults(reason: str, bi_gbps: float | None = None) -> P2pTopologyReport:
        os.environ.setdefault("NCCL_P2P_DISABLE", "1")
        os.environ.setdefault("NCCL_P2P_LEVEL", "LOC")
        os.environ.setdefault("NCCL_SHM_USE_CUDA_MEMCPY", "1")
        os.environ.setdefault("NCCL_SHM_MEMCPY_MODE", "3")
        os.environ.setdefault("NCCL_MIN_NCHANNELS", "6")
        os.environ.setdefault("NCCL_MAX_NCHANNELS", "6")
        os.environ.setdefault("NCCL_BUFFSIZE", "4194304")
        return P2pTopologyReport(
            p2p_enabled=False,
            pxb_shm_enabled=False,
            nccl_p2p_level=os.environ["NCCL_P2P_LEVEL"],
            symmetric_pxb_pairs=False,
            unique_numa_nodes=0,
            measured_pxb_bi_gbps=bi_gbps,
            reason=reason,
        )

    if mode in {"0", "off", "false", "disable", "disabled"}:
        return _apply_non_p2p_defaults("ZSTEP_P2P_MODE=off")
    if user_p2p_disable == "1":
        return _apply_non_p2p_defaults("NCCL_P2P_DISABLE=1")
    if world_size < 2:
        return _apply_non_p2p_defaults("world_size<2")
    if alert_file.is_file() and mode != "force":
        _LOG.warning(
            "Detected PCIe ACS alert file %s; falling back to non-P2P Host-SHM collectives",
            alert_file,
        )
        return _apply_non_p2p_defaults("acs_alert_file_present")

    try:
        cudart = _load_cudart()
        peer_mat = _query_peer_access_matrix(cudart, world_size)
        bdfs = [_query_cuda_pci_bus_id(i) for i in range(world_size)]
        pci_infos = [_read_sysfs_pci_info(bdf) for bdf in bdfs]
    except Exception as exc:
        _LOG.warning("PCIe P2P topology probe failed (%s); falling back to Host-SHM", exc)
        return _apply_non_p2p_defaults(f"probe_error:{exc}")

    numa_set = {numa for _, numa in pci_infos if numa >= 0}
    unique_numa = len(numa_set) if numa_set else 1

    # Check if every `(2k, 2k+1)` pair sits under a shared PCIe switch (`PIX` or `PXB`)
    symmetric_switch_pairs = (world_size % 2 == 0) and all(
        classify_pci_pair_relation(
            pci_infos[2 * k][0],
            pci_infos[2 * k][1],
            pci_infos[2 * k + 1][0],
            pci_infos[2 * k + 1][1],
        )
        in {"PIX", "PXB"}
        for k in range(world_size // 2)
    )
    all_peers_ok = all(peer_mat[i][j] for i in range(world_size) for j in range(world_size))
    all_switch_pairs_peer_ok = symmetric_switch_pairs and all(
        peer_mat[2 * k][2 * k + 1] and peer_mat[2 * k + 1][2 * k]
        for k in range(world_size // 2)
    )

    if not all_peers_ok and not all_switch_pairs_peer_ok:
        return _apply_non_p2p_defaults("can_device_access_peer=0")

    # Determine target NCCL_P2P_LEVEL from global cluster-wide topology:
    # - Single NUMA (e.g. TP=2 / TP=4 within one socket): `PHB` enables full intra-socket P2P.
    # - Multi-NUMA with intra-switch GPU pairs (e.g. 8x RTX 6000D CX8): `PXB` uses 103 GB/s
    #   intra-switch P2P while keeping cross-Root-Port / UPI hops on Xeon DDIO Host-SHM.
    if unique_numa == 1 and all_peers_ok:
        recommended_level = "PHB"
    elif all_switch_pairs_peer_ok:
        pair_rels = {
            classify_pci_pair_relation(
                pci_infos[2 * k][0],
                pci_infos[2 * k][1],
                pci_infos[2 * k + 1][0],
                pci_infos[2 * k + 1][1],
            )
            for k in range(world_size // 2)
        }
        recommended_level = "PIX" if pair_rels == {"PIX"} else "PXB"
    elif all_peers_ok:
        recommended_level = "PHB"
    else:
        return _apply_non_p2p_defaults("partial_peer_access_unsupported")

    min_bi_gbps = float(
        os.environ.get("ZSTEP_P2P_MIN_PXB_BI_GBPS", str(DEFAULT_MIN_PXB_BI_GBPS))
    )
    measured_bi_gbps: float | None = None
    if (
        dist_init_file is not None
        and all_switch_pairs_peer_ok
        and mode != "force"
        and min_bi_gbps > 0.0
    ):
        measured_bi_gbps = _probe_pxb_ipc_bandwidth_all_pairs(
            cudart,
            rank=rank,
            world_size=world_size,
            device_index=device_index,
            dist_init_file=dist_init_file,
        )
        if measured_bi_gbps is None or measured_bi_gbps < min_bi_gbps:
            _LOG.warning(
                "PCIe P2P BAR reported OK, but intra-switch bidirectional bandwidth is %s GB/s "
                "(min %.1f GB/s; check PCIe ACS ReqRedir+ / 6000d-acs-off.service). "
                "Falling back to safe non-P2P Host-SHM collectives.",
                measured_bi_gbps,
                min_bi_gbps,
            )
            return _apply_non_p2p_defaults(
                f"pxb_bandwidth_too_low:{measured_bi_gbps}", bi_gbps=measured_bi_gbps
            )

    os.environ.setdefault("NCCL_P2P_DISABLE", "0")
    os.environ.setdefault("NCCL_P2P_LEVEL", recommended_level)
    pxb_shm_env = os.environ.get("ZSTEP_SHM_ENABLE_PXB_P2P", "1").strip().lower()
    pxb_shm_enabled = (
        pxb_shm_env not in {"0", "off", "false"}
        and world_size == 8
        and all_switch_pairs_peer_ok
    )
    report = P2pTopologyReport(
        p2p_enabled=True,
        pxb_shm_enabled=pxb_shm_enabled,
        nccl_p2p_level=os.environ["NCCL_P2P_LEVEL"],
        symmetric_pxb_pairs=symmetric_switch_pairs,
        unique_numa_nodes=unique_numa,
        measured_pxb_bi_gbps=measured_bi_gbps,
        reason="verified_p2p",
    )
    if rank == 0:
        _LOG.info(
            "Adaptive P2P probe: p2p_enabled=%s pxb_shm=%s nccl_level=%s numa_nodes=%d pxb_bi_gbps=%s",
            report.p2p_enabled,
            report.pxb_shm_enabled,
            report.nccl_p2p_level,
            report.unique_numa_nodes,
            report.measured_pxb_bi_gbps,
        )
    return report
