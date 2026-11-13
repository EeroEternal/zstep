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

"""PCIe -> host NUMA discovery and CPU affinity pinning for GPU workers.

On multi-socket / multi-switch PCIe workstations without NVLink (e.g. 4-NUMA
topologies where GPUs map to `[0, 0, 0, 1, 2, 2, 2, 3]`), binding each worker
process to its GPU's local CPU list before first-touching host-mapped shared
memory avoids 30-50% cross-socket interconnect penalties.
"""

from __future__ import annotations

import ctypes
import ctypes.util
from dataclasses import dataclass
import os
from pathlib import Path


@dataclass(frozen=True)
class DeviceNumaTopology:
    """Resolved PCIe bus ID, host NUMA node, and pinned CPU set for one CUDA device."""

    device_index: int
    pci_bus_id: str
    numa_node: int
    pinned_cpus: tuple[int, ...]


def parse_cpulist(raw: str) -> set[int]:
    """Parse Linux sysfs cpulist syntax such as `'0-15,64-79'` into integer core IDs."""
    cpus: set[int] = set()
    stripped = raw.strip()
    if not stripped:
        return cpus
    for part in stripped.split(","):
        item = part.strip()
        if not item:
            continue
        if "-" in item:
            lo_str, hi_str = item.split("-", 1)
            lo = int(lo_str)
            hi = int(hi_str)
            if hi < lo:
                raise ValueError(f"invalid CPU range {item!r}")
            cpus.update(range(lo, hi + 1))
        else:
            cpus.add(int(item))
    return cpus


_PINNED_CACHE: dict[tuple[int, int], DeviceNumaTopology] = {}


def _load_cudart() -> ctypes.CDLL:
    candidates: list[str] = ["libcudart.so", "libcudart.so.12", "libcudart.so.13"]
    found = ctypes.util.find_library("cudart")
    if found and found not in candidates:
        candidates.append(found)
    last_err: OSError | None = None
    for name in candidates:
        try:
            return ctypes.CDLL(name)
        except OSError as exc:
            last_err = exc
    raise RuntimeError(f"failed to load CUDA runtime library from {candidates}: {last_err}")


def _query_cuda_pci_bus_id(device_index: int) -> str:
    cudart = _load_cudart()
    func = cudart.cudaDeviceGetPCIBusId
    func.argtypes = [ctypes.c_char_p, ctypes.c_int, ctypes.c_int]
    func.restype = ctypes.c_int
    buf = ctypes.create_string_buffer(32)
    status = func(buf, 32, device_index)
    if status != 0:
        raise RuntimeError(f"cudaDeviceGetPCIBusId({device_index}) failed with status {status}")
    raw_bdf = buf.value.decode("ascii").strip().lower()
    # CUDA sometimes formats the domain with 8 hex digits (`00000000:16:00.0`);
    # Linux `/sys/bus/pci/devices` uses 4 hex digits (`0000:16:00.0`) for domain 0.
    parts = raw_bdf.split(":")
    if len(parts) == 3 and len(parts[0]) > 4:
        domain_int = int(parts[0], 16)
        raw_bdf = f"{domain_int:04x}:{parts[1]}:{parts[2]}"
    return raw_bdf


def pin_process_to_device_numa(device_index: int) -> DeviceNumaTopology:
    """Discover `device_index`'s PCIe NUMA domain and pin the calling process's CPU affinity."""
    cache_key = (os.getpid(), int(device_index))
    cached = _PINNED_CACHE.get(cache_key)
    if cached is not None:
        return cached
    pci_bus_id = _query_cuda_pci_bus_id(device_index)
    sysfs_dir = Path("/sys/bus/pci/devices") / pci_bus_id
    numa_node = -1
    numa_file = sysfs_dir / "numa_node"
    if numa_file.is_file():
        try:
            numa_node = int(numa_file.read_text(encoding="utf-8").strip())
        except ValueError:
            numa_node = -1

    allowed_cpus = set(os.sched_getaffinity(0))
    target_cpus: set[int] = set()
    cpulist_file = sysfs_dir / "local_cpulist"
    if cpulist_file.is_file():
        local_cpus = parse_cpulist(cpulist_file.read_text(encoding="utf-8"))
        target_cpus = local_cpus & allowed_cpus

    if not target_cpus and numa_node >= 0:
        node_cpulist = Path(f"/sys/devices/system/node/node{numa_node}/cpulist")
        if node_cpulist.is_file():
            target_cpus = parse_cpulist(node_cpulist.read_text(encoding="utf-8")) & allowed_cpus

    if target_cpus:
        os.sched_setaffinity(0, target_cpus)
        pinned = tuple(sorted(target_cpus))
    else:
        pinned = tuple(sorted(allowed_cpus))

    topo = DeviceNumaTopology(
        device_index=device_index,
        pci_bus_id=pci_bus_id,
        numa_node=numa_node,
        pinned_cpus=pinned,
    )
    _PINNED_CACHE[cache_key] = topo
    return topo
