# Copyright (c) 2026 zstep Contributors
# SPDX-License-Identifier: MIT

"""ctypes CUDA Driver API guardrails for GIL-free KV transfers and low-jitter sync.

Separates libcuda calls across two ctypes library handles:
- `ctypes.CDLL("libcuda.so.1")` releases the Python GIL for potentially long-running
  operations (`cuMemcpyBatchAsync`, `cuMemcpyAsync`, `cuEventSynchronize`), preventing
  background/offload KV transfers from stalling the Python worker thread.
- `ctypes.PyDLL("libcuda.so.1")` retains the Python GIL for microsecond non-blocking
  driver calls (`cuEventCreate`, `cuEventDestroy_v2`, `cuEventRecord`, `cuEventQuery`,
  `cuStreamWaitEvent`), avoiding GIL release/reacquire churn on the main compute loop.
"""

from __future__ import annotations

from collections.abc import Sequence
import ctypes
from dataclasses import dataclass
import logging
import os
from typing import Any

_LOG = logging.getLogger(__name__)

CUDA_SUCCESS: int = 0
CUDA_ERROR_NOT_READY: int = 600
CUDA_ERROR_NOT_SUPPORTED: int = 801
CU_EVENT_DISABLE_TIMING: int = 0x2

CU_MEM_ALLOC_TYPE_PINNED: int = 1
CU_MEM_LOCATION_TYPE_DEVICE: int = 1
CU_MEM_HANDLE_TYPE_NONE: int = 0
CU_MEM_HANDLE_TYPE_POSIX_FILE_DESCRIPTOR: int = 1
CU_MEM_ALLOC_GRANULARITY_RECOMMENDED: int = 1
CU_MEM_ACCESS_FLAGS_PROT_READWRITE: int = 3
DEFAULT_VMM_BASE_VA: int = 0x0800_0000_0000
DEFAULT_VMM_RANK_STRIDE: int = 1 << 40


class CUmemLocation(ctypes.Structure):
    _fields_ = [
        ("type", ctypes.c_int),
        ("id", ctypes.c_int),
    ]


class CUmemAllocationProp(ctypes.Structure):
    _fields_ = [
        ("type", ctypes.c_int),
        ("requestedHandleTypes", ctypes.c_int),
        ("location", CUmemLocation),
        ("win32HandleMetaData", ctypes.c_void_p),
        ("allocFlags", ctypes.c_ubyte * 8),
        ("_reserved_tail", ctypes.c_ubyte * 32),
    ]


class CUmemAccessDesc(ctypes.Structure):
    _fields_ = [
        ("location", CUmemLocation),
        ("flags", ctypes.c_int),
    ]


@dataclass
class VmmAllocationHandle:
    """CUDA 13 Virtual Memory Management fixed-VA allocation and optional POSIX FD."""

    va_ptr: int
    size_bytes: int
    granularity_bytes: int
    alloc_handle: int
    device_id: int
    exact_va_honored: bool
    exported_fd: int = -1
    peer_device_ids: tuple[int, ...] = ()
    closed: bool = False


class CudaDriverError(RuntimeError):
    """Raised when a CUDA Driver API call returns a non-zero CUresult."""

    def __init__(self, api: str, code: int) -> None:
        super().__init__(f"{api} failed with CUresult={code}")
        self.api = api
        self.code = int(code)


def _check_cu(api: str, code: int) -> None:
    if code != CUDA_SUCCESS:
        raise CudaDriverError(api, code)


class CudaDriverGuard:
    """Thin ctypes wrapper over `libcuda.so.1` with explicit CDLL / PyDLL GIL split."""

    def __init__(self, cdll: Any, pydll: Any) -> None:
        self._cdll = cdll
        self._pydll = pydll
        self._batch_version: int = 0
        self._cu_memcpy_batch_async: Any = None
        self._cu_memcpy_async: Any = None
        self._cu_event_sync: Any = None
        self._cu_event_create: Any = None
        self._cu_event_destroy: Any = None
        self._cu_event_record: Any = None
        self._cu_event_query: Any = None
        self._cu_stream_wait_event: Any = None
        self._cu_mem_get_granularity: Any = None
        self._cu_mem_create: Any = None
        self._cu_mem_release: Any = None
        self._cu_mem_addr_reserve: Any = None
        self._cu_mem_addr_free: Any = None
        self._cu_mem_map: Any = None
        self._cu_mem_unmap: Any = None
        self._cu_mem_set_access: Any = None
        self._cu_mem_export_fd: Any = None
        self._cu_mem_import_fd: Any = None
        self._bind_symbols()

    def _bind_symbols(self) -> None:
        # 1. GIL-releasing CDLL symbols (batch/async memcpy + blocking event sync).
        batch_v2 = getattr(self._cdll, "cuMemcpyBatchAsync_v2", None)
        batch_v1 = getattr(self._cdll, "cuMemcpyBatchAsync", None)
        if batch_v2 is not None:
            batch_v2.argtypes = [
                ctypes.POINTER(ctypes.c_uint64),  # dsts
                ctypes.POINTER(ctypes.c_uint64),  # srcs
                ctypes.POINTER(ctypes.c_size_t),  # sizes
                ctypes.c_size_t,  # count
                ctypes.c_void_p,  # attrs
                ctypes.POINTER(ctypes.c_size_t),  # attrsIdxs
                ctypes.c_size_t,  # numAttrs
                ctypes.c_void_p,  # hStream
            ]
            batch_v2.restype = ctypes.c_int
            self._cu_memcpy_batch_async = batch_v2
            self._batch_version = 2
        elif batch_v1 is not None:
            batch_v1.argtypes = [
                ctypes.POINTER(ctypes.c_uint64),  # dsts
                ctypes.POINTER(ctypes.c_uint64),  # srcs
                ctypes.POINTER(ctypes.c_size_t),  # sizes
                ctypes.c_size_t,  # count
                ctypes.c_void_p,  # attrs
                ctypes.POINTER(ctypes.c_size_t),  # attrsIdxs
                ctypes.c_size_t,  # numAttrs
                ctypes.POINTER(ctypes.c_size_t),  # failIdx
                ctypes.c_void_p,  # hStream
            ]
            batch_v1.restype = ctypes.c_int
            self._cu_memcpy_batch_async = batch_v1
            self._batch_version = 1

        memcpy_async = getattr(self._cdll, "cuMemcpyAsync", None)
        if memcpy_async is not None:
            memcpy_async.argtypes = [
                ctypes.c_uint64,  # dst
                ctypes.c_uint64,  # src
                ctypes.c_size_t,  # ByteCount
                ctypes.c_void_p,  # hStream
            ]
            memcpy_async.restype = ctypes.c_int
            self._cu_memcpy_async = memcpy_async

        event_sync = getattr(self._cdll, "cuEventSynchronize", None)
        if event_sync is not None:
            event_sync.argtypes = [ctypes.c_void_p]
            event_sync.restype = ctypes.c_int
            self._cu_event_sync = event_sync

        # 2. GIL-retaining PyDLL symbols (microsecond non-blocking event/stream ops).
        event_create = getattr(self._pydll, "cuEventCreate", None)
        if event_create is not None:
            event_create.argtypes = [ctypes.POINTER(ctypes.c_void_p), ctypes.c_uint]
            event_create.restype = ctypes.c_int
            self._cu_event_create = event_create

        event_destroy = getattr(self._pydll, "cuEventDestroy_v2", None) or getattr(
            self._pydll, "cuEventDestroy", None
        )
        if event_destroy is not None:
            event_destroy.argtypes = [ctypes.c_void_p]
            event_destroy.restype = ctypes.c_int
            self._cu_event_destroy = event_destroy

        event_record = getattr(self._pydll, "cuEventRecord", None)
        if event_record is not None:
            event_record.argtypes = [ctypes.c_void_p, ctypes.c_void_p]
            event_record.restype = ctypes.c_int
            self._cu_event_record = event_record

        event_query = getattr(self._pydll, "cuEventQuery", None)
        if event_query is not None:
            event_query.argtypes = [ctypes.c_void_p]
            event_query.restype = ctypes.c_int
            self._cu_event_query = event_query

        stream_wait = getattr(self._pydll, "cuStreamWaitEvent", None)
        if stream_wait is not None:
            stream_wait.argtypes = [ctypes.c_void_p, ctypes.c_void_p, ctypes.c_uint]
            stream_wait.restype = ctypes.c_int
            self._cu_stream_wait_event = stream_wait

        # 3. CUDA Virtual Memory Management (VMM) symbols via CDLL (control-plane setup).
        gran = getattr(self._cdll, "cuMemGetAllocationGranularity", None)
        if gran is not None:
            gran.argtypes = [
                ctypes.POINTER(ctypes.c_size_t),
                ctypes.POINTER(CUmemAllocationProp),
                ctypes.c_int,
            ]
            gran.restype = ctypes.c_int
            self._cu_mem_get_granularity = gran

        mem_create = getattr(self._cdll, "cuMemCreate", None)
        if mem_create is not None:
            mem_create.argtypes = [
                ctypes.POINTER(ctypes.c_uint64),
                ctypes.c_size_t,
                ctypes.POINTER(CUmemAllocationProp),
                ctypes.c_ulonglong,
            ]
            mem_create.restype = ctypes.c_int
            self._cu_mem_create = mem_create

        mem_release = getattr(self._cdll, "cuMemRelease", None)
        if mem_release is not None:
            mem_release.argtypes = [ctypes.c_uint64]
            mem_release.restype = ctypes.c_int
            self._cu_mem_release = mem_release

        addr_res = getattr(self._cdll, "cuMemAddressReserve", None)
        if addr_res is not None:
            addr_res.argtypes = [
                ctypes.POINTER(ctypes.c_uint64),
                ctypes.c_size_t,
                ctypes.c_size_t,
                ctypes.c_uint64,
                ctypes.c_ulonglong,
            ]
            addr_res.restype = ctypes.c_int
            self._cu_mem_addr_reserve = addr_res

        addr_free = getattr(self._cdll, "cuMemAddressFree", None)
        if addr_free is not None:
            addr_free.argtypes = [ctypes.c_uint64, ctypes.c_size_t]
            addr_free.restype = ctypes.c_int
            self._cu_mem_addr_free = addr_free

        mem_map = getattr(self._cdll, "cuMemMap", None)
        if mem_map is not None:
            mem_map.argtypes = [
                ctypes.c_uint64,
                ctypes.c_size_t,
                ctypes.c_size_t,
                ctypes.c_uint64,
                ctypes.c_ulonglong,
            ]
            mem_map.restype = ctypes.c_int
            self._cu_mem_map = mem_map

        mem_unmap = getattr(self._cdll, "cuMemUnmap", None)
        if mem_unmap is not None:
            mem_unmap.argtypes = [ctypes.c_uint64, ctypes.c_size_t]
            mem_unmap.restype = ctypes.c_int
            self._cu_mem_unmap = mem_unmap

        set_acc = getattr(self._cdll, "cuMemSetAccess", None)
        if set_acc is not None:
            set_acc.argtypes = [
                ctypes.c_uint64,
                ctypes.c_size_t,
                ctypes.POINTER(CUmemAccessDesc),
                ctypes.c_size_t,
            ]
            set_acc.restype = ctypes.c_int
            self._cu_mem_set_access = set_acc

        exp_fd = getattr(self._cdll, "cuMemExportToShareableHandle", None)
        if exp_fd is not None:
            exp_fd.argtypes = [
                ctypes.c_void_p,
                ctypes.c_uint64,
                ctypes.c_int,
                ctypes.c_ulonglong,
            ]
            exp_fd.restype = ctypes.c_int
            self._cu_mem_export_fd = exp_fd

        imp_fd = getattr(self._cdll, "cuMemImportFromShareableHandle", None)
        if imp_fd is not None:
            imp_fd.argtypes = [
                ctypes.POINTER(ctypes.c_uint64),
                ctypes.c_void_p,
                ctypes.c_int,
            ]
            imp_fd.restype = ctypes.c_int
            self._cu_mem_import_fd = imp_fd

    @property
    def batch_memcpy_version(self) -> int:
        """Return 2 (`cuMemcpyBatchAsync_v2`), 1 (`cuMemcpyBatchAsync`), or 0 (fallback)."""
        return self._batch_version

    @property
    def has_batch_memcpy(self) -> bool:
        return self._cu_memcpy_batch_async is not None

    @property
    def has_event_ops(self) -> bool:
        return all(
            fn is not None
            for fn in (
                self._cu_event_create,
                self._cu_event_destroy,
                self._cu_event_record,
                self._cu_event_query,
                self._cu_stream_wait_event,
                self._cu_event_sync,
            )
        )

    def create_event(self, flags: int = CU_EVENT_DISABLE_TIMING) -> int:
        if self._cu_event_create is None:
            raise CudaDriverError("cuEventCreate", CUDA_ERROR_NOT_SUPPORTED)
        handle = ctypes.c_void_p(0)
        _check_cu("cuEventCreate", int(self._cu_event_create(ctypes.byref(handle), int(flags))))
        return int(handle.value or 0)

    def destroy_event(self, event: int) -> None:
        if not event:
            return
        if self._cu_event_destroy is None:
            raise CudaDriverError("cuEventDestroy_v2", CUDA_ERROR_NOT_SUPPORTED)
        _check_cu("cuEventDestroy_v2", int(self._cu_event_destroy(ctypes.c_void_p(int(event)))))

    def record_event(self, event: int, stream: int) -> None:
        """Record `event` on `stream` via `PyDLL` (keeps GIL, microsecond call)."""
        if self._cu_event_record is None:
            raise CudaDriverError("cuEventRecord", CUDA_ERROR_NOT_SUPPORTED)
        _check_cu(
            "cuEventRecord",
            int(self._cu_event_record(ctypes.c_void_p(int(event)), ctypes.c_void_p(int(stream)))),
        )

    def stream_wait_event(self, stream: int, event: int, flags: int = 0) -> None:
        """Make `stream` wait on `event` via `PyDLL` (keeps GIL, microsecond call)."""
        if self._cu_stream_wait_event is None:
            raise CudaDriverError("cuStreamWaitEvent", CUDA_ERROR_NOT_SUPPORTED)
        _check_cu(
            "cuStreamWaitEvent",
            int(
                self._cu_stream_wait_event(
                    ctypes.c_void_p(int(stream)),
                    ctypes.c_void_p(int(event)),
                    int(flags),
                )
            ),
        )

    def query_event(self, event: int) -> bool:
        """Non-blocking completion poll via `PyDLL` (`True` if complete, `False` if pending)."""
        if self._cu_event_query is None:
            raise CudaDriverError("cuEventQuery", CUDA_ERROR_NOT_SUPPORTED)
        res = int(self._cu_event_query(ctypes.c_void_p(int(event))))
        if res == CUDA_SUCCESS:
            return True
        if res == CUDA_ERROR_NOT_READY:
            return False
        raise CudaDriverError("cuEventQuery", res)

    def sync_event(self, event: int) -> None:
        """Blocking wait for `event` via `CDLL` (releases GIL while waiting)."""
        if self._cu_event_sync is None:
            raise CudaDriverError("cuEventSynchronize", CUDA_ERROR_NOT_SUPPORTED)
        _check_cu("cuEventSynchronize", int(self._cu_event_sync(ctypes.c_void_p(int(event)))))

    def memcpy_batch_async(
        self,
        dsts: Sequence[int],
        srcs: Sequence[int],
        sizes: Sequence[int],
        stream: int,
    ) -> str:
        """Launch batched asynchronous device/host copies via `CDLL` (releases GIL).

        Uses CUDA 13.0+ `cuMemcpyBatchAsync_v2` or CUDA 12.8+ `cuMemcpyBatchAsync`
        when available, falling back to a `CDLL` `cuMemcpyAsync` loop if unsupported.
        Returns the dispatch mode (`"batch_v2"`, `"batch_v1"`, `"async_loop"`, or `"noop"`).
        """
        count = len(dsts)
        if len(srcs) != count or len(sizes) != count:
            raise ValueError(
                f"memcpy_batch_async length mismatch: dsts={count}, srcs={len(srcs)}, sizes={len(sizes)}"
            )
        if count == 0:
            return "noop"

        if self._cu_memcpy_batch_async is not None:
            dst_arr = (ctypes.c_uint64 * count)(*(int(x) for x in dsts))
            src_arr = (ctypes.c_uint64 * count)(*(int(x) for x in srcs))
            size_arr = (ctypes.c_size_t * count)(*(int(x) for x in sizes))
            if self._batch_version == 2:
                res = int(
                    self._cu_memcpy_batch_async(
                        dst_arr,
                        src_arr,
                        size_arr,
                        count,
                        None,
                        None,
                        0,
                        ctypes.c_void_p(int(stream)),
                    )
                )
                if res == CUDA_SUCCESS:
                    return "batch_v2"
                if res != CUDA_ERROR_NOT_SUPPORTED:
                    raise CudaDriverError("cuMemcpyBatchAsync_v2", res)
            else:
                fail_idx = ctypes.c_size_t(0)
                res = int(
                    self._cu_memcpy_batch_async(
                        dst_arr,
                        src_arr,
                        size_arr,
                        count,
                        None,
                        None,
                        0,
                        ctypes.byref(fail_idx),
                        ctypes.c_void_p(int(stream)),
                    )
                )
                if res == CUDA_SUCCESS:
                    return "batch_v1"
                if res != CUDA_ERROR_NOT_SUPPORTED:
                    raise CudaDriverError(
                        f"cuMemcpyBatchAsync(failIdx={fail_idx.value})", res
                    )
            # Permanently disable batch path on drivers reporting NOT_SUPPORTED.
            self._cu_memcpy_batch_async = None
            self._batch_version = 0

        if self._cu_memcpy_async is None:
            raise CudaDriverError("cuMemcpyAsync", CUDA_ERROR_NOT_SUPPORTED)
        stream_ptr = ctypes.c_void_p(int(stream))
        for dst, src, nbytes in zip(dsts, srcs, sizes, strict=True):
            _check_cu(
                "cuMemcpyAsync",
                int(self._cu_memcpy_async(int(dst), int(src), int(nbytes), stream_ptr)),
            )
        return "async_loop"

    @property
    def has_vmm_ops(self) -> bool:
        return all(
            fn is not None
            for fn in (
                self._cu_mem_get_granularity,
                self._cu_mem_create,
                self._cu_mem_release,
                self._cu_mem_addr_reserve,
                self._cu_mem_addr_free,
                self._cu_mem_map,
                self._cu_mem_unmap,
                self._cu_mem_set_access,
            )
        )

    def _make_vmm_prop(self, device_id: int, *, export_posix_fd: bool) -> CUmemAllocationProp:
        prop = CUmemAllocationProp()
        prop.type = CU_MEM_ALLOC_TYPE_PINNED
        prop.requestedHandleTypes = (
            CU_MEM_HANDLE_TYPE_POSIX_FILE_DESCRIPTOR
            if export_posix_fd
            else CU_MEM_HANDLE_TYPE_NONE
        )
        prop.location.type = CU_MEM_LOCATION_TYPE_DEVICE
        prop.location.id = int(device_id)
        return prop

    def vmm_allocation_granularity(
        self, device_id: int, *, export_posix_fd: bool = False
    ) -> int:
        if not self.has_vmm_ops:
            raise CudaDriverError("cuMemGetAllocationGranularity", CUDA_ERROR_NOT_SUPPORTED)
        prop = self._make_vmm_prop(device_id, export_posix_fd=export_posix_fd)
        gran = ctypes.c_size_t(0)
        _check_cu(
            "cuMemGetAllocationGranularity",
            int(
                self._cu_mem_get_granularity(
                    ctypes.byref(gran),
                    ctypes.byref(prop),
                    CU_MEM_ALLOC_GRANULARITY_RECOMMENDED,
                )
            ),
        )
        return int(gran.value)

    def vmm_alloc_and_map_fixed_va(
        self,
        *,
        device_id: int,
        size_bytes: int,
        target_va: int = DEFAULT_VMM_BASE_VA,
        peer_device_ids: Sequence[int] = (),
        export_posix_fd: bool = False,
    ) -> VmmAllocationHandle:
        """Allocate physical HBM via `cuMemCreate` and map at deterministic fixed VA."""
        if size_bytes <= 0:
            raise ValueError(f"size_bytes must be > 0 (got {size_bytes})")
        if not self.has_vmm_ops:
            raise CudaDriverError("cuMemCreate", CUDA_ERROR_NOT_SUPPORTED)

        gran = self.vmm_allocation_granularity(device_id, export_posix_fd=export_posix_fd)
        aligned_size = ((int(size_bytes) + gran - 1) // gran) * gran
        prop = self._make_vmm_prop(device_id, export_posix_fd=export_posix_fd)

        alloc_handle = ctypes.c_uint64(0)
        _check_cu(
            "cuMemCreate",
            int(self._cu_mem_create(ctypes.byref(alloc_handle), aligned_size, ctypes.byref(prop), 0)),
        )
        va_ptr = ctypes.c_uint64(0)
        mapped = False
        fd_val = -1
        try:
            _check_cu(
                "cuMemAddressReserve",
                int(
                    self._cu_mem_addr_reserve(
                        ctypes.byref(va_ptr),
                        aligned_size,
                        gran,
                        int(target_va),
                        0,
                    )
                ),
            )
            _check_cu(
                "cuMemMap",
                int(self._cu_mem_map(va_ptr.value, aligned_size, 0, alloc_handle.value, 0)),
            )
            mapped = True

            access_devs = [int(device_id)]
            for pid in peer_device_ids:
                if int(pid) not in access_devs:
                    access_devs.append(int(pid))
            desc_arr = (CUmemAccessDesc * len(access_devs))()
            for idx, did in enumerate(access_devs):
                desc_arr[idx].location.type = CU_MEM_LOCATION_TYPE_DEVICE
                desc_arr[idx].location.id = did
                desc_arr[idx].flags = CU_MEM_ACCESS_FLAGS_PROT_READWRITE
            _check_cu(
                "cuMemSetAccess",
                int(self._cu_mem_set_access(va_ptr.value, aligned_size, desc_arr, len(access_devs))),
            )

            if export_posix_fd:
                if self._cu_mem_export_fd is None:
                    raise CudaDriverError("cuMemExportToShareableHandle", CUDA_ERROR_NOT_SUPPORTED)
                raw_fd = ctypes.c_int(-1)
                _check_cu(
                    "cuMemExportToShareableHandle",
                    int(
                        self._cu_mem_export_fd(
                            ctypes.byref(raw_fd),
                            alloc_handle.value,
                            CU_MEM_HANDLE_TYPE_POSIX_FILE_DESCRIPTOR,
                            0,
                        )
                    ),
                )
                fd_val = int(raw_fd.value)
        except BaseException:
            if fd_val >= 0:
                try:
                    os.close(fd_val)
                except OSError:
                    pass
            if mapped and va_ptr.value:
                self._cu_mem_unmap(va_ptr.value, aligned_size)
            if va_ptr.value:
                self._cu_mem_addr_free(va_ptr.value, aligned_size)
            if alloc_handle.value:
                self._cu_mem_release(alloc_handle.value)
            raise

        return VmmAllocationHandle(
            va_ptr=int(va_ptr.value),
            size_bytes=aligned_size,
            granularity_bytes=gran,
            alloc_handle=int(alloc_handle.value),
            device_id=int(device_id),
            exact_va_honored=(int(va_ptr.value) == int(target_va)),
            exported_fd=fd_val,
            peer_device_ids=tuple(int(p) for p in peer_device_ids),
        )

    def vmm_free(self, handle: VmmAllocationHandle) -> None:
        """Unmap, free reserved VA, close exported POSIX FD, and release physical HBM idempotently."""
        if handle.closed:
            return
        handle.closed = True
        if handle.exported_fd >= 0:
            try:
                os.close(handle.exported_fd)
            except OSError:
                pass
            handle.exported_fd = -1
        if handle.va_ptr and self._cu_mem_unmap is not None:
            res = int(self._cu_mem_unmap(handle.va_ptr, handle.size_bytes))
            if res != CUDA_SUCCESS:
                _LOG.warning("cuMemUnmap(%#x) returned CUresult=%d", handle.va_ptr, res)
        if handle.va_ptr and self._cu_mem_addr_free is not None:
            res = int(self._cu_mem_addr_free(handle.va_ptr, handle.size_bytes))
            if res != CUDA_SUCCESS:
                _LOG.warning("cuMemAddressFree(%#x) returned CUresult=%d", handle.va_ptr, res)
        if handle.alloc_handle and self._cu_mem_release is not None:
            res = int(self._cu_mem_release(handle.alloc_handle))
            if res != CUDA_SUCCESS:
                _LOG.warning("cuMemRelease(%#x) returned CUresult=%d", handle.alloc_handle, res)


_CACHED_DRIVER: CudaDriverGuard | None = None
_DRIVER_PROBED: bool = False


def load_cuda_driver() -> CudaDriverGuard | None:
    """Load and cache `CudaDriverGuard` from `libcuda.so.1`, or `None` if unavailable."""
    global _CACHED_DRIVER, _DRIVER_PROBED
    if _DRIVER_PROBED:
        return _CACHED_DRIVER
    _DRIVER_PROBED = True
    try:
        cdll = ctypes.CDLL("libcuda.so.1")
        pydll = ctypes.PyDLL("libcuda.so.1")
        guard = CudaDriverGuard(cdll, pydll)
        if not guard.has_event_ops:
            _LOG.warning("libcuda.so.1 loaded but missing required event symbols; disabling guard")
            return None
        _CACHED_DRIVER = guard
        _LOG.info(
            "Loaded CudaDriverGuard from libcuda.so.1 (batch_memcpy_version=%d)",
            guard.batch_memcpy_version,
        )
    except OSError:
        _CACHED_DRIVER = None
    return _CACHED_DRIVER
