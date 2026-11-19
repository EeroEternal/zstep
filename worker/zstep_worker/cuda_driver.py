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
import logging
from typing import Any

_LOG = logging.getLogger(__name__)

CUDA_SUCCESS: int = 0
CUDA_ERROR_NOT_READY: int = 600
CUDA_ERROR_NOT_SUPPORTED: int = 801
CU_EVENT_DISABLE_TIMING: int = 0x2


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
