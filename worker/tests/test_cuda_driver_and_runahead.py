# Copyright (c) 2026 zstep Contributors
# SPDX-License-Identifier: MIT

"""Unit tests for CudaDriverGuard (CDLL/PyDLL GIL split) and PrefillRunAheadGuard."""

from __future__ import annotations

import numpy as np
import pytest
import torch

from zstep_worker.cuda_driver import (
    CUDA_ERROR_NOT_READY,
    CUDA_ERROR_NOT_SUPPORTED,
    CUDA_SUCCESS,
    CudaDriverError,
    CudaDriverGuard,
)
from zstep_worker.kv_dram_tier import (
    KV_TRANSFER_DIR_D2H,
    KV_TRANSFER_DIR_H2D,
    HostDramKvTier,
)
from zstep_worker.runahead import PrefillRunAheadGuard


class _FakeFunc:
    def __init__(self, impl) -> None:
        self._impl = impl
        self.argtypes = None
        self.restype = None

    def __call__(self, *args):
        return self._impl(*args)


class _FakeCudaLibs:
    """In-memory mock of separate `CDLL` (GIL-free) and `PyDLL` (GIL-kept) handles."""

    def __init__(
        self,
        *,
        batch_version: int = 1,
        batch_return_code: int = CUDA_SUCCESS,
    ) -> None:
        self.calls: list[tuple[str, str, tuple]] = []
        self.next_event_id: int = 100
        self.ready_events: set[int] = set()
        self.batch_return_code = batch_return_code

        self.cdll = type("_FakeCDLL", (), {})()
        self.pydll = type("_FakePyDLL", (), {})()

        if batch_version == 2:
            self.cdll.cuMemcpyBatchAsync_v2 = _FakeFunc(self._batch_v2)
        elif batch_version == 1:
            self.cdll.cuMemcpyBatchAsync = _FakeFunc(self._batch_v1)

        self.cdll.cuMemcpyAsync = _FakeFunc(self._memcpy_async)
        self.cdll.cuEventSynchronize = _FakeFunc(self._event_sync)

        self.pydll.cuEventCreate = _FakeFunc(self._event_create)
        self.pydll.cuEventDestroy_v2 = _FakeFunc(self._event_destroy)
        self.pydll.cuEventRecord = _FakeFunc(self._event_record)
        self.pydll.cuEventQuery = _FakeFunc(self._event_query)
        self.pydll.cuStreamWaitEvent = _FakeFunc(self._stream_wait_event)

    def _batch_v2(self, dsts, srcs, sizes, count, _attrs, _attr_idxs, _num_attrs, stream) -> int:
        triples = [(int(dsts[i]), int(srcs[i]), int(sizes[i])) for i in range(count)]
        self.calls.append(("CDLL", "cuMemcpyBatchAsync_v2", (triples, int(stream.value or 0))))
        return self.batch_return_code

    def _batch_v1(
        self, dsts, srcs, sizes, count, _attrs, _attr_idxs, _num_attrs, _fail_idx, stream
    ) -> int:
        triples = [(int(dsts[i]), int(srcs[i]), int(sizes[i])) for i in range(count)]
        self.calls.append(("CDLL", "cuMemcpyBatchAsync", (triples, int(stream.value or 0))))
        return self.batch_return_code

    def _memcpy_async(self, dst, src, nbytes, stream) -> int:
        self.calls.append(
            ("CDLL", "cuMemcpyAsync", (int(dst), int(src), int(nbytes), int(stream.value or 0)))
        )
        return CUDA_SUCCESS

    def _event_sync(self, event) -> int:
        ev = int(event.value or 0)
        self.calls.append(("CDLL", "cuEventSynchronize", (ev,)))
        self.ready_events.add(ev)
        return CUDA_SUCCESS

    def _event_create(self, out_ptr, flags) -> int:
        ev = self.next_event_id
        self.next_event_id += 1
        out_ptr._obj.value = ev
        self.calls.append(("PyDLL", "cuEventCreate", (ev, int(flags))))
        return CUDA_SUCCESS

    def _event_destroy(self, event) -> int:
        ev = int(event.value or 0)
        self.calls.append(("PyDLL", "cuEventDestroy_v2", (ev,)))
        self.ready_events.discard(ev)
        return CUDA_SUCCESS

    def _event_record(self, event, stream) -> int:
        ev = int(event.value or 0)
        self.ready_events.discard(ev)
        self.calls.append(("PyDLL", "cuEventRecord", (ev, int(stream.value or 0))))
        return CUDA_SUCCESS

    def _event_query(self, event) -> int:
        ev = int(event.value or 0)
        self.calls.append(("PyDLL", "cuEventQuery", (ev,)))
        return CUDA_SUCCESS if ev in self.ready_events else CUDA_ERROR_NOT_READY

    def _stream_wait_event(self, stream, event, flags) -> int:
        self.calls.append(
            ("PyDLL", "cuStreamWaitEvent", (int(stream.value or 0), int(event.value or 0), int(flags)))
        )
        return CUDA_SUCCESS


def test_cuda_driver_guard_cdll_pydll_split_v1_and_v2() -> None:
    for version, expected_mode, expected_api in [
        (1, "batch_v1", "cuMemcpyBatchAsync"),
        (2, "batch_v2", "cuMemcpyBatchAsync_v2"),
    ]:
        libs = _FakeCudaLibs(batch_version=version)
        guard = CudaDriverGuard(libs.cdll, libs.pydll)
        assert guard.batch_memcpy_version == version
        assert guard.has_batch_memcpy is True
        assert guard.has_event_ops is True

        ev = guard.create_event()
        guard.record_event(ev, stream=0x1000)
        guard.stream_wait_event(stream=0x2000, event=ev)
        assert guard.query_event(ev) is False
        mode = guard.memcpy_batch_async([10, 20], [30, 40], [128, 256], stream=0x2000)
        assert mode == expected_mode
        guard.sync_event(ev)
        assert guard.query_event(ev) is True
        guard.destroy_event(ev)

        by_api = [(lib, api) for lib, api, _ in libs.calls]
        assert by_api == [
            ("PyDLL", "cuEventCreate"),
            ("PyDLL", "cuEventRecord"),
            ("PyDLL", "cuStreamWaitEvent"),
            ("PyDLL", "cuEventQuery"),
            ("CDLL", expected_api),
            ("CDLL", "cuEventSynchronize"),
            ("PyDLL", "cuEventQuery"),
            ("PyDLL", "cuEventDestroy_v2"),
        ]


def test_cuda_driver_guard_falls_back_to_async_loop_when_batch_unsupported() -> None:
    libs = _FakeCudaLibs(batch_version=1, batch_return_code=CUDA_ERROR_NOT_SUPPORTED)
    guard = CudaDriverGuard(libs.cdll, libs.pydll)
    mode = guard.memcpy_batch_async([100, 200], [300, 400], [64, 128], stream=0x55)
    assert mode == "async_loop"
    assert guard.has_batch_memcpy is False
    # Subsequent call skips batch symbol directly.
    libs.calls.clear()
    assert guard.memcpy_batch_async([500], [600], [32], stream=0x55) == "async_loop"
    assert libs.calls == [("CDLL", "cuMemcpyAsync", (500, 600, 32, 0x55))]

    with pytest.raises(ValueError, match="length mismatch"):
        guard.memcpy_batch_async([1], [2, 3], [4], stream=0)
    assert guard.memcpy_batch_async([], [], [], stream=0) == "noop"


def test_cuda_driver_guard_raises_on_driver_error() -> None:
    libs = _FakeCudaLibs(batch_version=1, batch_return_code=700)
    guard = CudaDriverGuard(libs.cdll, libs.pydll)
    with pytest.raises(CudaDriverError) as exc_info:
        guard.memcpy_batch_async([1], [2], [4], stream=0)
    assert exc_info.value.code == 700


def test_host_dram_kv_tier_driver_batch_pointer_math_and_event_ordering() -> None:
    libs = _FakeCudaLibs(batch_version=1)
    guard = CudaDriverGuard(libs.cdll, libs.pydll)
    layer0 = torch.zeros((4, 16), dtype=torch.uint8)
    layer1 = torch.zeros((4, 32), dtype=torch.uint8)
    tier = HostDramKvTier(
        [layer0, layer1],
        total_hbm_blocks=4,
        num_dram_blocks=3,
        device=torch.device("cpu"),
        enable_driver_batch_memcpy=True,
        cuda_driver=guard,
    )
    libs.calls.clear()

    tier._dispatch_driver_batch(
        np.array([2, 1], dtype=np.int32),
        np.array([0, 2], dtype=np.int32),
        np.array([KV_TRANSFER_DIR_D2H, KV_TRANSFER_DIR_H2D], dtype=np.int32),
        2,
        compute_stream_ptr=0x111,
        copy_stream_ptr=0x222,
    )
    assert tier.offload_ops == 1
    assert tier.reload_ops == 1
    assert tier.batch_memcpy_calls == 1
    assert tier.batch_memcpy_segments == 4
    assert tier.last_batch_mode == "batch_v1"

    assert [(lib, api) for lib, api, _ in libs.calls] == [
        ("PyDLL", "cuEventRecord"),
        ("PyDLL", "cuStreamWaitEvent"),
        ("CDLL", "cuMemcpyBatchAsync"),
        ("PyDLL", "cuEventRecord"),
        ("PyDLL", "cuStreamWaitEvent"),
    ]
    triples, copy_stream = libs.calls[2][2]
    assert copy_stream == 0x222
    dram_base = int(tier.dram_arena.data_ptr())
    l0_base = int(layer0.data_ptr())
    l1_base = int(layer1.data_ptr())
    # bpb: layer0=16, layer1=32 -> row stride = 48
    assert triples == [
        # Page 0: D2H hbm=2 -> dram=0
        (dram_base + 0 * 48 + 0, l0_base + 2 * 16, 16),
        (dram_base + 0 * 48 + 16, l1_base + 2 * 32, 32),
        # Page 1: H2D dram=2 -> hbm=1
        (l0_base + 1 * 16, dram_base + 2 * 48 + 0, 16),
        (l1_base + 1 * 32, dram_base + 2 * 48 + 16, 32),
    ]

    tier.close()
    tier.close()  # idempotent
    destroyed = [api for _, api, _ in libs.calls if api == "cuEventDestroy_v2"]
    assert len(destroyed) == 2


def test_prefill_runahead_guard_throttling_reaping_and_pool_reuse() -> None:
    libs = _FakeCudaLibs(batch_version=1)
    driver = CudaDriverGuard(libs.cdll, libs.pydll)
    guard = PrefillRunAheadGuard(max_depth=2, backend=driver)
    assert guard.enabled is True

    # Launch 2 steps within max_depth=2 -> zero blocking waits.
    guard.before_launch()
    guard.after_launch(stream=0x10)
    guard.before_launch()
    guard.after_launch(stream=0x10)
    assert guard.in_flight_depth == 2
    assert guard.launched_steps == 2
    assert guard.throttle_waits == 0

    # 3rd launch while oldest is still in-flight -> blocks on oldest via CDLL sync_event.
    guard.before_launch()
    assert guard.throttle_waits == 1
    assert guard.in_flight_depth == 1
    guard.after_launch(stream=0x10)
    assert guard.in_flight_depth == 2

    # Mark both in-flight events ready -> next before_launch reaps non-blockingly!
    libs.ready_events.update(guard._in_flight)
    guard.before_launch()
    assert guard.in_flight_depth == 0
    assert guard.throttle_waits == 1  # unchanged

    # Pool reuse: only 2 event handles were ever created across 3+ launches.
    created = [api for _, api, _ in libs.calls if api == "cuEventCreate"]
    assert len(created) == 2

    guard.after_launch(stream=0x10)
    guard.drain()
    assert guard.in_flight_depth == 0
    guard.close()
    guard.close()  # idempotent
    destroyed = [api for _, api, _ in libs.calls if api == "cuEventDestroy_v2"]
    assert len(destroyed) == 2


def test_prefill_runahead_guard_disabled_when_zero() -> None:
    guard = PrefillRunAheadGuard(max_depth=0)
    assert guard.enabled is False
    guard.before_launch()
    guard.after_launch()
    guard.drain()
    guard.close()
    assert guard.launched_steps == 0
    assert guard.throttle_waits == 0

    with pytest.raises(ValueError, match="max_depth"):
        PrefillRunAheadGuard(max_depth=-1)


def test_prefill_runahead_abort_close_and_error_teardown_safety() -> None:
    libs = _FakeCudaLibs(batch_version=1)
    driver = CudaDriverGuard(libs.cdll, libs.pydll)
    guard = PrefillRunAheadGuard(max_depth=2, backend=driver)

    guard.before_launch()
    guard.after_launch(stream=0x10)
    guard.before_launch()
    guard.after_launch(stream=0x10)
    assert guard.in_flight_depth == 2

    # Make the first cuEventDestroy_v2 fail with sticky CUDA error (700) to verify
    # abort close skips cuEventSynchronize AND still destroys every tracked handle.
    destroy_attempts: list[int] = []

    def _flaky_destroy(event) -> int:
        ev = int(event.value or 0)
        destroy_attempts.append(ev)
        libs.calls.append(("PyDLL", "cuEventDestroy_v2", (ev,)))
        return 700 if len(destroy_attempts) == 1 else CUDA_SUCCESS

    libs.pydll.cuEventDestroy_v2 = _FakeFunc(_flaky_destroy)
    driver._cu_event_destroy = libs.pydll.cuEventDestroy_v2

    libs.calls.clear()
    guard.close(abort=True)
    assert guard.enabled is False
    assert guard.in_flight_depth == 0
    assert not any(api == "cuEventSynchronize" for _, api, _ in libs.calls)
    assert len(destroy_attempts) == 2


def test_host_dram_kv_tier_partial_init_and_sticky_error_close_safety() -> None:
    libs = _FakeCudaLibs(batch_version=1)
    created_count = 0

    def _fail_second_create(out_ptr, flags) -> int:
        nonlocal created_count
        created_count += 1
        if created_count == 2:
            return 700
        return libs._event_create(out_ptr, flags)

    libs.pydll.cuEventCreate = _FakeFunc(_fail_second_create)
    driver = CudaDriverGuard(libs.cdll, libs.pydll)
    layer0 = torch.zeros((2, 16), dtype=torch.uint8)
    with pytest.raises(CudaDriverError):
        HostDramKvTier(
            [layer0],
            total_hbm_blocks=2,
            num_dram_blocks=2,
            device=torch.device("cpu"),
            enable_driver_batch_memcpy=True,
            cuda_driver=driver,
        )
    assert [api for _, api, _ in libs.calls if api == "cuEventDestroy_v2"] == ["cuEventDestroy_v2"]

    # Sticky error during HostDramKvTier.close() must not raise or skip second event.
    libs2 = _FakeCudaLibs(batch_version=1)
    driver2 = CudaDriverGuard(libs2.cdll, libs2.pydll)
    tier = HostDramKvTier(
        [layer0],
        total_hbm_blocks=2,
        num_dram_blocks=2,
        device=torch.device("cpu"),
        enable_driver_batch_memcpy=True,
        cuda_driver=driver2,
    )
    destroyed: list[int] = []

    def _fail_first_destroy(event) -> int:
        ev = int(event.value or 0)
        destroyed.append(ev)
        return 700 if len(destroyed) == 1 else CUDA_SUCCESS

    libs2.pydll.cuEventDestroy_v2 = _FakeFunc(_fail_first_destroy)
    driver2._cu_event_destroy = libs2.pydll.cuEventDestroy_v2
    tier.close()
    assert len(destroyed) == 2
    assert tier._cuda_driver is None


def test_cuda_driver_guard_vmm_fixed_va_alloc_access_and_idempotent_free() -> None:
    from zstep_worker.cuda_driver import DEFAULT_VMM_BASE_VA

    libs = _FakeCudaLibs(batch_version=2)
    vmm_calls: list[tuple[str, tuple]] = []

    def _gran(out_gran, _prop, _option) -> int:
        out_gran._obj.value = 2 * 1024 * 1024
        vmm_calls.append(("cuMemGetAllocationGranularity", ()))
        return CUDA_SUCCESS

    def _create(out_handle, size, _prop, _flags) -> int:
        out_handle._obj.value = 0xCAFE_1234
        vmm_calls.append(("cuMemCreate", (int(size),)))
        return CUDA_SUCCESS

    def _reserve(out_ptr, size, align, addr, _flags) -> int:
        out_ptr._obj.value = int(addr)
        vmm_calls.append(("cuMemAddressReserve", (int(size), int(align), int(addr))))
        return CUDA_SUCCESS

    def _map(ptr, size, _off, handle, _flags) -> int:
        vmm_calls.append(("cuMemMap", (int(ptr), int(size), int(handle))))
        return CUDA_SUCCESS

    def _set_access(ptr, size, _descs, count) -> int:
        vmm_calls.append(("cuMemSetAccess", (int(ptr), int(size), int(count))))
        return CUDA_SUCCESS

    def _unmap(ptr, size) -> int:
        vmm_calls.append(("cuMemUnmap", (int(ptr), int(size))))
        return CUDA_SUCCESS

    def _addr_free(ptr, size) -> int:
        vmm_calls.append(("cuMemAddressFree", (int(ptr), int(size))))
        return CUDA_SUCCESS

    def _release(handle) -> int:
        vmm_calls.append(("cuMemRelease", (int(handle),)))
        return CUDA_SUCCESS

    libs.cdll.cuMemGetAllocationGranularity = _FakeFunc(_gran)
    libs.cdll.cuMemCreate = _FakeFunc(_create)
    libs.cdll.cuMemAddressReserve = _FakeFunc(_reserve)
    libs.cdll.cuMemMap = _FakeFunc(_map)
    libs.cdll.cuMemSetAccess = _FakeFunc(_set_access)
    libs.cdll.cuMemUnmap = _FakeFunc(_unmap)
    libs.cdll.cuMemAddressFree = _FakeFunc(_addr_free)
    libs.cdll.cuMemRelease = _FakeFunc(_release)

    guard = CudaDriverGuard(libs.cdll, libs.pydll)
    assert guard.has_vmm_ops is True
    h = guard.vmm_alloc_and_map_fixed_va(
        device_id=0,
        size_bytes=3 * 1024 * 1024,  # rounds up to 4 MiB (2 MiB granularity)
        target_va=DEFAULT_VMM_BASE_VA,
        peer_device_ids=(1, 2, 3),
    )
    assert h.va_ptr == DEFAULT_VMM_BASE_VA
    assert h.exact_va_honored is True
    assert h.size_bytes == 4 * 1024 * 1024
    assert h.granularity_bytes == 2 * 1024 * 1024
    assert h.peer_device_ids == (1, 2, 3)

    guard.vmm_free(h)
    guard.vmm_free(h)  # idempotent
    assert [name for name, _ in vmm_calls] == [
        "cuMemGetAllocationGranularity",
        "cuMemCreate",
        "cuMemAddressReserve",
        "cuMemMap",
        "cuMemSetAccess",
        "cuMemUnmap",
        "cuMemAddressFree",
        "cuMemRelease",
    ]
