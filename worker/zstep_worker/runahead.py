# Copyright (c) 2026 zstep Contributors
# SPDX-License-Identifier: MIT

"""Explicit Prefill RunAhead launch-depth guardrail for eager GPU steps.

Extends the decode `in_flight_depth=1` discipline to eager chunked-prefill steps so
back-to-back multi-layer launches cannot saturate the CUDA driver's ~1024-kernel
submission queue and stall the worker main loop or heartbeat thread.
"""

from __future__ import annotations

from collections import deque
import logging
from typing import Protocol

from zstep_worker.cuda_driver import CU_EVENT_DISABLE_TIMING, CudaDriverGuard, load_cuda_driver

_LOG = logging.getLogger(__name__)


class _EventBackend(Protocol):
    def create_event(self, flags: int = CU_EVENT_DISABLE_TIMING) -> int: ...
    def destroy_event(self, event: int) -> None: ...
    def record_event(self, event: int, stream: int) -> None: ...
    def query_event(self, event: int) -> bool: ...
    def sync_event(self, event: int) -> None: ...


class _TorchEventAdapter:
    """Fallback backend using `torch.cuda.Event` when `libcuda.so.1` is unavailable."""

    def __init__(self, device: object) -> None:
        self._device = device
        self._next_id: int = 1
        self._events: dict[int, object] = {}

    def create_event(self, flags: int = CU_EVENT_DISABLE_TIMING) -> int:
        import torch

        del flags
        ev = torch.cuda.Event(enable_timing=False)
        token = self._next_id
        self._next_id += 1
        self._events[token] = ev
        return token

    def destroy_event(self, event: int) -> None:
        self._events.pop(event, None)

    def record_event(self, event: int, stream: int) -> None:
        import torch

        del stream
        ev = self._events[event]
        ev.record(torch.cuda.current_stream(self._device))  # type: ignore[attr-defined]

    def query_event(self, event: int) -> bool:
        return bool(self._events[event].query())  # type: ignore[attr-defined]

    def sync_event(self, event: int) -> None:
        self._events[event].synchronize()  # type: ignore[attr-defined]


class PrefillRunAheadGuard:
    """Caps concurrent in-flight eager prefill steps using a pooled ring of CUDA events.

    When `max_depth <= 0`, throttling is disabled and all methods become no-ops.
    """

    def __init__(
        self,
        max_depth: int = 2,
        *,
        device: object | None = None,
        backend: _EventBackend | None = None,
    ) -> None:
        if max_depth < 0:
            raise ValueError(f"max_depth must be >= 0 (got {max_depth})")
        self.max_depth: int = int(max_depth)
        self._device = device
        self._backend: _EventBackend | None = None
        if self.max_depth > 0:
            if backend is not None:
                self._backend = backend
            elif device is not None and getattr(device, "type", None) == "cuda":
                driver: CudaDriverGuard | None = load_cuda_driver()
                self._backend = driver if driver is not None else _TorchEventAdapter(device)

        self._in_flight: deque[int] = deque()
        self._pool: list[int] = []
        self._closed: bool = False
        self.launched_steps: int = 0
        self.throttle_waits: int = 0
        self.reaped_events: int = 0

    @property
    def enabled(self) -> bool:
        return not self._closed and self.max_depth > 0 and self._backend is not None

    @property
    def in_flight_depth(self) -> int:
        return len(self._in_flight)

    def _reap_completed(self) -> None:
        if self._backend is None:
            return
        while self._in_flight and self._backend.query_event(self._in_flight[0]):
            ev = self._in_flight.popleft()
            self._pool.append(ev)
            self.reaped_events += 1

    def before_launch(self) -> None:
        """Ensure in-flight eager prefill depth is strictly below `max_depth`."""
        if not self.enabled or self._backend is None:
            return
        self._reap_completed()
        while len(self._in_flight) >= self.max_depth:
            oldest = self._in_flight.popleft()
            try:
                self._backend.sync_event(oldest)
            finally:
                self._pool.append(oldest)
            self.throttle_waits += 1
            self.reaped_events += 1

    def after_launch(self, stream: int | None = None) -> None:
        """Record a completion fence for the just-launched eager prefill step."""
        if not self.enabled or self._backend is None:
            return
        ev = self._pool.pop() if self._pool else self._backend.create_event()
        try:
            if stream is None:
                if isinstance(self._backend, CudaDriverGuard) and self._device is not None:
                    import torch

                    stream_ptr = int(torch.cuda.current_stream(self._device).cuda_stream)
                else:
                    stream_ptr = 0
            else:
                stream_ptr = int(stream)
            self._backend.record_event(ev, stream_ptr)
        except BaseException:
            self._pool.append(ev)
            raise
        self._in_flight.append(ev)
        self.launched_steps += 1

    def drain(self) -> None:
        """Wait for all in-flight prefill fence events and return them to the pool."""
        if not self.enabled or self._backend is None:
            return
        while self._in_flight:
            ev = self._in_flight.popleft()
            try:
                if not self._backend.query_event(ev):
                    self._backend.sync_event(ev)
            finally:
                self._pool.append(ev)
            self.reaped_events += 1

    def close(self, *, abort: bool = False) -> None:
        """Destroy all tracked CUDA driver event handles safely and idempotently.

        When `abort=True`, skips blocking `drain()` so aborting on a hung GPU kernel
        never deadlocks in `cuEventSynchronize`. Driver errors during teardown are
        logged and suppressed so every event handle is attempted and caller cleanup
        always proceeds.
        """
        if self._closed:
            return
        if self._backend is None:
            self._closed = True
            return
        backend = self._backend
        try:
            if not abort:
                try:
                    self.drain()
                except Exception as exc:
                    _LOG.warning("PrefillRunAheadGuard.drain() failed during close: %s", exc)
        finally:
            self._closed = True
            while self._in_flight:
                self._pool.append(self._in_flight.popleft())
            while self._pool:
                ev = self._pool.pop()
                try:
                    backend.destroy_event(ev)
                except Exception as exc:
                    _LOG.warning("PrefillRunAheadGuard.destroy_event(%s) failed: %s", ev, exc)
