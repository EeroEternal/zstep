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

"""Verify Python StepPlanLayout invariants and Rust zs-wire parity across VRAM profiles."""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
import struct
import tempfile
import threading

import pytest

from zstep_worker.shm_bridge import (
    MAX_DECODE_GRAPH_SEQS,
    SEQ_OUTPUT_LOGPROBS_OFFSET,
    SEQ_STEP_OUTPUT_BYTES,
    SHM_SUPERBLOCK_REGION_BYTES,
    SHM_SUPERBLOCK_STRUCT_BYTES,
    STEP_FLAG_DECODE_ONLY,
    STEP_FLAG_DEVICE_TOKENS,
    STEP_FLAG_NEXT_PREPUBLISHED,
    STEP_HEADER_BYTES,
    STEP_HEADER_FLAGS_OFFSET,
    STEP_RESULT_HEADER_BYTES,
    SUPERBLOCK_LAYOUT_HASH_OFFSET,
    SUPERBLOCK_MAX_DECODE_GRAPH_SEQS_OFFSET,
    SUPERBLOCK_PLAN_OFFSETS_OFFSET,
    SUPERBLOCK_PLAN_STRIDE_OFFSET,
    SUPERBLOCK_RESULT_OFFSETS_OFFSET,
    SUPERBLOCK_RESULT_STRIDE_OFFSET,
    SUPERBLOCK_TOTAL_BYTES_OFFSET,
    WORKER_ERROR_OFFSET,
    WORKER_FREE_VRAM_OFFSET,
    WORKER_HEARTBEAT_OFFSET,
    WORKER_NUMA_OFFSET,
    WORKER_PID_OFFSET,
    WORKER_SM_ARCH_OFFSET,
    WORKER_STATE_BYTES,
    WORKER_STATES_OFFSET,
    WORKER_STATES_REGION_BYTES,
    WORKER_STATUS_OFFSET,
    WORKER_TOTAL_VRAM_OFFSET,
    WORKER_TRITON_RECOMP_OFFSET,
    WORKER_WATCHED_KERNELS_OFFSET,
    ZSTEP_PROTOCOL_VERSION,
    ZSTEP_SHM_MAGIC,
    ShmBridge,
    ShmGeometry,
    StepPlanLayout,
    compute_layout_hash,
    compute_result_slot_stride,
    select_decode_bucket,
)


def _assert_layout_invariants(geom: ShmGeometry, layout: StepPlanLayout) -> None:
    regions = [
        (layout.header_offset, 64),
        (layout.token_ids_offset, geom.max_step_tokens * 4),
        (layout.positions_offset, geom.max_step_tokens * 4),
        (layout.req_slot_ids_offset, geom.max_step_tokens * 4),
        (layout.replay_mask_offset, geom.max_step_tokens),
        (layout.engram_prev_window_offset, geom.max_step_tokens * 12),
        (layout.narrow_keep_indices_offset, geom.max_decoder_rows * 4),
        (layout.sample_row_indices_offset, geom.max_decoder_rows * 4),
        (layout.seq_lens_offset, geom.max_seqs * 4),
        (layout.query_start_loc_offset, (geom.max_seqs + 1) * 4),
        (
            layout.global_kv_page_table_offset,
            geom.max_seqs * geom.max_pages_per_seq * 4,
        ),
        (layout.swa_page_table_offset, geom.max_seqs * 16),
        (layout.compressor_slot_ids_offset, geom.max_seqs * 4),
        (layout.temperatures_offset, geom.max_seqs * 4),
    ]
    for (prev_off, prev_len), (next_off, _next_len) in zip(
        regions, regions[1:], strict=False
    ):
        assert prev_off + prev_len <= next_off
        assert next_off % 256 == 0
    last_off, last_len = regions[-1]
    assert last_off + last_len <= layout.total_slot_bytes
    assert layout.total_slot_bytes % 4096 == 0


def test_step_plan_layout_invariants_and_vram_derivation() -> None:
    geom_32g = ShmGeometry.from_vram_budget(
        world_size=8,
        per_gpu_vram_mb=32_607,
        reserved_weights_and_workspace_mb=12_288,
        kv_bytes_per_token_per_gpu=10_240,
        max_seqs=128,
        max_step_tokens=8_192,
        max_decoder_rows=8_192,
        page_block_tokens=64,
        spec_verify_width=4,
    )
    assert geom_32g is not None
    assert geom_32g.max_pages_per_seq > 0
    _assert_layout_invariants(geom_32g, StepPlanLayout.compute(geom_32g))

    odd_geom = ShmGeometry(
        world_size=1,
        max_seqs=3,
        max_step_tokens=19,
        max_decoder_rows=7,
        max_pages_per_seq=11,
        page_block_tokens=64,
        spec_verify_width=1,
    )
    _assert_layout_invariants(odd_geom, StepPlanLayout.compute(odd_geom))

    assert (
        ShmGeometry.from_vram_budget(
            world_size=8,
            per_gpu_vram_mb=16_384,
            reserved_weights_and_workspace_mb=20_000,
            kv_bytes_per_token_per_gpu=10_240,
            max_seqs=32,
            max_step_tokens=1024,
            max_decoder_rows=1024,
            page_block_tokens=64,
            spec_verify_width=1,
        )
        is None
    )


def test_cross_language_golden_layout_and_hash_parity() -> None:
    golden_path = Path(__file__).resolve().parent / "fixtures" / "shm_layout_golden.json"
    golden = json.loads(golden_path.read_text(encoding="utf-8"))
    assert golden["protocol_version"] == ZSTEP_PROTOCOL_VERSION

    pod = golden["pod_offsets"]
    sb = pod["ShmSuperblock"]
    assert sb["total_shm_bytes"] == SUPERBLOCK_TOTAL_BYTES_OFFSET
    assert sb["plan_slot_offsets"] == SUPERBLOCK_PLAN_OFFSETS_OFFSET
    assert sb["plan_slot_stride"] == SUPERBLOCK_PLAN_STRIDE_OFFSET
    assert sb["result_slot_offsets"] == SUPERBLOCK_RESULT_OFFSETS_OFFSET
    assert sb["result_slot_stride"] == SUPERBLOCK_RESULT_STRIDE_OFFSET
    assert sb["layout_hash"] == SUPERBLOCK_LAYOUT_HASH_OFFSET
    assert sb["max_decode_graph_seqs"] == SUPERBLOCK_MAX_DECODE_GRAPH_SEQS_OFFSET == 112
    assert sb["_pad_u32"] == 116
    assert sb["_sizeof"] == SHM_SUPERBLOCK_STRUCT_BYTES

    ws = pod["WorkerSlotState"]
    assert ws["heartbeat_ns"] == WORKER_HEARTBEAT_OFFSET
    assert ws["worker_pid"] == WORKER_PID_OFFSET
    assert ws["device_sm_arch"] == WORKER_SM_ARCH_OFFSET
    assert ws["status_code"] == WORKER_STATUS_OFFSET
    assert ws["error_code"] == WORKER_ERROR_OFFSET
    assert ws["free_vram_mb"] == WORKER_FREE_VRAM_OFFSET
    assert ws["total_vram_mb"] == WORKER_TOTAL_VRAM_OFFSET
    assert ws["numa_node"] == WORKER_NUMA_OFFSET
    assert ws["triton_recompilations"] == WORKER_TRITON_RECOMP_OFFSET
    assert ws["watched_kernels"] == WORKER_WATCHED_KERNELS_OFFSET
    assert ws["_sizeof"] == WORKER_STATE_BYTES

    sh = pod["StepHeader"]
    assert sh["flags"] == STEP_HEADER_FLAGS_OFFSET
    assert sh["_sizeof"] == STEP_HEADER_BYTES

    srh = pod["StepResultHeader"]
    assert srh["_sizeof"] == STEP_RESULT_HEADER_BYTES

    sso = pod["SeqStepOutput"]
    assert sso["accepted_logprobs_bits"] == SEQ_OUTPUT_LOGPROBS_OFFSET
    assert sso["_sizeof"] == SEQ_STEP_OUTPUT_BYTES

    cases = golden["cases"]
    assert len(cases) >= 3
    for case in cases:
        g = case["geometry"]
        geom = ShmGeometry(
            world_size=g["world_size"],
            max_seqs=g["max_seqs"],
            max_step_tokens=g["max_step_tokens"],
            max_decoder_rows=g["max_decoder_rows"],
            max_pages_per_seq=g["max_pages_per_seq"],
            page_block_tokens=g["page_block_tokens"],
            spec_verify_width=g["spec_verify_width"],
        )
        layout = StepPlanLayout.compute(geom)
        _assert_layout_invariants(geom, layout)
        exp = case["expected"]
        for field_name in (
            "header_offset",
            "token_ids_offset",
            "positions_offset",
            "req_slot_ids_offset",
            "replay_mask_offset",
            "engram_prev_window_offset",
            "narrow_keep_indices_offset",
            "sample_row_indices_offset",
            "seq_lens_offset",
            "query_start_loc_offset",
            "global_kv_page_table_offset",
            "swa_page_table_offset",
            "compressor_slot_ids_offset",
            "temperatures_offset",
            "total_slot_bytes",
        ):
            actual = getattr(layout, field_name)
            assert actual == exp[field_name], (
                f"{case['name']}.{field_name}: python={actual} != golden={exp[field_name]}"
            )
        stride = compute_result_slot_stride(geom.max_seqs)
        assert stride == exp["result_slot_stride"]
        digest_hex = f"0x{compute_layout_hash(layout, stride):016x}"
        assert digest_hex == exp["layout_hash_hex"], (
            f"{case['name']} hash mismatch: {digest_hex} != {exp['layout_hash_hex']}"
        )


def test_superblock_binary_roundtrip_and_pad_isolation() -> None:
    """Regression test: offset 112 (`max_decode_graph_seqs`) must never read `_pad_u32` (116)."""
    golden_path = Path(__file__).resolve().parent / "fixtures" / "shm_layout_golden.json"
    sb_offsets = json.loads(golden_path.read_text(encoding="utf-8"))["pod_offsets"][
        "ShmSuperblock"
    ]
    geom = ShmGeometry(
        world_size=8,
        max_seqs=128,
        max_step_tokens=4096,
        max_decoder_rows=4096,
        max_pages_per_seq=256,
        page_block_tokens=128,
        spec_verify_width=6,
    )
    layout = StepPlanLayout.compute(geom)
    result_stride = compute_result_slot_stride(geom.max_seqs)
    layout_hash = compute_layout_hash(layout, result_stride)
    plans_base = SHM_SUPERBLOCK_REGION_BYTES + WORKER_STATES_REGION_BYTES
    results_base = plans_base + 2 * layout.total_slot_bytes
    total_bytes = results_base + 2 * result_stride

    buf = bytearray(SHM_SUPERBLOCK_REGION_BYTES)
    struct.pack_into("<Q", buf, sb_offsets["magic"], ZSTEP_SHM_MAGIC)
    struct.pack_into("<I", buf, sb_offsets["protocol_version"], ZSTEP_PROTOCOL_VERSION)
    struct.pack_into("<I", buf, sb_offsets["world_size"], geom.world_size)
    struct.pack_into("<I", buf, sb_offsets["max_seqs"], geom.max_seqs)
    struct.pack_into("<I", buf, sb_offsets["max_step_tokens"], geom.max_step_tokens)
    struct.pack_into("<I", buf, sb_offsets["max_decoder_rows"], geom.max_decoder_rows)
    struct.pack_into("<I", buf, sb_offsets["max_pages_per_seq"], geom.max_pages_per_seq)
    struct.pack_into("<I", buf, sb_offsets["page_block_tokens"], geom.page_block_tokens)
    struct.pack_into("<I", buf, sb_offsets["spec_verify_width"], geom.spec_verify_width)
    struct.pack_into("<I", buf, sb_offsets["master_pid"], 424242)
    # Poison implicit C alignment hole [44..48] and `_pad_u32` [116..120] + `_reserved` [120..128]
    struct.pack_into("<I", buf, 44, 777777)
    struct.pack_into("<Q", buf, sb_offsets["total_shm_bytes"], total_bytes)
    struct.pack_into(
        "<2Q",
        buf,
        sb_offsets["plan_slot_offsets"],
        plans_base,
        plans_base + layout.total_slot_bytes,
    )
    struct.pack_into("<Q", buf, sb_offsets["plan_slot_stride"], layout.total_slot_bytes)
    struct.pack_into(
        "<2Q",
        buf,
        sb_offsets["result_slot_offsets"],
        results_base,
        results_base + result_stride,
    )
    struct.pack_into("<Q", buf, sb_offsets["result_slot_stride"], result_stride)
    struct.pack_into("<Q", buf, sb_offsets["layout_hash"], layout_hash)
    struct.pack_into("<I", buf, sb_offsets["max_decode_graph_seqs"], 128)
    struct.pack_into("<I", buf, sb_offsets["_pad_u32"], 999)
    struct.pack_into("<Q", buf, sb_offsets["_reserved"], 0xDEAD_BEEF_CAFE_BABE)

    parsed = ShmBridge._parse_superblock(bytes(buf))
    assert parsed.magic == ZSTEP_SHM_MAGIC
    assert parsed.protocol_version == ZSTEP_PROTOCOL_VERSION
    assert parsed.geometry == geom
    assert parsed.master_pid == 424242
    assert parsed.total_shm_bytes == total_bytes
    assert parsed.plan_slot_offsets == (plans_base, plans_base + layout.total_slot_bytes)
    assert parsed.plan_slot_stride == layout.total_slot_bytes
    assert parsed.result_slot_offsets == (results_base, results_base + result_stride)
    assert parsed.result_slot_stride == result_stride
    assert parsed.layout_hash == layout_hash
    assert parsed.max_decode_graph_seqs == 128

    # Corrupting `max_decode_graph_seqs` at offset 112 (`0` or `999`) must fail loudly.
    for bad_max in (0, 999):
        bad_buf = bytearray(buf)
        struct.pack_into("<I", bad_buf, sb_offsets["max_decode_graph_seqs"], bad_max)
        with pytest.raises(RuntimeError, match="max_decode_graph_seqs"):
            ShmBridge._parse_superblock(bytes(bad_buf))


def test_select_decode_bucket_never_clamps_overflow() -> None:
    assert MAX_DECODE_GRAPH_SEQS == 128
    assert STEP_FLAG_DECODE_ONLY == (1 << 0)
    assert STEP_FLAG_DEVICE_TOKENS == (1 << 1)
    assert STEP_FLAG_NEXT_PREPUBLISHED == (1 << 2)
    assert select_decode_bucket(1, 128) == 1
    assert select_decode_bucket(3, 128) == 4
    assert select_decode_bucket(65, 128) == 128
    with pytest.raises(ValueError, match="outside supported CUDA Graph range"):
        select_decode_bucket(129, 128)
    with pytest.raises(ValueError, match="outside supported CUDA Graph range"):
        select_decode_bucket(33, 32)
    with pytest.raises(ValueError, match="outside supported CUDA Graph range"):
        select_decode_bucket(0, 128)


def _make_runtime_loop_fixture(runner: object) -> tuple[object, ShmBridge, int, int, Path]:
    from zstep_worker.numa_pin import DeviceNumaTopology
    from zstep_worker.runtime_loop import RuntimeLoop

    golden_path = Path(__file__).resolve().parent / "fixtures" / "shm_layout_golden.json"
    sb_offsets = json.loads(golden_path.read_text(encoding="utf-8"))["pod_offsets"][
        "ShmSuperblock"
    ]
    geom = ShmGeometry(
        world_size=1,
        max_seqs=8,
        max_step_tokens=64,
        max_decoder_rows=64,
        max_pages_per_seq=16,
        page_block_tokens=64,
        spec_verify_width=1,
    )
    layout = StepPlanLayout.compute(geom)
    result_stride = compute_result_slot_stride(geom.max_seqs)
    layout_hash = compute_layout_hash(layout, result_stride)
    plans_base = SHM_SUPERBLOCK_REGION_BYTES + WORKER_STATES_REGION_BYTES
    results_base = plans_base + 2 * layout.total_slot_bytes
    total_bytes = results_base + 2 * result_stride

    fd, raw_path = tempfile.mkstemp(prefix="zstep_rtloop_")
    shm_path = Path(raw_path)
    os.ftruncate(fd, total_bytes)
    buf = bytearray(SHM_SUPERBLOCK_REGION_BYTES)
    struct.pack_into("<Q", buf, sb_offsets["magic"], ZSTEP_SHM_MAGIC)
    struct.pack_into("<I", buf, sb_offsets["protocol_version"], ZSTEP_PROTOCOL_VERSION)
    struct.pack_into("<I", buf, sb_offsets["world_size"], geom.world_size)
    struct.pack_into("<I", buf, sb_offsets["max_seqs"], geom.max_seqs)
    struct.pack_into("<I", buf, sb_offsets["max_step_tokens"], geom.max_step_tokens)
    struct.pack_into("<I", buf, sb_offsets["max_decoder_rows"], geom.max_decoder_rows)
    struct.pack_into("<I", buf, sb_offsets["max_pages_per_seq"], geom.max_pages_per_seq)
    struct.pack_into("<I", buf, sb_offsets["page_block_tokens"], geom.page_block_tokens)
    struct.pack_into("<I", buf, sb_offsets["spec_verify_width"], geom.spec_verify_width)
    struct.pack_into("<I", buf, sb_offsets["master_pid"], os.getpid())
    struct.pack_into("<Q", buf, sb_offsets["total_shm_bytes"], total_bytes)
    struct.pack_into(
        "<2Q",
        buf,
        sb_offsets["plan_slot_offsets"],
        plans_base,
        plans_base + layout.total_slot_bytes,
    )
    struct.pack_into("<Q", buf, sb_offsets["plan_slot_stride"], layout.total_slot_bytes)
    struct.pack_into(
        "<2Q",
        buf,
        sb_offsets["result_slot_offsets"],
        results_base,
        results_base + result_stride,
    )
    struct.pack_into("<Q", buf, sb_offsets["result_slot_stride"], result_stride)
    struct.pack_into("<Q", buf, sb_offsets["layout_hash"], layout_hash)
    struct.pack_into("<I", buf, sb_offsets["max_decode_graph_seqs"], 8)
    os.pwrite(fd, bytes(buf), 0)

    master_fd = os.dup(fd)
    worker_bridge = ShmBridge(name="zstep_rtloop_test", shm_fd=fd)
    master_bridge = ShmBridge(name="zstep_rtloop_test", shm_fd=master_fd)
    plan_fd = os.eventfd(0)
    done_fd = os.eventfd(0)

    loop = RuntimeLoop.__new__(RuntimeLoop)
    loop.args = argparse.Namespace(
        rank=0,
        plan_fd=plan_fd,
        done_fd=done_fd,
        model_dir="",
    )
    loop.numa_topology = DeviceNumaTopology(
        device_index=0,
        pci_bus_id="0000:00:00.0",
        numa_node=0,
        pinned_cpus=(0,),
    )
    loop.bridge = worker_bridge
    loop.runner = runner
    loop.model_type = ""
    loop._watchdog_stop = threading.Event()
    loop._watchdog_thread = threading.Thread(target=lambda: None)
    loop._free_vram_mb = 1024
    loop._total_vram_mb = 2048
    loop.current_step = 0
    loop.last_step = 0
    loop._pending_plan_signals = 0
    return loop, master_bridge, plan_fd, done_fd, shm_path


def _publish_test_step(
    bridge: ShmBridge,
    *,
    step_id: int,
    opcode: int,
    flags: int = 0,
    num_seqs: int = 1,
) -> None:
    plan = bridge.plan(step_id)
    fields = struct.pack(
        "<12I",
        opcode,
        num_seqs,
        num_seqs,
        num_seqs,
        num_seqs,
        num_seqs,
        num_seqs,
        0,
        flags,
        0,
        0,
        0,
    )
    bridge._mapping[plan.base + 8 : plan.base + 56] = fields
    bridge.store_u64(plan.base, step_id)


@pytest.mark.skipif(not hasattr(os, "eventfd"), reason="Linux eventfd required")
def test_runtime_loop_pipelined_prepublished_burst_and_tail_step() -> None:
    from zstep_worker.runtime_loop import OPCODE_NORMAL_STEP, OPCODE_SHUTDOWN
    from zstep_worker.shm_bridge import SeqResult

    events: list[str] = []

    class _FakePipelinedRunner:
        def launch(self, _plan: object, header: object) -> int:
            events.append(f"launch:{header.step_id}:{header.flags}")
            return header.step_id

        def finish(self, launched_step_id: int) -> tuple[list[SeqResult], int]:
            events.append(f"finish:{launched_step_id}")
            return (
                [
                    SeqResult(
                        req_slot=0,
                        accepted_token=100 + launched_step_id,
                        accepted_logprob_bits=0,
                    )
                ],
                10,
            )

        def close(self, *, abort: bool = False) -> None:
            pass

    loop, master, plan_fd, done_fd, shm_path = _make_runtime_loop_fixture(_FakePipelinedRunner())
    worker_exc: list[BaseException] = []
    worker_rc: list[int] = []

    def _run_worker() -> None:
        try:
            worker_rc.append(loop.run())
        except BaseException as exc:
            worker_exc.append(exc)

    thread = threading.Thread(target=_run_worker, daemon=True)
    thread.start()
    try:
        # Drain initial ready signal (`step 0`).
        assert os.eventfd_read(done_fd) == 1
        # Publish step 1 (seed pre-published) + step 2 (device tokens + pre-published) together.
        _publish_test_step(
            master,
            step_id=1,
            opcode=OPCODE_NORMAL_STEP,
            flags=STEP_FLAG_DECODE_ONLY | STEP_FLAG_NEXT_PREPUBLISHED,
        )
        _publish_test_step(
            master,
            step_id=2,
            opcode=OPCODE_NORMAL_STEP,
            flags=STEP_FLAG_DECODE_ONLY | STEP_FLAG_DEVICE_TOKENS | STEP_FLAG_NEXT_PREPUBLISHED,
        )
        os.eventfd_write(plan_fd, 2)
        # Step 1 finishes only after step 2 is launched behind it.
        assert os.eventfd_read(done_fd) == 1
        assert events[:3] == [
            f"launch:1:{STEP_FLAG_DECODE_ONLY | STEP_FLAG_NEXT_PREPUBLISHED}",
            f"launch:2:{STEP_FLAG_DECODE_ONLY | STEP_FLAG_DEVICE_TOKENS | STEP_FLAG_NEXT_PREPUBLISHED}",
            "finish:1",
        ]
        # Publish tail step 3 (`promise == False`): worker launches 3, finishes 2, and finishes 3.
        _publish_test_step(
            master,
            step_id=3,
            opcode=OPCODE_NORMAL_STEP,
            flags=STEP_FLAG_DECODE_ONLY | STEP_FLAG_DEVICE_TOKENS,
        )
        os.eventfd_write(plan_fd, 1)
        drained = 0
        while drained < 2:
            drained += os.eventfd_read(done_fd)
        assert drained == 2
        assert master.load_u64(WORKER_STATES_OFFSET) == 3

        _publish_test_step(master, step_id=4, opcode=OPCODE_SHUTDOWN, flags=0, num_seqs=0)
        os.eventfd_write(plan_fd, 1)
        assert os.eventfd_read(done_fd) == 1
        thread.join(timeout=5.0)
        assert not worker_exc
        assert worker_rc == [0]
        assert events == [
            f"launch:1:{STEP_FLAG_DECODE_ONLY | STEP_FLAG_NEXT_PREPUBLISHED}",
            f"launch:2:{STEP_FLAG_DECODE_ONLY | STEP_FLAG_DEVICE_TOKENS | STEP_FLAG_NEXT_PREPUBLISHED}",
            "finish:1",
            f"launch:3:{STEP_FLAG_DECODE_ONLY | STEP_FLAG_DEVICE_TOKENS}",
            "finish:2",
            "finish:3",
        ]
    finally:
        master.close()
        loop.close(abort=True)
        shm_path.unlink(missing_ok=True)


@pytest.mark.skipif(not hasattr(os, "eventfd"), reason="Linux eventfd required")
@pytest.mark.parametrize(
    ("pipelined", "step1_opcode", "step1_flags", "step2_opcode", "expected_match"),
    [
        (False, 1, STEP_FLAG_DECODE_ONLY | STEP_FLAG_DEVICE_TOKENS, None, "non-pipelined runner"),
        (True, 1, STEP_FLAG_DEVICE_TOKENS, None, "non-decode step"),
        (
            True,
            1,
            STEP_FLAG_DECODE_ONLY | STEP_FLAG_NEXT_PREPUBLISHED,
            99,
            "was still pending",
        ),
    ],
)
def test_runtime_loop_rejects_invalid_overlap_flags(
    pipelined: bool,
    step1_opcode: int,
    step1_flags: int,
    step2_opcode: int | None,
    expected_match: str,
) -> None:
    class _NonPipelinedRunner:
        def execute(self, _plan: object, _header: object) -> tuple[list, int]:
            return [], 1

        def close(self, *, abort: bool = False) -> None:
            pass

    class _PipelinedRunner:
        def launch(self, _plan: object, header: object) -> int:
            return header.step_id

        def finish(self, _launched: int) -> tuple[list, int]:
            return [], 1

        def close(self, *, abort: bool = False) -> None:
            pass

    runner = _PipelinedRunner() if pipelined else _NonPipelinedRunner()
    loop, master, plan_fd, _done_fd, shm_path = _make_runtime_loop_fixture(runner)
    try:
        _publish_test_step(master, step_id=1, opcode=step1_opcode, flags=step1_flags)
        os.eventfd_write(plan_fd, 1)
        if step2_opcode is not None:
            _publish_test_step(master, step_id=2, opcode=step2_opcode, flags=0, num_seqs=0)
            os.eventfd_write(plan_fd, 1)
        with pytest.raises(RuntimeError, match=expected_match):
            loop.run()
    finally:
        master.close()
        loop.close(abort=True)
        shm_path.unlink(missing_ok=True)


@pytest.mark.skipif(not hasattr(os, "eventfd"), reason="Linux eventfd required")
def test_runtime_loop_missing_torch_guard_allows_fake_runner_and_rejects_real_worker(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    import sys

    class _NoopRunner:
        def execute(self, _plan: object, _header: object) -> tuple[list, int]:
            return [], 1

        def close(self, *, abort: bool = False) -> None:
            pass

    # Setting a module entry to `None` in `sys.modules` forces Python's import
    # machinery to raise `ModuleNotFoundError` deterministically even on hosts
    # where `torch` is installed.
    monkeypatch.setitem(sys.modules, "zstep_worker.kernels_sm120.moe_router", None)

    # 1. Fake runner (`model_dir == ""`) must degrade cleanly without `torch`.
    loop_fake, master_fake, plan_fd_fake, done_fd_fake, shm_path_fake = (
        _make_runtime_loop_fixture(_NoopRunner())
    )
    try:
        _publish_test_step(master_fake, step_id=1, opcode=99, flags=0, num_seqs=0)
        os.eventfd_write(plan_fd_fake, 1)
        loop_fake.run()
        # 1 signal from initial READY handshake + 1 signal from shutdown step 1
        assert os.eventfd_read(done_fd_fake) == 2
    finally:
        master_fake.close()
        loop_fake.close(abort=True)
        shm_path_fake.unlink(missing_ok=True)

    # 2. Real model worker (`bool(model_dir) is True`) must re-raise `ModuleNotFoundError` loudly.
    loop_real, master_real, _plan_fd_real, _done_fd_real, shm_path_real = (
        _make_runtime_loop_fixture(_NoopRunner())
    )
    loop_real.args.model_dir = "/models/nonexistent-test-model"
    try:
        with pytest.raises(ModuleNotFoundError):
            loop_real.run()
    finally:
        master_real.close()
        loop_real.close(abort=True)
        shm_path_real.unlink(missing_ok=True)


if __name__ == "__main__":
    test_step_plan_layout_invariants_and_vram_derivation()
    test_cross_language_golden_layout_and_hash_parity()
    test_superblock_binary_roundtrip_and_pad_isolation()
    test_select_decode_bucket_never_clamps_overflow()
    print("Python <-> Rust StepPlanLayout invariants, golden hash, and superblock bounds OK")
