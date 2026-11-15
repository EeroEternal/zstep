# Copyright (c) 2026 zstep Contributors
# SPDX-License-Identifier: MIT

from __future__ import annotations

import contextlib
import sys
import types
from types import SimpleNamespace

import numpy as np
import pytest

from zstep_worker.spec_decode import (
    VerifiedSequence,
    seq_results,
    verify_batch,
    verify_greedy,
)


def test_greedy_verification_accepts_matches_and_stops_at_mismatch():
    assert verify_greedy([10, 11, 99, 13], [5, 10, 11, 12]) == [10, 11, 99]
    assert verify_greedy([10, 11, 12], [5, 10, 11]) == [10, 11, 12]
    assert verify_greedy([99, 11, 12], [5, 10, 11]) == [99]
    assert verify_greedy([42], [7]) == [42]
    with pytest.raises(ValueError):
        verify_greedy([], [])
    with pytest.raises(ValueError):
        verify_greedy([1, 2], [1])


def test_varlen_verification_groups_sequences_and_preserves_logprobs():
    # Seq 0 (rows 0..2): partial prefill chunk (not in sample_rows) -> skipped
    # Seq 1 (rows 2..5): 2 drafts [10, 12], first matches (10), second rejected (12 != 11)
    #   -> accepts 2 tokens [10, 11], discarding prediction 99 conditioned on rejected draft 12
    # Seq 2 (rows 5..7): 1 draft [20], matches -> accepts 2 tokens [20, 21]
    verified = verify_batch(
        predictions=np.array([10, 11, 99, 20, 21], dtype=np.int32),
        logprob_bits=np.array([-1, 2, 3, 4, 5], dtype=np.int32),
        token_ids=np.array([1, 2, 5, 10, 12, 6, 20], dtype=np.int32),
        sample_rows=np.array([2, 3, 4, 5, 6], dtype=np.int32),
        query_start_loc=np.array([0, 2, 5, 7], dtype=np.int32),
        req_slot_ids=np.array([3, 3, 7, 7, 7, 9, 9], dtype=np.int32),
    )
    assert [(item.req_slot, item.batch_index, item.accepted_tokens) for item in verified] == [
        (7, 1, [10, 11]),
        (9, 2, [20, 21]),
    ]
    assert verified[0].accepted_logprobs_bits == [0xFFFF_FFFF, 2]

    results = seq_results(verified, {7: [100, 101, 102]})
    assert len(results) == 2
    assert results[0].req_slot == 7
    assert results[0].accepted_tokens == [10, 11]
    assert results[0].next_draft_tokens == [100, 101, 102]
    assert results[1].req_slot == 9
    assert results[1].next_draft_tokens == []


def _make_fake_mtp_drafter(k: int = 3, max_pages_per_seq: int = 2):
    torch = pytest.importorskip("torch")
    pytest.importorskip("triton")
    from zstep_worker.models.qwen35_mtp import Qwen35MtpDrafter

    drafter = object.__new__(Qwen35MtpDrafter)
    drafter.num_draft_tokens = k
    drafter.device = torch.device("cpu")
    drafter.runner = SimpleNamespace(rank=0, max_pages_per_seq=max_pages_per_seq)
    drafter._pending_chunk_tail = {}
    drafter.mtp_graphs = {}
    drafter.mtp_tail_graph = None
    drafter.step_positions_cpu = torch.zeros(k, dtype=torch.int32)
    drafter.step_seq_lens_cpu = torch.zeros(k, dtype=torch.int32)
    drafter.step_positions_gpu = torch.zeros(k, dtype=torch.int32)
    drafter.step_seq_lens_gpu = torch.zeros(k, dtype=torch.int32)
    drafter.slots_cpu = torch.zeros(8, dtype=torch.int32)
    drafter.slots_gpu = torch.zeros(8, dtype=torch.int32)
    drafter.draft_out_tokens_gpu = torch.zeros((k, 1, 2), dtype=torch.int32)
    drafter.draft_out_tokens_batch_gpu = torch.zeros((8, k), dtype=torch.int32)
    drafter.draft_out_tokens_cpu = torch.zeros((8, k), dtype=torch.int32)

    recorded_ctx: list[tuple[list[int], list[int], int]] = []
    recorded_tail_pos: list[tuple[int, int]] = []

    def fake_stage_and_run(*, tokens, positions, req_slot, hidden):
        recorded_ctx.append((tokens.tolist(), positions.tolist(), int(req_slot)))
        return hidden

    def fake_forward_gpu(*, tokens_gpu, positions_gpu, slots_gpu, target_hidden, seq_lens_gpu, rows):
        recorded_tail_pos.append((int(positions_gpu[0].item()), int(seq_lens_gpu[0].item())))
        return target_hidden

    def fake_sample_into_step(hidden_1row, step_idx: int) -> None:
        drafter.draft_out_tokens_gpu[step_idx, 0, 0] = 700 + step_idx

    drafter._stage_and_run_prefill_context = fake_stage_and_run
    drafter._forward_gpu = fake_forward_gpu
    drafter._sample_into_step = fake_sample_into_step
    return drafter, recorded_ctx, recorded_tail_pos


def test_qwen35_mtp_discards_stale_pending_chunk_tail_on_slot_reuse_and_apc_restore():
    torch = pytest.importorskip("torch")
    drafter, recorded_ctx, _ = _make_fake_mtp_drafter(k=2, max_pages_per_seq=8)
    hidden = torch.zeros((8, 16), dtype=torch.float32)

    # 1) Partial chunk 1 on slot 3 (`positions 0..4`): writes `0..3` and stashes `prev_pos = 3`.
    out = drafter.draft(
        [],
        token_ids=np.array([10, 11, 12, 13], dtype=np.int32),
        positions=np.array([0, 1, 2, 3], dtype=np.int32),
        req_slot_ids=np.array([3, 3, 3, 3], dtype=np.int32),
        query_start_loc=np.array([0, 4], dtype=np.int32),
        target_hidden=hidden,
    )
    assert out == {}
    assert 3 in drafter._pending_chunk_tail
    assert recorded_ctx == [([11, 12, 13], [0, 1, 2], 3)]
    recorded_ctx.clear()

    # 2) Request cancels and slot 3 is reused by a fresh cold request (`positions 0..2`):
    #    stale `(prev_pos=3)` MUST be discarded without writing MTP KV at pos 3!
    verified_fresh = [
        VerifiedSequence(
            req_slot=3,
            batch_index=0,
            rows=(1,),
            accepted_tokens=[99],
            accepted_logprobs_bits=[0],
        )
    ]
    drafter.draft(
        verified_fresh,
        token_ids=np.array([20, 21], dtype=np.int32),
        positions=np.array([0, 1], dtype=np.int32),
        req_slot_ids=np.array([3, 3], dtype=np.int32),
        query_start_loc=np.array([0, 2], dtype=np.int32),
        target_hidden=hidden,
    )
    assert recorded_ctx == [([21, 99], [0, 1], 3)]
    recorded_ctx.clear()

    # 3) Stash `(prev_pos=63)` on slot 3, cancel, then reuse slot 3 for a 65-token prompt
    #    hitting a 64-token APC checkpoint (`start_pos == 64 == prev_pos + 1`, `apc_restored={0}`):
    #    stale tail MUST STILL be discarded so shared APC pages are never overwritten!
    drafter._pending_chunk_tail[3] = (63, hidden[0:1].clone())
    verified_apc = [
        VerifiedSequence(
            req_slot=3,
            batch_index=0,
            rows=(0,),
            accepted_tokens=[88],
            accepted_logprobs_bits=[0],
        )
    ]
    drafter.draft(
        verified_apc,
        token_ids=np.array([55], dtype=np.int32),
        positions=np.array([64], dtype=np.int32),
        req_slot_ids=np.array([3], dtype=np.int32),
        query_start_loc=np.array([0, 1], dtype=np.int32),
        target_hidden=hidden,
        apc_restored_seq_indices=frozenset({0}),
    )
    assert recorded_ctx == [([88], [64], 3)]
    recorded_ctx.clear()

    # 4) True contiguous Chunk 2 (`prev_pos=63 -> start_pos=64`, NOT restored from APC):
    #    consumes `(63, h_63)` with `token_ids[0]=55` before writing `pos=64`.
    drafter._pending_chunk_tail[3] = (63, hidden[0:1].clone())
    drafter.draft(
        verified_apc,
        token_ids=np.array([55], dtype=np.int32),
        positions=np.array([64], dtype=np.int32),
        req_slot_ids=np.array([3], dtype=np.int32),
        query_start_loc=np.array([0, 1], dtype=np.int32),
        target_hidden=hidden,
        apc_restored_seq_indices=frozenset(),
    )
    assert recorded_ctx == [([55], [63], 3), ([88], [64], 3)]


def test_qwen35_mtp_truncates_and_skips_drafts_at_max_page_boundary():
    torch = pytest.importorskip("torch")
    # `max_pages_per_seq = 2` -> `max_ctx = 32` (valid positions `0..31`).
    drafter, recorded_ctx, recorded_tail_pos = _make_fake_mtp_drafter(
        k=3, max_pages_per_seq=2
    )
    hidden = torch.zeros((4, 16), dtype=torch.float32)

    # At `last_pos = 29`, target can verify at most 2 more tokens (`pos 30, 31`),
    # so `avail_k == 2` and tail runs ONLY `pos 30` (never staging `pos >= 32`).
    res_29 = drafter.draft(
        [
            VerifiedSequence(
                req_slot=1,
                batch_index=0,
                rows=(0,),
                accepted_tokens=[41],
                accepted_logprobs_bits=[0],
            )
        ],
        token_ids=np.array([40], dtype=np.int32),
        positions=np.array([29], dtype=np.int32),
        req_slot_ids=np.array([1], dtype=np.int32),
        query_start_loc=np.array([0, 1], dtype=np.int32),
        target_hidden=hidden,
    )
    assert res_29 == {1: [700, 701]}
    assert recorded_tail_pos == [(30, 31)]
    assert int(drafter.step_positions_gpu[:2].max().item()) < 32
    recorded_ctx.clear()
    recorded_tail_pos.clear()

    # At `last_pos = 31` (`max_ctx - 1`), zero draft tokens fit -> skipped cleanly.
    res_31 = drafter.draft(
        [
            VerifiedSequence(
                req_slot=1,
                batch_index=0,
                rows=(0,),
                accepted_tokens=[42],
                accepted_logprobs_bits=[0],
            )
        ],
        token_ids=np.array([41], dtype=np.int32),
        positions=np.array([31], dtype=np.int32),
        req_slot_ids=np.array([1], dtype=np.int32),
        query_start_loc=np.array([0, 1], dtype=np.int32),
        target_hidden=hidden,
    )
    assert res_31 == {}
    assert recorded_ctx == []
    assert recorded_tail_pos == []


def test_deepseek_v41_dspark_guards_max_context_page_table_bounds(monkeypatch):
    torch = pytest.importorskip("torch")
    from zstep_worker.models.deepseek_v41_dspark import DeepSeekV41DSparkDrafter

    if "vllm.config" not in sys.modules:
        vllm_mod = types.ModuleType("vllm")
        cfg_mod = types.ModuleType("vllm.config")
        cfg_mod.CUDAGraphMode = SimpleNamespace(NONE="NONE")
        fwd_mod = types.ModuleType("vllm.forward_context")

        @contextlib.contextmanager
        def _noop_ctx(*args, **kwargs):
            yield

        fwd_mod.set_forward_context = _noop_ctx
        monkeypatch.setitem(sys.modules, "vllm", vllm_mod)
        monkeypatch.setitem(sys.modules, "vllm.config", cfg_mod)
        monkeypatch.setitem(sys.modules, "vllm.forward_context", fwd_mod)

    k = 5
    bt_width = 2  # `max_ctx = 256` (valid positions `0..255`, `page_table` cols `0..1`)
    ms = 4
    rows_cap = ms * k
    runner = SimpleNamespace(
        rank=0,
        device=torch.device("cpu"),
        max_model_len=256,
        num_kv_pages=8,
        dummy_comp_block=99,
        decode_graph_max_seq_len=256,
        vllm_config=SimpleNamespace(),
        mla_bt_cpu=torch.zeros((ms, bt_width), dtype=torch.int32),
        mla_bt_gpu=torch.zeros((ms, bt_width), dtype=torch.int32),
        swa_bt_cpu=torch.zeros((ms, bt_width * 2), dtype=torch.int32),
        swa_bt_gpu=torch.zeros((ms, bt_width * 2), dtype=torch.int32),
        comp_bt_cpu=torch.zeros((ms, 1), dtype=torch.int32),
        comp_bt_gpu=torch.zeros((ms, 1), dtype=torch.int32),
        seq_lens_cpu=torch.zeros(ms, dtype=torch.int32),
        seq_lens_gpu=torch.zeros(ms, dtype=torch.int32),
        decode_is_prefilling_cpu=torch.zeros(ms, dtype=torch.bool),
        _build_step_attn_metadata=lambda **kwargs: kwargs,
    )
    stored_slots: list[list[int]] = []
    model = SimpleNamespace(
        combine_hidden_states=lambda x: x,
        precompute_and_store_context_kv=lambda x, pos, slots: stored_slots.append(
            slots[0].tolist()
        ),
    )
    drafter = object.__new__(DeepSeekV41DSparkDrafter)
    drafter.runner = runner
    drafter.model = model
    drafter.noise_token_id = 0
    drafter.num_draft_tokens = k
    drafter.layer_names = {"draft.0"}
    drafter.num_draft_layers = 1
    drafter.draft_buckets = []
    drafter.draft_graphs = {}
    drafter.ctx_positions_cpu = torch.zeros(16, dtype=torch.int64)
    drafter.ctx_slot_mapping_cpu = torch.zeros(16, dtype=torch.int64)
    drafter.ctx_positions_gpu = torch.zeros(16, dtype=torch.int64)
    drafter.ctx_slot_mapping_gpu = torch.zeros(16, dtype=torch.int64)
    drafter.draft_qsl_cpu = torch.arange(ms + 1, dtype=torch.int32) * k
    drafter.draft_qsl_gpu = torch.arange(ms + 1, dtype=torch.int32) * k
    drafter.draft_tokens_cpu = torch.zeros(rows_cap, dtype=torch.int32)
    drafter.draft_tokens_gpu = torch.zeros(rows_cap, dtype=torch.int32)
    drafter.draft_positions_cpu = torch.zeros(rows_cap, dtype=torch.int64)
    drafter.draft_positions_gpu = torch.zeros(rows_cap, dtype=torch.int64)
    drafter.draft_slot_mapping_cpu = torch.zeros(rows_cap, dtype=torch.int64)
    drafter.draft_slot_mapping_gpu = torch.zeros(rows_cap, dtype=torch.int64)
    drafter.draft_out_tokens_gpu = torch.zeros((k, ms, 2), dtype=torch.int32)
    drafter.draft_out_tokens_cpu = torch.zeros((k, ms), dtype=torch.int32)

    def fake_run_draft(eff_b: int) -> None:
        for step_i in range(k):
            drafter.draft_out_tokens_gpu[step_i, :eff_b, 0] = 800 + step_i

    drafter._run_draft_forward_and_markov = fake_run_draft

    # Seq 0 at `pos = 253` (`pos_next = 254`, 2 tokens `< 256` remain; `254 + 5 = 259` crosses `bt_width=2`).
    # Seq 1 at `pos = 255` (`pos_next = 256 >= max_ctx`, zero tokens remain -> skipped).
    page_table = np.array([[2, 5], [3, 6]], dtype=np.int32)
    verified = [
        VerifiedSequence(
            req_slot=0,
            batch_index=0,
            rows=(0,),
            accepted_tokens=[50],
            accepted_logprobs_bits=[0],
        ),
        VerifiedSequence(
            req_slot=1,
            batch_index=1,
            rows=(1,),
            accepted_tokens=[60],
            accepted_logprobs_bits=[0],
        ),
    ]
    out = drafter.draft(
        verified,
        aux_hidden_states=[torch.zeros((2, 8), dtype=torch.float32)],
        positions=np.array([253, 255], dtype=np.int32),
        req_slot_ids=np.array([0, 1], dtype=np.int32),
        query_start_loc=np.array([0, 1, 2], dtype=np.int32),
        page_table=page_table,
    )
    assert out == {0: [800, 801]}
    assert int(runner.seq_lens_gpu[0].item()) == 256
    # Positions 254, 255 map to physical block `5 + 1 = 6`; overflow positions 256..258 map to `-1`.
    assert drafter.draft_slot_mapping_gpu[:k].tolist() == [
        6 * 128 + 126,
        6 * 128 + 127,
        -1,
        -1,
        -1,
    ]
    assert int(drafter.draft_positions_gpu[:k].max().item()) == 255

