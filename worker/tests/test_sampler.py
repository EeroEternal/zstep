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

"""Unit tests for the zero-sync distributed GPU sampler (temperature, top_p, top_k, seed)."""

from __future__ import annotations

import pytest

torch = pytest.importorskip("torch")

from zstep_worker.sampler import (  # noqa: E402
    SlotPenaltyState,
    sample_tp8_logits,
    splitmix64_uniform_f32,
)


def _splitmix64_py(x: int) -> int:
    mask = 0xFFFF_FFFF_FFFF_FFFF
    x = (x + 0x9E37_79B9_7F4A_7C15) & mask
    x = ((x ^ (x >> 30)) * 0xBF58_476D_1CE4_E5B9) & mask
    x = ((x ^ (x >> 27)) * 0x94D0_49BB_1331_11EB) & mask
    return (x ^ (x >> 31)) & mask


def test_splitmix64_uniform_matches_u64_reference_and_bounds() -> None:
    raw_u64 = [
        0,
        1,
        42,
        0x7FFF_FFFF_FFFF_FFFF,
        0x8000_0000_0000_0000,
        0xFFFF_FFFF_FFFF_FFFF,
        0xDEAD_BEEF_CAFE_BABE,
    ]
    # Reinterpret u64 as signed int64 bit-pattern (same as numpy uint64 -> view(int64))
    signed_i64 = [v - (1 << 64) if v >= (1 << 63) else v for v in raw_u64]
    seeds = torch.tensor(signed_i64, dtype=torch.int64)
    uniforms = splitmix64_uniform_f32(seeds)
    assert uniforms.dtype == torch.float32
    assert torch.all(uniforms > 0.0)
    assert torch.all(uniforms < 1.0)
    for idx, raw in enumerate(raw_u64):
        expected = (((_splitmix64_py(raw) >> 41) & 0x007F_FFFF) + 0.5) / 8388608.0
        assert uniforms[idx].item() == pytest.approx(expected, rel=1e-7, abs=1e-7)


def test_greedy_and_top_k_one_return_exact_argmax() -> None:
    logits = torch.tensor(
        [
            [0.1, 3.5, 1.2, 2.9, -1.0],
            [-2.0, -0.5, -3.0, 0.25, 0.24],
        ],
        dtype=torch.float32,
    )
    seeds = torch.tensor([12345, 67890], dtype=torch.int64)

    greedy = sample_tp8_logits(
        logits,
        temperatures=torch.tensor([0.0, 0.0], dtype=torch.float32),
        top_p=torch.tensor([1.0, 1.0], dtype=torch.float32),
        top_k=torch.tensor([0, 0], dtype=torch.int32),
        rng_seeds=seeds,
    )
    assert greedy.dtype == torch.int32
    assert greedy.tolist() == [1, 3]

    topk1 = sample_tp8_logits(
        logits,
        temperatures=torch.tensor([1.5, 2.0], dtype=torch.float32),
        top_p=torch.tensor([1.0, 1.0], dtype=torch.float32),
        top_k=torch.tensor([1, 1], dtype=torch.int32),
        rng_seeds=seeds,
    )
    assert topk1.tolist() == [1, 3]


def test_top_k_and_top_p_respect_support_and_never_empty() -> None:
    # Token 4 (top1), token 1 (top2), token 6 (top3)
    row = torch.tensor([-5.0, 2.0, -5.0, -5.0, 2.2, -5.0, 1.9, -5.0], dtype=torch.float32)
    batch = 64
    logits = row.unsqueeze(0).expand(batch, -1).contiguous()
    seeds = torch.arange(1, batch + 1, dtype=torch.int64)

    # top_k=3 must only ever emit {4, 1, 6} and must explore more than one token at temp=1.0
    out_k3 = sample_tp8_logits(
        logits,
        temperatures=torch.full((batch,), 1.0, dtype=torch.float32),
        top_p=torch.full((batch,), 1.0, dtype=torch.float32),
        top_k=torch.full((batch,), 3, dtype=torch.int32),
        rng_seeds=seeds,
    )
    observed_k3 = set(out_k3.tolist())
    assert observed_k3.issubset({4, 1, 6})
    assert len(observed_k3) >= 2

    # Tiny top_p=1e-6 must still keep top-1 without NaN or out-of-bounds
    out_tiny_p = sample_tp8_logits(
        logits,
        temperatures=torch.full((batch,), 1.0, dtype=torch.float32),
        top_p=torch.full((batch,), 1e-6, dtype=torch.float32),
        top_k=torch.zeros((batch,), dtype=torch.int32),
        rng_seeds=seeds,
    )
    assert set(out_tiny_p.tolist()) == {4}


def test_seed_reproducibility_and_mixed_greedy_stochastic_batch() -> None:
    logits = torch.tensor(
        [
            [1.0, 1.1, 1.2, 1.3, 1.4, 1.5],  # row 0: greedy (argmax = 5)
            [1.0, 1.0, 1.0, 1.0, 1.0, 1.0],  # row 1: uniform stochastic
        ],
        dtype=torch.float32,
    )
    temps = torch.tensor([0.0, 1.0], dtype=torch.float32)
    top_p = torch.tensor([1.0, 1.0], dtype=torch.float32)
    top_k = torch.tensor([0, 0], dtype=torch.int32)
    seeds_a = torch.tensor([0, 42], dtype=torch.int64)

    run1 = sample_tp8_logits(logits, temperatures=temps, top_p=top_p, top_k=top_k, rng_seeds=seeds_a)
    run2 = sample_tp8_logits(logits, temperatures=temps, top_p=top_p, top_k=top_k, rng_seeds=seeds_a)
    assert run1.tolist() == run2.tolist()
    assert run1[0].item() == 5

    # Distinct seeds across 32 trials must produce diverse tokens on uniform row 1
    draws = {
        sample_tp8_logits(
            logits,
            temperatures=temps,
            top_p=top_p,
            top_k=top_k,
            rng_seeds=torch.tensor([0, s], dtype=torch.int64),
        )[1].item()
        for s in range(1, 33)
    }
    assert len(draws) >= 4


def test_multi_rank_all_gather_layout_and_vocab_padding_trim(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Verify `[world_size, M, V_local]` -> `[M, world_size * V_local]` transpose and vocab trim."""
    # Simulate 2 TP ranks, M=2 rows, V_local=4 (total 8), with true vocab_size=6
    # Rank 0 owns tokens 0..4, Rank 1 owns tokens 4..8 (tokens 6, 7 are padding!)
    rank0_logits = torch.tensor(
        [
            [0.0, 1.0, 0.0, 0.0],  # row 0: token 1
            [0.0, 0.0, 0.0, 0.0],  # row 1: loses to rank 1 token 5
        ],
        dtype=torch.float32,
    )
    rank1_logits = torch.tensor(
        [
            [0.0, 0.0, 99.0, 99.0],  # padding tokens 6, 7 have huge logits — must be trimmed!
            [0.0, 5.0, 99.0, 99.0],  # token 5 wins for row 1; padding 6, 7 trimmed
        ],
        dtype=torch.float32,
    )

    def _fake_all_gather(output: torch.Tensor, _inp: torch.Tensor, group: object = None) -> None:
        assert output.shape == (2, 2, 4)
        output[0].copy_(rank0_logits)
        output[1].copy_(rank1_logits)

    monkeypatch.setattr("torch.distributed.all_gather_into_tensor", _fake_all_gather)

    out = sample_tp8_logits(
        rank0_logits,
        temperatures=torch.tensor([0.0, 0.0], dtype=torch.float32),
        top_p=torch.tensor([1.0, 1.0], dtype=torch.float32),
        top_k=torch.tensor([0, 0], dtype=torch.int32),
        rng_seeds=torch.tensor([1, 2], dtype=torch.int64),
        world_size=2,
        vocab_size=6,
    )
    assert out.tolist() == [1, 5]


def test_max_uniform_never_samples_masked_out_tail_and_nonfinite_falls_back(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Regression test for P1-1/P1-2/P2-1: u=1-2^-24 stays within top_k and NaN falls back."""
    max_u = 1.0 - (1.0 / 16777216.0)
    monkeypatch.setattr(
        "zstep_worker.sampler.splitmix64_uniform_f32",
        lambda seeds: torch.full((seeds.shape[0],), max_u, dtype=torch.float32),
    )
    # Large vocab where top-2 are tokens 7 and 13; token 4095 has lowest logit (-100)
    vocab = 4096
    logits = torch.full((2, vocab), -10.0, dtype=torch.float32)
    logits[0, 7] = 3.0
    logits[0, 13] = 2.9
    logits[0, vocab - 1] = -100.0
    # Row 1 has +inf logit (which makes softmax NaN) and should fall back cleanly to argmax
    logits[1, 42] = float("inf")
    logits[1, vocab - 1] = -100.0

    out = sample_tp8_logits(
        logits,
        temperatures=torch.tensor([1.0, 1.0], dtype=torch.float32),
        top_p=torch.tensor([0.99, 1.0], dtype=torch.float32),
        top_k=torch.tensor([2, 0], dtype=torch.int32),
        rng_seeds=torch.tensor([1, 2], dtype=torch.int64),
    )
    assert out[0].item() in {7, 13}
    assert out[1].item() == 42


def test_presence_and_frequency_penalties_with_slot_reset_and_ring_wrap() -> None:
    """Verify exact OpenAI presence/frequency penalty progression, slot reset, and ring wrap."""
    state = SlotPenaltyState(max_seqs=4, device=torch.device("cpu"), max_history=3)
    # Token 2 has highest base logit (3.0), token 0 has 2.2, token 1 has 1.6
    logits = torch.tensor([[2.2, 1.6, 3.0, 0.0]], dtype=torch.float32)
    zeros_temp = torch.tensor([0.0], dtype=torch.float32)
    ones_p = torch.tensor([1.0], dtype=torch.float32)
    zeros_k = torch.tensor([0], dtype=torch.int32)
    seeds = torch.tensor([1], dtype=torch.int64)
    slots = torch.tensor([1], dtype=torch.int64)

    # Step 0 (prefill completion, reset=1): no prior tokens -> picks token 2 (logit 3.0)
    t0 = sample_tp8_logits(
        logits,
        temperatures=zeros_temp,
        top_p=ones_p,
        top_k=zeros_k,
        rng_seeds=seeds,
        penalty_state=state,
        slots=slots,
        reset_mask=torch.tensor([1], dtype=torch.uint8),
        presence_penalties=torch.tensor([1.0], dtype=torch.float32),
        frequency_penalties=torch.tensor([0.0], dtype=torch.float32),
    )
    assert t0.tolist() == [2]

    # Step 1 (decode, reset=0): token 2 now penalized by presence=1.0 -> 3.0-1.0=2.0 < 2.2 -> picks token 0
    t1 = sample_tp8_logits(
        logits,
        temperatures=zeros_temp,
        top_p=ones_p,
        top_k=zeros_k,
        rng_seeds=seeds,
        penalty_state=state,
        slots=slots,
        reset_mask=torch.tensor([0], dtype=torch.uint8),
        presence_penalties=torch.tensor([1.0], dtype=torch.float32),
        frequency_penalties=torch.tensor([0.0], dtype=torch.float32),
    )
    assert t1.tolist() == [0]

    # Step 2 (frequency penalty=0.5, presence=0.0): token 2 (count=1 -> 2.5) beats token 0 (count=1 -> 1.7)
    t2 = sample_tp8_logits(
        logits,
        temperatures=zeros_temp,
        top_p=ones_p,
        top_k=zeros_k,
        rng_seeds=seeds,
        penalty_state=state,
        slots=slots,
        reset_mask=torch.tensor([0], dtype=torch.uint8),
        presence_penalties=torch.tensor([0.0], dtype=torch.float32),
        frequency_penalties=torch.tensor([0.5], dtype=torch.float32),
    )
    assert t2.tolist() == [2]

    # Step 3 (frequency penalty=0.5): token 2 now count=2 -> 3.0 - 1.0 = 2.0 > token 0 (2.2 - 0.5 = 1.7) -> token 2
    # And write cursor wraps cleanly modulo max_history=3 without out-of-bounds!
    t3 = sample_tp8_logits(
        logits,
        temperatures=zeros_temp,
        top_p=ones_p,
        top_k=zeros_k,
        rng_seeds=seeds,
        penalty_state=state,
        slots=slots,
        reset_mask=torch.tensor([0], dtype=torch.uint8),
        presence_penalties=torch.tensor([0.0], dtype=torch.float32),
        frequency_penalties=torch.tensor([0.5], dtype=torch.float32),
    )
    assert t3.tolist() == [2]

    # Slot reuse with reset=1 clears history completely -> picks token 2 again even under strong penalty
    t_reset = sample_tp8_logits(
        logits,
        temperatures=zeros_temp,
        top_p=ones_p,
        top_k=zeros_k,
        rng_seeds=seeds,
        penalty_state=state,
        slots=slots,
        reset_mask=torch.tensor([1], dtype=torch.uint8),
        presence_penalties=torch.tensor([2.0], dtype=torch.float32),
        frequency_penalties=torch.tensor([2.0], dtype=torch.float32),
    )
    assert t_reset.tolist() == [2]

    # Mixed batch: row 0 (slot 0, unpenalized), row 1 (slot 1, positive penalty),
    # row 2 (slot 2, negative penalty encouraging repetition)
    mix_logits = torch.tensor(
        [
            [2.0, 3.0, 1.0, 0.0],  # slot 0: argmax = 1
            [2.2, 1.6, 3.0, 0.0],  # slot 1: already has token 2 recorded from t_reset
            [2.0, 2.5, 1.0, 0.0],  # slot 2: reset=1 first, then negative penalty reinforces token 1
        ],
        dtype=torch.float32,
    )
    mix_slots = torch.tensor([0, 1, 2], dtype=torch.int64)
    mix_out = sample_tp8_logits(
        mix_logits,
        temperatures=torch.tensor([0.0, 0.0, 0.0], dtype=torch.float32),
        top_p=torch.tensor([1.0, 1.0, 1.0], dtype=torch.float32),
        top_k=torch.tensor([0, 0, 0], dtype=torch.int32),
        rng_seeds=torch.tensor([1, 2, 3], dtype=torch.int64),
        penalty_state=state,
        slots=mix_slots,
        reset_mask=torch.tensor([1, 0, 1], dtype=torch.uint8),
        presence_penalties=torch.tensor([0.0, 1.0, -1.5], dtype=torch.float32),
        frequency_penalties=torch.tensor([0.0, 0.0, 0.0], dtype=torch.float32),
    )
    # Row 0 unpenalized -> 1; Row 1 penalized token 2 (3.0-1.0=2.0 < 2.2) -> 0; Row 2 fresh -> 1
    assert mix_out.tolist() == [1, 0, 1]

    # Next step with slot 2 having token 1 boosted by negative presence_penalty=-1.5 (+1.5 to token 1)
    mix_logits_step2 = torch.tensor(
        [
            [2.0, 3.0, 1.0, 0.0],  # slot 0 unpenalized -> still 1
            [2.2, 1.6, 3.0, 0.0],  # slot 1 has {2, 0} penalized -> 2.0 vs 1.2 vs 1.6 -> 2
            [3.5, 2.5, 1.0, 0.0],  # slot 2 token 1 boosted: 2.5 - (-1.5) = 4.0 > 3.5 -> 1
        ],
        dtype=torch.float32,
    )
    mix_out2 = sample_tp8_logits(
        mix_logits_step2,
        temperatures=torch.tensor([0.0, 0.0, 0.0], dtype=torch.float32),
        top_p=torch.tensor([1.0, 1.0, 1.0], dtype=torch.float32),
        top_k=torch.tensor([0, 0, 0], dtype=torch.int32),
        rng_seeds=torch.tensor([4, 5, 6], dtype=torch.int64),
        penalty_state=state,
        slots=mix_slots,
        reset_mask=torch.tensor([0, 0, 0], dtype=torch.uint8),
        presence_penalties=torch.tensor([0.0, 1.0, -1.5], dtype=torch.float32),
        frequency_penalties=torch.tensor([0.0, 0.0, 0.0], dtype=torch.float32),
    )
    assert mix_out2.tolist() == [1, 2, 1]


def test_sample_tp8_logits_return_top2() -> None:
    logits = torch.tensor(
        [
            [10.0, 5.0, 1.0, 0.0],
            [2.0, 8.0, 4.0, 1.0],
        ],
        dtype=torch.float32,
    )
    tokens, lps, ru_toks, ru_lps = sample_tp8_logits(
        logits,
        temperatures=torch.tensor([0.0, 0.0], dtype=torch.float32),
        top_p=torch.tensor([1.0, 1.0], dtype=torch.float32),
        top_k=torch.tensor([0, 0], dtype=torch.int32),
        rng_seeds=torch.tensor([0, 0], dtype=torch.int64),
        return_top2=True,
    )
    assert tokens.tolist() == [0, 1]
    assert lps.tolist() == [10.0, 8.0]
    assert ru_toks.tolist() == [1, 2]
    assert ru_lps.tolist() == [5.0, 4.0]

