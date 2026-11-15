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

"""Shared deterministic target verification for all speculative methods."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from zstep_worker.shm_bridge import SeqResult


@dataclass(frozen=True)
class VerifiedSequence:
    req_slot: int
    batch_index: int
    rows: tuple[int, ...]
    accepted_tokens: list[int]
    accepted_logprobs_bits: list[int]


def verify_greedy(
    predictions: list[int], input_tokens: list[int]
) -> list[int]:
    """Accept matching drafts and the first mismatching or bonus target token."""
    if not predictions or len(input_tokens) != len(predictions):
        raise ValueError("verification requires equal nonzero prediction/input rows")
    accepted = [int(predictions[0])]
    for row in range(1, len(predictions)):
        if predictions[row - 1] != input_tokens[row]:
            break
        accepted.append(int(predictions[row]))
    return accepted


def verify_batch(
    *,
    predictions: np.ndarray,
    logprob_bits: np.ndarray,
    token_ids: np.ndarray,
    sample_rows: np.ndarray,
    query_start_loc: np.ndarray,
    req_slot_ids: np.ndarray,
) -> list[VerifiedSequence]:
    """Group sampled varlen rows by sequence and verify each independently."""
    sample_lookup = {int(row): index for index, row in enumerate(sample_rows)}
    verified: list[VerifiedSequence] = []
    for seq_idx in range(len(query_start_loc) - 1):
        begin = int(query_start_loc[seq_idx])
        end = int(query_start_loc[seq_idx + 1])
        rows = tuple(row for row in range(begin, end) if row in sample_lookup)
        if not rows:
            continue
        sample_indices = [sample_lookup[row] for row in rows]
        row_predictions = [int(predictions[index]) for index in sample_indices]
        row_inputs = [int(token_ids[row]) for row in rows]
        accepted = verify_greedy(row_predictions, row_inputs)
        accepted_bits = [
            int(logprob_bits[index]) & 0xFFFF_FFFF
            for index in sample_indices[: len(accepted)]
        ]
        verified.append(
            VerifiedSequence(
                req_slot=int(req_slot_ids[rows[0]]),
                batch_index=seq_idx,
                rows=rows,
                accepted_tokens=accepted,
                accepted_logprobs_bits=accepted_bits,
            )
        )
    return verified


def seq_results(
    verified: list[VerifiedSequence], drafts: dict[int, list[int]]
) -> list[SeqResult]:
    """Build rank-0 SHM results from identical verification data on every rank."""
    outputs: list[SeqResult] = []
    for sequence in verified:
        accepted = sequence.accepted_tokens
        logprobs = sequence.accepted_logprobs_bits
        next_drafts = drafts.get(sequence.req_slot, [])
        outputs.append(
            SeqResult(
                req_slot=sequence.req_slot,
                accepted_token=accepted[0],
                accepted_logprob_bits=logprobs[0],
                accepted_tokens=accepted,
                accepted_logprobs_bits=logprobs,
                next_draft_tokens=next_drafts,
            )
        )
    return outputs
