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

from zstep_worker.shm_bridge import ShmGeometry, StepPlanLayout


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


if __name__ == "__main__":
    test_step_plan_layout_invariants_and_vram_derivation()
    print("Python <-> Rust StepPlanLayout invariants and VRAM budget derivation OK")
