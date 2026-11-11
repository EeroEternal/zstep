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

"""Verify Python StepPlanLayout offsets match Rust zs-wire byte-for-byte."""

from __future__ import annotations

from zstep_worker.shm_bridge import ShmGeometry, StepPlanLayout


def test_step_plan_layout_matches_rust_wire() -> None:
    geom = ShmGeometry(
        world_size=8,
        max_seqs=64,
        max_step_tokens=16384,
        max_decoder_rows=8192,
        max_pages_per_seq=2048,
        page_block_tokens=64,
        spec_verify_width=6,
    )
    layout = StepPlanLayout.compute(geom)
    assert layout.token_ids_offset == 256
    assert layout.positions_offset == 256 + 65536
    assert layout.global_kv_page_table_offset % 256 == 0
    assert layout.total_slot_bytes % 4096 == 0
    assert layout.total_slot_bytes == 1_003_520


if __name__ == "__main__":
    test_step_plan_layout_matches_rust_wire()
    print("Python <-> Rust StepPlanLayout byte parity OK (980 KiB/slot)")
