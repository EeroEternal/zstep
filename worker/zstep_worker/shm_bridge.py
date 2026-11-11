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

"""Zero-copy POSIX shared-memory reader matching crates/zs-wire/src/lib.rs."""

from __future__ import annotations

from dataclasses import dataclass

ZSTEP_SHM_MAGIC: int = 0x5A53_5445_505F_5631
ZSTEP_PROTOCOL_VERSION: int = 1
STEP_HEADER_BYTES: int = 64
VEC_ALIGN: int = 256


def _align_up(offset: int, align: int) -> int:
    return (offset + align - 1) & ~(align - 1)


@dataclass(frozen=True)
class ShmGeometry:
    """Geometry parameters mirrored identically from Rust `zs_wire::ShmGeometry`."""

    world_size: int
    max_seqs: int
    max_step_tokens: int
    max_decoder_rows: int
    max_pages_per_seq: int
    page_block_tokens: int
    spec_verify_width: int


@dataclass(frozen=True)
class StepPlanLayout:
    """Byte offsets inside one double-buffered `StepPlanSlot`."""

    total_slot_bytes: int
    header_offset: int
    token_ids_offset: int
    positions_offset: int
    req_slot_ids_offset: int
    replay_mask_offset: int
    engram_prev_window_offset: int
    narrow_keep_indices_offset: int
    sample_row_indices_offset: int
    seq_lens_offset: int
    query_start_loc_offset: int
    global_kv_page_table_offset: int
    swa_page_table_offset: int
    compressor_slot_ids_offset: int
    temperatures_offset: int

    @classmethod
    def compute(cls, g: ShmGeometry) -> StepPlanLayout:
        max_t = g.max_step_tokens
        max_dec = g.max_decoder_rows
        max_b = g.max_seqs
        max_p = g.max_pages_per_seq

        header_offset = 0
        cursor = _align_up(STEP_HEADER_BYTES, VEC_ALIGN)

        token_ids_offset = cursor
        cursor = _align_up(cursor + max_t * 4, VEC_ALIGN)

        positions_offset = cursor
        cursor = _align_up(cursor + max_t * 4, VEC_ALIGN)

        req_slot_ids_offset = cursor
        cursor = _align_up(cursor + max_t * 4, VEC_ALIGN)

        replay_mask_offset = cursor
        cursor = _align_up(cursor + max_t, VEC_ALIGN)

        engram_prev_window_offset = cursor
        cursor = _align_up(cursor + max_t * 3 * 4, VEC_ALIGN)

        narrow_keep_indices_offset = cursor
        cursor = _align_up(cursor + max_dec * 4, VEC_ALIGN)

        sample_row_indices_offset = cursor
        cursor = _align_up(cursor + max_dec * 4, VEC_ALIGN)

        seq_lens_offset = cursor
        cursor = _align_up(cursor + max_b * 4, VEC_ALIGN)

        query_start_loc_offset = cursor
        cursor = _align_up(cursor + (max_b + 1) * 4, VEC_ALIGN)

        global_kv_page_table_offset = cursor
        cursor = _align_up(cursor + max_b * max_p * 4, VEC_ALIGN)

        swa_page_table_offset = cursor
        cursor = _align_up(cursor + max_b * 4 * 4, VEC_ALIGN)

        compressor_slot_ids_offset = cursor
        cursor = _align_up(cursor + max_b * 4, VEC_ALIGN)

        temperatures_offset = cursor
        cursor = _align_up(cursor + max_b * 4, 4096)

        return cls(
            total_slot_bytes=cursor,
            header_offset=header_offset,
            token_ids_offset=token_ids_offset,
            positions_offset=positions_offset,
            req_slot_ids_offset=req_slot_ids_offset,
            replay_mask_offset=replay_mask_offset,
            engram_prev_window_offset=engram_prev_window_offset,
            narrow_keep_indices_offset=narrow_keep_indices_offset,
            sample_row_indices_offset=sample_row_indices_offset,
            seq_lens_offset=seq_lens_offset,
            query_start_loc_offset=query_start_loc_offset,
            global_kv_page_table_offset=global_kv_page_table_offset,
            swa_page_table_offset=swa_page_table_offset,
            compressor_slot_ids_offset=compressor_slot_ids_offset,
            temperatures_offset=temperatures_offset,
        )
