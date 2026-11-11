// Copyright (c) 2026 zstep Contributors
//
// Permission is hereby granted, free of charge, to any person obtaining a copy
// of this software and associated documentation files (the "Software"), to deal
// in the Software without restriction, including without limitation the rights
// to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
// copies of the Software, and to permit persons to whom the Software is
// furnished to do so, subject to the following conditions:
//
// The above copyright notice and this permission notice shall be included in all
// copies or substantial portions of the Software.
//
// THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
// IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
// FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
// AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
// LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
// OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
// SOFTWARE.

//! Zero-copy `#[repr(C)]` shared-memory protocol definitions for Mode B
//! (1 pure-Rust Master Control Plane + N GPU Worker processes).
//!
//! See `docs/03-shm-protocol.md` for the full memory map and barrier rules.

use std::sync::atomic::{AtomicU32, AtomicU64};

/// ASCII `"ZSTEP_V1"` in little-endian (`0x5A53_5445_505F_5631`).
pub const ZSTEP_SHM_MAGIC: u64 = 0x5A53_5445_505F_5631;
/// Wire ABI version. Must match exactly between master and GPU workers.
pub const ZSTEP_PROTOCOL_VERSION: u32 = 1;
/// Maximum GPU workers supported per single-node shared-memory segment.
pub const MAX_WORKERS: usize = 8;
/// Token ID sentinel indicating the GPU worker should self-bootstrap the token
/// on-device from the previous in-flight step's output slot (`in_flight_depth = 1`).
pub const DEVICE_BOOTSTRAP_TOKEN_ID: i32 = -1;

/// Step operation codes published in [`StepHeader::opcode`].
#[repr(u32)]
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum StepOpcode {
    Empty = 0,
    NormalStep = 1,
    WarmupCaptureGraphs = 2,
    ReloadForward = 3,
    Shutdown = 99,
}

/// Region 0 (`0x0000_0000`, 4 KiB reserved): immutable geometry handshake header
/// written once by the Rust master before spawning GPU workers.
#[repr(C, align(64))]
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct ShmSuperblock {
    pub magic: u64,
    pub protocol_version: u32,
    pub world_size: u32,
    pub max_seqs: u32,
    pub max_step_tokens: u32,
    pub max_decoder_rows: u32,
    pub max_pages_per_seq: u32,
    pub page_block_tokens: u32,
    pub spec_verify_width: u32,
    pub master_pid: u32,
    pub total_shm_bytes: u64,
    pub plan_slot_offsets: [u64; 2],
    pub plan_slot_stride: u64,
    pub result_slot_offsets: [u64; 2],
    pub _reserved: [u64; 4],
}

/// Per-worker status block inside Region 1 (`0x0000_1000`).
/// Aligned to 64 bytes so each GPU worker writes its own cache line without
/// false sharing across ranks.
#[repr(C, align(64))]
pub struct WorkerSlotState {
    pub ready_step_id: AtomicU64,
    pub heartbeat_ns: AtomicU64,
    pub worker_pid: AtomicU32,
    pub device_sm_arch: AtomicU32,
    pub status_code: AtomicU32,
    pub error_code: AtomicU32,
    pub free_vram_mb: AtomicU32,
    pub total_vram_mb: AtomicU32,
    pub _pad: [u64; 2],
}

/// Header at the start of each double-buffered `StepPlanSlot` (`slot = step_id & 1`).
///
/// The master populates all payload arrays in the slot first, then publishes
/// `step_id` with `Ordering::Release` before signaling `plan_efd`.
#[repr(C, align(64))]
pub struct StepHeader {
    pub step_id: AtomicU64,
    pub opcode: u32,
    pub num_seqs: u32,
    /// Stage 1 (Layers 0..19) active token rows (`T_enc`).
    pub num_encoder_tokens: u32,
    /// Stage 1 padded CUDA Graph bucket (`0` selects eager execution).
    pub encoder_graph_bucket: u32,
    /// Stage 3 (Layers 21..39) narrowed active rows (`T_dec`) after Layer 20 CED cut.
    pub num_decoder_rows: u32,
    /// Stage 3 padded CUDA Graph bucket (`0` selects eager execution).
    pub decoder_graph_bucket: u32,
    /// Rows entering the target LM head / verifier (`<= num_decoder_rows`).
    pub num_sample_rows: u32,
    /// `1` when executing DSpark 6-token speculative verification + next draft.
    pub is_dspark_verify: u32,
    pub flags: u32,
    pub _reserved: [u32; 3],
}

/// Byte offsets of every flat tensor slice inside one `StepPlanSlot`,
/// computed deterministically from [`ShmGeometry`].
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct StepPlanLayout {
    pub total_slot_bytes: usize,
    pub header_offset: usize,
    pub token_ids_offset: usize,
    pub positions_offset: usize,
    pub req_slot_ids_offset: usize,
    pub replay_mask_offset: usize,
    pub engram_prev_window_offset: usize,
    pub narrow_keep_indices_offset: usize,
    pub sample_row_indices_offset: usize,
    pub seq_lens_offset: usize,
    pub query_start_loc_offset: usize,
    pub global_kv_page_table_offset: usize,
    pub swa_page_table_offset: usize,
    pub compressor_slot_ids_offset: usize,
    pub temperatures_offset: usize,
}

/// Geometry parameters that size the shared-memory segment.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct ShmGeometry {
    pub world_size: u32,
    pub max_seqs: u32,
    pub max_step_tokens: u32,
    pub max_decoder_rows: u32,
    pub max_pages_per_seq: u32,
    pub page_block_tokens: u32,
    pub spec_verify_width: u32,
}

const fn align_up(offset: usize, align: usize) -> usize {
    (offset + align - 1) & !(align - 1)
}

impl StepPlanLayout {
    /// Compute 256-byte-aligned array offsets for vectorized GPU H2D copies.
    pub const fn compute(g: ShmGeometry) -> Self {
        const VEC_ALIGN: usize = 256;
        let max_t = g.max_step_tokens as usize;
        let max_dec = g.max_decoder_rows as usize;
        let max_b = g.max_seqs as usize;
        let max_p = g.max_pages_per_seq as usize;

        let header_offset = 0;
        let mut cursor = align_up(std::mem::size_of::<StepHeader>(), VEC_ALIGN);

        let token_ids_offset = cursor;
        cursor = align_up(cursor + max_t * 4, VEC_ALIGN);

        let positions_offset = cursor;
        cursor = align_up(cursor + max_t * 4, VEC_ALIGN);

        let req_slot_ids_offset = cursor;
        cursor = align_up(cursor + max_t * 4, VEC_ALIGN);

        let replay_mask_offset = cursor;
        cursor = align_up(cursor + max_t, VEC_ALIGN);

        // 3-token normalized history window per row: [max_step_tokens, 3] i32
        let engram_prev_window_offset = cursor;
        cursor = align_up(cursor + max_t * 3 * 4, VEC_ALIGN);

        let narrow_keep_indices_offset = cursor;
        cursor = align_up(cursor + max_dec * 4, VEC_ALIGN);

        let sample_row_indices_offset = cursor;
        cursor = align_up(cursor + max_dec * 4, VEC_ALIGN);

        let seq_lens_offset = cursor;
        cursor = align_up(cursor + max_b * 4, VEC_ALIGN);

        let query_start_loc_offset = cursor;
        cursor = align_up(cursor + (max_b + 1) * 4, VEC_ALIGN);

        let global_kv_page_table_offset = cursor;
        cursor = align_up(cursor + max_b * max_p * 4, VEC_ALIGN);

        // SWA ring uses at most 4 physical pages per sequence
        let swa_page_table_offset = cursor;
        cursor = align_up(cursor + max_b * 4 * 4, VEC_ALIGN);

        let compressor_slot_ids_offset = cursor;
        cursor = align_up(cursor + max_b * 4, VEC_ALIGN);

        let temperatures_offset = cursor;
        cursor = align_up(cursor + max_b * 4, 4096);

        Self {
            total_slot_bytes: cursor,
            header_offset,
            token_ids_offset,
            positions_offset,
            req_slot_ids_offset,
            replay_mask_offset,
            engram_prev_window_offset,
            narrow_keep_indices_offset,
            sample_row_indices_offset,
            seq_lens_offset,
            query_start_loc_offset,
            global_kv_page_table_offset,
            swa_page_table_offset,
            compressor_slot_ids_offset,
            temperatures_offset,
        }
    }
}

/// Header at the beginning of each `StepResultSlot` written by Worker Rank 0.
#[repr(C, align(64))]
pub struct StepResultHeader {
    pub completed_step_id: AtomicU64,
    pub gpu_exec_us: u32,
    pub status_code: u32,
    pub num_seq_results: u32,
    pub _pad: [u32; 11],
}

/// Per-request output entry inside `StepResultSlot` (supports both 1-token
/// standard decode and up to 6-token DSpark speculative verification).
#[repr(C)]
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct SeqStepOutput {
    pub req_slot: u32,
    pub num_accepted: u32,
    pub accepted_tokens: [i32; 8],
    pub next_draft_tokens: [i32; 8],
    pub accepted_logprobs_bits: [u32; 8],
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn cacheline_alignments_and_sizes_match_spec() {
        assert_eq!(std::mem::align_of::<ShmSuperblock>(), 64);
        assert_eq!(std::mem::size_of::<ShmSuperblock>(), 128);
        assert_eq!(std::mem::align_of::<WorkerSlotState>(), 64);
        assert_eq!(std::mem::size_of::<WorkerSlotState>(), 64);
        assert_eq!(std::mem::align_of::<StepHeader>(), 64);
        assert_eq!(std::mem::size_of::<StepHeader>(), 64);
        assert_eq!(std::mem::align_of::<StepResultHeader>(), 64);
        assert_eq!(std::mem::size_of::<StepResultHeader>(), 64);
    }

    #[test]
    fn dsv41_flash_8x6000d_geometry_fits_budget() {
        let geom = ShmGeometry {
            world_size: 8,
            max_seqs: 64,
            max_step_tokens: 16384,
            max_decoder_rows: 8192,
            max_pages_per_seq: 2048,
            page_block_tokens: 64,
            spec_verify_width: 6,
        };
        let layout = StepPlanLayout::compute(geom);
        assert_eq!(layout.token_ids_offset % 256, 0);
        assert_eq!(layout.global_kv_page_table_offset % 256, 0);
        assert_eq!(layout.total_slot_bytes % 4096, 0);
        // Single StepPlanSlot for 64 seqs @ 128K ctx is exactly 980 KiB (1,003,520 bytes)
        assert_eq!(layout.total_slot_bytes, 1_003_520);
    }
}
