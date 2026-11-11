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

use std::sync::atomic::{AtomicI32, AtomicU32, AtomicU64};

/// ASCII `"ZSTEP_V1"` in little-endian (`0x5A53_5445_505F_5631`).
pub const ZSTEP_SHM_MAGIC: u64 = 0x5A53_5445_505F_5631;
/// Wire ABI version. Must match exactly between master and GPU workers.
pub const ZSTEP_PROTOCOL_VERSION: u32 = 1;
/// Maximum GPU workers supported per single-node shared-memory segment.
pub const MAX_WORKERS: usize = 8;
/// Token ID sentinel indicating the GPU worker should self-bootstrap the token
/// on-device from the previous in-flight step's output slot (`in_flight_depth = 1`).
pub const DEVICE_BOOTSTRAP_TOKEN_ID: i32 = -1;
/// Reserved bytes at the start of the mapping for [`ShmSuperblock`].
pub const SHM_SUPERBLOCK_REGION_BYTES: usize = 4096;
/// Byte offset of the fixed eight-entry worker status array.
pub const WORKER_STATES_OFFSET: usize = 4096;
/// Reserved bytes for the worker status array.
pub const WORKER_STATES_REGION_BYTES: usize = 4096;
/// Worker initialization completed and the process is ready for plans.
pub const WORKER_STATUS_READY: u32 = 1;
/// Worker is executing a published plan.
pub const WORKER_STATUS_RUNNING: u32 = 2;
/// Worker stopped because of an exception.
pub const WORKER_STATUS_ERROR: u32 = 99;

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
    /// Host NUMA node (`>= 0`) detected from `/sys/bus/pci/devices/<bdf>/numa_node`
    /// and pinned via `sched_setaffinity`, or `-1` when the platform exposes no NUMA affinity.
    pub numa_node: AtomicI32,
    pub _pad_u32: u32,
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
    /// Stage 1 active token rows (`T_enc`).
    pub num_encoder_tokens: u32,
    /// Stage 1 padded CUDA Graph bucket (`0` selects eager execution).
    pub encoder_graph_bucket: u32,
    /// Stage 2 narrowed active rows (`T_dec`) entering the second half / LM head.
    pub num_decoder_rows: u32,
    /// Stage 2 padded CUDA Graph bucket (`0` selects eager execution).
    pub decoder_graph_bucket: u32,
    /// Rows entering the target LM head / verifier (`<= num_decoder_rows`).
    pub num_sample_rows: u32,
    /// `1` when executing speculative verification + next draft.
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

/// Probed device and model memory parameters used to derive [`ShmGeometry`].
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct VramGeometrySpec {
    pub world_size: u32,
    pub per_gpu_vram_mb: u32,
    pub reserved_weights_and_workspace_mb: u32,
    pub kv_bytes_per_token_per_gpu: u32,
    pub max_seqs: u32,
    pub max_step_tokens: u32,
    pub max_decoder_rows: u32,
    pub page_block_tokens: u32,
    pub spec_verify_width: u32,
}

impl ShmGeometry {
    /// Derive shared-memory plan geometry from a probed per-GPU VRAM capacity, per-GPU weight
    /// + workspace reservation, and per-token KV footprint across all paged attention layers.
    pub fn from_vram_budget(spec: VramGeometrySpec) -> Option<Self> {
        if spec.world_size == 0
            || spec.world_size as usize > MAX_WORKERS
            || spec.max_seqs == 0
            || spec.max_step_tokens == 0
            || spec.max_decoder_rows == 0
            || spec.max_decoder_rows > spec.max_step_tokens
            || spec.page_block_tokens == 0
            || spec.kv_bytes_per_token_per_gpu == 0
            || spec.per_gpu_vram_mb <= spec.reserved_weights_and_workspace_mb
        {
            return None;
        }
        let kv_budget_bytes =
            u64::from(spec.per_gpu_vram_mb - spec.reserved_weights_and_workspace_mb) * 1024 * 1024;
        let bytes_per_page =
            u64::from(spec.kv_bytes_per_token_per_gpu) * u64::from(spec.page_block_tokens);
        let total_pages = (kv_budget_bytes / bytes_per_page) as u32;
        if total_pages == 0 {
            return None;
        }
        let max_pages_per_seq = total_pages.min(4096);
        Some(Self {
            world_size: spec.world_size,
            max_seqs: spec.max_seqs,
            max_step_tokens: spec.max_step_tokens,
            max_decoder_rows: spec.max_decoder_rows,
            max_pages_per_seq,
            page_block_tokens: spec.page_block_tokens,
            spec_verify_width: spec.spec_verify_width,
        })
    }
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
/// standard decode and up to 6-token speculative verification).
///
/// When `num_accepted == 1`, `accepted_tokens[0]` and `accepted_logprobs_bits[0]`
/// store the greedy argmax token and its IEEE-754 `f32` top-1 logit bits, while
/// `accepted_tokens[1]` (when `>= 0`) and `accepted_logprobs_bits[1]` carry the
/// global runner-up top-2 token and `f32` top-2 logit bits across all TP ranks.
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
    fn layout_invariants_hold_for_probed_vram_profiles() {
        // Verified 8x32GB profile (e.g. Qwen3.5-35B-A3B TP=8: ~8.5 GiB weights + ~3.5 GiB workspace,
        // 10 full-attention GQA layers * 2 KV heads * 256 dim * 2B / TP-replicated = 10 KiB/tok/GPU).
        let geom_32g = ShmGeometry::from_vram_budget(VramGeometrySpec {
            world_size: 8,
            per_gpu_vram_mb: 32_607,
            reserved_weights_and_workspace_mb: 12_288,
            kv_bytes_per_token_per_gpu: 10_240,
            max_seqs: 128,
            max_step_tokens: 8_192,
            max_decoder_rows: 8_192,
            page_block_tokens: 64,
            spec_verify_width: 4,
        })
        .expect("valid 32GB geometry");
        assert!(geom_32g.max_pages_per_seq > 0);

        for geom in [
            geom_32g,
            ShmGeometry {
                world_size: 1,
                max_seqs: 1,
                max_step_tokens: 7,
                max_decoder_rows: 3,
                max_pages_per_seq: 5,
                page_block_tokens: 64,
                spec_verify_width: 1,
            },
        ] {
            let layout = StepPlanLayout::compute(geom);
            let regions = [
                (layout.header_offset, std::mem::size_of::<StepHeader>()),
                (layout.token_ids_offset, geom.max_step_tokens as usize * 4),
                (layout.positions_offset, geom.max_step_tokens as usize * 4),
                (
                    layout.req_slot_ids_offset,
                    geom.max_step_tokens as usize * 4,
                ),
                (layout.replay_mask_offset, geom.max_step_tokens as usize),
                (
                    layout.engram_prev_window_offset,
                    geom.max_step_tokens as usize * 12,
                ),
                (
                    layout.narrow_keep_indices_offset,
                    geom.max_decoder_rows as usize * 4,
                ),
                (
                    layout.sample_row_indices_offset,
                    geom.max_decoder_rows as usize * 4,
                ),
                (layout.seq_lens_offset, geom.max_seqs as usize * 4),
                (
                    layout.query_start_loc_offset,
                    (geom.max_seqs as usize + 1) * 4,
                ),
                (
                    layout.global_kv_page_table_offset,
                    geom.max_seqs as usize * geom.max_pages_per_seq as usize * 4,
                ),
                (layout.swa_page_table_offset, geom.max_seqs as usize * 16),
                (
                    layout.compressor_slot_ids_offset,
                    geom.max_seqs as usize * 4,
                ),
                (layout.temperatures_offset, geom.max_seqs as usize * 4),
            ];
            for window in regions.windows(2) {
                let (prev_off, prev_len) = window[0];
                let (next_off, _) = window[1];
                assert!(
                    prev_off + prev_len <= next_off,
                    "overlapping SHM slices: {prev_off}+{prev_len} > {next_off}"
                );
                assert_eq!(next_off % 256, 0, "payload slice must be 256B aligned");
            }
            let (last_off, last_len) = *regions.last().expect("regions nonempty");
            assert!(last_off + last_len <= layout.total_slot_bytes);
            assert_eq!(layout.total_slot_bytes % 4096, 0);
        }

        // Insufficient VRAM budget must return None cleanly.
        assert!(
            ShmGeometry::from_vram_budget(VramGeometrySpec {
                world_size: 8,
                per_gpu_vram_mb: 16_384,
                reserved_weights_and_workspace_mb: 20_000,
                kv_bytes_per_token_per_gpu: 10_240,
                max_seqs: 32,
                max_step_tokens: 1024,
                max_decoder_rows: 1024,
                page_block_tokens: 64,
                spec_verify_width: 1,
            })
            .is_none()
        );
    }
}
