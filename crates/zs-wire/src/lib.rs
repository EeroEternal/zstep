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
pub const ZSTEP_PROTOCOL_VERSION: u32 = 3;
/// Maximum GPU workers supported per single-node shared-memory segment.
pub const MAX_WORKERS: usize = 8;
/// Token ID sentinel indicating the GPU worker should self-bootstrap the token
/// on-device from the previous in-flight step's output slot (`in_flight_depth = 1`).
pub const DEVICE_BOOTSTRAP_TOKEN_ID: i32 = -1;
/// Reserved bytes at the start of the mapping for [`ShmSuperblock`].
pub const SHM_SUPERBLOCK_REGION_BYTES: usize = 4096;
/// Cacheline-aligned Region 0 offset (`0x80`) of the `AtomicU32` cluster abort word.
pub const SUPERBLOCK_ABORT_OFFSET: usize = 128;
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
/// [`StepHeader::flags`] bit indicating all active sequences have `q_len == 1` and
/// execute through a pre-captured decode CUDA Graph bucket.
pub const STEP_FLAG_DECODE_ONLY: u32 = 1 << 0;
/// [`StepHeader::flags`] bit for decode-only steps: each row's input token is the token the
/// worker sampled for that `req_slot_id` in the previous decode step, kept on the GPU, and
/// `token_ids` is ignored. Lets the master publish a step before reading the previous result.
pub const STEP_FLAG_DEVICE_TOKENS: u32 = 1 << 1;
/// [`StepHeader::flags`] bit promising that the master publishes step `step_id + 1` before it
/// waits for this step's result, so a worker may launch that next step before completing this one.
pub const STEP_FLAG_NEXT_PREPUBLISHED: u32 = 1 << 2;
/// Canonical CUDA Graph batch buckets for decode-only steps.
pub const DECODE_GRAPH_BUCKETS: [u32; 8] = [1, 2, 4, 8, 16, 32, 64, 128];
/// Maximum active decode sequences supported by pre-captured decode CUDA Graphs.
pub const MAX_DECODE_GRAPH_SEQS: u32 = 128;

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
    pub result_slot_stride: u64,
    pub layout_hash: u64,
    pub max_decode_graph_seqs: u32,
    pub _pad_u32: u32,
    pub _reserved: [u64; 1],
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
    /// Unexpected runtime Triton JIT compilations observed before shutdown.
    pub triton_recompilations: AtomicU32,
    /// Number of Triton kernels monitored by the runtime compilation spy.
    pub watched_kernels: AtomicU32,
    pub _pad_u32: u32,
    pub _pad: [u64; 1],
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

    /// Page-aligned byte stride of one `StepResultSlot` for `max_seqs`.
    pub const fn compute_result_slot_stride(max_seqs: u32) -> usize {
        let payload = std::mem::size_of::<StepResultHeader>()
            + (max_seqs as usize) * std::mem::size_of::<SeqStepOutput>();
        align_up(payload, 4096)
    }

    /// Deterministic 64-bit FNV-1a wire ABI hash over protocol constants, POD struct sizes,
    /// and all computed plan/result slot offsets so Python workers reject any layout drift.
    pub const fn layout_hash(&self, result_slot_stride: usize) -> u64 {
        const FNV_OFFSET_BASIS: u64 = 0xcbf2_9ce4_8422_2325;
        const FNV_PRIME: u64 = 0x0000_0100_0000_01b3;

        const fn mix_u64(mut hash: u64, value: u64) -> u64 {
            let bytes = value.to_le_bytes();
            let mut idx = 0;
            while idx < 8 {
                hash ^= bytes[idx] as u64;
                hash = hash.wrapping_mul(FNV_PRIME);
                idx += 1;
            }
            hash
        }

        let words = [
            ZSTEP_PROTOCOL_VERSION as u64,
            std::mem::size_of::<ShmSuperblock>() as u64,
            std::mem::size_of::<WorkerSlotState>() as u64,
            std::mem::size_of::<StepHeader>() as u64,
            std::mem::size_of::<StepResultHeader>() as u64,
            std::mem::size_of::<SeqStepOutput>() as u64,
            SUPERBLOCK_ABORT_OFFSET as u64,
            WORKER_STATES_OFFSET as u64,
            STEP_FLAG_DECODE_ONLY as u64,
            STEP_FLAG_DEVICE_TOKENS as u64,
            STEP_FLAG_NEXT_PREPUBLISHED as u64,
            MAX_DECODE_GRAPH_SEQS as u64,
            std::mem::offset_of!(ShmSuperblock, total_shm_bytes) as u64,
            std::mem::offset_of!(ShmSuperblock, plan_slot_offsets) as u64,
            std::mem::offset_of!(ShmSuperblock, plan_slot_stride) as u64,
            std::mem::offset_of!(ShmSuperblock, result_slot_offsets) as u64,
            std::mem::offset_of!(ShmSuperblock, result_slot_stride) as u64,
            std::mem::offset_of!(ShmSuperblock, layout_hash) as u64,
            std::mem::offset_of!(ShmSuperblock, max_decode_graph_seqs) as u64,
            std::mem::offset_of!(WorkerSlotState, heartbeat_ns) as u64,
            std::mem::offset_of!(WorkerSlotState, status_code) as u64,
            std::mem::offset_of!(WorkerSlotState, numa_node) as u64,
            std::mem::offset_of!(WorkerSlotState, triton_recompilations) as u64,
            std::mem::offset_of!(WorkerSlotState, watched_kernels) as u64,
            std::mem::offset_of!(StepHeader, flags) as u64,
            std::mem::offset_of!(SeqStepOutput, accepted_logprobs_bits) as u64,
            self.total_slot_bytes as u64,
            self.header_offset as u64,
            self.token_ids_offset as u64,
            self.positions_offset as u64,
            self.req_slot_ids_offset as u64,
            self.replay_mask_offset as u64,
            self.engram_prev_window_offset as u64,
            self.narrow_keep_indices_offset as u64,
            self.sample_row_indices_offset as u64,
            self.seq_lens_offset as u64,
            self.query_start_loc_offset as u64,
            self.global_kv_page_table_offset as u64,
            self.swa_page_table_offset as u64,
            self.compressor_slot_ids_offset as u64,
            self.temperatures_offset as u64,
            result_slot_stride as u64,
        ];

        let mut hash = FNV_OFFSET_BASIS;
        let mut i = 0;
        while i < words.len() {
            hash = mix_u64(hash, words[i]);
            i += 1;
        }
        hash
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

/// Optional Git SHA embedded at `cargo build` time when built from a Git checkout or with
/// `ZSTEP_GIT_SHA` set. Absent (`None`) when built from a `git archive` / sdist tree.
pub const BUILD_GIT_SHA: Option<&str> = option_env!("ZSTEP_BUILD_GIT_SHA");

fn normalize_git_sha(raw: &str, min_len: usize, short: bool) -> Result<String, String> {
    let trimmed = raw.trim();
    if trimmed.len() < min_len
        || trimmed.len() > 40
        || !trimmed.bytes().all(|b| b.is_ascii_hexdigit())
    {
        return Err(format!("invalid Git SHA provenance `{trimmed}`"));
    }
    let lower = trimmed.to_ascii_lowercase();
    if short {
        Ok(lower[..7].to_owned())
    } else {
        Ok(lower)
    }
}

fn read_git_dir_sha(repo_dir: &std::path::Path) -> Option<String> {
    let dot_git = repo_dir.join(".git");
    let git_dir = if dot_git.is_file() {
        let content = std::fs::read_to_string(&dot_git).ok()?;
        let rel = content.trim().strip_prefix("gitdir:")?.trim();
        let path = std::path::Path::new(rel);
        if path.is_absolute() {
            path.to_path_buf()
        } else {
            repo_dir.join(path)
        }
    } else if dot_git.is_dir() {
        dot_git
    } else {
        return None;
    };

    let head = std::fs::read_to_string(git_dir.join("HEAD")).ok()?;
    let head = head.trim();
    let raw_sha = if let Some(reference) = head.strip_prefix("ref: ") {
        let ref_path = git_dir.join(reference);
        if let Ok(contents) = std::fs::read_to_string(ref_path) {
            contents
        } else {
            let packed = std::fs::read_to_string(git_dir.join("packed-refs")).ok()?;
            packed
                .lines()
                .filter(|line| !line.starts_with('#') && !line.starts_with('^'))
                .find_map(|line| {
                    let (sha, packed_ref) = line.split_once(' ')?;
                    (packed_ref.trim() == reference).then(|| sha.to_owned())
                })?
        }
    } else {
        head.to_owned()
    };
    normalize_git_sha(&raw_sha, 7, false).ok()
}

/// Pure helper resolving Git provenance in `working_dir` with explicit overrides so tests can
/// exercise `git archive` / sdist directories deterministically without mutating process env.
pub fn resolve_git_revision_in(
    working_dir: &std::path::Path,
    env_override: Option<&str>,
    build_fallback: Option<&str>,
    short: bool,
) -> Result<String, String> {
    if let Some(raw_env) = env_override {
        let min_len = if short { 7 } else { 40 };
        return normalize_git_sha(raw_env, min_len, short);
    }

    if let Ok(output) = std::process::Command::new("git")
        .args(["rev-parse", "HEAD"])
        .current_dir(working_dir)
        .output()
        && output.status.success()
        && let Ok(stdout) = String::from_utf8(output.stdout)
        && let Ok(parsed) = normalize_git_sha(&stdout, 7, short)
    {
        return Ok(parsed);
    }

    if let Some(file_sha) = read_git_dir_sha(working_dir) {
        return normalize_git_sha(&file_sha, 7, short);
    }

    if let Some(build_sha) = build_fallback
        && let Ok(parsed) = normalize_git_sha(build_sha, 7, short)
    {
        return Ok(parsed);
    }

    Ok("unknown".to_owned())
}

/// Resolve the current Git commit SHA (`short = true` -> 7-char prefix, `short = false` -> 40-char
/// full SHA) from `ZSTEP_GIT_SHA`, `git rev-parse HEAD`, `.git/HEAD`, compile-time
/// [`BUILD_GIT_SHA`], or `"unknown"` when running from a `git archive` / sdist source tree.
pub fn resolve_git_sha(short: bool) -> Result<String, String> {
    let env_val = match std::env::var("ZSTEP_GIT_SHA") {
        Ok(val) if !val.trim().is_empty() => Some(val),
        Ok(_) | Err(std::env::VarError::NotPresent) => None,
        Err(err) => return Err(format!("invalid ZSTEP_GIT_SHA environment variable: {err}")),
    };
    let cwd = std::env::current_dir().unwrap_or_else(|_| std::path::PathBuf::from("."));
    resolve_git_revision_in(&cwd, env_val.as_deref(), BUILD_GIT_SHA, short)
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
        assert_eq!(std::mem::size_of::<SeqStepOutput>(), 104);
    }

    #[test]
    fn pod_field_offsets_match_wire_spec() {
        assert_eq!(std::mem::offset_of!(ShmSuperblock, magic), 0);
        assert_eq!(std::mem::offset_of!(ShmSuperblock, protocol_version), 8);
        assert_eq!(std::mem::offset_of!(ShmSuperblock, world_size), 12);
        assert_eq!(std::mem::offset_of!(ShmSuperblock, max_seqs), 16);
        assert_eq!(std::mem::offset_of!(ShmSuperblock, max_step_tokens), 20);
        assert_eq!(std::mem::offset_of!(ShmSuperblock, max_decoder_rows), 24);
        assert_eq!(std::mem::offset_of!(ShmSuperblock, max_pages_per_seq), 28);
        assert_eq!(std::mem::offset_of!(ShmSuperblock, page_block_tokens), 32);
        assert_eq!(std::mem::offset_of!(ShmSuperblock, spec_verify_width), 36);
        assert_eq!(std::mem::offset_of!(ShmSuperblock, master_pid), 40);
        assert_eq!(std::mem::offset_of!(ShmSuperblock, total_shm_bytes), 48);
        assert_eq!(std::mem::offset_of!(ShmSuperblock, plan_slot_offsets), 56);
        assert_eq!(std::mem::offset_of!(ShmSuperblock, plan_slot_stride), 72);
        assert_eq!(std::mem::offset_of!(ShmSuperblock, result_slot_offsets), 80);
        assert_eq!(std::mem::offset_of!(ShmSuperblock, result_slot_stride), 96);
        assert_eq!(std::mem::offset_of!(ShmSuperblock, layout_hash), 104);
        assert_eq!(
            std::mem::offset_of!(ShmSuperblock, max_decode_graph_seqs),
            112
        );
        assert_eq!(std::mem::offset_of!(ShmSuperblock, _pad_u32), 116);
        assert_eq!(std::mem::offset_of!(ShmSuperblock, _reserved), 120);

        assert_eq!(std::mem::offset_of!(WorkerSlotState, ready_step_id), 0);
        assert_eq!(std::mem::offset_of!(WorkerSlotState, heartbeat_ns), 8);
        assert_eq!(std::mem::offset_of!(WorkerSlotState, worker_pid), 16);
        assert_eq!(std::mem::offset_of!(WorkerSlotState, device_sm_arch), 20);
        assert_eq!(std::mem::offset_of!(WorkerSlotState, status_code), 24);
        assert_eq!(std::mem::offset_of!(WorkerSlotState, error_code), 28);
        assert_eq!(std::mem::offset_of!(WorkerSlotState, free_vram_mb), 32);
        assert_eq!(std::mem::offset_of!(WorkerSlotState, total_vram_mb), 36);
        assert_eq!(std::mem::offset_of!(WorkerSlotState, numa_node), 40);
        assert_eq!(
            std::mem::offset_of!(WorkerSlotState, triton_recompilations),
            44
        );
        assert_eq!(std::mem::offset_of!(WorkerSlotState, watched_kernels), 48);

        assert_eq!(std::mem::offset_of!(StepHeader, step_id), 0);
        assert_eq!(std::mem::offset_of!(StepHeader, opcode), 8);
        assert_eq!(std::mem::offset_of!(StepHeader, num_seqs), 12);
        assert_eq!(std::mem::offset_of!(StepHeader, num_encoder_tokens), 16);
        assert_eq!(std::mem::offset_of!(StepHeader, encoder_graph_bucket), 20);
        assert_eq!(std::mem::offset_of!(StepHeader, num_decoder_rows), 24);
        assert_eq!(std::mem::offset_of!(StepHeader, decoder_graph_bucket), 28);
        assert_eq!(std::mem::offset_of!(StepHeader, num_sample_rows), 32);
        assert_eq!(std::mem::offset_of!(StepHeader, is_dspark_verify), 36);
        assert_eq!(std::mem::offset_of!(StepHeader, flags), 40);

        assert_eq!(std::mem::offset_of!(StepResultHeader, completed_step_id), 0);
        assert_eq!(std::mem::offset_of!(StepResultHeader, gpu_exec_us), 8);
        assert_eq!(std::mem::offset_of!(StepResultHeader, status_code), 12);
        assert_eq!(std::mem::offset_of!(StepResultHeader, num_seq_results), 16);

        assert_eq!(std::mem::offset_of!(SeqStepOutput, req_slot), 0);
        assert_eq!(std::mem::offset_of!(SeqStepOutput, num_accepted), 4);
        assert_eq!(std::mem::offset_of!(SeqStepOutput, accepted_tokens), 8);
        assert_eq!(std::mem::offset_of!(SeqStepOutput, next_draft_tokens), 40);
        assert_eq!(
            std::mem::offset_of!(SeqStepOutput, accepted_logprobs_bits),
            72
        );
    }

    #[test]
    fn git_provenance_resolves_without_dot_git_directory() {
        let isolated = std::env::temp_dir().join(format!("zstep_gitless_{}", std::process::id()));
        let _ = std::fs::remove_dir_all(&isolated);
        std::fs::create_dir_all(&isolated).expect("create temp dir");

        // 1) Pure `git archive` / sdist tree (no `.git`, no `ZSTEP_GIT_SHA`, no build SHA):
        assert_eq!(
            resolve_git_revision_in(&isolated, None, None, true).expect("short fallback"),
            "unknown"
        );
        assert_eq!(
            resolve_git_revision_in(&isolated, None, None, false).expect("full fallback"),
            "unknown"
        );

        // 2) `ZSTEP_GIT_SHA` override works in a git-less directory:
        let sample_sha = "0123456789abcdef0123456789abcdef01234567";
        assert_eq!(
            resolve_git_revision_in(&isolated, Some(sample_sha), None, true).expect("env short"),
            "0123456"
        );
        assert_eq!(
            resolve_git_revision_in(&isolated, Some(sample_sha), None, false).expect("env full"),
            sample_sha
        );

        // 3) Malformed `ZSTEP_GIT_SHA` still fails loudly:
        assert!(resolve_git_revision_in(&isolated, Some("not-hex!"), None, false).is_err());

        // 4) Compile-time `BUILD_GIT_SHA` rescues binary copied outside `.git`:
        assert_eq!(
            resolve_git_revision_in(&isolated, None, Some(sample_sha), true).expect("build short"),
            "0123456"
        );

        let _ = std::fs::remove_dir_all(&isolated);
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

        // Verified 8x84GB profile (DeepSeek-V4.1-Flash TP=8 on 8x RTX 6000D 85,651 MiB:
        // ~65.1 GiB weights incl. VRAM-resident Engram + ~4.9 GiB workspace,
        // 4 MLA+Indexer KV source layers [2,8,14,20] with compress_ratios [2,2,2,1] ->
        // 227,840 B per 256-token page = 890 B/tok/GPU).
        let geom_84g = ShmGeometry::from_vram_budget(VramGeometrySpec {
            world_size: 8,
            per_gpu_vram_mb: 85_651,
            reserved_weights_and_workspace_mb: 71_680,
            kv_bytes_per_token_per_gpu: 890,
            max_seqs: 128,
            max_step_tokens: 8_192,
            max_decoder_rows: 8_192,
            page_block_tokens: 256,
            spec_verify_width: 6,
        })
        .expect("valid 84GB DeepSeek-V4.1-Flash geometry");
        assert_eq!(geom_84g.page_block_tokens, 256);
        assert!(geom_84g.max_pages_per_seq >= 128);

        for geom in [
            geom_32g,
            geom_84g,
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
