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

//! Continuous-batching serving scheduler and multi-tier benchmark harness.
//!
//! Owns request slots and physical 16-token KV pages via RAII handles (`SlotLease`,
//! `zs_kv::BlockLease`) and overlaps next-step decode preparation (`in_flight_depth = 1`)
//! while the 8 GPU workers execute the current step.

use std::collections::VecDeque;
use std::sync::atomic::AtomicU64;
use std::sync::{Arc, Mutex};
use std::time::Instant;

use zs_kv::{
    BlockLease, BlockPool, CacheGroupKind, DramPrefixCache, DramTierConfig, DramTierStats,
    KvTransferOp, PrefixLease, PrefixTree,
};
use zs_wire::{
    DECODE_GRAPH_BUCKETS, KV_TRANSFER_DIR_D2H, KV_TRANSFER_DIR_H2D, MAX_DECODE_GRAPH_SEQS,
    STEP_FLAG_DECODE_ONLY, STEP_FLAG_DEVICE_TOKENS, STEP_FLAG_NEXT_PREPUBLISHED, STEP_FLAG_PENALTY,
    STEP_FLAG_SAMPLING, StepHeader, StepOpcode,
};

use crate::coordinator::{
    EngineConfig, EngineError, Result, SpeculativeConfig, SpeculativeMethod, WorkerCoordinator,
};
use crate::select_graph_bucket;

struct SlotPoolInner {
    free_slots: Vec<u32>,
    capacity: usize,
}

/// Free-list pool of active sequence slot IDs `0..max_seqs` with RAII release on `Drop`.
#[derive(Clone)]
pub struct SlotPool {
    inner: Arc<Mutex<SlotPoolInner>>,
}

impl SlotPool {
    pub fn new(max_seqs: u32) -> Self {
        let free_slots = (0..max_seqs).rev().collect();
        Self {
            inner: Arc::new(Mutex::new(SlotPoolInner {
                free_slots,
                capacity: max_seqs as usize,
            })),
        }
    }

    pub fn available(&self) -> usize {
        self.inner
            .lock()
            .unwrap_or_else(|poisoned| poisoned.into_inner())
            .free_slots
            .len()
    }

    pub fn capacity(&self) -> usize {
        self.inner
            .lock()
            .unwrap_or_else(|poisoned| poisoned.into_inner())
            .capacity
    }

    pub fn allocate(&self) -> Option<SlotLease> {
        let mut guard = self
            .inner
            .lock()
            .unwrap_or_else(|poisoned| poisoned.into_inner());
        let slot = guard.free_slots.pop()?;
        Some(SlotLease {
            pool: Arc::clone(&self.inner),
            slot: Some(slot),
        })
    }
}

/// RAII handle returning a request slot ID to [`SlotPool`] on `Drop` without panicking.
pub struct SlotLease {
    pool: Arc<Mutex<SlotPoolInner>>,
    slot: Option<u32>,
}

impl SlotLease {
    pub fn id(&self) -> u32 {
        self.slot.unwrap_or(0)
    }
}

impl Drop for SlotLease {
    fn drop(&mut self) {
        if let Some(slot) = self.slot.take() {
            let mut guard = self
                .pool
                .lock()
                .unwrap_or_else(|poisoned| poisoned.into_inner());
            guard.free_slots.push(slot);
        }
    }
}

mod report;
pub use self::report::*;

/// Per-step greedy Top-2 logit diagnostics gathered across all TP ranks.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct StepTop2Logits {
    pub top1_token: i32,
    pub top1_logit: f32,
    pub top2_token: i32,
    pub top2_logit: f32,
    pub top1_minus_top2_margin: f32,
}

/// Generated token IDs paired with per-step Top-2 logit diagnostics for a batch of prompts.
pub type GreedyTop2Batch = (Vec<Vec<i32>>, Vec<Vec<StepTop2Logits>>);

#[derive(Debug, Clone)]
struct HarvestedOutput {
    req_slot: u32,
    accepted_tokens: Vec<i32>,
    accepted_logprobs_bits: Vec<u32>,
    next_draft_tokens: Vec<i32>,
    runner_up_token: i32,
    runner_up_logprob_bits: u32,
}

/// Per-request sampling configuration forwarded to GPU workers via SHM v5.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct SamplingParams {
    /// Softmax temperature (`0.0` = zero-overhead greedy argmax fast path when penalties are 0).
    pub temperature: f32,
    /// Nucleus cumulative probability threshold in `(0.0, 1.0]`.
    pub top_p: f32,
    /// Top-K truncation (`0` or `-1` = disabled; `>= 1` = keep top-k tokens).
    pub top_k: i32,
    /// Deterministic base PRNG seed for per-step SplitMix64 sampling.
    pub seed: u64,
    /// OpenAI presence penalty in `[-2.0, 2.0]` (`0.0` = disabled).
    pub presence_penalty: f32,
    /// OpenAI frequency penalty in `[-2.0, 2.0]` (`0.0` = disabled).
    pub frequency_penalty: f32,
    /// Continue generating past EOS tokens until `max_tokens` or caller stop strings.
    pub ignore_eos: bool,
    /// Whether to return log probabilities for output tokens.
    pub logprobs: bool,
    /// Number of top log probabilities to return per token (0..=5).
    pub top_logprobs: u32,
}

impl Default for SamplingParams {
    fn default() -> Self {
        Self {
            temperature: 0.0,
            top_p: 1.0,
            top_k: 0,
            seed: 0,
            presence_penalty: 0.0,
            frequency_penalty: 0.0,
            ignore_eos: false,
            logprobs: false,
            top_logprobs: 0,
        }
    }
}

impl SamplingParams {
    #[inline]
    pub fn is_stochastic(self) -> bool {
        self.temperature > 0.0
    }

    #[inline]
    pub fn has_penalty(self) -> bool {
        self.presence_penalty != 0.0 || self.frequency_penalty != 0.0
    }

    #[inline]
    pub fn needs_sampler(self) -> bool {
        self.is_stochastic() || self.has_penalty()
    }

    /// Deterministic per-step 64-bit RNG seed for generation index `step_idx` (`0, 1, 2, ...`).
    #[inline]
    pub fn step_seed(self, step_idx: u64) -> u64 {
        splitmix64(
            self.seed
                .wrapping_add(step_idx.wrapping_mul(0x9E37_79B9_7F4A_7C15)),
        )
    }
}

/// Default hard starvation cap (ms) before an aged waiting request preempts SRPT ordering.
pub const DEFAULT_PREFILL_STARVATION_CAP_MS: u64 = 60_000;
/// Default remaining prefill backlog (tokens) at or below which mixed steps stay at `mixed_cap`.
pub const DEFAULT_BACKLOG_LOW_TOKENS: usize = 2_048;
/// Default remaining prefill backlog (tokens) at or above which mixed steps scale to `max_step_tokens`.
pub const DEFAULT_BACKLOG_HIGH_TOKENS: usize = 16_384;

/// Prefill scheduling policy for mixed-length and APC-heavy workloads (Issue #40 Proposal 4).
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct PrefillSchedulingPolicy {
    /// Order waiting/partial-prefill requests by Shortest Remaining Processing Time (SRPT)
    /// over effective uncached prompt tokens (`prompt_len - cached_prefix_tokens`).
    /// Equal uncached lengths tie-break by `(submit_time, request_id)` (exact FIFO).
    pub srpt_enabled: bool,
    /// Hard starvation cap in milliseconds. Any waiting request whose age since `submit_time`
    /// reaches this cap preempts SRPT ordering (oldest starved request first) and holds back
    /// shorter admissions until KV pages drain enough to admit it. `0` disables starvation promotion.
    pub starvation_cap_ms: u64,
    /// Dynamically ramp mixed-step token budget from `mixed_cap` up to `max_step_tokens`
    /// as remaining prefill backlog grows from `backlog_low_tokens` to `backlog_high_tokens`.
    pub dynamic_budget_enabled: bool,
    /// Remaining prefill backlog (tokens) at or below which mixed steps stay at `mixed_cap`.
    pub backlog_low_tokens: usize,
    /// Remaining prefill backlog (tokens) at or above which mixed steps use `max_step_tokens`.
    pub backlog_high_tokens: usize,
}

impl Default for PrefillSchedulingPolicy {
    fn default() -> Self {
        Self {
            srpt_enabled: true,
            starvation_cap_ms: DEFAULT_PREFILL_STARVATION_CAP_MS,
            dynamic_budget_enabled: true,
            backlog_low_tokens: DEFAULT_BACKLOG_LOW_TOKENS,
            backlog_high_tokens: DEFAULT_BACKLOG_HIGH_TOKENS,
        }
    }
}

impl PrefillSchedulingPolicy {
    /// Strict FIFO waiting admission with fixed `mixed_cap` budget (legacy baseline behavior).
    pub const fn fifo() -> Self {
        Self {
            srpt_enabled: false,
            starvation_cap_ms: 0,
            dynamic_budget_enabled: false,
            backlog_low_tokens: DEFAULT_BACKLOG_LOW_TOKENS,
            backlog_high_tokens: DEFAULT_BACKLOG_HIGH_TOKENS,
        }
    }
}

struct PendingRequest {
    request_id: usize,
    prompt_tokens: Vec<i32>,
    target_output_len: usize,
    sampling: SamplingParams,
    permit_time: Instant,
    submit_time: Instant,
}

struct ActiveSequence {
    request_id: usize,
    slot_lease: SlotLease,
    /// Pinned prefix nodes reused or promoted into `PrefixTree` (`0..cached_pages`).
    prefix_lease: Option<PrefixLease>,
    /// Exclusively owned single-page `BlockLease`s for `cached_pages..page_ids.len()`.
    kv_leases: Vec<BlockLease>,
    page_ids: Vec<i32>,
    prompt_tokens: Vec<i32>,
    prefilled_tokens: usize,
    /// Prompt tokens whose GPU prefill step has completed (`<= prefilled_tokens`).
    computed_prompt_tokens: usize,
    /// Prefix tokens satisfied from `PrefixTree` at admission (`< prompt_tokens.len()`).
    cached_prefix_tokens: usize,
    /// Hybrid recurrent checkpoint slot ID to restore on the first prefill chunk (`-1` = none).
    restore_checkpoint_id: i32,
    /// Pending `(end_tokens, recurrent_slot_lease)` checkpoints staged for the in-flight prefill.
    pending_checkpoints: Vec<(usize, BlockLease)>,
    /// Stops further checkpoint slot allocation for this sequence once a boundary misses.
    checkpoint_alloc_stopped: bool,
    current_seq_len: usize,
    last_token: i32,
    generated_tokens: Vec<i32>,
    generated_top2: Vec<StepTop2Logits>,
    pending_draft_tokens: Vec<i32>,
    staged_verify: Option<(usize, usize)>,
    target_output_len: usize,
    sampling: SamplingParams,
    /// KV pages promised at admission for the whole request; only part may be allocated yet.
    reserved_pages: usize,
    permit_time: Instant,
    submit_time: Instant,
    first_token_time: Option<Instant>,
    last_token_time: Option<Instant>,
}

fn allocate_kv_pages_with_eviction(
    delta: usize,
    kv_pool: &BlockPool,
    prefix_tree: Option<&mut PrefixTree>,
    evictions_pages: &mut u64,
) -> Option<BlockLease> {
    if kv_pool.available() < delta
        && let Some(tree) = prefix_tree
    {
        let shortfall = delta - kv_pool.available();
        let nodes_needed = shortfall.div_ceil(tree.pages_per_node());
        let evicted_nodes = tree.evict_lru(nodes_needed);
        *evictions_pages += (evicted_nodes * tree.pages_per_node()) as u64;
    }
    kv_pool.allocate(delta)
}

impl ActiveSequence {
    #[cfg(test)]
    fn ensure_kv_pages(
        &mut self,
        target_seq_len: usize,
        page_block_tokens: usize,
        max_pages_per_seq: usize,
        kv_pool: &BlockPool,
        shadow_page_table: &mut [i32],
        dirty_pt_slots: &mut [u128; 2],
    ) -> Result<()> {
        let mut ignored_evictions = 0u64;
        self.ensure_kv_pages_with_cache(
            target_seq_len,
            page_block_tokens,
            max_pages_per_seq,
            kv_pool,
            None,
            &mut ignored_evictions,
            shadow_page_table,
            dirty_pt_slots,
        )
    }

    #[allow(clippy::too_many_arguments)]
    fn ensure_kv_pages_with_cache(
        &mut self,
        target_seq_len: usize,
        page_block_tokens: usize,
        max_pages_per_seq: usize,
        kv_pool: &BlockPool,
        prefix_tree: Option<&mut PrefixTree>,
        evictions_pages: &mut u64,
        shadow_page_table: &mut [i32],
        dirty_pt_slots: &mut [u128; 2],
    ) -> Result<()> {
        let needed_pages = target_seq_len.div_ceil(page_block_tokens);
        if needed_pages > max_pages_per_seq {
            return Err(EngineError::Protocol(format!(
                "sequence length {target_seq_len} exceeds max_pages_per_seq {max_pages_per_seq}"
            )));
        }
        if needed_pages > self.page_ids.len() {
            let delta = needed_pages - self.page_ids.len();
            let lease =
                allocate_kv_pages_with_eviction(delta, kv_pool, prefix_tree, evictions_pages)
                    .ok_or_else(|| {
                        EngineError::Protocol(format!(
                            "KV BlockPool exhausted allocating {delta} pages (free={})",
                            kv_pool.available()
                        ))
                    })?;
            let slot_id = self.slot_lease.id() as usize;
            let slot_row_base = slot_id * max_pages_per_seq;
            for &pid in lease.ids() {
                let page_col = self.page_ids.len();
                self.page_ids.push(pid);
                shadow_page_table[slot_row_base + page_col] = pid;
            }
            let dirty_bit = 1u128 << slot_id;
            dirty_pt_slots[0] |= dirty_bit;
            dirty_pt_slots[1] |= dirty_bit;
            self.kv_leases.extend(lease.split_individual());
        }
        Ok(())
    }

    /// Chronological 3-token accepted history immediately preceding `start_pos`,
    /// padded with `-1` (`DEAD_ID`) before sequence start (`pos < 0`).
    fn lookback_window(&self, start_pos: usize) -> [i32; 3] {
        let prompt_len = self.prompt_tokens.len();
        [3usize, 2, 1].map(|back| {
            let Some(idx) = start_pos.checked_sub(back) else {
                return -1;
            };
            if idx < prompt_len {
                self.prompt_tokens[idx]
            } else {
                self.generated_tokens
                    .get(idx - prompt_len)
                    .copied()
                    .unwrap_or(-1)
            }
        })
    }

    /// Token id and Engram window for a decode step that was planned while the
    /// previous step was still in flight.
    ///
    /// `stage_decode_step(..., commit_tokens_immediately = false)` freezes
    /// positions from `current_seq_len` before `generated_tokens` contains the
    /// token that step produced. Call this only after that token has been
    /// appended, and while `current_seq_len` is still the staged position.
    fn revise_overlapped_decode(&mut self) -> (i32, [i32; 3]) {
        let token = self.last_token;
        let lookback = self.lookback_window(self.current_seq_len);
        self.current_seq_len += 1;
        (token, lookback)
    }

    fn refresh_draft_tokens(&mut self, neural_drafts: &[i32], config: &SpeculativeConfig) {
        self.pending_draft_tokens.clear();
        if self.sampling.needs_sampler() {
            return;
        }
        match config.method {
            SpeculativeMethod::None => {}
            SpeculativeMethod::Mtp | SpeculativeMethod::Dspark => {
                self.pending_draft_tokens.extend(
                    neural_drafts
                        .iter()
                        .copied()
                        .filter(|&token| token >= 0)
                        .take(config.num_speculative_tokens),
                );
            }
            SpeculativeMethod::Ngram => {
                let history: Vec<i32> = self
                    .prompt_tokens
                    .iter()
                    .chain(&self.generated_tokens)
                    .copied()
                    .collect();
                self.pending_draft_tokens = ngram_draft_tokens(&history, config);
            }
        }
    }
}

fn rewind_rejected_suffix_pages(
    seq: &mut ActiveSequence,
    verified_len: usize,
    page_block_tokens: usize,
    max_pages_per_seq: usize,
    dummy_page: i32,
    shadow_page_table: &mut [i32],
    dirty_pt_slots: &mut [u128; 2],
) {
    let needed_pages = verified_len.div_ceil(page_block_tokens);
    if needed_pages >= seq.page_ids.len() {
        return;
    }
    let slot = seq.slot_lease.id() as usize;
    let row_base = slot * max_pages_per_seq;
    shadow_page_table[row_base + needed_pages..row_base + seq.page_ids.len()].fill(dummy_page);
    seq.page_ids.truncate(needed_pages);
    let prefix_pages = seq
        .prefix_lease
        .as_ref()
        .map_or(0, |lease| lease.block_ids().len());
    seq.kv_leases
        .truncate(needed_pages.saturating_sub(prefix_pages));
    let dirty_bit = 1u128 << slot;
    dirty_pt_slots[0] |= dirty_bit;
    dirty_pt_slots[1] |= dirty_bit;
}

fn ngram_draft_tokens(history: &[i32], config: &SpeculativeConfig) -> Vec<i32> {
    if history.len() < 2 || config.num_speculative_tokens == 0 {
        return Vec::new();
    }
    let max_n = config.prompt_lookup_max.min(history.len() - 1);
    for n in (config.prompt_lookup_min..=max_n).rev() {
        let suffix = &history[history.len() - n..];
        for start in (0..=history.len() - n - 1).rev() {
            if &history[start..start + n] == suffix {
                let draft_start = start + n;
                let draft_end = (draft_start + config.num_speculative_tokens).min(history.len());
                return history[draft_start..draft_end].to_vec();
            }
        }
    }
    Vec::new()
}

#[derive(Debug, Clone, Default)]
pub struct GenerationScheduleStats {
    pub prefill_steps: u64,
    pub decode_steps: u64,
    pub chunked_prefill_steps: u64,
    pub decode_steps_with_partial_prefill: u64,
    pub max_active_sequences: usize,
}

/// Live continuous-batching engine keeping the GPU workers warm across runs.
pub struct ServingEngine {
    coordinator: WorkerCoordinator,
    slot_pool: SlotPool,
    kv_pool: BlockPool,
    prefix_tree: Option<PrefixTree>,
    dram_cache: Option<DramPrefixCache>,
    recurrent_pool: Option<BlockPool>,
    pages_per_prefix_node: usize,
    prefix_cache_queries_tokens: u64,
    prefix_cache_hits_tokens: u64,
    prefix_cache_evictions_pages: u64,
    shadow_page_table: Vec<i32>,
    dirty_pt_slots: [u128; 2],
    step_cursor: u64,
    init_wall_us: u64,
    worker_numa_nodes: Vec<i32>,
    max_mixed_step_tokens: Option<usize>,
    max_prefill_coalesce_wait_ms: u64,
    prefill_chunk_align: usize,
    prefill_policy: PrefillSchedulingPolicy,
    device_token_overlap: bool,
    speculative: SpeculativeConfig,
    poisoned: bool,
    last_generation_schedule: GenerationScheduleStats,
}

enum MixedStepOutcome {
    MixedOrPrefill { had_decodes: bool, had_chunk: bool },
    PureDecodeFallback,
}

fn validate_serving_max_seqs(max_seqs: u32) -> Result<()> {
    if max_seqs == 0 || max_seqs > MAX_DECODE_GRAPH_SEQS {
        return Err(EngineError::Protocol(format!(
            "max_seqs={max_seqs} is outside supported decode CUDA Graph range 1..={MAX_DECODE_GRAPH_SEQS}"
        )));
    }
    Ok(())
}

/// Whether every active sequence is prompt-complete and still needs a decode step after the
/// step being planned, given `in_flight` earlier steps whose tokens are not harvested yet.
fn decode_continues_for_all(
    active: &[ActiveSequence],
    waiting_admissible: bool,
    in_flight: usize,
    has_prefix_cache: bool,
) -> bool {
    !waiting_admissible
        && !active.is_empty()
        && active.iter().all(|seq| {
            seq.prefilled_tokens == seq.prompt_tokens.len()
                && (!has_prefix_cache || seq.computed_prompt_tokens == seq.prompt_tokens.len())
                && seq.generated_tokens.len() + in_flight + 1 < seq.target_output_len
        })
}

/// Peek prefix-cache reuse and reservation headroom without mutating LRU ticks or pins.
///
/// Returns `(demand_pages, cached_prefix_tokens, extra_demand_pages, reclaimable_after_pin)`.
fn peek_request_admission(
    prompt_tokens: &[i32],
    target_output_len: usize,
    page_block_tokens: usize,
    kv_pool: &BlockPool,
    prefix_tree: Option<&PrefixTree>,
) -> (usize, usize, usize, usize) {
    let demand_pages =
        request_kv_page_demand(prompt_tokens.len(), target_output_len, page_block_tokens);
    if let Some(tree) = prefix_tree {
        let node_tokens = tree.block_tokens() as usize;
        let max_reusable_nodes = prompt_tokens.len().saturating_sub(1) / node_tokens;
        let (matched_nodes, unpinned_matched_nodes) =
            tree.peek_prefix_bounded(prompt_tokens, max_reusable_nodes);
        let pages_per_node = tree.pages_per_node();
        let matched_pages = matched_nodes * pages_per_node;
        let newly_pinned_pages = unpinned_matched_nodes * pages_per_node;
        let reclaimable_after_pin =
            kv_pool.available() + tree.evictable_kv_pages().saturating_sub(newly_pinned_pages);
        let extra_demand = demand_pages.saturating_sub(matched_pages);
        (
            demand_pages,
            matched_nodes * node_tokens,
            extra_demand,
            reclaimable_after_pin,
        )
    } else {
        (demand_pages, 0, demand_pages, kv_pool.available())
    }
}

/// Admit `req` into `ActiveSequence`, pinning any matched prefix nodes in `prefix_tree`.
#[allow(clippy::too_many_arguments)]
fn admit_request_into_sequence(
    req: PendingRequest,
    slot_lease: SlotLease,
    demand_pages: usize,
    max_pages_per_seq: usize,
    prefix_tree: Option<&mut PrefixTree>,
    has_recurrent_pool: bool,
    queries_tokens: &mut u64,
    hits_tokens: &mut u64,
    shadow_page_table: &mut [i32],
    dirty_pt_slots: &mut [u128; 2],
) -> Result<ActiveSequence> {
    let slot_id = slot_lease.id() as usize;
    let last_token = *req.prompt_tokens.last().unwrap_or(&0);
    let mut prefix_lease = None;
    let mut page_ids = Vec::new();
    let mut cached_tokens = 0usize;
    let mut restore_checkpoint_id = -1i32;

    if let Some(tree) = prefix_tree {
        *queries_tokens += req.prompt_tokens.len() as u64;
        let node_tokens = tree.block_tokens() as usize;
        let max_reusable_nodes = req.prompt_tokens.len().saturating_sub(1) / node_tokens;
        let lease = tree.match_prefix_bounded(&req.prompt_tokens, max_reusable_nodes);
        cached_tokens = lease.matched_tokens() as usize;
        *hits_tokens += cached_tokens as u64;
        if cached_tokens > 0 {
            if lease.block_ids().len() > max_pages_per_seq {
                return Err(EngineError::Protocol(format!(
                    "matched prefix pages {} exceed max_pages_per_seq {max_pages_per_seq}",
                    lease.block_ids().len()
                )));
            }
            if has_recurrent_pool {
                let anchor_block_id = lease.leaf_anchor_block_id().ok_or_else(|| {
                    EngineError::Protocol(
                        "hybrid prefix node missing canonical anchor KV Block-ID".to_owned(),
                    )
                })?;
                let aliased_slot = tree.resolve_aux_for_kv_block(anchor_block_id).ok_or_else(|| {
                    EngineError::Protocol(format!(
                        "hybrid prefix anchor block {anchor_block_id} missing unified recurrent alias"
                    ))
                })?;
                if lease.leaf_aux_id() != Some(aliased_slot) {
                    return Err(EngineError::Protocol(format!(
                        "hybrid prefix alias mismatch: anchor block {anchor_block_id} mapped to slot {aliased_slot}, leaf_aux_id={:?}",
                        lease.leaf_aux_id()
                    )));
                }
                restore_checkpoint_id = aliased_slot;
            }
            page_ids.extend_from_slice(lease.block_ids());
            let row_base = slot_id * max_pages_per_seq;
            shadow_page_table[row_base..row_base + page_ids.len()].copy_from_slice(&page_ids);
            let dirty_bit = 1u128 << slot_id;
            dirty_pt_slots[0] |= dirty_bit;
            dirty_pt_slots[1] |= dirty_bit;
        }
        prefix_lease = Some(lease);
    }

    Ok(ActiveSequence {
        request_id: req.request_id,
        slot_lease,
        prefix_lease,
        kv_leases: Vec::new(),
        page_ids,
        prompt_tokens: req.prompt_tokens,
        prefilled_tokens: cached_tokens,
        computed_prompt_tokens: cached_tokens,
        cached_prefix_tokens: cached_tokens,
        restore_checkpoint_id,
        pending_checkpoints: Vec::new(),
        checkpoint_alloc_stopped: false,
        current_seq_len: cached_tokens,
        last_token,
        generated_tokens: Vec::with_capacity(req.target_output_len),
        generated_top2: Vec::new(),
        pending_draft_tokens: Vec::new(),
        staged_verify: None,
        target_output_len: req.target_output_len,
        sampling: req.sampling,
        reserved_pages: demand_pages,
        permit_time: req.permit_time,
        submit_time: req.submit_time,
        first_token_time: None,
        last_token_time: None,
    })
}

/// Promote newly completed full prompt prefix nodes of `active` into `prefix_tree`.
#[allow(clippy::too_many_arguments)]
fn promote_completed_prefixes_in_batch(
    active: &mut [ActiveSequence],
    mut prefix_tree: Option<&mut PrefixTree>,
    has_recurrent_pool: bool,
    hold_incomplete_tail_token: bool,
    pages_per_node: usize,
    max_pages_per_seq: usize,
    shadow_page_table: &mut [i32],
    dirty_pt_slots: &mut [u128; 2],
) -> Result<()> {
    for seq in active.iter_mut() {
        if seq.computed_prompt_tokens >= seq.prefilled_tokens {
            continue;
        }
        seq.computed_prompt_tokens = seq.prefilled_tokens;
        let Some(tree) = prefix_tree.as_deref_mut() else {
            continue;
        };
        let node_tokens = tree.block_tokens() as usize;
        let already_promoted_nodes = seq
            .prefix_lease
            .as_ref()
            .map_or(0, PrefixLease::matched_nodes);
        let promotable_prompt_tokens =
            if hold_incomplete_tail_token && seq.computed_prompt_tokens < seq.prompt_tokens.len() {
                seq.computed_prompt_tokens.saturating_sub(1)
            } else {
                seq.computed_prompt_tokens
            };
        let total_completed_nodes = promotable_prompt_tokens / node_tokens;
        if total_completed_nodes <= already_promoted_nodes {
            continue;
        }

        let mut aux_leases = Vec::new();
        let promotable_nodes = if has_recurrent_pool {
            let mut count = 0usize;
            for node_idx in already_promoted_nodes..total_completed_nodes {
                let expected_end = (node_idx + 1) * node_tokens;
                if seq
                    .pending_checkpoints
                    .first()
                    .is_some_and(|(end, _)| *end == expected_end)
                {
                    let (_, ckpt_lease) = seq.pending_checkpoints.remove(0);
                    aux_leases.push(ckpt_lease);
                    count += 1;
                } else {
                    break;
                }
            }
            count
        } else {
            total_completed_nodes - already_promoted_nodes
        };
        if promotable_nodes == 0 {
            continue;
        }

        let promote_pages = promotable_nodes * pages_per_node;
        if seq.kv_leases.len() < promote_pages {
            return Err(EngineError::Protocol(format!(
                "sequence {} holds {} exclusive KV leases, fewer than {} promotable prefix pages",
                seq.request_id,
                seq.kv_leases.len(),
                promote_pages
            )));
        }
        let drained: Vec<BlockLease> = seq.kv_leases.drain(..promote_pages).collect();
        let mut node_kv_leases = Vec::with_capacity(promotable_nodes);
        let mut iter = drained.into_iter();
        for _ in 0..promotable_nodes {
            let chunk: Vec<BlockLease> = iter.by_ref().take(pages_per_node).collect();
            let merged = BlockLease::merge(chunk).ok_or_else(|| {
                EngineError::Protocol("failed to merge per-node KV BlockLeases".to_owned())
            })?;
            node_kv_leases.push(merged);
        }

        let start_tokens = already_promoted_nodes * node_tokens;
        let target_tokens = (already_promoted_nodes + promotable_nodes) * node_tokens;
        let existing_lease = seq
            .prefix_lease
            .take()
            .unwrap_or_else(|| tree.match_prefix_bounded(&[], 0));
        let new_aux = if has_recurrent_pool {
            Some(aux_leases)
        } else {
            None
        };
        let new_prefix_lease = tree
            .extend_prefix_with_aux(
                existing_lease,
                &seq.prompt_tokens[start_tokens..target_tokens],
                node_kv_leases,
                new_aux,
            )
            .map_err(|err| EngineError::Protocol(err.to_string()))?;
        let canonical_pages = new_prefix_lease.block_ids();
        let slot_id = seq.slot_lease.id() as usize;
        let row_base = slot_id * max_pages_per_seq;
        if seq.page_ids[..canonical_pages.len()] != *canonical_pages {
            seq.page_ids[..canonical_pages.len()].copy_from_slice(canonical_pages);
            shadow_page_table[row_base..row_base + canonical_pages.len()]
                .copy_from_slice(canonical_pages);
            let dirty_bit = 1u128 << slot_id;
            dirty_pt_slots[0] |= dirty_bit;
            dirty_pt_slots[1] |= dirty_bit;
        }
        seq.prefix_lease = Some(new_prefix_lease);
    }
    Ok(())
}

/// KV pages a request occupies at its longest. The last sampled token is never fed back,
/// so its KV row is never written.
pub fn request_kv_page_demand(
    prompt_len: usize,
    target_output_len: usize,
    page_block_tokens: usize,
) -> usize {
    prompt_len
        .saturating_add(target_output_len.saturating_sub(1))
        .div_ceil(page_block_tokens.max(1))
}

/// Pages promised to active sequences but not allocated yet.
fn outstanding_reserved_pages(active: &[ActiveSequence]) -> usize {
    active
        .iter()
        .map(|seq| seq.reserved_pages.saturating_sub(seq.page_ids.len()))
        .sum()
}

/// Admit only when the whole request fits next to what earlier admissions were promised,
/// so a running sequence can never run out of KV pages mid-decode.
fn kv_demand_fits(kv_available: usize, outstanding_reserved: usize, demand_pages: usize) -> bool {
    kv_available.saturating_sub(outstanding_reserved) >= demand_pages
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
struct WaitingSelection {
    index: usize,
    demand_pages: usize,
    cached_tokens: usize,
}

#[inline]
fn is_starved(submit_time: Instant, now: Instant, starvation_cap_ms: u64) -> bool {
    starvation_cap_ms > 0
        && now.saturating_duration_since(submit_time).as_millis() >= u128::from(starvation_cap_ms)
}

/// Compute the effective mixed-step token budget under [`PrefillSchedulingPolicy`].
///
/// Scales linearly from `base_budget` (at `backlog_tokens <= policy.backlog_low_tokens`)
/// up to `max_step_tokens` (at `backlog_tokens >= policy.backlog_high_tokens`).
pub(crate) fn compute_dynamic_mixed_step_budget(
    base_budget: usize,
    max_step_tokens: usize,
    backlog_tokens: usize,
    policy: PrefillSchedulingPolicy,
) -> usize {
    let base = base_budget.min(max_step_tokens);
    if !policy.dynamic_budget_enabled || base >= max_step_tokens {
        return base;
    }
    let low = policy.backlog_low_tokens;
    let high = policy.backlog_high_tokens.max(low);
    if backlog_tokens <= low {
        return base;
    }
    if backlog_tokens >= high || high == low {
        return max_step_tokens;
    }
    let span = high - low;
    let excess = backlog_tokens - low;
    let extra = (max_step_tokens - base).saturating_mul(excess) / span;
    base.saturating_add(extra).min(max_step_tokens)
}

fn estimate_prefill_backlog_tokens(
    active: &[ActiveSequence],
    waiting_permitted: &VecDeque<PendingRequest>,
    page_block_tokens: usize,
    kv_pool: &BlockPool,
    prefix_tree: Option<&PrefixTree>,
    stop_at: usize,
) -> usize {
    let mut backlog = 0usize;
    for seq in active {
        backlog =
            backlog.saturating_add(seq.prompt_tokens.len().saturating_sub(seq.prefilled_tokens));
        if backlog >= stop_at {
            return backlog;
        }
    }
    for req in waiting_permitted {
        let (_, cached_tokens, _, _) = peek_request_admission(
            &req.prompt_tokens,
            req.target_output_len,
            page_block_tokens,
            kv_pool,
            prefix_tree,
        );
        backlog = backlog.saturating_add(req.prompt_tokens.len().saturating_sub(cached_tokens));
        if backlog >= stop_at {
            return backlog;
        }
    }
    backlog
}

fn ordered_partial_prefill_indices(
    active: &[ActiveSequence],
    policy: PrefillSchedulingPolicy,
    now: Instant,
) -> Vec<usize> {
    let mut indices: Vec<usize> = active
        .iter()
        .enumerate()
        .filter_map(|(idx, seq)| (seq.prefilled_tokens < seq.prompt_tokens.len()).then_some(idx))
        .collect();
    if policy.srpt_enabled && indices.len() > 1 {
        indices.sort_by_key(|&idx| {
            let seq = &active[idx];
            let starved = is_starved(seq.submit_time, now, policy.starvation_cap_ms);
            let remaining = seq.prompt_tokens.len().saturating_sub(seq.prefilled_tokens);
            // Starved requests come first (oldest submit_time first); otherwise SRPT.
            (
                !starved,
                if starved { 0 } else { remaining },
                seq.submit_time,
                seq.request_id,
            )
        });
    }
    indices
}

#[allow(clippy::too_many_arguments)]
fn select_next_waiting_candidate(
    active: &[ActiveSequence],
    waiting_permitted: &VecDeque<PendingRequest>,
    slot_pool: &SlotPool,
    kv_pool: &BlockPool,
    prefix_tree: Option<&PrefixTree>,
    max_seqs: usize,
    page_block_tokens: usize,
    loading_area_reserve: usize,
    chunk_align: usize,
    policy: PrefillSchedulingPolicy,
    now: Instant,
    // Optional `(remaining_budget, max_prefill_chunk, num_seqs_already_staged)` filter.
    budget_filter: Option<(usize, usize, usize)>,
) -> Option<WaitingSelection> {
    if waiting_permitted.is_empty() || active.len() >= max_seqs || slot_pool.available() == 0 {
        return None;
    }
    // Reserve the 10% HBM Loading Area exclusively for DRAM -> HBM reloads so regular new
    // sequence admissions cannot exhaust it and trigger a reload/eviction deadlock.
    let outstanding = outstanding_reserved_pages(active).saturating_add(loading_area_reserve);

    let budget_accepts = |prompt_len: usize, cached_tokens: usize| -> bool {
        let Some((remaining_budget, max_prefill_chunk, num_seqs_staged)) = budget_filter else {
            return true;
        };
        let uncached_prompt_len = prompt_len.saturating_sub(cached_tokens);
        if num_seqs_staged > 0 && remaining_budget < 16 && uncached_prompt_len > remaining_budget {
            return false;
        }
        prefill_chunk_range_aligned(
            prompt_len,
            cached_tokens,
            remaining_budget,
            max_prefill_chunk,
            chunk_align,
        )
        .is_some()
    };

    if !policy.srpt_enabled {
        let front = waiting_permitted.front()?;
        let (demand_pages, cached_tokens, extra_demand, reclaimable_after_pin) =
            peek_request_admission(
                &front.prompt_tokens,
                front.target_output_len,
                page_block_tokens,
                kv_pool,
                prefix_tree,
            );
        if !kv_demand_fits(reclaimable_after_pin, outstanding, extra_demand)
            || !budget_accepts(front.prompt_tokens.len(), cached_tokens)
        {
            return None;
        }
        return Some(WaitingSelection {
            index: 0,
            demand_pages,
            cached_tokens,
        });
    }

    // Starvation barrier: if any waiting request has waited >= `starvation_cap_ms`, the oldest
    // starved request gets exclusive admission priority. If it cannot fit in KV yet, hold
    // admission so active sequences drain and free pages for it instead of starving it forever.
    if policy.starvation_cap_ms > 0 {
        let oldest_starved = waiting_permitted
            .iter()
            .enumerate()
            .filter(|(_, req)| is_starved(req.submit_time, now, policy.starvation_cap_ms))
            .min_by_key(|&(idx, req)| (req.submit_time, req.request_id, idx));
        if let Some((idx, req)) = oldest_starved {
            let (demand_pages, cached_tokens, extra_demand, reclaimable_after_pin) =
                peek_request_admission(
                    &req.prompt_tokens,
                    req.target_output_len,
                    page_block_tokens,
                    kv_pool,
                    prefix_tree,
                );
            if kv_demand_fits(reclaimable_after_pin, outstanding, extra_demand)
                && budget_accepts(req.prompt_tokens.len(), cached_tokens)
            {
                return Some(WaitingSelection {
                    index: idx,
                    demand_pages,
                    cached_tokens,
                });
            }
            return None;
        }
    }

    // SRPT: pick the admissible waiting request with the fewest remaining uncached prompt tokens,
    // tie-breaking by `(submit_time, request_id)` (exact FIFO for equal effective lengths).
    let mut best: Option<((usize, Instant, usize), WaitingSelection)> = None;
    for (idx, req) in waiting_permitted.iter().enumerate() {
        let (demand_pages, cached_tokens, extra_demand, reclaimable_after_pin) =
            peek_request_admission(
                &req.prompt_tokens,
                req.target_output_len,
                page_block_tokens,
                kv_pool,
                prefix_tree,
            );
        if !kv_demand_fits(reclaimable_after_pin, outstanding, extra_demand)
            || !budget_accepts(req.prompt_tokens.len(), cached_tokens)
        {
            continue;
        }
        let uncached_tokens = req.prompt_tokens.len().saturating_sub(cached_tokens);
        let key = (uncached_tokens, req.submit_time, req.request_id);
        if best.is_none_or(|(best_key, _)| key < best_key) {
            best = Some((
                key,
                WaitingSelection {
                    index: idx,
                    demand_pages,
                    cached_tokens,
                },
            ));
        }
    }
    best.map(|(_, sel)| sel)
}

#[cfg(test)]
fn front_request_admissible_in_pools(
    active: &[ActiveSequence],
    waiting_permitted: &VecDeque<PendingRequest>,
    slot_pool: &SlotPool,
    kv_pool: &BlockPool,
    prefix_tree: Option<&PrefixTree>,
    max_seqs: usize,
    page_block_tokens: usize,
) -> bool {
    select_next_waiting_candidate(
        active,
        waiting_permitted,
        slot_pool,
        kv_pool,
        prefix_tree,
        max_seqs,
        page_block_tokens,
        0,
        1,
        PrefillSchedulingPolicy::fifo(),
        Instant::now(),
        None,
    )
    .is_some()
}

fn validate_request_kv_demand_with_limits(
    requests: impl Iterator<Item = (usize, usize)>,
    page_block_tokens: usize,
    max_pages_per_seq: usize,
    kv_capacity: usize,
) -> Result<()> {
    for (prompt_len, target_output_len) in requests {
        if prompt_len == 0 {
            return Err(EngineError::Protocol(
                "prompt length must be positive".to_owned(),
            ));
        }
        let Some(max_kv_tokens) = prompt_len.checked_add(target_output_len.saturating_sub(1))
        else {
            return Err(EngineError::Protocol(format!(
                "request with prompt_len={prompt_len} and output_len={target_output_len} overflows KV token length"
            )));
        };
        let demand = max_kv_tokens.div_ceil(page_block_tokens);
        if demand > max_pages_per_seq || demand > kv_capacity {
            return Err(EngineError::Protocol(format!(
                "request with prompt_len={prompt_len} and output_len={target_output_len} needs {demand} KV pages, exceeding max_pages_per_seq={max_pages_per_seq} or num_kv_pages={kv_capacity}"
            )));
        }
    }
    Ok(())
}

fn sync_dirty_page_table_rows(
    plan: &mut zs_shm::StepPlanSlotMut<'_>,
    step_id: u64,
    max_pages_per_seq: usize,
    shadow_page_table: &[i32],
    dirty_pt_slots: &mut [u128; 2],
) -> Result<()> {
    let slot_idx = (step_id & 1) as usize;
    let mut mask = dirty_pt_slots[slot_idx];
    if mask == 0 {
        return Ok(());
    }
    let dst_table = plan.global_kv_page_table_mut()?;
    while mask != 0 {
        let row = mask.trailing_zeros() as usize;
        mask &= mask - 1;
        let start = row * max_pages_per_seq;
        let end = start + max_pages_per_seq;
        dst_table[start..end].copy_from_slice(&shadow_page_table[start..end]);
    }
    dirty_pt_slots[slot_idx] = 0;
    Ok(())
}

impl ServingEngine {
    pub fn launch(config: EngineConfig) -> Result<Self> {
        let geometry = config.geometry;
        validate_serving_max_seqs(geometry.max_seqs)?;
        let num_kv_pages = config.launch.num_kv_pages as usize;
        let dummy_page = config.launch.num_kv_pages as i32;
        let speculative = config.speculative.clone();
        let mut coordinator = WorkerCoordinator::launch(config)?;
        let (init_wall_us, worker_numa_nodes) = coordinator.wait_workers_ready()?;

        let pt_len = geometry.max_seqs as usize * geometry.max_pages_per_seq as usize;
        let shadow_page_table = vec![dummy_page; pt_len];
        for slot_step in [0u64, 1u64] {
            coordinator
                .plan_slot_mut(slot_step)?
                .global_kv_page_table_mut()?
                .fill(dummy_page);
        }
        Ok(Self {
            coordinator,
            slot_pool: SlotPool::new(geometry.max_seqs),
            kv_pool: BlockPool::new(CacheGroupKind::GlobalAndIndex, num_kv_pages),
            prefix_tree: None,
            dram_cache: None,
            recurrent_pool: None,
            pages_per_prefix_node: 1,
            prefix_cache_queries_tokens: 0,
            prefix_cache_hits_tokens: 0,
            prefix_cache_evictions_pages: 0,
            shadow_page_table,
            dirty_pt_slots: [0, 0],
            step_cursor: 0,
            init_wall_us,
            worker_numa_nodes,
            max_mixed_step_tokens: None,
            max_prefill_coalesce_wait_ms: DEFAULT_PREFILL_COALESCE_WAIT_MS,
            prefill_chunk_align: 4,
            prefill_policy: PrefillSchedulingPolicy::default(),
            device_token_overlap: false,
            speculative,
            poisoned: false,
            last_generation_schedule: GenerationScheduleStats::default(),
        })
    }

    /// Enable Automatic Prefix Caching (APC) across requests.
    ///
    /// - Pure paged attention (`DeepSeek-V4.1-Flash`): `pages_per_node = 1` (128 tok/node),
    ///   `num_recurrent_checkpoints = 0`.
    /// - Hybrid `GatedDeltaNet` (`Qwen3.5-35B-A3B`): `pages_per_node = 4` (64 tok/node),
    ///   `num_recurrent_checkpoints = 255` (slot IDs `0..=254` in SHM `replay_mask`).
    pub fn enable_prefix_caching(
        &mut self,
        pages_per_node: usize,
        num_recurrent_checkpoints: usize,
    ) -> Result<()> {
        if pages_per_node == 0 {
            return Err(EngineError::Protocol(
                "pages_per_node must be positive".to_owned(),
            ));
        }
        if num_recurrent_checkpoints > 255 {
            return Err(EngineError::Protocol(format!(
                "num_recurrent_checkpoints={num_recurrent_checkpoints} exceeds uint8 SHM limit 255"
            )));
        }
        let page_block_tokens = self.coordinator.config().geometry.page_block_tokens;
        let node_tokens = page_block_tokens
            .checked_mul(pages_per_node as u32)
            .ok_or_else(|| EngineError::Protocol("prefix node_tokens overflow".to_owned()))?;
        let tree = PrefixTree::with_pages_per_node(node_tokens, pages_per_node)
            .map_err(|err| EngineError::Protocol(err.to_string()))?;
        self.prefix_tree = Some(tree);
        self.pages_per_prefix_node = pages_per_node;
        self.recurrent_pool = (num_recurrent_checkpoints > 0)
            .then(|| BlockPool::new(CacheGroupKind::RecurrentState, num_recurrent_checkpoints));
        Ok(())
    }

    pub fn has_prefix_cache(&self) -> bool {
        self.prefix_tree.is_some()
    }

    /// Enable the Exclusive Host DRAM KV tier with Victim Chain demotion and a reserved
    /// HBM Loading Area (Issue #40 Proposals 1 & 2).
    pub fn enable_dram_kv_cache(&mut self, config: DramTierConfig) -> Result<()> {
        if !config.is_enabled() {
            self.dram_cache = None;
            return Ok(());
        }
        let Some(ref tree) = self.prefix_tree else {
            return Err(EngineError::Protocol(
                "enable_dram_kv_cache requires enable_prefix_caching to be called first".to_owned(),
            ));
        };
        if self.pages_per_prefix_node != 1 {
            return Err(EngineError::Protocol(format!(
                "Host DRAM KV tier currently requires pages_per_prefix_node == 1 (got {})",
                self.pages_per_prefix_node
            )));
        }
        let cache = DramPrefixCache::new(tree.block_tokens(), self.kv_pool.capacity(), config)
            .map_err(|err| EngineError::Protocol(err.to_string()))?;
        self.dram_cache = Some(cache);
        Ok(())
    }

    pub fn dram_tier_stats(&self) -> Option<DramTierStats> {
        self.dram_cache.as_ref().map(DramPrefixCache::stats)
    }

    /// Number of HBM KV pages reserved exclusively for DRAM -> HBM reloads (`0` when disabled).
    pub fn loading_area_pages(&self) -> usize {
        self.dram_cache
            .as_ref()
            .map_or(0, DramPrefixCache::loading_area_pages)
    }

    /// Publish and await one or more [`StepOpcode::KvTransfer`] micro-steps across all workers.
    fn execute_kv_transfer(&mut self, ops: &[KvTransferOp]) -> Result<()> {
        if ops.is_empty() {
            return Ok(());
        }
        let max_ops_per_step = (self.coordinator.config().geometry.max_step_tokens as usize).max(1);
        for chunk in ops.chunks(max_ops_per_step) {
            self.step_cursor += 1;
            let transfer_sid = self.step_cursor;
            {
                let mut plan = self.coordinator.plan_slot_mut(transfer_sid)?;
                {
                    let token_ids = plan.token_ids_mut()?;
                    for (i, op) in chunk.iter().enumerate() {
                        token_ids[i] = op.hbm_page_id;
                    }
                }
                {
                    let positions = plan.positions_mut()?;
                    for (i, op) in chunk.iter().enumerate() {
                        positions[i] = op.dram_slot_id;
                    }
                }
                {
                    let directions = plan.req_slot_ids_mut()?;
                    for (i, op) in chunk.iter().enumerate() {
                        directions[i] = if op.is_h2d {
                            KV_TRANSFER_DIR_H2D
                        } else {
                            KV_TRANSFER_DIR_D2H
                        };
                    }
                }
                *plan.header_mut()? = StepHeader {
                    step_id: AtomicU64::new(0),
                    opcode: StepOpcode::KvTransfer as u32,
                    num_seqs: chunk.len() as u32,
                    num_encoder_tokens: chunk.len() as u32,
                    encoder_graph_bucket: 0,
                    num_decoder_rows: chunk.len() as u32,
                    decoder_graph_bucket: 0,
                    num_sample_rows: 0,
                    is_dspark_verify: 0,
                    flags: 0,
                    _reserved: [0; 3],
                };
            }
            self.coordinator.publish_step(transfer_sid)?;
            self.coordinator.wait_step_completed(transfer_sid)?;
        }
        Ok(())
    }

    /// When HBM free pages are below `needed_pages` and Host DRAM tier is active, demote
    /// unpinned LRU HBM prefix leaves into Exclusive Host DRAM via D2H `KvTransfer` while
    /// keeping their HBM `BlockLease`s pinned until the transfer completes.
    fn prepare_hbm_for_pages(&mut self, needed_pages: usize) -> Result<()> {
        if needed_pages == 0 || self.kv_pool.available() >= needed_pages {
            return Ok(());
        }
        if self.dram_cache.is_none() || self.prefix_tree.is_none() {
            return Ok(());
        }
        let deficit = needed_pages.saturating_sub(self.kv_pool.available());
        let victims = self
            .prefix_tree
            .as_ref()
            .map(|tree| tree.evict_lru_victims(deficit))
            .unwrap_or_default();
        if victims.is_empty() {
            return Ok(());
        }
        let mut ops = Vec::with_capacity(victims.len());
        let mut admitted_paths = Vec::with_capacity(victims.len());
        if let Some(dram) = self.dram_cache.as_mut() {
            for victim in &victims {
                if let Some(&hbm_page_id) = victim.lease.ids().first()
                    && let Some(op) = dram.admit_victim(victim.token_path.clone(), hbm_page_id)
                {
                    ops.push(op);
                    admitted_paths.push(victim.token_path.clone());
                }
            }
        }
        if !ops.is_empty()
            && let Err(err) = self.execute_kv_transfer(&ops)
        {
            if let Some(dram) = self.dram_cache.as_mut() {
                for path in &admitted_paths {
                    dram.rollback_failed_offload(path);
                }
            }
            drop(victims);
            return Err(err);
        }
        self.prefix_cache_evictions_pages += victims.len() as u64;
        // Dropping `victims` returns the offloaded HBM `BlockLease`s back to `self.kv_pool`
        // strictly after the worker's D2H transfer has completed.
        drop(victims);
        Ok(())
    }

    /// Reload contiguous DRAM-resident prefix blocks immediately following the current HBM
    /// prefix match into HBM (drawing from the 10% Loading Area / free HBM pool) and promote
    /// them into `PrefixTree` before staging prefill.
    fn reload_dram_prefix_into_hbm(&mut self, prompt_tokens: &[i32]) -> Result<usize> {
        if self.dram_cache.is_none() || self.prefix_tree.is_none() || prompt_tokens.len() <= 1 {
            return Ok(0);
        }
        let hbm_matched_tokens = {
            let tree = self.prefix_tree.as_ref().expect("checked");
            let max_nodes = (prompt_tokens.len() - 1) / (tree.block_tokens() as usize);
            let (matched_nodes, _) = tree.peek_prefix_bounded(prompt_tokens, max_nodes);
            matched_nodes * (tree.block_tokens() as usize)
        };
        let avail_dram_blocks = self.dram_cache.as_ref().map_or(0, |d| {
            d.peek_continuation_blocks(prompt_tokens, hbm_matched_tokens)
        });
        if avail_dram_blocks == 0 {
            return Ok(0);
        }
        let max_reload = avail_dram_blocks.min(self.kv_pool.available());
        if max_reload == 0 {
            return Ok(0);
        }
        let mut hbm_leases = Vec::with_capacity(max_reload);
        for _ in 0..max_reload {
            if let Some(lease) = self.kv_pool.allocate(1) {
                hbm_leases.push(lease);
            } else {
                break;
            }
        }
        if hbm_leases.is_empty() {
            return Ok(0);
        }
        let taken = self
            .dram_cache
            .as_mut()
            .expect("checked")
            .take_continuation_for_reload(prompt_tokens, hbm_matched_tokens, hbm_leases.len());
        hbm_leases.truncate(taken.len());
        if taken.is_empty() {
            return Ok(0);
        }
        let dram_slots: Vec<i32> = taken.iter().map(|(_, slot)| *slot).collect();
        let h2d_ops: Vec<KvTransferOp> = hbm_leases
            .iter()
            .zip(taken.iter())
            .map(|(lease, (_, dram_slot_id))| KvTransferOp {
                hbm_page_id: lease.ids()[0],
                dram_slot_id: *dram_slot_id,
                is_h2d: true,
            })
            .collect();
        let transfer_res = self.execute_kv_transfer(&h2d_ops);
        if let Some(dram) = self.dram_cache.as_mut() {
            dram.release_dram_slots(&dram_slots);
        }
        transfer_res?;

        let reloaded_blocks = hbm_leases.len();
        if let Some(tree) = self.prefix_tree.as_ref() {
            let full_len = hbm_matched_tokens + reloaded_blocks * (tree.block_tokens() as usize);
            let base_lease = tree.match_prefix_bounded(
                prompt_tokens,
                hbm_matched_tokens / (tree.block_tokens() as usize),
            );
            let _promoted = tree
                .extend_prefix(
                    base_lease,
                    &prompt_tokens[hbm_matched_tokens..full_len],
                    hbm_leases,
                )
                .map_err(|err| EngineError::Protocol(err.to_string()))?;
        }
        Ok(reloaded_blocks)
    }

    /// Pre-step DRAM tier maintenance (executed strictly *before* allocating the next
    /// `NormalStep` `step_id` so `KvTransfer` steps stay strictly monotonic in SHM ring slots).
    fn prepare_dram_tier_for_step(
        &mut self,
        active: &[ActiveSequence],
        waiting_permitted: &VecDeque<PendingRequest>,
        max_seqs: usize,
        page_block_tokens: usize,
        now: Instant,
    ) -> Result<()> {
        if self.dram_cache.is_none() || self.prefix_tree.is_none() {
            return Ok(());
        }
        // 1. Demote HBM LRU victims to DRAM if active decode/partial-prefill sequences need new pages.
        let mut needed_hbm_pages = 0usize;
        for seq in active {
            if seq.prefilled_tokens == seq.prompt_tokens.len() {
                let next_pages = (seq.current_seq_len + 1).div_ceil(page_block_tokens);
                needed_hbm_pages += next_pages.saturating_sub(seq.page_ids.len());
            }
        }
        if needed_hbm_pages > 0 {
            self.prepare_hbm_for_pages(needed_hbm_pages)?;
        }
        // 2. If a waiting request is admissible, reload any contiguous DRAM prefix continuation
        //    into HBM (drawing from the 10% Loading Area) and demote victims if needed for its
        //    remaining uncached pages.
        if let Some(sel) = select_next_waiting_candidate(
            active,
            waiting_permitted,
            &self.slot_pool,
            &self.kv_pool,
            self.prefix_tree.as_ref(),
            max_seqs,
            page_block_tokens,
            self.loading_area_pages(),
            self.prefill_chunk_align,
            self.prefill_policy,
            now,
            None,
        ) {
            let prompt = waiting_permitted[sel.index].prompt_tokens.clone();
            let target_output_len = waiting_permitted[sel.index].target_output_len;
            let _ = self.reload_dram_prefix_into_hbm(&prompt)?;
            let (_, _, extra_demand, _) = peek_request_admission(
                &prompt,
                target_output_len,
                page_block_tokens,
                &self.kv_pool,
                self.prefix_tree.as_ref(),
            );
            let total_needed = needed_hbm_pages.saturating_add(extra_demand);
            if total_needed > 0 {
                self.prepare_hbm_for_pages(total_needed)?;
            }
        }
        Ok(())
    }

    /// Free physical KV pages plus unpinned `PrefixTree` pages reclaimable via LRU eviction.
    pub fn reclaimable_kv_pages(&self) -> usize {
        self.kv_pool.available()
            + self
                .prefix_tree
                .as_ref()
                .map_or(0, PrefixTree::evictable_kv_pages)
    }

    /// Cumulative `(queries_tokens, hits_tokens, evicted_kv_pages)` counters.
    pub fn prefix_cache_counters(&self) -> (u64, u64, u64) {
        (
            self.prefix_cache_queries_tokens,
            self.prefix_cache_hits_tokens,
            self.prefix_cache_evictions_pages,
        )
    }

    fn promote_completed_prefixes(
        &mut self,
        active: &mut [ActiveSequence],
        max_pages_per_seq: usize,
    ) -> Result<()> {
        promote_completed_prefixes_in_batch(
            active,
            self.prefix_tree.as_mut(),
            self.recurrent_pool.is_some(),
            self.speculative.method == SpeculativeMethod::Mtp,
            self.pages_per_prefix_node,
            max_pages_per_seq,
            &mut self.shadow_page_table,
            &mut self.dirty_pt_slots,
        )
    }

    pub fn worker_numa_nodes(&self) -> &[i32] {
        &self.worker_numa_nodes
    }

    pub fn init_wall_ms(&self) -> f64 {
        self.init_wall_us as f64 / 1000.0
    }

    pub fn last_generation_schedule(&self) -> &GenerationScheduleStats {
        &self.last_generation_schedule
    }

    pub fn set_max_prefill_chunk(&mut self, max_prefill_chunk: usize) -> Result<()> {
        if max_prefill_chunk == 0 {
            return Err(EngineError::Protocol(
                "max_prefill_chunk must be positive".to_owned(),
            ));
        }
        self.coordinator.config_mut().max_prefill_chunk = max_prefill_chunk;
        Ok(())
    }

    pub fn set_max_mixed_step_tokens(&mut self, max_mixed_step_tokens: Option<usize>) {
        self.max_mixed_step_tokens = max_mixed_step_tokens.filter(|&cap| cap > 0);
    }

    pub fn set_max_prefill_coalesce_wait_ms(&mut self, wait_ms: u64) {
        self.max_prefill_coalesce_wait_ms = wait_ms;
    }

    pub fn prefill_scheduling_policy(&self) -> PrefillSchedulingPolicy {
        self.prefill_policy
    }

    pub fn set_prefill_scheduling_policy(&mut self, policy: PrefillSchedulingPolicy) {
        self.prefill_policy = policy;
    }

    /// Align non-final chunked-prefill boundaries to `align` tokens (default `4` for
    /// DeepSeek-V4.1 `kpool` `compress_ratio == 4`).
    pub fn set_prefill_chunk_align(&mut self, align: usize) {
        self.prefill_chunk_align = align.max(1);
    }

    /// Attach a non-blocking [`zs_kv::KvCacheEventSink`] to the active `PrefixTree` (if APC is enabled).
    pub fn set_prefix_event_sink(&mut self, sink: Option<zs_kv::KvCacheEventSink>) {
        if let Some(tree) = self.prefix_tree.as_ref() {
            tree.set_event_sink(sink);
        }
    }

    /// Publish decode steps before the previous step's result is read.
    ///
    /// Workers must implement `launch`/`finish`, `STEP_FLAG_DEVICE_TOKENS`, and
    /// `STEP_FLAG_NEXT_PREPUBLISHED`, maintaining device-resident sampled tokens (and any
    /// per-slot Engram lookback window) across overlapped decode bursts.
    pub fn set_device_token_overlap(&mut self, enabled: bool) {
        self.device_token_overlap = enabled;
    }

    fn harvested_outputs(&mut self, step_id: u64) -> Result<Vec<HarvestedOutput>> {
        let result = self.coordinator.result_slot(step_id)?;
        if result.completed_step_id()? != step_id {
            return Err(EngineError::Protocol(format!(
                "completed_step_id mismatch on step {step_id}"
            )));
        }
        let header = result.header()?;
        if header.status_code != 0 {
            return Err(EngineError::Protocol(format!(
                "worker failed step {step_id} with status {}",
                header.status_code
            )));
        }
        result.outputs()?[..header.num_seq_results as usize]
            .iter()
            .map(|output| {
                let count = output.num_accepted as usize;
                if count == 0 || count > output.accepted_tokens.len() {
                    return Err(EngineError::Protocol(format!(
                        "invalid num_accepted={count} on step {step_id}"
                    )));
                }
                Ok(HarvestedOutput {
                    req_slot: output.req_slot,
                    accepted_tokens: output.accepted_tokens[..count].to_vec(),
                    accepted_logprobs_bits: output.accepted_logprobs_bits[..count].to_vec(),
                    next_draft_tokens: output
                        .next_draft_tokens
                        .iter()
                        .copied()
                        .filter(|&token| token >= 0)
                        .collect(),
                    runner_up_token: output.accepted_tokens[1],
                    runner_up_logprob_bits: output.accepted_logprobs_bits[1],
                })
            })
            .collect()
    }

    fn plan_flags(&mut self, step_id: u64) -> Result<u32> {
        Ok(self.coordinator.plan_slot_mut(step_id)?.header_mut()?.flags)
    }

    fn or_plan_flags(&mut self, step_id: u64, bits: u32) -> Result<()> {
        self.coordinator.plan_slot_mut(step_id)?.header_mut()?.flags |= bits;
        Ok(())
    }

    /// Whether any waiting request can be admitted now under [`Self::prefill_scheduling_policy`].
    fn front_request_admissible(
        &self,
        active: &[ActiveSequence],
        waiting_permitted: &VecDeque<PendingRequest>,
        max_seqs: usize,
        page_block_tokens: usize,
        now: Instant,
    ) -> bool {
        select_next_waiting_candidate(
            active,
            waiting_permitted,
            &self.slot_pool,
            &self.kv_pool,
            self.prefix_tree.as_ref(),
            max_seqs,
            page_block_tokens,
            self.loading_area_pages(),
            self.prefill_chunk_align,
            self.prefill_policy,
            now,
            None,
        )
        .is_some()
    }

    /// Reject requests that could never be admitted before any step is published, so the
    /// engine stays usable instead of failing mid-run.
    fn validate_request_kv_demand(
        &self,
        requests: impl Iterator<Item = (usize, usize)>,
        page_block_tokens: usize,
        max_pages_per_seq: usize,
    ) -> Result<()> {
        validate_request_kv_demand_with_limits(
            requests,
            page_block_tokens,
            max_pages_per_seq,
            self.kv_pool
                .capacity()
                .saturating_sub(self.loading_area_pages()),
        )
    }

    /// Generate exact greedy completions for `prompts` (`target_output_len` new tokens each).
    pub fn generate_greedy(
        &mut self,
        prompts: &[Vec<i32>],
        target_output_len: usize,
    ) -> Result<Vec<Vec<i32>>> {
        if target_output_len == 0 {
            return Ok(vec![Vec::new(); prompts.len()]);
        }
        self.generate_greedy_with_lengths(prompts, &vec![target_output_len; prompts.len()])
    }

    /// Generate exact greedy completions with an explicit output length for every prompt.
    pub fn generate_greedy_with_lengths(
        &mut self,
        prompts: &[Vec<i32>],
        target_output_lengths: &[usize],
    ) -> Result<Vec<Vec<i32>>> {
        self.generate_greedy_internal(prompts, target_output_lengths, false)
            .map(|(tokens, _)| tokens)
    }

    /// Generate exact greedy completions along with per-step global Top-2 logit diagnostics.
    pub fn generate_greedy_with_top2(
        &mut self,
        prompts: &[Vec<i32>],
        target_output_lengths: &[usize],
    ) -> Result<GreedyTop2Batch> {
        self.generate_greedy_internal(prompts, target_output_lengths, true)
    }

    fn generate_greedy_internal(
        &mut self,
        prompts: &[Vec<i32>],
        target_output_lengths: &[usize],
        record_top2: bool,
    ) -> Result<GreedyTop2Batch> {
        if prompts.len() != target_output_lengths.len() {
            return Err(EngineError::Protocol(format!(
                "prompt count {} differs from output length count {}",
                prompts.len(),
                target_output_lengths.len()
            )));
        }
        if target_output_lengths.contains(&0) {
            return Err(EngineError::Protocol(
                "greedy output lengths must be positive".to_owned(),
            ));
        }
        let mut completed = vec![Vec::new(); prompts.len()];
        let mut completed_top2 = if record_top2 {
            vec![Vec::new(); prompts.len()]
        } else {
            Vec::new()
        };
        if prompts.is_empty() {
            return Ok((completed, completed_top2));
        }
        self.last_generation_schedule = GenerationScheduleStats::default();
        let now = Instant::now();
        let mut waiting: VecDeque<PendingRequest> = prompts
            .iter()
            .zip(target_output_lengths)
            .enumerate()
            .map(|(idx, (prompt, &target_output_len))| PendingRequest {
                request_id: idx,
                prompt_tokens: prompt.clone(),
                target_output_len,
                sampling: SamplingParams::default(),
                permit_time: now,
                submit_time: now,
            })
            .collect();
        let mut raw_prompts = VecDeque::new();
        let mut ttfts = Vec::new();
        let mut submit_ttfts = Vec::new();
        let mut tpots = Vec::new();
        let mut itls = Vec::new();
        let mut steps = 0u64;
        let mut overlapped = 0u64;

        self.drive_requests(
            prompts.len(),
            &mut waiting,
            &mut raw_prompts,
            &mut completed,
            if record_top2 {
                Some(&mut completed_top2)
            } else {
                None
            },
            &mut ttfts,
            &mut submit_ttfts,
            &mut tpots,
            &mut itls,
            &mut steps,
            &mut overlapped,
        )?;
        Ok((completed, completed_top2))
    }

    /// Execute one continuous-batching benchmark tier with default seed `BENCHMARK_RANDOM_SEED`.
    pub fn run_benchmark_tier(&mut self, spec: &BenchmarkTierSpec) -> Result<BenchmarkTierResult> {
        self.run_benchmark_tier_with_seed(spec, BENCHMARK_RANDOM_SEED)
    }

    /// Execute one continuous-batching benchmark tier with semaphore `spec.concurrency` and explicit `seed`.
    pub fn run_benchmark_tier_with_seed(
        &mut self,
        spec: &BenchmarkTierSpec,
        seed: u64,
    ) -> Result<BenchmarkTierResult> {
        if spec.concurrency == 0
            || spec.num_prompts == 0
            || spec.input_len == 0
            || spec.output_len == 0
        {
            return Err(EngineError::Protocol(
                "benchmark tier dimensions must all be positive".to_owned(),
            ));
        }
        let all_prompts =
            build_deterministic_random_prompts_seeded(spec.num_prompts, spec.input_len, seed);
        let initial_admit = spec.concurrency.min(spec.num_prompts);

        let mut completed = vec![Vec::new(); spec.num_prompts];
        let mut ttfts_us = Vec::with_capacity(spec.num_prompts);
        let mut submit_ttfts_us = Vec::with_capacity(spec.num_prompts);
        let mut tpots_us = Vec::with_capacity(spec.num_prompts);
        let mut itls_us = Vec::with_capacity(spec.num_prompts * spec.output_len);
        let mut steps = 0u64;
        let mut overlapped_decode_steps = 0u64;

        let wall_start = Instant::now();
        let mut waiting_permitted: VecDeque<PendingRequest> =
            VecDeque::with_capacity(spec.concurrency);
        let mut unpermitted_pool = VecDeque::with_capacity(spec.num_prompts - initial_admit);

        for (req_id, prompt) in all_prompts.into_iter().enumerate() {
            if req_id < initial_admit {
                waiting_permitted.push_back(PendingRequest {
                    request_id: req_id,
                    prompt_tokens: prompt,
                    target_output_len: spec.output_len,
                    sampling: SamplingParams::default(),
                    permit_time: wall_start,
                    submit_time: wall_start,
                });
            } else {
                unpermitted_pool.push_back((req_id, prompt, spec.output_len, wall_start));
            }
        }

        self.drive_requests(
            spec.concurrency,
            &mut waiting_permitted,
            &mut unpermitted_pool,
            &mut completed,
            None,
            &mut ttfts_us,
            &mut submit_ttfts_us,
            &mut tpots_us,
            &mut itls_us,
            &mut steps,
            &mut overlapped_decode_steps,
        )?;
        let wall_elapsed = wall_start.elapsed();
        let wall_secs = wall_elapsed.as_secs_f64();
        let total_out_tokens = (spec.num_prompts * spec.output_len) as f64;
        let total_all_tokens = (spec.num_prompts * (spec.input_len + spec.output_len)) as f64;

        Ok(BenchmarkTierResult {
            label: spec.label.clone(),
            repeat: 0,
            seed,
            concurrency: spec.concurrency,
            num_prompts: spec.num_prompts,
            input_len: spec.input_len,
            output_len: spec.output_len,
            steps,
            overlapped_decode_steps,
            wall_time_ms: wall_secs * 1000.0,
            output_throughput_tok_s: total_out_tokens / wall_secs,
            total_throughput_tok_s: total_all_tokens / wall_secs,
            mean_ttft_ms: mean_us_as_ms(&ttfts_us),
            median_ttft_ms: percentile_us_as_ms(&ttfts_us, 50),
            p99_ttft_ms: percentile_us_as_ms(&ttfts_us, 99),
            mean_submit_ttft_ms: mean_us_as_ms(&submit_ttfts_us),
            median_submit_ttft_ms: percentile_us_as_ms(&submit_ttfts_us, 50),
            p99_submit_ttft_ms: percentile_us_as_ms(&submit_ttfts_us, 99),
            mean_tpot_ms: mean_us_as_ms(&tpots_us),
            median_tpot_ms: percentile_us_as_ms(&tpots_us, 50),
            p99_tpot_ms: percentile_us_as_ms(&tpots_us, 99),
            mean_itl_ms: mean_us_as_ms(&itls_us),
            median_itl_ms: percentile_us_as_ms(&itls_us, 50),
            p99_itl_ms: percentile_us_as_ms(&itls_us, 99),
        })
    }

    #[allow(clippy::too_many_arguments)]
    fn drive_requests(
        &mut self,
        concurrency_cap: usize,
        waiting_permitted: &mut VecDeque<PendingRequest>,
        unpermitted_pool: &mut VecDeque<(usize, Vec<i32>, usize, Instant)>,
        completed: &mut [Vec<i32>],
        mut completed_top2: Option<&mut [Vec<StepTop2Logits>]>,
        ttfts_us: &mut Vec<u64>,
        submit_ttfts_us: &mut Vec<u64>,
        tpots_us: &mut Vec<u64>,
        itls_us: &mut Vec<u64>,
        steps_out: &mut u64,
        overlapped_out: &mut u64,
    ) -> Result<()> {
        if self.poisoned {
            return Err(EngineError::Protocol(
                "serving engine is poisoned after a failed in-flight step".to_owned(),
            ));
        }
        let geometry = self.coordinator.config().geometry;
        let max_seqs = geometry.max_seqs as usize;
        let max_step_tokens = geometry.max_step_tokens as usize;
        let max_prefill_chunk = self.coordinator.config().max_prefill_chunk;
        let page_block_tokens = geometry.page_block_tokens as usize;
        let max_pages_per_seq = geometry.max_pages_per_seq as usize;
        self.validate_request_kv_demand(
            waiting_permitted
                .iter()
                .map(|req| (req.prompt_tokens.len(), req.target_output_len))
                .chain(
                    unpermitted_pool
                        .iter()
                        .map(|(_, prompt, target_out, _)| (prompt.len(), *target_out)),
                ),
            page_block_tokens,
            max_pages_per_seq,
        )?;
        self.poisoned = true;

        let mut active: Vec<ActiveSequence> = Vec::with_capacity(max_seqs.min(concurrency_cap));
        let mut slot_to_active: Vec<usize> = vec![usize::MAX; max_seqs];
        let mut consecutive_prefills = 0usize;
        let mut prestaged_step_id: Option<u64> = None;
        // A decode step published before its predecessor's result was read, and whether its
        // plan promised that the step after it is also published ahead.
        let mut published_ahead: Option<(u64, bool)> = None;

        while !active.is_empty() || !waiting_permitted.is_empty() || !unpermitted_pool.is_empty() {
            let mut ahead_promise = None;
            let step_id = if let Some((ahead_step, promise)) = published_ahead.take() {
                *overlapped_out += 1;
                ahead_promise = Some(promise);
                ahead_step
            } else if let Some(ready_step) = prestaged_step_id.take() {
                *overlapped_out += 1;
                ready_step
            } else {
                let step_now = Instant::now();
                self.prepare_dram_tier_for_step(
                    &active,
                    waiting_permitted,
                    max_seqs,
                    page_block_tokens,
                    step_now,
                )?;
                let waiting_admissible = self.front_request_admissible(
                    &active,
                    waiting_permitted,
                    max_seqs,
                    page_block_tokens,
                    step_now,
                );
                self.step_cursor += 1;
                let sid = self.step_cursor;
                let decode_active_count = active
                    .iter()
                    .filter(|seq| seq.prefilled_tokens == seq.prompt_tokens.len())
                    .count();
                let has_partial_prefill = decode_active_count != active.len();
                if !self.speculative.enabled()
                    && let Some(mixed_cap) = self.max_mixed_step_tokens
                {
                    let can_attempt_mixed = has_partial_prefill || waiting_admissible;
                    if can_attempt_mixed {
                        match self.stage_mixed_step(
                            sid,
                            mixed_cap,
                            &mut active,
                            &mut slot_to_active,
                            waiting_permitted,
                            page_block_tokens,
                            max_pages_per_seq,
                            step_now,
                        )? {
                            MixedStepOutcome::MixedOrPrefill {
                                had_decodes,
                                had_chunk,
                            } => {
                                self.last_generation_schedule.prefill_steps += 1;
                                if had_chunk {
                                    self.last_generation_schedule.chunked_prefill_steps += 1;
                                }
                                if had_decodes && (has_partial_prefill || had_chunk) {
                                    self.last_generation_schedule
                                        .decode_steps_with_partial_prefill += 1;
                                }
                                self.last_generation_schedule.max_active_sequences = self
                                    .last_generation_schedule
                                    .max_active_sequences
                                    .max(active.len());
                                consecutive_prefills += 1;
                            }
                            MixedStepOutcome::PureDecodeFallback => {
                                self.last_generation_schedule.decode_steps += 1;
                                if has_partial_prefill {
                                    self.last_generation_schedule
                                        .decode_steps_with_partial_prefill += 1;
                                }
                                consecutive_prefills = 0;
                            }
                        }
                    } else {
                        self.stage_decode_step(
                            sid,
                            &mut active,
                            page_block_tokens,
                            max_pages_per_seq,
                            true,
                        )?;
                        self.last_generation_schedule.decode_steps += 1;
                        consecutive_prefills = 0;
                    }
                } else {
                    let run_prefill = (has_partial_prefill || waiting_admissible)
                        && should_schedule_prefill(
                            decode_active_count,
                            active.len(),
                            has_partial_prefill,
                            max_seqs,
                            max_step_tokens,
                            max_prefill_chunk,
                            waiting_permitted,
                            consecutive_prefills,
                            self.max_prefill_coalesce_wait_ms,
                        );
                    if run_prefill {
                        let continued_chunk = active.iter().any(|seq| {
                            seq.prefilled_tokens > seq.cached_prefix_tokens
                                && seq.prefilled_tokens < seq.prompt_tokens.len()
                        });
                        self.stage_prefill_step(
                            sid,
                            &mut active,
                            &mut slot_to_active,
                            waiting_permitted,
                            page_block_tokens,
                            max_pages_per_seq,
                            step_now,
                        )?;
                        self.last_generation_schedule.prefill_steps += 1;
                        if continued_chunk
                            || active
                                .iter()
                                .any(|seq| seq.prefilled_tokens < seq.prompt_tokens.len())
                        {
                            self.last_generation_schedule.chunked_prefill_steps += 1;
                        }
                        self.last_generation_schedule.max_active_sequences = self
                            .last_generation_schedule
                            .max_active_sequences
                            .max(active.len());
                        consecutive_prefills += 1;
                    } else {
                        self.stage_decode_step(
                            sid,
                            &mut active,
                            page_block_tokens,
                            max_pages_per_seq,
                            true,
                        )?;
                        self.last_generation_schedule.decode_steps += 1;
                        if has_partial_prefill {
                            self.last_generation_schedule
                                .decode_steps_with_partial_prefill += 1;
                        }
                        consecutive_prefills = 0;
                    }
                }
                sid
            };
            let waiting_admissible = self.front_request_admissible(
                &active,
                waiting_permitted,
                max_seqs,
                page_block_tokens,
                Instant::now(),
            );

            let has_prefix_cache = self.has_prefix_cache();
            let publish_next = match ahead_promise {
                Some(promise) => promise,
                None => {
                    let promise = self.device_token_overlap
                        && !self.speculative.enabled()
                        && self.plan_flags(step_id)? & STEP_FLAG_DECODE_ONLY != 0
                        && decode_continues_for_all(
                            &active,
                            waiting_admissible,
                            0,
                            has_prefix_cache,
                        );
                    if promise {
                        self.or_plan_flags(step_id, STEP_FLAG_NEXT_PREPUBLISHED)?;
                    }
                    self.coordinator.publish_step(step_id)?;
                    *steps_out += 1;
                    promise
                }
            };

            // Device-token overlap: while the GPUs run `step_id`, publish the next decode step
            // from known sequence lengths. Workers take its input tokens from the tokens they
            // sampled on the GPU, so the GPUs never wait for this loop to read `step_id`'s result.
            if publish_next {
                let next_sid = self.step_cursor + 1;
                self.step_cursor = next_sid;
                self.last_generation_schedule.decode_steps += 1;
                self.stage_decode_step(
                    next_sid,
                    &mut active,
                    page_block_tokens,
                    max_pages_per_seq,
                    true,
                )?;
                let promise = !self.speculative.enabled()
                    && decode_continues_for_all(&active, waiting_admissible, 1, has_prefix_cache);
                let mut flags = STEP_FLAG_DEVICE_TOKENS;
                if promise {
                    flags |= STEP_FLAG_NEXT_PREPUBLISHED;
                }
                self.or_plan_flags(next_sid, flags)?;
                self.coordinator.publish_step(next_sid)?;
                *steps_out += 1;
                published_ahead = Some((next_sid, promise));
            }

            // `in_flight_depth = 1` overlap: if no prefills are waiting and every active
            // sequence still needs >= 2 tokens (so none retires after `step_id`), pre-allocate
            // pages and pre-write `plan_slot(step_id + 1)` while the 8 GPUs execute `step_id`.
            let can_overlap_next = !publish_next
                && !self.speculative.enabled()
                && decode_continues_for_all(&active, waiting_admissible, 0, has_prefix_cache);

            if can_overlap_next {
                let next_sid = self.step_cursor + 1;
                self.last_generation_schedule.decode_steps += 1;
                self.stage_decode_step(
                    next_sid,
                    &mut active,
                    page_block_tokens,
                    max_pages_per_seq,
                    false,
                )?;
                prestaged_step_id = Some(next_sid);
            }

            self.coordinator.wait_step_completed(step_id)?;
            let now = Instant::now();

            let emitted = self.harvested_outputs(step_id)?;

            self.promote_completed_prefixes(&mut active, max_pages_per_seq)?;

            let collect_top2 = completed_top2.is_some();
            for output in emitted {
                let idx = *slot_to_active
                    .get(output.req_slot as usize)
                    .unwrap_or(&usize::MAX);
                if idx >= active.len() {
                    continue;
                }
                let seq = &mut active[idx];
                let mut accepted = output.accepted_tokens;
                accepted.truncate(
                    seq.target_output_len
                        .saturating_sub(seq.generated_tokens.len()),
                );
                if let Some((start_pos, _)) = seq.staged_verify.take() {
                    seq.current_seq_len = start_pos + accepted.len();
                    rewind_rejected_suffix_pages(
                        seq,
                        seq.current_seq_len,
                        page_block_tokens,
                        max_pages_per_seq,
                        self.coordinator.config().launch.num_kv_pages as i32,
                        &mut self.shadow_page_table,
                        &mut self.dirty_pt_slots,
                    );
                }
                for (accepted_idx, tok) in accepted.into_iter().enumerate() {
                    if let Some(prev_t) = seq.last_token_time {
                        let itl = now.duration_since(prev_t).as_micros() as u64;
                        itls_us.push(itl);
                    } else {
                        let (ttft, submit_ttft) =
                            first_token_latency_us(seq.permit_time, seq.submit_time, now);
                        ttfts_us.push(ttft);
                        submit_ttfts_us.push(submit_ttft);
                        seq.first_token_time = Some(now);
                    }
                    seq.last_token_time = Some(now);
                    seq.last_token = tok;
                    seq.generated_tokens.push(tok);
                    if collect_top2 {
                        let top1_logit = f32::from_bits(
                            output
                                .accepted_logprobs_bits
                                .get(accepted_idx)
                                .copied()
                                .unwrap_or(0),
                        );
                        let top2_logit = f32::from_bits(output.runner_up_logprob_bits);
                        seq.generated_top2.push(StepTop2Logits {
                            top1_token: tok,
                            top1_logit,
                            top2_token: output.runner_up_token,
                            top2_logit,
                            top1_minus_top2_margin: top1_logit - top2_logit,
                        });
                    }
                }
                seq.refresh_draft_tokens(&output.next_draft_tokens, &self.speculative);
            }

            // `decode_continues_for_all` guaranteed that no sequence retires after `step_id`.
            if published_ahead.is_some() {
                consecutive_prefills = 0;
                continue;
            }

            if let Some(next_sid) = prestaged_step_id {
                self.step_cursor = next_sid;
                {
                    let mut plan = self.coordinator.plan_slot_mut(next_sid)?;
                    for (i, seq) in active.iter_mut().enumerate() {
                        // The in-flight step's token is now in `generated_tokens`.
                        // Recompute the 3-token window the pre-stage froze too early.
                        let (token, lookback) = seq.revise_overlapped_decode();
                        plan.token_ids_mut()?[i] = token;
                        plan.engram_prev_window_mut()?[i * 3..i * 3 + 3].copy_from_slice(&lookback);
                    }
                }
                consecutive_prefills = 0;
                continue;
            }

            // Retire completed requests (RAII drops `SlotLease` and `Vec<BlockLease>` automatically)
            let dummy_page = self.coordinator.config().launch.num_kv_pages as i32;
            let mut idx = 0;
            while idx < active.len() {
                if active[idx].generated_tokens.len() >= active[idx].target_output_len {
                    let mut finished = active.swap_remove(idx);
                    let freed_slot = finished.slot_lease.id() as usize;
                    slot_to_active[freed_slot] = usize::MAX;
                    let row_start = freed_slot * max_pages_per_seq;
                    self.shadow_page_table[row_start..row_start + max_pages_per_seq]
                        .fill(dummy_page);
                    let dirty_bit = 1u128 << freed_slot;
                    self.dirty_pt_slots[0] |= dirty_bit;
                    self.dirty_pt_slots[1] |= dirty_bit;
                    if idx < active.len() {
                        let moved_slot = active[idx].slot_lease.id() as usize;
                        slot_to_active[moved_slot] = idx;
                    }

                    if finished.generated_tokens.len() > 1
                        && let (Some(first_t), Some(last_t)) =
                            (finished.first_token_time, finished.last_token_time)
                    {
                        let span_us = last_t.duration_since(first_t).as_micros() as u64;
                        let denom = (finished.generated_tokens.len() - 1) as u64;
                        tpots_us.push(span_us / denom);
                    }
                    completed[finished.request_id] = std::mem::take(&mut finished.generated_tokens);
                    if let Some(top2_out) = completed_top2.as_deref_mut() {
                        top2_out[finished.request_id] =
                            std::mem::take(&mut finished.generated_top2);
                    }
                    drop(finished);

                    // Release concurrency permit immediately to next waiting request
                    if let Some((req_id, prompt, target_out, submit_time)) =
                        unpermitted_pool.pop_front()
                    {
                        waiting_permitted.push_back(PendingRequest {
                            request_id: req_id,
                            prompt_tokens: prompt,
                            target_output_len: target_out,
                            sampling: SamplingParams::default(),
                            permit_time: now,
                            submit_time,
                        });
                    }
                } else {
                    idx += 1;
                }
            }
        }
        self.poisoned = false;
        Ok(())
    }

    #[allow(clippy::too_many_arguments)]
    fn stage_prefill_step(
        &mut self,
        step_id: u64,
        active: &mut Vec<ActiveSequence>,
        slot_to_active: &mut [usize],
        waiting_permitted: &mut VecDeque<PendingRequest>,
        page_block_tokens: usize,
        max_pages_per_seq: usize,
        now: Instant,
    ) -> Result<()> {
        let geometry = self.coordinator.config().geometry;
        let max_seqs = geometry.max_seqs as usize;
        let max_step_tokens = geometry.max_step_tokens as usize;
        let max_prefill_chunk = self.coordinator.config().max_prefill_chunk;

        let mut token_cursor = 0usize;
        let mut num_seqs = 0usize;
        let mut num_sample_rows = 0usize;
        let mut any_sampling = false;
        let mut any_penalty = false;

        let partial_indices = ordered_partial_prefill_indices(active, self.prefill_policy, now);
        for active_idx in partial_indices {
            let seq = &mut active[active_idx];
            let Some((chunk_start, chunk_end)) = prefill_chunk_range_aligned(
                seq.prompt_tokens.len(),
                seq.prefilled_tokens,
                max_step_tokens - token_cursor,
                max_prefill_chunk,
                self.prefill_chunk_align,
            ) else {
                continue;
            };
            let sample_row_index = (chunk_end == seq.prompt_tokens.len()).then(|| {
                let row = num_sample_rows;
                num_sample_rows += 1;
                row
            });
            if sample_row_index.is_some() {
                if seq.sampling.needs_sampler() {
                    any_sampling = true;
                }
                if seq.sampling.has_penalty() {
                    any_penalty = true;
                }
            }
            self.stage_prefill_row(
                step_id,
                seq,
                num_seqs,
                token_cursor,
                chunk_start,
                chunk_end,
                sample_row_index,
                page_block_tokens,
                max_pages_per_seq,
            )?;
            token_cursor += chunk_end - chunk_start;
            num_seqs += 1;
        }

        while active.len() < max_seqs && token_cursor < max_step_tokens {
            let Some(sel) = select_next_waiting_candidate(
                active,
                waiting_permitted,
                &self.slot_pool,
                &self.kv_pool,
                self.prefix_tree.as_ref(),
                max_seqs,
                page_block_tokens,
                self.loading_area_pages(),
                self.prefill_chunk_align,
                self.prefill_policy,
                now,
                Some((max_step_tokens - token_cursor, max_prefill_chunk, 0)),
            ) else {
                break;
            };
            let prompt_len = waiting_permitted[sel.index].prompt_tokens.len();
            if prompt_len == 0 {
                return Err(EngineError::Protocol(
                    "prompt length must be positive".to_owned(),
                ));
            }
            let Some((chunk_start, chunk_end)) = prefill_chunk_range_aligned(
                prompt_len,
                sel.cached_tokens,
                max_step_tokens - token_cursor,
                max_prefill_chunk,
                self.prefill_chunk_align,
            ) else {
                break;
            };
            let req = waiting_permitted.remove(sel.index).ok_or_else(|| {
                EngineError::Protocol("waiting queue unexpectedly empty".to_owned())
            })?;
            let slot_lease = self.slot_pool.allocate().ok_or_else(|| {
                EngineError::Protocol("SlotPool unexpectedly exhausted".to_owned())
            })?;
            let slot_id = slot_lease.id() as usize;
            let mut seq = admit_request_into_sequence(
                req,
                slot_lease,
                sel.demand_pages,
                max_pages_per_seq,
                self.prefix_tree.as_mut(),
                self.recurrent_pool.is_some(),
                &mut self.prefix_cache_queries_tokens,
                &mut self.prefix_cache_hits_tokens,
                &mut self.shadow_page_table,
                &mut self.dirty_pt_slots,
            )?;
            let sample_row_index = (chunk_end == prompt_len).then(|| {
                let row = num_sample_rows;
                num_sample_rows += 1;
                row
            });
            if sample_row_index.is_some() {
                if seq.sampling.needs_sampler() {
                    any_sampling = true;
                }
                if seq.sampling.has_penalty() {
                    any_penalty = true;
                }
            }
            self.stage_prefill_row(
                step_id,
                &mut seq,
                num_seqs,
                token_cursor,
                chunk_start,
                chunk_end,
                sample_row_index,
                page_block_tokens,
                max_pages_per_seq,
            )?;
            slot_to_active[slot_id] = active.len();
            active.push(seq);
            token_cursor += chunk_end - chunk_start;
            num_seqs += 1;
        }

        if num_seqs == 0 {
            return Err(EngineError::Protocol(
                "prefill step scheduled zero sequences".to_owned(),
            ));
        }

        let mut plan = self.coordinator.plan_slot_mut(step_id)?;
        plan.query_start_loc_mut()?[num_seqs] = token_cursor as i32;
        sync_dirty_page_table_rows(
            &mut plan,
            step_id,
            max_pages_per_seq,
            &self.shadow_page_table,
            &mut self.dirty_pt_slots,
        )?;
        let mut flags = 0u32;
        if any_sampling {
            flags |= STEP_FLAG_SAMPLING;
        }
        if any_penalty {
            flags |= STEP_FLAG_PENALTY;
        }
        *plan.header_mut()? = StepHeader {
            step_id: AtomicU64::new(0),
            opcode: StepOpcode::NormalStep as u32,
            num_seqs: num_seqs as u32,
            num_encoder_tokens: token_cursor as u32,
            encoder_graph_bucket: 0,
            num_decoder_rows: token_cursor as u32,
            decoder_graph_bucket: 0,
            num_sample_rows: num_sample_rows as u32,
            is_dspark_verify: u32::from(self.speculative.method.is_neural()),
            flags,
            _reserved: [0; 3],
        };
        Ok(())
    }

    #[allow(clippy::too_many_arguments)]
    fn stage_mixed_step(
        &mut self,
        step_id: u64,
        mixed_cap: usize,
        active: &mut Vec<ActiveSequence>,
        slot_to_active: &mut [usize],
        waiting_permitted: &mut VecDeque<PendingRequest>,
        page_block_tokens: usize,
        max_pages_per_seq: usize,
        now: Instant,
    ) -> Result<MixedStepOutcome> {
        let geometry = self.coordinator.config().geometry;
        let max_seqs = geometry.max_seqs as usize;
        let max_step_tokens = geometry.max_step_tokens as usize;
        let max_prefill_chunk = self.coordinator.config().max_prefill_chunk;

        let decode_active_count = active
            .iter()
            .filter(|seq| seq.prefilled_tokens == seq.prompt_tokens.len())
            .count();
        let continued_chunk = active.iter().any(|seq| {
            seq.prefilled_tokens > seq.cached_prefix_tokens
                && seq.prefilled_tokens < seq.prompt_tokens.len()
        });

        let step_budget = if decode_active_count > 0 {
            let base_budget = mixed_cap.max(decode_active_count + 1).min(max_step_tokens);
            if self.prefill_policy.dynamic_budget_enabled && base_budget < max_step_tokens {
                let backlog_tokens = estimate_prefill_backlog_tokens(
                    active,
                    waiting_permitted,
                    page_block_tokens,
                    &self.kv_pool,
                    self.prefix_tree.as_ref(),
                    self.prefill_policy
                        .backlog_high_tokens
                        .max(self.prefill_policy.backlog_low_tokens),
                );
                compute_dynamic_mixed_step_budget(
                    base_budget,
                    max_step_tokens,
                    backlog_tokens,
                    self.prefill_policy,
                )
            } else {
                base_budget
            }
        } else {
            max_step_tokens
        };

        // Pass 1 (Region 0): Reserve KV pages and stage all prompt-complete active decode
        // sequences first (`q_len == 1, pos > 0`) so in-flight decodes never starve.
        for seq in active
            .iter_mut()
            .filter(|seq| seq.prefilled_tokens == seq.prompt_tokens.len())
        {
            let next_seq_len = seq.current_seq_len + 1;
            seq.ensure_kv_pages_with_cache(
                next_seq_len,
                page_block_tokens,
                max_pages_per_seq,
                &self.kv_pool,
                self.prefix_tree.as_mut(),
                &mut self.prefix_cache_evictions_pages,
                &mut self.shadow_page_table,
                &mut self.dirty_pt_slots,
            )?;
        }

        let mut token_cursor = 0usize;
        let mut num_seqs = 0usize;
        let mut num_sample_rows = 0usize;
        let mut any_sampling = false;
        let mut any_penalty = false;
        let has_recurrent_pool = self.recurrent_pool.is_some();

        if decode_active_count > 0 {
            let mut staged_tokens = Vec::with_capacity(decode_active_count);
            let mut staged_positions = Vec::with_capacity(decode_active_count);
            let mut staged_slots = Vec::with_capacity(decode_active_count);
            let mut staged_lookbacks = Vec::with_capacity(decode_active_count * 3);
            let mut staged_seq_lens = Vec::with_capacity(decode_active_count);
            let mut staged_temps = Vec::with_capacity(decode_active_count);
            let mut staged_top_p = Vec::with_capacity(decode_active_count);
            let mut staged_top_k = Vec::with_capacity(decode_active_count);
            let mut staged_seeds = Vec::with_capacity(decode_active_count);
            let mut staged_pres_pen = Vec::with_capacity(decode_active_count);
            let mut staged_freq_pen = Vec::with_capacity(decode_active_count);
            for seq in active
                .iter_mut()
                .filter(|seq| seq.prefilled_tokens == seq.prompt_tokens.len())
            {
                let start_pos = seq.current_seq_len;
                let lookback = seq.lookback_window(start_pos);
                let pos = start_pos as i32;
                let next_len = pos + 1;
                let step_idx = start_pos.saturating_sub(seq.prompt_tokens.len()) as u64 + 1;
                if seq.sampling.needs_sampler() {
                    any_sampling = true;
                }
                if seq.sampling.has_penalty() {
                    any_penalty = true;
                }
                staged_tokens.push(seq.last_token);
                staged_positions.push(pos);
                staged_slots.push(seq.slot_lease.id() as i32);
                staged_lookbacks.extend_from_slice(&lookback);
                staged_seq_lens.push(next_len);
                staged_temps.push(seq.sampling.temperature);
                staged_top_p.push(seq.sampling.top_p);
                staged_top_k.push(seq.sampling.top_k);
                staged_seeds.push(seq.sampling.step_seed(step_idx));
                staged_pres_pen.push(seq.sampling.presence_penalty);
                staged_freq_pen.push(seq.sampling.frequency_penalty);
                seq.current_seq_len = next_len as usize;
                token_cursor += 1;
                num_seqs += 1;
                num_sample_rows += 1;
            }
            let mut plan = self.coordinator.plan_slot_mut(step_id)?;
            plan.token_ids_mut()?[..num_seqs].copy_from_slice(&staged_tokens);
            plan.positions_mut()?[..num_seqs].copy_from_slice(&staged_positions);
            plan.req_slot_ids_mut()?[..num_seqs].copy_from_slice(&staged_slots);
            plan.engram_prev_window_mut()?[..num_seqs * 3].copy_from_slice(&staged_lookbacks);
            plan.seq_lens_mut()?[..num_seqs].copy_from_slice(&staged_seq_lens);
            plan.temperatures_mut()?[..num_seqs].copy_from_slice(&staged_temps);
            plan.top_p_mut()?[..num_seqs].copy_from_slice(&staged_top_p);
            plan.top_k_mut()?[..num_seqs].copy_from_slice(&staged_top_k);
            plan.rng_seeds_mut()?[..num_seqs].copy_from_slice(&staged_seeds);
            plan.presence_penalties_mut()?[..num_seqs].copy_from_slice(&staged_pres_pen);
            plan.frequency_penalties_mut()?[..num_seqs].copy_from_slice(&staged_freq_pen);
            plan.penalty_reset_mask_mut()?[..num_seqs].fill(0);
            if has_recurrent_pool {
                plan.compressor_slot_ids_mut()?[..num_seqs].fill(-1);
                plan.replay_mask_mut()?[..token_cursor].fill(0xFF);
            }
            {
                let sample_rows = plan.sample_row_indices_mut()?;
                for (row, dst) in sample_rows[..num_sample_rows].iter_mut().enumerate() {
                    *dst = row as i32;
                }
            }
            {
                let qsl = plan.query_start_loc_mut()?;
                for (row, dst) in qsl[..num_seqs].iter_mut().enumerate() {
                    *dst = row as i32;
                }
            }
        }

        let mut prefill_seqs_staged = 0usize;

        // Pass 2 (Region 2 / 0): Stage partial-prefill continuations (`first_pos > 0`).
        let partial_indices = ordered_partial_prefill_indices(active, self.prefill_policy, now);
        for active_idx in partial_indices {
            if token_cursor >= step_budget {
                break;
            }
            let prompt_len = active[active_idx].prompt_tokens.len();
            let prefilled = active[active_idx].prefilled_tokens;
            let Some((chunk_start, chunk_end)) = prefill_chunk_range_aligned(
                prompt_len,
                prefilled,
                step_budget - token_cursor,
                max_prefill_chunk,
                self.prefill_chunk_align,
            ) else {
                continue;
            };
            let needed_pages = chunk_end.div_ceil(page_block_tokens);
            let extra_pages = needed_pages.saturating_sub(active[active_idx].page_ids.len());
            if extra_pages > self.reclaimable_kv_pages() {
                continue;
            }
            let seq = &mut active[active_idx];
            let sample_row_index = (chunk_end == prompt_len).then(|| {
                let row = num_sample_rows;
                num_sample_rows += 1;
                row
            });
            if sample_row_index.is_some() {
                if seq.sampling.needs_sampler() {
                    any_sampling = true;
                }
                if seq.sampling.has_penalty() {
                    any_penalty = true;
                }
            }
            self.stage_prefill_row(
                step_id,
                seq,
                num_seqs,
                token_cursor,
                chunk_start,
                chunk_end,
                sample_row_index,
                page_block_tokens,
                max_pages_per_seq,
            )?;
            token_cursor += chunk_end - chunk_start;
            num_seqs += 1;
            prefill_seqs_staged += 1;
        }

        // Pass 3 (Region 3 / 1): Admit new waiting requests.
        while active.len() < max_seqs && token_cursor < step_budget {
            let remaining_budget = step_budget - token_cursor;
            let Some(sel) = select_next_waiting_candidate(
                active,
                waiting_permitted,
                &self.slot_pool,
                &self.kv_pool,
                self.prefix_tree.as_ref(),
                max_seqs,
                page_block_tokens,
                self.loading_area_pages(),
                self.prefill_chunk_align,
                self.prefill_policy,
                now,
                Some((remaining_budget, max_prefill_chunk, num_seqs)),
            ) else {
                break;
            };
            let prompt_len = waiting_permitted[sel.index].prompt_tokens.len();
            if prompt_len == 0 {
                return Err(EngineError::Protocol(
                    "prompt length must be positive".to_owned(),
                ));
            }
            let Some((chunk_start, chunk_end)) = prefill_chunk_range_aligned(
                prompt_len,
                sel.cached_tokens,
                remaining_budget,
                max_prefill_chunk,
                self.prefill_chunk_align,
            ) else {
                break;
            };
            let req = waiting_permitted.remove(sel.index).ok_or_else(|| {
                EngineError::Protocol("waiting queue unexpectedly empty".to_owned())
            })?;
            let slot_lease = self.slot_pool.allocate().ok_or_else(|| {
                EngineError::Protocol("SlotPool unexpectedly exhausted".to_owned())
            })?;
            let slot_id = slot_lease.id() as usize;
            let mut seq = admit_request_into_sequence(
                req,
                slot_lease,
                sel.demand_pages,
                max_pages_per_seq,
                self.prefix_tree.as_mut(),
                self.recurrent_pool.is_some(),
                &mut self.prefix_cache_queries_tokens,
                &mut self.prefix_cache_hits_tokens,
                &mut self.shadow_page_table,
                &mut self.dirty_pt_slots,
            )?;
            let sample_row_index = (chunk_end == prompt_len).then(|| {
                let row = num_sample_rows;
                num_sample_rows += 1;
                row
            });
            if sample_row_index.is_some() {
                if seq.sampling.needs_sampler() {
                    any_sampling = true;
                }
                if seq.sampling.has_penalty() {
                    any_penalty = true;
                }
            }
            self.stage_prefill_row(
                step_id,
                &mut seq,
                num_seqs,
                token_cursor,
                chunk_start,
                chunk_end,
                sample_row_index,
                page_block_tokens,
                max_pages_per_seq,
            )?;
            slot_to_active[slot_id] = active.len();
            active.push(seq);
            token_cursor += chunk_end - chunk_start;
            num_seqs += 1;
            prefill_seqs_staged += 1;
        }

        if num_seqs == 0 {
            return Err(EngineError::Protocol(
                "mixed step scheduled zero sequences (KV pool or slots exhausted)".to_owned(),
            ));
        }

        let candidate_decode = prefill_seqs_staged == 0 && decode_active_count == num_seqs;
        let bucket = if candidate_decode {
            select_graph_bucket(num_seqs as u32, &DECODE_GRAPH_BUCKETS)
        } else {
            0
        };
        let is_pure_decode_fallback = candidate_decode && bucket > 0;
        let mut flags = if is_pure_decode_fallback {
            STEP_FLAG_DECODE_ONLY
        } else {
            0
        };
        if any_sampling {
            flags |= STEP_FLAG_SAMPLING;
        }
        if any_penalty {
            flags |= STEP_FLAG_PENALTY;
        }

        let mut plan = self.coordinator.plan_slot_mut(step_id)?;
        plan.query_start_loc_mut()?[num_seqs] = token_cursor as i32;
        sync_dirty_page_table_rows(
            &mut plan,
            step_id,
            max_pages_per_seq,
            &self.shadow_page_table,
            &mut self.dirty_pt_slots,
        )?;
        *plan.header_mut()? = StepHeader {
            step_id: AtomicU64::new(0),
            opcode: StepOpcode::NormalStep as u32,
            num_seqs: num_seqs as u32,
            num_encoder_tokens: token_cursor as u32,
            encoder_graph_bucket: bucket,
            num_decoder_rows: token_cursor as u32,
            decoder_graph_bucket: bucket,
            num_sample_rows: num_sample_rows as u32,
            is_dspark_verify: u32::from(self.speculative.method.is_neural()),
            flags,
            _reserved: [0; 3],
        };

        if is_pure_decode_fallback {
            Ok(MixedStepOutcome::PureDecodeFallback)
        } else {
            let had_chunk = continued_chunk
                || active
                    .iter()
                    .any(|seq| seq.prefilled_tokens < seq.prompt_tokens.len());
            Ok(MixedStepOutcome::MixedOrPrefill {
                had_decodes: decode_active_count > 0,
                had_chunk,
            })
        }
    }

    #[allow(clippy::too_many_arguments)]
    fn stage_prefill_row(
        &mut self,
        step_id: u64,
        seq: &mut ActiveSequence,
        seq_batch_pos: usize,
        token_cursor: usize,
        chunk_start: usize,
        chunk_end: usize,
        sample_row_index: Option<usize>,
        page_block_tokens: usize,
        max_pages_per_seq: usize,
    ) -> Result<()> {
        seq.ensure_kv_pages_with_cache(
            chunk_end,
            page_block_tokens,
            max_pages_per_seq,
            &self.kv_pool,
            self.prefix_tree.as_mut(),
            &mut self.prefix_cache_evictions_pages,
            &mut self.shadow_page_table,
            &mut self.dirty_pt_slots,
        )?;
        let chunk_len = chunk_end - chunk_start;
        let slot_i32 = seq.slot_lease.id() as i32;
        let lookback = seq.lookback_window(chunk_start);

        let has_recurrent_pool = self.recurrent_pool.is_some();
        let restore_id = if has_recurrent_pool
            && chunk_start == seq.cached_prefix_tokens
            && seq.restore_checkpoint_id >= 0
        {
            let id = seq.restore_checkpoint_id;
            seq.restore_checkpoint_id = -1;
            id
        } else {
            -1
        };

        let mut checkpoint_rows: Vec<(usize, u8)> = Vec::new();
        if let (Some(recurrent_pool), Some(node_tokens)) = (
            self.recurrent_pool.as_ref(),
            self.prefix_tree.as_ref().map(|t| t.block_tokens() as usize),
        ) && !seq.checkpoint_alloc_stopped
        {
            let mut end_tokens = (chunk_start / node_tokens + 1) * node_tokens;
            while end_tokens <= chunk_end {
                if recurrent_pool.available() == 0
                    && let Some(tree) = self.prefix_tree.as_mut()
                {
                    while recurrent_pool.available() == 0 {
                        let evicted = tree.evict_lru(1);
                        if evicted == 0 {
                            break;
                        }
                        self.prefix_cache_evictions_pages +=
                            (evicted * tree.pages_per_node()) as u64;
                    }
                }
                let Some(ckpt_lease) = recurrent_pool.allocate(1) else {
                    seq.checkpoint_alloc_stopped = true;
                    break;
                };
                let ckpt_id = ckpt_lease.ids()[0];
                if !(0..255).contains(&ckpt_id) {
                    return Err(EngineError::Protocol(format!(
                        "recurrent checkpoint slot {ckpt_id} outside valid uint8 range 0..255"
                    )));
                }
                let rel_row = (end_tokens - 1) - chunk_start;
                checkpoint_rows.push((rel_row, ckpt_id as u8));
                seq.pending_checkpoints.push((end_tokens, ckpt_lease));
                end_tokens += node_tokens;
            }
        }

        let mut plan = self.coordinator.plan_slot_mut(step_id)?;
        plan.token_ids_mut()?[token_cursor..token_cursor + chunk_len]
            .copy_from_slice(&seq.prompt_tokens[chunk_start..chunk_end]);
        {
            let positions = &mut plan.positions_mut()?[token_cursor..token_cursor + chunk_len];
            for (offset, dst) in positions.iter_mut().enumerate() {
                *dst = (chunk_start + offset) as i32;
            }
        }
        plan.req_slot_ids_mut()?[token_cursor..token_cursor + chunk_len].fill(slot_i32);
        plan.engram_prev_window_mut()?[seq_batch_pos * 3..seq_batch_pos * 3 + 3]
            .copy_from_slice(&lookback);
        plan.query_start_loc_mut()?[seq_batch_pos] = token_cursor as i32;
        plan.seq_lens_mut()?[seq_batch_pos] = chunk_end as i32;
        if has_recurrent_pool {
            plan.compressor_slot_ids_mut()?[seq_batch_pos] = restore_id;
            let replay_slice = &mut plan.replay_mask_mut()?[token_cursor..token_cursor + chunk_len];
            replay_slice.fill(0xFF);
            for (rel_row, ckpt_u8) in checkpoint_rows {
                replay_slice[rel_row] = ckpt_u8;
            }
        }
        if let Some(sample_row) = sample_row_index {
            plan.sample_row_indices_mut()?[sample_row] = (token_cursor + chunk_len - 1) as i32;
            plan.temperatures_mut()?[sample_row] = seq.sampling.temperature;
            plan.top_p_mut()?[sample_row] = seq.sampling.top_p;
            plan.top_k_mut()?[sample_row] = seq.sampling.top_k;
            plan.rng_seeds_mut()?[sample_row] = seq.sampling.step_seed(0);
            plan.presence_penalties_mut()?[sample_row] = seq.sampling.presence_penalty;
            plan.frequency_penalties_mut()?[sample_row] = seq.sampling.frequency_penalty;
            plan.penalty_reset_mask_mut()?[sample_row] = 1;
        }
        seq.prefilled_tokens = chunk_end;
        seq.current_seq_len = chunk_end;
        Ok(())
    }

    fn stage_speculative_decode_step(
        &mut self,
        step_id: u64,
        active: &mut [ActiveSequence],
        page_block_tokens: usize,
        max_pages_per_seq: usize,
    ) -> Result<()> {
        let config = self.speculative.clone();
        let mut staged_tokens = Vec::new();
        let mut staged_positions = Vec::new();
        let mut staged_slots = Vec::new();
        let mut staged_lookbacks = Vec::new();
        let mut staged_seq_lens = Vec::new();
        let mut query_start_loc = vec![0i32];
        let mut num_seqs = 0usize;

        for seq in active
            .iter_mut()
            .filter(|seq| seq.prefilled_tokens == seq.prompt_tokens.len())
        {
            let start_pos = seq.current_seq_len;
            let remaining = seq
                .target_output_len
                .saturating_sub(seq.generated_tokens.len());
            let k_eff = seq
                .pending_draft_tokens
                .len()
                .min(config.num_speculative_tokens)
                .min(remaining.saturating_sub(1));
            let mut rows = Vec::with_capacity(k_eff + 1);
            rows.push(seq.last_token);
            rows.extend_from_slice(&seq.pending_draft_tokens[..k_eff]);
            let target_len = start_pos + rows.len();
            seq.ensure_kv_pages_with_cache(
                target_len,
                page_block_tokens,
                max_pages_per_seq,
                &self.kv_pool,
                self.prefix_tree.as_mut(),
                &mut self.prefix_cache_evictions_pages,
                &mut self.shadow_page_table,
                &mut self.dirty_pt_slots,
            )?;

            for (row, &token) in rows.iter().enumerate() {
                staged_tokens.push(token);
                staged_positions.push((start_pos + row) as i32);
                staged_slots.push(seq.slot_lease.id() as i32);
            }
            staged_lookbacks.extend_from_slice(&seq.lookback_window(start_pos));
            staged_seq_lens.push(target_len as i32);
            query_start_loc.push(staged_tokens.len() as i32);
            seq.staged_verify = Some((start_pos, k_eff));
            seq.current_seq_len = target_len;
            num_seqs += 1;
        }

        if num_seqs == 0 {
            return Err(EngineError::Protocol(
                "speculative decode scheduled zero prompt-complete sequences".to_owned(),
            ));
        }
        let total_tokens = staged_tokens.len();
        let mut plan = self.coordinator.plan_slot_mut(step_id)?;
        plan.token_ids_mut()?[..total_tokens].copy_from_slice(&staged_tokens);
        plan.positions_mut()?[..total_tokens].copy_from_slice(&staged_positions);
        plan.req_slot_ids_mut()?[..total_tokens].copy_from_slice(&staged_slots);
        plan.engram_prev_window_mut()?[..num_seqs * 3].copy_from_slice(&staged_lookbacks);
        plan.seq_lens_mut()?[..num_seqs].copy_from_slice(&staged_seq_lens);
        plan.query_start_loc_mut()?[..=num_seqs].copy_from_slice(&query_start_loc);
        {
            let sample_rows = plan.sample_row_indices_mut()?;
            for (row, dst) in sample_rows[..total_tokens].iter_mut().enumerate() {
                *dst = row as i32;
            }
        }
        if self.recurrent_pool.is_some() {
            plan.compressor_slot_ids_mut()?[..num_seqs].fill(-1);
            plan.replay_mask_mut()?[..total_tokens].fill(0xFF);
        }
        sync_dirty_page_table_rows(
            &mut plan,
            step_id,
            max_pages_per_seq,
            &self.shadow_page_table,
            &mut self.dirty_pt_slots,
        )?;
        *plan.header_mut()? = StepHeader {
            step_id: AtomicU64::new(0),
            opcode: StepOpcode::NormalStep as u32,
            num_seqs: num_seqs as u32,
            num_encoder_tokens: total_tokens as u32,
            encoder_graph_bucket: 0,
            num_decoder_rows: total_tokens as u32,
            decoder_graph_bucket: 0,
            num_sample_rows: total_tokens as u32,
            is_dspark_verify: 1,
            flags: 0,
            _reserved: [0; 3],
        };
        Ok(())
    }

    fn stage_decode_step(
        &mut self,
        step_id: u64,
        active: &mut [ActiveSequence],
        page_block_tokens: usize,
        max_pages_per_seq: usize,
        commit_tokens_immediately: bool,
    ) -> Result<()> {
        let any_sampling = active.iter().any(|seq| {
            seq.prefilled_tokens == seq.prompt_tokens.len() && seq.sampling.needs_sampler()
        });
        let any_penalty = active.iter().any(|seq| {
            seq.prefilled_tokens == seq.prompt_tokens.len() && seq.sampling.has_penalty()
        });
        let has_drafts = !any_sampling
            && active.iter().any(|seq| {
                seq.prefilled_tokens == seq.prompt_tokens.len()
                    && !seq.pending_draft_tokens.is_empty()
                    && seq.generated_tokens.len() + 1 < seq.target_output_len
            });
        if self.speculative.enabled() {
            if !commit_tokens_immediately {
                return Err(EngineError::Protocol(
                    "speculative decode cannot be pre-staged before drafts are harvested"
                        .to_owned(),
                ));
            }
            if has_drafts {
                return self.stage_speculative_decode_step(
                    step_id,
                    active,
                    page_block_tokens,
                    max_pages_per_seq,
                );
            }
        }
        let num_seqs = active
            .iter()
            .filter(|seq| seq.prefilled_tokens == seq.prompt_tokens.len())
            .count();
        if num_seqs == 0 {
            return Err(EngineError::Protocol(
                "decode step scheduled zero prompt-complete sequences".to_owned(),
            ));
        }

        let bucket = select_graph_bucket(num_seqs as u32, &DECODE_GRAPH_BUCKETS);
        if bucket == 0 {
            return Err(EngineError::Protocol(format!(
                "decode batch size {num_seqs} exceeds max decode CUDA Graph bucket ({MAX_DECODE_GRAPH_SEQS})"
            )));
        }

        for seq in active
            .iter_mut()
            .filter(|seq| seq.prefilled_tokens == seq.prompt_tokens.len())
        {
            let next_seq_len = seq.current_seq_len + 1;
            seq.ensure_kv_pages_with_cache(
                next_seq_len,
                page_block_tokens,
                max_pages_per_seq,
                &self.kv_pool,
                self.prefix_tree.as_mut(),
                &mut self.prefix_cache_evictions_pages,
                &mut self.shadow_page_table,
                &mut self.dirty_pt_slots,
            )?;
        }

        let mut staged_tokens = Vec::with_capacity(if commit_tokens_immediately {
            num_seqs
        } else {
            0
        });
        let mut staged_positions = Vec::with_capacity(num_seqs);
        let mut staged_slots = Vec::with_capacity(num_seqs);
        let mut staged_lookbacks = Vec::with_capacity(num_seqs * 3);
        let mut staged_seq_lens = Vec::with_capacity(num_seqs);
        let mut staged_temps = Vec::with_capacity(if any_sampling { num_seqs } else { 0 });
        let mut staged_top_p = Vec::with_capacity(if any_sampling { num_seqs } else { 0 });
        let mut staged_top_k = Vec::with_capacity(if any_sampling { num_seqs } else { 0 });
        let mut staged_seeds = Vec::with_capacity(if any_sampling { num_seqs } else { 0 });
        let mut staged_pres_pen = Vec::with_capacity(if any_penalty { num_seqs } else { 0 });
        let mut staged_freq_pen = Vec::with_capacity(if any_penalty { num_seqs } else { 0 });
        for seq in active
            .iter_mut()
            .filter(|seq| seq.prefilled_tokens == seq.prompt_tokens.len())
        {
            let start_pos = seq.current_seq_len;
            let lookback = seq.lookback_window(start_pos);
            let pos = start_pos as i32;
            let next_len = pos + 1;
            if commit_tokens_immediately {
                staged_tokens.push(seq.last_token);
                seq.current_seq_len = next_len as usize;
            }
            staged_positions.push(pos);
            staged_slots.push(seq.slot_lease.id() as i32);
            staged_lookbacks.extend_from_slice(&lookback);
            staged_seq_lens.push(next_len);
            if any_sampling {
                let step_idx = start_pos.saturating_sub(seq.prompt_tokens.len()) as u64 + 1;
                staged_temps.push(seq.sampling.temperature);
                staged_top_p.push(seq.sampling.top_p);
                staged_top_k.push(seq.sampling.top_k);
                staged_seeds.push(seq.sampling.step_seed(step_idx));
                if any_penalty {
                    staged_pres_pen.push(seq.sampling.presence_penalty);
                    staged_freq_pen.push(seq.sampling.frequency_penalty);
                }
            }
        }

        let mut plan = self.coordinator.plan_slot_mut(step_id)?;
        if commit_tokens_immediately {
            plan.token_ids_mut()?[..num_seqs].copy_from_slice(&staged_tokens);
        }
        plan.positions_mut()?[..num_seqs].copy_from_slice(&staged_positions);
        plan.req_slot_ids_mut()?[..num_seqs].copy_from_slice(&staged_slots);
        plan.engram_prev_window_mut()?[..num_seqs * 3].copy_from_slice(&staged_lookbacks);
        plan.seq_lens_mut()?[..num_seqs].copy_from_slice(&staged_seq_lens);
        if any_sampling {
            plan.temperatures_mut()?[..num_seqs].copy_from_slice(&staged_temps);
            plan.top_p_mut()?[..num_seqs].copy_from_slice(&staged_top_p);
            plan.top_k_mut()?[..num_seqs].copy_from_slice(&staged_top_k);
            plan.rng_seeds_mut()?[..num_seqs].copy_from_slice(&staged_seeds);
            if any_penalty {
                plan.presence_penalties_mut()?[..num_seqs].copy_from_slice(&staged_pres_pen);
                plan.frequency_penalties_mut()?[..num_seqs].copy_from_slice(&staged_freq_pen);
                plan.penalty_reset_mask_mut()?[..num_seqs].fill(0);
            }
        }
        {
            let sample_rows = plan.sample_row_indices_mut()?;
            for (row, dst) in sample_rows[..num_seqs].iter_mut().enumerate() {
                *dst = row as i32;
            }
        }
        {
            let qsl = plan.query_start_loc_mut()?;
            for (row, dst) in qsl[..num_seqs].iter_mut().enumerate() {
                *dst = row as i32;
            }
            qsl[num_seqs] = num_seqs as i32;
        }
        sync_dirty_page_table_rows(
            &mut plan,
            step_id,
            max_pages_per_seq,
            &self.shadow_page_table,
            &mut self.dirty_pt_slots,
        )?;
        let mut flags = STEP_FLAG_DECODE_ONLY;
        if any_sampling {
            flags |= STEP_FLAG_SAMPLING;
        }
        if any_penalty {
            flags |= STEP_FLAG_PENALTY;
        }
        *plan.header_mut()? = StepHeader {
            step_id: AtomicU64::new(0),
            opcode: StepOpcode::NormalStep as u32,
            num_seqs: num_seqs as u32,
            num_encoder_tokens: num_seqs as u32,
            encoder_graph_bucket: bucket,
            num_decoder_rows: num_seqs as u32,
            decoder_graph_bucket: bucket,
            num_sample_rows: num_seqs as u32,
            is_dspark_verify: 0,
            flags,
            _reserved: [0; 3],
        };
        Ok(())
    }

    pub fn shutdown(mut self) -> Result<()> {
        if self.poisoned {
            return Err(EngineError::Protocol(
                "cannot gracefully shut down engine after failed in-flight step; cluster aborted"
                    .to_owned(),
            ));
        }
        self.step_cursor += 1;
        let stop_step = self.step_cursor;
        self.coordinator.shutdown_and_wait(stop_step)
    }

    pub fn shutdown_with_stats(mut self) -> Result<(u32, u32)> {
        if self.poisoned {
            return Err(EngineError::Protocol(
                "cannot gracefully shut down engine after failed in-flight step; cluster aborted"
                    .to_owned(),
            ));
        }
        self.step_cursor += 1;
        let stop_step = self.step_cursor;
        self.coordinator.shutdown_and_wait_with_stats(stop_step)
    }
}

fn prefill_chunk_range_aligned(
    prompt_len: usize,
    prefilled_tokens: usize,
    token_budget: usize,
    max_prefill_chunk: usize,
    align: usize,
) -> Option<(usize, usize)> {
    if prefilled_tokens >= prompt_len || token_budget == 0 || max_prefill_chunk == 0 {
        return None;
    }
    let chunk_len = (prompt_len - prefilled_tokens)
        .min(token_budget)
        .min(max_prefill_chunk);
    let raw_end = prefilled_tokens + chunk_len;
    if raw_end == prompt_len || align <= 1 {
        return Some((prefilled_tokens, raw_end));
    }
    // Keep non-final chunk boundaries aligned to `align` (e.g. 4 tokens for DeepSeek-V4.1
    // `kpool` `compress_ratio == 4`) so intermediate mixed-step chunks never split a pool group.
    let aligned_end = (raw_end / align) * align;
    (aligned_end > prefilled_tokens).then_some((prefilled_tokens, aligned_end))
}

#[cfg(test)]
fn prefill_chunk_range(
    prompt_len: usize,
    prefilled_tokens: usize,
    token_budget: usize,
    max_prefill_chunk: usize,
) -> Option<(usize, usize)> {
    prefill_chunk_range_aligned(
        prompt_len,
        prefilled_tokens,
        token_budget,
        max_prefill_chunk,
        1,
    )
}

fn first_token_latency_us(
    permit_time: Instant,
    submit_time: Instant,
    first_token_time: Instant,
) -> (u64, u64) {
    (
        first_token_time.duration_since(permit_time).as_micros() as u64,
        first_token_time.duration_since(submit_time).as_micros() as u64,
    )
}

#[allow(clippy::too_many_arguments)]
fn should_schedule_prefill(
    decode_active_count: usize,
    total_active_count: usize,
    has_partial_prefill: bool,
    max_seqs: usize,
    max_step_tokens: usize,
    max_prefill_chunk: usize,
    waiting_permitted: &VecDeque<PendingRequest>,
    consecutive_prefills: usize,
    max_prefill_coalesce_wait_ms: u64,
) -> bool {
    if (total_active_count >= max_seqs || waiting_permitted.is_empty()) && !has_partial_prefill {
        return false;
    }
    if decode_active_count > 0 && consecutive_prefills >= 1 {
        return false;
    }
    if has_partial_prefill || decode_active_count == 0 {
        return true;
    }
    let first_len = waiting_permitted.front().map_or(1, |req| {
        req.prompt_tokens
            .len()
            .min(max_prefill_chunk)
            .min(max_step_tokens)
            .max(1)
    });
    let full_prefill_batch = (max_step_tokens / first_len).max(1);
    let free_slots = max_seqs - total_active_count;
    if waiting_permitted.len() >= full_prefill_batch.min(free_slots) {
        return true;
    }
    waiting_permitted.front().is_some_and(|req| {
        req.permit_time.elapsed().as_millis() >= u128::from(max_prefill_coalesce_wait_ms)
    })
}

/// SplitMix64 bit-mixing hash used for deterministic prompt generation and sampling seeds.
#[inline]
pub fn splitmix64(mut x: u64) -> u64 {
    x = x.wrapping_add(0x9E37_79B9_7F4A_7C15);
    x = (x ^ (x >> 30)).wrapping_mul(0xBF58_476D_1CE4_E5B9);
    x = (x ^ (x >> 27)).wrapping_mul(0x94D0_49BB_1331_11EB);
    x ^ (x >> 31)
}

/// Generate `num_prompts` deterministic pseudo-random token sequences in `[256, 65536)`
/// using the default [`BENCHMARK_RANDOM_SEED`].
pub fn build_deterministic_random_prompts(num_prompts: usize, input_len: usize) -> Vec<Vec<i32>> {
    build_deterministic_random_prompts_seeded(num_prompts, input_len, BENCHMARK_RANDOM_SEED)
}

/// Generate `num_prompts` deterministic pseudo-random token sequences in `[256, 65536)`
/// parameterized by `seed` so each `(round, tier)` receives fresh uncached prompts.
pub fn build_deterministic_random_prompts_seeded(
    num_prompts: usize,
    input_len: usize,
    seed: u64,
) -> Vec<Vec<i32>> {
    let mut prompts = Vec::with_capacity(num_prompts);
    let mut state: u64 = splitmix64(seed);
    for p in 0..num_prompts {
        let mut row = Vec::with_capacity(input_len);
        let prompt_mix = splitmix64(
            seed.wrapping_add(
                (p as u64)
                    .wrapping_add(1)
                    .wrapping_mul(0xBF58_476D_1CE4_E5B9),
            ),
        );
        state = state.rotate_left(17) ^ prompt_mix;
        for _ in 0..input_len {
            state = state
                .wrapping_mul(6_364_136_223_846_793_005)
                .wrapping_add(1_442_695_040_888_963_407);
            let tok = 256 + ((state >> 32) as i32).rem_euclid(65_000);
            row.push(tok);
        }
        prompts.push(row);
    }
    prompts
}

#[cfg(test)]
mod tests;

#[path = "serve_loop.rs"]
pub mod serve_loop;
pub use serve_loop::{
    EngineState, EventSink, LoopEvent, LoopFinishReason, MockScript, OnlineConfig, OnlineMetrics,
    ServeLoopHandle, StepLogprob, StepTopLogprob, SubmitError, SubmitRequest,
    TPOT_BUCKET_BOUNDS_SECS, TTFT_BUCKET_BOUNDS_SECS, compute_step_logprob,
    spawn as spawn_serve_loop, spawn_mock, spawn_with_state as spawn_serve_loop_with_state,
};
