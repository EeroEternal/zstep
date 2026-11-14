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

//! Continuous-batching Mode B serving scheduler and multi-tier benchmark harness.
//!
//! Owns request slots and physical 16-token KV pages via RAII handles (`SlotLease`,
//! `zs_kv::BlockLease`) and overlaps next-step decode preparation (`in_flight_depth = 1`)
//! while the 8 GPU workers execute the current step.

use std::collections::VecDeque;
use std::process::Command;
use std::sync::atomic::AtomicU64;
use std::sync::{Arc, Mutex};
use std::time::Instant;

use zs_kv::{BlockLease, BlockPool, CacheGroupKind};
use zs_wire::{
    DECODE_GRAPH_BUCKETS, MAX_DECODE_GRAPH_SEQS, STEP_FLAG_DECODE_ONLY, STEP_FLAG_DEVICE_TOKENS,
    STEP_FLAG_NEXT_PREPUBLISHED, StepHeader, StepOpcode,
};

use crate::mode_b::{ModeBConfig, ModeBCoordinator, ModeBError, Result};
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

/// One concurrency tier specification matching `vllm bench serve` random workload parameters.
#[derive(Debug, Clone)]
pub struct BenchmarkTierSpec {
    pub label: String,
    pub concurrency: usize,
    pub num_prompts: usize,
    pub input_len: usize,
    pub output_len: usize,
}

pub const BENCHMARK_RANDOM_SEED: u64 = 0x9E37_79B9_7F4A_7C15;
/// Default maximum wait (ms) before flushing a non-full prefill batch while decodes are active.
pub const DEFAULT_PREFILL_COALESCE_WAIT_MS: u64 = 25;

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

/// Per-tier continuous-batching throughput and latency metrics for one repeat.
#[derive(Debug, Clone)]
pub struct BenchmarkTierResult {
    pub label: String,
    pub repeat: usize,
    pub seed: u64,
    pub concurrency: usize,
    pub num_prompts: usize,
    pub input_len: usize,
    pub output_len: usize,
    pub steps: u64,
    pub overlapped_decode_steps: u64,
    pub wall_time_ms: f64,
    pub output_throughput_tok_s: f64,
    pub total_throughput_tok_s: f64,
    pub mean_ttft_ms: f64,
    pub median_ttft_ms: f64,
    pub p99_ttft_ms: f64,
    pub mean_submit_ttft_ms: f64,
    pub median_submit_ttft_ms: f64,
    pub p99_submit_ttft_ms: f64,
    pub mean_tpot_ms: f64,
    pub median_tpot_ms: f64,
    pub p99_tpot_ms: f64,
    pub mean_itl_ms: f64,
    pub median_itl_ms: f64,
    pub p99_itl_ms: f64,
}

impl BenchmarkTierResult {
    fn to_json(&self) -> String {
        format!(
            concat!(
                "    {{\n",
                "      \"label\": {},\n",
                "      \"repeat\": {},\n",
                "      \"seed\": {},\n",
                "      \"concurrency\": {},\n",
                "      \"num_prompts\": {},\n",
                "      \"input_len\": {},\n",
                "      \"output_len\": {},\n",
                "      \"steps\": {},\n",
                "      \"overlapped_decode_steps\": {},\n",
                "      \"wall_time_ms\": {:.3},\n",
                "      \"output_throughput_tok_s\": {:.2},\n",
                "      \"total_throughput_tok_s\": {:.2},\n",
                "      \"mean_ttft_ms\": {:.2},\n",
                "      \"median_ttft_ms\": {:.2},\n",
                "      \"p99_ttft_ms\": {:.2},\n",
                "      \"mean_submit_ttft_ms\": {:.2},\n",
                "      \"median_submit_ttft_ms\": {:.2},\n",
                "      \"p99_submit_ttft_ms\": {:.2},\n",
                "      \"mean_tpot_ms\": {:.2},\n",
                "      \"median_tpot_ms\": {:.2},\n",
                "      \"p99_tpot_ms\": {:.2},\n",
                "      \"mean_itl_ms\": {:.2},\n",
                "      \"median_itl_ms\": {:.2},\n",
                "      \"p99_itl_ms\": {:.2}\n",
                "    }}"
            ),
            json_string(&self.label),
            self.repeat,
            self.seed,
            self.concurrency,
            self.num_prompts,
            self.input_len,
            self.output_len,
            self.steps,
            self.overlapped_decode_steps,
            self.wall_time_ms,
            self.output_throughput_tok_s,
            self.total_throughput_tok_s,
            self.mean_ttft_ms,
            self.median_ttft_ms,
            self.p99_ttft_ms,
            self.mean_submit_ttft_ms,
            self.median_submit_ttft_ms,
            self.p99_submit_ttft_ms,
            self.mean_tpot_ms,
            self.median_tpot_ms,
            self.p99_tpot_ms,
            self.mean_itl_ms,
            self.median_itl_ms,
            self.p99_itl_ms,
        )
    }
}

#[derive(Debug, Clone)]
pub struct AggregateMetric {
    pub mean: f64,
    pub std: f64,
    pub min: f64,
    pub max: f64,
}

impl AggregateMetric {
    fn from_samples(samples: &[f64]) -> Option<Self> {
        if samples.is_empty() {
            return None;
        }
        let mean = samples.iter().sum::<f64>() / samples.len() as f64;
        let variance = samples
            .iter()
            .map(|sample| (sample - mean).powi(2))
            .sum::<f64>()
            / samples.len() as f64;
        let min = samples.iter().copied().fold(f64::INFINITY, f64::min);
        let max = samples.iter().copied().fold(f64::NEG_INFINITY, f64::max);
        Some(Self {
            mean,
            std: variance.sqrt(),
            min,
            max,
        })
    }

    fn to_json(&self) -> String {
        format!(
            "{{\"mean\": {:.6}, \"std\": {:.6}, \"min\": {:.6}, \"max\": {:.6}}}",
            self.mean, self.std, self.min, self.max
        )
    }
}

#[derive(Debug, Clone)]
pub struct TierAggregateStats {
    pub label: String,
    pub repeats: usize,
    pub output_throughput_tok_s: AggregateMetric,
    pub total_throughput_tok_s: AggregateMetric,
    pub mean_ttft_ms: AggregateMetric,
    pub median_ttft_ms: AggregateMetric,
    pub p99_ttft_ms: AggregateMetric,
    pub mean_submit_ttft_ms: AggregateMetric,
    pub median_submit_ttft_ms: AggregateMetric,
    pub p99_submit_ttft_ms: AggregateMetric,
    pub mean_tpot_ms: AggregateMetric,
    pub p99_tpot_ms: AggregateMetric,
    pub p99_itl_ms: AggregateMetric,
}

impl TierAggregateStats {
    pub fn from_results(label: String, results: &[BenchmarkTierResult]) -> Option<Self> {
        macro_rules! metric {
            ($field:ident) => {
                AggregateMetric::from_samples(
                    &results
                        .iter()
                        .map(|result| result.$field)
                        .collect::<Vec<_>>(),
                )?
            };
        }
        Some(Self {
            label,
            repeats: results.len(),
            output_throughput_tok_s: metric!(output_throughput_tok_s),
            total_throughput_tok_s: metric!(total_throughput_tok_s),
            mean_ttft_ms: metric!(mean_ttft_ms),
            median_ttft_ms: metric!(median_ttft_ms),
            p99_ttft_ms: metric!(p99_ttft_ms),
            mean_submit_ttft_ms: metric!(mean_submit_ttft_ms),
            median_submit_ttft_ms: metric!(median_submit_ttft_ms),
            p99_submit_ttft_ms: metric!(p99_submit_ttft_ms),
            mean_tpot_ms: metric!(mean_tpot_ms),
            p99_tpot_ms: metric!(p99_tpot_ms),
            p99_itl_ms: metric!(p99_itl_ms),
        })
    }

    fn to_json(&self) -> String {
        format!(
            concat!(
                "    {{\n",
                "      \"label\": {},\n",
                "      \"repeats\": {},\n",
                "      \"output_throughput_tok_s\": {},\n",
                "      \"total_throughput_tok_s\": {},\n",
                "      \"mean_ttft_ms\": {},\n",
                "      \"median_ttft_ms\": {},\n",
                "      \"p99_ttft_ms\": {},\n",
                "      \"mean_submit_ttft_ms\": {},\n",
                "      \"median_submit_ttft_ms\": {},\n",
                "      \"p99_submit_ttft_ms\": {},\n",
                "      \"mean_tpot_ms\": {},\n",
                "      \"p99_tpot_ms\": {},\n",
                "      \"p99_itl_ms\": {}\n",
                "    }}"
            ),
            json_string(&self.label),
            self.repeats,
            self.output_throughput_tok_s.to_json(),
            self.total_throughput_tok_s.to_json(),
            self.mean_ttft_ms.to_json(),
            self.median_ttft_ms.to_json(),
            self.p99_ttft_ms.to_json(),
            self.mean_submit_ttft_ms.to_json(),
            self.median_submit_ttft_ms.to_json(),
            self.p99_submit_ttft_ms.to_json(),
            self.mean_tpot_ms.to_json(),
            self.p99_tpot_ms.to_json(),
            self.p99_itl_ms.to_json(),
        )
    }
}

#[derive(Debug, Clone)]
pub struct BenchmarkProvenance {
    pub git_commit: String,
    pub nvidia_driver_version: String,
    pub gpu_graphics_clocks_mhz: Vec<u32>,
    pub gpu_mem_clocks_mhz: Vec<u32>,
    pub seed: u64,
    pub repeats: usize,
    pub warmup_prompts: usize,
    pub max_step_tokens: u32,
    pub max_prefill_chunk: usize,
}

impl BenchmarkProvenance {
    pub fn capture(
        seed: u64,
        repeats: usize,
        warmup_prompts: usize,
        max_step_tokens: u32,
        max_prefill_chunk: usize,
    ) -> Result<Self> {
        let git_commit = zs_wire::resolve_git_sha(true).map_err(ModeBError::Protocol)?;

        let gpu_output = Command::new("nvidia-smi")
            .args([
                "--query-gpu=driver_version,clocks.current.graphics,clocks.current.memory",
                "--format=csv,noheader,nounits",
            ])
            .output()?;
        if !gpu_output.status.success() {
            return Err(ModeBError::Protocol(
                "nvidia-smi failed while capturing provenance".to_owned(),
            ));
        }
        let gpu_csv = String::from_utf8(gpu_output.stdout)
            .map_err(|error| ModeBError::Protocol(format!("invalid nvidia-smi output: {error}")))?;
        let mut driver_versions = Vec::new();
        let mut gpu_graphics_clocks_mhz = Vec::new();
        let mut gpu_mem_clocks_mhz = Vec::new();
        for line in gpu_csv.lines().filter(|line| !line.trim().is_empty()) {
            let fields: Vec<&str> = line.split(',').map(str::trim).collect();
            if fields.len() != 3 {
                return Err(ModeBError::Protocol(format!(
                    "unexpected nvidia-smi provenance row: {line}"
                )));
            }
            driver_versions.push(fields[0].to_owned());
            gpu_graphics_clocks_mhz.push(fields[1].parse::<u32>().map_err(|error| {
                ModeBError::Protocol(format!("invalid graphics clock in `{line}`: {error}"))
            })?);
            gpu_mem_clocks_mhz.push(fields[2].parse::<u32>().map_err(|error| {
                ModeBError::Protocol(format!("invalid memory clock in `{line}`: {error}"))
            })?);
        }
        let nvidia_driver_version = driver_versions.first().cloned().ok_or_else(|| {
            ModeBError::Protocol("nvidia-smi returned no GPU provenance rows".to_owned())
        })?;
        Ok(Self {
            git_commit,
            nvidia_driver_version,
            gpu_graphics_clocks_mhz,
            gpu_mem_clocks_mhz,
            seed,
            repeats,
            warmup_prompts,
            max_step_tokens,
            max_prefill_chunk,
        })
    }

    fn to_json(&self) -> String {
        let graphics_clocks = self
            .gpu_graphics_clocks_mhz
            .iter()
            .map(u32::to_string)
            .collect::<Vec<_>>()
            .join(", ");
        let mem_clocks = self
            .gpu_mem_clocks_mhz
            .iter()
            .map(u32::to_string)
            .collect::<Vec<_>>()
            .join(", ");
        format!(
            concat!(
                "  \"provenance\": {{\n",
                "    \"git_commit\": {},\n",
                "    \"nvidia_driver_version\": {},\n",
                "    \"gpu_graphics_clocks_mhz\": [{}],\n",
                "    \"gpu_mem_clocks_mhz\": [{}],\n",
                "    \"seed\": {},\n",
                "    \"repeats\": {},\n",
                "    \"warmup_prompts\": {},\n",
                "    \"max_step_tokens\": {},\n",
                "    \"max_prefill_chunk\": {}\n",
                "  }}"
            ),
            json_string(&self.git_commit),
            json_string(&self.nvidia_driver_version),
            graphics_clocks,
            mem_clocks,
            self.seed,
            self.repeats,
            self.warmup_prompts,
            self.max_step_tokens,
            self.max_prefill_chunk,
        )
    }
}

/// Complete multi-tier benchmark report produced by [`ModeBServingEngine`].
#[derive(Debug, Clone)]
pub struct ServingBenchmarkReport {
    pub model_name: String,
    pub world_size: u32,
    pub worker_numa_nodes: Vec<i32>,
    pub init_wall_ms: f64,
    pub tiers: Vec<BenchmarkTierResult>,
    pub tier_aggregates: Vec<TierAggregateStats>,
    pub provenance: BenchmarkProvenance,
}

impl ServingBenchmarkReport {
    pub fn to_json_pretty(&self) -> String {
        let numa_str = self
            .worker_numa_nodes
            .iter()
            .map(i32::to_string)
            .collect::<Vec<_>>()
            .join(", ");
        let tiers_json = self
            .tiers
            .iter()
            .map(BenchmarkTierResult::to_json)
            .collect::<Vec<_>>()
            .join(",\n");
        let aggregates_json = self
            .tier_aggregates
            .iter()
            .map(TierAggregateStats::to_json)
            .collect::<Vec<_>>()
            .join(",\n");
        format!(
            concat!(
                "{{\n",
                "  \"engine\": \"zstep-mode-b\",\n",
                "  \"model\": {},\n",
                "  \"world_size\": {},\n",
                "  \"worker_numa_nodes\": [{}],\n",
                "  \"init_wall_ms\": {:.2},\n",
                "{},\n",
                "  \"tiers\": [\n{}\n  ],\n",
                "  \"tier_aggregates\": [\n{}\n  ]\n",
                "}}\n"
            ),
            json_string(&self.model_name),
            self.world_size,
            numa_str,
            self.init_wall_ms,
            self.provenance.to_json(),
            tiers_json,
            aggregates_json,
        )
    }
}

fn json_string(value: &str) -> String {
    let mut out = String::with_capacity(value.len() + 2);
    out.push('"');
    for ch in value.chars() {
        match ch {
            '"' => out.push_str("\\\""),
            '\\' => out.push_str("\\\\"),
            '\u{08}' => out.push_str("\\b"),
            '\u{0c}' => out.push_str("\\f"),
            '\n' => out.push_str("\\n"),
            '\r' => out.push_str("\\r"),
            '\t' => out.push_str("\\t"),
            c if (c as u32) < 0x20 => {
                use std::fmt::Write as _;
                let _ = write!(out, "\\u{:04x}", c as u32);
            }
            c => out.push(c),
        }
    }
    out.push('"');
    out
}

struct PendingRequest {
    request_id: usize,
    prompt_tokens: Vec<i32>,
    target_output_len: usize,
    permit_time: Instant,
    submit_time: Instant,
}

struct ActiveSequence {
    request_id: usize,
    slot_lease: SlotLease,
    kv_leases: Vec<BlockLease>,
    page_ids: Vec<i32>,
    prompt_tokens: Vec<i32>,
    prefilled_tokens: usize,
    current_seq_len: usize,
    last_token: i32,
    generated_tokens: Vec<i32>,
    generated_top2: Vec<StepTop2Logits>,
    target_output_len: usize,
    permit_time: Instant,
    submit_time: Instant,
    first_token_time: Option<Instant>,
    last_token_time: Option<Instant>,
}

impl ActiveSequence {
    fn ensure_kv_pages(
        &mut self,
        target_seq_len: usize,
        page_block_tokens: usize,
        max_pages_per_seq: usize,
        kv_pool: &BlockPool,
        shadow_page_table: &mut [i32],
        dirty_pt_slots: &mut [u128; 2],
    ) -> Result<()> {
        let needed_pages = target_seq_len.div_ceil(page_block_tokens);
        if needed_pages > max_pages_per_seq {
            return Err(ModeBError::Protocol(format!(
                "sequence length {target_seq_len} exceeds max_pages_per_seq {max_pages_per_seq}"
            )));
        }
        if needed_pages > self.page_ids.len() {
            let delta = needed_pages - self.page_ids.len();
            let lease = kv_pool.allocate(delta).ok_or_else(|| {
                ModeBError::Protocol(format!(
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
            self.kv_leases.push(lease);
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
}

#[derive(Debug, Clone, Default)]
pub struct GenerationScheduleStats {
    pub prefill_steps: u64,
    pub decode_steps: u64,
    pub chunked_prefill_steps: u64,
    pub decode_steps_with_partial_prefill: u64,
    pub max_active_sequences: usize,
}

/// Live Mode B continuous-batching engine keeping the 8 GPU workers warm across runs.
pub struct ModeBServingEngine {
    coordinator: ModeBCoordinator,
    slot_pool: SlotPool,
    kv_pool: BlockPool,
    shadow_page_table: Vec<i32>,
    dirty_pt_slots: [u128; 2],
    step_cursor: u64,
    init_wall_us: u64,
    worker_numa_nodes: Vec<i32>,
    max_mixed_step_tokens: Option<usize>,
    max_prefill_coalesce_wait_ms: u64,
    device_token_overlap: bool,
    poisoned: bool,
    last_generation_schedule: GenerationScheduleStats,
}

enum MixedStepOutcome {
    MixedOrPrefill { had_decodes: bool, had_chunk: bool },
    PureDecodeFallback,
}

fn validate_serving_max_seqs(max_seqs: u32) -> Result<()> {
    if max_seqs == 0 || max_seqs > MAX_DECODE_GRAPH_SEQS {
        return Err(ModeBError::Protocol(format!(
            "max_seqs={max_seqs} is outside supported decode CUDA Graph range 1..={MAX_DECODE_GRAPH_SEQS}"
        )));
    }
    Ok(())
}

/// Whether every active sequence is prompt-complete and still needs a decode step after the
/// step being planned, given `in_flight` earlier steps whose tokens are not harvested yet.
fn decode_continues_for_all(
    active: &[ActiveSequence],
    waiting_permitted: &VecDeque<PendingRequest>,
    in_flight: usize,
) -> bool {
    waiting_permitted.is_empty()
        && !active.is_empty()
        && active.iter().all(|seq| {
            seq.prefilled_tokens == seq.prompt_tokens.len()
                && seq.generated_tokens.len() + in_flight + 1 < seq.target_output_len
        })
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

impl ModeBServingEngine {
    pub fn launch(config: ModeBConfig) -> Result<Self> {
        let geometry = config.geometry;
        validate_serving_max_seqs(geometry.max_seqs)?;
        let num_kv_pages = config.launch.num_kv_pages as usize;
        let dummy_page = config.launch.num_kv_pages as i32;
        let mut coordinator = ModeBCoordinator::launch(config)?;
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
            shadow_page_table,
            dirty_pt_slots: [0, 0],
            step_cursor: 0,
            init_wall_us,
            worker_numa_nodes,
            max_mixed_step_tokens: None,
            max_prefill_coalesce_wait_ms: DEFAULT_PREFILL_COALESCE_WAIT_MS,
            device_token_overlap: false,
            poisoned: false,
            last_generation_schedule: GenerationScheduleStats::default(),
        })
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
            return Err(ModeBError::Protocol(
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

    /// Publish decode steps before the previous step's result is read.
    ///
    /// Workers must implement `launch`/`finish`, `STEP_FLAG_DEVICE_TOKENS`, and
    /// `STEP_FLAG_NEXT_PREPUBLISHED`, and must not rely on host-updated `engram_prev_window`
    /// during overlapped decode bursts (enable only for models without host Engram lookback).
    pub fn set_device_token_overlap(&mut self, enabled: bool) {
        self.device_token_overlap = enabled;
    }

    fn plan_flags(&mut self, step_id: u64) -> Result<u32> {
        Ok(self.coordinator.plan_slot_mut(step_id)?.header_mut()?.flags)
    }

    fn or_plan_flags(&mut self, step_id: u64, bits: u32) -> Result<()> {
        self.coordinator.plan_slot_mut(step_id)?.header_mut()?.flags |= bits;
        Ok(())
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
            return Err(ModeBError::Protocol(format!(
                "prompt count {} differs from output length count {}",
                prompts.len(),
                target_output_lengths.len()
            )));
        }
        if target_output_lengths.contains(&0) {
            return Err(ModeBError::Protocol(
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
            return Err(ModeBError::Protocol(
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
            return Err(ModeBError::Protocol(
                "serving engine is poisoned after a failed in-flight step".to_owned(),
            ));
        }
        self.poisoned = true;
        let geometry = self.coordinator.config().geometry;
        let max_seqs = geometry.max_seqs as usize;
        let max_step_tokens = geometry.max_step_tokens as usize;
        let max_prefill_chunk = self.coordinator.config().max_prefill_chunk;
        let page_block_tokens = geometry.page_block_tokens as usize;
        let max_pages_per_seq = geometry.max_pages_per_seq as usize;

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
                self.step_cursor += 1;
                let sid = self.step_cursor;
                let decode_active_count = active
                    .iter()
                    .filter(|seq| seq.prefilled_tokens == seq.prompt_tokens.len())
                    .count();
                let has_partial_prefill = decode_active_count != active.len();
                if let Some(mixed_cap) = self.max_mixed_step_tokens {
                    let can_attempt_mixed = has_partial_prefill
                        || (active.len() < max_seqs && !waiting_permitted.is_empty());
                    if can_attempt_mixed {
                        match self.stage_mixed_step(
                            sid,
                            mixed_cap,
                            &mut active,
                            &mut slot_to_active,
                            waiting_permitted,
                            page_block_tokens,
                            max_pages_per_seq,
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
                    let run_prefill = should_schedule_prefill(
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
                            seq.prefilled_tokens > 0
                                && seq.prefilled_tokens < seq.prompt_tokens.len()
                        });
                        self.stage_prefill_step(
                            sid,
                            &mut active,
                            &mut slot_to_active,
                            waiting_permitted,
                            page_block_tokens,
                            max_pages_per_seq,
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

            let publish_next = match ahead_promise {
                Some(promise) => promise,
                None => {
                    let promise = self.device_token_overlap
                        && self.plan_flags(step_id)? & STEP_FLAG_DECODE_ONLY != 0
                        && decode_continues_for_all(&active, waiting_permitted, 0);
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
                let promise = decode_continues_for_all(&active, waiting_permitted, 1);
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
                && waiting_permitted.is_empty()
                && !active.is_empty()
                && active.iter().all(|seq| {
                    seq.prefilled_tokens == seq.prompt_tokens.len()
                        && seq.generated_tokens.len() + 1 < seq.target_output_len
                });

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

            // Harvest accepted tokens and Top-2 logits from rank 0's ResultSlot
            let emitted: Vec<(u32, i32, i32, u32, u32)> = {
                let result = self.coordinator.result_slot(step_id)?;
                if result.completed_step_id()? != step_id {
                    return Err(ModeBError::Protocol(format!(
                        "completed_step_id mismatch on step {step_id}"
                    )));
                }
                let header = result.header()?;
                if header.status_code != 0 {
                    return Err(ModeBError::Protocol(format!(
                        "worker failed step {step_id} with status {}",
                        header.status_code
                    )));
                }
                let outputs = result.outputs()?;
                outputs[..header.num_seq_results as usize]
                    .iter()
                    .map(|o| {
                        (
                            o.req_slot,
                            o.accepted_tokens[0],
                            o.accepted_tokens[1],
                            o.accepted_logprobs_bits[0],
                            o.accepted_logprobs_bits[1],
                        )
                    })
                    .collect()
            };

            let collect_top2 = completed_top2.is_some();
            for (req_slot, tok, tok2, lp1_bits, lp2_bits) in emitted {
                let idx = *slot_to_active.get(req_slot as usize).unwrap_or(&usize::MAX);
                if idx < active.len() {
                    let seq = &mut active[idx];
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
                        let top1_logit = f32::from_bits(lp1_bits);
                        let top2_logit = f32::from_bits(lp2_bits);
                        seq.generated_top2.push(StepTop2Logits {
                            top1_token: tok,
                            top1_logit,
                            top2_token: tok2,
                            top2_logit,
                            top1_minus_top2_margin: top1_logit - top2_logit,
                        });
                    }
                }
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
                    let finished = active.swap_remove(idx);
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
                    completed[finished.request_id] = finished.generated_tokens;
                    if let Some(top2_out) = completed_top2.as_deref_mut() {
                        top2_out[finished.request_id] = finished.generated_top2;
                    }
                    drop(finished.kv_leases);
                    drop(finished.slot_lease);

                    // Release concurrency permit immediately to next waiting request
                    if let Some((req_id, prompt, target_out, submit_time)) =
                        unpermitted_pool.pop_front()
                    {
                        waiting_permitted.push_back(PendingRequest {
                            request_id: req_id,
                            prompt_tokens: prompt,
                            target_output_len: target_out,
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

    fn stage_prefill_step(
        &mut self,
        step_id: u64,
        active: &mut Vec<ActiveSequence>,
        slot_to_active: &mut [usize],
        waiting_permitted: &mut VecDeque<PendingRequest>,
        page_block_tokens: usize,
        max_pages_per_seq: usize,
    ) -> Result<()> {
        let geometry = self.coordinator.config().geometry;
        let max_seqs = geometry.max_seqs as usize;
        let max_step_tokens = geometry.max_step_tokens as usize;
        let max_prefill_chunk = self.coordinator.config().max_prefill_chunk;

        let mut token_cursor = 0usize;
        let mut num_seqs = 0usize;
        let mut num_sample_rows = 0usize;

        for seq in active.iter_mut() {
            let Some((chunk_start, chunk_end)) = prefill_chunk_range(
                seq.prompt_tokens.len(),
                seq.prefilled_tokens,
                max_step_tokens - token_cursor,
                max_prefill_chunk,
            ) else {
                continue;
            };
            let sample_row_index = (chunk_end == seq.prompt_tokens.len()).then(|| {
                let row = num_sample_rows;
                num_sample_rows += 1;
                row
            });
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
            let Some(front) = waiting_permitted.front() else {
                break;
            };
            let prompt_len = front.prompt_tokens.len();
            if prompt_len == 0 {
                return Err(ModeBError::Protocol(
                    "prompt length must be positive".to_owned(),
                ));
            }
            let Some((chunk_start, chunk_end)) = prefill_chunk_range(
                prompt_len,
                0,
                max_step_tokens - token_cursor,
                max_prefill_chunk,
            ) else {
                break;
            };
            let needed_pages = chunk_end.div_ceil(page_block_tokens);
            if self.kv_pool.available() < needed_pages || self.slot_pool.available() == 0 {
                break;
            }
            let req = waiting_permitted.pop_front().ok_or_else(|| {
                ModeBError::Protocol("waiting queue unexpectedly empty".to_owned())
            })?;
            let slot_lease = self.slot_pool.allocate().ok_or_else(|| {
                ModeBError::Protocol("SlotPool unexpectedly exhausted".to_owned())
            })?;
            let slot_id = slot_lease.id() as usize;
            let last_token = *req.prompt_tokens.last().unwrap_or(&0);
            let mut seq = ActiveSequence {
                request_id: req.request_id,
                slot_lease,
                kv_leases: Vec::new(),
                page_ids: Vec::new(),
                prompt_tokens: req.prompt_tokens,
                prefilled_tokens: 0,
                current_seq_len: 0,
                last_token,
                generated_tokens: Vec::with_capacity(req.target_output_len),
                generated_top2: Vec::new(),
                target_output_len: req.target_output_len,
                permit_time: req.permit_time,
                submit_time: req.submit_time,
                first_token_time: None,
                last_token_time: None,
            };
            let sample_row_index = (chunk_end == prompt_len).then(|| {
                let row = num_sample_rows;
                num_sample_rows += 1;
                row
            });
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
            return Err(ModeBError::Protocol(
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
        *plan.header_mut()? = StepHeader {
            step_id: AtomicU64::new(0),
            opcode: StepOpcode::NormalStep as u32,
            num_seqs: num_seqs as u32,
            num_encoder_tokens: token_cursor as u32,
            encoder_graph_bucket: 0,
            num_decoder_rows: token_cursor as u32,
            decoder_graph_bucket: 0,
            num_sample_rows: num_sample_rows as u32,
            is_dspark_verify: 0,
            flags: 0,
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
    ) -> Result<MixedStepOutcome> {
        let geometry = self.coordinator.config().geometry;
        let max_seqs = geometry.max_seqs as usize;
        let max_step_tokens = geometry.max_step_tokens as usize;
        let max_prefill_chunk = self.coordinator.config().max_prefill_chunk;

        let decode_active_count = active
            .iter()
            .filter(|seq| seq.prefilled_tokens == seq.prompt_tokens.len())
            .count();
        let continued_chunk = active
            .iter()
            .any(|seq| seq.prefilled_tokens > 0 && seq.prefilled_tokens < seq.prompt_tokens.len());

        let step_budget = if decode_active_count > 0 {
            mixed_cap.max(decode_active_count + 1).min(max_step_tokens)
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
            seq.ensure_kv_pages(
                next_seq_len,
                page_block_tokens,
                max_pages_per_seq,
                &self.kv_pool,
                &mut self.shadow_page_table,
                &mut self.dirty_pt_slots,
            )?;
        }

        let mut token_cursor = 0usize;
        let mut num_seqs = 0usize;
        let mut num_sample_rows = 0usize;

        if decode_active_count > 0 {
            let mut staged_tokens = Vec::with_capacity(decode_active_count);
            let mut staged_positions = Vec::with_capacity(decode_active_count);
            let mut staged_slots = Vec::with_capacity(decode_active_count);
            let mut staged_lookbacks = Vec::with_capacity(decode_active_count * 3);
            let mut staged_seq_lens = Vec::with_capacity(decode_active_count);
            for seq in active
                .iter_mut()
                .filter(|seq| seq.prefilled_tokens == seq.prompt_tokens.len())
            {
                let start_pos = seq.current_seq_len;
                let lookback = seq.lookback_window(start_pos);
                let pos = start_pos as i32;
                let next_len = pos + 1;
                staged_tokens.push(seq.last_token);
                staged_positions.push(pos);
                staged_slots.push(seq.slot_lease.id() as i32);
                staged_lookbacks.extend_from_slice(&lookback);
                staged_seq_lens.push(next_len);
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
        for seq in active.iter_mut() {
            if token_cursor >= step_budget {
                break;
            }
            if seq.prefilled_tokens >= seq.prompt_tokens.len() {
                continue;
            }
            let prompt_len = seq.prompt_tokens.len();
            let prefilled = seq.prefilled_tokens;
            let Some((chunk_start, chunk_end)) = prefill_chunk_range(
                prompt_len,
                prefilled,
                step_budget - token_cursor,
                max_prefill_chunk,
            ) else {
                continue;
            };
            let needed_pages = chunk_end.div_ceil(page_block_tokens);
            let extra_pages = needed_pages.saturating_sub(seq.page_ids.len());
            if extra_pages > self.kv_pool.available() {
                continue;
            }
            let sample_row_index = (chunk_end == prompt_len).then(|| {
                let row = num_sample_rows;
                num_sample_rows += 1;
                row
            });
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

        // Pass 3 (Region 3 / 1): Admit new waiting requests (`first_pos == 0`).
        while active.len() < max_seqs && token_cursor < step_budget {
            let Some(front) = waiting_permitted.front() else {
                break;
            };
            let prompt_len = front.prompt_tokens.len();
            if prompt_len == 0 {
                return Err(ModeBError::Protocol(
                    "prompt length must be positive".to_owned(),
                ));
            }
            let remaining_budget = step_budget - token_cursor;
            if num_seqs > 0 && remaining_budget < 16 && prompt_len > remaining_budget {
                break;
            }
            let Some((chunk_start, chunk_end)) =
                prefill_chunk_range(prompt_len, 0, remaining_budget, max_prefill_chunk)
            else {
                break;
            };
            let needed_pages = chunk_end.div_ceil(page_block_tokens);
            if self.kv_pool.available() < needed_pages || self.slot_pool.available() == 0 {
                break;
            }
            let req = waiting_permitted.pop_front().ok_or_else(|| {
                ModeBError::Protocol("waiting queue unexpectedly empty".to_owned())
            })?;
            let slot_lease = self.slot_pool.allocate().ok_or_else(|| {
                ModeBError::Protocol("SlotPool unexpectedly exhausted".to_owned())
            })?;
            let slot_id = slot_lease.id() as usize;
            let last_token = *req.prompt_tokens.last().unwrap_or(&0);
            let mut seq = ActiveSequence {
                request_id: req.request_id,
                slot_lease,
                kv_leases: Vec::new(),
                page_ids: Vec::new(),
                prompt_tokens: req.prompt_tokens,
                prefilled_tokens: 0,
                current_seq_len: 0,
                last_token,
                generated_tokens: Vec::with_capacity(req.target_output_len),
                generated_top2: Vec::new(),
                target_output_len: req.target_output_len,
                permit_time: req.permit_time,
                submit_time: req.submit_time,
                first_token_time: None,
                last_token_time: None,
            };
            let sample_row_index = (chunk_end == prompt_len).then(|| {
                let row = num_sample_rows;
                num_sample_rows += 1;
                row
            });
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
            return Err(ModeBError::Protocol(
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
        let flags = if is_pure_decode_fallback {
            STEP_FLAG_DECODE_ONLY
        } else {
            0
        };

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
            is_dspark_verify: 0,
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
        seq.ensure_kv_pages(
            chunk_end,
            page_block_tokens,
            max_pages_per_seq,
            &self.kv_pool,
            &mut self.shadow_page_table,
            &mut self.dirty_pt_slots,
        )?;
        let chunk_len = chunk_end - chunk_start;
        let slot_i32 = seq.slot_lease.id() as i32;
        let lookback = seq.lookback_window(chunk_start);
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
        if let Some(sample_row) = sample_row_index {
            plan.sample_row_indices_mut()?[sample_row] = (token_cursor + chunk_len - 1) as i32;
        }
        seq.prefilled_tokens = chunk_end;
        seq.current_seq_len = chunk_end;
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
        let num_seqs = active
            .iter()
            .filter(|seq| seq.prefilled_tokens == seq.prompt_tokens.len())
            .count();
        if num_seqs == 0 {
            return Err(ModeBError::Protocol(
                "decode step scheduled zero prompt-complete sequences".to_owned(),
            ));
        }

        let bucket = select_graph_bucket(num_seqs as u32, &DECODE_GRAPH_BUCKETS);
        if bucket == 0 {
            return Err(ModeBError::Protocol(format!(
                "decode batch size {num_seqs} exceeds max decode CUDA Graph bucket ({MAX_DECODE_GRAPH_SEQS})"
            )));
        }

        for seq in active
            .iter_mut()
            .filter(|seq| seq.prefilled_tokens == seq.prompt_tokens.len())
        {
            let next_seq_len = seq.current_seq_len + 1;
            seq.ensure_kv_pages(
                next_seq_len,
                page_block_tokens,
                max_pages_per_seq,
                &self.kv_pool,
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
        }

        let mut plan = self.coordinator.plan_slot_mut(step_id)?;
        if commit_tokens_immediately {
            plan.token_ids_mut()?[..num_seqs].copy_from_slice(&staged_tokens);
        }
        plan.positions_mut()?[..num_seqs].copy_from_slice(&staged_positions);
        plan.req_slot_ids_mut()?[..num_seqs].copy_from_slice(&staged_slots);
        plan.engram_prev_window_mut()?[..num_seqs * 3].copy_from_slice(&staged_lookbacks);
        plan.seq_lens_mut()?[..num_seqs].copy_from_slice(&staged_seq_lens);
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
            flags: STEP_FLAG_DECODE_ONLY,
            _reserved: [0; 3],
        };
        Ok(())
    }

    pub fn shutdown(mut self) -> Result<()> {
        if self.poisoned {
            return Err(ModeBError::Protocol(
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
            return Err(ModeBError::Protocol(
                "cannot gracefully shut down engine after failed in-flight step; cluster aborted"
                    .to_owned(),
            ));
        }
        self.step_cursor += 1;
        let stop_step = self.step_cursor;
        self.coordinator.shutdown_and_wait_with_stats(stop_step)
    }
}

fn prefill_chunk_range(
    prompt_len: usize,
    prefilled_tokens: usize,
    token_budget: usize,
    max_prefill_chunk: usize,
) -> Option<(usize, usize)> {
    if prefilled_tokens >= prompt_len || token_budget == 0 || max_prefill_chunk == 0 {
        return None;
    }
    let chunk_len = (prompt_len - prefilled_tokens)
        .min(token_budget)
        .min(max_prefill_chunk);
    Some((prefilled_tokens, prefilled_tokens + chunk_len))
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

fn splitmix64(mut x: u64) -> u64 {
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

fn mean_us_as_ms(samples: &[u64]) -> f64 {
    if samples.is_empty() {
        return 0.0;
    }
    let sum: u128 = samples.iter().map(|&v| u128::from(v)).sum();
    (sum as f64 / samples.len() as f64) / 1000.0
}

fn percentile_us_as_ms(samples: &[u64], pct: usize) -> f64 {
    if samples.is_empty() {
        return 0.0;
    }
    let mut sorted = samples.to_vec();
    sorted.sort_unstable();
    let idx = (sorted.len() - 1) * pct / 100;
    sorted[idx] as f64 / 1000.0
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn slot_pool_raii_returns_slots_on_drop_and_unwind() {
        let pool = SlotPool::new(4);
        assert_eq!(pool.available(), 4);
        let s0 = pool.allocate().expect("slot 0");
        let s1 = pool.allocate().expect("slot 1");
        assert_eq!(s0.id(), 0);
        assert_eq!(s1.id(), 1);
        assert_eq!(pool.available(), 2);
        drop(s0);
        assert_eq!(pool.available(), 3);

        let pool_clone = pool.clone();
        let unwind_res = std::panic::catch_unwind(move || {
            let _s_panic = pool_clone.allocate().expect("panic slot");
            assert_eq!(pool_clone.available(), 2);
            panic!("simulated panic while holding SlotLease");
        });
        assert!(unwind_res.is_err());
        assert_eq!(pool.available(), 3);
        drop(s1);
        assert_eq!(pool.available(), 4);
    }

    #[test]
    fn active_sequence_kv_page_growth_and_raii_release() {
        let slot_pool = SlotPool::new(2);
        let kv_pool = BlockPool::new(CacheGroupKind::GlobalAndIndex, 8);
        let mut shadow_pt = vec![-1i32; 2 * 4];

        {
            let mut seq = ActiveSequence {
                request_id: 0,
                slot_lease: slot_pool.allocate().expect("slot"),
                kv_leases: Vec::new(),
                page_ids: Vec::new(),
                prompt_tokens: vec![42],
                prefilled_tokens: 1,
                current_seq_len: 0,
                last_token: 42,
                generated_tokens: Vec::new(),
                generated_top2: Vec::new(),
                target_output_len: 10,
                permit_time: Instant::now(),
                submit_time: Instant::now(),
                first_token_time: None,
                last_token_time: None,
            };
            let mut dirty_pt_slots = [0u128; 2];
            seq.ensure_kv_pages(17, 16, 4, &kv_pool, &mut shadow_pt, &mut dirty_pt_slots)
                .expect("allocate 2 pages");
            assert_eq!(seq.page_ids.len(), 2);
            assert_eq!(kv_pool.available(), 6);
            assert_eq!(shadow_pt[0], seq.page_ids[0]);
            assert_eq!(shadow_pt[1], seq.page_ids[1]);
            assert_eq!(dirty_pt_slots, [1, 1]);

            dirty_pt_slots = [0, 0];
            // Growing from 17 -> 32 stays within the existing 2 pages (2 * 16 = 32)
            seq.ensure_kv_pages(32, 16, 4, &kv_pool, &mut shadow_pt, &mut dirty_pt_slots)
                .expect("still 2 pages");
            assert_eq!(kv_pool.available(), 6);
            assert_eq!(dirty_pt_slots, [0, 0]);

            // Token 33 crosses boundary and allocates page 3
            seq.ensure_kv_pages(33, 16, 4, &kv_pool, &mut shadow_pt, &mut dirty_pt_slots)
                .expect("allocate 3rd page");
            assert_eq!(seq.page_ids.len(), 3);
            assert_eq!(kv_pool.available(), 5);
            assert_eq!(dirty_pt_slots, [1, 1]);
        }

        // Dropping `seq` releases all 3 KV blocks and the sequence slot
        assert_eq!(kv_pool.available(), 8);
        assert_eq!(slot_pool.available(), 2);
    }

    #[test]
    fn max_seqs_above_decode_graph_limit_is_rejected() {
        assert!(validate_serving_max_seqs(0).is_err());
        assert!(validate_serving_max_seqs(1).is_ok());
        assert!(validate_serving_max_seqs(128).is_ok());
        assert!(validate_serving_max_seqs(129).is_err());
        assert!(validate_serving_max_seqs(256).is_err());
    }

    #[test]
    fn json_string_escapes_rfc8259_and_preserves_utf8() {
        assert_eq!(
            json_string("DeepSeek-V4.1 \"中文\" \\ \n\t\x01\x1f"),
            "\"DeepSeek-V4.1 \\\"中文\\\" \\\\ \\n\\t\\u0001\\u001f\""
        );
    }

    #[test]
    fn overlapped_decode_lookback_includes_token_from_inflight_step() {
        let slot_pool = SlotPool::new(1);
        let now = Instant::now();
        let mut seq = ActiveSequence {
            request_id: 0,
            slot_lease: slot_pool.allocate().expect("slot"),
            kv_leases: Vec::new(),
            page_ids: Vec::new(),
            prompt_tokens: vec![10, 11, 12, 13],
            prefilled_tokens: 4,
            current_seq_len: 4,
            last_token: 13,
            generated_tokens: Vec::new(),
            generated_top2: Vec::new(),
            target_output_len: 4,
            permit_time: now,
            submit_time: now,
            first_token_time: None,
            last_token_time: None,
        };

        // Commit-true staging of the first decode, matching `stage_decode_step`.
        let inflight_pos = seq.current_seq_len;
        assert_eq!(seq.lookback_window(inflight_pos), [11, 12, 13]);
        seq.current_seq_len = inflight_pos + 1;

        // Pre-stage of the next step runs before harvest, so index 4 is absent.
        let stale = seq.lookback_window(seq.current_seq_len);
        assert_eq!(stale, [12, 13, -1]);

        seq.last_token = 99;
        seq.generated_tokens.push(99);
        let (token, repaired) = seq.revise_overlapped_decode();
        assert_eq!(token, 99);
        assert_eq!(repaired, [12, 13, 99]);
        assert_eq!(seq.current_seq_len, 6);
    }

    #[test]
    fn chunked_prefill_progresses_to_exact_prompt_end_and_samples_only_final_chunk() {
        let prompt_len = 10;
        let mut prefilled_tokens = 0;
        let mut ranges = Vec::new();
        while let Some((start, end)) = prefill_chunk_range(prompt_len, prefilled_tokens, 6, 4) {
            assert!(end - start <= 4);
            ranges.push((start, end));
            prefilled_tokens = end;
        }
        assert_eq!(ranges, vec![(0, 4), (4, 8), (8, 10)]);
        assert_eq!(prefilled_tokens, prompt_len);
        let sample_chunks = ranges.iter().filter(|(_, end)| *end == prompt_len).count();
        assert_eq!(sample_chunks, 1);
        assert_eq!(prefill_chunk_range(10, 4, 2, 4), Some((4, 6)));
        assert_eq!(prefill_chunk_range(10, 10, 6, 4), None);
    }

    #[test]
    fn decode_sequences_force_single_prefill_interleaving() {
        let now = Instant::now();
        let waiting: VecDeque<PendingRequest> = (0..4)
            .map(|request_id| PendingRequest {
                request_id,
                prompt_tokens: vec![256; 1024],
                target_output_len: 8,
                permit_time: now,
                submit_time: now,
            })
            .collect();
        assert!(should_schedule_prefill(
            1,
            1,
            false,
            8,
            4096,
            2048,
            &waiting,
            0,
            DEFAULT_PREFILL_COALESCE_WAIT_MS,
        ));
        assert!(!should_schedule_prefill(
            1,
            1,
            false,
            8,
            4096,
            2048,
            &waiting,
            1,
            DEFAULT_PREFILL_COALESCE_WAIT_MS,
        ));
        assert!(should_schedule_prefill(
            1,
            2,
            true,
            8,
            4096,
            2048,
            &waiting,
            0,
            DEFAULT_PREFILL_COALESCE_WAIT_MS,
        ));
        assert!(!should_schedule_prefill(
            1,
            2,
            true,
            8,
            4096,
            2048,
            &waiting,
            1,
            DEFAULT_PREFILL_COALESCE_WAIT_MS,
        ));
        assert!(should_schedule_prefill(
            0,
            1,
            false,
            8,
            4096,
            2048,
            &waiting,
            4,
            DEFAULT_PREFILL_COALESCE_WAIT_MS,
        ));
    }

    #[test]
    fn ttft_and_submit_ttft_use_their_respective_start_times() {
        let submit_time = Instant::now();
        let permit_time = submit_time + std::time::Duration::from_millis(40);
        let first_token_time = submit_time + std::time::Duration::from_millis(100);
        let (ttft_us, submit_ttft_us) =
            first_token_latency_us(permit_time, submit_time, first_token_time);
        assert_eq!(ttft_us, 60_000);
        assert_eq!(submit_ttft_us, 100_000);
    }

    #[test]
    fn tier_aggregates_include_permit_and_submit_ttft_population_stats() {
        let results = (0..3)
            .map(|repeat| BenchmarkTierResult {
                label: "tier".to_owned(),
                repeat,
                seed: BENCHMARK_RANDOM_SEED + repeat as u64,
                concurrency: 4,
                num_prompts: 12,
                input_len: 512,
                output_len: 256,
                steps: 100,
                overlapped_decode_steps: 90,
                wall_time_ms: 1_000.0,
                output_throughput_tok_s: 100.0,
                total_throughput_tok_s: 300.0,
                mean_ttft_ms: 40.0,
                median_ttft_ms: 40.0,
                p99_ttft_ms: 45.0,
                mean_submit_ttft_ms: 100.0 * (repeat + 1) as f64,
                median_submit_ttft_ms: 100.0 * (repeat + 1) as f64,
                p99_submit_ttft_ms: 110.0 * (repeat + 1) as f64,
                mean_tpot_ms: 4.0,
                median_tpot_ms: 4.0,
                p99_tpot_ms: 5.0,
                mean_itl_ms: 4.0,
                median_itl_ms: 4.0,
                p99_itl_ms: 6.0,
            })
            .collect::<Vec<_>>();
        let aggregate = TierAggregateStats::from_results("tier".to_owned(), &results)
            .expect("three repeat samples");
        assert_eq!(aggregate.repeats, 3);
        assert_eq!(aggregate.mean_ttft_ms.mean, 40.0);
        assert_eq!(aggregate.mean_submit_ttft_ms.mean, 200.0);
        assert!((aggregate.mean_submit_ttft_ms.std - 81.649_658).abs() < 1e-6);
        assert_eq!(aggregate.mean_submit_ttft_ms.min, 100.0);
        assert_eq!(aggregate.mean_submit_ttft_ms.max, 300.0);
        let encoded = aggregate.to_json();
        assert!(encoded.contains("\"median_submit_ttft_ms\""));
        assert!(encoded.contains("\"p99_submit_ttft_ms\""));
    }

    #[test]
    fn random_prompts_are_distinct_and_in_valid_vocab_range() {
        let prompts = build_deterministic_random_prompts(16, 64);
        assert_eq!(prompts.len(), 16);
        assert_ne!(prompts[0], prompts[1]);
        for p in &prompts {
            assert_eq!(p.len(), 64);
            assert!(p.iter().all(|&tok| (256..65_256).contains(&tok)));
        }

        let round0 = build_deterministic_random_prompts_seeded(8, 64, BENCHMARK_RANDOM_SEED);
        let round1 =
            build_deterministic_random_prompts_seeded(8, 64, BENCHMARK_RANDOM_SEED + 1_000);
        let round0_again = build_deterministic_random_prompts_seeded(8, 64, BENCHMARK_RANDOM_SEED);
        assert_eq!(round0, round0_again);
        for (p0, p1) in round0.iter().zip(&round1) {
            assert_ne!(p0, p1);
        }
    }

    #[test]
    fn engram_lookback_window_spans_prompt_boundary_and_generated_tokens() {
        let slot_pool = SlotPool::new(1);
        let mut seq = ActiveSequence {
            request_id: 0,
            slot_lease: slot_pool.allocate().expect("slot"),
            kv_leases: Vec::new(),
            page_ids: Vec::new(),
            prompt_tokens: vec![101, 102],
            prefilled_tokens: 2,
            current_seq_len: 2,
            last_token: 201,
            generated_tokens: vec![201, 202],
            generated_top2: Vec::new(),
            target_output_len: 8,
            permit_time: Instant::now(),
            submit_time: Instant::now(),
            first_token_time: None,
            last_token_time: None,
        };
        // Initial prefill start (`chunk_start = 0`)
        assert_eq!(seq.lookback_window(0), [-1, -1, -1]);
        // Chunked prefill at offset 1 (`chunk_start = 1`)
        assert_eq!(seq.lookback_window(1), [-1, -1, 101]);
        // Decode step 1 (`pos = 2`, feeding `201`)
        assert_eq!(seq.lookback_window(2), [-1, 101, 102]);
        // Overlapped decode step 2 (`pos = 3`, feeding `202`)
        assert_eq!(seq.lookback_window(3), [101, 102, 201]);
        // Steady-state decode step 3 (`pos = 4`)
        seq.generated_tokens.push(203);
        assert_eq!(seq.lookback_window(4), [102, 201, 202]);
        assert_eq!(seq.lookback_window(5), [201, 202, 203]);
    }

    #[test]
    fn decode_continues_for_all_respects_inflight_prefill_and_waiting() {
        let slot_pool = SlotPool::new(2);
        let now = Instant::now();
        let make_seq = |slot_pool: &SlotPool, prefilled: usize, generated: usize, target: usize| {
            ActiveSequence {
                request_id: 0,
                slot_lease: slot_pool.allocate().expect("slot"),
                kv_leases: Vec::new(),
                page_ids: Vec::new(),
                prompt_tokens: vec![10, 20, 30, 40],
                prefilled_tokens: prefilled,
                current_seq_len: prefilled + generated.saturating_sub(1),
                last_token: 99,
                generated_tokens: vec![99; generated],
                generated_top2: Vec::new(),
                target_output_len: target,
                permit_time: now,
                submit_time: now,
                first_token_time: None,
                last_token_time: None,
            }
        };

        let empty_waiting = VecDeque::new();
        assert!(!decode_continues_for_all(&[], &empty_waiting, 0));

        // 1 generated out of 4: in_flight=0 -> 1+0+1=2 < 4 (true); in_flight=1 -> 1+1+1=3 < 4 (true);
        // in_flight=2 -> 1+2+1=4 < 4 (false, retires right after planned step).
        let active = vec![make_seq(&slot_pool, 4, 1, 4)];
        assert!(decode_continues_for_all(&active, &empty_waiting, 0));
        assert!(decode_continues_for_all(&active, &empty_waiting, 1));
        assert!(!decode_continues_for_all(&active, &empty_waiting, 2));

        // Waiting permitted request blocks overlap so prefill interleaving is not starved.
        let mut non_empty_waiting = VecDeque::new();
        non_empty_waiting.push_back(PendingRequest {
            request_id: 1,
            prompt_tokens: vec![1, 2],
            target_output_len: 4,
            permit_time: now,
            submit_time: now,
        });
        assert!(!decode_continues_for_all(&active, &non_empty_waiting, 0));
        drop(active);

        // Partial prefill sequence in `active` blocks decode-only pre-publishing.
        let partial = vec![make_seq(&slot_pool, 2, 0, 4)];
        assert!(!decode_continues_for_all(&partial, &empty_waiting, 0));
    }
}
