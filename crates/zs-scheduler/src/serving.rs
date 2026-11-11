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
use std::sync::atomic::AtomicU64;
use std::sync::{Arc, Mutex};
use std::time::Instant;

use zs_kv::{BlockLease, BlockPool, CacheGroupKind};
use zs_wire::{StepHeader, StepOpcode};

use crate::mode_b::{ModeBConfig, ModeBCoordinator, ModeBError, Result};
use crate::select_graph_bucket;

const DECODE_BUCKETS: [u32; 8] = [1, 2, 4, 8, 16, 32, 64, 128];

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

/// Per-tier continuous-batching throughput and latency metrics.
#[derive(Debug, Clone)]
pub struct BenchmarkTierResult {
    pub label: String,
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
    pub mean_tpot_ms: f64,
    pub median_tpot_ms: f64,
    pub p99_tpot_ms: f64,
    pub mean_itl_ms: f64,
    pub median_itl_ms: f64,
    pub p99_itl_ms: f64,
}

/// Complete multi-tier benchmark report produced by [`ModeBServingEngine`].
#[derive(Debug, Clone)]
pub struct ServingBenchmarkReport {
    pub model_name: String,
    pub world_size: u32,
    pub worker_numa_nodes: Vec<i32>,
    pub init_wall_ms: f64,
    pub tiers: Vec<BenchmarkTierResult>,
}

impl ServingBenchmarkReport {
    pub fn to_json_pretty(&self) -> String {
        let numa_str = self
            .worker_numa_nodes
            .iter()
            .map(i32::to_string)
            .collect::<Vec<_>>()
            .join(", ");
        let mut tiers_json = String::new();
        for (idx, t) in self.tiers.iter().enumerate() {
            if idx > 0 {
                tiers_json.push_str(",\n");
            }
            tiers_json.push_str(&format!(
                concat!(
                    "    {{\n",
                    "      \"label\": \"{}\",\n",
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
                    "      \"mean_tpot_ms\": {:.2},\n",
                    "      \"median_tpot_ms\": {:.2},\n",
                    "      \"p99_tpot_ms\": {:.2},\n",
                    "      \"mean_itl_ms\": {:.2},\n",
                    "      \"median_itl_ms\": {:.2},\n",
                    "      \"p99_itl_ms\": {:.2}\n",
                    "    }}"
                ),
                t.label,
                t.concurrency,
                t.num_prompts,
                t.input_len,
                t.output_len,
                t.steps,
                t.overlapped_decode_steps,
                t.wall_time_ms,
                t.output_throughput_tok_s,
                t.total_throughput_tok_s,
                t.mean_ttft_ms,
                t.median_ttft_ms,
                t.p99_ttft_ms,
                t.mean_tpot_ms,
                t.median_tpot_ms,
                t.p99_tpot_ms,
                t.mean_itl_ms,
                t.median_itl_ms,
                t.p99_itl_ms,
            ));
        }
        format!(
            concat!(
                "{{\n",
                "  \"engine\": \"zstep-mode-b\",\n",
                "  \"model\": \"{}\",\n",
                "  \"world_size\": {},\n",
                "  \"worker_numa_nodes\": [{}],\n",
                "  \"init_wall_ms\": {:.2},\n",
                "  \"tiers\": [\n{}\n  ]\n",
                "}}\n"
            ),
            self.model_name, self.world_size, numa_str, self.init_wall_ms, tiers_json
        )
    }
}

struct PendingRequest {
    request_id: usize,
    prompt_tokens: Vec<i32>,
    target_output_len: usize,
    arrival_time: Instant,
}

struct ActiveSequence {
    request_id: usize,
    slot_lease: SlotLease,
    kv_leases: Vec<BlockLease>,
    page_ids: Vec<i32>,
    current_seq_len: usize,
    last_token: i32,
    generated_tokens: Vec<i32>,
    target_output_len: usize,
    arrival_time: Instant,
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
            let slot_row_base = self.slot_lease.id() as usize * max_pages_per_seq;
            for &pid in lease.ids() {
                let page_col = self.page_ids.len();
                self.page_ids.push(pid);
                shadow_page_table[slot_row_base + page_col] = pid;
            }
            self.kv_leases.push(lease);
        }
        Ok(())
    }
}

/// Live Mode B continuous-batching engine keeping the 8 GPU workers warm across runs.
pub struct ModeBServingEngine {
    coordinator: ModeBCoordinator,
    slot_pool: SlotPool,
    kv_pool: BlockPool,
    shadow_page_table: Vec<i32>,
    step_cursor: u64,
    init_wall_us: u64,
    worker_numa_nodes: Vec<i32>,
}

impl ModeBServingEngine {
    pub fn launch(config: ModeBConfig) -> Result<Self> {
        let geometry = config.geometry;
        let num_kv_pages = config.launch.num_kv_pages as usize;
        let dummy_page = config.launch.num_kv_pages as i32;
        let mut coordinator = ModeBCoordinator::launch(config)?;
        let (init_wall_us, worker_numa_nodes) = coordinator.wait_workers_ready()?;

        let pt_len = geometry.max_seqs as usize * geometry.max_pages_per_seq as usize;
        let shadow_page_table = vec![dummy_page; pt_len];
        Ok(Self {
            coordinator,
            slot_pool: SlotPool::new(geometry.max_seqs),
            kv_pool: BlockPool::new(CacheGroupKind::GlobalAndIndex, num_kv_pages),
            shadow_page_table,
            step_cursor: 0,
            init_wall_us,
            worker_numa_nodes,
        })
    }

    pub fn worker_numa_nodes(&self) -> &[i32] {
        &self.worker_numa_nodes
    }

    pub fn init_wall_ms(&self) -> f64 {
        self.init_wall_us as f64 / 1000.0
    }

    /// Generate exact greedy completions for `prompts` (`target_output_len` new tokens each).
    pub fn generate_greedy(
        &mut self,
        prompts: &[Vec<i32>],
        target_output_len: usize,
    ) -> Result<Vec<Vec<i32>>> {
        let mut completed = vec![Vec::new(); prompts.len()];
        if prompts.is_empty() || target_output_len == 0 {
            return Ok(completed);
        }
        let now = Instant::now();
        let mut waiting: VecDeque<PendingRequest> = prompts
            .iter()
            .enumerate()
            .map(|(idx, p)| PendingRequest {
                request_id: idx,
                prompt_tokens: p.clone(),
                target_output_len,
                arrival_time: now,
            })
            .collect();
        let mut raw_prompts: VecDeque<(usize, Vec<i32>, usize)> = VecDeque::new();
        let mut ttfts = Vec::new();
        let mut tpots = Vec::new();
        let mut itls = Vec::new();
        let mut steps = 0u64;
        let mut overlapped = 0u64;

        self.drive_requests(
            prompts.len(),
            &mut waiting,
            &mut raw_prompts,
            &mut completed,
            &mut ttfts,
            &mut tpots,
            &mut itls,
            &mut steps,
            &mut overlapped,
        )?;
        Ok(completed)
    }

    /// Execute one continuous-batching benchmark tier with semaphore `spec.concurrency`.
    pub fn run_benchmark_tier(&mut self, spec: &BenchmarkTierSpec) -> Result<BenchmarkTierResult> {
        if spec.concurrency == 0
            || spec.num_prompts == 0
            || spec.input_len == 0
            || spec.output_len == 0
        {
            return Err(ModeBError::Protocol(
                "benchmark tier dimensions must all be positive".to_owned(),
            ));
        }
        let all_prompts = build_deterministic_random_prompts(spec.num_prompts, spec.input_len);
        let initial_admit = spec.concurrency.min(spec.num_prompts);

        let mut completed = vec![Vec::new(); spec.num_prompts];
        let mut ttfts_us = Vec::with_capacity(spec.num_prompts);
        let mut tpots_us = Vec::with_capacity(spec.num_prompts);
        let mut itls_us = Vec::with_capacity(spec.num_prompts * spec.output_len);
        let mut steps = 0u64;
        let mut overlapped_decode_steps = 0u64;

        let wall_start = Instant::now();
        let mut waiting_permitted: VecDeque<PendingRequest> =
            VecDeque::with_capacity(spec.concurrency);
        let mut unpermitted_pool: VecDeque<(usize, Vec<i32>, usize)> =
            VecDeque::with_capacity(spec.num_prompts.saturating_sub(initial_admit));

        for (req_id, prompt) in all_prompts.into_iter().enumerate() {
            if req_id < initial_admit {
                waiting_permitted.push_back(PendingRequest {
                    request_id: req_id,
                    prompt_tokens: prompt,
                    target_output_len: spec.output_len,
                    arrival_time: wall_start,
                });
            } else {
                unpermitted_pool.push_back((req_id, prompt, spec.output_len));
            }
        }

        self.drive_requests(
            spec.concurrency,
            &mut waiting_permitted,
            &mut unpermitted_pool,
            &mut completed,
            &mut ttfts_us,
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
        unpermitted_pool: &mut VecDeque<(usize, Vec<i32>, usize)>,
        completed: &mut [Vec<i32>],
        ttfts_us: &mut Vec<u64>,
        tpots_us: &mut Vec<u64>,
        itls_us: &mut Vec<u64>,
        steps_out: &mut u64,
        overlapped_out: &mut u64,
    ) -> Result<()> {
        let geometry = self.coordinator.config().geometry;
        let max_seqs = geometry.max_seqs as usize;
        let max_step_tokens = geometry.max_step_tokens as usize;
        let page_block_tokens = geometry.page_block_tokens as usize;
        let max_pages_per_seq = geometry.max_pages_per_seq as usize;

        let mut active: Vec<ActiveSequence> = Vec::with_capacity(max_seqs.min(concurrency_cap));
        let mut slot_to_active: Vec<usize> = vec![usize::MAX; max_seqs];
        let mut consecutive_prefills = 0usize;
        let mut prestaged_step_id: Option<u64> = None;

        while !active.is_empty() || !waiting_permitted.is_empty() || !unpermitted_pool.is_empty() {
            let step_id = if let Some(ready_step) = prestaged_step_id.take() {
                *overlapped_out += 1;
                ready_step
            } else {
                self.step_cursor += 1;
                let sid = self.step_cursor;
                let run_prefill = should_schedule_prefill(
                    active.len(),
                    max_seqs,
                    max_step_tokens,
                    waiting_permitted,
                    consecutive_prefills,
                );
                if run_prefill {
                    self.stage_prefill_step(
                        sid,
                        &mut active,
                        &mut slot_to_active,
                        waiting_permitted,
                        page_block_tokens,
                        max_pages_per_seq,
                    )?;
                    consecutive_prefills += 1;
                } else {
                    self.stage_decode_step(
                        sid,
                        &mut active,
                        page_block_tokens,
                        max_pages_per_seq,
                        true,
                    )?;
                    consecutive_prefills = 0;
                }
                sid
            };

            self.coordinator.publish_step(step_id)?;
            *steps_out += 1;

            // `in_flight_depth = 1` overlap: if no prefills are waiting and every active
            // sequence still needs >= 2 tokens (so none retires after `step_id`), pre-allocate
            // pages and pre-write `plan_slot(step_id + 1)` while the 8 GPUs execute `step_id`.
            let can_overlap_next = waiting_permitted.is_empty()
                && !active.is_empty()
                && active
                    .iter()
                    .all(|seq| seq.generated_tokens.len() + 1 < seq.target_output_len);

            if can_overlap_next {
                let next_sid = self.step_cursor + 1;
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

            // Harvest accepted tokens from rank 0's ResultSlot
            let emitted: Vec<(u32, i32)> = {
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
                    .map(|o| (o.req_slot, o.accepted_tokens[0]))
                    .collect()
            };

            for (req_slot, tok) in emitted {
                let idx = *slot_to_active.get(req_slot as usize).unwrap_or(&usize::MAX);
                if idx < active.len() {
                    let seq = &mut active[idx];
                    if let Some(prev_t) = seq.last_token_time {
                        let itl = now.duration_since(prev_t).as_micros() as u64;
                        itls_us.push(itl);
                    } else {
                        let ttft = now.duration_since(seq.arrival_time).as_micros() as u64;
                        ttfts_us.push(ttft);
                        seq.first_token_time = Some(now);
                    }
                    seq.last_token_time = Some(now);
                    seq.last_token = tok;
                    seq.generated_tokens.push(tok);
                }
            }

            if let Some(next_sid) = prestaged_step_id {
                self.step_cursor = next_sid;
                {
                    let mut plan = self.coordinator.plan_slot_mut(next_sid)?;
                    let tids = plan.token_ids_mut()?;
                    for (i, seq) in active.iter_mut().enumerate() {
                        tids[i] = seq.last_token;
                        seq.current_seq_len += 1;
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
                    drop(finished.kv_leases);
                    drop(finished.slot_lease);

                    // Release concurrency permit immediately to next waiting request
                    if let Some((req_id, prompt, target_out)) = unpermitted_pool.pop_front() {
                        waiting_permitted.push_back(PendingRequest {
                            request_id: req_id,
                            prompt_tokens: prompt,
                            target_output_len: target_out,
                            arrival_time: now,
                        });
                    }
                } else {
                    idx += 1;
                }
            }
        }
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

        let mut token_cursor = 0usize;
        let mut scheduled_indices: Vec<usize> = Vec::new();

        while active.len() < max_seqs {
            let Some(front) = waiting_permitted.front() else {
                break;
            };
            let plen = front.prompt_tokens.len();
            if plen == 0 || plen > max_step_tokens {
                return Err(ModeBError::Protocol(format!(
                    "prompt length {plen} invalid for max_step_tokens={max_step_tokens}"
                )));
            }
            if token_cursor > 0 && token_cursor + plen > max_step_tokens {
                break;
            }
            let needed_pages = plen.div_ceil(page_block_tokens);
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
            let mut seq = ActiveSequence {
                request_id: req.request_id,
                slot_lease,
                kv_leases: Vec::new(),
                page_ids: Vec::new(),
                current_seq_len: plen,
                last_token: *req.prompt_tokens.last().unwrap_or(&0),
                generated_tokens: Vec::with_capacity(req.target_output_len),
                target_output_len: req.target_output_len,
                arrival_time: req.arrival_time,
                first_token_time: None,
                last_token_time: None,
            };
            seq.ensure_kv_pages(
                plen,
                page_block_tokens,
                max_pages_per_seq,
                &self.kv_pool,
                &mut self.shadow_page_table,
            )?;

            // Temporarily stash prompt tokens in `generated_tokens`? No, write directly to SHM below!
            let act_idx = active.len();
            slot_to_active[slot_id] = act_idx;
            scheduled_indices.push(act_idx);

            {
                let mut plan = self.coordinator.plan_slot_mut(step_id)?;
                let slot_i32 = slot_id as i32;
                for (offset, &tok) in req.prompt_tokens.iter().enumerate() {
                    plan.token_ids_mut()?[token_cursor + offset] = tok;
                    plan.positions_mut()?[token_cursor + offset] = offset as i32;
                    plan.req_slot_ids_mut()?[token_cursor + offset] = slot_i32;
                }
                let seq_batch_pos = scheduled_indices.len() - 1;
                plan.query_start_loc_mut()?[seq_batch_pos] = token_cursor as i32;
                plan.seq_lens_mut()?[seq_batch_pos] = plen as i32;
                plan.sample_row_indices_mut()?[seq_batch_pos] = (token_cursor + plen - 1) as i32;
            }
            token_cursor += plen;
            active.push(seq);
        }

        let num_seqs = scheduled_indices.len() as u32;
        if num_seqs == 0 {
            return Err(ModeBError::Protocol(
                "prefill step scheduled zero sequences".to_owned(),
            ));
        }

        let mut plan = self.coordinator.plan_slot_mut(step_id)?;
        plan.query_start_loc_mut()?[num_seqs as usize] = token_cursor as i32;
        plan.global_kv_page_table_mut()?
            .copy_from_slice(&self.shadow_page_table);
        *plan.header_mut()? = StepHeader {
            step_id: AtomicU64::new(0),
            opcode: StepOpcode::NormalStep as u32,
            num_seqs,
            num_encoder_tokens: token_cursor as u32,
            encoder_graph_bucket: 0,
            num_decoder_rows: token_cursor as u32,
            decoder_graph_bucket: 0,
            num_sample_rows: num_seqs,
            is_dspark_verify: 0,
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
        let num_seqs = active.len();
        if num_seqs == 0 {
            return Err(ModeBError::Protocol(
                "decode step scheduled zero active sequences".to_owned(),
            ));
        }

        for seq in active.iter_mut() {
            let next_seq_len = seq.current_seq_len + 1;
            seq.ensure_kv_pages(
                next_seq_len,
                page_block_tokens,
                max_pages_per_seq,
                &self.kv_pool,
                &mut self.shadow_page_table,
            )?;
        }

        let bucket = select_graph_bucket(num_seqs as u32, &DECODE_BUCKETS);
        let mut plan = self.coordinator.plan_slot_mut(step_id)?;
        for (row, seq) in active.iter_mut().enumerate() {
            let pos = seq.current_seq_len as i32;
            let next_len = pos + 1;
            if commit_tokens_immediately {
                plan.token_ids_mut()?[row] = seq.last_token;
                seq.current_seq_len = next_len as usize;
            }
            plan.positions_mut()?[row] = pos;
            plan.req_slot_ids_mut()?[row] = seq.slot_lease.id() as i32;
            plan.sample_row_indices_mut()?[row] = row as i32;
            plan.seq_lens_mut()?[row] = next_len;
            plan.query_start_loc_mut()?[row] = row as i32;
        }
        plan.query_start_loc_mut()?[num_seqs] = num_seqs as i32;
        plan.global_kv_page_table_mut()?
            .copy_from_slice(&self.shadow_page_table);
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
            flags: 1,
            _reserved: [0; 3],
        };
        Ok(())
    }

    pub fn shutdown(mut self) -> Result<()> {
        self.step_cursor += 1;
        let stop_step = self.step_cursor;
        self.coordinator.shutdown_and_wait(stop_step)
    }
}

fn should_schedule_prefill(
    active_count: usize,
    max_seqs: usize,
    max_step_tokens: usize,
    waiting_permitted: &VecDeque<PendingRequest>,
    consecutive_prefills: usize,
) -> bool {
    if waiting_permitted.is_empty() || active_count >= max_seqs {
        return false;
    }
    if active_count == 0 {
        return true;
    }
    // Interleave 1 CUDA-Graph decode step after every 4 prefill batches during ramp-up so
    // already-admitted requests make steady progress and stagger completions.
    if consecutive_prefills >= 4 {
        return false;
    }
    let first_len = waiting_permitted
        .front()
        .map_or(1, |req| req.prompt_tokens.len().max(1));
    let full_prefill_batch = (max_step_tokens / first_len).max(1);
    let free_slots = max_seqs - active_count;
    if waiting_permitted.len() >= full_prefill_batch.min(free_slots) {
        return true;
    }
    // Also admit small remainder batches once waiting >= 25 ms so TTFT stays bounded.
    waiting_permitted
        .front()
        .is_some_and(|req| req.arrival_time.elapsed().as_millis() >= 25)
}

/// Generate `num_prompts` deterministic pseudo-random token sequences in `[256, 65536)`
/// (avoiding EOS/control token IDs and avoiding artificial prefix-cache collisions).
pub fn build_deterministic_random_prompts(num_prompts: usize, input_len: usize) -> Vec<Vec<i32>> {
    let mut prompts = Vec::with_capacity(num_prompts);
    let mut state: u64 = 0x9E37_79B9_7F4A_7C15;
    for p in 0..num_prompts {
        let mut row = Vec::with_capacity(input_len);
        state ^= (p as u64).wrapping_mul(0xBF58_476D_1CE4_E5B9);
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
                current_seq_len: 0,
                last_token: 42,
                generated_tokens: Vec::new(),
                target_output_len: 10,
                arrival_time: Instant::now(),
                first_token_time: None,
                last_token_time: None,
            };
            seq.ensure_kv_pages(17, 16, 4, &kv_pool, &mut shadow_pt)
                .expect("allocate 2 pages");
            assert_eq!(seq.page_ids.len(), 2);
            assert_eq!(kv_pool.available(), 6);
            assert_eq!(shadow_pt[0], seq.page_ids[0]);
            assert_eq!(shadow_pt[1], seq.page_ids[1]);

            // Growing from 17 -> 32 stays within the existing 2 pages (2 * 16 = 32)
            seq.ensure_kv_pages(32, 16, 4, &kv_pool, &mut shadow_pt)
                .expect("still 2 pages");
            assert_eq!(kv_pool.available(), 6);

            // Token 33 crosses boundary and allocates page 3
            seq.ensure_kv_pages(33, 16, 4, &kv_pool, &mut shadow_pt)
                .expect("allocate 3rd page");
            assert_eq!(seq.page_ids.len(), 3);
            assert_eq!(kv_pool.available(), 5);
        }

        // Dropping `seq` releases all 3 KV blocks and the sequence slot
        assert_eq!(kv_pool.available(), 8);
        assert_eq!(slot_pool.available(), 2);
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
    }
}
