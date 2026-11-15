// Copyright (c) 2026 zstep Contributors
// SPDX-License-Identifier: MIT

//! Persistent online `ServeLoop` (Slice Z2 of the Nebula integration plan).
//!
//! This module is a child module of `serving`, reusing the exact `ActiveSequence`,
//! whole-request KV reservation, `stage_*`, and SHM harvest machinery of
//! [`ServingEngine`] while keeping the offline benchmark/parity paths untouched.
//!
//! Threading model:
//!
//! ```text
//!   HTTP handler ──submit()──► ServeLoopHandle ──channel──► ServeLoop thread
//!        │                         │                          │  (owns engine)
//!        └──── EventSink ◄── LoopEvent (token / finished) ◄────┘
//! ```
//!
//! Cancellation and caller stop-string completion are evaluated strictly at step
//! boundaries (after the in-flight GPU step converges), preserving `SlotLease` and
//! `BlockLease` RAII invariants and clearing any pre-staged next-step plan whenever
//! active batch membership changes.

use std::collections::{HashMap, HashSet, VecDeque};
use std::sync::atomic::{AtomicU8, AtomicU64, Ordering};
use std::sync::mpsc::{SyncSender, TrySendError};
use std::sync::{Arc, Mutex};
use std::time::{Duration, Instant};

use super::*;

/// Prometheus histogram bucket upper bounds (seconds) for TTFT.
pub const TTFT_BUCKET_BOUNDS_SECS: [f64; 10] =
    [0.025, 0.05, 0.1, 0.25, 0.5, 1.0, 2.5, 5.0, 10.0, 30.0];
const TTFT_BUCKET_BOUNDS_US: [u64; 10] = [
    25_000, 50_000, 100_000, 250_000, 500_000, 1_000_000, 2_500_000, 5_000_000, 10_000_000,
    30_000_000,
];

/// Prometheus histogram bucket upper bounds (seconds) for TPOT / ITL.
pub const TPOT_BUCKET_BOUNDS_SECS: [f64; 10] =
    [0.005, 0.01, 0.015, 0.025, 0.05, 0.1, 0.25, 0.5, 1.0, 2.5];
const TPOT_BUCKET_BOUNDS_US: [u64; 10] = [
    5_000, 10_000, 15_000, 25_000, 50_000, 100_000, 250_000, 500_000, 1_000_000, 2_500_000,
];

/// Why a request finished (mirrors `zs_tokenizer::FinishReason` without a dependency edge).
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum LoopFinishReason {
    Length,
    StopToken,
    Cancelled,
    Failed,
}

/// Event pushed to the HTTP layer by the loop thread. Sinks must be non-blocking.
#[derive(Debug, Clone)]
pub enum LoopEvent {
    Token {
        request_id: u64,
        token: u32,
    },
    Finished {
        request_id: u64,
        reason: LoopFinishReason,
    },
    Failed {
        request_id: u64,
        message: String,
    },
}

/// Non-blocking event sink created by the HTTP layer.
pub type EventSink = Arc<dyn Fn(LoopEvent) + Send + Sync>;

/// A request submitted to the loop.
pub struct SubmitRequest {
    pub prompt_tokens: Vec<i32>,
    pub max_new_tokens: usize,
    pub sink: EventSink,
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub enum SubmitError {
    /// Bounded waiting queue full or total in-flight request cap reached.
    Overloaded,
    /// Request exceeds configured KV page geometry and could never be admitted.
    InvalidDemand(String),
    /// Instance is starting / draining / faulted / shut down.
    Unavailable(EngineState),
}

/// Coarse engine lifecycle used by the HTTP probes (`nebula/docs/dev/zstep.md` §3).
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
#[repr(u8)]
pub enum EngineState {
    Starting = 0,
    Ready = 1,
    Draining = 2,
    Faulted = 3,
}

impl EngineState {
    pub fn from_u8(value: u8) -> Self {
        match value {
            1 => Self::Ready,
            2 => Self::Draining,
            3 => Self::Faulted,
            _ => Self::Starting,
        }
    }
}

/// Lock-free runtime metrics scraped by `GET /metrics`.
#[derive(Debug, Default)]
pub struct OnlineMetrics {
    pub waiting: AtomicU64,
    pub running: AtomicU64,
    pub kv_pages_used: AtomicU64,
    pub kv_pages_capacity: AtomicU64,
    pub preemptions: AtomicU64,
    pub finished: AtomicU64,
    pub cancelled: AtomicU64,
    pub rejected_overloaded: AtomicU64,
    pub failed: AtomicU64,
    pub ttft_count: AtomicU64,
    pub ttft_micros: AtomicU64,
    pub ttft_buckets: [AtomicU64; 10],
    pub tpot_count: AtomicU64,
    pub tpot_micros: AtomicU64,
    pub tpot_buckets: [AtomicU64; 10],
    pub triton_recompilations: AtomicU64,
    pub max_active: AtomicU64,
}

impl OnlineMetrics {
    pub fn kv_cache_usage(&self) -> f64 {
        let capacity = self.kv_pages_capacity.load(Ordering::Relaxed);
        if capacity == 0 {
            return 0.0;
        }
        let used = self.kv_pages_used.load(Ordering::Relaxed);
        (used as f64 / capacity as f64).clamp(0.0, 1.0)
    }

    pub fn observe_ttft_us(&self, us: u64) {
        self.ttft_count.fetch_add(1, Ordering::Relaxed);
        self.ttft_micros.fetch_add(us, Ordering::Relaxed);
        for (bound, bucket) in TTFT_BUCKET_BOUNDS_US.iter().zip(&self.ttft_buckets) {
            if us <= *bound {
                bucket.fetch_add(1, Ordering::Relaxed);
            }
        }
    }

    pub fn observe_tpot_us(&self, us: u64) {
        self.tpot_count.fetch_add(1, Ordering::Relaxed);
        self.tpot_micros.fetch_add(us, Ordering::Relaxed);
        for (bound, bucket) in TPOT_BUCKET_BOUNDS_US.iter().zip(&self.tpot_buckets) {
            if us <= *bound {
                bucket.fetch_add(1, Ordering::Relaxed);
            }
        }
    }
}

struct OnlineSequenceMeta {
    sink: EventSink,
}

struct OnlineState {
    waiting: VecDeque<PendingRequest>,
    active: Vec<ActiveSequence>,
    slot_to_active: Vec<usize>,
    meta: HashMap<u64, OnlineSequenceMeta>,
    cancelled: HashSet<u64>,
    completed_stops: HashSet<u64>,
    eos_tokens: Vec<u32>,
    next_request_id: u64,
    max_waiting: usize,
    pre_staged: Option<u64>,
    consecutive_prefills: usize,
    steps: u64,
    pending: Arc<AtomicU64>,
}

enum LoopCommand {
    Submit(
        SubmitRequest,
        SyncSender<std::result::Result<u64, SubmitError>>,
    ),
    Cancel(u64),
    CompleteStop(u64),
    Drain,
    Shutdown,
}

/// Handle cloned into the HTTP layer.
#[derive(Clone)]
pub struct ServeLoopHandle {
    tx: Arc<Mutex<SyncSender<LoopCommand>>>,
    state: Arc<AtomicU8>,
    metrics: Arc<OnlineMetrics>,
    thread: Arc<Mutex<Option<std::thread::JoinHandle<()>>>>,
    pending: Arc<AtomicU64>,
    max_in_flight_total: u64,
    page_block_tokens: usize,
    max_pages_per_seq: usize,
    kv_capacity: usize,
}

impl ServeLoopHandle {
    pub fn state(&self) -> EngineState {
        EngineState::from_u8(self.state.load(Ordering::Acquire))
    }

    pub fn state_arc(&self) -> Arc<AtomicU8> {
        Arc::clone(&self.state)
    }

    pub fn set_state(&self, state: EngineState) {
        self.state.store(state as u8, Ordering::Release);
    }

    pub fn metrics(&self) -> Arc<OnlineMetrics> {
        Arc::clone(&self.metrics)
    }

    /// Submit a request. Returns the loop-assigned request id or a typed admission error.
    pub fn submit(&self, request: SubmitRequest) -> std::result::Result<u64, SubmitError> {
        let cur_state = self.state();
        if cur_state != EngineState::Ready {
            return Err(SubmitError::Unavailable(cur_state));
        }
        let target_output_len = request.max_new_tokens.max(1);
        if let Err(err) = validate_request_kv_demand_with_limits(
            std::iter::once((request.prompt_tokens.len(), target_output_len)),
            self.page_block_tokens,
            self.max_pages_per_seq,
            self.kv_capacity,
        ) {
            return Err(SubmitError::InvalidDemand(err.to_string()));
        }

        // Contract §4.2: bounded waiting queue + active slots exhaustion returns 429, never 5xx.
        let in_flight = self.pending.fetch_add(1, Ordering::Relaxed);
        if in_flight >= self.max_in_flight_total {
            self.pending.fetch_sub(1, Ordering::Relaxed);
            self.metrics
                .rejected_overloaded
                .fetch_add(1, Ordering::Relaxed);
            return Err(SubmitError::Overloaded);
        }
        let (reply_tx, reply_rx) = std::sync::mpsc::sync_channel(1);
        {
            let guard = self
                .tx
                .lock()
                .map_err(|_| SubmitError::Unavailable(self.state()))?;
            match guard.try_send(LoopCommand::Submit(request, reply_tx)) {
                Ok(()) => {}
                Err(TrySendError::Full(_)) => {
                    self.pending.fetch_sub(1, Ordering::Relaxed);
                    self.metrics
                        .rejected_overloaded
                        .fetch_add(1, Ordering::Relaxed);
                    return Err(SubmitError::Overloaded);
                }
                Err(TrySendError::Disconnected(_)) => {
                    self.pending.fetch_sub(1, Ordering::Relaxed);
                    return Err(SubmitError::Unavailable(self.state()));
                }
            }
        }
        let result = reply_rx
            .recv()
            .unwrap_or_else(|_| Err(SubmitError::Unavailable(self.state())));
        if result.is_err() {
            self.pending.fetch_sub(1, Ordering::Relaxed);
        }
        result
    }

    /// Cancel an in-flight or queued request due to client disconnect / timeout / backpressure.
    pub fn cancel(&self, request_id: u64) {
        if let Ok(guard) = self.tx.lock() {
            let _ = guard.try_send(LoopCommand::Cancel(request_id));
        }
    }

    /// Reclaim an active sequence that matched a caller-supplied `stop` string (`outcome="finished"`).
    pub fn complete_stop(&self, request_id: u64) {
        if let Ok(guard) = self.tx.lock() {
            let _ = guard.try_send(LoopCommand::CompleteStop(request_id));
        }
    }

    /// Enter `Draining`: reject new admissions while keeping workers, `/health == 200`,
    /// and `/metrics` alive until [`Self::shutdown`] is invoked on process termination.
    pub fn drain(&self) {
        let _ = self.state.compare_exchange(
            EngineState::Ready as u8,
            EngineState::Draining as u8,
            Ordering::AcqRel,
            Ordering::Acquire,
        );
        if let Ok(guard) = self.tx.lock() {
            let _ = guard.try_send(LoopCommand::Drain);
        }
    }

    /// Stop the loop thread (cancelling any remaining requests) and tear down the engine.
    pub fn shutdown(&self) {
        let _ = self.state.compare_exchange(
            EngineState::Ready as u8,
            EngineState::Draining as u8,
            Ordering::AcqRel,
            Ordering::Acquire,
        );
        if let Ok(guard) = self.tx.lock() {
            let _ = guard.try_send(LoopCommand::Shutdown);
        }
    }

    pub fn is_running(&self) -> bool {
        self.thread
            .lock()
            .map(|guard| guard.is_some())
            .unwrap_or(false)
    }

    /// Wait for the loop thread to finish (call after [`Self::shutdown`]).
    pub fn join(&self, timeout: Duration) -> bool {
        let handle = self.thread.lock().ok().and_then(|mut guard| guard.take());
        let Some(handle) = handle else {
            return true;
        };
        let deadline = Instant::now() + timeout;
        while Instant::now() < deadline {
            if handle.is_finished() {
                let _ = handle.join();
                return true;
            }
            std::thread::sleep(Duration::from_millis(20));
        }
        false
    }
}

/// Configuration for the online loop (EOS ids + bounded queue size).
#[derive(Debug, Clone)]
pub struct OnlineConfig {
    pub eos_token_ids: Vec<u32>,
    pub max_waiting_requests: usize,
}

/// Spawn the persistent online loop around an already-launched [`ServingEngine`].
pub fn spawn(engine: ServingEngine, config: OnlineConfig) -> ServeLoopHandle {
    let state = Arc::new(AtomicU8::new(EngineState::Ready as u8));
    spawn_with_state(engine, config, state)
}

/// Spawn the persistent online loop sharing `state` directly with the HTTP probe state.
pub fn spawn_with_state(
    mut engine: ServingEngine,
    config: OnlineConfig,
    state: Arc<AtomicU8>,
) -> ServeLoopHandle {
    // Online serving keeps `device_token_overlap = false` so EOS / stop-string sequences
    // never pre-publish speculative post-EOS GPU steps.
    engine.set_device_token_overlap(false);
    let geometry = engine.coordinator.config().geometry;
    let max_seqs = geometry.max_seqs as usize;
    let max_waiting = config.max_waiting_requests.max(1);
    let kv_capacity = engine.kv_pool.capacity();
    let cmd_capacity = (max_seqs + max_waiting).saturating_mul(2).max(32);
    let (tx, rx) = std::sync::mpsc::sync_channel::<LoopCommand>(cmd_capacity);
    let metrics = Arc::new(OnlineMetrics::default());
    let pending = Arc::new(AtomicU64::new(0));
    let handle = ServeLoopHandle {
        tx: Arc::new(Mutex::new(tx)),
        state: Arc::clone(&state),
        metrics: Arc::clone(&metrics),
        thread: Arc::new(Mutex::new(None)),
        pending: Arc::clone(&pending),
        max_in_flight_total: (max_seqs + max_waiting) as u64,
        page_block_tokens: geometry.page_block_tokens.max(1) as usize,
        max_pages_per_seq: geometry.max_pages_per_seq.max(1) as usize,
        kv_capacity,
    };
    let loop_state = Arc::clone(&state);
    let loop_metrics = Arc::clone(&metrics);
    let thread = std::thread::Builder::new()
        .name("zstep-serve-loop".to_owned())
        .spawn(move || run_loop(engine, config, rx, loop_state, loop_metrics, pending))
        .expect("spawn serve loop thread");
    if let Ok(mut guard) = handle.thread.lock() {
        *guard = Some(thread);
    }
    handle
}

fn run_loop(
    mut engine: ServingEngine,
    config: OnlineConfig,
    rx: std::sync::mpsc::Receiver<LoopCommand>,
    state: Arc<AtomicU8>,
    metrics: Arc<OnlineMetrics>,
    pending: Arc<AtomicU64>,
) {
    let capacity = engine.kv_pool.capacity();
    metrics
        .kv_pages_capacity
        .store(capacity as u64, Ordering::Relaxed);
    let max_seqs = engine.coordinator.config().geometry.max_seqs as usize;
    let mut online = OnlineState {
        waiting: VecDeque::new(),
        active: Vec::new(),
        slot_to_active: vec![usize::MAX; max_seqs],
        meta: HashMap::new(),
        cancelled: HashSet::new(),
        completed_stops: HashSet::new(),
        eos_tokens: config.eos_token_ids,
        next_request_id: 1,
        max_waiting: config.max_waiting_requests.max(1),
        pre_staged: None,
        consecutive_prefills: 0,
        steps: 0,
        pending,
    };
    let idle = Duration::from_millis(2);
    let mut draining = EngineState::from_u8(state.load(Ordering::Acquire)) == EngineState::Draining;
    let mut shutting_down = false;
    let mut fault: Option<String> = None;

    loop {
        loop {
            match rx.try_recv() {
                Ok(LoopCommand::Submit(request, reply)) => {
                    let cur_state = EngineState::from_u8(state.load(Ordering::Acquire));
                    let result = if draining || shutting_down || cur_state != EngineState::Ready {
                        let unavailable = if cur_state == EngineState::Ready {
                            EngineState::Draining
                        } else {
                            cur_state
                        };
                        Err(SubmitError::Unavailable(unavailable))
                    } else if online.waiting.len() >= online.max_waiting {
                        metrics.rejected_overloaded.fetch_add(1, Ordering::Relaxed);
                        Err(SubmitError::Overloaded)
                    } else {
                        let request_id = online.next_request_id;
                        online.next_request_id += 1;
                        let now = Instant::now();
                        online
                            .meta
                            .insert(request_id, OnlineSequenceMeta { sink: request.sink });
                        online.waiting.push_back(PendingRequest {
                            request_id: request_id as usize,
                            prompt_tokens: request.prompt_tokens,
                            target_output_len: request.max_new_tokens.max(1),
                            permit_time: now,
                            submit_time: now,
                        });
                        Ok(request_id)
                    };
                    let _ = reply.send(result);
                }
                Ok(LoopCommand::Cancel(request_id)) => {
                    online.cancelled.insert(request_id);
                }
                Ok(LoopCommand::CompleteStop(request_id)) => {
                    online.completed_stops.insert(request_id);
                }
                Ok(LoopCommand::Drain) => {
                    draining = true;
                }
                Ok(LoopCommand::Shutdown) => {
                    draining = true;
                    shutting_down = true;
                }
                Err(std::sync::mpsc::TryRecvError::Empty) => break,
                Err(std::sync::mpsc::TryRecvError::Disconnected) => {
                    draining = true;
                    shutting_down = true;
                    break;
                }
            }
        }

        if shutting_down {
            // Drain deadline reached or process exiting: cancel all remaining queued/active work.
            for req in &online.waiting {
                online.cancelled.insert(req.request_id as u64);
            }
            for seq in &online.active {
                online.cancelled.insert(seq.request_id as u64);
            }
        }

        let max_pages_per_seq = engine.coordinator.config().geometry.max_pages_per_seq as usize;
        let dummy_page = dummy_page_value(&engine);
        if reclaim_stopped_and_cancelled(
            &mut engine.shadow_page_table,
            &mut engine.dirty_pt_slots,
            max_pages_per_seq,
            dummy_page,
            &mut online,
            &metrics,
        ) {
            update_gauges(&engine, &online, &metrics);
        }

        if shutting_down && online.active.is_empty() && online.waiting.is_empty() {
            break;
        }

        if online.active.is_empty() && online.waiting.is_empty() {
            if let Err(err) = engine.coordinator.check_workers_healthy() {
                fault = Some(err.to_string());
                break;
            }
            update_gauges(&engine, &online, &metrics);
            std::thread::sleep(idle);
            continue;
        }

        match online_step(&mut engine, &mut online, &metrics) {
            Ok(true) => {
                update_gauges(&engine, &online, &metrics);
            }
            Ok(false) => {
                if let Err(err) = engine.coordinator.check_workers_healthy() {
                    fault = Some(err.to_string());
                    break;
                }
                update_gauges(&engine, &online, &metrics);
                std::thread::sleep(idle);
            }
            Err(error) => {
                fault = Some(error.to_string());
                break;
            }
        }
    }

    if let Some(message) = fault {
        engine.poisoned = true;
        state.store(EngineState::Faulted as u8, Ordering::Release);
        fail_all(&mut online, &metrics, &message);
        update_gauges(&engine, &online, &metrics);
        online.metrics_waiting_zero(&metrics);
        // Skip graceful shutdown on fault: `Drop` signals SHM abort and reaps workers immediately.
        drop(engine);
        return;
    }

    update_gauges(&engine, &online, &metrics);
    online.metrics_waiting_zero(&metrics);
    if engine.shutdown().is_err() {
        state.store(EngineState::Faulted as u8, Ordering::Release);
    } else if state.load(Ordering::Acquire) != EngineState::Faulted as u8 {
        state.store(EngineState::Draining as u8, Ordering::Release);
    }
}

impl OnlineState {
    fn metrics_waiting_zero(&self, metrics: &OnlineMetrics) {
        metrics.waiting.store(0, Ordering::Relaxed);
        metrics.running.store(0, Ordering::Relaxed);
    }
}

fn update_gauges(engine: &ServingEngine, online: &OnlineState, metrics: &OnlineMetrics) {
    metrics
        .waiting
        .store(online.waiting.len() as u64, Ordering::Relaxed);
    metrics
        .running
        .store(online.active.len() as u64, Ordering::Relaxed);
    let capacity = engine.kv_pool.capacity();
    metrics
        .kv_pages_capacity
        .store(capacity as u64, Ordering::Relaxed);
    metrics.kv_pages_used.store(
        capacity.saturating_sub(engine.kv_pool.available()) as u64,
        Ordering::Relaxed,
    );
    metrics.triton_recompilations.store(
        u64::from(engine.coordinator.current_triton_recompilations()),
        Ordering::Relaxed,
    );
    metrics
        .max_active
        .fetch_max(online.active.len() as u64, Ordering::Relaxed);
}

fn emit(online: &OnlineState, event: LoopEvent) {
    let request_id = match &event {
        LoopEvent::Token { request_id, .. }
        | LoopEvent::Finished { request_id, .. }
        | LoopEvent::Failed { request_id, .. } => *request_id,
    };
    if let Some(meta) = online.meta.get(&request_id) {
        let terminal = matches!(event, LoopEvent::Finished { .. } | LoopEvent::Failed { .. });
        (meta.sink)(event);
        if terminal {
            online.pending.fetch_sub(1, Ordering::Relaxed);
        }
    }
}

fn fail_all(online: &mut OnlineState, metrics: &OnlineMetrics, message: &str) {
    for request_id in online.meta.keys().copied().collect::<Vec<_>>() {
        emit(
            online,
            LoopEvent::Failed {
                request_id,
                message: message.to_owned(),
            },
        );
        metrics.failed.fetch_add(1, Ordering::Relaxed);
    }
    online.pre_staged = None;
    online.meta.clear();
    online.active.clear();
    online.waiting.clear();
}

fn retire_active_at(
    index: usize,
    active: &mut Vec<ActiveSequence>,
    slot_to_active: &mut [usize],
    shadow_page_table: &mut [i32],
    dirty_pt_slots: &mut [u128; 2],
    max_pages_per_seq: usize,
    dummy_page: i32,
) {
    let finished = active.swap_remove(index);
    let slot_id = finished.slot_lease.id() as usize;
    slot_to_active[slot_id] = usize::MAX;
    let row_start = slot_id * max_pages_per_seq;
    shadow_page_table[row_start..row_start + max_pages_per_seq].fill(dummy_page);
    let dirty_bit = 1u128 << slot_id;
    dirty_pt_slots[0] |= dirty_bit;
    dirty_pt_slots[1] |= dirty_bit;
    if index < active.len() {
        let moved_slot = active[index].slot_lease.id() as usize;
        slot_to_active[moved_slot] = index;
    }
    drop(finished);
}

/// Reclaim cancelled or stop-string-completed sequences at a step boundary.
/// Clears `online.pre_staged` whenever active batch membership changes.
fn reclaim_stopped_and_cancelled(
    shadow_page_table: &mut [i32],
    dirty_pt_slots: &mut [u128; 2],
    max_pages_per_seq: usize,
    dummy_page: i32,
    online: &mut OnlineState,
    metrics: &OnlineMetrics,
) -> bool {
    if online.cancelled.is_empty() && online.completed_stops.is_empty() {
        return false;
    }
    let mut changed = false;
    let mut active_changed = false;

    let mut index = 0;
    while index < online.waiting.len() {
        let id = online.waiting[index].request_id as u64;
        let is_stop = online.completed_stops.contains(&id);
        let is_cancel = online.cancelled.contains(&id);
        if is_stop || is_cancel {
            online.waiting.remove(index);
            let reason = if is_stop {
                metrics.finished.fetch_add(1, Ordering::Relaxed);
                LoopFinishReason::StopToken
            } else {
                metrics.cancelled.fetch_add(1, Ordering::Relaxed);
                LoopFinishReason::Cancelled
            };
            emit(
                online,
                LoopEvent::Finished {
                    request_id: id,
                    reason,
                },
            );
            online.meta.remove(&id);
            changed = true;
        } else {
            index += 1;
        }
    }

    let mut index = 0;
    while index < online.active.len() {
        let id = online.active[index].request_id as u64;
        let is_stop = online.completed_stops.contains(&id);
        let is_cancel = online.cancelled.contains(&id);
        if is_stop || is_cancel {
            retire_active_at(
                index,
                &mut online.active,
                &mut online.slot_to_active,
                shadow_page_table,
                dirty_pt_slots,
                max_pages_per_seq,
                dummy_page,
            );
            let reason = if is_stop {
                metrics.finished.fetch_add(1, Ordering::Relaxed);
                LoopFinishReason::StopToken
            } else {
                metrics.cancelled.fetch_add(1, Ordering::Relaxed);
                LoopFinishReason::Cancelled
            };
            emit(
                online,
                LoopEvent::Finished {
                    request_id: id,
                    reason,
                },
            );
            online.meta.remove(&id);
            changed = true;
            active_changed = true;
        } else {
            index += 1;
        }
    }

    if active_changed {
        online.pre_staged = None;
    }
    online.cancelled.clear();
    online.completed_stops.clear();
    changed
}

/// Retire active sequences that reached `target_output_len` or emitted an EOS token.
/// Returns `true` if any active sequence retired (invalidating any `pre_staged` step).
fn retire_completed_sequences(
    shadow_page_table: &mut [i32],
    dirty_pt_slots: &mut [u128; 2],
    max_pages_per_seq: usize,
    dummy_page: i32,
    online: &mut OnlineState,
    metrics: &OnlineMetrics,
) -> bool {
    let mut any_retired = false;
    let mut index = 0;
    while index < online.active.len() {
        let (request_id, prefill_done, generated, target, last_token) = {
            let seq = &online.active[index];
            (
                seq.request_id as u64,
                seq.prefilled_tokens == seq.prompt_tokens.len(),
                seq.generated_tokens.len(),
                seq.target_output_len,
                seq.last_token,
            )
        };
        let eos = prefill_done && generated > 0 && online.eos_tokens.contains(&(last_token as u32));
        if generated >= target || eos {
            retire_active_at(
                index,
                &mut online.active,
                &mut online.slot_to_active,
                shadow_page_table,
                dirty_pt_slots,
                max_pages_per_seq,
                dummy_page,
            );
            let reason = if eos {
                LoopFinishReason::StopToken
            } else {
                LoopFinishReason::Length
            };
            emit(online, LoopEvent::Finished { request_id, reason });
            online.meta.remove(&request_id);
            metrics.finished.fetch_add(1, Ordering::Relaxed);
            any_retired = true;
        } else {
            index += 1;
        }
    }
    if any_retired {
        online.pre_staged = None;
    }
    any_retired
}

/// Execute at most one GPU step (`Ok(false)` when no sequence is currently runnable).
fn online_step(
    engine: &mut ServingEngine,
    online: &mut OnlineState,
    metrics: &OnlineMetrics,
) -> Result<bool> {
    let geometry = engine.coordinator.config().geometry;
    let max_seqs = geometry.max_seqs as usize;
    let max_step_tokens = geometry.max_step_tokens as usize;
    let max_prefill_chunk = engine.coordinator.config().max_prefill_chunk;
    let page_block_tokens = geometry.page_block_tokens as usize;
    let max_pages_per_seq = geometry.max_pages_per_seq as usize;

    let waiting_admissible = engine.front_request_admissible(
        &online.active,
        &online.waiting,
        max_seqs,
        page_block_tokens,
    );

    if online.active.is_empty() && !waiting_admissible {
        online.pre_staged = None;
        return Ok(false);
    }

    // Discard pre-staged pure-decode plan if a newly arrived request is now admissible.
    if waiting_admissible {
        online.pre_staged = None;
    }

    let step_id = if let Some(next_sid) = online.pre_staged.take() {
        engine.step_cursor = next_sid;
        {
            let mut plan = engine.coordinator.plan_slot_mut(next_sid)?;
            for (index, seq) in online.active.iter_mut().enumerate() {
                let (token, lookback) = seq.revise_overlapped_decode();
                plan.token_ids_mut()?[index] = token;
                plan.engram_prev_window_mut()?[index * 3..index * 3 + 3].copy_from_slice(&lookback);
            }
        }
        online.consecutive_prefills = 0;
        next_sid
    } else {
        engine.step_cursor += 1;
        let sid = engine.step_cursor;
        let decode_active_count = online
            .active
            .iter()
            .filter(|seq| seq.prefilled_tokens == seq.prompt_tokens.len())
            .count();
        let has_partial_prefill = decode_active_count != online.active.len();
        if let Some(mixed_cap) = engine.max_mixed_step_tokens {
            let can_attempt_mixed = has_partial_prefill || waiting_admissible;
            if can_attempt_mixed {
                match engine.stage_mixed_step(
                    sid,
                    mixed_cap,
                    &mut online.active,
                    &mut online.slot_to_active,
                    &mut online.waiting,
                    page_block_tokens,
                    max_pages_per_seq,
                )? {
                    MixedStepOutcome::MixedOrPrefill { .. } => {
                        online.consecutive_prefills += 1;
                    }
                    MixedStepOutcome::PureDecodeFallback => {
                        // `stage_mixed_step` already staged the pure-decode plan and advanced
                        // `seq.current_seq_len`; never call `stage_decode_step` a second time.
                        online.consecutive_prefills = 0;
                    }
                }
            } else {
                engine.stage_decode_step(
                    sid,
                    &mut online.active,
                    page_block_tokens,
                    max_pages_per_seq,
                    true,
                )?;
                online.consecutive_prefills = 0;
            }
        } else {
            let run_prefill = (has_partial_prefill || waiting_admissible)
                && should_schedule_prefill(
                    decode_active_count,
                    online.active.len(),
                    has_partial_prefill,
                    max_seqs,
                    max_step_tokens,
                    max_prefill_chunk,
                    &online.waiting,
                    online.consecutive_prefills,
                    engine.max_prefill_coalesce_wait_ms,
                );
            if run_prefill {
                engine.stage_prefill_step(
                    sid,
                    &mut online.active,
                    &mut online.slot_to_active,
                    &mut online.waiting,
                    page_block_tokens,
                    max_pages_per_seq,
                )?;
                online.consecutive_prefills += 1;
            } else {
                engine.stage_decode_step(
                    sid,
                    &mut online.active,
                    page_block_tokens,
                    max_pages_per_seq,
                    true,
                )?;
                online.consecutive_prefills = 0;
            }
        }
        sid
    };

    engine.coordinator.publish_step(step_id)?;
    online.steps += 1;

    let post_admit_waiting_admissible = engine.front_request_admissible(
        &online.active,
        &online.waiting,
        max_seqs,
        page_block_tokens,
    );
    let can_overlap_next = !post_admit_waiting_admissible
        && !online.active.is_empty()
        && online.active.iter().all(|seq| {
            seq.prefilled_tokens == seq.prompt_tokens.len()
                && seq.generated_tokens.len() + 1 < seq.target_output_len
        });
    if can_overlap_next {
        let next_sid = engine.step_cursor + 1;
        engine.stage_decode_step(
            next_sid,
            &mut online.active,
            page_block_tokens,
            max_pages_per_seq,
            false,
        )?;
        online.pre_staged = Some(next_sid);
    }

    engine.coordinator.wait_step_completed(step_id)?;
    let now = Instant::now();

    let emitted: Vec<(u32, i32)> = {
        let result = engine.coordinator.result_slot(step_id)?;
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
            .map(|output| (output.req_slot, output.accepted_tokens[0]))
            .collect()
    };

    for (req_slot, token) in emitted {
        let idx = *online
            .slot_to_active
            .get(req_slot as usize)
            .unwrap_or(&usize::MAX);
        if idx >= online.active.len() {
            continue;
        }
        let request_id = {
            let seq = &mut online.active[idx];
            if seq.first_token_time.is_none() {
                let (ttft_us, _) = first_token_latency_us(seq.permit_time, seq.submit_time, now);
                metrics.observe_ttft_us(ttft_us);
                seq.first_token_time = Some(now);
            } else {
                let prev = seq.last_token_time.unwrap_or(now);
                let itl_us = now.duration_since(prev).as_micros() as u64;
                metrics.observe_tpot_us(itl_us);
            }
            seq.last_token_time = Some(now);
            seq.last_token = token;
            seq.generated_tokens.push(token);
            seq.request_id as u64
        };
        emit(
            online,
            LoopEvent::Token {
                request_id,
                token: token as u32,
            },
        );
    }

    let dummy_page = dummy_page_value(engine);
    let _ = retire_completed_sequences(
        &mut engine.shadow_page_table,
        &mut engine.dirty_pt_slots,
        max_pages_per_seq,
        dummy_page,
        online,
        metrics,
    );
    Ok(true)
}

fn dummy_page_value(engine: &ServingEngine) -> i32 {
    engine.coordinator.config().launch.num_kv_pages as i32
}

// ---------------------------------------------------------------------------
// Mock loop (contract tests + Nebula integration harness)
// ---------------------------------------------------------------------------

/// Deterministic scripted backend used by HTTP contract tests and integration harnesses.
#[derive(Debug, Clone)]
pub struct MockScript {
    pub tokens: Vec<u32>,
    pub finish: LoopFinishReason,
    pub per_token_delay: Duration,
    pub max_waiting: usize,
    pub max_seqs: usize,
    pub page_block_tokens: usize,
    pub max_pages_per_seq: usize,
    pub kv_capacity: usize,
    pub fail_after: Option<usize>,
}

impl Default for MockScript {
    fn default() -> Self {
        Self {
            tokens: vec![1, 2, 3],
            finish: LoopFinishReason::Length,
            per_token_delay: Duration::from_millis(1),
            max_waiting: 8,
            max_seqs: 128,
            page_block_tokens: 16,
            max_pages_per_seq: 256,
            kv_capacity: 49_152,
            fail_after: None,
        }
    }
}

/// Spawn a scripted in-process loop (no GPU, no SHM).
pub fn spawn_mock(script: MockScript) -> ServeLoopHandle {
    let max_waiting = script.max_waiting.max(1);
    let max_seqs = script.max_seqs.max(1);
    let max_in_flight_total = (max_seqs + max_waiting) as u64;
    let (tx, rx) = std::sync::mpsc::sync_channel::<LoopCommand>((max_seqs + max_waiting) * 2 + 16);
    let state = Arc::new(AtomicU8::new(EngineState::Ready as u8));
    let metrics = Arc::new(OnlineMetrics::default());
    metrics
        .kv_pages_capacity
        .store(script.kv_capacity as u64, Ordering::Relaxed);
    let pending = Arc::new(AtomicU64::new(0));
    let handle = ServeLoopHandle {
        tx: Arc::new(Mutex::new(tx)),
        state: Arc::clone(&state),
        metrics: Arc::clone(&metrics),
        thread: Arc::new(Mutex::new(None)),
        pending: Arc::clone(&pending),
        max_in_flight_total,
        page_block_tokens: script.page_block_tokens.max(1),
        max_pages_per_seq: script.max_pages_per_seq.max(1),
        kv_capacity: script.kv_capacity.max(1),
    };
    let mock_pending = Arc::clone(&pending);
    let mock_state = Arc::clone(&state);
    let thread = std::thread::Builder::new()
        .name("zstep-serve-mock".to_owned())
        .spawn(move || {
            struct MockSeq {
                id: u64,
                sink: EventSink,
                cursor: usize,
                last_emit: Instant,
            }
            let mut active: Vec<MockSeq> = Vec::new();
            let mut waiting: VecDeque<(u64, EventSink)> = VecDeque::new();
            let mut cancelled: HashSet<u64> = HashSet::new();
            let mut completed_stops: HashSet<u64> = HashSet::new();
            let mut next_id = 1u64;
            let mut draining = false;
            let mut shutting_down = false;

            loop {
                loop {
                    match rx.try_recv() {
                        Ok(LoopCommand::Submit(request, reply)) => {
                            let cur_state =
                                EngineState::from_u8(mock_state.load(Ordering::Acquire));
                            if draining || shutting_down || cur_state != EngineState::Ready {
                                let unavailable = if cur_state == EngineState::Ready {
                                    EngineState::Draining
                                } else {
                                    cur_state
                                };
                                let _ = reply.send(Err(SubmitError::Unavailable(unavailable)));
                                continue;
                            }
                            if waiting.len() >= max_waiting {
                                metrics.rejected_overloaded.fetch_add(1, Ordering::Relaxed);
                                let _ = reply.send(Err(SubmitError::Overloaded));
                                continue;
                            }
                            let id = next_id;
                            next_id += 1;
                            let _ = reply.send(Ok(id));
                            waiting.push_back((id, request.sink));
                        }
                        Ok(LoopCommand::Cancel(id)) => {
                            cancelled.insert(id);
                        }
                        Ok(LoopCommand::CompleteStop(id)) => {
                            completed_stops.insert(id);
                        }
                        Ok(LoopCommand::Drain) => {
                            draining = true;
                        }
                        Ok(LoopCommand::Shutdown) => {
                            draining = true;
                            shutting_down = true;
                        }
                        Err(std::sync::mpsc::TryRecvError::Empty) => break,
                        Err(std::sync::mpsc::TryRecvError::Disconnected) => {
                            draining = true;
                            shutting_down = true;
                            break;
                        }
                    }
                }

                if shutting_down {
                    for (id, _) in &waiting {
                        cancelled.insert(*id);
                    }
                    for seq in &active {
                        cancelled.insert(seq.id);
                    }
                }

                // Reclaim cancelled / stop-completed queued or active requests.
                let mut w_idx = 0;
                while w_idx < waiting.len() {
                    let id = waiting[w_idx].0;
                    if completed_stops.contains(&id) || cancelled.contains(&id) {
                        let (_, sink) = waiting.remove(w_idx).expect("waiting entry");
                        let reason = if completed_stops.contains(&id) {
                            metrics.finished.fetch_add(1, Ordering::Relaxed);
                            LoopFinishReason::StopToken
                        } else {
                            metrics.cancelled.fetch_add(1, Ordering::Relaxed);
                            LoopFinishReason::Cancelled
                        };
                        (sink)(LoopEvent::Finished {
                            request_id: id,
                            reason,
                        });
                        mock_pending.fetch_sub(1, Ordering::Relaxed);
                    } else {
                        w_idx += 1;
                    }
                }
                let mut a_idx = 0;
                while a_idx < active.len() {
                    let id = active[a_idx].id;
                    if completed_stops.contains(&id) || cancelled.contains(&id) {
                        let seq = active.swap_remove(a_idx);
                        let reason = if completed_stops.contains(&id) {
                            metrics.finished.fetch_add(1, Ordering::Relaxed);
                            LoopFinishReason::StopToken
                        } else {
                            metrics.cancelled.fetch_add(1, Ordering::Relaxed);
                            LoopFinishReason::Cancelled
                        };
                        (seq.sink)(LoopEvent::Finished {
                            request_id: id,
                            reason,
                        });
                        mock_pending.fetch_sub(1, Ordering::Relaxed);
                    } else {
                        a_idx += 1;
                    }
                }
                cancelled.clear();
                completed_stops.clear();

                while active.len() < max_seqs {
                    let Some((id, sink)) = waiting.pop_front() else {
                        break;
                    };
                    active.push(MockSeq {
                        id,
                        sink,
                        cursor: 0,
                        last_emit: Instant::now()
                            .checked_sub(script.per_token_delay)
                            .unwrap_or_else(Instant::now),
                    });
                }

                metrics
                    .waiting
                    .store(waiting.len() as u64, Ordering::Relaxed);
                metrics
                    .running
                    .store(active.len() as u64, Ordering::Relaxed);

                if shutting_down && waiting.is_empty() && active.is_empty() {
                    break;
                }

                let now = Instant::now();
                let mut progressed = false;
                let mut idx = 0;
                while idx < active.len() {
                    if now.duration_since(active[idx].last_emit) < script.per_token_delay {
                        idx += 1;
                        continue;
                    }
                    progressed = true;
                    let step_idx = active[idx].cursor;
                    if script.fail_after == Some(step_idx) {
                        let seq = active.swap_remove(idx);
                        mock_state.store(EngineState::Faulted as u8, Ordering::Release);
                        (seq.sink)(LoopEvent::Failed {
                            request_id: seq.id,
                            message: "mock worker failure".to_owned(),
                        });
                        mock_pending.fetch_sub(1, Ordering::Relaxed);
                        metrics.failed.fetch_add(1, Ordering::Relaxed);
                        continue;
                    }
                    if let Some(&token) = script.tokens.get(step_idx) {
                        active[idx].cursor += 1;
                        active[idx].last_emit = now;
                        if step_idx == 0 {
                            metrics.observe_ttft_us(1_000);
                        } else {
                            metrics.observe_tpot_us(script.per_token_delay.as_micros() as u64);
                        }
                        (active[idx].sink)(LoopEvent::Token {
                            request_id: active[idx].id,
                            token,
                        });
                    }
                    if active[idx].cursor >= script.tokens.len() {
                        let seq = active.swap_remove(idx);
                        (seq.sink)(LoopEvent::Finished {
                            request_id: seq.id,
                            reason: script.finish,
                        });
                        mock_pending.fetch_sub(1, Ordering::Relaxed);
                        match script.finish {
                            LoopFinishReason::Cancelled => {
                                metrics.cancelled.fetch_add(1, Ordering::Relaxed);
                            }
                            LoopFinishReason::Failed => {
                                metrics.failed.fetch_add(1, Ordering::Relaxed);
                            }
                            _ => {
                                metrics.finished.fetch_add(1, Ordering::Relaxed);
                            }
                        }
                    } else {
                        idx += 1;
                    }
                }

                metrics
                    .waiting
                    .store(waiting.len() as u64, Ordering::Relaxed);
                metrics
                    .running
                    .store(active.len() as u64, Ordering::Relaxed);

                if !progressed {
                    std::thread::sleep(Duration::from_millis(1));
                }
            }
            metrics.waiting.store(0, Ordering::Relaxed);
            metrics.running.store(0, Ordering::Relaxed);
        })
        .expect("spawn mock loop");
    if let Ok(mut guard) = handle.thread.lock() {
        *guard = Some(thread);
    }
    handle
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn random_50_percent_cancellation_across_prefill_and_decode_leaks_zero_slots_or_kv_pages() {
        let max_seqs = 8usize;
        let max_pages_per_seq = 8usize;
        let page_block_tokens = 16usize;
        let kv_capacity = 48usize;
        let dummy_page = kv_capacity as i32;

        let slot_pool = SlotPool::new(max_seqs as u32);
        let kv_pool = BlockPool::new(CacheGroupKind::GlobalAndIndex, kv_capacity);
        let mut shadow_pt = vec![dummy_page; max_seqs * max_pages_per_seq];
        let mut dirty_pt_slots = [0u128; 2];
        let pending = Arc::new(AtomicU64::new(0));
        let metrics = OnlineMetrics::default();

        let mut online = OnlineState {
            waiting: VecDeque::new(),
            active: Vec::new(),
            slot_to_active: vec![usize::MAX; max_seqs],
            meta: HashMap::new(),
            cancelled: HashSet::new(),
            completed_stops: HashSet::new(),
            eos_tokens: vec![2],
            next_request_id: 1,
            max_waiting: 64,
            pre_staged: Some(7),
            consecutive_prefills: 0,
            steps: 0,
            pending: Arc::clone(&pending),
        };

        let mut rng = 0x1234_5678_9ABC_DEF0u64;
        for step in 0..160u64 {
            // Submit 1..=3 synthetic requests each step.
            rng = splitmix64(rng);
            let arrivals = (rng % 3) + 1;
            for _ in 0..arrivals {
                let id = online.next_request_id;
                online.next_request_id += 1;
                rng = splitmix64(rng);
                let prompt_len = 8 + (rng as usize % 32);
                let target_out = 4 + (rng as usize % 20);
                pending.fetch_add(1, Ordering::Relaxed);
                online.meta.insert(
                    id,
                    OnlineSequenceMeta {
                        sink: Arc::new(|_| {}),
                    },
                );
                online.waiting.push_back(PendingRequest {
                    request_id: id as usize,
                    prompt_tokens: vec![11; prompt_len],
                    target_output_len: target_out,
                    permit_time: Instant::now(),
                    submit_time: Instant::now(),
                });
            }

            // Admit admissible waiting requests into `active` (whole-request KV reservation).
            while front_request_admissible_in_pools(
                &online.active,
                &online.waiting,
                &slot_pool,
                &kv_pool,
                max_seqs,
                page_block_tokens,
            ) {
                let req = online.waiting.pop_front().expect("front");
                let slot_lease = slot_pool.allocate().expect("slot");
                let slot_id = slot_lease.id() as usize;
                let demand = request_kv_page_demand(
                    req.prompt_tokens.len(),
                    req.target_output_len,
                    page_block_tokens,
                );
                let mut seq = ActiveSequence {
                    request_id: req.request_id,
                    slot_lease,
                    kv_leases: Vec::new(),
                    page_ids: Vec::new(),
                    prefilled_tokens: req.prompt_tokens.len() / 2,
                    current_seq_len: req.prompt_tokens.len() / 2,
                    last_token: 11,
                    prompt_tokens: req.prompt_tokens,
                    generated_tokens: Vec::new(),
                    generated_top2: Vec::new(),
                    target_output_len: req.target_output_len,
                    reserved_pages: demand,
                    permit_time: req.permit_time,
                    submit_time: req.submit_time,
                    first_token_time: None,
                    last_token_time: None,
                };
                let init_len = seq.current_seq_len.max(1);
                seq.ensure_kv_pages(
                    init_len,
                    page_block_tokens,
                    max_pages_per_seq,
                    &kv_pool,
                    &mut shadow_pt,
                    &mut dirty_pt_slots,
                )
                .expect("initial pages");
                online.slot_to_active[slot_id] = online.active.len();
                online.active.push(seq);
            }

            // Advance active prefill / decode sequences by one step.
            for seq in &mut online.active {
                if seq.prefilled_tokens < seq.prompt_tokens.len() {
                    seq.prefilled_tokens = seq.prompt_tokens.len();
                    seq.current_seq_len = seq.prompt_tokens.len();
                } else {
                    seq.current_seq_len += 1;
                }
                seq.ensure_kv_pages(
                    seq.current_seq_len,
                    page_block_tokens,
                    max_pages_per_seq,
                    &kv_pool,
                    &mut shadow_pt,
                    &mut dirty_pt_slots,
                )
                .expect("step kv growth");
                seq.generated_tokens.push(42);
                seq.last_token = 42;
            }
            online.pre_staged = Some(step + 100);

            // Randomly cancel ~50% of active and waiting requests at the step boundary.
            for req in &online.waiting {
                rng = splitmix64(rng);
                if rng & 1 == 0 {
                    online.cancelled.insert(req.request_id as u64);
                }
            }
            for seq in &online.active {
                rng = splitmix64(rng);
                if rng & 1 == 0 {
                    if rng & 2 == 0 {
                        online.cancelled.insert(seq.request_id as u64);
                    } else {
                        online.completed_stops.insert(seq.request_id as u64);
                    }
                }
            }

            let had_active_cancel = online.active.iter().any(|seq| {
                let id = seq.request_id as u64;
                online.cancelled.contains(&id) || online.completed_stops.contains(&id)
            });
            reclaim_stopped_and_cancelled(
                &mut shadow_pt,
                &mut dirty_pt_slots,
                max_pages_per_seq,
                dummy_page,
                &mut online,
                &metrics,
            );
            if had_active_cancel {
                assert!(
                    online.pre_staged.is_none(),
                    "active cancellation must invalidate pre_staged plan"
                );
            }

            retire_completed_sequences(
                &mut shadow_pt,
                &mut dirty_pt_slots,
                max_pages_per_seq,
                dummy_page,
                &mut online,
                &metrics,
            );
        }

        // Cancel everything left and verify 100% of slots, KV pages, and page-table cells return.
        for req in &online.waiting {
            online.cancelled.insert(req.request_id as u64);
        }
        for seq in &online.active {
            online.cancelled.insert(seq.request_id as u64);
        }
        reclaim_stopped_and_cancelled(
            &mut shadow_pt,
            &mut dirty_pt_slots,
            max_pages_per_seq,
            dummy_page,
            &mut online,
            &metrics,
        );

        assert!(online.active.is_empty());
        assert!(online.waiting.is_empty());
        assert!(online.meta.is_empty());
        assert_eq!(pending.load(Ordering::Relaxed), 0);
        assert_eq!(slot_pool.available(), max_seqs);
        assert_eq!(kv_pool.available(), kv_capacity);
        assert!(shadow_pt.iter().all(|&cell| cell == dummy_page));
    }

    #[test]
    fn eos_retirement_invalidates_prestaged_decode_step() {
        let slot_pool = SlotPool::new(4);
        let kv_pool = BlockPool::new(CacheGroupKind::GlobalAndIndex, 16);
        let mut shadow_pt = vec![16i32; 4 * 4];
        let mut dirty = [0u128; 2];
        let pending = Arc::new(AtomicU64::new(1));
        let metrics = OnlineMetrics::default();
        let events = Arc::new(Mutex::new(Vec::new()));
        let events_sink = Arc::clone(&events);

        let mut online = OnlineState {
            waiting: VecDeque::new(),
            active: Vec::new(),
            slot_to_active: vec![usize::MAX; 4],
            meta: HashMap::new(),
            cancelled: HashSet::new(),
            completed_stops: HashSet::new(),
            eos_tokens: vec![151645],
            next_request_id: 2,
            max_waiting: 8,
            pre_staged: Some(42),
            consecutive_prefills: 0,
            steps: 1,
            pending: Arc::clone(&pending),
        };
        online.meta.insert(
            1,
            OnlineSequenceMeta {
                sink: Arc::new(move |ev| events_sink.lock().expect("lock").push(ev)),
            },
        );
        let slot = slot_pool.allocate().expect("slot");
        let page_lease = kv_pool.allocate(1).expect("kv page");
        let page_id = page_lease.ids()[0];
        online.slot_to_active[slot.id() as usize] = 0;
        online.active.push(ActiveSequence {
            request_id: 1,
            slot_lease: slot,
            kv_leases: vec![page_lease],
            page_ids: vec![page_id],
            prompt_tokens: vec![10, 11],
            prefilled_tokens: 2,
            current_seq_len: 2,
            last_token: 151645,
            generated_tokens: vec![151645],
            generated_top2: Vec::new(),
            target_output_len: 64,
            reserved_pages: 5,
            permit_time: Instant::now(),
            submit_time: Instant::now(),
            first_token_time: Some(Instant::now()),
            last_token_time: Some(Instant::now()),
        });

        let retired =
            retire_completed_sequences(&mut shadow_pt, &mut dirty, 4, 16, &mut online, &metrics);
        assert!(retired);
        assert!(online.pre_staged.is_none());
        assert!(online.active.is_empty());
        assert_eq!(slot_pool.available(), 4);
        assert_eq!(kv_pool.available(), 16);
        assert_eq!(pending.load(Ordering::Relaxed), 0);
        let recorded = events.lock().expect("lock");
        assert!(matches!(
            recorded.as_slice(),
            [LoopEvent::Finished {
                request_id: 1,
                reason: LoopFinishReason::StopToken
            }]
        ));
    }

    #[test]
    fn mock_drain_keeps_loop_alive_until_shutdown_and_tracks_stop_outcomes() {
        let handle = spawn_mock(MockScript {
            tokens: vec![1, 2, 3, 4, 5],
            per_token_delay: Duration::from_millis(20),
            max_waiting: 2,
            max_seqs: 2,
            ..Default::default()
        });
        let done = Arc::new(Mutex::new(Vec::new()));
        let done_clone = Arc::clone(&done);
        let req_id = handle
            .submit(SubmitRequest {
                prompt_tokens: vec![1, 2],
                max_new_tokens: 10,
                sink: Arc::new(move |ev| done_clone.lock().expect("lock").push(ev)),
            })
            .expect("submit");
        handle.complete_stop(req_id);
        std::thread::sleep(Duration::from_millis(30));
        assert_eq!(handle.metrics().finished.load(Ordering::Relaxed), 1);
        assert_eq!(handle.metrics().cancelled.load(Ordering::Relaxed), 0);

        // `drain()` transitions state to Draining and rejects new work, while keeping loop alive.
        handle.drain();
        assert_eq!(handle.state(), EngineState::Draining);
        let rejected = handle.submit(SubmitRequest {
            prompt_tokens: vec![1],
            max_new_tokens: 4,
            sink: Arc::new(|_| {}),
        });
        assert_eq!(
            rejected,
            Err(SubmitError::Unavailable(EngineState::Draining))
        );

        // `shutdown()` + `join()` cleanly terminates the loop thread.
        handle.shutdown();
        assert!(handle.join(Duration::from_secs(2)));
        assert_eq!(handle.state(), EngineState::Draining);
    }
}
