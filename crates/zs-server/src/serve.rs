// Copyright (c) 2026 zstep Contributors
// SPDX-License-Identifier: MIT

//! `zstep serve` / `zs-server serve`: OpenAI-compatible HTTP/SSE front end for the
//! persistent online `ServeLoop` (Slices Z3 + Z4 of the Nebula integration plan).
//!
//! Contract source: `nebula/docs/dev/zstep.md` (four-state probes, `429` vs `503`,
//! request-id propagation, `/metrics` field names & histograms, SSE `[DONE]` uniqueness).

use std::collections::BTreeMap;
use std::net::SocketAddr;
use std::path::{Path, PathBuf};
use std::sync::atomic::{AtomicBool, AtomicU8, AtomicU64, Ordering};
use std::sync::{Arc, Condvar, Mutex};
use std::time::{Duration, Instant, SystemTime, UNIX_EPOCH};

use axum::body::{Body, Bytes};
use axum::extract::State;
use axum::http::{HeaderMap, HeaderValue, StatusCode, header};
use axum::response::{IntoResponse, Response};
use axum::routing::{get, post};
use axum::{Json, Router};
use futures_util::stream;
use serde::Deserialize;
use serde_json::{Value, json};
use tokio::sync::{mpsc, oneshot};
use zs_scheduler::{
    DEFAULT_HEARTBEAT_TIMEOUT, EngineConfig, EngineState, LoopEvent, LoopFinishReason,
    OnlineConfig, OnlineMetrics, ServeLoopHandle, ServingEngine, SubmitError, SubmitRequest,
    TPOT_BUCKET_BOUNDS_SECS, TTFT_BUCKET_BOUNDS_SECS, WorkerLaunchConfig, request_kv_page_demand,
    spawn_serve_loop_with_state,
};
use zs_tokenizer::{
    ChatMessage, ChatTemplateFormat, ChatTemplateOptions, FinishReason as TokenizerFinishReason,
    StreamDecoder, ThinkingMode, Tokenizer, TokenizerError,
};
use zs_wire::{MAX_DECODE_GRAPH_SEQS, ShmGeometry};

const DEFAULT_MAX_TOKENS: usize = 256;
const MAX_STOP_SEQUENCES: usize = 4;
const MAX_STOP_BYTES: usize = 256;
const SSE_CHANNEL_CAPACITY: usize = 256;
const MAX_HTTP_BODY_BYTES: usize = 4 * 1024 * 1024;

pub const SERVE_USAGE: &str = "zstep serve — persistent OpenAI-compatible HTTP/SSE engine\n\
Usage: zstep serve --model-dir <PATH> [OPTIONS]\n\
  --model-dir PATH               Model weights & tokenizer.json directory (required)\n\
  --served-model-name ID         Model ID exposed on /v1/models (default: directory basename)\n\
  --host IP                      Bind address (default: 0.0.0.0)\n\
  --port PORT                    Bind port (default: 43537)\n\
  --world-size N                 GPU TP workers, 1..=8 (default: 8)\n\
  --gpu-memory-utilization F     KV page calibration factor in (0.0, 1.0] (default: 0.95)\n\
  --max-model-len N              Max total prompt + output tokens (default: 32768)\n\
  --max-seqs N                   Max concurrent sequences, 1..=128 (default: 128)\n\
  --max-step-tokens N            Max tokens per step >= max-seqs (default: 4096)\n\
  --max-prefill-chunk N          Max prefill chunk tokens (default: 2048)\n\
  --page-block-tokens N          Physical KV page size (default: 128 on deepseek_v41, 16 otherwise)\n\
  --num-kv-pages N               Explicit physical KV pages per GPU (overrides calibration)\n\
  --kv-cache-dtype DTYPE         KV dtype auto|fp8|bf16 (default: auto)\n\
  --tokenizer-mode MODE          auto | deepseek_v41 | chatml (default: auto)\n\
  --no-enable-prefix-caching     Explicitly disable prefix caching (default in P0)\n\
  --max-waiting-requests N       Bounded HTTP waiting queue depth (default: 1024)\n\
  --drain-timeout-secs N         SIGTERM drain grace period in seconds (default: 30)\n\
  --python PATH                  Python executable for GPU workers\n\
  --worker-root PATH             Python zstep_worker package root\n\
  --ready-timeout-secs N         Worker initialization timeout (default: 1800)\n\
  --step-timeout-secs N          Per-step timeout (default: 300)\n\
  --max-mixed-step-tokens N      Piggyback mixed-step cap (default: 3072 on deepseek_v41, 0=off)\n\
  --allreduce-blocks N           SHM AllReduce CUDA blocks (default: 8)\n\
  --allreduce-threads N          SHM AllReduce threads/block (default: 256)\n";

// ---------------------------------------------------------------------------
// CLI
// ---------------------------------------------------------------------------

/// Parsed `zstep serve` arguments (contract: `nebula/docs/dev/zstep.md` §2.1/§2.2).
#[derive(Debug, Clone)]
pub struct ServeArgs {
    pub model_dir: PathBuf,
    pub served_model_name: String,
    pub host: String,
    pub port: u16,
    pub world_size: u32,
    pub gpu_memory_utilization: f64,
    pub max_model_len: usize,
    pub max_seqs: u32,
    pub max_step_tokens: u32,
    pub max_prefill_chunk: usize,
    pub page_block_tokens: u32,
    pub num_kv_pages: Option<u32>,
    pub kv_cache_dtype: String,
    pub tokenizer_mode: String,
    pub max_waiting_requests: usize,
    pub drain_timeout: Duration,
    pub python: String,
    pub worker_root: PathBuf,
    pub ready_timeout: Duration,
    pub step_timeout: Duration,
    pub max_mixed_step_tokens: Option<usize>,
    pub allreduce_blocks: u32,
    pub allreduce_threads: u32,
}

fn default_python_executable() -> String {
    if let Ok(env_py) = std::env::var("ZSTEP_PYTHON")
        && !env_py.trim().is_empty()
    {
        return env_py;
    }
    for candidate in ["/opt/venv/bin/python", "/usr/bin/python3"] {
        if Path::new(candidate).is_file() {
            return candidate.to_owned();
        }
    }
    "python3".to_owned()
}

fn default_worker_root() -> PathBuf {
    if let Ok(env_root) = std::env::var("ZSTEP_WORKER_ROOT")
        && !env_root.trim().is_empty()
    {
        return PathBuf::from(env_root);
    }
    for candidate in ["./worker", "/workspace/worker"] {
        let path = PathBuf::from(candidate);
        if path.join("zstep_worker").is_dir() {
            return path;
        }
    }
    PathBuf::from("./worker")
}

/// Validate CLI flags, ensuring no flag is missing its value and no unknown option is ignored.
pub fn validate_known_flags(argv: &[String]) -> Result<BTreeMap<String, String>, String> {
    const VALUE_FLAGS: &[&str] = &[
        "--model-dir",
        "--served-model-name",
        "--host",
        "--port",
        "--world-size",
        "--gpu-memory-utilization",
        "--max-model-len",
        "--max-seqs",
        "--max-step-tokens",
        "--max-prefill-chunk",
        "--page-block-tokens",
        "--num-kv-pages",
        "--kv-cache-dtype",
        "--tokenizer-mode",
        "--max-waiting-requests",
        "--drain-timeout-secs",
        "--python",
        "--worker-root",
        "--ready-timeout-secs",
        "--step-timeout-secs",
        "--max-mixed-step-tokens",
        "--allreduce-blocks",
        "--allreduce-threads",
    ];
    let mut values = BTreeMap::new();
    let mut index = 0;
    while index < argv.len() {
        let flag = argv[index].as_str();
        if flag == "--no-enable-prefix-caching" {
            index += 1;
            continue;
        }
        if flag == "--enable-prefix-caching" {
            return Err(
                "--enable-prefix-caching is not supported in P0 online serving; pass --no-enable-prefix-caching"
                    .to_owned(),
            );
        }
        if !flag.starts_with("--") {
            return Err(format!("unexpected positional argument `{flag}`"));
        }
        if !VALUE_FLAGS.contains(&flag) {
            return Err(format!("unknown option `{flag}` for `serve`"));
        }
        let Some(value) = argv.get(index + 1) else {
            return Err(format!("missing value for `{flag}`"));
        };
        if value.starts_with("--") {
            return Err(format!("missing value for `{flag}` (got `{value}`)"));
        }
        values.insert(flag.to_owned(), value.clone());
        index += 2;
    }
    Ok(values)
}

fn parse_map_number<T: std::str::FromStr>(
    values: &BTreeMap<String, String>,
    name: &str,
    default: T,
) -> Result<T, String> {
    match values.get(name) {
        None => Ok(default),
        Some(raw) => raw
            .parse()
            .map_err(|_| format!("invalid value `{raw}` for {name}")),
    }
}

impl ServeArgs {
    pub fn parse(argv: &[String]) -> Result<Self, String> {
        let values = validate_known_flags(argv)?;
        let model_dir_raw = values
            .get("--model-dir")
            .ok_or_else(|| "--model-dir is required for `serve`".to_owned())?;
        let model_dir = PathBuf::from(model_dir_raw);
        let served_model_name = values
            .get("--served-model-name")
            .cloned()
            .unwrap_or_else(|| {
                model_dir
                    .file_name()
                    .map(|name| name.to_string_lossy().into_owned())
                    .unwrap_or_else(|| "zstep".to_owned())
            });
        let world_size: u32 = parse_map_number(&values, "--world-size", 8u32)?;
        if !(1..=8).contains(&world_size) {
            return Err(format!("--world-size {world_size} must be in 1..=8"));
        }
        let max_model_len: usize = parse_map_number(&values, "--max-model-len", 32_768)?;
        if max_model_len == 0 {
            return Err("--max-model-len must be positive".to_owned());
        }
        let is_dsv41 = is_deepseek_v41_dir(&model_dir);
        let default_page_block = if is_dsv41 { 128 } else { 16 };
        let page_block_tokens: u32 =
            parse_map_number(&values, "--page-block-tokens", default_page_block)?;
        if page_block_tokens == 0 {
            return Err("--page-block-tokens must be positive".to_owned());
        }
        let max_seqs: u32 = parse_map_number(&values, "--max-seqs", 128)?;
        if max_seqs == 0 || max_seqs > MAX_DECODE_GRAPH_SEQS {
            return Err(format!(
                "--max-seqs {max_seqs} is outside supported decode CUDA Graph range 1..={MAX_DECODE_GRAPH_SEQS}"
            ));
        }
        let max_step_tokens: u32 = parse_map_number(&values, "--max-step-tokens", 4096)?;
        if max_step_tokens < max_seqs {
            return Err(format!(
                "--max-step-tokens ({max_step_tokens}) must be >= --max-seqs ({max_seqs})"
            ));
        }
        let max_prefill_chunk: usize = parse_map_number(&values, "--max-prefill-chunk", 2048usize)?;
        if max_prefill_chunk == 0 {
            return Err("--max-prefill-chunk must be positive".to_owned());
        }
        let gpu_memory_utilization: f64 =
            parse_map_number(&values, "--gpu-memory-utilization", 0.95_f64)?;
        if !(0.0..=1.0).contains(&gpu_memory_utilization) || gpu_memory_utilization == 0.0 {
            return Err("--gpu-memory-utilization must be in (0.0, 1.0]".to_owned());
        }
        let num_kv_pages = values
            .get("--num-kv-pages")
            .map(|raw| raw.parse::<u32>())
            .transpose()
            .map_err(|_| "--num-kv-pages must be a positive integer".to_owned())?;
        if num_kv_pages == Some(0) {
            return Err("--num-kv-pages must be positive".to_owned());
        }
        let kv_cache_dtype = values
            .get("--kv-cache-dtype")
            .cloned()
            .unwrap_or_else(|| "auto".to_owned());
        if !matches!(
            kv_cache_dtype.as_str(),
            "auto" | "fp8" | "fp8_e4m3" | "bf16" | "bfloat16"
        ) {
            return Err(format!("unsupported --kv-cache-dtype `{kv_cache_dtype}`"));
        }
        if is_dsv41 && !matches!(kv_cache_dtype.as_str(), "auto" | "fp8" | "fp8_e4m3") {
            return Err(format!(
                "--kv-cache-dtype `{kv_cache_dtype}` is incompatible with deepseek_v41 runner (uses fp8_ds_mla; pass `auto` or `fp8`)"
            ));
        }
        if !is_dsv41 && !matches!(kv_cache_dtype.as_str(), "auto" | "bf16" | "bfloat16") {
            return Err(format!(
                "--kv-cache-dtype `{kv_cache_dtype}` is incompatible with BF16 model runner (pass `auto` or `bf16`)"
            ));
        }
        let tokenizer_mode = values
            .get("--tokenizer-mode")
            .cloned()
            .unwrap_or_else(|| "auto".to_owned());
        if !matches!(tokenizer_mode.as_str(), "auto" | "deepseek_v41" | "chatml") {
            return Err(format!("unsupported --tokenizer-mode `{tokenizer_mode}`"));
        }
        let max_waiting_requests: usize =
            parse_map_number(&values, "--max-waiting-requests", 1024)?;
        if max_waiting_requests == 0 {
            return Err("--max-waiting-requests must be positive".to_owned());
        }
        let drain_timeout_secs: u64 = parse_map_number(&values, "--drain-timeout-secs", 30)?;
        let ready_timeout_secs: u64 = parse_map_number(&values, "--ready-timeout-secs", 1800u64)?;
        let step_timeout_secs: u64 = parse_map_number(&values, "--step-timeout-secs", 300u64)?;
        if ready_timeout_secs == 0 || step_timeout_secs == 0 {
            return Err("--ready-timeout-secs and --step-timeout-secs must be positive".to_owned());
        }
        Ok(Self {
            model_dir,
            served_model_name,
            host: values
                .get("--host")
                .cloned()
                .unwrap_or_else(|| "0.0.0.0".to_owned()),
            port: parse_map_number(&values, "--port", 43537u16)?,
            world_size,
            gpu_memory_utilization,
            max_model_len,
            max_seqs,
            max_step_tokens,
            max_prefill_chunk,
            page_block_tokens,
            num_kv_pages,
            kv_cache_dtype,
            tokenizer_mode,
            max_waiting_requests,
            drain_timeout: Duration::from_secs(drain_timeout_secs),
            python: values
                .get("--python")
                .cloned()
                .unwrap_or_else(default_python_executable),
            worker_root: values
                .get("--worker-root")
                .map(PathBuf::from)
                .unwrap_or_else(default_worker_root),
            ready_timeout: Duration::from_secs(ready_timeout_secs),
            step_timeout: Duration::from_secs(step_timeout_secs),
            max_mixed_step_tokens: values
                .get("--max-mixed-step-tokens")
                .map(|raw| raw.parse::<usize>())
                .transpose()
                .map_err(|_| "--max-mixed-step-tokens must be an integer".to_owned())?,
            allreduce_blocks: parse_map_number(&values, "--allreduce-blocks", 8u32)?,
            allreduce_threads: parse_map_number(&values, "--allreduce-threads", 256u32)?,
        })
    }

    pub fn resolved_num_kv_pages(&self) -> u32 {
        self.num_kv_pages.unwrap_or_else(|| {
            let base = if is_deepseek_v41_dir(&self.model_dir) {
                2_560.0
            } else {
                49_152.0
            };
            let scaled = (self.gpu_memory_utilization / 0.95) * base;
            (scaled.round() as u32).max(128)
        })
    }

    pub fn resolved_max_pages_per_seq(&self) -> u32 {
        self.max_model_len
            .div_ceil(self.page_block_tokens as usize)
            .max(1) as u32
    }

    fn engine_config(&self, cancel_flag: Option<Arc<AtomicBool>>) -> EngineConfig {
        let max_pages_per_seq = self.resolved_max_pages_per_seq();
        let num_kv_pages = self.resolved_num_kv_pages();
        EngineConfig {
            shm_name: unique_shm_name(&self.served_model_name),
            geometry: ShmGeometry {
                world_size: self.world_size,
                max_seqs: self.max_seqs,
                max_step_tokens: self.max_step_tokens,
                max_decoder_rows: self.max_step_tokens,
                max_pages_per_seq,
                page_block_tokens: self.page_block_tokens,
                spec_verify_width: 6,
            },
            max_prefill_chunk: self.max_prefill_chunk,
            steps: 1,
            token_counts: vec![1],
            launch: WorkerLaunchConfig {
                python_executable: PathBuf::from(&self.python),
                worker_root: self.worker_root.clone(),
                hidden_size: 5120,
                output_features: 129_280,
                router_experts: 384,
                router_top_k: 6,
                allreduce_blocks: self.allreduce_blocks,
                allreduce_threads: self.allreduce_threads,
                model_dir: Some(self.model_dir.clone()),
                num_kv_pages,
                max_model_len: Some(self.max_model_len),
                cancel_flag,
            },
            ready_timeout: self.ready_timeout,
            step_timeout: self.step_timeout,
            heartbeat_timeout: DEFAULT_HEARTBEAT_TIMEOUT,
        }
    }
}

fn is_deepseek_v41_dir(model_dir: &Path) -> bool {
    let config = model_dir.join("config.json");
    std::fs::read_to_string(config)
        .ok()
        .and_then(|raw| serde_json::from_str::<Value>(&raw).ok())
        .and_then(|value| {
            value
                .get("model_type")
                .and_then(|kind| kind.as_str())
                .map(str::to_owned)
        })
        .is_some_and(|kind| kind == "deepseek_v41")
}

fn unique_shm_name(label: &str) -> String {
    let nanos = SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .map(|elapsed| elapsed.as_nanos())
        .unwrap_or(0);
    let safe: String = label
        .chars()
        .take(16)
        .map(|ch| if ch.is_ascii_alphanumeric() { ch } else { '_' })
        .collect();
    format!("zstep_srv_{safe}_{}_{nanos:x}", std::process::id())
}

// ---------------------------------------------------------------------------
// Shared state
// ---------------------------------------------------------------------------

struct AppState {
    phase: Arc<AtomicU8>,
    handle: Mutex<Option<ServeLoopHandle>>,
    shutdown_requested: Arc<AtomicBool>,
    tokenizer: Tokenizer,
    served_model_name: String,
    tokenizer_mode: String,
    max_model_len: usize,
    page_block_tokens: usize,
    max_pages_per_seq: usize,
    num_kv_pages: usize,
    created_at: u64,
    started: Instant,
    request_counter: AtomicU64,
}

impl AppState {
    fn phase(&self) -> EngineState {
        EngineState::from_u8(self.phase.load(Ordering::Acquire))
    }

    fn set_phase(&self, state: EngineState) {
        self.phase.store(state as u8, Ordering::Release);
    }

    fn handle(&self) -> Option<ServeLoopHandle> {
        self.handle.lock().ok().and_then(|guard| guard.clone())
    }

    fn metrics(&self) -> Option<Arc<OnlineMetrics>> {
        self.handle().map(|handle| handle.metrics())
    }

    fn template_format(&self) -> ChatTemplateFormat {
        match self.tokenizer_mode.as_str() {
            "deepseek_v41" => ChatTemplateFormat::DeepSeekV41,
            "chatml" => ChatTemplateFormat::ChatMl,
            _ => self.tokenizer.default_template_format(),
        }
    }
}

fn error_body(kind: &str, code: &str, message: &str) -> Value {
    json!({"error": {"message": message, "type": kind, "code": code}})
}

fn error_response(status: StatusCode, kind: &str, code: &str, message: &str) -> Response {
    (status, Json(error_body(kind, code, message))).into_response()
}

fn with_request_id(mut response: Response, request_id: &str) -> Response {
    let val = HeaderValue::from_str(request_id).unwrap_or_else(|_| HeaderValue::from_static("req"));
    response.headers_mut().insert("x-request-id", val);
    response
}

fn unavailable_code(state: EngineState) -> &'static str {
    match state {
        EngineState::Starting | EngineState::Ready => "engine_starting",
        EngineState::Draining => "engine_draining",
        EngineState::Faulted => "engine_faulted",
    }
}

fn lifecycle_error(state: &AppState) -> Option<Response> {
    let phase = state.phase();
    if phase == EngineState::Ready {
        return None;
    }
    Some(error_response(
        StatusCode::SERVICE_UNAVAILABLE,
        "server_error",
        unavailable_code(phase),
        "zstep replica is not ready to serve requests",
    ))
}

// ---------------------------------------------------------------------------
// Router & operational probes
// ---------------------------------------------------------------------------

fn router(state: Arc<AppState>) -> Router {
    Router::new()
        .route("/livez", get(livez))
        .route("/health", get(health))
        .route("/ready", get(ready))
        .route("/metrics", get(metrics))
        .route("/admin/drain", post(admin_drain))
        .route("/v1/models", get(list_models))
        .route("/v1/chat/completions", post(chat_completions))
        .with_state(state)
}

fn publish_launched_handle(state: &AppState, handle: ServeLoopHandle) {
    let mut guard = state.handle.lock().expect("handle mutex");
    *guard = Some(handle.clone());
    if let Err(cur) = state.phase.compare_exchange(
        EngineState::Starting as u8,
        EngineState::Ready as u8,
        Ordering::AcqRel,
        Ordering::Acquire,
    ) && EngineState::from_u8(cur) == EngineState::Draining
    {
        handle.drain();
    }
}

fn request_drain(state: &AppState) {
    let guard = state.handle.lock().expect("handle mutex");
    if state.phase() != EngineState::Faulted {
        state.set_phase(EngineState::Draining);
    }
    if let Some(handle) = guard.as_ref() {
        handle.drain();
    }
}

async fn livez() -> StatusCode {
    StatusCode::OK
}

async fn health(State(state): State<Arc<AppState>>) -> Response {
    match state.phase() {
        EngineState::Ready => (StatusCode::OK, Json(json!({"status": "ok"}))).into_response(),
        // Draining MUST keep `/health == 200` once launched so Nebula continues scraping `/metrics`
        // until `pending_requests == 0` (`nebula/docs/dev/zstep.md` §3). If drained before engine
        // warmup completed (`handle == None`), stay `503 starting` until launch finishes.
        EngineState::Draining if state.handle().is_some() => {
            (StatusCode::OK, Json(json!({"status": "draining"}))).into_response()
        }
        EngineState::Starting | EngineState::Draining => (
            StatusCode::SERVICE_UNAVAILABLE,
            Json(json!({"status": "starting"})),
        )
            .into_response(),
        EngineState::Faulted => (
            StatusCode::SERVICE_UNAVAILABLE,
            Json(json!({"status": "faulted"})),
        )
            .into_response(),
    }
}

async fn ready(State(state): State<Arc<AppState>>) -> Response {
    match state.phase() {
        EngineState::Ready => (StatusCode::OK, Json(json!({"status": "ready"}))).into_response(),
        EngineState::Draining => (
            StatusCode::SERVICE_UNAVAILABLE,
            Json(json!({"status": "draining"})),
        )
            .into_response(),
        EngineState::Starting => (
            StatusCode::SERVICE_UNAVAILABLE,
            Json(json!({"status": "starting"})),
        )
            .into_response(),
        EngineState::Faulted => (
            StatusCode::SERVICE_UNAVAILABLE,
            Json(json!({"status": "faulted"})),
        )
            .into_response(),
    }
}

async fn list_models(State(state): State<Arc<AppState>>) -> Response {
    if let Some(response) = lifecycle_error(&state) {
        return response;
    }
    (
        StatusCode::OK,
        Json(json!({
            "object": "list",
            "data": [{
                "id": state.served_model_name,
                "object": "model",
                "created": state.created_at,
                "owned_by": "zstep",
                "max_model_len": state.max_model_len,
            }]
        })),
    )
        .into_response()
}

async fn admin_drain(State(state): State<Arc<AppState>>) -> Response {
    request_drain(&state);
    (StatusCode::OK, Json(json!({"status": "draining"}))).into_response()
}

async fn metrics(State(state): State<Arc<AppState>>) -> Response {
    let text = render_metrics(&state);
    (
        StatusCode::OK,
        [(header::CONTENT_TYPE, "text/plain; version=0.0.4")],
        text,
    )
        .into_response()
}

fn append_histogram<const N: usize>(
    out: &mut String,
    name: &str,
    help: &str,
    bounds: &[f64; N],
    buckets: &[AtomicU64; N],
    count: u64,
    sum_secs: f64,
) {
    out.push_str(&format!("# HELP {name} {help}\n# TYPE {name} histogram\n"));
    for (bound, counter) in bounds.iter().zip(buckets) {
        let cumulative = counter.load(Ordering::Relaxed);
        out.push_str(&format!("{name}_bucket{{le=\"{bound}\"}} {cumulative}\n"));
    }
    out.push_str(&format!("{name}_bucket{{le=\"+Inf\"}} {count}\n"));
    out.push_str(&format!("{name}_sum {sum_secs:.6}\n"));
    out.push_str(&format!("{name}_count {count}\n"));
}

fn render_metrics(state: &AppState) -> String {
    let mut out = String::new();
    let Some(metrics) = state.metrics() else {
        out.push_str(
            "# HELP zstep:num_requests_waiting Requests accepted by HTTP and waiting for a slot lease\n# TYPE zstep:num_requests_waiting gauge\nzstep:num_requests_waiting 0\n",
        );
        out.push_str(
            "# HELP zstep:num_requests_running Active sequences holding a slot lease (prefill + decode)\n# TYPE zstep:num_requests_running gauge\nzstep:num_requests_running 0\n",
        );
        out.push_str(
            "# HELP zstep:kv_cache_usage_perc Physical KV page occupancy in [0,1]\n# TYPE zstep:kv_cache_usage_perc gauge\nzstep:kv_cache_usage_perc 0\n",
        );
        out.push_str(
            "# HELP zstep:num_preemptions_total Sequence preemptions\n# TYPE zstep:num_preemptions_total counter\nzstep:num_preemptions_total 0\n",
        );
        out.push_str(
            "# HELP zstep:triton_recompilations_total Runtime Triton JIT recompilations\n# TYPE zstep:triton_recompilations_total gauge\nzstep:triton_recompilations_total 0\n",
        );
        return out;
    };

    let gauges = [
        (
            "zstep:num_requests_waiting",
            metrics.waiting.load(Ordering::Relaxed) as f64,
            "Requests accepted by HTTP and waiting for a slot lease",
        ),
        (
            "zstep:num_requests_running",
            metrics.running.load(Ordering::Relaxed) as f64,
            "Active sequences holding a slot lease (prefill + decode)",
        ),
        (
            "zstep:kv_cache_usage_perc",
            metrics.kv_cache_usage(),
            "Physical KV page occupancy in [0,1]",
        ),
    ];
    for (name, value, help) in gauges {
        out.push_str(&format!("# HELP {name} {help}\n# TYPE {name} gauge\n"));
        out.push_str(&format!("{name} {value}\n"));
    }
    out.push_str("# HELP zstep:num_preemptions_total Sequence preemptions\n# TYPE zstep:num_preemptions_total counter\n");
    out.push_str(&format!(
        "zstep:num_preemptions_total {}\n",
        metrics.preemptions.load(Ordering::Relaxed)
    ));
    let outcomes = [
        ("finished", metrics.finished.load(Ordering::Relaxed)),
        ("cancelled", metrics.cancelled.load(Ordering::Relaxed)),
        (
            "rejected_overloaded",
            metrics.rejected_overloaded.load(Ordering::Relaxed),
        ),
        ("failed", metrics.failed.load(Ordering::Relaxed)),
    ];
    out.push_str("# HELP zstep:request_outcomes_total Request terminal outcomes\n# TYPE zstep:request_outcomes_total counter\n");
    for (outcome, value) in outcomes {
        out.push_str(&format!(
            "zstep:request_outcomes_total{{outcome=\"{outcome}\"}} {value}\n"
        ));
    }

    let ttft_count = metrics.ttft_count.load(Ordering::Relaxed);
    let ttft_sum = metrics.ttft_micros.load(Ordering::Relaxed) as f64 / 1e6;
    append_histogram(
        &mut out,
        "zstep:time_to_first_token_seconds",
        "Time to first token in seconds",
        &TTFT_BUCKET_BOUNDS_SECS,
        &metrics.ttft_buckets,
        ttft_count,
        ttft_sum,
    );

    let tpot_count = metrics.tpot_count.load(Ordering::Relaxed);
    let tpot_sum = metrics.tpot_micros.load(Ordering::Relaxed) as f64 / 1e6;
    append_histogram(
        &mut out,
        "zstep:time_per_output_token_seconds",
        "Inter-token latency in seconds",
        &TPOT_BUCKET_BOUNDS_SECS,
        &metrics.tpot_buckets,
        tpot_count,
        tpot_sum,
    );

    out.push_str("# HELP zstep:triton_recompilations_total Runtime Triton JIT recompilations\n# TYPE zstep:triton_recompilations_total gauge\n");
    out.push_str(&format!(
        "zstep:triton_recompilations_total {}\n",
        metrics.triton_recompilations.load(Ordering::Relaxed)
    ));
    out.push_str(&format!(
        "# zstep uptime_seconds {:.3}\n",
        state.started.elapsed().as_secs_f64()
    ));
    out
}

// ---------------------------------------------------------------------------
// Chat completions
// ---------------------------------------------------------------------------

#[derive(Debug, Default, Deserialize)]
struct StreamOptions {
    #[serde(default)]
    include_usage: bool,
    #[serde(flatten)]
    extra: BTreeMap<String, Value>,
}

#[derive(Debug, Default, Deserialize)]
struct ChatTemplateKwargs {
    #[serde(default)]
    enable_thinking: Option<bool>,
    #[serde(default)]
    thinking_mode: Option<String>,
    #[serde(flatten)]
    extra: BTreeMap<String, Value>,
}

#[derive(Debug, Deserialize, Clone)]
#[serde(untagged)]
enum StopField {
    One(String),
    Many(Vec<String>),
}

impl StopField {
    fn into_vec(self) -> Vec<String> {
        match self {
            Self::One(value) => vec![value],
            Self::Many(values) => values,
        }
    }
}

#[derive(Debug, Deserialize)]
struct ChatRequest {
    #[serde(default)]
    model: Option<String>,
    messages: Vec<ChatMessage>,
    #[serde(default)]
    max_tokens: Option<usize>,
    #[serde(default)]
    max_completion_tokens: Option<usize>,
    #[serde(default)]
    stream: Option<bool>,
    #[serde(default)]
    stream_options: Option<StreamOptions>,
    #[serde(default)]
    stop: Option<StopField>,
    #[serde(default)]
    temperature: Option<f64>,
    #[serde(default)]
    top_p: Option<f64>,
    #[serde(default)]
    n: Option<u32>,
    #[serde(default)]
    seed: Option<i64>,
    #[serde(default)]
    ignore_eos: Option<bool>,
    #[serde(default)]
    tools: Option<Value>,
    #[serde(default)]
    tool_choice: Option<Value>,
    #[serde(default)]
    response_format: Option<Value>,
    #[serde(default)]
    logprobs: Option<bool>,
    #[serde(default)]
    top_logprobs: Option<u32>,
    #[serde(default)]
    presence_penalty: Option<f64>,
    #[serde(default)]
    frequency_penalty: Option<f64>,
    #[serde(default)]
    reasoning_effort: Option<String>,
    #[serde(default)]
    chat_template_kwargs: Option<ChatTemplateKwargs>,
    #[serde(default)]
    user: Option<String>,
    #[serde(flatten)]
    extra: BTreeMap<String, Value>,
}

fn extract_request_id(headers: &HeaderMap, state: &AppState) -> String {
    for header_name in ["x-nebula-request-id", "x-request-id"] {
        if let Some(value) = headers.get(header_name)
            && let Ok(text) = value.to_str()
            && !text.trim().is_empty()
        {
            return text.trim().to_owned();
        }
    }
    let counter = state.request_counter.fetch_add(1, Ordering::Relaxed);
    let nanos = SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .map(|elapsed| elapsed.as_nanos())
        .unwrap_or(0);
    format!("chatcmpl-{nanos:x}-{counter:x}")
}

fn extract_traceparent(headers: &HeaderMap) -> Option<&str> {
    headers
        .get("traceparent")
        .and_then(|v| v.to_str().ok())
        .map(str::trim)
        .filter(|s| !s.is_empty())
}

fn reject_unsupported(request: &ChatRequest) -> Option<Response> {
    if request.max_tokens == Some(0) || request.max_completion_tokens == Some(0) {
        return Some(error_response(
            StatusCode::BAD_REQUEST,
            "invalid_request_error",
            "invalid_parameter",
            "max_tokens / max_completion_tokens must be >= 1",
        ));
    }
    if let Some(n) = request.n
        && n != 1
    {
        return Some(error_response(
            StatusCode::BAD_REQUEST,
            "invalid_request_error",
            "invalid_parameter",
            &format!("n={n} is not supported"),
        ));
    }
    if request.temperature.is_some_and(|value| value != 0.0) {
        return Some(error_response(
            StatusCode::BAD_REQUEST,
            "invalid_request_error",
            "invalid_parameter",
            "temperature sampling is not supported in P0 (greedy temperature=0 only)",
        ));
    }
    if request.top_p.is_some_and(|value| value != 1.0) {
        return Some(error_response(
            StatusCode::BAD_REQUEST,
            "invalid_request_error",
            "invalid_parameter",
            "top_p sampling is not supported in P0 (greedy only)",
        ));
    }
    if request.presence_penalty.is_some_and(|value| value != 0.0)
        || request.frequency_penalty.is_some_and(|value| value != 0.0)
    {
        return Some(error_response(
            StatusCode::BAD_REQUEST,
            "invalid_request_error",
            "invalid_parameter",
            "presence/frequency penalties are not supported",
        ));
    }
    if request.tools.is_some() || request.tool_choice.is_some() {
        return Some(error_response(
            StatusCode::BAD_REQUEST,
            "invalid_request_error",
            "invalid_parameter",
            "tool calling is not supported",
        ));
    }
    if request.response_format.is_some() {
        return Some(error_response(
            StatusCode::BAD_REQUEST,
            "invalid_request_error",
            "invalid_parameter",
            "response_format is not supported",
        ));
    }
    if request.logprobs.is_some_and(|v| v) || request.top_logprobs.is_some() {
        return Some(error_response(
            StatusCode::BAD_REQUEST,
            "invalid_request_error",
            "invalid_parameter",
            "logprobs are not supported",
        ));
    }
    if request.ignore_eos.is_some_and(|v| v) {
        return Some(error_response(
            StatusCode::BAD_REQUEST,
            "invalid_request_error",
            "invalid_parameter",
            "ignore_eos is not supported",
        ));
    }
    if let Some(opts) = &request.stream_options
        && let Some(unknown) = opts.extra.keys().next()
    {
        return Some(error_response(
            StatusCode::BAD_REQUEST,
            "invalid_request_error",
            "invalid_parameter",
            &format!("unsupported stream_options parameter `{unknown}`"),
        ));
    }
    if let Some(kwargs) = &request.chat_template_kwargs
        && let Some(unknown) = kwargs.extra.keys().next()
    {
        return Some(error_response(
            StatusCode::BAD_REQUEST,
            "invalid_request_error",
            "invalid_parameter",
            &format!("unsupported chat_template_kwargs parameter `{unknown}`"),
        ));
    }
    if let Some((unknown, value)) = request.extra.iter().find(|(_, v)| !v.is_null()) {
        return Some(error_response(
            StatusCode::BAD_REQUEST,
            "invalid_request_error",
            "invalid_parameter",
            &format!("unsupported parameter `{unknown}` ({value})"),
        ));
    }
    None
}

#[allow(clippy::result_large_err)]
fn normalise_stops(stops: Option<StopField>) -> Result<Vec<String>, Response> {
    let Some(stop) = stops else {
        return Ok(Vec::new());
    };
    let values = stop.into_vec();
    if values.len() > MAX_STOP_SEQUENCES {
        return Err(error_response(
            StatusCode::BAD_REQUEST,
            "invalid_request_error",
            "invalid_parameter",
            &format!("at most {MAX_STOP_SEQUENCES} stop strings are supported"),
        ));
    }
    let mut out = Vec::with_capacity(values.len());
    for value in values {
        if value.is_empty() {
            return Err(error_response(
                StatusCode::BAD_REQUEST,
                "invalid_request_error",
                "invalid_parameter",
                "stop strings must not be empty",
            ));
        }
        if value.len() > MAX_STOP_BYTES {
            return Err(error_response(
                StatusCode::BAD_REQUEST,
                "invalid_request_error",
                "invalid_parameter",
                &format!("stop strings must be <= {MAX_STOP_BYTES} bytes"),
            ));
        }
        out.push(value);
    }
    Ok(out)
}

fn chunk_json(
    id: &str,
    model: &str,
    created: u64,
    delta: Value,
    finish_reason: Option<&str>,
) -> String {
    json!({
        "id": id,
        "object": "chat.completion.chunk",
        "created": created,
        "model": model,
        "choices": [{
            "index": 0,
            "delta": delta,
            "finish_reason": finish_reason,
        }],
    })
    .to_string()
}

fn usage_chunk_json(
    id: &str,
    model: &str,
    created: u64,
    prompt_tokens: usize,
    completion_tokens: usize,
) -> String {
    json!({
        "id": id,
        "object": "chat.completion.chunk",
        "created": created,
        "model": model,
        "choices": [],
        "usage": {
            "prompt_tokens": prompt_tokens,
            "completion_tokens": completion_tokens,
            "total_tokens": prompt_tokens + completion_tokens,
        }
    })
    .to_string()
}

fn finish_reason_text(reason: LoopFinishReason) -> &'static str {
    match reason {
        LoopFinishReason::Length => "length",
        LoopFinishReason::StopToken | LoopFinishReason::Cancelled | LoopFinishReason::Failed => {
            "stop"
        }
    }
}

struct CancelGuard {
    handle: Option<ServeLoopHandle>,
    request_id: u64,
    armed: bool,
}

impl CancelGuard {
    fn disarm(&mut self) {
        self.armed = false;
    }
}

impl Drop for CancelGuard {
    fn drop(&mut self) {
        if self.armed
            && let Some(handle) = self.handle.take()
        {
            handle.cancel(self.request_id);
        }
    }
}

fn resolve_thinking_mode(request: &ChatRequest) -> Result<ThinkingMode, String> {
    if let Some(kwargs) = &request.chat_template_kwargs {
        if let Some(mode_str) = kwargs.thinking_mode.as_deref() {
            return match mode_str {
                "thinking" => Ok(ThinkingMode::Thinking),
                "chat" => Ok(ThinkingMode::Chat),
                other => Err(format!(
                    "invalid chat_template_kwargs.thinking_mode `{other}`"
                )),
            };
        }
        if let Some(enable) = kwargs.enable_thinking {
            return Ok(if enable {
                ThinkingMode::Thinking
            } else {
                ThinkingMode::Chat
            });
        }
    }
    if request.reasoning_effort.is_some() {
        return Ok(ThinkingMode::Thinking);
    }
    Ok(ThinkingMode::Chat)
}

async fn chat_completions(
    State(state): State<Arc<AppState>>,
    headers: HeaderMap,
    body: Body,
) -> Response {
    let request_id = extract_request_id(&headers, &state);
    let traceparent = extract_traceparent(&headers).map(str::to_owned);

    let raw_bytes = match axum::body::to_bytes(body, MAX_HTTP_BODY_BYTES).await {
        Ok(bytes) => bytes,
        Err(_) => {
            return with_request_id(
                error_response(
                    StatusCode::PAYLOAD_TOO_LARGE,
                    "invalid_request_error",
                    "payload_too_large",
                    "request body exceeds maximum size of 4 MiB",
                ),
                &request_id,
            );
        }
    };
    let request: ChatRequest = match serde_json::from_slice(&raw_bytes) {
        Ok(parsed) => parsed,
        Err(err) => {
            return with_request_id(
                error_response(
                    StatusCode::BAD_REQUEST,
                    "invalid_request_error",
                    "invalid_json",
                    &format!("invalid chat completion JSON body: {err}"),
                ),
                &request_id,
            );
        }
    };

    if let Some(response) = lifecycle_error(&state) {
        return with_request_id(response, &request_id);
    }
    if let Some(model) = request.model.as_deref()
        && model != state.served_model_name
    {
        return with_request_id(
            error_response(
                StatusCode::BAD_REQUEST,
                "invalid_request_error",
                "model_not_found",
                &format!(
                    "model `{model}` is not served by this replica (`{}`)",
                    state.served_model_name
                ),
            ),
            &request_id,
        );
    }
    if let Some(response) = reject_unsupported(&request) {
        return with_request_id(response, &request_id);
    }
    let stops = match normalise_stops(request.stop.clone()) {
        Ok(stops) => stops,
        Err(response) => return with_request_id(response, &request_id),
    };
    if request.messages.is_empty() {
        return with_request_id(
            error_response(
                StatusCode::BAD_REQUEST,
                "invalid_request_error",
                "invalid_parameter",
                "messages must not be empty",
            ),
            &request_id,
        );
    }
    let thinking_mode = match resolve_thinking_mode(&request) {
        Ok(mode) => mode,
        Err(message) => {
            return with_request_id(
                error_response(
                    StatusCode::BAD_REQUEST,
                    "invalid_request_error",
                    "invalid_parameter",
                    &message,
                ),
                &request_id,
            );
        }
    };
    let options = ChatTemplateOptions {
        format: state.template_format(),
        thinking_mode,
        reasoning_effort: request.reasoning_effort.clone(),
        ..ChatTemplateOptions::default()
    };
    let encoded = match state.tokenizer.encode_chat(&request.messages, &options) {
        Ok(encoded) => encoded,
        Err(error) => {
            return with_request_id(
                error_response(
                    StatusCode::BAD_REQUEST,
                    "invalid_request_error",
                    "invalid_parameter",
                    &error.to_string(),
                ),
                &request_id,
            );
        }
    };
    let prompt_tokens = encoded.token_ids.len();
    let max_new_tokens = request
        .max_completion_tokens
        .or(request.max_tokens)
        .unwrap_or(DEFAULT_MAX_TOKENS);
    if let Err(error) =
        state
            .tokenizer
            .validate_lengths(prompt_tokens, max_new_tokens, state.max_model_len)
    {
        let code = match error {
            TokenizerError::ContextLengthExceeded { .. } => "context_length_exceeded",
            _ => "invalid_parameter",
        };
        return with_request_id(
            error_response(
                StatusCode::BAD_REQUEST,
                "invalid_request_error",
                code,
                &error.to_string(),
            ),
            &request_id,
        );
    }
    let page_demand =
        request_kv_page_demand(prompt_tokens, max_new_tokens, state.page_block_tokens);
    if page_demand > state.max_pages_per_seq || page_demand > state.num_kv_pages {
        return with_request_id(
            error_response(
                StatusCode::BAD_REQUEST,
                "invalid_request_error",
                "context_length_exceeded",
                &format!(
                    "request requires {page_demand} KV pages, exceeding max_pages_per_seq={} or num_kv_pages={}",
                    state.max_pages_per_seq, state.num_kv_pages
                ),
            ),
            &request_id,
        );
    }

    let Some(handle) = state.handle() else {
        return with_request_id(
            error_response(
                StatusCode::SERVICE_UNAVAILABLE,
                "server_error",
                unavailable_code(state.phase()),
                "engine is not ready",
            ),
            &request_id,
        );
    };

    let stream_mode = request.stream.unwrap_or(false);
    let include_usage = request
        .stream_options
        .as_ref()
        .is_some_and(|options| options.include_usage);
    let prompt_ids: Vec<i32> = encoded.token_ids.iter().map(|id| *id as i32).collect();

    if stream_mode {
        let (tx, rx) = mpsc::channel::<LoopEvent>(SSE_CHANNEL_CAPACITY);
        let overflowed = Arc::new(AtomicBool::new(false));
        let assigned_loop_id = Arc::new(AtomicU64::new(0));
        let sink: zs_scheduler::EventSink = {
            let overflowed = Arc::clone(&overflowed);
            let assigned_loop_id = Arc::clone(&assigned_loop_id);
            let handle_for_sink = handle.clone();
            Arc::new(move |event: LoopEvent| {
                if tx.try_send(event).is_err() {
                    overflowed.store(true, Ordering::Release);
                    let id = assigned_loop_id.load(Ordering::Acquire);
                    if id != 0 {
                        handle_for_sink.cancel(id);
                    }
                }
            })
        };

        let loop_request_id = match handle.submit(SubmitRequest {
            prompt_tokens: prompt_ids,
            max_new_tokens,
            sink,
        }) {
            Ok(id) => {
                assigned_loop_id.store(id, Ordering::Release);
                if overflowed.load(Ordering::Acquire) {
                    handle.cancel(id);
                }
                id
            }
            Err(err) => return with_request_id(map_submit_error(err), &request_id),
        };

        println!(
            "zstep_serve request_id={request_id} traceparent={:?} loop_id={loop_request_id} prompt_tokens={prompt_tokens} max_tokens={max_new_tokens} stream=true stops={} seed={:?} user={:?}",
            traceparent,
            stops.len(),
            request.seed,
            request.user
        );

        let created = now_secs();
        let model = state.served_model_name.clone();
        let decoder = StreamDecoder::new(&state.tokenizer, stops);
        let guard = CancelGuard {
            handle: Some(handle.clone()),
            request_id: loop_request_id,
            armed: true,
        };
        let req_id_for_stream = request_id.clone();

        let event_stream = stream::unfold(
            (
                rx,
                decoder,
                guard,
                handle,
                loop_request_id,
                req_id_for_stream,
                model,
                created,
                include_usage,
                prompt_tokens,
                false,
                false,
            ),
            |(
                mut rx,
                mut decoder,
                mut guard,
                handle,
                loop_id,
                req_id,
                model,
                created,
                include_usage,
                prompt_tokens,
                sent_role,
                sent_done,
            )| async move {
                if sent_done {
                    return None;
                }
                let mut payloads: Vec<String> = Vec::new();
                match rx.recv().await {
                    Some(LoopEvent::Token { token, .. }) => {
                        if !sent_role {
                            payloads.push(format!(
                                "data: {}\n\n",
                                chunk_json(
                                    &req_id,
                                    &model,
                                    created,
                                    json!({"role": "assistant", "content": ""}),
                                    None,
                                )
                            ));
                        }
                        let delta = decoder.push(token);
                        if !delta.text.is_empty() {
                            payloads.push(format!(
                                "data: {}\n\n",
                                chunk_json(
                                    &req_id,
                                    &model,
                                    created,
                                    json!({"content": delta.text}),
                                    None,
                                )
                            ));
                        }
                        if delta.finished == Some(TokenizerFinishReason::StopString) {
                            guard.armed = false;
                            handle.complete_stop(loop_id);
                            payloads.push(format!(
                                "data: {}\n\n",
                                chunk_json(&req_id, &model, created, json!({}), Some("stop"))
                            ));
                            if include_usage {
                                payloads.push(format!(
                                    "data: {}\n\n",
                                    usage_chunk_json(
                                        &req_id,
                                        &model,
                                        created,
                                        prompt_tokens,
                                        decoder.token_count(),
                                    )
                                ));
                            }
                            payloads.push("data: [DONE]\n\n".to_owned());
                            return Some((
                                Ok::<_, std::convert::Infallible>(bytes_from(payloads)),
                                (
                                    rx,
                                    decoder,
                                    guard,
                                    handle,
                                    loop_id,
                                    req_id,
                                    model,
                                    created,
                                    include_usage,
                                    prompt_tokens,
                                    true,
                                    true,
                                ),
                            ));
                        }
                        Some((
                            Ok::<_, std::convert::Infallible>(bytes_from(payloads)),
                            (
                                rx,
                                decoder,
                                guard,
                                handle,
                                loop_id,
                                req_id,
                                model,
                                created,
                                include_usage,
                                prompt_tokens,
                                true,
                                false,
                            ),
                        ))
                    }
                    Some(LoopEvent::Finished { reason, .. }) => {
                        guard.armed = false;
                        if !sent_role {
                            payloads.push(format!(
                                "data: {}\n\n",
                                chunk_json(
                                    &req_id,
                                    &model,
                                    created,
                                    json!({"role": "assistant", "content": ""}),
                                    None,
                                )
                            ));
                        }
                        let tail = decoder.finish(finish_reason_to_tokenizer(reason));
                        if !tail.text.is_empty() {
                            payloads.push(format!(
                                "data: {}\n\n",
                                chunk_json(
                                    &req_id,
                                    &model,
                                    created,
                                    json!({"content": tail.text}),
                                    None,
                                )
                            ));
                        }
                        let final_reason =
                            if tail.finished == Some(TokenizerFinishReason::StopString) {
                                "stop"
                            } else {
                                finish_reason_text(reason)
                            };
                        payloads.push(format!(
                            "data: {}\n\n",
                            chunk_json(&req_id, &model, created, json!({}), Some(final_reason))
                        ));
                        if include_usage {
                            payloads.push(format!(
                                "data: {}\n\n",
                                usage_chunk_json(
                                    &req_id,
                                    &model,
                                    created,
                                    prompt_tokens,
                                    decoder.token_count(),
                                )
                            ));
                        }
                        payloads.push("data: [DONE]\n\n".to_owned());
                        Some((
                            Ok::<_, std::convert::Infallible>(bytes_from(payloads)),
                            (
                                rx,
                                decoder,
                                guard,
                                handle,
                                loop_id,
                                req_id,
                                model,
                                created,
                                include_usage,
                                prompt_tokens,
                                true,
                                true,
                            ),
                        ))
                    }
                    Some(LoopEvent::Failed { message, .. }) => {
                        guard.armed = false;
                        payloads.push(format!(
                            "data: {}\n\n",
                            error_body("server_error", "engine_faulted", &message)
                        ));
                        payloads.push("data: [DONE]\n\n".to_owned());
                        Some((
                            Ok::<_, std::convert::Infallible>(bytes_from(payloads)),
                            (
                                rx,
                                decoder,
                                guard,
                                handle,
                                loop_id,
                                req_id,
                                model,
                                created,
                                include_usage,
                                prompt_tokens,
                                sent_role,
                                true,
                            ),
                        ))
                    }
                    None => {
                        guard.disarm();
                        None
                    }
                }
            },
        );
        let mut response = Response::new(Body::from_stream(event_stream));
        response.headers_mut().insert(
            header::CONTENT_TYPE,
            HeaderValue::from_static("text/event-stream"),
        );
        response
            .headers_mut()
            .insert(header::CACHE_CONTROL, HeaderValue::from_static("no-cache"));
        return with_request_id(response, &request_id);
    }

    // Non-streaming: unbounded receiver prevents silent token drops on fast bursts.
    let (tx, mut rx) = mpsc::unbounded_channel::<LoopEvent>();
    let sink: zs_scheduler::EventSink = Arc::new(move |event: LoopEvent| {
        let _ = tx.send(event);
    });
    let loop_request_id = match handle.submit(SubmitRequest {
        prompt_tokens: prompt_ids,
        max_new_tokens,
        sink,
    }) {
        Ok(id) => id,
        Err(err) => return with_request_id(map_submit_error(err), &request_id),
    };

    println!(
        "zstep_serve request_id={request_id} traceparent={:?} loop_id={loop_request_id} prompt_tokens={prompt_tokens} max_tokens={max_new_tokens} stream=false stops={} seed={:?} user={:?}",
        traceparent,
        stops.len(),
        request.seed,
        request.user
    );

    let created = now_secs();
    let model = state.served_model_name.clone();
    let mut decoder = StreamDecoder::new(&state.tokenizer, stops);
    let mut guard = CancelGuard {
        handle: Some(handle.clone()),
        request_id: loop_request_id,
        armed: true,
    };
    let mut text = String::new();
    let mut finish_reason_str = "length";
    let mut failure: Option<String> = None;

    while let Some(event) = rx.recv().await {
        match event {
            LoopEvent::Token { token, .. } => {
                let delta = decoder.push(token);
                text.push_str(&delta.text);
                if delta.finished == Some(TokenizerFinishReason::StopString) {
                    guard.armed = false;
                    handle.complete_stop(loop_request_id);
                    finish_reason_str = "stop";
                    break;
                }
            }
            LoopEvent::Finished {
                reason: finished, ..
            } => {
                guard.armed = false;
                let tail = decoder.finish(finish_reason_to_tokenizer(finished));
                text.push_str(&tail.text);
                finish_reason_str = if tail.finished == Some(TokenizerFinishReason::StopString) {
                    "stop"
                } else {
                    finish_reason_text(finished)
                };
                break;
            }
            LoopEvent::Failed { message, .. } => {
                guard.armed = false;
                failure = Some(message);
                break;
            }
        }
    }
    guard.armed = false;
    if let Some(message) = failure {
        return with_request_id(
            error_response(
                StatusCode::SERVICE_UNAVAILABLE,
                "server_error",
                "engine_faulted",
                &message,
            ),
            &request_id,
        );
    }
    let completion_tokens = decoder.token_count();
    let body = json!({
        "id": request_id,
        "object": "chat.completion",
        "created": created,
        "model": model,
        "choices": [{
            "index": 0,
            "message": {"role": "assistant", "content": text},
            "finish_reason": finish_reason_str,
        }],
        "usage": {
            "prompt_tokens": prompt_tokens,
            "completion_tokens": completion_tokens,
            "total_tokens": prompt_tokens + completion_tokens,
        },
    });
    with_request_id(Json(body).into_response(), &request_id)
}

fn map_submit_error(error: SubmitError) -> Response {
    match error {
        SubmitError::Overloaded => error_response(
            StatusCode::TOO_MANY_REQUESTS,
            "rate_limit_error",
            "engine_overloaded",
            "zstep waiting queue or slot pool is full",
        ),
        SubmitError::InvalidDemand(msg) => error_response(
            StatusCode::BAD_REQUEST,
            "invalid_request_error",
            "context_length_exceeded",
            &msg,
        ),
        SubmitError::Unavailable(state) => error_response(
            StatusCode::SERVICE_UNAVAILABLE,
            "server_error",
            unavailable_code(state),
            "zstep is not accepting new requests",
        ),
    }
}

fn finish_reason_to_tokenizer(reason: LoopFinishReason) -> TokenizerFinishReason {
    match reason {
        LoopFinishReason::Length => TokenizerFinishReason::Length,
        LoopFinishReason::StopToken => TokenizerFinishReason::StopToken,
        LoopFinishReason::Cancelled => TokenizerFinishReason::Cancelled,
        LoopFinishReason::Failed => TokenizerFinishReason::Failed,
    }
}

fn bytes_from(payloads: Vec<String>) -> Bytes {
    Bytes::from(payloads.concat())
}

fn now_secs() -> u64 {
    SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .map(|elapsed| elapsed.as_secs())
        .unwrap_or(0)
}

// ---------------------------------------------------------------------------
// Entry point
// ---------------------------------------------------------------------------

/// Run `zstep serve`.
pub fn run(argv: &[String]) -> Result<(), Box<dyn std::error::Error>> {
    let args = ServeArgs::parse(argv)?;
    let tokenizer = Tokenizer::from_model_dir(&args.model_dir)?;
    let git_sha = zs_wire::resolve_git_sha(false).unwrap_or_else(|_| "unknown".to_owned());
    let num_kv_pages = args.resolved_num_kv_pages() as usize;
    let max_pages_per_seq = args.resolved_max_pages_per_seq() as usize;

    let state = Arc::new(AppState {
        phase: Arc::new(AtomicU8::new(EngineState::Starting as u8)),
        handle: Mutex::new(None),
        shutdown_requested: Arc::new(AtomicBool::new(false)),
        tokenizer,
        served_model_name: args.served_model_name.clone(),
        tokenizer_mode: args.tokenizer_mode.clone(),
        max_model_len: args.max_model_len,
        page_block_tokens: args.page_block_tokens as usize,
        max_pages_per_seq,
        num_kv_pages,
        created_at: now_secs(),
        started: Instant::now(),
        request_counter: AtomicU64::new(0),
    });

    let runtime = tokio::runtime::Builder::new_multi_thread()
        .enable_all()
        .build()?;
    runtime.block_on(async move {
        let address: SocketAddr = format!("{}:{}", args.host, args.port).parse()?;
        let listener = tokio::net::TcpListener::bind(address).await?;
        let bound = listener.local_addr()?;
        println!(
            "zstep_serve git_sha={git_sha} listening={bound} model={} model_dir={} world_size={} page_block_tokens={} num_kv_pages={num_kv_pages} max_model_len={} max_seqs={} max_step_tokens={} kv_cache_dtype={}",
            args.served_model_name,
            args.model_dir.display(),
            args.world_size,
            args.page_block_tokens,
            args.max_model_len,
            args.max_seqs,
            args.max_step_tokens,
            args.kv_cache_dtype,
        );

        // Launch GPU workers after HTTP listener binds so Nebula observes `Starting`
        // (`/livez == 200`, `/health == 503`, `/v1/models == 503`).
        let (launch_done_tx, launch_done_rx) = oneshot::channel::<()>();
        let exit_thread_pair = Arc::new((Mutex::new(false), Condvar::new()));
        let launch_exit_pair = Arc::clone(&exit_thread_pair);
        let launch_state = Arc::clone(&state);
        let launch_args = args.clone();
        let launch_join_handle = std::thread::Builder::new()
            .name("zstep-engine-launch".to_owned())
            .spawn(move || {
                launch_engine(launch_state, launch_args, launch_done_tx, launch_exit_pair)
            })
            .expect("spawn engine launch thread");

        let shutdown_state = Arc::clone(&state);
        let drain_timeout = args.drain_timeout;
        let app = router(state);
        axum::serve(listener, app)
            .with_graceful_shutdown(async move {
                wait_for_shutdown_signal().await;
                println!(
                    "zstep_serve sigterm: draining (timeout={}s)",
                    drain_timeout.as_secs()
                );
                shutdown_state
                    .shutdown_requested
                    .store(true, Ordering::Release);
                request_drain(&shutdown_state);
                let _ = launch_done_rx.await;
                if let Some(handle) = shutdown_state.handle() {
                    handle.drain();
                    let deadline = Instant::now() + drain_timeout;
                    loop {
                        if shutdown_state.phase() == EngineState::Faulted {
                            break;
                        }
                        let metrics = handle.metrics();
                        let pending = metrics.waiting.load(Ordering::Relaxed)
                            + metrics.running.load(Ordering::Relaxed);
                        if pending == 0 {
                            break;
                        }
                        if Instant::now() >= deadline {
                            println!("zstep_serve drain timeout with pending={pending}");
                            break;
                        }
                        tokio::time::sleep(Duration::from_millis(50)).await;
                    }
                    handle.shutdown();
                    let _ = tokio::task::spawn_blocking(move || {
                        handle.join(Duration::from_secs(30))
                    })
                    .await;
                }
                let (lock, cvar) = &*exit_thread_pair;
                if let Ok(mut stop) = lock.lock() {
                    *stop = true;
                    cvar.notify_all();
                }
                let _ = tokio::task::spawn_blocking(move || {
                    let _ = launch_join_handle.join();
                })
                .await;
            })
            .await?;
        Ok(())
    })
}

fn launch_engine(
    state: Arc<AppState>,
    args: ServeArgs,
    launch_done_tx: oneshot::Sender<()>,
    exit_thread_pair: Arc<(Mutex<bool>, Condvar)>,
) {
    let config = args.engine_config(Some(Arc::clone(&state.shutdown_requested)));
    let is_dsv41 = is_deepseek_v41_dir(&args.model_dir);
    let mixed = match args.max_mixed_step_tokens {
        Some(0) => None,
        Some(explicit) => Some(explicit),
        None if is_dsv41 => Some(3072),
        None => None,
    };
    let eos_token_ids = state.tokenizer.eos_token_ids().to_vec();
    match ServingEngine::launch(config) {
        Ok(mut engine) => {
            if state.shutdown_requested.load(Ordering::Acquire) {
                let _ = engine.shutdown();
                let _ = launch_done_tx.send(());
                return;
            }
            engine.set_max_mixed_step_tokens(mixed);
            let online = OnlineConfig {
                eos_token_ids,
                max_waiting_requests: args.max_waiting_requests,
            };
            let handle = spawn_serve_loop_with_state(engine, online, Arc::clone(&state.phase));
            publish_launched_handle(&state, handle);
            println!("zstep_serve engine ready (mixed_step_tokens={mixed:?})");
            let _ = launch_done_tx.send(());
            // Release `Arc<AppState>` before keeping this spawning thread alive until shutdown
            // completes (`PR_SET_PDEATHSIG` on Linux tracks the spawning thread's lifetime).
            drop(state);
            let (lock, cvar) = &*exit_thread_pair;
            let mut stop = lock.lock().expect("exit mutex");
            while !*stop {
                stop = cvar.wait(stop).expect("exit cvar");
            }
        }
        Err(error) => {
            if !state.shutdown_requested.load(Ordering::Acquire) {
                eprintln!("zstep_serve engine launch failed: {error}");
                state.set_phase(EngineState::Faulted);
            }
            let _ = launch_done_tx.send(());
        }
    }
}

async fn wait_for_shutdown_signal() {
    let mut terminate = tokio::signal::unix::signal(tokio::signal::unix::SignalKind::terminate())
        .expect("install SIGTERM handler");
    let mut interrupt = tokio::signal::unix::signal(tokio::signal::unix::SignalKind::interrupt())
        .expect("install SIGINT handler");
    tokio::select! {
        _ = terminate.recv() => {}
        _ = interrupt.recv() => {}
    }
}

// ---------------------------------------------------------------------------
// HTTP contract tests (mock loop, no GPU, no model weights)
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;
    use std::io::{Read, Write};
    use std::net::TcpStream;

    const TINY_TOKENIZER: &str = r#"{
      "version": "1.0",
      "truncation": null,
      "padding": null,
      "added_tokens": [
        {"id": 10, "content": "<｜begin▁of▁sentence｜>", "single_word": false, "lstrip": false, "rstrip": false, "normalized": false, "special": true},
        {"id": 11, "content": "<｜end▁of▁sentence｜>", "single_word": false, "lstrip": false, "rstrip": false, "normalized": false, "special": true},
        {"id": 12, "content": "<｜System｜>", "single_word": false, "lstrip": false, "rstrip": false, "normalized": false, "special": true},
        {"id": 13, "content": "<｜User｜>", "single_word": false, "lstrip": false, "rstrip": false, "normalized": false, "special": true},
        {"id": 14, "content": "<｜Assistant｜>", "single_word": false, "lstrip": false, "rstrip": false, "normalized": false, "special": true}
      ],
      "normalizer": null,
      "pre_tokenizer": {"type": "Whitespace"},
      "post_processor": null,
      "decoder": null,
      "model": {
        "type": "WordLevel",
        "vocab": {
          "[UNK]": 0, "hello": 1, "world": 2, "!": 3, "hi": 4, "Capital": 5, "France": 6, "?": 7,
          "<think>": 8, "</think>": 9, "<｜begin▁of▁sentence｜>": 10, "<｜end▁of▁sentence｜>": 11,
          "<｜System｜>": 12, "<｜User｜>": 13, "<｜Assistant｜>": 14
        },
        "unk_token": "[UNK]"
      }
    }"#;

    struct TempModelDir(PathBuf);

    impl Drop for TempModelDir {
        fn drop(&mut self) {
            let _ = std::fs::remove_dir_all(&self.0);
        }
    }

    fn test_tokenizer() -> Tokenizer {
        static COUNTER: AtomicU64 = AtomicU64::new(0);
        let dir = TempModelDir(std::env::temp_dir().join(format!(
            "zstep_serve_tok_{}_{}",
            std::process::id(),
            COUNTER.fetch_add(1, Ordering::Relaxed)
        )));
        std::fs::create_dir_all(&dir.0).expect("temp dir");
        std::fs::write(dir.0.join("tokenizer.json"), TINY_TOKENIZER).expect("write tokenizer");
        Tokenizer::from_model_dir(&dir.0).expect("load tiny tokenizer")
    }

    fn test_state(phase: EngineState, handle: Option<ServeLoopHandle>) -> Arc<AppState> {
        let phase_arc = if let Some(ref h) = handle {
            h.set_state(phase);
            h.state_arc()
        } else {
            Arc::new(AtomicU8::new(phase as u8))
        };
        Arc::new(AppState {
            phase: phase_arc,
            handle: Mutex::new(handle),
            shutdown_requested: Arc::new(AtomicBool::new(false)),
            tokenizer: test_tokenizer(),
            served_model_name: "zstep-test".to_owned(),
            tokenizer_mode: "auto".to_owned(),
            max_model_len: 4096,
            page_block_tokens: 16,
            max_pages_per_seq: 256,
            num_kv_pages: 4096,
            created_at: 0,
            started: Instant::now(),
            request_counter: AtomicU64::new(0),
        })
    }

    async fn spawn_server(state: Arc<AppState>) -> SocketAddr {
        let listener = tokio::net::TcpListener::bind("127.0.0.1:0")
            .await
            .expect("bind");
        let address = listener.local_addr().expect("addr");
        tokio::spawn(async move {
            let _ = axum::serve(listener, router(state)).await;
        });
        address
    }

    fn http(
        address: SocketAddr,
        method: &str,
        path: &str,
        body: Option<&str>,
        headers: &[(&str, &str)],
    ) -> (u16, String, String) {
        let mut stream = TcpStream::connect(address).expect("connect");
        stream
            .set_read_timeout(Some(Duration::from_secs(10)))
            .expect("timeout");
        let mut request =
            format!("{method} {path} HTTP/1.1\r\nHost: test\r\nConnection: close\r\n");
        for (name, value) in headers {
            request.push_str(&format!("{name}: {value}\r\n"));
        }
        if let Some(body) = body {
            request.push_str("Content-Type: application/json\r\n");
            request.push_str(&format!("Content-Length: {}\r\n", body.len()));
        }
        request.push_str("\r\n");
        if let Some(body) = body {
            request.push_str(body);
        }
        stream.write_all(request.as_bytes()).expect("write");
        let mut raw = Vec::new();
        stream.read_to_end(&mut raw).expect("read");
        let text = String::from_utf8_lossy(&raw).into_owned();
        let status = text
            .split_whitespace()
            .nth(1)
            .and_then(|code| code.parse().ok())
            .unwrap_or(0);
        let (head, body) = text.split_once("\r\n\r\n").unwrap_or((&text, ""));
        (status, head.to_owned(), body.to_owned())
    }

    fn chat_body(content: &str, stream: bool, extra: &str) -> String {
        format!(
            r#"{{"messages":[{{"role":"user","content":"{content}"}}],"stream":{stream}{extra}}}"#
        )
    }

    #[test]
    fn cli_parser_enforces_nebula_contract_flags_and_model_defaults() {
        let parse = |items: &[&str]| {
            let raw: Vec<String> = items.iter().copied().map(str::to_owned).collect();
            ServeArgs::parse(&raw)
        };
        assert!(parse(&["--model-dir"]).is_err(), "missing value must fail");
        assert!(
            parse(&["--model-dir", "/tmp/m", "--enable-prefix-caching"]).is_err(),
            "enable-prefix-caching must be rejected in P0"
        );
        assert!(
            parse(&["--model-dir", "/tmp/m", "--page-block-tokens", "0"]).is_err(),
            "zero page_block_tokens must fail"
        );
        assert!(
            parse(&["--model-dir", "/tmp/m", "--world-size", "16"]).is_err(),
            "world_size > 8 must fail"
        );
        assert!(
            parse(&["--model-dir", "/tmp/qwen35", "--kv-cache-dtype", "fp8",]).is_err(),
            "fp8 kv_cache_dtype must fail loudly on BF16 model runner"
        );
        let ok = parse(&[
            "--model-dir",
            "/tmp/qwen35",
            "--no-enable-prefix-caching",
            "--kv-cache-dtype",
            "bf16",
            "--max-model-len",
            "16384",
            "--gpu-memory-utilization",
            "0.95",
        ])
        .expect("valid qwen args");
        assert_eq!(ok.page_block_tokens, 16);
        assert_eq!(ok.resolved_num_kv_pages(), 49_152);
        assert_eq!(ok.engine_config(None).launch.max_model_len, Some(16_384));

        let dsv41_dir = TempModelDir(
            std::env::temp_dir().join(format!("zstep_dsv41_cfg_{}", std::process::id())),
        );
        std::fs::create_dir_all(&dsv41_dir.0).expect("dsv41 temp dir");
        std::fs::write(
            dsv41_dir.0.join("config.json"),
            r#"{"model_type":"deepseek_v41"}"#,
        )
        .expect("write dsv41 config");
        let dsv41_path = dsv41_dir.0.to_str().expect("utf8 path");
        assert!(
            parse(&["--model-dir", dsv41_path, "--kv-cache-dtype", "bf16"]).is_err(),
            "bf16 kv_cache_dtype must fail loudly on deepseek_v41 runner"
        );
        let dsv41_ok =
            parse(&["--model-dir", dsv41_path, "--kv-cache-dtype", "fp8"]).expect("dsv41 fp8 ok");
        assert_eq!(dsv41_ok.page_block_tokens, 128);
        assert_eq!(dsv41_ok.resolved_num_kv_pages(), 2_560);
    }

    #[tokio::test(flavor = "multi_thread", worker_threads = 2)]
    async fn startup_publication_and_drain_during_starting_plus_early_stream_fault() {
        // 1) Drain requested while still `Starting` (`handle == None`):
        //    - `/health` stays 503 `starting` until `publish_launched_handle` finishes.
        //    - `publish_launched_handle` preserves `Draining` (never clobbers to `Ready`)
        //      and immediately drains the handle so new requests get `503 engine_draining`.
        let starting_state = test_state(EngineState::Starting, None);
        let addr = spawn_server(starting_state.clone()).await;
        let (drain_status, _, _) =
            tokio::task::spawn_blocking(move || http(addr, "POST", "/admin/drain", None, &[]))
                .await
                .expect("join");
        assert_eq!(drain_status, 200);
        let (health_before, _, body_before) =
            tokio::task::spawn_blocking(move || http(addr, "GET", "/health", None, &[]))
                .await
                .expect("join");
        assert_eq!(health_before, 503);
        assert!(body_before.contains("starting"), "{body_before}");

        let mock_handle = zs_scheduler::spawn_mock(zs_scheduler::MockScript::default());
        publish_launched_handle(&starting_state, mock_handle.clone());
        assert_eq!(starting_state.phase(), EngineState::Draining);
        assert!(starting_state.handle().is_some());

        let (health_after, _, body_after) =
            tokio::task::spawn_blocking(move || http(addr, "GET", "/health", None, &[]))
                .await
                .expect("join");
        assert_eq!(health_after, 200);
        assert!(body_after.contains("draining"), "{body_after}");

        let chat = chat_body("hi", false, "");
        let (chat_code, _, chat_err) = tokio::task::spawn_blocking(move || {
            http(addr, "POST", "/v1/chat/completions", Some(&chat), &[])
        })
        .await
        .expect("join");
        assert_eq!(chat_code, 503);
        assert!(chat_err.contains("engine_draining"), "{chat_err}");

        mock_handle.shutdown();
        let _ = mock_handle.join(Duration::from_secs(2));

        // 2) Normal `Starting -> Ready` publication + streaming failure before first token:
        //    SSE emits ONLY `engine_faulted` + `[DONE]` without an empty assistant role chunk.
        let fail0_handle = zs_scheduler::spawn_mock(zs_scheduler::MockScript {
            tokens: vec![1],
            fail_after: Some(0),
            ..Default::default()
        });
        fail0_handle.set_state(EngineState::Starting);
        let ready_state = Arc::new(AppState {
            phase: fail0_handle.state_arc(),
            handle: Mutex::new(None),
            shutdown_requested: Arc::new(AtomicBool::new(false)),
            tokenizer: test_tokenizer(),
            served_model_name: "zstep-test".to_owned(),
            tokenizer_mode: "auto".to_owned(),
            max_model_len: 4096,
            page_block_tokens: 16,
            max_pages_per_seq: 256,
            num_kv_pages: 4096,
            created_at: 0,
            started: Instant::now(),
            request_counter: AtomicU64::new(0),
        });
        publish_launched_handle(&ready_state, fail0_handle.clone());
        assert_eq!(ready_state.phase(), EngineState::Ready);
        assert!(ready_state.handle().is_some());

        let addr2 = spawn_server(ready_state).await;
        let stream_req = chat_body("hi", true, r#","max_tokens":4"#);
        let (sse_code, _, sse_body) = tokio::task::spawn_blocking(move || {
            http(
                addr2,
                "POST",
                "/v1/chat/completions",
                Some(&stream_req),
                &[],
            )
        })
        .await
        .expect("join");
        assert_eq!(sse_code, 200);
        assert!(sse_body.contains("engine_faulted"), "{sse_body}");
        assert!(
            !sse_body.contains("\"role\":\"assistant\""),
            "early failure before first token must not emit assistant role chunk: {sse_body}"
        );
        assert_eq!(sse_body.matches("data: [DONE]").count(), 1, "{sse_body}");

        fail0_handle.shutdown();
        let _ = fail0_handle.join(Duration::from_secs(2));
    }

    #[tokio::test(flavor = "multi_thread", worker_threads = 2)]
    async fn probe_matrix_and_runtime_fault_propagate_to_health_503() {
        let handle = zs_scheduler::spawn_mock(zs_scheduler::MockScript {
            tokens: vec![1, 2, 3],
            fail_after: Some(1),
            ..Default::default()
        });
        let state = test_state(EngineState::Starting, Some(handle.clone()));
        let address = spawn_server(state.clone()).await;
        let call = |method: &'static str, path: &'static str| {
            std::thread::spawn(move || http(address, method, path, None, &[]))
                .join()
                .expect("join")
        };
        let (livez, _, _) = call("GET", "/livez");
        let (health, _, _) = call("GET", "/health");
        let (ready, _, _) = call("GET", "/ready");
        let (models, _, _) = call("GET", "/v1/models");
        assert_eq!((livez, health, ready, models), (200, 503, 503, 503));

        state.set_phase(EngineState::Ready);
        assert_eq!(call("GET", "/health").0, 200);
        assert_eq!(call("GET", "/ready").0, 200);
        assert_eq!(call("GET", "/v1/models").0, 200);

        // Trigger mid-step failure in ServeLoop -> shared atomic flips `/health` to 503!
        let req_body = chat_body("hi", false, r#","max_tokens":4"#);
        let (status, _, err_body) = tokio::task::spawn_blocking(move || {
            http(
                address,
                "POST",
                "/v1/chat/completions",
                Some(&req_body),
                &[],
            )
        })
        .await
        .expect("join");
        assert_eq!(status, 503);
        assert!(err_body.contains("engine_faulted"), "{err_body}");
        let (health_after_fault, _, h_body) = call("GET", "/health");
        assert_eq!(health_after_fault, 503);
        assert!(h_body.contains("faulted"), "{h_body}");

        handle.shutdown();
        let _ = handle.join(Duration::from_secs(2));
    }

    #[tokio::test(flavor = "multi_thread", worker_threads = 2)]
    async fn admin_drain_keeps_health_200_and_metrics_live() {
        let handle = zs_scheduler::spawn_mock(zs_scheduler::MockScript::default());
        let state = test_state(EngineState::Ready, Some(handle.clone()));
        let address = spawn_server(state).await;
        let (drain_code, _, _) =
            tokio::task::spawn_blocking(move || http(address, "POST", "/admin/drain", None, &[]))
                .await
                .expect("join");
        assert_eq!(drain_code, 200);

        let (health_code, _, health_body) =
            tokio::task::spawn_blocking(move || http(address, "GET", "/health", None, &[]))
                .await
                .expect("join");
        assert_eq!(health_code, 200);
        assert!(health_body.contains("draining"));

        let (ready_code, _, _) =
            tokio::task::spawn_blocking(move || http(address, "GET", "/ready", None, &[]))
                .await
                .expect("join");
        assert_eq!(ready_code, 503);

        let body = chat_body("hi", false, "");
        let (chat_code, _, chat_err) = tokio::task::spawn_blocking(move || {
            http(address, "POST", "/v1/chat/completions", Some(&body), &[])
        })
        .await
        .expect("join");
        assert_eq!(chat_code, 503);
        assert!(chat_err.contains("engine_draining"), "{chat_err}");

        handle.shutdown();
        let _ = handle.join(Duration::from_secs(2));
    }

    #[tokio::test(flavor = "multi_thread", worker_threads = 2)]
    async fn non_stream_and_stream_stop_string_cancels_backend_and_emits_choices_empty_usage() {
        let handle = zs_scheduler::spawn_mock(zs_scheduler::MockScript {
            tokens: vec![1, 2, 3], // decodes to "hello world !"
            per_token_delay: Duration::from_millis(5),
            ..Default::default()
        });
        let state = test_state(EngineState::Ready, Some(handle.clone()));
        let address = spawn_server(state).await;

        // 1) Streaming with stop=["world"] + include_usage=true
        let stream_body = chat_body(
            "hi",
            true,
            r#","max_tokens":16,"stop":["world"],"stream_options":{"include_usage":true}"#,
        );
        let (status, headers, sse) = tokio::task::spawn_blocking(move || {
            http(
                address,
                "POST",
                "/v1/chat/completions",
                Some(&stream_body),
                &[("x-nebula-request-id", "req_sse_stop")],
            )
        })
        .await
        .expect("join");
        assert_eq!(status, 200, "{sse}");
        assert!(
            headers
                .to_lowercase()
                .contains("x-request-id: req_sse_stop")
        );
        assert_eq!(sse.matches("data: [DONE]").count(), 1, "{sse}");
        assert!(sse.contains("\"finish_reason\":\"stop\""), "{sse}");
        assert!(!sse.contains("world"), "stop string must be clipped: {sse}");
        assert!(
            sse.contains("\"choices\":[],\"usage\"")
                || sse.contains("\"choices\":[]") && sse.contains("\"usage\""),
            "usage chunk must carry empty choices array: {sse}"
        );

        // 2) Non-streaming with stop=["world"] + metrics histogram validation
        let non_stream_body = chat_body("hi", false, r#","max_tokens":16,"stop":["world"]"#);
        let (status2, _, json_resp) = tokio::task::spawn_blocking(move || {
            http(
                address,
                "POST",
                "/v1/chat/completions",
                Some(&non_stream_body),
                &[],
            )
        })
        .await
        .expect("join");
        assert_eq!(status2, 200, "{json_resp}");
        assert!(
            json_resp.contains("\"finish_reason\":\"stop\""),
            "{json_resp}"
        );
        assert!(!json_resp.contains("world"), "{json_resp}");

        let (_, _, prom) =
            tokio::task::spawn_blocking(move || http(address, "GET", "/metrics", None, &[]))
                .await
                .expect("join");
        assert!(prom.contains("zstep:time_to_first_token_seconds_bucket{le=\"+Inf\"}"));
        assert!(prom.contains("zstep:time_per_output_token_seconds_bucket{le=\"+Inf\"}"));
        assert!(!prom.contains("prefix_cache_hits_total"));

        handle.shutdown();
        let _ = handle.join(Duration::from_secs(2));
    }

    #[tokio::test(flavor = "multi_thread", worker_threads = 2)]
    async fn client_disconnect_mid_stream_cancels_in_flight_sequence() {
        let handle = zs_scheduler::spawn_mock(zs_scheduler::MockScript {
            tokens: vec![1; 40],
            per_token_delay: Duration::from_millis(25),
            ..Default::default()
        });
        let metrics = handle.metrics();
        let state = test_state(EngineState::Ready, Some(handle.clone()));
        let address = spawn_server(state).await;

        tokio::task::spawn_blocking(move || {
            let mut stream = TcpStream::connect(address).expect("connect");
            let body = chat_body("hi", true, r#","max_tokens":40"#);
            let req = format!(
                "POST /v1/chat/completions HTTP/1.1\r\nHost: test\r\nContent-Type: application/json\r\nContent-Length: {}\r\n\r\n{body}",
                body.len()
            );
            stream.write_all(req.as_bytes()).expect("write");
            let mut buf = [0u8; 128];
            let _ = stream.read(&mut buf);
            // Drop socket abruptly mid-stream!
        })
        .await
        .expect("join");

        let deadline = Instant::now() + Duration::from_secs(3);
        while Instant::now() < deadline {
            if metrics.cancelled.load(Ordering::Relaxed) >= 1
                && metrics.running.load(Ordering::Relaxed) == 0
            {
                break;
            }
            tokio::time::sleep(Duration::from_millis(20)).await;
        }
        assert_eq!(metrics.running.load(Ordering::Relaxed), 0);
        assert!(metrics.cancelled.load(Ordering::Relaxed) >= 1);

        handle.shutdown();
        let _ = handle.join(Duration::from_secs(2));
    }

    #[tokio::test(flavor = "multi_thread", worker_threads = 2)]
    async fn unsupported_unknown_and_malformed_requests_return_400_json() {
        let handle = zs_scheduler::spawn_mock(zs_scheduler::MockScript::default());
        let state = test_state(EngineState::Ready, Some(handle.clone()));
        let address = spawn_server(state).await;
        let cases = [
            chat_body("hi", false, r#","temperature":0.7"#),
            chat_body("hi", false, r#","max_tokens":0"#),
            chat_body("hi", false, r#","n":2"#),
            chat_body("hi", false, r#","tools":[]"#),
            chat_body("hi", false, r#","logprobs":true"#),
            chat_body("hi", false, r#","logit_bias":{"1":5}"#),
            chat_body("hi", false, r#","stop":["a","b","c","d","e"]"#),
            r#"{"model":"other-model","messages":[{"role":"user","content":"hi"}]}"#.to_owned(),
            r#"{"messages":[]}"#.to_owned(),
            r#"{"messages":[{"role":"user","content":[{"type":"image_url"}]}]}"#.to_owned(),
            r#"{"not_valid_json""#.to_owned(),
        ];
        for body in cases {
            let request_body = body.clone();
            let (status, headers, response) = tokio::task::spawn_blocking(move || {
                http(
                    address,
                    "POST",
                    "/v1/chat/completions",
                    Some(&request_body),
                    &[("x-request-id", "req_bad_400")],
                )
            })
            .await
            .expect("join");
            assert_eq!(status, 400, "body={body} response={response}");
            assert!(response.contains("invalid_request_error"), "{response}");
            assert!(headers.to_lowercase().contains("x-request-id: req_bad_400"));
        }

        handle.shutdown();
        let _ = handle.join(Duration::from_secs(2));
    }

    #[tokio::test(flavor = "multi_thread", worker_threads = 2)]
    async fn overload_returns_429_not_503() {
        let handle = zs_scheduler::spawn_mock(zs_scheduler::MockScript {
            tokens: vec![1, 2, 3],
            per_token_delay: Duration::from_millis(250),
            max_waiting: 1,
            max_seqs: 1,
            ..Default::default()
        });
        let state = test_state(EngineState::Ready, Some(handle.clone()));
        let address = spawn_server(state).await;
        let mut tasks = Vec::new();
        for _ in 0..5 {
            tasks.push(tokio::task::spawn_blocking(move || {
                let body = chat_body("hi", false, r#","max_tokens":4"#);
                http(address, "POST", "/v1/chat/completions", Some(&body), &[])
            }));
        }
        let mut saw_429 = false;
        for task in tasks {
            let (status, _, body) = task.await.expect("join");
            assert_ne!(status, 503, "overload must never return 503: {body}");
            if status == 429 {
                assert!(body.contains("engine_overloaded"), "{body}");
                saw_429 = true;
            }
        }
        assert!(
            saw_429,
            "expected at least one 429 response under saturation"
        );

        handle.shutdown();
        let _ = handle.join(Duration::from_secs(2));
    }
}
