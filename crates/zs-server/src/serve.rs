// Copyright (c) 2026 zstep Contributors
// SPDX-License-Identifier: MIT

//! `zstep serve` / `zs-server serve`: OpenAI-compatible HTTP/SSE front end for the
//! persistent online `ServeLoop` (Slices Z3 + Z4 of the Nebula integration plan).
//!
//! Contract source: `nebula/docs/dev/zstep.md` (four-state probes, `429` vs `503`,
//! request-id propagation, `/metrics` field names & histograms, SSE `[DONE]` uniqueness).

use std::collections::BTreeMap;
use std::fmt;
use std::net::{IpAddr, SocketAddr};
use std::path::{Path, PathBuf};
use std::sync::atomic::{AtomicBool, AtomicU8, AtomicU64, Ordering};
use std::sync::{Arc, Condvar, Mutex};
use std::time::{Duration, Instant, SystemTime, UNIX_EPOCH};

use axum::body::{Body, Bytes};
use axum::extract::{ConnectInfo, State};
use axum::http::{HeaderMap, HeaderValue, Request, StatusCode, header};
use axum::response::{IntoResponse, Response};
use axum::routing::{get, post};
use axum::{Json, Router};
use futures_util::stream;
use serde::Deserialize;
use serde_json::{Value, json};
use tokio::sync::{mpsc, oneshot, watch};
use zs_scheduler::{
    DEFAULT_HEARTBEAT_TIMEOUT, EngineConfig, EngineState, LoopEvent, LoopFinishReason,
    OnlineConfig, OnlineMetrics, SamplingParams, ServeLoopHandle, ServingEngine, SpeculativeConfig,
    SpeculativeMethod, SubmitError, SubmitRequest, TPOT_BUCKET_BOUNDS_SECS,
    TTFT_BUCKET_BOUNDS_SECS, WorkerLaunchConfig, request_kv_page_demand,
    spawn_serve_loop_with_state, splitmix64,
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

mod auth;
mod cli;
mod kv_events;
mod metrics;

use self::auth::{authorize_bearer, authorize_ops_request};
use self::cli::is_deepseek_v41_dir;
pub use self::cli::{SERVE_USAGE, SecretToken, ServeArgs};
use self::kv_events::KvEventPublisher;
use self::metrics::render_metrics;

// ---------------------------------------------------------------------------
// Shared state
// ---------------------------------------------------------------------------

struct AppState {
    phase: Arc<AtomicU8>,
    handle: Mutex<Option<ServeLoopHandle>>,
    shutdown_requested: Arc<AtomicBool>,
    api_key: Option<SecretToken>,
    allow_remote_metrics: bool,
    tokenizer: Tokenizer,
    served_model_name: String,
    tokenizer_mode: String,
    enable_prefix_caching: bool,
    speculative_enabled: bool,
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
        self.handle
            .lock()
            .unwrap_or_else(|poisoned| poisoned.into_inner())
            .clone()
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
// Authentication, access control & routers
// ---------------------------------------------------------------------------
#[cfg(test)]
fn router(state: Arc<AppState>) -> Router {
    router_with_ops(state, true)
}

fn router_with_ops(state: Arc<AppState>, include_ops: bool) -> Router {
    let base = Router::new()
        .route("/livez", get(livez))
        .route("/health", get(health))
        .route("/ready", get(ready))
        .route("/v1/models", get(list_models))
        .route("/v1/chat/completions", post(chat_completions));
    if include_ops {
        base.route("/metrics", get(metrics))
            .route("/admin/drain", post(admin_drain))
            .with_state(state)
    } else {
        base.with_state(state)
    }
}

fn admin_loopback_router(state: Arc<AppState>) -> Router {
    Router::new()
        .route("/livez", get(livez))
        .route("/health", get(health))
        .route("/ready", get(ready))
        .route("/metrics", get(metrics))
        .route("/admin/drain", post(admin_drain))
        .with_state(state)
}

fn publish_launched_handle(state: &AppState, handle: ServeLoopHandle) {
    let mut guard = state
        .handle
        .lock()
        .unwrap_or_else(|poisoned| poisoned.into_inner());
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
    let guard = state
        .handle
        .lock()
        .unwrap_or_else(|poisoned| poisoned.into_inner());
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

async fn list_models(State(state): State<Arc<AppState>>, headers: HeaderMap) -> Response {
    if let Some(response) = authorize_bearer(&headers, &state) {
        return response;
    }
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

async fn admin_drain(State(state): State<Arc<AppState>>, req: Request<Body>) -> Response {
    if let Some(response) = authorize_ops_request(&req, &state, false) {
        return response;
    }
    request_drain(&state);
    (StatusCode::OK, Json(json!({"status": "draining"}))).into_response()
}

async fn metrics(State(state): State<Arc<AppState>>, req: Request<Body>) -> Response {
    if let Some(response) = authorize_ops_request(&req, &state, state.allow_remote_metrics) {
        return response;
    }
    let text = render_metrics(&state);
    (
        StatusCode::OK,
        [(header::CONTENT_TYPE, "text/plain; version=0.0.4")],
        text,
    )
        .into_response()
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
    top_k: Option<i32>,
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
fn resolve_sampling_params(
    request: &ChatRequest,
    state: &AppState,
) -> Result<SamplingParams, Response> {
    let temperature = request.temperature.unwrap_or(0.0);
    if !temperature.is_finite() || !(0.0..=2.0).contains(&temperature) {
        return Err(error_response(
            StatusCode::BAD_REQUEST,
            "invalid_request_error",
            "invalid_parameter",
            "temperature must be finite and within [0.0, 2.0]",
        ));
    }
    let top_p = request.top_p.unwrap_or(1.0);
    let top_p_f32 = top_p as f32;
    if !top_p.is_finite() || !top_p_f32.is_finite() || top_p_f32 <= 0.0 || top_p_f32 > 1.0 {
        return Err(error_response(
            StatusCode::BAD_REQUEST,
            "invalid_request_error",
            "invalid_parameter",
            "top_p must be finite and within (0.0, 1.0]",
        ));
    }
    let raw_top_k = request.top_k.unwrap_or(0);
    if raw_top_k < -1 {
        return Err(error_response(
            StatusCode::BAD_REQUEST,
            "invalid_request_error",
            "invalid_parameter",
            "top_k must be -1, 0 (disabled), or >= 1",
        ));
    }
    let top_k = if raw_top_k <= 0 { 0 } else { raw_top_k };
    let presence_penalty = request.presence_penalty.unwrap_or(0.0);
    if !presence_penalty.is_finite() || !(-2.0..=2.0).contains(&presence_penalty) {
        return Err(error_response(
            StatusCode::BAD_REQUEST,
            "invalid_request_error",
            "invalid_parameter",
            "presence_penalty must be finite and within [-2.0, 2.0]",
        ));
    }
    let frequency_penalty = request.frequency_penalty.unwrap_or(0.0);
    if !frequency_penalty.is_finite() || !(-2.0..=2.0).contains(&frequency_penalty) {
        return Err(error_response(
            StatusCode::BAD_REQUEST,
            "invalid_request_error",
            "invalid_parameter",
            "frequency_penalty must be finite and within [-2.0, 2.0]",
        ));
    }
    let ignore_eos = request.ignore_eos.unwrap_or(false);
    let logprobs = request.logprobs.unwrap_or(false);
    let top_logprobs = request.top_logprobs.unwrap_or(0);
    if request.top_logprobs.is_some() && !logprobs {
        return Err(error_response(
            StatusCode::BAD_REQUEST,
            "invalid_request_error",
            "invalid_parameter",
            "top_logprobs can only be specified when logprobs is true",
        ));
    }
    if top_logprobs > 5 {
        return Err(error_response(
            StatusCode::BAD_REQUEST,
            "invalid_request_error",
            "invalid_parameter",
            "top_logprobs must be between 0 and 5",
        ));
    }
    let temperature_f32 = temperature as f32;
    let seed = if temperature_f32 <= 0.0 {
        0
    } else if let Some(explicit) = request.seed {
        explicit as u64
    } else {
        let counter = state.request_counter.fetch_add(1, Ordering::Relaxed);
        let nanos = SystemTime::now()
            .duration_since(UNIX_EPOCH)
            .map(|elapsed| elapsed.as_nanos() as u64)
            .unwrap_or(0);
        splitmix64(
            nanos
                ^ counter.rotate_left(29)
                ^ u64::from(std::process::id()).wrapping_mul(0x9E37_79B9_7F4A_7C15),
        )
    };
    Ok(SamplingParams {
        temperature: temperature_f32,
        top_p: top_p_f32,
        top_k,
        seed,
        presence_penalty: presence_penalty as f32,
        frequency_penalty: frequency_penalty as f32,
        ignore_eos,
        logprobs,
        top_logprobs,
    })
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
    logprobs: Option<Value>,
) -> String {
    json!({
        "id": id,
        "object": "chat.completion.chunk",
        "created": created,
        "model": model,
        "choices": [{
            "index": 0,
            "delta": delta,
            "logprobs": logprobs,
            "finish_reason": finish_reason,
        }],
    })
    .to_string()
}

fn format_logprob_item(
    tokenizer: &Tokenizer,
    token: u32,
    logprob_info: Option<&zs_scheduler::StepLogprob>,
    top_logprobs_requested: bool,
) -> Value {
    let (token_str, token_bytes) = tokenizer.decode_single_token(token);
    let chosen_logprob = logprob_info.map_or(0.0, |lp| lp.logprob);
    let top_logprobs_val = if top_logprobs_requested {
        let mut entries = Vec::new();
        if let Some(lp) = logprob_info {
            for cand in &lp.top_logprobs {
                let (cand_str, cand_bytes) = tokenizer.decode_single_token(cand.token);
                entries.push(json!({
                    "token": cand_str,
                    "logprob": cand.logprob,
                    "bytes": cand_bytes,
                }));
            }
        }
        json!(entries)
    } else {
        json!([])
    };
    json!({
        "token": token_str,
        "logprob": chosen_logprob,
        "bytes": token_bytes,
        "top_logprobs": top_logprobs_val,
    })
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

struct StreamState {
    rx: mpsc::Receiver<LoopEvent>,
    decoder: StreamDecoder,
    guard: CancelGuard,
    handle: ServeLoopHandle,
    tokenizer: Tokenizer,
    loop_id: u64,
    req_id: String,
    model: String,
    created: u64,
    include_usage: bool,
    prompt_tokens: usize,
    enable_logprobs: bool,
    top_logprobs_requested: bool,
    buffered_logprobs: Vec<Value>,
    sent_role: bool,
    sent_done: bool,
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
    if let Some(response) = authorize_bearer(&headers, &state) {
        return with_request_id(response, &request_id);
    }
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
    let sampling = match resolve_sampling_params(&request, &state) {
        Ok(sampling) => sampling,
        Err(response) => return with_request_id(response, &request_id),
    };
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
            sampling,
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
        let initial_state = StreamState {
            rx,
            decoder,
            guard,
            handle,
            tokenizer: state.tokenizer.clone(),
            loop_id: loop_request_id,
            req_id: request_id.clone(),
            model,
            created,
            include_usage,
            prompt_tokens,
            enable_logprobs: request.logprobs.unwrap_or(false),
            top_logprobs_requested: request.top_logprobs.is_some(),
            buffered_logprobs: Vec::new(),
            sent_role: false,
            sent_done: false,
        };

        let event_stream = stream::unfold(initial_state, |mut s| async move {
            if s.sent_done {
                return None;
            }
            let mut payloads: Vec<String> = Vec::new();
            match s.rx.recv().await {
                Some(LoopEvent::Token { token, logprob, .. }) => {
                    if !s.sent_role {
                        payloads.push(format!(
                            "data: {}\n\n",
                            chunk_json(
                                &s.req_id,
                                &s.model,
                                s.created,
                                json!({"role": "assistant", "content": ""}),
                                None,
                                None,
                            )
                        ));
                        s.sent_role = true;
                    }
                    let delta = s.decoder.push(token);
                    if s.enable_logprobs {
                        s.buffered_logprobs.push(format_logprob_item(
                            &s.tokenizer,
                            token,
                            logprob.as_ref(),
                            s.top_logprobs_requested,
                        ));
                    }
                    if !delta.text.is_empty() {
                        let logprobs_chunk = if s.enable_logprobs {
                            Some(json!({
                                "content": std::mem::take(&mut s.buffered_logprobs),
                            }))
                        } else {
                            None
                        };
                        payloads.push(format!(
                            "data: {}\n\n",
                            chunk_json(
                                &s.req_id,
                                &s.model,
                                s.created,
                                json!({"content": delta.text}),
                                None,
                                logprobs_chunk,
                            )
                        ));
                    }
                    if delta.finished == Some(TokenizerFinishReason::StopString) {
                        s.guard.disarm();
                        s.handle.complete_stop(s.loop_id);
                        payloads.push(format!(
                            "data: {}\n\n",
                            chunk_json(
                                &s.req_id,
                                &s.model,
                                s.created,
                                json!({}),
                                Some("stop"),
                                None
                            )
                        ));
                        if s.include_usage {
                            payloads.push(format!(
                                "data: {}\n\n",
                                usage_chunk_json(
                                    &s.req_id,
                                    &s.model,
                                    s.created,
                                    s.prompt_tokens,
                                    s.decoder.token_count(),
                                )
                            ));
                        }
                        payloads.push("data: [DONE]\n\n".to_owned());
                        s.sent_done = true;
                        return Some((Ok::<_, std::convert::Infallible>(bytes_from(payloads)), s));
                    }
                    Some((Ok::<_, std::convert::Infallible>(bytes_from(payloads)), s))
                }
                Some(LoopEvent::Finished { reason, .. }) => {
                    s.guard.disarm();
                    if !s.sent_role {
                        payloads.push(format!(
                            "data: {}\n\n",
                            chunk_json(
                                &s.req_id,
                                &s.model,
                                s.created,
                                json!({"role": "assistant", "content": ""}),
                                None,
                                None,
                            )
                        ));
                    }
                    let tail = s.decoder.finish(finish_reason_to_tokenizer(reason));
                    if !tail.text.is_empty() {
                        let logprobs_chunk = if s.enable_logprobs {
                            Some(json!({
                                "content": std::mem::take(&mut s.buffered_logprobs),
                            }))
                        } else {
                            None
                        };
                        payloads.push(format!(
                            "data: {}\n\n",
                            chunk_json(
                                &s.req_id,
                                &s.model,
                                s.created,
                                json!({"content": tail.text}),
                                None,
                                logprobs_chunk,
                            )
                        ));
                    }
                    let final_reason = if tail.finished == Some(TokenizerFinishReason::StopString) {
                        "stop"
                    } else {
                        finish_reason_text(reason)
                    };
                    payloads.push(format!(
                        "data: {}\n\n",
                        chunk_json(
                            &s.req_id,
                            &s.model,
                            s.created,
                            json!({}),
                            Some(final_reason),
                            None
                        )
                    ));
                    if s.include_usage {
                        payloads.push(format!(
                            "data: {}\n\n",
                            usage_chunk_json(
                                &s.req_id,
                                &s.model,
                                s.created,
                                s.prompt_tokens,
                                s.decoder.token_count(),
                            )
                        ));
                    }
                    payloads.push("data: [DONE]\n\n".to_owned());
                    s.sent_done = true;
                    Some((Ok::<_, std::convert::Infallible>(bytes_from(payloads)), s))
                }
                Some(LoopEvent::Failed { message, .. }) => {
                    s.guard.disarm();
                    payloads.push(format!(
                        "data: {}\n\n",
                        error_body("server_error", "engine_faulted", &message)
                    ));
                    payloads.push("data: [DONE]\n\n".to_owned());
                    s.sent_done = true;
                    Some((Ok::<_, std::convert::Infallible>(bytes_from(payloads)), s))
                }
                None => {
                    s.guard.disarm();
                    None
                }
            }
        });
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
        sampling,
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
    let enable_logprobs = request.logprobs.unwrap_or(false);
    let top_logprobs_requested = request.top_logprobs.is_some();
    let mut logprobs_content = Vec::new();
    let mut text = String::new();
    let mut finish_reason_str = "length";
    let mut failure: Option<String> = None;

    while let Some(event) = rx.recv().await {
        match event {
            LoopEvent::Token { token, logprob, .. } => {
                if enable_logprobs {
                    logprobs_content.push(format_logprob_item(
                        &state.tokenizer,
                        token,
                        logprob.as_ref(),
                        top_logprobs_requested,
                    ));
                }
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
    let logprobs_payload = if enable_logprobs {
        Some(json!({
            "content": logprobs_content,
        }))
    } else {
        None
    };
    let body = json!({
        "id": request_id,
        "object": "chat.completion",
        "created": created,
        "model": model,
        "choices": [{
            "index": 0,
            "message": {"role": "assistant", "content": text},
            "logprobs": logprobs_payload,
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
        api_key: args.api_key.clone(),
        allow_remote_metrics: args.allow_remote_metrics,
        tokenizer,
        served_model_name: args.served_model_name.clone(),
        tokenizer_mode: args.tokenizer_mode.clone(),
        enable_prefix_caching: args.enable_prefix_caching,
        speculative_enabled: args.speculative.enabled(),
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
        let admin_listener = if let Some(admin_port) = args.admin_port {
            let admin_addr = SocketAddr::from(([127, 0, 0, 1], admin_port));
            Some(tokio::net::TcpListener::bind(admin_addr).await?)
        } else {
            None
        };
        let admin_bound = match &admin_listener {
            Some(listener) => Some(listener.local_addr()?),
            None => None,
        };
        println!(
            "zstep_serve git_sha={git_sha} listening={bound} admin_listening={admin_bound:?} auth_enabled={} model={} model_dir={} world_size={} page_block_tokens={} num_kv_pages={num_kv_pages} max_model_len={} max_seqs={} max_step_tokens={} kv_cache_dtype={} enable_prefix_caching={}",
            args.api_key.is_some(),
            args.served_model_name,
            args.model_dir.display(),
            args.world_size,
            args.page_block_tokens,
            args.max_model_len,
            args.max_seqs,
            args.max_step_tokens,
            args.kv_cache_dtype,
            args.enable_prefix_caching,
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
            })?;

        let (admin_stop_tx, mut admin_stop_rx) = watch::channel(false);
        let admin_task = if let Some(admin_listener) = admin_listener {
            let admin_app = admin_loopback_router(Arc::clone(&state))
                .into_make_service_with_connect_info::<SocketAddr>();
            Some(tokio::spawn(async move {
                let _ = axum::serve(admin_listener, admin_app)
                    .with_graceful_shutdown(async move {
                        let _ = admin_stop_rx.changed().await;
                    })
                    .await;
            }))
        } else {
            None
        };

        let launch_done_rx = Arc::new(tokio::sync::Mutex::new(Some(launch_done_rx)));
        let shutdown_launch_rx = Arc::clone(&launch_done_rx);
        let shutdown_state = Arc::clone(&state);
        let drain_timeout = args.drain_timeout;
        let app = router_with_ops(Arc::clone(&state), args.admin_port.is_none())
            .into_make_service_with_connect_info::<SocketAddr>();
        let serve_res = axum::serve(listener, app)
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
                if let Some(rx) = shutdown_launch_rx.lock().await.take() {
                    let _ = rx.await;
                }
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
                    // Shut down `ServeLoop` before `with_graceful_shutdown` returns so any
                    // remaining SSE streams drop their event channels immediately when
                    // `drain_timeout` expires instead of holding `axum::serve` open.
                    handle.shutdown();
                    let _ = tokio::task::spawn_blocking(move || {
                        handle.join(Duration::from_secs(30))
                    })
                    .await;
                }
            })
            .await;

        // Unconditional idempotent teardown on both graceful signal exit and `axum::serve` errors
        // so `zstep-engine-launch`, `admin_task`, and SHM worker segments never leak.
        state.shutdown_requested.store(true, Ordering::Release);
        request_drain(&state);
        let _ = admin_stop_tx.send(true);
        if let Some(rx) = launch_done_rx.lock().await.take() {
            let _ = rx.await;
        }
        if let Some(handle) = state.handle() {
            handle.shutdown();
            let _ = tokio::task::spawn_blocking(move || handle.join(Duration::from_secs(30))).await;
        }
        {
            let (lock, cvar) = &*exit_thread_pair;
            let mut stop = lock.lock().unwrap_or_else(|poisoned| poisoned.into_inner());
            *stop = true;
            cvar.notify_all();
        }
        if let Some(admin_task) = admin_task {
            let _ = admin_task.await;
        }
        let _ = tokio::task::spawn_blocking(move || {
            let _ = launch_join_handle.join();
        })
        .await;
        serve_res?;
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
    let (apc_pages_per_node, apc_recurrent_checkpoints) =
        args.resolved_apc_geometry().unwrap_or((0, 0));
    let tokio_handle = tokio::runtime::Handle::current();
    match ServingEngine::launch(config) {
        Ok(mut engine) => {
            if state.shutdown_requested.load(Ordering::Acquire) {
                let _ = engine.shutdown();
                let _ = launch_done_tx.send(());
                return;
            }
            if apc_pages_per_node > 0 {
                if let Err(error) = engine
                    .enable_prefix_caching(apc_pages_per_node, apc_recurrent_checkpoints as usize)
                {
                    let _ = engine.shutdown();
                    if !state.shutdown_requested.load(Ordering::Acquire) {
                        eprintln!("zstep_serve prefix caching initialization failed: {error}");
                        state.set_phase(EngineState::Faulted);
                    }
                    let _ = launch_done_tx.send(());
                    return;
                }
                if let Some(ref endpoint) = args.kv_events_endpoint {
                    let topic = args
                        .kv_events_topic
                        .clone()
                        .unwrap_or_else(|| format!("{}/0", args.served_model_name));
                    let publisher = KvEventPublisher::spawn(endpoint.clone(), topic, &tokio_handle);
                    engine.set_prefix_event_sink(Some(publisher.sink()));
                }
            }
            if args.dram_tier.is_enabled()
                && let Err(error) = engine.enable_dram_kv_cache(args.dram_tier)
            {
                let _ = engine.shutdown();
                if !state.shutdown_requested.load(Ordering::Acquire) {
                    eprintln!("zstep_serve Host DRAM KV tier initialization failed: {error}");
                    state.set_phase(EngineState::Faulted);
                }
                let _ = launch_done_tx.send(());
                return;
            }
            engine.set_max_mixed_step_tokens(mixed);
            engine.set_prefill_scheduling_policy(args.prefill_policy);
            engine.set_prefill_chunk_align(if is_dsv41 { 4 } else { 1 });
            engine.set_device_token_overlap(
                std::env::var("ZSTEP_DECODE_DEVICE_TOKENS").map_or(true, |v| v != "0"),
            );
            let online = OnlineConfig {
                eos_token_ids,
                max_waiting_requests: args.max_waiting_requests,
            };
            let handle = spawn_serve_loop_with_state(engine, online, Arc::clone(&state.phase));
            publish_launched_handle(&state, handle);
            println!(
                "zstep_serve engine ready (mixed_step_tokens={mixed:?}, enable_prefix_caching={}, apc_pages_per_node={apc_pages_per_node}, apc_recurrent_checkpoints={apc_recurrent_checkpoints})",
                args.enable_prefix_caching
            );
            let _ = launch_done_tx.send(());
            // Release `Arc<AppState>` before keeping this spawning thread alive until shutdown
            // completes (`PR_SET_PDEATHSIG` on Linux tracks the spawning thread's lifetime).
            drop(state);
            let (lock, cvar) = &*exit_thread_pair;
            let mut stop = lock.lock().unwrap_or_else(|poisoned| poisoned.into_inner());
            while !*stop {
                stop = cvar
                    .wait(stop)
                    .unwrap_or_else(|poisoned| poisoned.into_inner());
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

#[cfg(test)]
mod tests;
