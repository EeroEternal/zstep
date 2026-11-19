// Copyright (c) 2026 zstep Contributors
// SPDX-License-Identifier: MIT

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
        api_key: None,
        allow_remote_metrics: true,
        tokenizer: test_tokenizer(),
        served_model_name: "zstep-test".to_owned(),
        tokenizer_mode: "auto".to_owned(),
        enable_prefix_caching: false,
        speculative_enabled: false,
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
        let svc = router(state).into_make_service_with_connect_info::<SocketAddr>();
        let _ = axum::serve(listener, svc).await;
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
    let mut request = format!("{method} {path} HTTP/1.1\r\nHost: test\r\nConnection: close\r\n");
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
    format!(r#"{{"messages":[{{"role":"user","content":"{content}"}}],"stream":{stream}{extra}}}"#)
}

#[test]
fn cli_parser_enforces_nebula_contract_flags_and_model_defaults() {
    let parse = |items: &[&str]| {
        let raw: Vec<String> = items.iter().copied().map(str::to_owned).collect();
        ServeArgs::parse(&raw)
    };
    assert!(parse(&["--model-dir"]).is_err(), "missing value must fail");
    assert!(
        parse(&["--model-dir", "/tmp/m", "--page-block-tokens", "0"]).is_err(),
        "zero page_block_tokens must fail"
    );
    assert!(
        parse(&["--model-dir", "/tmp/m", "--page-block-tokens", "24"]).is_err(),
        "hybrid APC requires page_block_tokens dividing 64"
    );
    assert!(
        parse(&["--model-dir", "/tmp/m", "--world-size", "16"]).is_err(),
        "world_size > 8 must fail"
    );
    assert!(
        parse(&["--model-dir", "/tmp/qwen35", "--kv-cache-dtype", "fp8",]).is_err(),
        "fp8 kv_cache_dtype must fail loudly on BF16 model runner"
    );
    let default_apc = parse(&["--model-dir", "/tmp/qwen35"]).expect("default qwen args");
    assert!(default_apc.enable_prefix_caching);
    assert_eq!(default_apc.resolved_apc_geometry().expect("apc"), (4, 255));
    assert_eq!(
        default_apc
            .engine_config(None)
            .launch
            .num_recurrent_checkpoints,
        255
    );

    let ok = parse(&[
        "--model-dir",
        "/tmp/qwen35",
        "--enable-prefix-caching",
        "--no-enable-prefix-caching",
        "--kv-cache-dtype",
        "bf16",
        "--max-model-len",
        "16384",
        "--gpu-memory-utilization",
        "0.95",
    ])
    .expect("valid qwen args");
    assert!(!ok.enable_prefix_caching);
    assert_eq!(ok.resolved_apc_geometry().expect("disabled apc"), (0, 0));
    assert_eq!(ok.page_block_tokens, 16);
    assert_eq!(ok.resolved_num_kv_pages(), 49_152);
    assert_eq!(ok.engine_config(None).launch.max_model_len, Some(16_384));
    assert_eq!(ok.engine_config(None).launch.num_recurrent_checkpoints, 0);

    let reenabled = parse(&[
        "--model-dir",
        "/tmp/qwen35",
        "--no-enable-prefix-caching",
        "--enable-prefix-caching",
    ])
    .expect("last flag wins");
    assert!(reenabled.enable_prefix_caching);

    let dsv41_dir =
        TempModelDir(std::env::temp_dir().join(format!("zstep_dsv41_cfg_{}", std::process::id())));
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
    assert!(
        parse(&["--model-dir", dsv41_path, "--page-block-tokens", "64"]).is_err(),
        "deepseek_v41 APC requires 128-token pages"
    );
    let dsv41_ok =
        parse(&["--model-dir", dsv41_path, "--kv-cache-dtype", "fp8"]).expect("dsv41 fp8 ok");
    assert!(dsv41_ok.enable_prefix_caching);
    assert_eq!(dsv41_ok.page_block_tokens, 128);
    assert_eq!(dsv41_ok.resolved_apc_geometry().expect("dsv41 apc"), (1, 0));
    assert_eq!(dsv41_ok.resolved_num_kv_pages(), 2_560);
    let dsv41_dspark = parse(&[
        "--model-dir",
        dsv41_path,
        "--speculative-method",
        "dspark",
        "--num-speculative-tokens",
        "5",
    ])
    .expect("dsv41 dspark ok");
    assert_eq!(dsv41_dspark.resolved_num_kv_pages(), 1_536);
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
        api_key: None,
        allow_remote_metrics: true,
        tokenizer: test_tokenizer(),
        served_model_name: "zstep-test".to_owned(),
        tokenizer_mode: "auto".to_owned(),
        enable_prefix_caching: false,
        speculative_enabled: false,
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
        chat_body("hi", false, r#","temperature":-0.1"#),
        chat_body("hi", false, r#","temperature":2.5"#),
        chat_body("hi", false, r#","top_p":0.0"#),
        chat_body("hi", false, r#","top_p":1e-50"#),
        chat_body("hi", false, r#","top_p":1.5"#),
        chat_body("hi", false, r#","top_k":-2"#),
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

#[tokio::test(flavor = "multi_thread", worker_threads = 2)]
async fn metrics_exposes_apc_counters_iff_prefix_caching_is_enabled() {
    let handle = zs_scheduler::spawn_mock(zs_scheduler::MockScript::default());
    let metrics = handle.metrics();
    metrics.prefix_cache_queries.store(1024, Ordering::Relaxed);
    metrics.prefix_cache_hits.store(768, Ordering::Relaxed);
    metrics.prefix_cache_evictions.store(16, Ordering::Relaxed);

    let apc_state = Arc::new(AppState {
        phase: handle.state_arc(),
        handle: Mutex::new(Some(handle.clone())),
        shutdown_requested: Arc::new(AtomicBool::new(false)),
        api_key: None,
        allow_remote_metrics: true,
        tokenizer: test_tokenizer(),
        served_model_name: "zstep-test".to_owned(),
        tokenizer_mode: "auto".to_owned(),
        enable_prefix_caching: true,
        speculative_enabled: false,
        max_model_len: 4096,
        page_block_tokens: 16,
        max_pages_per_seq: 256,
        num_kv_pages: 4096,
        created_at: 0,
        started: Instant::now(),
        request_counter: AtomicU64::new(0),
    });
    let address = spawn_server(apc_state).await;
    let (status, _, prom) =
        tokio::task::spawn_blocking(move || http(address, "GET", "/metrics", None, &[]))
            .await
            .expect("join");
    assert_eq!(status, 200);
    assert!(
        prom.contains("zstep:prefix_cache_queries_total 1024\n"),
        "{prom}"
    );
    assert!(
        prom.contains("zstep:prefix_cache_hits_total 768\n"),
        "{prom}"
    );
    assert!(
        prom.contains("zstep:prefix_cache_evictions_total 16\n"),
        "{prom}"
    );

    handle.shutdown();
    let _ = handle.join(Duration::from_secs(2));
}

#[test]
fn speculative_cli_validation_rejects_mismatched_knobs() {
    let tmp = std::env::temp_dir().join(format!(
        "zstep_serve_cli_test_{}_{}",
        std::process::id(),
        std::time::SystemTime::now()
            .duration_since(std::time::UNIX_EPOCH)
            .expect("clock")
            .as_nanos()
    ));
    let qwen_dir = tmp.join("qwen35");
    let ds_dir = tmp.join("deepseek_v41");
    std::fs::create_dir_all(&qwen_dir).expect("mkdir qwen");
    std::fs::create_dir_all(&ds_dir).expect("mkdir ds");
    std::fs::write(
        qwen_dir.join("config.json"),
        r#"{"model_type":"qwen3_5_moe"}"#,
    )
    .expect("write qwen config");
    std::fs::write(
        ds_dir.join("config.json"),
        r#"{"model_type":"deepseek_v41"}"#,
    )
    .expect("write ds config");

    let parse = |dir: &std::path::Path, extra: &[&str]| {
        let mut argv = vec!["--model-dir".to_owned(), dir.display().to_string()];
        argv.extend(extra.iter().map(|item| (*item).to_owned()));
        ServeArgs::parse(&argv)
    };

    // Explicit `--num-speculative-tokens` without `--speculative-method` (or with `none`) must fail.
    assert!(
        parse(&qwen_dir, &["--num-speculative-tokens", "3"])
            .unwrap_err()
            .contains("--num-speculative-tokens requires --speculative-method")
    );
    assert!(
        parse(
            &qwen_dir,
            &[
                "--speculative-method",
                "none",
                "--num-speculative-tokens",
                "0"
            ]
        )
        .unwrap_err()
        .contains("--num-speculative-tokens requires --speculative-method")
    );

    // `--prompt-lookup-max` / `--prompt-lookup-min` require `ngram`.
    assert!(
        parse(
            &qwen_dir,
            &["--speculative-method", "mtp", "--prompt-lookup-max", "4"]
        )
        .unwrap_err()
        .contains("require --speculative-method ngram")
    );
    assert!(
        parse(
            &ds_dir,
            &["--speculative-method", "dspark", "--prompt-lookup-min", "2"]
        )
        .unwrap_err()
        .contains("require --speculative-method ngram")
    );

    // Model-family compatibility checks.
    assert!(
        parse(&qwen_dir, &["--speculative-method", "dspark"])
            .unwrap_err()
            .contains("requires a deepseek_v41 model")
    );
    assert!(
        parse(&ds_dir, &["--speculative-method", "mtp"])
            .unwrap_err()
            .contains("requires the Qwen3.5 runner")
    );

    // Valid configurations parse cleanly with expected defaults and overrides.
    let mtp = parse(
        &qwen_dir,
        &[
            "--speculative-method",
            "mtp",
            "--num-speculative-tokens",
            "3",
            "--worker-root",
            "/opt/zstep/worker",
        ],
    )
    .expect("valid mtp");
    assert_eq!(mtp.speculative.method, SpeculativeMethod::Mtp);
    assert_eq!(mtp.speculative.num_speculative_tokens, 3);
    assert_eq!(mtp.worker_root, std::path::Path::new("/opt/zstep/worker"));

    let dspark = parse(&ds_dir, &["--speculative-method", "dspark"]).expect("valid dspark");
    assert_eq!(dspark.speculative.method, SpeculativeMethod::Dspark);
    assert_eq!(dspark.speculative.num_speculative_tokens, 5);

    let ngram = parse(
        &qwen_dir,
        &[
            "--speculative-method",
            "ngram",
            "--num-speculative-tokens",
            "4",
            "--prompt-lookup-max",
            "6",
            "--prompt-lookup-min",
            "2",
        ],
    )
    .expect("valid ngram");
    assert_eq!(ngram.speculative.method, SpeculativeMethod::Ngram);
    assert_eq!(ngram.speculative.num_speculative_tokens, 4);
    assert_eq!(ngram.speculative.prompt_lookup_max, 6);
    assert_eq!(ngram.speculative.prompt_lookup_min, 2);

    let _ = std::fs::remove_dir_all(&tmp);
}

#[test]
fn metrics_exposes_spec_decode_three_state_contract() {
    let handle = zs_scheduler::spawn_mock(zs_scheduler::MockScript::default());

    // 1) `speculative_enabled == false`: strictly omit `zstep:spec_decode_*`.
    let disabled_state = test_state(EngineState::Ready, Some(handle.clone()));
    let prom_disabled = render_metrics(&disabled_state);
    assert!(
        !prom_disabled.contains("zstep:spec_decode_"),
        "{prom_disabled}"
    );

    // 2) `speculative_enabled == true` with `drafts == 0`: emit zero counters, omit gauge `tau`.
    let idle_spec_state = AppState {
        phase: handle.state_arc(),
        handle: Mutex::new(Some(handle.clone())),
        shutdown_requested: Arc::new(AtomicBool::new(false)),
        api_key: None,
        allow_remote_metrics: true,
        tokenizer: test_tokenizer(),
        served_model_name: "zstep-test".to_owned(),
        tokenizer_mode: "auto".to_owned(),
        enable_prefix_caching: false,
        speculative_enabled: true,
        max_model_len: 4096,
        page_block_tokens: 16,
        max_pages_per_seq: 256,
        num_kv_pages: 4096,
        created_at: 0,
        started: Instant::now(),
        request_counter: AtomicU64::new(0),
    };
    let prom_idle = render_metrics(&idle_spec_state);
    assert!(
        prom_idle.contains("zstep:spec_decode_num_drafts_total 0\n"),
        "{prom_idle}"
    );
    assert!(
        prom_idle.contains("zstep:spec_decode_num_draft_tokens_total 0\n"),
        "{prom_idle}"
    );
    assert!(
        prom_idle.contains("zstep:spec_decode_num_accepted_tokens_total 0\n"),
        "{prom_idle}"
    );
    assert!(
        !prom_idle.contains("zstep:spec_decode_mean_acceptance_length"),
        "{prom_idle}"
    );

    // 3) `speculative_enabled == true` with `drafts > 0`: emit `1 + accepted / drafts`.
    let metrics = handle.metrics();
    metrics.spec_decode_num_drafts.store(10, Ordering::Relaxed);
    metrics
        .spec_decode_num_draft_tokens
        .store(30, Ordering::Relaxed);
    metrics
        .spec_decode_num_accepted_tokens
        .store(24, Ordering::Relaxed);
    let prom_active = render_metrics(&idle_spec_state);
    assert!(
        prom_active.contains("zstep:spec_decode_num_drafts_total 10\n"),
        "{prom_active}"
    );
    assert!(
        prom_active.contains("zstep:spec_decode_num_draft_tokens_total 30\n"),
        "{prom_active}"
    );
    assert!(
        prom_active.contains("zstep:spec_decode_num_accepted_tokens_total 24\n"),
        "{prom_active}"
    );
    assert!(
        prom_active.contains("zstep:spec_decode_mean_acceptance_length 3.4\n"),
        "{prom_active}"
    );

    handle.shutdown();
    let _ = handle.join(Duration::from_secs(2));
}

#[test]
fn default_host_is_loopback_and_security_flags_validate_and_redact() {
    let raw = vec!["--model-dir".to_owned(), "/tmp/qwen35".to_owned()];
    let defaults = ServeArgs::parse_with_env(&raw, None, None).expect("defaults");
    assert_eq!(defaults.host, "127.0.0.1");
    assert_eq!(defaults.port, 43537);
    assert!(defaults.api_key.is_none());
    assert!(defaults.admin_port.is_none());
    assert!(defaults.allow_remote_metrics);

    // `ZSTEP_API_KEY` env fallback + `ZSTEP_ALLOW_REMOTE_METRICS=0` opt-out + redacted Debug.
    let from_env = ServeArgs::parse_with_env(&raw, Some("  sk-env-secret-123  "), Some(false))
        .expect("env key");
    assert_eq!(
        from_env.api_key.as_ref().map(SecretToken::as_bytes),
        Some(b"sk-env-secret-123".as_slice())
    );
    assert!(!from_env.allow_remote_metrics);
    let dbg = format!("{from_env:?}");
    assert!(
        !dbg.contains("sk-env-secret-123"),
        "secret leaked in Debug: {dbg}"
    );
    assert!(dbg.contains("[REDACTED]"), "{dbg}");

    // CLI `--api-key` overrides `ZSTEP_API_KEY` (including non-flag tokens beginning with `--`),
    // and `--no-allow-remote-metrics` overrides `env_remote_metrics = Some(true)`.
    let cli_override: Vec<String> = [
        "--model-dir",
        "/tmp/qwen35",
        "--api-key",
        "--leading-dash-token",
        "--admin-port",
        "9100",
        "--no-allow-remote-metrics",
    ]
    .iter()
    .map(|s| (*s).to_owned())
    .collect();
    let parsed =
        ServeArgs::parse_with_env(&cli_override, Some("sk-env"), Some(true)).expect("cli override");
    assert_eq!(
        parsed.api_key.as_ref().map(SecretToken::as_bytes),
        Some(b"--leading-dash-token".as_slice())
    );
    assert_eq!(parsed.admin_port, Some(9100));
    assert!(!parsed.allow_remote_metrics);

    // Empty `--api-key`, missing `--api-key` value swallowing a known flag, and colliding
    // `--admin-port == --port` must all fail loudly.
    let empty_key: Vec<String> = ["--model-dir", "/tmp/qwen35", "--api-key", "   "]
        .iter()
        .map(|s| (*s).to_owned())
        .collect();
    assert!(ServeArgs::parse_with_env(&empty_key, None, None).is_err());

    let swallowed_flag: Vec<String> = [
        "--model-dir",
        "/tmp/qwen35",
        "--api-key",
        "--enable-prefix-caching",
    ]
    .iter()
    .map(|s| (*s).to_owned())
    .collect();
    assert!(ServeArgs::parse_with_env(&swallowed_flag, None, None).is_err());

    let colliding_port: Vec<String> = [
        "--model-dir",
        "/tmp/qwen35",
        "--port",
        "43537",
        "--admin-port",
        "43537",
    ]
    .iter()
    .map(|s| (*s).to_owned())
    .collect();
    assert!(ServeArgs::parse_with_env(&colliding_port, None, None).is_err());
}

#[tokio::test(flavor = "multi_thread", worker_threads = 2)]
async fn bearer_auth_protects_v1_admin_and_metrics_while_keeping_probes_open() {
    let handle = zs_scheduler::spawn_mock(zs_scheduler::MockScript {
        tokens: vec![1],
        ..Default::default()
    });
    handle.set_state(EngineState::Ready);
    let state = Arc::new(AppState {
        phase: handle.state_arc(),
        handle: Mutex::new(Some(handle.clone())),
        shutdown_requested: Arc::new(AtomicBool::new(false)),
        api_key: Some(SecretToken::new("sk-top-secret".to_owned())),
        allow_remote_metrics: false,
        tokenizer: test_tokenizer(),
        served_model_name: "zstep-test".to_owned(),
        tokenizer_mode: "auto".to_owned(),
        enable_prefix_caching: false,
        speculative_enabled: false,
        max_model_len: 4096,
        page_block_tokens: 16,
        max_pages_per_seq: 256,
        num_kv_pages: 4096,
        created_at: 0,
        started: Instant::now(),
        request_counter: AtomicU64::new(0),
    });
    let address = spawn_server(state).await;

    // Operational health probes remain accessible without credentials.
    for path in ["/livez", "/health", "/ready"] {
        let (code, _, _) =
            tokio::task::spawn_blocking(move || http(address, "GET", path, None, &[]))
                .await
                .expect("join");
        assert_eq!(code, 200, "probe {path} must stay 200");
    }

    // Missing or wrong Bearer token returns 401 + WWW-Authenticate on /v1/*, /metrics, /admin/drain.
    let body = chat_body("hi", false, "");
    for (method, path, payload) in [
        ("GET", "/v1/models", None),
        ("POST", "/v1/chat/completions", Some(body.as_str())),
        ("GET", "/metrics", None),
        ("POST", "/admin/drain", None),
    ] {
        let p = payload.map(str::to_owned);
        let (code_missing, hdrs_missing, err_missing) = tokio::task::spawn_blocking(move || {
            http(
                address,
                method,
                path,
                p.as_deref(),
                &[("x-request-id", "req_auth_401")],
            )
        })
        .await
        .expect("join");
        assert_eq!(code_missing, 401, "{method} {path}: {err_missing}");
        assert!(
            hdrs_missing
                .to_lowercase()
                .contains("www-authenticate: bearer"),
            "{hdrs_missing}"
        );
        assert!(err_missing.contains("invalid_api_key"), "{err_missing}");

        let p_wrong = payload.map(str::to_owned);
        let (code_wrong, _, _) = tokio::task::spawn_blocking(move || {
            http(
                address,
                method,
                path,
                p_wrong.as_deref(),
                &[("Authorization", "Bearer wrong-token")],
            )
        })
        .await
        .expect("join");
        assert_eq!(code_wrong, 401, "{method} {path} wrong token");
    }

    // Valid case-insensitive `bearer` scheme succeeds across /v1/*, /metrics, and /admin/drain.
    let auth_hdr = [("Authorization", "bearer sk-top-secret")];
    let (models_ok, _, _) =
        tokio::task::spawn_blocking(move || http(address, "GET", "/v1/models", None, &auth_hdr))
            .await
            .expect("join");
    assert_eq!(models_ok, 200);

    let chat_ok_body = chat_body("hi", false, r#","max_tokens":2"#);
    let (chat_ok, _, _) = tokio::task::spawn_blocking(move || {
        http(
            address,
            "POST",
            "/v1/chat/completions",
            Some(&chat_ok_body),
            &auth_hdr,
        )
    })
    .await
    .expect("join");
    assert_eq!(chat_ok, 200);

    let (metrics_ok, _, _) =
        tokio::task::spawn_blocking(move || http(address, "GET", "/metrics", None, &auth_hdr))
            .await
            .expect("join");
    assert_eq!(metrics_ok, 200);

    let (drain_ok, _, _) =
        tokio::task::spawn_blocking(move || http(address, "POST", "/admin/drain", None, &auth_hdr))
            .await
            .expect("join");
    assert_eq!(drain_ok, 200);

    handle.shutdown();
    let _ = handle.join(Duration::from_secs(2));
}

#[tokio::test(flavor = "multi_thread", worker_threads = 2)]
async fn separate_admin_port_and_remote_peer_guard_block_unauthenticated_lan_drain() {
    let handle = zs_scheduler::spawn_mock(zs_scheduler::MockScript::default());
    let state = test_state(EngineState::Ready, Some(handle.clone()));

    // 1) Non-loopback peer (`172.17.0.1:51000`, e.g. Docker bridge NAT / K8s Pod scraper)
    //    without `api_key`: mutating `/admin/drain` is strictly 403, read-only `/metrics` is
    //    200 by default (`allow_remote_metrics = true`) and 403 when `--no-allow-remote-metrics`.
    let mut remote_drain_req = Request::builder()
        .method("POST")
        .uri("/admin/drain")
        .body(Body::empty())
        .expect("req");
    remote_drain_req
        .extensions_mut()
        .insert(ConnectInfo(SocketAddr::from(([172, 17, 0, 1], 51000))));
    let drain_resp = admin_drain(State(state.clone()), remote_drain_req).await;
    assert_eq!(drain_resp.status(), StatusCode::FORBIDDEN);
    assert_eq!(
        state.phase(),
        EngineState::Ready,
        "remote 403 must not drain"
    );

    let mut remote_metrics_default = Request::builder()
        .method("GET")
        .uri("/metrics")
        .body(Body::empty())
        .expect("req");
    remote_metrics_default
        .extensions_mut()
        .insert(ConnectInfo(SocketAddr::from(([172, 17, 0, 1], 51001))));
    let metrics_default_resp = metrics(State(state.clone()), remote_metrics_default).await;
    assert_eq!(metrics_default_resp.status(), StatusCode::OK);

    let locked_metrics_state = Arc::new(AppState {
        phase: handle.state_arc(),
        handle: Mutex::new(Some(handle.clone())),
        shutdown_requested: Arc::new(AtomicBool::new(false)),
        api_key: None,
        allow_remote_metrics: false,
        tokenizer: test_tokenizer(),
        served_model_name: "zstep-test".to_owned(),
        tokenizer_mode: "auto".to_owned(),
        enable_prefix_caching: false,
        speculative_enabled: false,
        max_model_len: 4096,
        page_block_tokens: 16,
        max_pages_per_seq: 256,
        num_kv_pages: 4096,
        created_at: 0,
        started: Instant::now(),
        request_counter: AtomicU64::new(0),
    });
    let mut remote_metrics_locked = Request::builder()
        .method("GET")
        .uri("/metrics")
        .body(Body::empty())
        .expect("req");
    remote_metrics_locked
        .extensions_mut()
        .insert(ConnectInfo(SocketAddr::from(([172, 17, 0, 1], 51002))));
    let metrics_locked_resp = metrics(State(locked_metrics_state), remote_metrics_locked).await;
    assert_eq!(metrics_locked_resp.status(), StatusCode::FORBIDDEN);

    // 2) Dedicated `--admin-port`: public router omits `/metrics` and `/admin/drain` (404),
    //    while loopback admin router serves both cleanly AND enforces `--api-key` when set.
    let pub_listener = tokio::net::TcpListener::bind("127.0.0.1:0")
        .await
        .expect("bind pub");
    let pub_addr = pub_listener.local_addr().expect("pub addr");
    let pub_state = state.clone();
    tokio::spawn(async move {
        let svc =
            router_with_ops(pub_state, false).into_make_service_with_connect_info::<SocketAddr>();
        let _ = axum::serve(pub_listener, svc).await;
    });

    let adm_listener = tokio::net::TcpListener::bind("127.0.0.1:0")
        .await
        .expect("bind adm");
    let adm_addr = adm_listener.local_addr().expect("adm addr");
    let adm_state = state.clone();
    tokio::spawn(async move {
        let svc =
            admin_loopback_router(adm_state).into_make_service_with_connect_info::<SocketAddr>();
        let _ = axum::serve(adm_listener, svc).await;
    });

    let (pub_metrics, _, _) =
        tokio::task::spawn_blocking(move || http(pub_addr, "GET", "/metrics", None, &[]))
            .await
            .expect("join");
    let (pub_drain, _, _) =
        tokio::task::spawn_blocking(move || http(pub_addr, "POST", "/admin/drain", None, &[]))
            .await
            .expect("join");
    assert_eq!((pub_metrics, pub_drain), (404, 404));

    let (adm_metrics, _, _) =
        tokio::task::spawn_blocking(move || http(adm_addr, "GET", "/metrics", None, &[]))
            .await
            .expect("join");
    let (adm_drain, _, _) =
        tokio::task::spawn_blocking(move || http(adm_addr, "POST", "/admin/drain", None, &[]))
            .await
            .expect("join");
    assert_eq!((adm_metrics, adm_drain), (200, 200));
    assert_eq!(state.phase(), EngineState::Draining);

    // 3) `--admin-port` combined with `--api-key` still enforces Bearer auth on loopback.
    handle.set_state(EngineState::Ready);
    let keyed_adm_state = Arc::new(AppState {
        phase: handle.state_arc(),
        handle: Mutex::new(Some(handle.clone())),
        shutdown_requested: Arc::new(AtomicBool::new(false)),
        api_key: Some(SecretToken::new("sk-admin-key".to_owned())),
        allow_remote_metrics: true,
        tokenizer: test_tokenizer(),
        served_model_name: "zstep-test".to_owned(),
        tokenizer_mode: "auto".to_owned(),
        enable_prefix_caching: false,
        speculative_enabled: false,
        max_model_len: 4096,
        page_block_tokens: 16,
        max_pages_per_seq: 256,
        num_kv_pages: 4096,
        created_at: 0,
        started: Instant::now(),
        request_counter: AtomicU64::new(0),
    });
    let keyed_adm_listener = tokio::net::TcpListener::bind("127.0.0.1:0")
        .await
        .expect("bind keyed adm");
    let keyed_adm_addr = keyed_adm_listener.local_addr().expect("keyed adm addr");
    tokio::spawn(async move {
        let svc = admin_loopback_router(keyed_adm_state)
            .into_make_service_with_connect_info::<SocketAddr>();
        let _ = axum::serve(keyed_adm_listener, svc).await;
    });

    let (keyed_unauth_m, _, _) =
        tokio::task::spawn_blocking(move || http(keyed_adm_addr, "GET", "/metrics", None, &[]))
            .await
            .expect("join");
    let (keyed_unauth_d, _, _) = tokio::task::spawn_blocking(move || {
        http(keyed_adm_addr, "POST", "/admin/drain", None, &[])
    })
    .await
    .expect("join");
    assert_eq!((keyed_unauth_m, keyed_unauth_d), (401, 401));

    let auth = [("Authorization", "Bearer sk-admin-key")];
    let (keyed_ok_m, _, _) =
        tokio::task::spawn_blocking(move || http(keyed_adm_addr, "GET", "/metrics", None, &auth))
            .await
            .expect("join");
    let (keyed_ok_d, _, _) = tokio::task::spawn_blocking(move || {
        http(keyed_adm_addr, "POST", "/admin/drain", None, &auth)
    })
    .await
    .expect("join");
    assert_eq!((keyed_ok_m, keyed_ok_d), (200, 200));

    handle.shutdown();
    let _ = handle.join(Duration::from_secs(2));
}

#[tokio::test(flavor = "multi_thread", worker_threads = 2)]
async fn poisoned_handle_mutex_recovers_without_panicking_handlers() {
    let handle = zs_scheduler::spawn_mock(zs_scheduler::MockScript::default());
    let state = test_state(EngineState::Ready, Some(handle.clone()));
    let poison_ref = Arc::clone(&state);
    let _ = std::panic::catch_unwind(std::panic::AssertUnwindSafe(move || {
        let _guard = poison_ref.handle.lock().unwrap();
        panic!("simulate panic while holding handle mutex");
    }));
    assert!(state.handle.is_poisoned());

    let address = spawn_server(state.clone()).await;
    let (health_code, _, _) =
        tokio::task::spawn_blocking(move || http(address, "GET", "/health", None, &[]))
            .await
            .expect("join");
    let (metrics_code, _, _) =
        tokio::task::spawn_blocking(move || http(address, "GET", "/metrics", None, &[]))
            .await
            .expect("join");
    let (drain_code, _, _) =
        tokio::task::spawn_blocking(move || http(address, "POST", "/admin/drain", None, &[]))
            .await
            .expect("join");
    assert_eq!((health_code, metrics_code, drain_code), (200, 200, 200));
    assert_eq!(state.phase(), EngineState::Draining);

    handle.shutdown();
    let _ = handle.join(Duration::from_secs(2));
}

#[tokio::test(flavor = "multi_thread", worker_threads = 2)]
async fn valid_sampling_params_accepted_for_stochastic_and_greedy_requests() {
    let handle = zs_scheduler::spawn_mock(zs_scheduler::MockScript::default());
    let state = test_state(EngineState::Ready, Some(handle.clone()));
    let address = spawn_server(state).await;
    let cases = [
        chat_body("hi", false, r#","temperature":0.0,"top_p":1.0,"top_k":0"#),
        chat_body(
            "hi",
            false,
            r#","temperature":0.7,"top_p":0.9,"top_k":40,"seed":42"#,
        ),
        chat_body("hi", false, r#","temperature":1.2,"top_k":-1"#),
        chat_body("hi", false, r#","temperature":2.0,"top_p":0.01,"seed":-7"#),
    ];
    for body in cases {
        let request_body = body.clone();
        let (status, _, response) = tokio::task::spawn_blocking(move || {
            http(
                address,
                "POST",
                "/v1/chat/completions",
                Some(&request_body),
                &[],
            )
        })
        .await
        .expect("join");
        assert_eq!(status, 200, "body={body} response={response}");
    }

    handle.shutdown();
    let _ = handle.join(Duration::from_secs(2));
}
