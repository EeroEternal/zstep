// Copyright (c) 2026 zstep Contributors
// SPDX-License-Identifier: MIT

use super::*;

pub const SERVE_USAGE: &str = "zstep serve — persistent OpenAI-compatible HTTP/SSE engine\n\
Usage: zstep serve --model-dir <PATH> [OPTIONS]\n\
  --model-dir PATH               Model weights & tokenizer.json directory (required)\n\
  --served-model-name ID         Model ID exposed on /v1/models (default: directory basename)\n\
  --host IP                      Bind address (default: 127.0.0.1)\n\
  --port PORT                    Bind port (default: 43537)\n\
  --api-key TOKEN                Optional Bearer token for /v1/*, /admin/* & /metrics (or ZSTEP_API_KEY)\n\
  --admin-port PORT              Optional loopback-only port (127.0.0.1:PORT) for /admin/* & /metrics\n\
  --allow-remote-metrics         Allow unauthenticated non-loopback GET /metrics (default: enabled)\n\
  --no-allow-remote-metrics      Restrict unauthenticated GET /metrics to loopback (or ZSTEP_ALLOW_REMOTE_METRICS=0)\n\
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
  --enable-prefix-caching        Enable Automatic Prefix Caching (APC, default: enabled)\n\
  --no-enable-prefix-caching     Disable Automatic Prefix Caching (APC)\n\
  --apc-gdn-node-tokens N        GDN hybrid APC node granularity: 64 | 128 (default: 128)\n\
  --kv-events-endpoint ADDR      Optional ZeroMQ PUB endpoint for BlockStored/BlockRemoved events\n\
  --kv-events-topic TOPIC        ZeroMQ topic (default: <served-model-name>/0)\n\
  --dram-kv-blocks N             Host DRAM KV blocks per GPU (default: 0 = disabled; enables Exclusive DRAM + sm_h2d)\n\
  --kv-loading-area-ratio F      HBM fraction reserved for DRAM->HBM reloads in (0.0, 0.5] (default: 0.10)\n\
  --max-waiting-requests N       Bounded HTTP waiting queue depth (default: 1024)\n\
  --prefill-policy MODE          Waiting prefill policy: srpt | fifo (default: srpt)\n\
  --prefill-starvation-cap-ms N  Max wait (ms) before long prompt preempts SRPT (default: 60000, 0=off)\n\
  --max-consecutive-prefills N   Max standalone prefill steps before decode (default: 1, 0=prefill-first)\n\
  --dynamic-prefill-budget       Scale mixed-step budget from --max-mixed-step-tokens to --max-step-tokens (default on srpt)\n\
  --no-dynamic-prefill-budget    Keep mixed-step prefill budget fixed at --max-mixed-step-tokens\n\
  --drain-timeout-secs N         SIGTERM drain grace period in seconds (default: 30)\n\
  --python PATH                  Python executable for GPU workers\n\
  --worker-root PATH             Python zstep_worker package root\n\
  --ready-timeout-secs N         Worker initialization timeout (default: 1800)\n\
  --step-timeout-secs N          Per-step timeout (default: 300)\n\
  --max-mixed-step-tokens N      Piggyback mixed-step cap (default: 3072 on deepseek_v41, 0=off)\n\
  --allreduce-blocks N           SHM AllReduce CUDA blocks (default: 8)\n\
  --allreduce-threads N          SHM AllReduce threads/block (default: 256)\n\
  --speculative-method METHOD    none | ngram | mtp | dspark (default: none)\n\
  --num-speculative-tokens K     Draft tokens per verify step, 1..=7\n\
  --prompt-lookup-max N          Maximum ngram suffix window (default: 5)\n\
  --prompt-lookup-min N          Minimum ngram suffix window (default: 2)\n";

// ---------------------------------------------------------------------------
// CLI
// ---------------------------------------------------------------------------

/// Redacted Bearer API token wrapper so `{args:?}` never leaks credentials into logs.
#[derive(Clone, PartialEq, Eq)]
pub struct SecretToken(String);

impl SecretToken {
    pub fn new(raw: String) -> Self {
        Self(raw)
    }

    pub fn as_bytes(&self) -> &[u8] {
        self.0.as_bytes()
    }
}

impl fmt::Debug for SecretToken {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.write_str("\"[REDACTED]\"")
    }
}

/// Parsed `zstep serve` arguments (contract: `nebula/docs/dev/zstep.md` §2.1/§2.2).
#[derive(Debug, Clone)]
pub struct ServeArgs {
    pub model_dir: PathBuf,
    pub served_model_name: String,
    pub host: String,
    pub port: u16,
    pub api_key: Option<SecretToken>,
    pub admin_port: Option<u16>,
    pub allow_remote_metrics: bool,
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
    pub enable_prefix_caching: bool,
    pub apc_gdn_node_tokens: u32,
    pub kv_events_endpoint: Option<String>,
    pub kv_events_topic: Option<String>,
    pub max_waiting_requests: usize,
    pub drain_timeout: Duration,
    pub python: String,
    pub worker_root: PathBuf,
    pub ready_timeout: Duration,
    pub step_timeout: Duration,
    pub max_mixed_step_tokens: Option<usize>,
    pub prefill_policy: zs_scheduler::PrefillSchedulingPolicy,
    pub dram_tier: zs_scheduler::DramTierConfig,
    pub allreduce_blocks: u32,
    pub allreduce_threads: u32,
    pub speculative: SpeculativeConfig,
    pub swa_kv_pages_env: Option<usize>,
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
        "--api-key",
        "--admin-port",
        "--world-size",
        "--gpu-memory-utilization",
        "--max-model-len",
        "--max-seqs",
        "--max-step-tokens",
        "--max-prefill-chunk",
        "--page-block-tokens",
        "--apc-gdn-node-tokens",
        "--kv-events-endpoint",
        "--kv-events-topic",
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
        "--prefill-policy",
        "--prefill-starvation-cap-ms",
        "--max-consecutive-prefills",
        "--dram-kv-blocks",
        "--kv-loading-area-ratio",
        "--allreduce-blocks",
        "--allreduce-threads",
        "--speculative-method",
        "--num-speculative-tokens",
        "--prompt-lookup-max",
        "--prompt-lookup-min",
    ];
    const BOOL_FLAGS: &[&str] = &[
        "--enable-prefix-caching",
        "--no-enable-prefix-caching",
        "--allow-remote-metrics",
        "--no-allow-remote-metrics",
        "--dynamic-prefill-budget",
        "--no-dynamic-prefill-budget",
    ];
    let mut values = BTreeMap::new();
    let mut index = 0;
    while index < argv.len() {
        let flag = argv[index].as_str();
        if flag == "--no-enable-prefix-caching" {
            values.insert("--enable-prefix-caching".to_owned(), "false".to_owned());
            index += 1;
            continue;
        }
        if flag == "--enable-prefix-caching" {
            values.insert("--enable-prefix-caching".to_owned(), "true".to_owned());
            index += 1;
            continue;
        }
        if flag == "--allow-remote-metrics" {
            values.insert("--allow-remote-metrics".to_owned(), "true".to_owned());
            index += 1;
            continue;
        }
        if flag == "--no-allow-remote-metrics" {
            values.insert("--allow-remote-metrics".to_owned(), "false".to_owned());
            index += 1;
            continue;
        }
        if flag == "--dynamic-prefill-budget" {
            values.insert("--dynamic-prefill-budget".to_owned(), "true".to_owned());
            index += 1;
            continue;
        }
        if flag == "--no-dynamic-prefill-budget" {
            values.insert("--dynamic-prefill-budget".to_owned(), "false".to_owned());
            index += 1;
            continue;
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
        let is_known_flag = VALUE_FLAGS.contains(&value.as_str())
            || BOOL_FLAGS.contains(&value.as_str())
            || matches!(value.as_str(), "--help" | "-h");
        if is_known_flag || (flag != "--api-key" && value.starts_with("--")) {
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
        let env_api_key = std::env::var("ZSTEP_API_KEY").ok();
        let env_remote_metrics = std::env::var("ZSTEP_ALLOW_REMOTE_METRICS")
            .ok()
            .filter(|v| !v.trim().is_empty())
            .map(|v| {
                !matches!(
                    v.trim(),
                    "0" | "false" | "FALSE" | "no" | "NO" | "off" | "OFF"
                )
            });
        let env_swa_kv_pages = std::env::var("ZSTEP_SWA_KV_PAGES").ok();
        Self::parse_with_all_env(
            argv,
            env_api_key.as_deref(),
            env_remote_metrics,
            env_swa_kv_pages.as_deref(),
        )
    }

    #[cfg(test)]
    pub fn parse_with_env(
        argv: &[String],
        env_api_key: Option<&str>,
        env_remote_metrics: Option<bool>,
    ) -> Result<Self, String> {
        Self::parse_with_all_env(argv, env_api_key, env_remote_metrics, None)
    }

    pub fn parse_with_all_env(
        argv: &[String],
        env_api_key: Option<&str>,
        env_remote_metrics: Option<bool>,
        env_swa_kv_pages: Option<&str>,
    ) -> Result<Self, String> {
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
        let max_prefill_chunk: usize = parse_map_number(&values, "--max-prefill-chunk", 2048usize)?;
        if max_prefill_chunk == 0 {
            return Err("--max-prefill-chunk must be positive".to_owned());
        }
        let default_step_tokens = 4096u32.max(max_prefill_chunk.min(u32::MAX as usize) as u32);
        let max_step_tokens: u32 =
            parse_map_number(&values, "--max-step-tokens", default_step_tokens)?;
        if max_step_tokens < max_seqs {
            return Err(format!(
                "--max-step-tokens ({max_step_tokens}) must be >= --max-seqs ({max_seqs})"
            ));
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
        let enable_prefix_caching = values
            .get("--enable-prefix-caching")
            .is_none_or(|raw| raw == "true");
        let apc_gdn_node_tokens: u32 = parse_map_number(
            &values,
            "--apc-gdn-node-tokens",
            zs_kv::GDN_FINE_GRAINED_TOKENS,
        )?;
        if !matches!(apc_gdn_node_tokens, 64 | 128) {
            return Err(format!(
                "--apc-gdn-node-tokens must be 64 or 128 (got {apc_gdn_node_tokens})"
            ));
        }
        let kv_events_endpoint = values
            .get("--kv-events-endpoint")
            .map(|s| s.trim().to_owned())
            .filter(|s| !s.is_empty());
        let kv_events_topic = values
            .get("--kv-events-topic")
            .map(|s| s.trim().to_owned())
            .filter(|s| !s.is_empty());
        let max_waiting_requests: usize =
            parse_map_number(&values, "--max-waiting-requests", 1024)?;
        if max_waiting_requests == 0 {
            return Err("--max-waiting-requests must be positive".to_owned());
        }
        let prefill_policy = {
            let mut policy = match values
                .get("--prefill-policy")
                .map(String::as_str)
                .unwrap_or("srpt")
            {
                "srpt" => zs_scheduler::PrefillSchedulingPolicy::default(),
                "fifo" => zs_scheduler::PrefillSchedulingPolicy::fifo(),
                other => {
                    return Err(format!(
                        "unsupported --prefill-policy `{other}` (expected `srpt` or `fifo`)"
                    ));
                }
            };
            if let Some(raw) = values.get("--prefill-starvation-cap-ms") {
                policy.starvation_cap_ms = raw.parse::<u64>().map_err(|_| {
                    format!("invalid value `{raw}` for --prefill-starvation-cap-ms")
                })?;
            }
            if let Some(raw) = values.get("--max-consecutive-prefills") {
                policy.max_consecutive_prefills = raw
                    .parse::<usize>()
                    .map_err(|_| format!("invalid value `{raw}` for --max-consecutive-prefills"))?;
            }
            if let Some(raw) = values.get("--dynamic-prefill-budget") {
                policy.dynamic_budget_enabled = raw == "true";
            }
            policy
        };
        let dram_kv_blocks: usize = parse_map_number(&values, "--dram-kv-blocks", 0usize)?;
        let kv_loading_area_ratio: f64 = parse_map_number(
            &values,
            "--kv-loading-area-ratio",
            zs_scheduler::DEFAULT_LOADING_AREA_RATIO,
        )?;
        if dram_kv_blocks > 0 && !enable_prefix_caching {
            return Err("--dram-kv-blocks requires --enable-prefix-caching".to_owned());
        }
        let dram_tier =
            zs_scheduler::DramTierConfig::enabled(dram_kv_blocks, kv_loading_area_ratio)
                .map_err(|err| err.to_string())?;
        let drain_timeout_secs: u64 = parse_map_number(&values, "--drain-timeout-secs", 30)?;
        let ready_timeout_secs: u64 = parse_map_number(&values, "--ready-timeout-secs", 1800u64)?;
        let step_timeout_secs: u64 = parse_map_number(&values, "--step-timeout-secs", 300u64)?;
        if ready_timeout_secs == 0 || step_timeout_secs == 0 {
            return Err("--ready-timeout-secs and --step-timeout-secs must be positive".to_owned());
        }
        let speculative_method = match values
            .get("--speculative-method")
            .map(String::as_str)
            .unwrap_or("none")
        {
            "none" => SpeculativeMethod::None,
            "ngram" => SpeculativeMethod::Ngram,
            "mtp" => SpeculativeMethod::Mtp,
            "dspark" => SpeculativeMethod::Dspark,
            other => return Err(format!("unsupported --speculative-method `{other}`")),
        };
        if speculative_method == SpeculativeMethod::None
            && values.contains_key("--num-speculative-tokens")
        {
            return Err(
                "--num-speculative-tokens requires --speculative-method dspark, mtp, or ngram"
                    .to_owned(),
            );
        }
        if speculative_method != SpeculativeMethod::Ngram
            && (values.contains_key("--prompt-lookup-max")
                || values.contains_key("--prompt-lookup-min"))
        {
            return Err(
                "--prompt-lookup-max and --prompt-lookup-min require --speculative-method ngram"
                    .to_owned(),
            );
        }
        let default_k = match speculative_method {
            SpeculativeMethod::None => 0,
            SpeculativeMethod::Dspark => 5,
            SpeculativeMethod::Ngram => 3,
            SpeculativeMethod::Mtp => 1,
        };
        let speculative = SpeculativeConfig {
            method: speculative_method,
            num_speculative_tokens: parse_map_number(
                &values,
                "--num-speculative-tokens",
                default_k,
            )?,
            prompt_lookup_max: parse_map_number(&values, "--prompt-lookup-max", 5)?,
            prompt_lookup_min: parse_map_number(&values, "--prompt-lookup-min", 2)?,
        };
        speculative.validate()?;
        if speculative_method == SpeculativeMethod::Dspark && !is_dsv41 {
            return Err("--speculative-method dspark requires a deepseek_v41 model".to_owned());
        }
        if speculative_method == SpeculativeMethod::Mtp && is_dsv41 {
            return Err("--speculative-method mtp requires the Qwen3.5 runner".to_owned());
        }
        if speculative.enabled()
            && max_step_tokens < max_seqs * (speculative.num_speculative_tokens as u32 + 1)
        {
            return Err(
                "--max-step-tokens must be at least max-seqs * (num-speculative-tokens + 1)"
                    .to_owned(),
            );
        }
        let port = parse_map_number(&values, "--port", 43537u16)?;
        let admin_port = values
            .get("--admin-port")
            .map(|raw| raw.parse::<u16>())
            .transpose()
            .map_err(|_| "--admin-port must be a valid TCP port (0..=65535)".to_owned())?;
        if let Some(ap) = admin_port
            && ap != 0
            && ap == port
        {
            return Err(format!(
                "--admin-port ({ap}) must differ from --port ({port})"
            ));
        }
        let api_key = if let Some(cli_key) = values.get("--api-key") {
            let trimmed = cli_key.trim();
            if trimmed.is_empty() {
                return Err("--api-key must not be empty".to_owned());
            }
            Some(SecretToken::new(trimmed.to_owned()))
        } else {
            env_api_key
                .map(str::trim)
                .filter(|v| !v.is_empty())
                .map(|v| SecretToken::new(v.to_owned()))
        };
        let allow_remote_metrics = match values.get("--allow-remote-metrics").map(String::as_str) {
            Some("true") => true,
            Some("false") => false,
            _ => env_remote_metrics.unwrap_or(true),
        };
        let swa_kv_pages_env = match env_swa_kv_pages
            .map(str::trim)
            .filter(|raw| !raw.is_empty())
        {
            Some(raw) => Some(raw.parse::<usize>().map_err(|_| {
                format!("invalid ZSTEP_SWA_KV_PAGES={raw:?}: must be a non-negative integer")
            })?),
            None => None,
        };
        let args = Self {
            model_dir,
            served_model_name,
            host: values
                .get("--host")
                .cloned()
                .unwrap_or_else(|| "127.0.0.1".to_owned()),
            port,
            api_key,
            admin_port,
            allow_remote_metrics,
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
            enable_prefix_caching,
            apc_gdn_node_tokens,
            kv_events_endpoint,
            kv_events_topic,
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
            prefill_policy,
            dram_tier,
            allreduce_blocks: parse_map_number(&values, "--allreduce-blocks", 8u32)?,
            allreduce_threads: parse_map_number(&values, "--allreduce-threads", 256u32)?,
            speculative,
            swa_kv_pages_env,
        };
        if is_dsv41
            && let Some(explicit) = args.swa_kv_pages_env
            && explicit > 0
        {
            if args.dram_tier.is_enabled() {
                return Err(
                    "ZSTEP_SWA_KV_PAGES > 0 cannot be combined with --dram-kv-blocks > 0"
                        .to_owned(),
                );
            }
            let min_active_swa = 3 * args.max_seqs as usize;
            if explicit <= min_active_swa {
                return Err(format!(
                    "ZSTEP_SWA_KV_PAGES ({explicit}) must exceed 3 * max_seqs ({min_active_swa})"
                ));
            }
            let num_kv = args.resolved_num_kv_pages() as usize;
            if explicit >= num_kv {
                return Err(format!(
                    "ZSTEP_SWA_KV_PAGES ({explicit}) must be strictly less than resolved num_kv_pages ({num_kv})"
                ));
            }
        }
        let _ = args.resolved_apc_geometry()?;
        Ok(args)
    }

    /// Resolve `(pages_per_prefix_node, num_recurrent_checkpoints)` for Automatic Prefix Caching.
    pub fn resolved_apc_geometry(&self) -> Result<(usize, u32), String> {
        if !self.enable_prefix_caching {
            return Ok((0, 0));
        }
        if is_deepseek_v41_dir(&self.model_dir) {
            if self.page_block_tokens != 128 {
                return Err(format!(
                    "--enable-prefix-caching on deepseek_v41 requires --page-block-tokens 128 (got {})",
                    self.page_block_tokens
                ));
            }
            Ok((1, 0))
        } else {
            let gdn_node_tokens = self.apc_gdn_node_tokens;
            if self.page_block_tokens > gdn_node_tokens
                || gdn_node_tokens % self.page_block_tokens != 0
            {
                return Err(format!(
                    "--enable-prefix-caching on hybrid GDN runner requires --page-block-tokens dividing --apc-gdn-node-tokens {gdn_node_tokens} (got {})",
                    self.page_block_tokens
                ));
            }
            Ok(((gdn_node_tokens / self.page_block_tokens) as usize, 255))
        }
    }

    fn decoupled_swa_default_enabled(&self) -> bool {
        is_deepseek_v41_dir(&self.model_dir)
            && !self.dram_tier.is_enabled()
            && self.swa_kv_pages_env != Some(0)
    }

    pub fn resolved_num_kv_pages(&self) -> u32 {
        self.num_kv_pages.unwrap_or_else(|| {
            let base = if is_deepseek_v41_dir(&self.model_dir) {
                if self.decoupled_swa_default_enabled() {
                    10_240.0
                } else if self.speculative.method == SpeculativeMethod::Dspark {
                    2_560.0
                } else {
                    4_096.0
                }
            } else {
                49_152.0
            };
            let scaled = (self.gpu_memory_utilization / 0.95) * base;
            (scaled.round() as u32).max(128)
        })
    }

    /// Resolved decoupled SWA 128-token KV page count for DeepSeek-V4.1 (`0` = disabled / uniform).
    pub fn resolved_swa_kv_pages(&self) -> usize {
        if !is_deepseek_v41_dir(&self.model_dir) {
            return 0;
        }
        let num_kv = self.resolved_num_kv_pages() as usize;
        if let Some(explicit) = self.swa_kv_pages_env {
            return if explicit > 0 && explicit < num_kv {
                explicit
            } else {
                0
            };
        }
        if self.dram_tier.is_enabled() {
            return 0;
        }
        let base_swa = if self.speculative.method == SpeculativeMethod::Dspark {
            1_536.0
        } else {
            2_048.0
        };
        let min_active = 2 * self.max_seqs as usize + 128;
        let scaled = ((self.gpu_memory_utilization / 0.95) * base_swa).round() as usize;
        let swa = scaled.max(min_active);
        if swa < num_kv { swa } else { 0 }
    }

    pub fn resolved_max_pages_per_seq(&self) -> u32 {
        self.max_model_len
            .div_ceil(self.page_block_tokens as usize)
            .max(1) as u32
    }

    pub(super) fn engine_config(&self, cancel_flag: Option<Arc<AtomicBool>>) -> EngineConfig {
        let max_pages_per_seq = self.resolved_max_pages_per_seq();
        let num_kv_pages = self.resolved_num_kv_pages();
        let (_, num_recurrent_checkpoints) = self.resolved_apc_geometry().unwrap_or((0, 0));
        EngineConfig {
            shm_name: unique_shm_name(&self.served_model_name),
            geometry: ShmGeometry {
                world_size: self.world_size,
                max_seqs: self.max_seqs,
                max_step_tokens: self.max_step_tokens,
                max_decoder_rows: self.max_step_tokens,
                max_pages_per_seq,
                page_block_tokens: self.page_block_tokens,
                spec_verify_width: if self.speculative.enabled() {
                    self.speculative.num_speculative_tokens as u32 + 1
                } else {
                    6
                },
            },
            max_prefill_chunk: self.max_prefill_chunk,
            steps: 1,
            token_counts: vec![1],
            speculative: self.speculative.clone(),
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
                num_recurrent_checkpoints,
                dram_kv_blocks: self.dram_tier.num_dram_blocks,
                swa_kv_pages: self.resolved_swa_kv_pages(),
                max_model_len: Some(self.max_model_len),
                cancel_flag,
            },
            ready_timeout: self.ready_timeout,
            step_timeout: self.step_timeout,
            heartbeat_timeout: DEFAULT_HEARTBEAT_TIMEOUT,
        }
    }
}

pub(super) fn is_deepseek_v41_dir(model_dir: &Path) -> bool {
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

pub(super) fn unique_shm_name(label: &str) -> String {
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
