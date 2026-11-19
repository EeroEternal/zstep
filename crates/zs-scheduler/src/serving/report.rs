// Copyright (c) 2026 zstep Contributors
// SPDX-License-Identifier: MIT

use std::collections::BTreeMap;
use std::process::Command;

use crate::coordinator::{EngineError, Result};

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
    pub(super) fn to_json(&self) -> String {
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

    pub(super) fn to_json(&self) -> String {
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

    pub(super) fn to_json(&self) -> String {
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
    pub env_overrides: BTreeMap<String, String>,
}

pub(super) fn is_sensitive_env_key(key: &str) -> bool {
    key.split('_').any(|seg| {
        matches!(
            seg.to_ascii_uppercase().as_str(),
            "KEY"
                | "KEYS"
                | "SECRET"
                | "SECRETS"
                | "TOKEN"
                | "PASSWORD"
                | "PASSWD"
                | "CREDENTIAL"
                | "CREDENTIALS"
                | "AUTH"
                | "BEARER"
        )
    })
}

pub(super) fn collect_zstep_env_overrides<I, K, V>(vars: I) -> BTreeMap<String, String>
where
    I: IntoIterator<Item = (K, V)>,
    K: AsRef<str>,
    V: AsRef<str>,
{
    let mut overrides = BTreeMap::new();
    for (k, v) in vars {
        let key = k.as_ref();
        if !key.starts_with("ZSTEP_") {
            continue;
        }
        let val = if is_sensitive_env_key(key) {
            "[REDACTED]".to_owned()
        } else {
            v.as_ref().to_owned()
        };
        overrides.insert(key.to_owned(), val);
    }
    overrides
}

impl BenchmarkProvenance {
    pub fn capture(
        seed: u64,
        repeats: usize,
        warmup_prompts: usize,
        max_step_tokens: u32,
        max_prefill_chunk: usize,
    ) -> Result<Self> {
        let git_commit = zs_wire::resolve_git_sha(true).map_err(EngineError::Protocol)?;

        let gpu_output = Command::new("nvidia-smi")
            .args([
                "--query-gpu=driver_version,clocks.current.graphics,clocks.current.memory",
                "--format=csv,noheader,nounits",
            ])
            .output()?;
        if !gpu_output.status.success() {
            return Err(EngineError::Protocol(
                "nvidia-smi failed while capturing provenance".to_owned(),
            ));
        }
        let gpu_csv = String::from_utf8(gpu_output.stdout).map_err(|error| {
            EngineError::Protocol(format!("invalid nvidia-smi output: {error}"))
        })?;
        let mut driver_versions = Vec::new();
        let mut gpu_graphics_clocks_mhz = Vec::new();
        let mut gpu_mem_clocks_mhz = Vec::new();
        for line in gpu_csv.lines().filter(|line| !line.trim().is_empty()) {
            let fields: Vec<&str> = line.split(',').map(str::trim).collect();
            if fields.len() != 3 {
                return Err(EngineError::Protocol(format!(
                    "unexpected nvidia-smi provenance row: {line}"
                )));
            }
            driver_versions.push(fields[0].to_owned());
            gpu_graphics_clocks_mhz.push(fields[1].parse::<u32>().map_err(|error| {
                EngineError::Protocol(format!("invalid graphics clock in `{line}`: {error}"))
            })?);
            gpu_mem_clocks_mhz.push(fields[2].parse::<u32>().map_err(|error| {
                EngineError::Protocol(format!("invalid memory clock in `{line}`: {error}"))
            })?);
        }
        let nvidia_driver_version = driver_versions.first().cloned().ok_or_else(|| {
            EngineError::Protocol("nvidia-smi returned no GPU provenance rows".to_owned())
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
            env_overrides: collect_zstep_env_overrides(std::env::vars()),
        })
    }

    pub(super) fn to_json(&self) -> String {
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
        let env_json = if self.env_overrides.is_empty() {
            "{}".to_owned()
        } else {
            let entries = self
                .env_overrides
                .iter()
                .map(|(k, v)| format!("      {}: {}", json_string(k), json_string(v)))
                .collect::<Vec<_>>()
                .join(",\n");
            format!("{{\n{entries}\n    }}")
        };
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
                "    \"max_prefill_chunk\": {},\n",
                "    \"env_overrides\": {}\n",
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
            env_json,
        )
    }
}

/// Complete multi-tier benchmark report produced by [`ServingEngine`].
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
                "  \"engine\": \"zstep\",\n",
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

pub(super) fn json_string(value: &str) -> String {
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

pub(super) fn mean_us_as_ms(samples: &[u64]) -> f64 {
    if samples.is_empty() {
        return 0.0;
    }
    let sum: u128 = samples.iter().map(|&v| u128::from(v)).sum();
    (sum as f64 / samples.len() as f64) / 1000.0
}

pub(super) fn percentile_us_as_ms(samples: &[u64], pct: usize) -> f64 {
    if samples.is_empty() {
        return 0.0;
    }
    let mut sorted = samples.to_vec();
    sorted.sort_unstable();
    let idx = (sorted.len() - 1) * pct / 100;
    sorted[idx] as f64 / 1000.0
}
