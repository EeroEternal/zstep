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

use std::collections::hash_map::RandomState;
use std::hash::{BuildHasher, Hasher};
use std::path::PathBuf;
use std::process::ExitCode;
use std::time::{Duration, SystemTime, UNIX_EPOCH};

use serde::{Deserialize, Serialize};
use zs_scheduler::{
    BENCHMARK_RANDOM_SEED, BenchmarkProvenance, BenchmarkTierSpec, DEFAULT_HEARTBEAT_TIMEOUT,
    EngineConfig, GenerationScheduleStats, ServingBenchmarkReport, ServingEngine, StepTop2Logits,
    TierAggregateStats, WorkerCoordinator, WorkerLaunchConfig,
};
use zs_wire::{MAX_DECODE_GRAPH_SEQS, ShmGeometry};

#[derive(Debug, Deserialize)]
struct GreedyReference {
    vllm_version: String,
    cases: Vec<GreedyReferenceCase>,
}

#[derive(Debug, Clone, Deserialize, Serialize)]
struct VllmTop2Step {
    top1_token: i32,
    top1_logprob: f64,
    top2_token: i32,
    top2_logprob: f64,
    top1_minus_top2_margin: f64,
}

#[derive(Debug, Clone, Deserialize, Serialize)]
struct GreedyReferenceCase {
    name: String,
    prompt_token_ids: Vec<i32>,
    max_tokens: usize,
    expected_greedy_token_ids: Vec<i32>,
    #[serde(default)]
    top2_steps: Vec<VllmTop2Step>,
    #[serde(default)]
    require_exact: bool,
    #[serde(default)]
    max_mismatch_margin: Option<f32>,
}

#[derive(Debug, Clone, Deserialize, Serialize)]
struct ZstepTop2StepReport {
    top1_token: i32,
    top1_logit: f32,
    top2_token: i32,
    top2_logit: f32,
    top1_minus_top2_margin: f32,
}

impl From<StepTop2Logits> for ZstepTop2StepReport {
    fn from(step: StepTop2Logits) -> Self {
        Self {
            top1_token: step.top1_token,
            top1_logit: step.top1_logit,
            top2_token: step.top2_token,
            top2_logit: step.top2_logit,
            top1_minus_top2_margin: step.top1_minus_top2_margin,
        }
    }
}

#[derive(Debug, Clone, Deserialize, Serialize)]
struct FirstMismatchDiagnostics {
    token_index: usize,
    prefix_identical_at_step: bool,
    expected_token: i32,
    generated_token: i32,
    vllm_top2: Option<VllmTop2Step>,
    zstep_top2: ZstepTop2StepReport,
}

#[derive(Debug, Clone, Deserialize, Serialize)]
struct ModePairMismatchDiagnostics {
    token_index: usize,
    prefix_identical_at_step: bool,
    left_mode: String,
    right_mode: String,
    left_top2: ZstepTop2StepReport,
    right_top2: ZstepTop2StepReport,
}

#[derive(Debug, Clone, Deserialize, Serialize)]
struct GreedyMatchReport {
    exact_prefix_match_tokens: usize,
    total_tokens: usize,
    exact_token_matches: usize,
    exact_token_match_rate: f64,
    all_matched: bool,
    first_mismatch_token_index: Option<usize>,
    mean_top1_minus_top2_margin: f64,
    min_top1_minus_top2_margin: f64,
    first_mismatch_diagnostics: Option<FirstMismatchDiagnostics>,
}

#[derive(Debug, Deserialize, Serialize)]
struct GreedyCaseParityReport {
    name: String,
    prompt_token_ids: Vec<i32>,
    expected_greedy_token_ids: Vec<i32>,
    exact_prefix_match_tokens: usize,
    total_tokens: usize,
    exact_token_match_rate: f64,
    all_matched: bool,
    first_mismatch_token_index: Option<usize>,
    single_generated_token_ids: Vec<i32>,
    batched_generated_token_ids: Vec<i32>,
    chunked_generated_token_ids: Vec<i32>,
    single_match: GreedyMatchReport,
    batched_match: GreedyMatchReport,
    chunked_match: GreedyMatchReport,
    single_vs_batched_first_mismatch: Option<ModePairMismatchDiagnostics>,
    batched_vs_chunked_first_mismatch: Option<ModePairMismatchDiagnostics>,
}

#[derive(Debug, Deserialize, Serialize)]
struct ScheduleReport {
    prefill_steps: u64,
    decode_steps: u64,
    chunked_prefill_steps: u64,
    decode_steps_with_partial_prefill: u64,
    max_active_sequences: usize,
}

impl From<&GenerationScheduleStats> for ScheduleReport {
    fn from(stats: &GenerationScheduleStats) -> Self {
        Self {
            prefill_steps: stats.prefill_steps,
            decode_steps: stats.decode_steps,
            chunked_prefill_steps: stats.chunked_prefill_steps,
            decode_steps_with_partial_prefill: stats.decode_steps_with_partial_prefill,
            max_active_sequences: stats.max_active_sequences,
        }
    }
}

#[derive(Debug, Deserialize, Serialize)]
struct GreedyParityReport {
    git_sha: String,
    vllm_version: String,
    max_prefill_chunk: usize,
    all_matched: bool,
    batched_vs_single_exact_match: bool,
    chunked_vs_unchunked_exact_match: bool,
    chunked_schedule: ScheduleReport,
    unchunked_schedule: ScheduleReport,
    triton_recompilations: Option<u32>,
    watched_kernels: Option<u32>,
    cases: Vec<GreedyCaseParityReport>,
}

const USAGE: &str = "zstep master\n\
Usage: zstep [run] [OPTIONS]\n\
  --world-size N             GPU workers, 1..=8 (default: 8)\n\
  --steps N                  Overlapped micro-steps when --model-dir is omitted (default: 200)\n\
  --token-counts LIST        Runtime rows, comma separated (default: 6,48,192)\n\
  --python PATH              Python executable (default: /opt/venv/bin/python)\n\
  --worker-root PATH         Python package root (default: /workspace/worker)\n\
  --shm-name NAME            POSIX SHM object name (default: unique per run)\n\
  --metrics-output PATH      Also write the JSON metrics/report to this file\n\
  --model-dir PATH           Real-weight HuggingFace model directory (e.g. Qwen3.5-35B-A3B)\n\
  --num-kv-pages N           Physical 16-token KV pages per GPU (default: 49152)\n\
  --page-block-tokens N      Tokens per KV page block (default: 64 synthetic, 16 with --model-dir)\n\
  --prompt-ids LIST          Comma-separated prompt token IDs for greedy generation parity check\n\
  --max-new-tokens N         Generated tokens per prompt for --prompt-ids (default: 16)\n\
  --verify-greedy-json PATH  Multi-case vLLM greedy reference JSON\n\
  --verify-output-json PATH  Write measured single/batched/chunked parity report\n\
  --require-exact-greedy BOOL Fail with non-zero exit if any token differs (default: false)\n\
  --require-exact-cases LIST Comma-separated case names that must match 100% (optional)\n\
  --max-greedy-mismatch-margin FLOAT Fail if any mismatch Top1-Top2 margin exceeds FLOAT (optional)\n\
  --bench-tiers SPEC         Semicolon-separated tiers `label:conc:prompts:in_len:out_len`\n\
  --warmup-prompts N         Warm up 512->64 prompts at concurrency 16 (default: 16)\n\
  --repeats N                Interleaved repeat count across all tiers (default: 3)\n\
  --max-prefill-chunk N      Maximum prompt tokens per sequence per prefill step (default: 2048)\n\
  --max-mixed-step-tokens N  Piggyback decode+prefill token cap (default: 3072 on deepseek_v41, 0=off)\n\
  --max-seqs N               Sequence capacity (default: 8 synthetic, 128 with --model-dir)\n\
  --max-step-tokens N        Token capacity (default: 192 synthetic, 4096 with --model-dir)\n\
  --max-decoder-rows N       Decoder row capacity (default: 192 synthetic, 4096 with --model-dir)\n\
  --max-pages-per-seq N      Page-table capacity (default: 16 synthetic, 256 with --model-dir)\n\
  --hidden-size N            RMSNorm/GEMM hidden size (default: 5120)\n\
  --output-features N        GEMM output columns (default: 2048)\n\
  --router-experts N         Router expert count (default: 384)\n\
  --router-top-k N           Router selected experts (default: 6)\n\
  --allreduce-blocks N       SHM AllReduce CUDA blocks (default: 16)\n\
  --allreduce-threads N      SHM AllReduce threads/block (default: 256)\n\
  --ready-timeout-secs N     Initialization timeout (default: 600)\n\
  --step-timeout-secs N      Per-step timeout (default: 120)\n\
  --heartbeat-timeout-ms N   Per-worker heartbeat stall timeout (default: 5000)\n";

struct Arguments {
    world_size: u32,
    steps: u64,
    token_counts: Vec<u32>,
    python: PathBuf,
    worker_root: PathBuf,
    shm_name: Option<String>,
    metrics_output: Option<PathBuf>,
    model_dir: Option<PathBuf>,
    num_kv_pages: Option<u32>,
    page_block_tokens: Option<u32>,
    prompt_ids: Option<Vec<i32>>,
    max_new_tokens: usize,
    verify_greedy_json: Option<PathBuf>,
    verify_output_json: Option<PathBuf>,
    require_exact_greedy: bool,
    require_exact_cases: Vec<String>,
    max_greedy_mismatch_margin: Option<f32>,
    bench_tiers: Vec<BenchmarkTierSpec>,
    warmup_prompts: usize,
    repeats: usize,
    max_prefill_chunk: usize,
    max_mixed_step_tokens: Option<usize>,
    max_seqs: Option<u32>,
    max_step_tokens: Option<u32>,
    max_decoder_rows: Option<u32>,
    max_pages_per_seq: Option<u32>,
    hidden_size: u32,
    output_features: u32,
    router_experts: u32,
    router_top_k: u32,
    allreduce_blocks: u32,
    allreduce_threads: u32,
    ready_timeout_secs: u64,
    step_timeout_secs: u64,
    heartbeat_timeout_ms: u64,
}

impl Default for Arguments {
    fn default() -> Self {
        Self {
            world_size: 8,
            steps: 200,
            token_counts: vec![6, 48, 192],
            python: PathBuf::from("/opt/venv/bin/python"),
            worker_root: PathBuf::from("/workspace/worker"),
            shm_name: None,
            metrics_output: None,
            model_dir: None,
            num_kv_pages: None,
            page_block_tokens: None,
            prompt_ids: None,
            max_new_tokens: 16,
            verify_greedy_json: None,
            verify_output_json: None,
            require_exact_greedy: false,
            require_exact_cases: Vec::new(),
            max_greedy_mismatch_margin: None,
            bench_tiers: Vec::new(),
            warmup_prompts: 16,
            repeats: 3,
            max_prefill_chunk: 2048,
            max_mixed_step_tokens: None,
            max_seqs: None,
            max_step_tokens: None,
            max_decoder_rows: None,
            max_pages_per_seq: None,
            hidden_size: 5120,
            output_features: 2048,
            router_experts: 384,
            router_top_k: 6,
            allreduce_blocks: 16,
            allreduce_threads: 256,
            ready_timeout_secs: 600,
            step_timeout_secs: 120,
            heartbeat_timeout_ms: DEFAULT_HEARTBEAT_TIMEOUT.as_millis() as u64,
        }
    }
}

fn main() -> ExitCode {
    let raw_args: Vec<String> = std::env::args().skip(1).collect();
    if raw_args.is_empty() || raw_args.iter().any(|arg| arg == "--help" || arg == "-h") {
        print!("{USAGE}");
        return ExitCode::SUCCESS;
    }
    let args = match parse_arguments(&raw_args) {
        Ok(args) => args,
        Err(error) => {
            eprintln!("argument error: {error}");
            return ExitCode::FAILURE;
        }
    };
    match run_engine(args) {
        Ok(()) => ExitCode::SUCCESS,
        Err(error) => {
            eprintln!("zstep failed: {error}");
            ExitCode::FAILURE
        }
    }
}

fn greedy_match(
    expected: &[i32],
    vllm_top2: &[VllmTop2Step],
    generated: &[i32],
    zstep_top2: &[StepTop2Logits],
) -> GreedyMatchReport {
    let first_mismatch = expected
        .iter()
        .zip(generated)
        .position(|(expected_token, generated_token)| expected_token != generated_token);
    let exact_prefix_match_tokens = first_mismatch.unwrap_or(expected.len());
    let exact_token_matches = expected
        .iter()
        .zip(generated)
        .filter(|(expected_token, generated_token)| expected_token == generated_token)
        .count();
    let (mean_margin, min_margin) = if zstep_top2.is_empty() {
        (0.0, 0.0)
    } else {
        let sum: f64 = zstep_top2
            .iter()
            .map(|step| f64::from(step.top1_minus_top2_margin))
            .sum();
        let min = zstep_top2
            .iter()
            .map(|step| f64::from(step.top1_minus_top2_margin))
            .fold(f64::INFINITY, f64::min);
        (sum / zstep_top2.len() as f64, min)
    };
    let first_mismatch_diagnostics = first_mismatch.and_then(|idx| {
        zstep_top2
            .get(idx)
            .copied()
            .map(|z_step| FirstMismatchDiagnostics {
                token_index: idx,
                prefix_identical_at_step: true,
                expected_token: expected[idx],
                generated_token: generated[idx],
                vllm_top2: vllm_top2.get(idx).cloned(),
                zstep_top2: ZstepTop2StepReport::from(z_step),
            })
    });
    GreedyMatchReport {
        exact_prefix_match_tokens,
        total_tokens: expected.len(),
        exact_token_matches,
        exact_token_match_rate: exact_token_matches as f64 / expected.len() as f64,
        all_matched: exact_prefix_match_tokens == expected.len(),
        first_mismatch_token_index: first_mismatch,
        mean_top1_minus_top2_margin: mean_margin,
        min_top1_minus_top2_margin: min_margin,
        first_mismatch_diagnostics,
    }
}

fn mode_pair_mismatch(
    left_mode: &str,
    left_tokens: &[i32],
    left_top2: &[StepTop2Logits],
    right_mode: &str,
    right_tokens: &[i32],
    right_top2: &[StepTop2Logits],
) -> Option<ModePairMismatchDiagnostics> {
    let idx = left_tokens
        .iter()
        .zip(right_tokens)
        .position(|(a, b)| a != b)?;
    let left_step = *left_top2.get(idx)?;
    let right_step = *right_top2.get(idx)?;
    Some(ModePairMismatchDiagnostics {
        token_index: idx,
        prefix_identical_at_step: true,
        left_mode: left_mode.to_owned(),
        right_mode: right_mode.to_owned(),
        left_top2: ZstepTop2StepReport::from(left_step),
        right_top2: ZstepTop2StepReport::from(right_step),
    })
}

fn current_git_sha() -> Result<String, Box<dyn std::error::Error>> {
    zs_wire::resolve_git_sha(false).map_err(Into::into)
}

fn run_greedy_verification(
    engine: &mut ServingEngine,
    reference: &GreedyReference,
    max_prefill_chunk: usize,
) -> Result<GreedyParityReport, Box<dyn std::error::Error>> {
    if reference.cases.is_empty() {
        return Err("greedy reference contains no cases".into());
    }
    for case in &reference.cases {
        if case.prompt_token_ids.is_empty() || case.max_tokens == 0 {
            return Err(format!(
                "case {} has an empty prompt or zero output length",
                case.name
            )
            .into());
        }
        if case.expected_greedy_token_ids.len() != case.max_tokens {
            return Err(format!(
                "case {} expected {} tokens but max_tokens is {}",
                case.name,
                case.expected_greedy_token_ids.len(),
                case.max_tokens
            )
            .into());
        }
        if !case.top2_steps.is_empty() && case.top2_steps.len() != case.max_tokens {
            return Err(format!(
                "case {} has {} top2_steps entries, expected {}",
                case.name,
                case.top2_steps.len(),
                case.max_tokens
            )
            .into());
        }
    }

    let prompts: Vec<Vec<i32>> = reference
        .cases
        .iter()
        .map(|case| case.prompt_token_ids.clone())
        .collect();
    let output_lengths: Vec<usize> = reference.cases.iter().map(|case| case.max_tokens).collect();

    engine.set_max_prefill_chunk(usize::MAX)?;
    let mut single_outputs = Vec::with_capacity(reference.cases.len());
    let mut single_top2 = Vec::with_capacity(reference.cases.len());
    for case in &reference.cases {
        let (mut outputs, mut top2) = engine.generate_greedy_with_top2(
            std::slice::from_ref(&case.prompt_token_ids),
            std::slice::from_ref(&case.max_tokens),
        )?;
        single_outputs.push(outputs.remove(0));
        single_top2.push(top2.remove(0));
    }
    let (batched_outputs, batched_top2) =
        engine.generate_greedy_with_top2(&prompts, &output_lengths)?;
    let unchunked_schedule = ScheduleReport::from(engine.last_generation_schedule());

    engine.set_max_prefill_chunk(max_prefill_chunk)?;
    let (chunked_outputs, chunked_top2) =
        engine.generate_greedy_with_top2(&prompts, &output_lengths)?;
    let chunked_schedule = ScheduleReport::from(engine.last_generation_schedule());

    let batched_vs_single_exact_match = batched_outputs == single_outputs;
    let chunked_vs_unchunked_exact_match = chunked_outputs == batched_outputs;
    let mut case_reports = Vec::with_capacity(reference.cases.len());
    for (index, case) in reference.cases.iter().enumerate() {
        let single = &single_outputs[index];
        let batched = &batched_outputs[index];
        let chunked = &chunked_outputs[index];
        let single_steps = &single_top2[index];
        let batched_steps = &batched_top2[index];
        let chunked_steps = &chunked_top2[index];
        if single.len() != case.max_tokens
            || batched.len() != case.max_tokens
            || chunked.len() != case.max_tokens
            || single_steps.len() != case.max_tokens
            || batched_steps.len() != case.max_tokens
            || chunked_steps.len() != case.max_tokens
        {
            return Err(format!("case {} returned an unexpected token count", case.name).into());
        }
        let single_match = greedy_match(
            &case.expected_greedy_token_ids,
            &case.top2_steps,
            single,
            single_steps,
        );
        let batched_match = greedy_match(
            &case.expected_greedy_token_ids,
            &case.top2_steps,
            batched,
            batched_steps,
        );
        let chunked_match = greedy_match(
            &case.expected_greedy_token_ids,
            &case.top2_steps,
            chunked,
            chunked_steps,
        );
        let single_vs_batched_first_mismatch = mode_pair_mismatch(
            "single",
            single,
            single_steps,
            "batched",
            batched,
            batched_steps,
        );
        let batched_vs_chunked_first_mismatch = mode_pair_mismatch(
            "batched",
            batched,
            batched_steps,
            "chunked",
            chunked,
            chunked_steps,
        );
        let case_all_matched = single_match.all_matched
            && batched_match.all_matched
            && chunked_match.all_matched
            && single_vs_batched_first_mismatch.is_none()
            && batched_vs_chunked_first_mismatch.is_none();
        let exact_prefix_match_tokens = single_match
            .exact_prefix_match_tokens
            .min(batched_match.exact_prefix_match_tokens)
            .min(chunked_match.exact_prefix_match_tokens);
        let exact_token_match_rate = single_match
            .exact_token_match_rate
            .min(batched_match.exact_token_match_rate)
            .min(chunked_match.exact_token_match_rate);
        let first_mismatch_token_index = [
            single_match.first_mismatch_token_index,
            batched_match.first_mismatch_token_index,
            chunked_match.first_mismatch_token_index,
        ]
        .into_iter()
        .flatten()
        .min();
        case_reports.push(GreedyCaseParityReport {
            name: case.name.clone(),
            prompt_token_ids: case.prompt_token_ids.clone(),
            expected_greedy_token_ids: case.expected_greedy_token_ids.clone(),
            exact_prefix_match_tokens,
            total_tokens: single_match.total_tokens,
            exact_token_match_rate,
            all_matched: case_all_matched,
            first_mismatch_token_index,
            single_generated_token_ids: single.clone(),
            batched_generated_token_ids: batched.clone(),
            chunked_generated_token_ids: chunked.clone(),
            single_match,
            batched_match,
            chunked_match,
            single_vs_batched_first_mismatch,
            batched_vs_chunked_first_mismatch,
        });
    }
    let all_matched = batched_vs_single_exact_match
        && chunked_vs_unchunked_exact_match
        && case_reports.iter().all(|case| case.all_matched);
    Ok(GreedyParityReport {
        git_sha: current_git_sha()?,
        vllm_version: reference.vllm_version.clone(),
        max_prefill_chunk,
        all_matched,
        batched_vs_single_exact_match,
        chunked_vs_unchunked_exact_match,
        chunked_schedule,
        unchunked_schedule,
        triton_recompilations: None,
        watched_kernels: None,
        cases: case_reports,
    })
}

#[derive(Debug, Deserialize)]
struct ModelConfigProbe {
    #[serde(default)]
    model_type: String,
}

fn is_deepseek_v41_model_dir(model_dir: Option<&PathBuf>) -> bool {
    let Some(dir) = model_dir else {
        return false;
    };
    let Ok(bytes) = std::fs::read(dir.join("config.json")) else {
        return false;
    };
    serde_json::from_slice::<ModelConfigProbe>(&bytes)
        .is_ok_and(|probe| probe.model_type == "deepseek_v41")
}

#[derive(Debug, Default)]
struct GreedyGatePolicy<'a> {
    require_exact_all: bool,
    require_exact_cases: &'a [String],
    max_mismatch_margin: Option<f32>,
    reference_cases: &'a [GreedyReferenceCase],
}

fn validate_greedy_report_gate(
    report: &GreedyParityReport,
    policy: &GreedyGatePolicy<'_>,
) -> Result<(), String> {
    if let Some(recomp) = report.triton_recompilations
        && recomp > 0
    {
        return Err(format!(
            "unexpected runtime Triton JIT recompilations detected: {recomp}"
        ));
    }
    if policy.require_exact_all && !report.all_matched {
        return Err(
            "greedy parity report had token mismatches (--require-exact-greedy=true)".to_owned(),
        );
    }
    for required_name in policy.require_exact_cases {
        let Some(case) = report.cases.iter().find(|c| &c.name == required_name) else {
            return Err(format!(
                "--require-exact-cases specified unknown case `{required_name}`"
            ));
        };
        if !case.all_matched {
            return Err(format!(
                "case `{required_name}` had token mismatches (--require-exact-cases)"
            ));
        }
    }
    for case in &report.cases {
        let ref_case = policy
            .reference_cases
            .iter()
            .find(|rc| rc.name == case.name);
        if ref_case.is_some_and(|rc| rc.require_exact) && !case.all_matched {
            return Err(format!(
                "case `{}` had token mismatches (reference require_exact=true)",
                case.name
            ));
        }
        let margin_limit = policy
            .max_mismatch_margin
            .or_else(|| ref_case.and_then(|rc| rc.max_mismatch_margin));
        let Some(max_margin) = margin_limit else {
            continue;
        };
        if let Some(diag) = &case.single_match.first_mismatch_diagnostics
            && diag.zstep_top2.top1_minus_top2_margin > max_margin
        {
            return Err(format!(
                "case `{}` diverged from reference at token {} with margin {:.4} > {:.4}",
                case.name, diag.token_index, diag.zstep_top2.top1_minus_top2_margin, max_margin
            ));
        }
        for pair in [
            case.single_vs_batched_first_mismatch.as_ref(),
            case.batched_vs_chunked_first_mismatch.as_ref(),
        ]
        .into_iter()
        .flatten()
        {
            let max_pair_margin = pair
                .left_top2
                .top1_minus_top2_margin
                .max(pair.right_top2.top1_minus_top2_margin);
            if max_pair_margin > max_margin {
                return Err(format!(
                    "case `{}` diverged between {} and {} at token {} with margin {:.4} > {:.4}",
                    case.name,
                    pair.left_mode,
                    pair.right_mode,
                    pair.token_index,
                    max_pair_margin,
                    max_margin
                ));
            }
        }
    }
    Ok(())
}

fn run_engine(args: Arguments) -> Result<(), Box<dyn std::error::Error>> {
    let has_model = args.model_dir.is_some();
    let is_dsv41 = is_deepseek_v41_model_dir(args.model_dir.as_ref());
    let max_seqs = args.max_seqs.unwrap_or(if has_model { 128 } else { 8 });
    if max_seqs == 0 || (has_model && max_seqs > MAX_DECODE_GRAPH_SEQS) {
        return Err(format!(
            "--max-seqs {max_seqs} is outside supported decode CUDA Graph range 1..={MAX_DECODE_GRAPH_SEQS}"
        )
        .into());
    }
    let max_step_tokens = args
        .max_step_tokens
        .unwrap_or(if has_model { 4096 } else { 192 });
    let max_decoder_rows =
        args.max_decoder_rows
            .unwrap_or(if has_model { max_step_tokens } else { 192 });
    let max_pages_per_seq = args
        .max_pages_per_seq
        .unwrap_or(if has_model { 256 } else { 16 });
    let page_block_tokens = args.page_block_tokens.unwrap_or(if is_dsv41 {
        128
    } else if has_model {
        16
    } else {
        64
    });
    let num_kv_pages = args
        .num_kv_pages
        .unwrap_or(if is_dsv41 { 2_560 } else { 49_152 });

    let shm_name = args.shm_name.unwrap_or_else(unique_shm_name);
    let config = EngineConfig {
        shm_name,
        geometry: ShmGeometry {
            world_size: args.world_size,
            max_seqs,
            max_step_tokens,
            max_decoder_rows,
            max_pages_per_seq,
            page_block_tokens,
            spec_verify_width: 6,
        },
        max_prefill_chunk: args.max_prefill_chunk,
        steps: args.steps,
        token_counts: args.token_counts,
        launch: WorkerLaunchConfig {
            python_executable: args.python,
            worker_root: args.worker_root,
            hidden_size: args.hidden_size,
            output_features: args.output_features,
            router_experts: args.router_experts,
            router_top_k: args.router_top_k,
            allreduce_blocks: args.allreduce_blocks,
            allreduce_threads: args.allreduce_threads,
            model_dir: args.model_dir.clone(),
            num_kv_pages,
        },
        ready_timeout: Duration::from_secs(args.ready_timeout_secs),
        step_timeout: Duration::from_secs(args.step_timeout_secs),
        heartbeat_timeout: Duration::from_millis(args.heartbeat_timeout_ms),
    };

    if has_model {
        let model_label = args
            .model_dir
            .as_ref()
            .and_then(|p| p.file_name())
            .map_or_else(
                || "Qwen3.5-35B-A3B".to_owned(),
                |n| n.to_string_lossy().into_owned(),
            );
        let mixed_step_tokens = match args.max_mixed_step_tokens {
            Some(0) => None,
            Some(explicit) => Some(explicit),
            None if is_dsv41 => Some(3072),
            None => None,
        };
        let mut engine = ServingEngine::launch(config)?;
        engine.set_max_mixed_step_tokens(mixed_step_tokens);
        engine.set_device_token_overlap(
            !is_dsv41 && std::env::var("ZSTEP_DECODE_DEVICE_TOKENS").map_or(true, |v| v != "0"),
        );

        if let Some(reference_path) = args.verify_greedy_json {
            let output_path = args
                .verify_output_json
                .ok_or("--verify-output-json is required with --verify-greedy-json")?;
            let reference_bytes = std::fs::read(reference_path)?;
            let reference: GreedyReference = serde_json::from_slice(&reference_bytes)?;
            let mut report =
                run_greedy_verification(&mut engine, &reference, args.max_prefill_chunk)?;
            let (recompilations, watched) = engine.shutdown_with_stats()?;
            report.triton_recompilations = Some(recompilations);
            report.watched_kernels = Some(watched);
            let mut encoded = serde_json::to_string_pretty(&report)?;
            encoded.push('\n');
            std::fs::write(&output_path, encoded)?;
            println!("VERIFY_GREEDY_REPORT={}", output_path.display());
            validate_greedy_report_gate(
                &report,
                &GreedyGatePolicy {
                    require_exact_all: args.require_exact_greedy,
                    require_exact_cases: &args.require_exact_cases,
                    max_mismatch_margin: args.max_greedy_mismatch_margin,
                    reference_cases: &reference.cases,
                },
            )?;
            return Ok(());
        }

        if let Some(prompt) = args.prompt_ids {
            let outs =
                engine.generate_greedy(std::slice::from_ref(&prompt), args.max_new_tokens)?;
            println!("PROMPT_IDS={prompt:?}");
            println!("GENERATED_IDS={:?}", outs[0]);
        }

        if !args.bench_tiers.is_empty() {
            let provenance = BenchmarkProvenance::capture(
                BENCHMARK_RANDOM_SEED,
                args.repeats,
                args.warmup_prompts,
                max_step_tokens,
                args.max_prefill_chunk,
            )?;
            if args.warmup_prompts > 0 {
                let warmup = BenchmarkTierSpec {
                    label: "warmup".to_owned(),
                    concurrency: args.warmup_prompts.min(16),
                    num_prompts: args.warmup_prompts,
                    input_len: 512,
                    output_len: 64,
                };
                let warmup_result = engine.run_benchmark_tier_with_seed(
                    &warmup,
                    BENCHMARK_RANDOM_SEED.wrapping_add(900_000),
                )?;
                eprintln!(
                    "[warmup] prompts={} -> out_tok/s={:.2}, median_ttft={:.2}ms",
                    warmup_result.num_prompts,
                    warmup_result.output_throughput_tok_s,
                    warmup_result.median_ttft_ms,
                );
            }

            let mut tier_results = Vec::with_capacity(args.bench_tiers.len() * args.repeats);
            let mut results_by_tier =
                vec![Vec::with_capacity(args.repeats); args.bench_tiers.len()];
            for repeat in 0..args.repeats {
                for (tier_index, spec) in args.bench_tiers.iter().enumerate() {
                    let tier_seed = BENCHMARK_RANDOM_SEED
                        .wrapping_add((repeat as u64) * 1_000 + (tier_index as u64));
                    let mut res = engine.run_benchmark_tier_with_seed(spec, tier_seed)?;
                    res.repeat = repeat;
                    eprintln!(
                        "[{} repeat={}] conc={} prompts={} in={} out={} -> out_tok/s={:.2}, total_tok/s={:.2}, median_ttft={:.2}ms, median_submit_ttft={:.2}ms, mean_tpot={:.2}ms, p99_tpot={:.2}ms, p99_itl={:.2}ms (steps={}, overlapped={})",
                        res.label,
                        res.repeat,
                        res.concurrency,
                        res.num_prompts,
                        res.input_len,
                        res.output_len,
                        res.output_throughput_tok_s,
                        res.total_throughput_tok_s,
                        res.median_ttft_ms,
                        res.median_submit_ttft_ms,
                        res.mean_tpot_ms,
                        res.p99_tpot_ms,
                        res.p99_itl_ms,
                        res.steps,
                        res.overlapped_decode_steps,
                    );
                    results_by_tier[tier_index].push(res.clone());
                    tier_results.push(res);
                }
            }
            let tier_aggregates = args
                .bench_tiers
                .iter()
                .zip(&results_by_tier)
                .map(|(spec, results)| {
                    TierAggregateStats::from_results(spec.label.clone(), results)
                        .expect("repeats is validated as positive")
                })
                .collect();
            let report = ServingBenchmarkReport {
                model_name: model_label,
                world_size: args.world_size,
                worker_numa_nodes: engine.worker_numa_nodes().to_vec(),
                init_wall_ms: engine.init_wall_ms(),
                tiers: tier_results,
                tier_aggregates,
                provenance,
            };
            let encoded = report.to_json_pretty();
            if let Some(path) = args.metrics_output {
                std::fs::write(path, &encoded)?;
            }
            print!("{encoded}");
        }

        engine.shutdown()?;
        return Ok(());
    }

    let metrics = WorkerCoordinator::launch(config)?.run()?;
    let encoded = metrics.to_json_pretty();
    if let Some(path) = args.metrics_output {
        std::fs::write(path, &encoded)?;
    }
    print!("{encoded}");
    Ok(())
}

fn parse_arguments(raw: &[String]) -> Result<Arguments, String> {
    let mut args = Arguments::default();
    let mut index = 0;
    while index < raw.len() {
        let option = raw[index].as_str();
        index += 1;
        // Accept optional `run` subcommand or legacy `--mode-b` flag for script compatibility.
        if option == "--mode-b" || (index == 1 && option == "run") {
            continue;
        }
        if !option.starts_with("--") {
            return Err(format!("unknown option {option}"));
        }
        let value = raw
            .get(index)
            .ok_or_else(|| format!("{option} requires a value"))?;
        index += 1;
        match option {
            "--world-size" => args.world_size = parse_number(option, value)?,
            "--steps" => args.steps = parse_number(option, value)?,
            "--token-counts" => args.token_counts = parse_u32_list(option, value)?,
            "--python" => args.python = PathBuf::from(value),
            "--worker-root" => args.worker_root = PathBuf::from(value),
            "--shm-name" => args.shm_name = Some(value.clone()),
            "--metrics-output" => args.metrics_output = Some(PathBuf::from(value)),
            "--model-dir" => args.model_dir = Some(PathBuf::from(value)),
            "--num-kv-pages" => args.num_kv_pages = Some(parse_number(option, value)?),
            "--page-block-tokens" => args.page_block_tokens = Some(parse_number(option, value)?),
            "--prompt-ids" => args.prompt_ids = Some(parse_i32_list(option, value)?),
            "--max-new-tokens" => args.max_new_tokens = parse_number(option, value)?,
            "--verify-greedy-json" => args.verify_greedy_json = Some(PathBuf::from(value)),
            "--verify-output-json" => args.verify_output_json = Some(PathBuf::from(value)),
            "--require-exact-greedy" => {
                args.require_exact_greedy = match value.as_str() {
                    "true" | "1" => true,
                    "false" | "0" => false,
                    _ => {
                        return Err(format!(
                            "--require-exact-greedy expects true/false/1/0, got {value}"
                        ));
                    }
                };
            }
            "--require-exact-cases" => {
                let names: Vec<String> = value
                    .split(',')
                    .map(str::trim)
                    .filter(|s| !s.is_empty())
                    .map(str::to_owned)
                    .collect();
                if names.is_empty() {
                    return Err("--require-exact-cases requires at least one case name".to_owned());
                }
                args.require_exact_cases = names;
            }
            "--max-greedy-mismatch-margin" => {
                let parsed = value.parse::<f32>().map_err(|_| {
                    format!(
                        "--max-greedy-mismatch-margin expects a non-negative float, got {value}"
                    )
                })?;
                if !parsed.is_finite() || parsed < 0.0 {
                    return Err(format!(
                        "--max-greedy-mismatch-margin must be finite and >= 0.0, got {value}"
                    ));
                }
                args.max_greedy_mismatch_margin = Some(parsed);
            }
            "--bench-tiers" => args.bench_tiers = parse_bench_tiers(value)?,
            "--warmup-prompts" => args.warmup_prompts = parse_number(option, value)?,
            "--repeats" => args.repeats = parse_number(option, value)?,
            "--max-prefill-chunk" => args.max_prefill_chunk = parse_number(option, value)?,
            "--max-mixed-step-tokens" => {
                args.max_mixed_step_tokens = Some(parse_number(option, value)?);
            }
            "--max-seqs" => args.max_seqs = Some(parse_number(option, value)?),
            "--max-step-tokens" => args.max_step_tokens = Some(parse_number(option, value)?),
            "--max-decoder-rows" => args.max_decoder_rows = Some(parse_number(option, value)?),
            "--max-pages-per-seq" => args.max_pages_per_seq = Some(parse_number(option, value)?),
            "--hidden-size" => args.hidden_size = parse_number(option, value)?,
            "--output-features" => args.output_features = parse_number(option, value)?,
            "--router-experts" => args.router_experts = parse_number(option, value)?,
            "--router-top-k" => args.router_top_k = parse_number(option, value)?,
            "--allreduce-blocks" => args.allreduce_blocks = parse_number(option, value)?,
            "--allreduce-threads" => args.allreduce_threads = parse_number(option, value)?,
            "--ready-timeout-secs" => args.ready_timeout_secs = parse_number(option, value)?,
            "--step-timeout-secs" => args.step_timeout_secs = parse_number(option, value)?,
            "--heartbeat-timeout-ms" => args.heartbeat_timeout_ms = parse_number(option, value)?,
            _ => return Err(format!("unknown option {option}")),
        }
    }
    if args.repeats == 0 {
        return Err("--repeats must be positive".to_owned());
    }
    if args.max_prefill_chunk == 0 {
        return Err("--max-prefill-chunk must be positive".to_owned());
    }
    if args.verify_greedy_json.is_some() != args.verify_output_json.is_some() {
        return Err(
            "--verify-greedy-json and --verify-output-json must be provided together".to_owned(),
        );
    }
    if args.verify_greedy_json.is_some()
        && (args.model_dir.is_none() || args.prompt_ids.is_some() || !args.bench_tiers.is_empty())
    {
        return Err(
            "greedy verification requires --model-dir and excludes --prompt-ids/--bench-tiers"
                .to_owned(),
        );
    }
    if let Some(ms) = args.max_seqs {
        if ms == 0 {
            return Err("--max-seqs must be positive".to_owned());
        }
        if args.model_dir.is_some() && ms > MAX_DECODE_GRAPH_SEQS {
            return Err(format!(
                "--max-seqs {ms} exceeds maximum supported decode CUDA Graph capacity ({MAX_DECODE_GRAPH_SEQS})"
            ));
        }
    }
    Ok(args)
}

fn parse_number<T: std::str::FromStr>(option: &str, value: &str) -> Result<T, String> {
    value
        .parse::<T>()
        .map_err(|_| format!("{option} expects a valid integer, got {value}"))
}

fn parse_u32_list(option: &str, raw: &str) -> Result<Vec<u32>, String> {
    let values = raw
        .split(',')
        .filter(|entry| !entry.is_empty())
        .map(|entry| parse_number(option, entry))
        .collect::<Result<Vec<u32>, String>>()?;
    if values.is_empty() || values.contains(&0) {
        return Err(format!("{option} requires at least one positive value"));
    }
    Ok(values)
}

fn parse_i32_list(option: &str, raw: &str) -> Result<Vec<i32>, String> {
    let values = raw
        .split(',')
        .filter(|entry| !entry.is_empty())
        .map(|entry| parse_number(option, entry))
        .collect::<Result<Vec<i32>, String>>()?;
    if values.is_empty() {
        return Err(format!("{option} requires at least one token ID"));
    }
    Ok(values)
}

fn parse_bench_tiers(raw: &str) -> Result<Vec<BenchmarkTierSpec>, String> {
    let mut specs = Vec::new();
    for part in raw.split(';').filter(|s| !s.trim().is_empty()) {
        let fields: Vec<&str> = part.trim().split(':').collect();
        if fields.len() != 5 {
            return Err(format!(
                "invalid --bench-tiers entry `{part}`; expected `label:conc:prompts:in_len:out_len`"
            ));
        }
        specs.push(BenchmarkTierSpec {
            label: fields[0].to_owned(),
            concurrency: parse_number("concurrency", fields[1])?,
            num_prompts: parse_number("num_prompts", fields[2])?,
            input_len: parse_number("input_len", fields[3])?,
            output_len: parse_number("output_len", fields[4])?,
        });
    }
    if specs.is_empty() {
        return Err("--bench-tiers requires at least one tier".to_owned());
    }
    Ok(specs)
}

fn unique_shm_name() -> String {
    let nanos = SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .map_or(0, |duration| duration.as_nanos());
    let mut hasher = RandomState::new().build_hasher();
    hasher.write_u128(nanos);
    hasher.write_u32(std::process::id());
    let nonce = hasher.finish();
    format!("zstep_b_{}_{nonce:016x}", std::process::id())
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn parses_multi_tier_benchmark_spec() {
        let tiers = parse_bench_tiers("c1:1:20:1024:256;c32:32:200:1024:256;c128:128:400:1024:256")
            .expect("valid bench tiers");
        assert_eq!(tiers.len(), 3);
        assert_eq!(tiers[0].label, "c1");
        assert_eq!(tiers[0].concurrency, 1);
        assert_eq!(tiers[1].concurrency, 32);
        assert_eq!(tiers[2].num_prompts, 400);
        assert_eq!(tiers[2].input_len, 1024);
        assert_eq!(tiers[2].output_len, 256);
    }

    #[test]
    fn greedy_match_reports_prefix_rate_and_first_divergence() {
        let vllm_steps = vec![
            VllmTop2Step {
                top1_token: 10,
                top1_logprob: -0.01,
                top2_token: 50,
                top2_logprob: -4.01,
                top1_minus_top2_margin: 4.0,
            },
            VllmTop2Step {
                top1_token: 11,
                top1_logprob: -0.02,
                top2_token: 51,
                top2_logprob: -3.02,
                top1_minus_top2_margin: 3.0,
            },
            VllmTop2Step {
                top1_token: 12,
                top1_logprob: -0.68,
                top2_token: 99,
                top2_logprob: -0.71,
                top1_minus_top2_margin: 0.03,
            },
            VllmTop2Step {
                top1_token: 13,
                top1_logprob: -0.05,
                top2_token: 53,
                top2_logprob: -2.05,
                top1_minus_top2_margin: 2.0,
            },
        ];
        let zstep_steps = vec![
            StepTop2Logits {
                top1_token: 10,
                top1_logit: 20.0,
                top2_token: 50,
                top2_logit: 16.0,
                top1_minus_top2_margin: 4.0,
            },
            StepTop2Logits {
                top1_token: 11,
                top1_logit: 19.0,
                top2_token: 51,
                top2_logit: 16.0,
                top1_minus_top2_margin: 3.0,
            },
            StepTop2Logits {
                top1_token: 99,
                top1_logit: 18.125,
                top2_token: 12,
                top2_logit: 18.125,
                top1_minus_top2_margin: 0.0,
            },
            StepTop2Logits {
                top1_token: 13,
                top1_logit: 17.0,
                top2_token: 53,
                top2_logit: 15.0,
                top1_minus_top2_margin: 2.0,
            },
        ];
        let report = greedy_match(
            &[10, 11, 12, 13],
            &vllm_steps,
            &[10, 11, 99, 13],
            &zstep_steps,
        );
        assert_eq!(report.exact_prefix_match_tokens, 2);
        assert_eq!(report.total_tokens, 4);
        assert_eq!(report.exact_token_matches, 3);
        assert_eq!(report.exact_token_match_rate, 0.75);
        assert!(!report.all_matched);
        assert_eq!(report.first_mismatch_token_index, Some(2));
        assert!((report.mean_top1_minus_top2_margin - 2.25).abs() < 1e-6);
        assert_eq!(report.min_top1_minus_top2_margin, 0.0);
        let diag = report.first_mismatch_diagnostics.expect("mismatch diag");
        assert_eq!(diag.token_index, 2);
        assert!(diag.prefix_identical_at_step);
        assert_eq!(diag.expected_token, 12);
        assert_eq!(diag.generated_token, 99);
        assert_eq!(diag.zstep_top2.top2_token, 12);
    }

    #[test]
    fn greedy_verification_arguments_are_paired_and_model_scoped() {
        let args = |tail: &[&str]| {
            let raw: Vec<String> = tail.iter().copied().map(str::to_owned).collect();
            parse_arguments(&raw)
        };
        assert!(args(&["--verify-greedy-json", "reference.json"]).is_err());
        assert!(
            args(&[
                "--model-dir",
                "model",
                "--verify-greedy-json",
                "reference.json",
                "--verify-output-json",
                "report.json",
            ])
            .is_ok()
        );
        assert!(
            args(&[
                "--verify-greedy-json",
                "reference.json",
                "--verify-output-json",
                "report.json",
            ])
            .is_err()
        );
    }

    #[test]
    fn cli_accepts_direct_options_run_subcommand_and_legacy_mode_b_flag() {
        let parse = |items: &[&str]| {
            let raw: Vec<String> = items.iter().copied().map(str::to_owned).collect();
            parse_arguments(&raw).expect("valid CLI arguments")
        };

        let direct = parse(&["--world-size", "4", "--steps", "32"]);
        assert_eq!(direct.world_size, 4);
        assert_eq!(direct.steps, 32);

        let subcmd = parse(&["run", "--world-size", "2", "--steps", "16"]);
        assert_eq!(subcmd.world_size, 2);
        assert_eq!(subcmd.steps, 16);

        let legacy = parse(&["--mode-b", "--world-size", "8", "--steps", "64"]);
        assert_eq!(legacy.world_size, 8);
        assert_eq!(legacy.steps, 64);

        let err = parse_arguments(&["bogus".to_owned()])
            .err()
            .expect("unknown subcommand");
        assert!(err.contains("unknown option bogus"));
    }

    #[test]
    fn greedy_gate_matches_archived_phase2_phase3_and_phase36_reports() {
        let bench_dir = PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("../../docs/benchmarks");
        let load_report = |filename: &str| -> GreedyParityReport {
            let bytes = std::fs::read(bench_dir.join(filename)).expect(filename);
            serde_json::from_slice(&bytes).expect("valid GreedyParityReport JSON")
        };

        let phase2 = load_report("phase2_greedy_parity_report.json");
        let phase3 = load_report("phase3_greedy_parity_report.json");
        let phase36 = load_report("phase36_b12x_greedy_parity_report.json");
        let prime_only = vec!["prime_squares_table".to_owned()];

        for report in [&phase2, &phase3, &phase36] {
            // 1) Default documented invocation (`rc = 0`): enforces zero Triton recompilations.
            assert!(validate_greedy_report_gate(report, &GreedyGatePolicy::default()).is_ok());

            // 2) Case-scoped exact parity (`--require-exact-cases prime_squares_table`, `rc = 0`).
            assert!(
                validate_greedy_report_gate(
                    report,
                    &GreedyGatePolicy {
                        require_exact_cases: &prime_only,
                        ..GreedyGatePolicy::default()
                    },
                )
                .is_ok()
            );

            // 3) Full `--require-exact-greedy true` fails (`rc = 1`) when `all_matched == false`.
            assert!(
                validate_greedy_report_gate(
                    report,
                    &GreedyGatePolicy {
                        require_exact_all: true,
                        ..GreedyGatePolicy::default()
                    },
                )
                .is_err()
            );
        }

        // 4) Strict BF16 ULP tie gate (`--max-greedy-mismatch-margin 0.25`):
        //    passes (`rc = 0`) on Phase 2 BF16 (`<= 0.25`), fails (`rc = 1`) on Phase 3 / 3.6 MXFP8/MXFP4.
        let ulp_gate = GreedyGatePolicy {
            max_mismatch_margin: Some(0.25),
            ..GreedyGatePolicy::default()
        };
        assert!(validate_greedy_report_gate(&phase2, &ulp_gate).is_ok());
        assert!(validate_greedy_report_gate(&phase3, &ulp_gate).is_err());
        assert!(validate_greedy_report_gate(&phase36, &ulp_gate).is_err());

        // 5) Runtime Triton recompilation is unconditionally fatal (`rc = 1`).
        let mut recomp_report = phase2;
        recomp_report.triton_recompilations = Some(1);
        assert!(validate_greedy_report_gate(&recomp_report, &GreedyGatePolicy::default()).is_err());
    }
}
