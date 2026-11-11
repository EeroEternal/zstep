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

use std::path::PathBuf;
use std::process::ExitCode;
use std::time::{Duration, SystemTime, UNIX_EPOCH};

use zs_scheduler::{
    BenchmarkTierSpec, ModeBConfig, ModeBCoordinator, ModeBServingEngine, ServingBenchmarkReport,
    WorkerLaunchConfig,
};
use zs_wire::ShmGeometry;

const USAGE: &str = "zstep Mode B master\n\
Usage: zstep --mode-b [OPTIONS]\n\
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
  --bench-tiers SPEC         Semicolon-separated tiers `label:conc:prompts:in_len:out_len`\n\
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
  --step-timeout-secs N      Per-step timeout (default: 120)\n";

struct Arguments {
    world_size: u32,
    steps: u64,
    token_counts: Vec<u32>,
    python: PathBuf,
    worker_root: PathBuf,
    shm_name: Option<String>,
    metrics_output: Option<PathBuf>,
    model_dir: Option<PathBuf>,
    num_kv_pages: u32,
    page_block_tokens: Option<u32>,
    prompt_ids: Option<Vec<i32>>,
    max_new_tokens: usize,
    bench_tiers: Vec<BenchmarkTierSpec>,
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
            num_kv_pages: 49_152,
            page_block_tokens: None,
            prompt_ids: None,
            max_new_tokens: 16,
            bench_tiers: Vec::new(),
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
        }
    }
}

fn main() -> ExitCode {
    let raw_args: Vec<String> = std::env::args().skip(1).collect();
    if raw_args.is_empty() || raw_args.iter().any(|arg| arg == "--help" || arg == "-h") {
        print!("{USAGE}");
        return ExitCode::SUCCESS;
    }
    if raw_args.first().map(String::as_str) != Some("--mode-b") {
        eprintln!("--mode-b is required\n{USAGE}");
        return ExitCode::FAILURE;
    }
    let args = match parse_arguments(&raw_args[1..]) {
        Ok(args) => args,
        Err(error) => {
            eprintln!("argument error: {error}");
            return ExitCode::FAILURE;
        }
    };
    match run_mode_b(args) {
        Ok(()) => ExitCode::SUCCESS,
        Err(error) => {
            eprintln!("Mode B failed: {error}");
            ExitCode::FAILURE
        }
    }
}

fn run_mode_b(args: Arguments) -> Result<(), Box<dyn std::error::Error>> {
    let has_model = args.model_dir.is_some();
    let max_seqs = args.max_seqs.unwrap_or(if has_model { 128 } else { 8 });
    let max_step_tokens = args
        .max_step_tokens
        .unwrap_or(if has_model { 4096 } else { 192 });
    let max_decoder_rows =
        args.max_decoder_rows
            .unwrap_or(if has_model { max_step_tokens } else { 192 });
    let max_pages_per_seq = args
        .max_pages_per_seq
        .unwrap_or(if has_model { 256 } else { 16 });
    let page_block_tokens = args
        .page_block_tokens
        .unwrap_or(if has_model { 16 } else { 64 });

    let shm_name = args.shm_name.unwrap_or_else(unique_shm_name);
    let config = ModeBConfig {
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
            num_kv_pages: args.num_kv_pages,
        },
        ready_timeout: Duration::from_secs(args.ready_timeout_secs),
        step_timeout: Duration::from_secs(args.step_timeout_secs),
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
        let mut engine = ModeBServingEngine::launch(config)?;

        if let Some(prompt) = args.prompt_ids {
            let outs =
                engine.generate_greedy(std::slice::from_ref(&prompt), args.max_new_tokens)?;
            println!("PROMPT_IDS={prompt:?}");
            println!("GENERATED_IDS={:?}", outs[0]);
        }

        if !args.bench_tiers.is_empty() {
            let mut tier_results = Vec::with_capacity(args.bench_tiers.len());
            for spec in &args.bench_tiers {
                let res = engine.run_benchmark_tier(spec)?;
                eprintln!(
                    "[{}] conc={} prompts={} in={} out={} -> out_tok/s={:.2}, total_tok/s={:.2}, median_ttft={:.2}ms, mean_tpot={:.2}ms, p99_tpot={:.2}ms (steps={}, overlapped={})",
                    res.label,
                    res.concurrency,
                    res.num_prompts,
                    res.input_len,
                    res.output_len,
                    res.output_throughput_tok_s,
                    res.total_throughput_tok_s,
                    res.median_ttft_ms,
                    res.mean_tpot_ms,
                    res.p99_tpot_ms,
                    res.steps,
                    res.overlapped_decode_steps,
                );
                tier_results.push(res);
            }
            let report = ServingBenchmarkReport {
                model_name: model_label,
                world_size: args.world_size,
                worker_numa_nodes: engine.worker_numa_nodes().to_vec(),
                init_wall_ms: engine.init_wall_ms(),
                tiers: tier_results,
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

    let metrics = ModeBCoordinator::launch(config)?.run()?;
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
            "--num-kv-pages" => args.num_kv_pages = parse_number(option, value)?,
            "--page-block-tokens" => args.page_block_tokens = Some(parse_number(option, value)?),
            "--prompt-ids" => args.prompt_ids = Some(parse_i32_list(option, value)?),
            "--max-new-tokens" => args.max_new_tokens = parse_number(option, value)?,
            "--bench-tiers" => args.bench_tiers = parse_bench_tiers(value)?,
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
            _ => return Err(format!("unknown option {option}")),
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
    format!("zstep_b_{}_{}", std::process::id(), nanos % 1_000_000_000)
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
}
