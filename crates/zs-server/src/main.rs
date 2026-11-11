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

use zs_scheduler::{ModeBConfig, ModeBCoordinator, WorkerLaunchConfig};
use zs_wire::ShmGeometry;

const USAGE: &str = "zstep Mode B master\n\
Usage: zstep --mode-b [OPTIONS]\n\
  --world-size N             GPU workers, 1..=8 (default: 8)\n\
  --steps N                  Overlapped steps (default: 200)\n\
  --token-counts LIST        Runtime rows, comma separated (default: 6,48,192)\n\
  --python PATH              Python executable (default: /opt/venv/bin/python)\n\
  --worker-root PATH         Python package root (default: /workspace/worker)\n\
  --shm-name NAME            POSIX SHM object name (default: unique per run)\n\
  --metrics-output PATH      Also write the JSON metrics to this file\n\
  --max-seqs N               Sequence capacity (default: 8)\n\
  --max-step-tokens N        Token capacity (default: 192)\n\
  --max-decoder-rows N       Decoder row capacity (default: 192)\n\
  --max-pages-per-seq N      Page-table capacity (default: 16)\n\
  --hidden-size N            RMSNorm/GEMM hidden size (default: 5120)\n\
  --output-features N        GEMM output columns (default: 2048)\n\
  --router-experts N         Router expert count (default: 384)\n\
  --router-top-k N           Router selected experts (default: 6)\n\
  --allreduce-blocks N       SHM AllReduce CUDA blocks (default: 32)\n\
  --allreduce-threads N      SHM AllReduce threads/block (default: 256)\n\
  --ready-timeout-secs N     Initialization timeout (default: 300)\n\
  --step-timeout-secs N      Per-step timeout (default: 60)\n";

struct Arguments {
    world_size: u32,
    steps: u64,
    token_counts: Vec<u32>,
    python: PathBuf,
    worker_root: PathBuf,
    shm_name: Option<String>,
    metrics_output: Option<PathBuf>,
    max_seqs: u32,
    max_step_tokens: u32,
    max_decoder_rows: u32,
    max_pages_per_seq: u32,
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
            max_seqs: 8,
            max_step_tokens: 192,
            max_decoder_rows: 192,
            max_pages_per_seq: 16,
            hidden_size: 5120,
            output_features: 2048,
            router_experts: 384,
            router_top_k: 6,
            allreduce_blocks: 32,
            allreduce_threads: 256,
            ready_timeout_secs: 300,
            step_timeout_secs: 60,
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
    let shm_name = args.shm_name.unwrap_or_else(unique_shm_name);
    let config = ModeBConfig {
        shm_name,
        geometry: ShmGeometry {
            world_size: args.world_size,
            max_seqs: args.max_seqs,
            max_step_tokens: args.max_step_tokens,
            max_decoder_rows: args.max_decoder_rows,
            max_pages_per_seq: args.max_pages_per_seq,
            page_block_tokens: 64,
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
        },
        ready_timeout: Duration::from_secs(args.ready_timeout_secs),
        step_timeout: Duration::from_secs(args.step_timeout_secs),
    };
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
            "--token-counts" => args.token_counts = parse_token_counts(value)?,
            "--python" => args.python = PathBuf::from(value),
            "--worker-root" => args.worker_root = PathBuf::from(value),
            "--shm-name" => args.shm_name = Some(value.clone()),
            "--metrics-output" => args.metrics_output = Some(PathBuf::from(value)),
            "--max-seqs" => args.max_seqs = parse_number(option, value)?,
            "--max-step-tokens" => args.max_step_tokens = parse_number(option, value)?,
            "--max-decoder-rows" => args.max_decoder_rows = parse_number(option, value)?,
            "--max-pages-per-seq" => args.max_pages_per_seq = parse_number(option, value)?,
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
        .map_err(|_| format!("{option} expects a positive integer, got {value}"))
}

fn parse_token_counts(value: &str) -> Result<Vec<u32>, String> {
    let counts = value
        .split(',')
        .map(|entry| {
            entry
                .parse::<u32>()
                .map_err(|_| format!("invalid token count {entry}"))
        })
        .collect::<Result<Vec<_>, _>>()?;
    if counts.is_empty() || counts.contains(&0) {
        return Err("token counts must be positive".to_owned());
    }
    Ok(counts)
}

fn unique_shm_name() -> String {
    let nanos = SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .unwrap_or_default()
        .as_nanos();
    format!("zstep_ipc_{}_{nanos}", std::process::id())
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn explicit_runtime_token_counts_are_parsed() {
        let raw = vec![
            "--world-size".to_owned(),
            "2".to_owned(),
            "--steps".to_owned(),
            "4".to_owned(),
            "--token-counts".to_owned(),
            "6,17,48".to_owned(),
        ];
        let args = parse_arguments(&raw).expect("valid arguments");
        assert_eq!(args.world_size, 2);
        assert_eq!(args.steps, 4);
        assert_eq!(args.token_counts, [6, 17, 48]);
    }

    #[test]
    fn invalid_token_counts_are_rejected() {
        assert!(parse_token_counts("6,0,48").is_err());
        assert!(parse_token_counts("six").is_err());
    }
}
