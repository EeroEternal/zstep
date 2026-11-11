# Copyright (c) 2026 zstep Contributors
#
# Permission is hereby granted, free of charge, to any person obtaining a copy
# of this software and associated documentation files (the "Software"), to deal
# in the Software without restriction, including without limitation the rights
# to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
# copies of the Software, and to permit persons to whom the Software is
# furnished to do so, subject to the following conditions:
#
# The above copyright notice and this permission notice shall be included in all
# copies or substantial portions of the Software.
#
# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
# IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
# FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
# AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
# LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
# OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
# SOFTWARE.

"""Collect reproducible Qwen3.5 greedy and warm interleaved vLLM references."""

from __future__ import annotations

import argparse
from dataclasses import dataclass
from datetime import datetime, timezone
import importlib.metadata
import json
import os
from pathlib import Path
import shlex
import statistics
import subprocess
import time
from typing import Any
import urllib.request
from urllib.parse import urlsplit

DEFAULT_MODEL_DIR = "/models/Qwen3.5-35B-A3B"
DEFAULT_IMAGE = "vllm/vllm-openai:v0.33.0"
METRIC_NAMES = (
    "duration_s",
    "output_throughput_tok_s",
    "total_throughput_tok_s",
    "mean_ttft_ms",
    "median_ttft_ms",
    "p99_ttft_ms",
    "mean_tpot_ms",
    "median_tpot_ms",
    "p99_tpot_ms",
    "mean_itl_ms",
    "median_itl_ms",
    "p99_itl_ms",
)


@dataclass(frozen=True)
class PromptSpec:
    name: str
    text: str
    max_tokens: int


@dataclass(frozen=True)
class TierSpec:
    name: str
    input_len: int
    output_len: int
    num_prompts: int
    max_concurrency: int


PROMPTS = (
    PromptSpec(
        name="france_capital",
        text="The capital of France is",
        max_tokens=32,
    ),
    PromptSpec(
        name="ieee754_subnormals",
        text=(
            "Explain IEEE-754 floating point representation and subnormal numbers "
            "step by step, including sign, exponent, significand, bias, and gradual underflow."
        ),
        max_tokens=128,
    ),
    PromptSpec(
        name="dijkstra_shortest_path",
        text=(
            "Walk through Dijkstra shortest-path algorithm on a weighted graph. "
            "Explain the priority queue, distance initialization, relaxation, visited "
            "states, complexity, and why negative edge weights are invalid."
        ),
        max_tokens=96,
    ),
    PromptSpec(
        name="posix_shm_tlb_shootdown",
        text=(
            "Explain how Linux implements POSIX shared memory, how page tables map "
            "shm pages into processes, and why TLB shootdowns are required after "
            "unmapping. Include shm_open, mmap, page faults, huge pages, and "
            "multiprocessor invalidation. Discuss locking, cache coherency, IPI "
            "latency, memory ordering, and safe access across sockets."
        ),
        max_tokens=64,
    ),
)

TIERS = (
    TierSpec("in512_c1", 512, 256, 8, 1),
    TierSpec("in512_c16", 512, 256, 48, 16),
    TierSpec("in512_c128", 512, 256, 256, 128),
    TierSpec("in1024_c1", 1024, 256, 20, 1),
    TierSpec("in1024_c32", 1024, 256, 200, 32),
    TierSpec("in1024_c128", 1024, 256, 400, 128),
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--phase", required=True, choices=("greedy", "benchmark"))
    parser.add_argument("--vllm-url", required=True)
    parser.add_argument("--model-name", required=True)
    parser.add_argument("--output-json", required=True, type=Path)
    parser.add_argument("--seed", required=True, type=int)
    parser.add_argument(
        "--tokenizer-dir",
        default=os.environ.get("ZSTEP_QWEN35_MODEL_DIR", DEFAULT_MODEL_DIR),
    )
    parser.add_argument("--raw-results-dir", type=Path)
    parser.add_argument("--request-timeout-s", type=float, default=600.0)
    parser.add_argument("--bench-timeout-s", type=float, default=3600.0)
    args = parser.parse_args()
    if args.request_timeout_s <= 0 or args.bench_timeout_s <= 0:
        parser.error("request and benchmark timeouts must be positive")
    return args


def atomic_write_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(value, indent=2) + "\n", encoding="utf-8")
    temporary.replace(path)


def request_json(
    url: str,
    timeout_s: float,
    payload: dict[str, Any] | None = None,
) -> dict[str, Any]:
    data = None if payload is None else json.dumps(payload).encode("utf-8")
    headers = {} if data is None else {"Content-Type": "application/json"}
    request = urllib.request.Request(url, data=data, headers=headers)
    with urllib.request.urlopen(request, timeout=timeout_s) as response:
        result = json.loads(response.read().decode("utf-8"))
    if not isinstance(result, dict):
        raise RuntimeError(f"expected a JSON object from {url}")
    return result


def check_health(vllm_url: str, timeout_s: float) -> None:
    request = urllib.request.Request(f"{vllm_url}/health")
    with urllib.request.urlopen(request, timeout=timeout_s) as response:
        if response.status != 200:
            raise RuntimeError(f"vLLM health returned HTTP {response.status}")


def public_url(vllm_url: str) -> str:
    parsed = urlsplit(vllm_url)
    port = parsed.port or 8000
    return f"http://127.0.0.1:{port}"


def serve_arguments(model_name: str, seed: int, vllm_url: str) -> list[str]:
    model_dir = os.environ.get("ZSTEP_QWEN35_MODEL_DIR", DEFAULT_MODEL_DIR)
    port = urlsplit(public_url(vllm_url)).port
    return [
        model_dir,
        "--served-model-name",
        model_name,
        "--tensor-parallel-size",
        "8",
        "--dtype",
        "bfloat16",
        "--max-model-len",
        "32768",
        "--gpu-memory-utilization",
        "0.85",
        "--host",
        "0.0.0.0",
        "--port",
        str(port),
        "--seed",
        str(seed),
    ]


def serve_command(model_name: str, seed: int, vllm_url: str) -> str:
    return shlex.join(["vllm", "serve", *serve_arguments(model_name, seed, vllm_url)])


def capture_hardware() -> dict[str, Any]:
    output = subprocess.run(
        [
            "nvidia-smi",
            "--query-gpu=name,driver_version,clocks.current.graphics,clocks.current.memory,memory.total",
            "--format=csv,noheader,nounits",
        ],
        check=True,
        capture_output=True,
        text=True,
    )
    rows = []
    for line in output.stdout.splitlines():
        if not line.strip():
            continue
        name, driver, graphics, memory, memory_total = [
            field.strip() for field in line.split(",")
        ]
        rows.append(
            {
                "name": name,
                "driver_version": driver,
                "graphics_clock_mhz": int(graphics),
                "memory_clock_mhz": int(memory),
                "memory_total_mib": int(memory_total),
            }
        )
    if not rows:
        raise RuntimeError("nvidia-smi returned no GPU rows")
    driver_versions = {row["driver_version"] for row in rows}
    if len(driver_versions) != 1:
        raise RuntimeError(f"inconsistent GPU driver versions: {sorted(driver_versions)}")
    return {
        "nvidia_driver_version": rows[0]["driver_version"],
        "gpu_clocks_mhz": [row["graphics_clock_mhz"] for row in rows],
        "gpu_memory_clocks_mhz": [row["memory_clock_mhz"] for row in rows],
        "gpus": rows,
    }


def current_git_sha() -> str:
    environment_sha = os.environ.get("ZSTEP_GIT_SHA")
    if environment_sha is not None:
        candidates = [environment_sha]
    else:
        git_dir = Path(__file__).resolve().parents[3] / ".git"
        head = (git_dir / "HEAD").read_text(encoding="utf-8").strip()
        if head.startswith("ref: "):
            head = (git_dir / head.removeprefix("ref: ")).read_text(
                encoding="utf-8"
            ).strip()
        candidates = [head]
    sha = candidates[0]
    if len(sha) != 40 or any(character not in "0123456789abcdef" for character in sha):
        raise RuntimeError(f"invalid Git SHA provenance: {sha}")
    return sha


def common_provenance(args: argparse.Namespace) -> dict[str, Any]:
    return {
        "engine": "vllm",
        "vllm_version": importlib.metadata.version("vllm"),
        "model": args.model_name,
        "container_image": os.environ.get("ZSTEP_VLLM_IMAGE", DEFAULT_IMAGE),
        "dtype": "bfloat16",
        "tp_size": 8,
        "seed": args.seed,
        "git_sha": current_git_sha(),
        "collection_timestamp_utc": datetime.now(timezone.utc).isoformat(),
        "serve_args": serve_arguments(args.model_name, args.seed, args.vllm_url),
        "vllm_serve_command": serve_command(
            args.model_name, args.seed, args.vllm_url
        ),
        **capture_hardware(),
    }


def encode_prompts(tokenizer_dir: str) -> list[list[int]]:
    from transformers import AutoTokenizer

    tokenizer = AutoTokenizer.from_pretrained(tokenizer_dir, local_files_only=True)
    encoded = [
        tokenizer.encode(prompt.text, add_special_tokens=False) for prompt in PROMPTS
    ]
    if encoded[0] != [760, 6511, 314, 9338, 369]:
        raise RuntimeError(f"unexpected Prompt 0 token IDs: {encoded[0]}")
    if not 20 <= len(encoded[1]) <= 45:
        raise RuntimeError(f"Prompt 1 length is not approximately 25 tokens: {len(encoded[1])}")
    if not 30 <= len(encoded[2]) <= 50:
        raise RuntimeError(f"Prompt 2 length is not approximately 40 tokens: {len(encoded[2])}")
    if not 65 <= len(encoded[3]) <= 72:
        raise RuntimeError(f"Prompt 3 must span three 24-token chunks: {len(encoded[3])}")
    return encoded


def collect_greedy(args: argparse.Namespace) -> dict[str, Any]:
    check_health(args.vllm_url, args.request_timeout_s)
    prompt_ids = encode_prompts(args.tokenizer_dir)
    cases = []
    response_metadata = []
    for prompt, tokens in zip(PROMPTS, prompt_ids, strict=True):
        payload = {
            "model": args.model_name,
            "prompt": tokens,
            "temperature": 0.0,
            "top_p": 1.0,
            "max_tokens": prompt.max_tokens,
            "logprobs": 1,
            "ignore_eos": True,
            "return_token_ids": True,
            "seed": args.seed,
        }
        response = request_json(
            f"{args.vllm_url}/v1/completions",
            args.request_timeout_s,
            payload,
        )
        choices = response.get("choices")
        if not isinstance(choices, list) or len(choices) != 1:
            raise RuntimeError(f"unexpected completion choices for {prompt.name}: {choices}")
        choice = choices[0]
        generated = choice.get("token_ids")
        if not isinstance(generated, list) or not all(
            isinstance(token, int) for token in generated
        ):
            raise RuntimeError(f"vLLM did not return exact token IDs for {prompt.name}")
        if len(generated) != prompt.max_tokens:
            raise RuntimeError(
                f"{prompt.name} returned {len(generated)} tokens, expected {prompt.max_tokens}"
            )
        returned_prompt = choice.get("prompt_token_ids")
        if returned_prompt is not None and returned_prompt != tokens:
            raise RuntimeError(f"vLLM changed the prompt token IDs for {prompt.name}")
        if choice.get("logprobs") is None:
            raise RuntimeError(f"vLLM omitted requested logprobs for {prompt.name}")
        cases.append(
            {
                "name": prompt.name,
                "prompt_text": prompt.text,
                "prompt_token_ids": tokens,
                "max_tokens": prompt.max_tokens,
                "expected_greedy_token_ids": generated,
            }
        )
        response_metadata.append(
            {
                "name": prompt.name,
                "response_id": response.get("id"),
                "finish_reason": choice.get("finish_reason"),
                "usage": response.get("usage"),
                "token_id_source": "choices[0].token_ids (return_token_ids=true)",
            }
        )
        print(f"collected {prompt.name}: prompt={len(tokens)} output={len(generated)}", flush=True)
    return {
        **common_provenance(args),
        "request_parameters": {
            "endpoint": "/v1/completions",
            "temperature": 0.0,
            "top_p": 1.0,
            "logprobs": 1,
            "ignore_eos": True,
            "return_token_ids": True,
        },
        "cases": cases,
        "response_metadata": response_metadata,
    }


def benchmark_command(
    args: argparse.Namespace,
    tier: TierSpec,
    result_filename: str,
    raw_results_dir: Path,
    recorded_url: str,
) -> list[str]:
    return [
        "vllm",
        "bench",
        "serve",
        "--backend",
        "openai-chat",
        "--endpoint",
        "/v1/chat/completions",
        "--base-url",
        recorded_url,
        "--model",
        args.model_name,
        "--tokenizer",
        args.tokenizer_dir,
        "--dataset-name",
        "random",
        "--random-input-len",
        str(tier.input_len),
        "--random-output-len",
        str(tier.output_len),
        "--random-range-ratio",
        "0.0",
        "--num-prompts",
        str(tier.num_prompts),
        "--max-concurrency",
        str(tier.max_concurrency),
        "--ignore-eos",
        "--temperature",
        "0.0",
        "--seed",
        str(args.seed),
        "--percentile-metrics",
        "ttft,tpot,itl",
        "--metric-percentiles",
        "99",
        "--save-result",
        "--result-dir",
        str(raw_results_dir),
        "--result-filename",
        result_filename,
    ]


def sanitize_raw(value: Any, actual_url: str, recorded_url: str) -> Any:
    if isinstance(value, dict):
        return {
            key: sanitize_raw(item, actual_url, recorded_url)
            for key, item in value.items()
        }
    if isinstance(value, list):
        return [sanitize_raw(item, actual_url, recorded_url) for item in value]
    if isinstance(value, str):
        return value.replace(actual_url, recorded_url)
    return value


def extract_percentile(raw: dict[str, Any], metric: str, percentile: int) -> float:
    direct_key = f"p{percentile}_{metric}_ms"
    if direct_key in raw:
        return float(raw[direct_key])
    values = raw.get(f"percentiles_{metric}_ms", [])
    for entry in values:
        if isinstance(entry, list) and len(entry) == 2 and int(entry[0]) == percentile:
            return float(entry[1])
    raise KeyError(f"missing p{percentile} {metric} in vLLM result")


def extract_metrics(raw: dict[str, Any]) -> dict[str, float]:
    metrics = {
        "duration_s": float(raw["duration"]),
        "output_throughput_tok_s": float(raw["output_throughput"]),
        "total_throughput_tok_s": float(raw["total_token_throughput"]),
    }
    for metric in ("ttft", "tpot", "itl"):
        metrics[f"mean_{metric}_ms"] = float(raw[f"mean_{metric}_ms"])
        metrics[f"median_{metric}_ms"] = float(raw[f"median_{metric}_ms"])
        metrics[f"p99_{metric}_ms"] = extract_percentile(raw, metric, 99)
    if set(metrics) != set(METRIC_NAMES):
        raise RuntimeError("internal metric schema mismatch")
    return metrics


def aggregate_metrics(rounds: list[dict[str, Any]]) -> dict[str, dict[str, float]]:
    aggregates = {}
    for metric in METRIC_NAMES:
        samples = [float(round_result["metrics"][metric]) for round_result in rounds]
        aggregates[metric] = {
            "mean": statistics.fmean(samples),
            "std": statistics.pstdev(samples),
            "min": min(samples),
            "max": max(samples),
        }
    return aggregates


def run_benchmark(
    args: argparse.Namespace,
    tier: TierSpec,
    result_filename: str,
    raw_results_dir: Path,
) -> tuple[dict[str, Any], list[str]]:
    result_path = raw_results_dir / result_filename
    result_path.unlink(missing_ok=True)
    recorded_url = public_url(args.vllm_url)
    recorded_command = benchmark_command(
        args, tier, result_filename, raw_results_dir, recorded_url
    )
    actual_command = benchmark_command(
        args, tier, result_filename, raw_results_dir, args.vllm_url
    )
    subprocess.run(actual_command, check=True, timeout=args.bench_timeout_s)
    raw = json.loads(result_path.read_text(encoding="utf-8"))
    raw = sanitize_raw(raw, args.vllm_url, recorded_url)
    atomic_write_json(result_path, raw)
    if int(raw["completed"]) != tier.num_prompts:
        raise RuntimeError(
            f"{result_filename} completed {raw['completed']} of {tier.num_prompts} prompts"
        )
    return raw, recorded_command


def collect_benchmark(args: argparse.Namespace) -> dict[str, Any]:
    check_health(args.vllm_url, args.request_timeout_s)
    raw_results_dir = args.raw_results_dir
    if raw_results_dir is None:
        raw_results_dir = args.output_json.parent / "phase2_vllm_raw"
    raw_results_dir.mkdir(parents=True, exist_ok=True)
    warmup = TierSpec("warmup", 512, 128, 32, 16)
    _, warmup_command = run_benchmark(
        args, warmup, "warmup.json", raw_results_dir
    )
    rounds = []
    tier_results: dict[str, list[dict[str, Any]]] = {tier.name: [] for tier in TIERS}
    commands: dict[str, list[str]] = {}
    for round_index in range(3):
        for tier in TIERS:
            filename = f"round{round_index}_{tier.name}.json"
            started = time.monotonic()
            raw, command = run_benchmark(args, tier, filename, raw_results_dir)
            metrics = extract_metrics(raw)
            result = {
                "round": round_index,
                "tier": tier.name,
                "input_len": tier.input_len,
                "output_len": tier.output_len,
                "num_prompts": tier.num_prompts,
                "max_concurrency": tier.max_concurrency,
                "completed": int(raw["completed"]),
                "raw_result_file": str(raw_results_dir / filename),
                "metrics": metrics,
                "vllm_bench_command": shlex.join(command),
            }
            rounds.append(result)
            tier_results[tier.name].append(result)
            commands[tier.name] = command
            print(
                f"round={round_index} tier={tier.name} "
                f"out={metrics['output_throughput_tok_s']:.2f} tok/s "
                f"p99_itl={metrics['p99_itl_ms']:.2f} ms "
                f"collector_wall={time.monotonic() - started:.1f}s",
                flush=True,
            )
    tiers = []
    for tier in TIERS:
        results = tier_results[tier.name]
        tiers.append(
            {
                "name": tier.name,
                "input_len": tier.input_len,
                "output_len": tier.output_len,
                "num_prompts": tier.num_prompts,
                "max_concurrency": tier.max_concurrency,
                "rounds": results,
                "aggregates": aggregate_metrics(results),
            }
        )
    return {
        **common_provenance(args),
        "benchmark": {
            "backend": "openai-chat",
            "endpoint": "/v1/chat/completions",
            "dataset_name": "random",
            "random_range_ratio": 0.0,
            "ignore_eos": True,
            "temperature": 0.0,
            "rounds": 3,
            "schedule": "warmup once, then round-major interleaving of all six tiers",
            "warmup": {
                "input_len": 512,
                "output_len": 128,
                "num_prompts": 32,
                "max_concurrency": 16,
                "raw_result_file": str(raw_results_dir / "warmup.json"),
                "vllm_bench_command": shlex.join(warmup_command),
            },
            "aggregate_std": "population standard deviation across three rounds",
        },
        "vllm_bench_command": shlex.join(commands[TIERS[0].name]),
        "vllm_bench_commands": {
            name: shlex.join(command) for name, command in commands.items()
        },
        "rounds": rounds,
        "tiers": tiers,
    }


def main() -> None:
    args = parse_args()
    if args.phase == "greedy":
        report = collect_greedy(args)
    else:
        report = collect_benchmark(args)
    atomic_write_json(args.output_json, report)
    print(f"wrote {args.output_json}", flush=True)


if __name__ == "__main__":
    main()
