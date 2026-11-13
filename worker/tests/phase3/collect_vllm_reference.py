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

"""Collect reproducible DeepSeek-V4.1-Flash greedy and warm interleaved vLLM references."""

from __future__ import annotations

import argparse
from dataclasses import dataclass
from datetime import UTC, datetime
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

DEFAULT_MODEL_DIR = "/models/DeepSeek-V4.1-Flash"
DEFAULT_IMAGE = "vllm/vllm-openai:v0.31.0"
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
    PromptSpec(
        name="prime_squares_table",
        text=(
            "Continue this exact numbered arithmetic table without commentary, "
            "one line per prime p:\n"
            "1. p=2, p^2=4, 2p+1=5\n"
            "2. p=3, p^2=9, 2p+1=7\n"
            "3. p=5, p^2=25, 2p+1=11\n"
            "4."
        ),
        max_tokens=128,
    ),
)

SIX_TIERS = (
    TierSpec("in512_c1", 512, 256, 8, 1),
    TierSpec("in512_c16", 512, 256, 48, 16),
    TierSpec("in512_c128", 512, 256, 256, 128),
    TierSpec("in1024_c1", 1024, 256, 20, 1),
    TierSpec("in1024_c32", 1024, 256, 200, 32),
    TierSpec("in1024_c128", 1024, 256, 400, 128),
)

IN512_OUT128_TIERS = (
    TierSpec("c1", 512, 128, 16, 1),
    TierSpec("c8", 512, 128, 32, 8),
    TierSpec("c16", 512, 128, 64, 16),
    TierSpec("c32", 512, 128, 128, 32),
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
        default=os.environ.get("ZSTEP_DSV41_MODEL_DIR", DEFAULT_MODEL_DIR),
    )
    parser.add_argument(
        "--tier-suite",
        choices=("six_tier", "in512_out128"),
        default="six_tier",
    )
    parser.add_argument("--rounds", type=int, default=3)
    parser.add_argument("--spec-mode", default="off")
    parser.add_argument("--engram-memory", default="device")
    parser.add_argument("--raw-results-dir", type=Path)
    parser.add_argument("--request-timeout-s", type=float, default=600.0)
    parser.add_argument("--bench-timeout-s", type=float, default=3600.0)
    args = parser.parse_args()
    if args.request_timeout_s <= 0 or args.bench_timeout_s <= 0 or args.rounds <= 0:
        parser.error("request timeout, benchmark timeout, and rounds must be positive")
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
    opener = urllib.request.build_opener(urllib.request.ProxyHandler({}))
    try:
        with opener.open(request, timeout=timeout_s) as response:
            result = json.loads(response.read().decode("utf-8"))
    except urllib.error.HTTPError as exc:
        body = exc.read().decode("utf-8", errors="replace")
        raise RuntimeError(f"HTTP {exc.code} {exc.reason} from {url}: {body}") from exc
    if not isinstance(result, dict):
        raise RuntimeError(f"expected a JSON object from {url}")
    return result


def check_health(vllm_url: str, timeout_s: float) -> None:
    request = urllib.request.Request(f"{vllm_url}/health")
    opener = urllib.request.build_opener(urllib.request.ProxyHandler({}))
    with opener.open(request, timeout=timeout_s) as response:
        if response.status != 200:
            raise RuntimeError(f"vLLM health returned HTTP {response.status}")


def public_url(vllm_url: str) -> str:
    parsed = urlsplit(vllm_url)
    port = parsed.port or 8000
    return f"http://127.0.0.1:{port}"


def serve_arguments(args: argparse.Namespace) -> list[str]:
    port = urlsplit(public_url(args.vllm_url)).port
    cpu_offload = args.engram_memory != "device"
    return [
        DEFAULT_MODEL_DIR,
        "--served-model-name",
        args.model_name,
        "--tensor-parallel-size",
        "8",
        "--dtype",
        "bfloat16",
        "--max-model-len",
        "32768",
        "--max-num-seqs",
        "128",
        "--gpu-memory-utilization",
        "0.95",
        "--engram-config",
        json.dumps({"cpu_offload": cpu_offload}),
        "--limit-mm-per-prompt",
        json.dumps({"image": 0}),
        "--no-enable-prefix-caching",
        "--host",
        "0.0.0.0",
        "--port",
        str(port),
        "--seed",
        str(args.seed),
    ]


def serve_command(args: argparse.Namespace) -> str:
    return shlex.join(["vllm", "serve", *serve_arguments(args)])


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
        name, driver, graphics, memory, memory_total = (
            field.strip() for field in line.split(",")
        )
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
    if environment_sha is not None and environment_sha.strip():
        sha = environment_sha.strip().lower()
        if len(sha) != 40 or any(c not in "0123456789abcdef" for c in sha):
            raise RuntimeError(f"invalid Git SHA provenance: {sha}")
        return sha
    try:
        sha = subprocess.check_output(
            ["git", "rev-parse", "HEAD"],
            cwd=Path(__file__).resolve().parents[3],
            text=True,
            timeout=5,
        ).strip().lower()
        if len(sha) == 40 and all(c in "0123456789abcdef" for c in sha):
            return sha
    except (OSError, subprocess.SubprocessError):
        pass
    return "unknown"


def common_provenance(args: argparse.Namespace) -> dict[str, Any]:
    return {
        "engine": "vllm",
        "vllm_version": importlib.metadata.version("vllm"),
        "model": args.model_name,
        "container_image": os.environ.get("ZSTEP_VLLM_IMAGE", DEFAULT_IMAGE),
        "dtype": "bfloat16",
        "tp_size": 8,
        "spec_mode": args.spec_mode,
        "engram_memory": args.engram_memory,
        "seed": args.seed,
        "git_sha": current_git_sha(),
        "collection_timestamp_utc": datetime.now(UTC).isoformat(),
        "serve_args": serve_arguments(args),
        "serve_env": {
            "NCCL_P2P_DISABLE": "1",
            "NCCL_P2P_LEVEL": "LOC",
            "sm120_flashinfer_page_block_size_64_shim": True,
        },
        "vllm_serve_command": serve_command(args),
        **capture_hardware(),
    }


def encode_prompts(tokenizer_dir: str) -> list[list[int]]:
    from transformers import AutoTokenizer

    tokenizer = AutoTokenizer.from_pretrained(
        tokenizer_dir, local_files_only=True, trust_remote_code=True
    )
    encoded = [
        tokenizer.encode(prompt.text, add_special_tokens=False) for prompt in PROMPTS
    ]
    if not 3 <= len(encoded[0]) <= 12:
        raise RuntimeError(f"unexpected Prompt 0 length: {len(encoded[0])}")
    if not 20 <= len(encoded[1]) <= 50:
        raise RuntimeError(f"unexpected Prompt 1 length: {len(encoded[1])}")
    if not 25 <= len(encoded[2]) <= 60:
        raise RuntimeError(f"unexpected Prompt 2 length: {len(encoded[2])}")
    if not 55 <= len(encoded[3]) <= 90:
        raise RuntimeError(f"Prompt 3 must span three 24-token chunks: {len(encoded[3])}")
    if not 45 <= len(encoded[4]) <= 95:
        raise RuntimeError(f"Prompt 4 unexpected length: {len(encoded[4])}")
    return encoded


def _parse_token_id_key(raw_key: str) -> int:
    if not raw_key.startswith("token_id:"):
        raise RuntimeError(
            f"expected return_tokens_as_token_ids format 'token_id:<int>', got {raw_key!r}"
        )
    return int(raw_key.removeprefix("token_id:"))


def _extract_top2_steps(
    prompt_name: str,
    generated: list[int],
    logprobs_obj: dict[str, Any],
) -> list[dict[str, Any]]:
    top_logprobs = logprobs_obj.get("top_logprobs")
    if not isinstance(top_logprobs, list) or len(top_logprobs) != len(generated):
        raise RuntimeError(
            f"{prompt_name}: expected {len(generated)} top_logprobs entries, got {type(top_logprobs)}"
        )
    steps: list[dict[str, Any]] = []
    for step_idx, (chosen_tok, step_dict) in enumerate(
        zip(generated, top_logprobs, strict=True)
    ):
        if not isinstance(step_dict, dict) or len(step_dict) < 2:
            raise RuntimeError(
                f"{prompt_name} step {step_idx}: expected >=2 top_logprobs candidates, got {step_dict}"
            )
        parsed = [
            (_parse_token_id_key(key), float(val)) for key, val in step_dict.items()
        ]
        chosen_matches = [lp for tok, lp in parsed if tok == chosen_tok]
        if not chosen_matches:
            raise RuntimeError(
                f"{prompt_name} step {step_idx}: chosen token {chosen_tok} missing from top_logprobs"
            )
        top1_lp = chosen_matches[0]
        others = sorted(
            [(tok, lp) for tok, lp in parsed if tok != chosen_tok],
            key=lambda item: (-item[1], item[0]),
        )
        top2_tok, top2_lp = others[0]
        steps.append(
            {
                "top1_token": int(chosen_tok),
                "top1_logprob": top1_lp,
                "top2_token": int(top2_tok),
                "top2_logprob": top2_lp,
                "top1_minus_top2_margin": top1_lp - top2_lp,
            }
        )
    return steps


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
            "logprobs": 5,
            "ignore_eos": True,
            "return_token_ids": True,
            "return_tokens_as_token_ids": True,
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
        logprobs_obj = choice.get("logprobs")
        if not isinstance(logprobs_obj, dict):
            raise RuntimeError(f"vLLM omitted requested logprobs for {prompt.name}")
        top2_steps = _extract_top2_steps(prompt.name, generated, logprobs_obj)
        if prompt.name == "prime_squares_table":
            windows_16 = {
                tuple(generated[idx : idx + 16]) for idx in range(len(generated) - 15)
            }
            if len(windows_16) != len(generated) - 15:
                raise RuntimeError(
                    f"prime_squares_table repeated a 16-token window: {generated[-32:]}"
                )
        margins = [step["top1_minus_top2_margin"] for step in top2_steps]
        cases.append(
            {
                "name": prompt.name,
                "prompt_text": prompt.text,
                "prompt_token_ids": tokens,
                "max_tokens": prompt.max_tokens,
                "expected_greedy_token_ids": generated,
                "top2_steps": top2_steps,
            }
        )
        response_metadata.append(
            {
                "name": prompt.name,
                "response_id": response.get("id"),
                "finish_reason": choice.get("finish_reason"),
                "usage": response.get("usage"),
                "token_id_source": "choices[0].token_ids (return_token_ids=true, return_tokens_as_token_ids=true)",
                "mean_top1_minus_top2_margin": statistics.fmean(margins),
                "min_top1_minus_top2_margin": min(margins),
            }
        )
        print(
            f"collected {prompt.name}: prompt={len(tokens)} output={len(generated)} "
            f"min_margin={min(margins):.4f} mean_margin={statistics.fmean(margins):.4f}",
            flush=True,
        )
    return {
        **common_provenance(args),
        "request_parameters": {
            "endpoint": "/v1/completions",
            "temperature": 0.0,
            "top_p": 1.0,
            "logprobs": 5,
            "ignore_eos": True,
            "return_token_ids": True,
            "return_tokens_as_token_ids": True,
        },
        "cases": cases,
        "response_metadata": response_metadata,
    }


def benchmark_command(
    args: argparse.Namespace,
    tier: TierSpec,
    tier_seed: int,
    result_filename: str,
    raw_results_dir: Path,
    recorded_url: str,
    recorded_tokenizer: str,
) -> list[str]:
    return [
        "vllm",
        "bench",
        "serve",
        "--backend",
        "openai",
        "--endpoint",
        "/v1/completions",
        "--base-url",
        recorded_url,
        "--model",
        args.model_name,
        "--tokenizer",
        recorded_tokenizer,
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
        str(tier_seed),
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


def sanitize_raw(
    value: Any,
    actual_url: str,
    recorded_url: str,
    actual_tokenizer: str,
    recorded_tokenizer: str,
) -> Any:
    if isinstance(value, dict):
        return {
            key: sanitize_raw(
                item, actual_url, recorded_url, actual_tokenizer, recorded_tokenizer
            )
            for key, item in value.items()
        }
    if isinstance(value, list):
        return [
            sanitize_raw(
                item, actual_url, recorded_url, actual_tokenizer, recorded_tokenizer
            )
            for item in value
        ]
    if isinstance(value, str):
        return value.replace(actual_url, recorded_url).replace(
            actual_tokenizer, recorded_tokenizer
        )
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
    tier_seed: int,
    result_filename: str,
    raw_results_dir: Path,
) -> tuple[dict[str, Any], list[str]]:
    result_path = raw_results_dir / result_filename
    result_path.unlink(missing_ok=True)
    recorded_url = public_url(args.vllm_url)
    recorded_command = benchmark_command(
        args, tier, tier_seed, result_filename, Path("raw_vllm"), recorded_url, DEFAULT_MODEL_DIR
    )
    actual_command = benchmark_command(
        args, tier, tier_seed, result_filename, raw_results_dir, args.vllm_url, args.tokenizer_dir
    )
    subprocess.run(actual_command, check=True, timeout=args.bench_timeout_s)
    raw = json.loads(result_path.read_text(encoding="utf-8"))
    raw = sanitize_raw(
        raw, args.vllm_url, recorded_url, args.tokenizer_dir, DEFAULT_MODEL_DIR
    )
    atomic_write_json(result_path, raw)
    if int(raw["completed"]) != tier.num_prompts:
        raise RuntimeError(
            f"{result_filename} completed {raw['completed']} of {tier.num_prompts} prompts"
        )
    return raw, recorded_command


def collect_benchmark(args: argparse.Namespace) -> dict[str, Any]:
    check_health(args.vllm_url, args.request_timeout_s)
    active_tiers = SIX_TIERS if args.tier_suite == "six_tier" else IN512_OUT128_TIERS
    raw_results_dir = args.raw_results_dir
    if raw_results_dir is None:
        raw_results_dir = args.output_json.parent / f"phase3_vllm_raw_{args.tier_suite}"
    raw_results_dir.mkdir(parents=True, exist_ok=True)
    warmup = TierSpec("warmup", 512, 128, 32, 16)
    warmup_seed = args.seed + 900_000
    _, warmup_command = run_benchmark(
        args, warmup, warmup_seed, "warmup.json", raw_results_dir
    )
    rounds = []
    tier_results: dict[str, list[dict[str, Any]]] = {tier.name: [] for tier in active_tiers}
    commands: dict[str, list[str]] = {}
    for round_index in range(args.rounds):
        for tier_index, tier in enumerate(active_tiers):
            tier_seed = args.seed + round_index * 1_000 + tier_index
            filename = f"round{round_index}_{tier.name}.json"
            started = time.monotonic()
            raw, command = run_benchmark(
                args, tier, tier_seed, filename, raw_results_dir
            )
            metrics = extract_metrics(raw)
            result = {
                "round": round_index,
                "tier": tier.name,
                "seed": tier_seed,
                "input_len": tier.input_len,
                "output_len": tier.output_len,
                "num_prompts": tier.num_prompts,
                "max_concurrency": tier.max_concurrency,
                "completed": int(raw["completed"]),
                "raw_result_file": f"raw_vllm/{filename}",
                "metrics": metrics,
                "vllm_bench_command": shlex.join(command),
            }
            rounds.append(result)
            tier_results[tier.name].append(result)
            if round_index == 0:
                commands[tier.name] = command
            print(
                f"round={round_index} tier={tier.name} seed={tier_seed} "
                f"out={metrics['output_throughput_tok_s']:.2f} tok/s "
                f"p99_itl={metrics['p99_itl_ms']:.2f} ms "
                f"collector_wall={time.monotonic() - started:.1f}s",
                flush=True,
            )
    tiers = []
    for tier in active_tiers:
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
            "backend": "openai",
            "endpoint": "/v1/completions",
            "tier_suite": args.tier_suite,
            "dataset_name": "random",
            "random_range_ratio": 0.0,
            "ignore_eos": True,
            "temperature": 0.0,
            "prefix_caching_enabled": False,
            "seed_schedule": "base_seed + round_index * 1000 + tier_index (warmup uses base_seed + 900000)",
            "rounds": args.rounds,
            "schedule": "warmup once, then round-major interleaving across all tiers with distinct per-(round, tier) seeds",
            "warmup": {
                "seed": warmup_seed,
                "input_len": 512,
                "output_len": 128,
                "num_prompts": 32,
                "max_concurrency": 16,
                "raw_result_file": "raw_vllm/warmup.json",
                "vllm_bench_command": shlex.join(warmup_command),
            },
            "aggregate_std": f"population standard deviation across {args.rounds} rounds",
        },
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
