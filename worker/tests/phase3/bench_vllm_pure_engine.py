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

"""Pure-engine in-process vLLM token-ID benchmark aligned with zstep Mode B metrology.

Eliminates HTTP loopback, FastAPI/Uvicorn, prompt tokenization, incremental
detokenization, and SSE framing by driving ``vllm.AsyncLLMEngine`` (`AsyncLLM`)
directly in-process with ``TokensPrompt(prompt_token_ids=...)``,
``detokenize=False``, and ``RequestOutputKind.DELTA`` under the exact same
SplitMix64 prompt token sequences, FIFO concurrency semaphore, warmup schedule,
and ``(repeat, tier)`` seed isolation as ``ModeBServingEngine::run_benchmark_tier_with_seed``.
"""

from __future__ import annotations

import argparse
import asyncio
from dataclasses import dataclass
from datetime import datetime, timezone
import importlib.metadata
import json
import math
import os
from pathlib import Path
import subprocess
import sys
import time
from typing import Any

BENCHMARK_RANDOM_SEED: int = 0x9E37_79B9_7F4A_7C15
_MASK64: int = (1 << 64) - 1
DEFAULT_MODEL_DIR = "/models/DeepSeek-V4.1-Flash"


@dataclass(frozen=True)
class TierSpec:
    label: str
    concurrency: int
    num_prompts: int
    input_len: int
    output_len: int


IN512_OUT128_TIERS: tuple[TierSpec, ...] = (
    TierSpec("c1", 1, 16, 512, 128),
    TierSpec("c8", 8, 32, 512, 128),
    TierSpec("c16", 16, 64, 512, 128),
    TierSpec("c32", 32, 128, 512, 128),
)

SIX_TIERS: tuple[TierSpec, ...] = (
    TierSpec("in512_c1", 1, 8, 512, 256),
    TierSpec("in512_c16", 16, 48, 512, 256),
    TierSpec("in512_c128", 128, 256, 512, 256),
    TierSpec("in1024_c1", 1, 20, 1024, 256),
    TierSpec("in1024_c32", 32, 200, 1024, 256),
    TierSpec("in1024_c128", 128, 400, 1024, 256),
)

SUITES: dict[str, tuple[TierSpec, ...]] = {
    "in512_out128": IN512_OUT128_TIERS,
    "six_tier": SIX_TIERS,
}


def _splitmix64(x: int) -> int:
    x = (x + 0x9E37_79B9_7F4A_7C15) & _MASK64
    x = ((x ^ (x >> 30)) * 0xBF58_476D_1CE4_E5B9) & _MASK64
    x = ((x ^ (x >> 27)) * 0x94D0_49BB_1331_11EB) & _MASK64
    return (x ^ (x >> 31)) & _MASK64


def _rotl64(x: int, k: int) -> int:
    return (((x << k) & _MASK64) | (x >> (64 - k))) & _MASK64


def build_deterministic_random_prompts_seeded(
    num_prompts: int,
    input_len: int,
    seed: int,
) -> list[list[int]]:
    """Bit-exact match to ``zs_scheduler::serving::build_deterministic_random_prompts_seeded``."""
    seed &= _MASK64
    prompts: list[list[int]] = []
    state = _splitmix64(seed)
    for p in range(num_prompts):
        mix_arg = (seed + (((p + 1) * 0xBF58_476D_1CE4_E5B9) & _MASK64)) & _MASK64
        prompt_mix = _splitmix64(mix_arg)
        state = _rotl64(state, 17) ^ prompt_mix
        row: list[int] = []
        for _ in range(input_len):
            state = (
                state * 6_364_136_223_846_793_005 + 1_442_695_040_888_963_407
            ) & _MASK64
            hi32 = (state >> 32) & 0xFFFF_FFFF
            signed_hi32 = hi32 - 0x1_0000_0000 if hi32 >= 0x8000_0000 else hi32
            row.append(256 + (signed_hi32 % 65_000))
        prompts.append(row)
    return prompts


def _mean_us_as_ms(samples: list[int]) -> float:
    if not samples:
        return 0.0
    return (sum(samples) / len(samples)) / 1000.0


def _percentile_us_as_ms(samples: list[int], pct: int) -> float:
    if not samples:
        return 0.0
    ordered = sorted(samples)
    idx = (len(ordered) - 1) * pct // 100
    return ordered[idx] / 1000.0


def _aggregate_metric(samples: list[float]) -> dict[str, float]:
    mean = sum(samples) / len(samples)
    variance = sum((x - mean) ** 2 for x in samples) / len(samples)
    return {
        "mean": round(mean, 6),
        "std": round(math.sqrt(variance), 6),
        "min": round(min(samples), 6),
        "max": round(max(samples), 6),
    }


def _query_nvidia_smi() -> tuple[str, list[int], list[int]]:
    try:
        out = subprocess.check_output(
            [
                "nvidia-smi",
                "--query-gpu=driver_version,clocks.current.graphics,clocks.current.memory",
                "--format=csv,noheader,nounits",
            ],
            text=True,
            timeout=10,
        )
    except (OSError, subprocess.SubprocessError):
        return ("unknown", [], [])
    driver = "unknown"
    gfx_clocks: list[int] = []
    mem_clocks: list[int] = []
    for raw_line in out.strip().splitlines():
        parts = [p.strip() for p in raw_line.split(",")]
        if len(parts) >= 3:
            driver = parts[0]
            gfx_clocks.append(int(parts[1]))
            mem_clocks.append(int(parts[2]))
    return (driver, gfx_clocks, mem_clocks)


def _git_commit() -> str:
    env_sha = os.environ.get("ZSTEP_GIT_SHA", "").strip()
    if env_sha:
        return env_sha[:7]
    try:
        return subprocess.check_output(
            ["git", "rev-parse", "--short=7", "HEAD"],
            text=True,
            timeout=5,
        ).strip()
    except (OSError, subprocess.SubprocessError):
        return "unknown"


async def _run_tier(
    engine: Any,
    spec: TierSpec,
    seed: int,
    repeat: int,
    run_tag: str,
) -> dict[str, Any]:
    from vllm import SamplingParams, TokensPrompt
    from vllm.sampling_params import RequestOutputKind

    prompts = build_deterministic_random_prompts_seeded(
        spec.num_prompts,
        spec.input_len,
        seed,
    )
    sampling_params = SamplingParams(
        temperature=0.0,
        max_tokens=spec.output_len,
        ignore_eos=True,
        detokenize=False,
        output_kind=RequestOutputKind.DELTA,
    )
    sem = asyncio.Semaphore(spec.concurrency)
    ttfts_us: list[int] = []
    submit_ttfts_us: list[int] = []
    tpots_us: list[int] = []
    itls_us: list[int] = []

    wall_start_ns = time.perf_counter_ns()

    async def _one_request(req_idx: int, prompt_ids: list[int]) -> None:
        submit_ns = wall_start_ns
        async with sem:
            permit_ns = time.perf_counter_ns()
            req_id = f"{run_tag}-{spec.label}-r{repeat}-q{req_idx}"
            first_ns: int | None = None
            last_ns: int | None = None
            emitted = 0
            async for out in engine.generate(
                TokensPrompt(prompt_token_ids=prompt_ids),
                sampling_params,
                req_id,
            ):
                if not out.outputs:
                    continue
                n_new = len(out.outputs[0].token_ids)
                if n_new <= 0:
                    continue
                now_ns = time.perf_counter_ns()
                if first_ns is None:
                    first_ns = now_ns
                    ttfts_us.append(max(0, (now_ns - permit_ns) // 1_000))
                    submit_ttfts_us.append(max(0, (now_ns - submit_ns) // 1_000))
                    for _ in range(n_new - 1):
                        itls_us.append(0)
                else:
                    assert last_ns is not None
                    itls_us.append(max(0, (now_ns - last_ns) // 1_000))
                    for _ in range(n_new - 1):
                        itls_us.append(0)
                last_ns = now_ns
                emitted += n_new

            if emitted != spec.output_len:
                raise RuntimeError(
                    f"request {req_id} emitted {emitted} tokens, expected {spec.output_len}"
                )
            if emitted > 1 and first_ns is not None and last_ns is not None:
                span_us = max(0, (last_ns - first_ns) // 1_000)
                tpots_us.append(span_us // (emitted - 1))

    tasks = [
        asyncio.create_task(_one_request(idx, prompt_ids))
        for idx, prompt_ids in enumerate(prompts)
    ]
    await asyncio.gather(*tasks)

    wall_ns = time.perf_counter_ns() - wall_start_ns
    wall_secs = wall_ns / 1e9
    total_out_tokens = float(spec.num_prompts * spec.output_len)
    total_all_tokens = float(spec.num_prompts * (spec.input_len + spec.output_len))

    return {
        "label": spec.label,
        "repeat": repeat,
        "seed": seed,
        "concurrency": spec.concurrency,
        "num_prompts": spec.num_prompts,
        "input_len": spec.input_len,
        "output_len": spec.output_len,
        "wall_time_ms": round(wall_secs * 1000.0, 3),
        "output_throughput_tok_s": round(total_out_tokens / wall_secs, 6),
        "total_throughput_tok_s": round(total_all_tokens / wall_secs, 6),
        "mean_ttft_ms": round(_mean_us_as_ms(ttfts_us), 6),
        "median_ttft_ms": round(_percentile_us_as_ms(ttfts_us, 50), 6),
        "p99_ttft_ms": round(_percentile_us_as_ms(ttfts_us, 99), 6),
        "mean_submit_ttft_ms": round(_mean_us_as_ms(submit_ttfts_us), 6),
        "median_submit_ttft_ms": round(_percentile_us_as_ms(submit_ttfts_us, 50), 6),
        "p99_submit_ttft_ms": round(_percentile_us_as_ms(submit_ttfts_us, 99), 6),
        "mean_tpot_ms": round(_mean_us_as_ms(tpots_us), 6),
        "median_tpot_ms": round(_percentile_us_as_ms(tpots_us, 50), 6),
        "p99_tpot_ms": round(_percentile_us_as_ms(tpots_us, 99), 6),
        "mean_itl_ms": round(_mean_us_as_ms(itls_us), 6),
        "median_itl_ms": round(_percentile_us_as_ms(itls_us, 50), 6),
        "p99_itl_ms": round(_percentile_us_as_ms(itls_us, 99), 6),
    }


def _build_aggregates(
    tiers: tuple[TierSpec, ...],
    results_by_tier: list[list[dict[str, Any]]],
) -> list[dict[str, Any]]:
    aggregates: list[dict[str, Any]] = []
    metric_keys = (
        "output_throughput_tok_s",
        "total_throughput_tok_s",
        "mean_ttft_ms",
        "median_ttft_ms",
        "p99_ttft_ms",
        "mean_submit_ttft_ms",
        "median_submit_ttft_ms",
        "p99_submit_ttft_ms",
        "mean_tpot_ms",
        "p99_tpot_ms",
        "p99_itl_ms",
    )
    for spec, bucket in zip(tiers, results_by_tier, strict=True):
        entry: dict[str, Any] = {
            "label": spec.label,
            "repeats": len(bucket),
        }
        for key in metric_keys:
            entry[key] = _aggregate_metric([float(item[key]) for item in bucket])
        aggregates.append(entry)
    return aggregates


async def _run_benchmarks(args: argparse.Namespace) -> None:
    from vllm import AsyncEngineArgs, AsyncLLMEngine
    from vllm.usage.usage_lib import UsageContext

    suite_pairs: list[tuple[str, Path]] = []
    for item in args.suite_outputs:
        if "=" not in item:
            raise ValueError(f"Expected SUITE=PATH in --suite-output, got {item!r}")
        suite_name, out_path = item.split("=", 1)
        if suite_name not in SUITES:
            raise ValueError(f"Unknown suite {suite_name!r}; valid: {sorted(SUITES)}")
        suite_pairs.append((suite_name, Path(out_path)))

    init_t0 = time.perf_counter()
    engine_args = AsyncEngineArgs(
        model=args.model_dir,
        tensor_parallel_size=args.tp_size,
        dtype="bfloat16",
        max_model_len=args.max_model_len,
        max_num_seqs=args.max_num_seqs,
        max_num_batched_tokens=args.max_num_batched_tokens,
        gpu_memory_utilization=args.gpu_memory_utilization,
        engram_config={"cpu_offload": False},
        limit_mm_per_prompt={"image": 0},
        enable_prefix_caching=False,
        seed=args.engine_seed,
        enable_log_requests=False,
        disable_log_stats=True,
    )
    engine = AsyncLLMEngine.from_engine_args(
        engine_args,
        usage_context=UsageContext.OPENAI_API_SERVER,
    )
    init_wall_ms = round((time.perf_counter() - init_t0) * 1000.0, 2)
    print(
        f"[vllm-pure] Engine initialized in {init_wall_ms:.2f} ms",
        file=sys.stderr,
        flush=True,
    )

    try:
        vllm_version = importlib.metadata.version("vllm")
    except importlib.metadata.PackageNotFoundError:
        vllm_version = "unknown"

    try:
        for suite_idx, (suite_name, out_path) in enumerate(suite_pairs):
            tiers = SUITES[suite_name]
            if args.warmup_prompts > 0:
                warmup_spec = TierSpec(
                    label="warmup",
                    concurrency=min(args.warmup_prompts, 16),
                    num_prompts=args.warmup_prompts,
                    input_len=512,
                    output_len=64,
                )
                warmup_seed = (
                    BENCHMARK_RANDOM_SEED + 900_000 + suite_idx * 10_000
                ) & _MASK64
                w_res = await _run_tier(
                    engine,
                    warmup_spec,
                    warmup_seed,
                    0,
                    f"warmup-{suite_name}",
                )
                print(
                    f"[{suite_name} warmup] prompts={w_res['num_prompts']} -> "
                    f"out_tok/s={w_res['output_throughput_tok_s']:.2f}, "
                    f"median_ttft={w_res['median_ttft_ms']:.2f}ms",
                    file=sys.stderr,
                    flush=True,
                )

            driver, gfx_clocks, mem_clocks = _query_nvidia_smi()
            tier_results: list[dict[str, Any]] = []
            results_by_tier: list[list[dict[str, Any]]] = [[] for _ in tiers]

            for repeat in range(args.repeats):
                for tier_idx, spec in enumerate(tiers):
                    tier_seed = (
                        BENCHMARK_RANDOM_SEED + repeat * 1_000 + tier_idx
                    ) & _MASK64
                    res = await _run_tier(
                        engine,
                        spec,
                        tier_seed,
                        repeat,
                        suite_name,
                    )
                    print(
                        f"[{suite_name}:{res['label']} repeat={repeat}] "
                        f"conc={res['concurrency']} prompts={res['num_prompts']} "
                        f"in={res['input_len']} out={res['output_len']} -> "
                        f"out_tok/s={res['output_throughput_tok_s']:.2f}, "
                        f"total_tok/s={res['total_throughput_tok_s']:.2f}, "
                        f"median_ttft={res['median_ttft_ms']:.2f}ms, "
                        f"mean_tpot={res['mean_tpot_ms']:.2f}ms, "
                        f"p99_itl={res['p99_itl_ms']:.2f}ms",
                        file=sys.stderr,
                        flush=True,
                    )
                    results_by_tier[tier_idx].append(res)
                    tier_results.append(res)

            report = {
                "engine": "vllm-pure-engine",
                "vllm_version": vllm_version,
                "measurement_boundary": (
                    "in_process_async_llm_tokens_prompt_no_http_no_tokenizer_"
                    "no_detokenizer_no_sse"
                ),
                "model": Path(args.model_dir).name,
                "world_size": args.tp_size,
                "suite": suite_name,
                "init_wall_ms": init_wall_ms,
                "collection_timestamp_utc": datetime.now(timezone.utc).isoformat(),
                "engine_config": {
                    "tensor_parallel_size": args.tp_size,
                    "dtype": "bfloat16",
                    "max_model_len": args.max_model_len,
                    "max_num_seqs": args.max_num_seqs,
                    "max_num_batched_tokens": args.max_num_batched_tokens,
                    "usage_context": "OPENAI_API_SERVER",
                    "gpu_memory_utilization": args.gpu_memory_utilization,
                    "engram_config": {"cpu_offload": False},
                    "enable_prefix_caching": False,
                    "detokenize": False,
                    "prompt_format": "TokensPrompt(prompt_token_ids)",
                    "output_kind": "DELTA",
                    "nccl_p2p_disable": os.environ.get("NCCL_P2P_DISABLE", ""),
                    "nccl_p2p_level": os.environ.get("NCCL_P2P_LEVEL", ""),
                },
                "provenance": {
                    "git_commit": _git_commit(),
                    "nvidia_driver_version": driver,
                    "gpu_graphics_clocks_mhz": gfx_clocks,
                    "gpu_mem_clocks_mhz": mem_clocks,
                    "seed": BENCHMARK_RANDOM_SEED,
                    "engine_seed": args.engine_seed,
                    "repeats": args.repeats,
                    "warmup_prompts": args.warmup_prompts,
                },
                "tiers": tier_results,
                "tier_aggregates": _build_aggregates(tiers, results_by_tier),
            }
            out_path.parent.mkdir(parents=True, exist_ok=True)
            out_path.write_text(
                json.dumps(report, indent=2) + "\n",
                encoding="utf-8",
            )
            print(
                f"[vllm-pure] Wrote {suite_name} report -> {out_path}",
                file=sys.stderr,
                flush=True,
            )
    finally:
        engine.shutdown()


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model-dir", default=DEFAULT_MODEL_DIR)
    parser.add_argument("--tp-size", type=int, default=8)
    parser.add_argument("--max-model-len", type=int, default=32768)
    parser.add_argument("--max-num-seqs", type=int, default=128)
    parser.add_argument("--max-num-batched-tokens", type=int, default=8192)
    parser.add_argument("--gpu-memory-utilization", type=float, default=0.95)
    parser.add_argument("--engine-seed", type=int, default=20261112)
    parser.add_argument("--warmup-prompts", type=int, default=16)
    parser.add_argument("--repeats", type=int, default=3)
    parser.add_argument(
        "--suite-output",
        dest="suite_outputs",
        action="append",
        required=True,
        help="SUITE=OUTPUT_JSON pair (e.g. in512_out128=/path/out.json); repeatable.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    asyncio.run(_run_benchmarks(args))


if __name__ == "__main__":
    main()
