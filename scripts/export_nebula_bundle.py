#!/usr/bin/env python3
# Copyright (c) 2026 zstep Contributors
# SPDX-License-Identifier: MIT
"""Export a validated Docker + Nebula deployment bundle for zstep.

Validates an optional Evolith-approved policy JSON against
``zstep_worker.env_config.WorkerRuntimeConfig`` and writes:
  - ``policy.env``: ``ZSTEP_*`` environment overrides for Docker/Nebula
  - ``nebula-deployment.json``: payload for Nebula Gateway ``POST /models/load``
  - ``docker-build.sh``: reproducible ``docker build`` script for the target backend
"""

from __future__ import annotations

import argparse
import json
import shlex
import sys
from pathlib import Path
from typing import Any

REPO_ROOT = Path(__file__).resolve().parent.parent
WORKER_DIR = REPO_ROOT / "worker"
if str(WORKER_DIR) not in sys.path:
    sys.path.insert(0, str(WORKER_DIR))

from zstep_worker.env_config import WorkerRuntimeConfig  # noqa: E402
from zstep_worker.sail_config import SailRuntimeConfig  # noqa: E402

SAIL_POLICY_TO_ENV: dict[str, str] = SailRuntimeConfig.policy_env_map()

SAIL_POLICY_METADATA_KEYS: frozenset[str] = frozenset({"version"})

BACKEND_DOCKERFILE: dict[str, str] = {
    "sm120": "deploy/docker/Dockerfile.sm120",
    "sail": "deploy/docker/Dockerfile.sail",
}


def policy_to_env_map(
    backend: str,
    policy_path: Path | None,
) -> dict[str, str]:
    """Validate policy JSON (if provided) and return ``ZSTEP_*`` env mapping."""
    if backend not in BACKEND_DOCKERFILE:
        raise ValueError(f"Unsupported backend {backend!r}; expected one of {sorted(BACKEND_DOCKERFILE)}")

    env_map: dict[str, str] = {
        "ZSTEP_BACKEND": "sm120" if backend == "sm120" else "sail",
    }
    if policy_path is None:
        WorkerRuntimeConfig.from_env(env_map)
        return env_map

    raw = json.loads(policy_path.read_text(encoding="utf-8"))
    if not isinstance(raw, dict):
        raise ValueError(f"Policy file {policy_path} must contain a JSON object")

    if backend == "sail":
        unknown = sorted(
            k for k in raw if k not in SAIL_POLICY_TO_ENV and k not in SAIL_POLICY_METADATA_KEYS
        )
        if unknown:
            raise ValueError(f"Unknown SAIL policy keys: {unknown}")
        for key, env_key in SAIL_POLICY_TO_ENV.items():
            if key in raw:
                env_map[env_key] = str(raw[key])
    else:
        for key, val in raw.items():
            env_key = key if key.startswith("ZSTEP_") else f"ZSTEP_{key.upper()}"
            env_map[env_key] = str(val)

    # Validate the resulting env map against zstep's trusted env_config parser.
    WorkerRuntimeConfig.from_env(env_map)
    return env_map


def export_bundle(
    *,
    backend: str,
    image: str,
    model_uid: str,
    model_dir: str,
    output_dir: Path,
    policy_path: Path | None = None,
    world_size: int = 8,
    gpu_memory_utilization: float = 0.88,
    max_seqs: int = 64,
    speculative_method: str = "none",
    num_speculative_tokens: int = 0,
    enable_prefix_caching: bool = True,
    replicas: int = 1,
    base_image: str | None = None,
) -> dict[str, Path]:
    env_map = policy_to_env_map(backend, policy_path)
    output_dir.mkdir(parents=True, exist_ok=True)

    # 1. Write policy.env
    policy_env_path = output_dir / "policy.env"
    env_lines = [f"{k}={v}" for k, v in sorted(env_map.items())]
    policy_env_path.write_text("\n".join(env_lines) + "\n", encoding="utf-8")

    # 2. Write Nebula deployment manifest
    extra_args: list[str] = [
        "--world-size",
        str(world_size),
        "--gpu-memory-utilization",
        str(gpu_memory_utilization),
        "--max-seqs",
        str(max_seqs),
    ]
    if enable_prefix_caching:
        extra_args.append("--enable-prefix-caching")
    else:
        extra_args.append("--no-enable-prefix-caching")
    if speculative_method != "none":
        extra_args.extend(["--speculative-method", speculative_method])
        if num_speculative_tokens > 0:
            extra_args.extend(["--num-speculative-tokens", str(num_speculative_tokens)])

    deployment: dict[str, Any] = {
        "model_uid": model_uid,
        "model_name": model_dir,
        "engine_type": "zstep",
        "replicas": replicas,
        "docker_image": image,
        "hardware_backend": backend,
        "extra_args": extra_args,
        "env": env_map,
    }
    manifest_path = output_dir / "nebula-deployment.json"
    manifest_path.write_text(json.dumps(deployment, indent=2) + "\n", encoding="utf-8")

    # 3. Write reproducible docker-build.sh
    dockerfile = BACKEND_DOCKERFILE[backend]
    build_cmd = ["docker", "build", "-f", dockerfile, "-t", image]
    if base_image:
        build_cmd.extend(["--build-arg", f"BASE_IMAGE={base_image}"])
    if policy_path is not None and backend == "sail":
        try:
            rel_policy = str(policy_path.resolve().relative_to(REPO_ROOT))
        except ValueError:
            rel_policy = str(policy_path)
        build_cmd.extend(["--build-arg", f"POLICY_FILE={rel_policy}"])
    build_cmd.append(".")

    build_script = (
        "#!/usr/bin/env bash\n"
        "set -euo pipefail\n"
        f"# Reproducible zstep ({backend}) image build for Nebula\n"
        f"{shlex.join(build_cmd)}\n"
    )
    build_script_path = output_dir / "docker-build.sh"
    build_script_path.write_text(build_script, encoding="utf-8")
    build_script_path.chmod(0o755)

    return {
        "policy_env": policy_env_path,
        "nebula_deployment": manifest_path,
        "docker_build": build_script_path,
    }


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Export zstep Docker + Nebula deployment bundle.")
    parser.add_argument("--backend", required=True, choices=sorted(BACKEND_DOCKERFILE))
    parser.add_argument("--image", required=True, help="Target Docker image tag, e.g. zstep:sail-gen3")
    parser.add_argument("--model-uid", required=True, help="Nebula model UID")
    parser.add_argument("--model-dir", required=True, help="Model path inside /model or host mount")
    parser.add_argument("--output", required=True, type=Path, help="Output bundle directory")
    parser.add_argument("--policy", type=Path, default=None, help="Optional Evolith approved policy JSON")
    parser.add_argument("--world-size", type=int, default=8)
    parser.add_argument("--gpu-memory-utilization", type=float, default=0.88)
    parser.add_argument("--max-seqs", type=int, default=64)
    parser.add_argument("--speculative-method", default="none", choices=["none", "dspark", "mtp", "ngram"])
    parser.add_argument("--num-speculative-tokens", type=int, default=0)
    parser.add_argument("--no-prefix-caching", action="store_true")
    parser.add_argument("--replicas", type=int, default=1)
    parser.add_argument("--base-image", default=None)
    args = parser.parse_args(argv)

    written = export_bundle(
        backend=args.backend,
        image=args.image,
        model_uid=args.model_uid,
        model_dir=args.model_dir,
        output_dir=args.output,
        policy_path=args.policy,
        world_size=args.world_size,
        gpu_memory_utilization=args.gpu_memory_utilization,
        max_seqs=args.max_seqs,
        speculative_method=args.speculative_method,
        num_speculative_tokens=args.num_speculative_tokens,
        enable_prefix_caching=not args.no_prefix_caching,
        replicas=args.replicas,
        base_image=args.base_image,
    )
    print(json.dumps({k: str(v) for k, v in written.items()}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
