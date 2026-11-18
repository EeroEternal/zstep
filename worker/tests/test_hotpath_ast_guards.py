# Copyright (c) 2026 zstep Contributors
# SPDX-License-Identifier: MIT

"""Static AST guards for SM120 worker hot paths and runtime env configuration.

Enforces the invariants documented in `docs/10-rsi-evolution-lessons.md` and `#31`:
1. Zero `.item()`, `.cpu()`, or `dist.broadcast(...)` in model/speculative runners,
   and `.tolist()` only on explicit CPU staging buffers.
2. Zero `torch.cuda.synchronize()` or `dist.barrier(...)` inside per-step / per-draft
   execution methods (`launch`, `finish`, `execute`, `draft`, `_draft_batch`, etc.).
3. Zero scalar literals passed to `torch.where(..., out=<scalar>)` (CUDAGraph hazard).
4. Zero explicit `int32` casts passed as the `index` argument to `index_select` or
   `index_copy_` (PyTorch requires `int64` index tensors).
5. All `ZSTEP_*` `os.environ` lookups in `zstep_worker` live in `env_config.py`.
"""

from __future__ import annotations

import ast
from pathlib import Path

WORKER_PKG = Path(__file__).resolve().parents[1] / "zstep_worker"

HOTPATH_MODEL_FILES: tuple[str, ...] = (
    "models/deepseek_v41.py",
    "models/qwen35_moe.py",
    "spec_decode.py",
    "models/deepseek_v41_dspark.py",
    "models/qwen35_mtp.py",
)

STEP_EXECUTION_METHODS: frozenset[str] = frozenset(
    {
        "launch",
        "finish",
        "execute",
        "draft",
        "_draft_batch",
        "_run_draft_forward_and_markov",
        "_forward_gpu",
        "_run_decode_bucket_forward",
        "_run_prefill_varlen_forward",
    }
)

ALLOWED_CPU_STAGING_TOLIST_SUBSTRINGS: tuple[str, ...] = (
    "out_io_cpu",
    "draft_out_tokens_cpu",
)


def _attr_chain(node: ast.AST) -> str:
    parts: list[str] = []
    cur = node
    while isinstance(cur, ast.Attribute):
        parts.append(cur.attr)
        cur = cur.value
    if isinstance(cur, ast.Name):
        parts.append(cur.id)
        return ".".join(reversed(parts))
    return ""


def _is_explicit_int32_expr(node: ast.AST) -> bool:
    if not isinstance(node, ast.Call):
        return False
    if isinstance(node.func, ast.Attribute) and node.func.attr == "int" and not node.args:
        return True
    if isinstance(node.func, ast.Attribute) and node.func.attr == "to":
        for arg in node.args:
            if _attr_chain(arg) in {"torch.int32", "torch.int"}:
                return True
        for kw in node.keywords:
            if kw.arg == "dtype" and _attr_chain(kw.value) in {"torch.int32", "torch.int"}:
                return True
    for kw in node.keywords:
        if kw.arg == "dtype" and _attr_chain(kw.value) in {"torch.int32", "torch.int"}:
            return True
    return False


def scan_hotpath_source(source: str, rel_path: str) -> list[str]:
    """Scan a model/speculative runner source string for hot-path sync/copy hazards."""
    tree = ast.parse(source, filename=rel_path)
    violations: list[str] = []

    for node in ast.walk(tree):
        if not isinstance(node, ast.Call):
            continue
        chain = _attr_chain(node.func)
        if isinstance(node.func, ast.Attribute) and node.func.attr in {"item", "cpu"}:
            violations.append(
                f"{rel_path}:{node.lineno}: forbidden `.{node.func.attr}()` in runner module"
            )
        elif isinstance(node.func, ast.Attribute) and node.func.attr == "tolist":
            expr_src = ast.unparse(node.func.value)
            if not any(tok in expr_src for tok in ALLOWED_CPU_STAGING_TOLIST_SUBSTRINGS):
                violations.append(
                    f"{rel_path}:{node.lineno}: `.tolist()` on non-staging expression `{expr_src}`"
                )
        elif chain in {"dist.broadcast", "torch.distributed.broadcast"}:
            violations.append(f"{rel_path}:{node.lineno}: forbidden `{chain}(...)`")

    for fn in ast.walk(tree):
        if not isinstance(fn, ast.FunctionDef | ast.AsyncFunctionDef):
            continue
        if fn.name not in STEP_EXECUTION_METHODS:
            continue
        for sub in ast.walk(fn):
            if not isinstance(sub, ast.Call):
                continue
            chain = _attr_chain(sub.func)
            if chain in {
                "torch.cuda.synchronize",
                "dist.barrier",
                "torch.distributed.barrier",
            }:
                violations.append(
                    f"{rel_path}:{sub.lineno}: forbidden `{chain}()` inside `{fn.name}`"
                )

    return violations


def scan_tensor_api_hazards(source: str, rel_path: str) -> list[str]:
    """Scan any worker Python module for `torch.where(..., out=scalar)` and `int32` indexing."""
    tree = ast.parse(source, filename=rel_path)
    violations: list[str] = []

    for node in ast.walk(tree):
        if not isinstance(node, ast.Call):
            continue
        chain = _attr_chain(node.func)
        func_attr = node.func.attr if isinstance(node.func, ast.Attribute) else ""

        if chain == "torch.where" or func_attr == "where":
            for kw in node.keywords:
                if (
                    kw.arg == "out"
                    and isinstance(kw.value, ast.Constant)
                    and isinstance(kw.value.value, int | float | bool)
                ):
                    violations.append(
                        f"{rel_path}:{node.lineno}: scalar literal `out={kw.value.value!r}` in `where(...)`"
                    )

        if chain in {"torch.index_select", "torch.index_copy"}:
            idx_arg = node.args[2] if len(node.args) >= 3 else None
            if idx_arg is None:
                for kw in node.keywords:
                    if kw.arg == "index":
                        idx_arg = kw.value
                        break
            if idx_arg is not None and _is_explicit_int32_expr(idx_arg):
                violations.append(
                    f"{rel_path}:{node.lineno}: explicit int32 index in `{chain}(...)`"
                )
        elif func_attr in {"index_select", "index_copy_", "index_copy"}:
            idx_arg = node.args[1] if len(node.args) >= 2 else None
            if idx_arg is None:
                for kw in node.keywords:
                    if kw.arg == "index":
                        idx_arg = kw.value
                        break
            if idx_arg is not None and _is_explicit_int32_expr(idx_arg):
                violations.append(
                    f"{rel_path}:{node.lineno}: explicit int32 index in `.{func_attr}(...)`"
                )

    return violations


def scan_direct_zstep_environ_reads(source: str, rel_path: str) -> list[str]:
    """Ensure `ZSTEP_*` `os.environ` / `os.getenv` reads live only in `env_config.py`."""
    if rel_path == "env_config.py":
        return []
    tree = ast.parse(source, filename=rel_path)
    violations: list[str] = []
    for node in ast.walk(tree):
        if (
            isinstance(node, ast.Call)
            and _attr_chain(node.func) in {"os.environ.get", "os.environ.pop", "os.getenv"}
            and node.args
            and isinstance(node.args[0], ast.Constant)
            and isinstance(node.args[0].value, str)
            and node.args[0].value.startswith("ZSTEP_")
        ):
            call_name = _attr_chain(node.func)
            violations.append(
                f"{rel_path}:{node.lineno}: direct `{call_name}({node.args[0].value!r})` outside `env_config.py`"
            )
        elif (
            isinstance(node, ast.Subscript)
            and _attr_chain(node.value) == "os.environ"
            and isinstance(node.slice, ast.Constant)
            and isinstance(node.slice.value, str)
            and node.slice.value.startswith("ZSTEP_")
        ):
            violations.append(
                f"{rel_path}:{node.lineno}: direct `os.environ[{node.slice.value!r}]` outside `env_config.py`"
            )
        elif (
            isinstance(node, ast.Compare)
            and isinstance(node.left, ast.Constant)
            and isinstance(node.left.value, str)
            and node.left.value.startswith("ZSTEP_")
            and any(isinstance(op, ast.In | ast.NotIn) for op in node.ops)
            and any(_attr_chain(comp) == "os.environ" for comp in node.comparators)
        ):
            violations.append(
                f"{rel_path}:{node.lineno}: direct `{node.left.value!r} in os.environ` check outside `env_config.py`"
            )
    return violations


def test_hotpath_runner_modules_have_zero_sync_or_host_copy_hazards() -> None:
    violations: list[str] = []
    for rel in HOTPATH_MODEL_FILES:
        path = WORKER_PKG / rel
        assert path.is_file(), f"Missing guarded hot-path module: {path}"
        violations.extend(scan_hotpath_source(path.read_text(encoding="utf-8"), rel))
    assert violations == []


def test_all_worker_modules_pass_tensor_api_and_env_centralization_guards() -> None:
    violations: list[str] = []
    py_files = sorted(WORKER_PKG.rglob("*.py"))
    assert py_files, f"No Python files found under {WORKER_PKG}"
    for path in py_files:
        rel = path.relative_to(WORKER_PKG).as_posix()
        src = path.read_text(encoding="utf-8")
        violations.extend(scan_tensor_api_hazards(src, rel))
        violations.extend(scan_direct_zstep_environ_reads(src, rel))
    assert violations == []


def test_ast_guard_scanners_detect_synthetic_regressions() -> None:
    bad_hotpath = """
def launch(self, plan, header):
    n = plan.tokens_gpu.item()
    arr = plan.tokens_gpu.cpu().tolist()
    dist.broadcast(plan.tokens_gpu, src=0)
    torch.cuda.synchronize()
    return n, arr
"""
    hp_hits = scan_hotpath_source(bad_hotpath, "synthetic_runner.py")
    assert any(".item()" in h for h in hp_hits)
    assert any(".cpu()" in h for h in hp_hits)
    assert any(".tolist()" in h for h in hp_hits)
    assert any("dist.broadcast" in h for h in hp_hits)
    assert any("torch.cuda.synchronize()" in h for h in hp_hits)

    bad_tensor = """
y = torch.where(mask, x, 0.0, out=0)
z = weight.index_select(0, idx.int())
buf.index_copy_(0, slots.to(torch.int32), src)
"""
    t_hits = scan_tensor_api_hazards(bad_tensor, "synthetic_ops.py")
    assert len(t_hits) == 3

    bad_env = """
import os
x = os.environ.get("ZSTEP_FOO", "1")
y = os.environ["ZSTEP_BAR"]
z = os.getenv("ZSTEP_BAZ")
w = os.environ.pop("ZSTEP_QUX", None)
flag = "ZSTEP_CORGE" in os.environ
"""
    e_hits = scan_direct_zstep_environ_reads(bad_env, "models/synthetic.py")
    assert len(e_hits) == 5
    assert scan_direct_zstep_environ_reads(bad_env, "env_config.py") == []
