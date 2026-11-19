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
import re
import shutil
import subprocess
import tempfile
import textwrap

WORKER_PKG = Path(__file__).resolve().parents[1] / "zstep_worker"
CSRC_DIR = WORKER_PKG / "kernels_sm120" / "csrc"

HOTPATH_MODEL_FILES: tuple[str, ...] = (
    "models/deepseek_v41.py",
    "models/qwen35_moe.py",
    "sampler.py",
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
        "_apply_stochastic_sampling",
        "sample_tp8_logits",
        "splitmix64_uniform_f32",
        "reset_and_apply_penalties",
        "record_sampled",
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


def _extract_load_inline_functions(py_path: Path) -> list[str]:
    tree = ast.parse(py_path.read_text(encoding="utf-8"), filename=str(py_path))
    for node in ast.walk(tree):
        if not isinstance(node, ast.Call):
            continue
        func_name = node.func.id if isinstance(node.func, ast.Name) else ""
        if func_name != "load_inline":
            continue
        for kw in node.keywords:
            if kw.arg == "functions" and isinstance(kw.value, ast.List):
                return [
                    elt.value
                    for elt in kw.value.elts
                    if isinstance(elt, ast.Constant) and isinstance(elt.value, str)
                ]
    raise AssertionError(f"No load_inline(functions=[...]) call found in {py_path}")


def _strip_cpp_comments_and_strings(src: str) -> str:
    pattern = re.compile(
        r"//.*?$|/\*.*?\*/|\"(?:\\.|[^\"\\])*\"|'(?:\\.|[^'\\\n]){1,4}'",
        re.DOTALL | re.MULTILINE,
    )
    return pattern.sub(" ", src)


def _check_balanced_delimiters(src: str, label: str) -> None:
    clean = _strip_cpp_comments_and_strings(src)
    pairs = {"}": "{", ")": "(", "]": "["}
    opens = set(pairs.values())
    stack: list[str] = []
    for ch in clean:
        if ch in opens:
            stack.append(ch)
        elif ch in pairs:
            assert stack and stack[-1] == pairs[ch], f"Unbalanced `{ch}` in {label}"
            stack.pop()
    assert not stack, f"Unclosed delimiters {stack} in {label}"


def test_sm120_csrc_sources_have_balanced_delimiters_and_bound_entrypoints() -> None:
    specs = (
        ("kernels_sm120/shm_allreduce.py", "shm_allreduce.cpp", "shm_allreduce_kernels.cu"),
        ("kernels_sm120/qwen35_ops.py", "qwen35_ops.cpp", "qwen35_ops_kernels.cu"),
    )
    decl_re = re.compile(r"\bvoid\s+(launch_[A-Za-z0-9_]+_cuda)\s*\(")
    for py_rel, cpp_name, cu_name in specs:
        py_path = WORKER_PKG / py_rel
        cpp_path = CSRC_DIR / cpp_name
        cu_path = CSRC_DIR / cu_name
        assert cpp_path.is_file(), f"Missing {cpp_path}"
        assert cu_path.is_file(), f"Missing {cu_path}"
        cpp_src = cpp_path.read_text(encoding="utf-8")
        cu_src = cu_path.read_text(encoding="utf-8")
        _check_balanced_delimiters(cpp_src, cpp_name)
        _check_balanced_delimiters(cu_src, cu_name)

        exported_fns = _extract_load_inline_functions(py_path)
        assert exported_fns, f"Empty exported function list in {py_rel}"
        for fn_name in exported_fns:
            assert re.search(rf"\b{re.escape(fn_name)}\s*\(", cpp_src), (
                f"Exported function `{fn_name}` from {py_rel} not defined in {cpp_name}"
            )

        forward_cuda_fns = set(decl_re.findall(cpp_src))
        assert forward_cuda_fns, f"No `launch_*_cuda` declarations found in {cpp_name}"
        for cuda_fn in sorted(forward_cuda_fns):
            assert re.search(rf"\bvoid\s+{re.escape(cuda_fn)}\s*\(", cu_src), (
                f"Forward-declared `{cuda_fn}` in {cpp_name} missing from {cu_name}"
            )


def test_sm120_bfp7_csrc_device_functions_compile_and_pass_host_cxx() -> None:
    cxx = shutil.which("c++") or shutil.which("g++") or shutil.which("clang++")
    assert cxx is not None, "C++ compiler required for csrc BFP7 verification"
    cu_src = (CSRC_DIR / "shm_allreduce_kernels.cu").read_text(encoding="utf-8")
    start = cu_src.index("__device__ __forceinline__ uint4 pack_fp12_bfp7_words(")
    end = cu_src.index("__device__ __forceinline__ void unpack_to_bf16_pair(")
    snippet = cu_src[start:end]

    harness = (
        textwrap.dedent(
            """
            #undef NDEBUG
            #include <algorithm>
            #include <cassert>
            #include <cmath>
            #include <cstdint>
            #include <cstring>
            #include <limits>

            #define __device__
            #define __forceinline__ inline
            struct uint4 { uint32_t x, y, z, w; };
            inline uint4 make_uint4(uint32_t x, uint32_t y, uint32_t z, uint32_t w) {
                return {x, y, z, w};
            }
            inline uint32_t __float_as_uint(float f) {
                uint32_t u;
                std::memcpy(&u, &f, sizeof(u));
                return u;
            }
            inline float __uint_as_float(uint32_t u) {
                float f;
                std::memcpy(&f, &u, sizeof(f));
                return f;
            }
            inline int __float2int_rn(float f) { return static_cast<int>(std::rint(f)); }
            using std::fmaxf;
            using std::max;
            using std::min;
            """
        )
        + snippet
        + textwrap.dedent(
            """
            int main() {
                // 1) Small normal finite sub-block + zero sub-block (no inv_sml overflow)
                {
                    float v[16] = {0.f};
                    v[0] = 1.0e-34f;
                    v[1] = -1.0e-34f;
                    float acc[16] = {0.f};
                    unpack_accum_fp12_bfp7(pack_fp12_bfp7(v), acc);
                    for (int i = 0; i < 16; ++i) assert(!std::isnan(acc[i]));
                    for (int i = 8; i < 16; ++i) assert(acc[i] == 0.0f);
                    assert(std::fabs(acc[0] - 1.0e-34f) < 3.0e-36f);
                }
                // 2) Sub-threshold <= 1e-36 sub-block flushes to 0.0f
                {
                    float v[16] = {0.f};
                    for (int i = 0; i < 8; ++i) v[i] = 1.0e-37f;
                    v[8] = 1.0e-34f;
                    float acc[16] = {0.f};
                    unpack_accum_fp12_bfp7(pack_fp12_bfp7(v), acc);
                    for (int i = 0; i < 8; ++i) assert(acc[i] == 0.0f);
                    assert(std::fabs(acc[8] - 1.0e-34f) < 3.0e-36f);
                }
                // 3) Dual sub-block +-Inf and +Inf + (-Inf) -> NaN
                {
                    float v1[16] = {0.f}, v2[16] = {0.f};
                    v1[0] = std::numeric_limits<float>::infinity();
                    v1[8] = -std::numeric_limits<float>::infinity();
                    v2[0] = -std::numeric_limits<float>::infinity();
                    v2[3] = std::numeric_limits<float>::quiet_NaN();
                    float acc[16] = {0.f};
                    unpack_accum_fp12_bfp7(pack_fp12_bfp7(v1), acc);
                    unpack_accum_fp12_bfp7(pack_fp12_bfp7(v2), acc);
                    assert(std::isnan(acc[0]) && std::isnan(acc[3]));
                    assert(std::isinf(acc[8]) && acc[8] < 0);
                    assert(acc[1] == 0.0f && acc[4] == 0.0f);
                }
                return 0;
            }
            """
        )
    )
    with tempfile.TemporaryDirectory(prefix="zstep_bfp7_cxx_") as tmp:
        cpp_file = Path(tmp) / "test_bfp7.cpp"
        bin_file = Path(tmp) / "test_bfp7"
        cpp_file.write_text(harness, encoding="utf-8")
        subprocess.run(
            [
                cxx,
                "-std=c++17",
                "-O2",
                "-Wall",
                "-Wextra",
                "-Wno-unknown-pragmas",
                "-Werror",
                str(cpp_file),
                "-o",
                str(bin_file),
            ],
            check=True,
        )
        subprocess.run([str(bin_file)], check=True)
