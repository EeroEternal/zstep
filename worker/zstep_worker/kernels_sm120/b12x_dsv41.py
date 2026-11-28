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

"""Selective SM120 (`sm_120`) compute integrations for DeepSeek-V4.1-Flash.

Integrates the measured winning SM120 operator families into `zstep`'s
`DeepSeekV41StepRunner` while keeping `zstep`'s Rust SHM control plane, `shm_v5`
NUMA-local collectives (`M <= 320` + 100% CUDAGraph), `NCCL_LOC` (`M > 320`),
and vLLM 0.31.0's `mhc_shifted_post_pre` / `ParallelLMHead` / FlashInfer sparse
MLA+SWA untouched:

1. **Routed MoE (`40 layers`, `E=384, top_k=6, H=5120, I_tp8=288`, MXFP4 + UE8M0)**:
   - Unpadded `I_tp8 = 288` (`288 % 32 == 0`, vs vLLM's `384` padding) saves
     ~10 GiB VRAM per GPU and 25% wasted MMA compute.
   - Uses `b12x.moe.fused_moe` (`ActivationMode.A8`, `W13Layout("w31")`,
     `swiglu_limit=10.0`, `B12X_W4A8_TINY_DECODE=0`) for `1.50x-2.50x` decode
     CUDAGraph speedup across `M = 1..128`.
2. **Block32 `[32, 32] UE8M0` 2D FP8 Dense Linears (`210 layers`, `7 shapes`:
   `fused_wqa_wkv`, `wq_b`, `indexer.wq_b`, `wo_b`,
   `shared_experts.gate_up_proj`, `shared_experts.down_proj`, `engram.wkv`)**:
   - Uses `b12x.gemm.block_fp8_linear` (`2.25x-4.64x` decode CUDAGraph speedup
     including `K % 32 == 0` `5120 x 288` `shared_experts.down_proj`) with
     pre-resolved fast bindings and per-shape `156-SM` autotuned eager Cutlass
     hybrid crossover (`rows > 2048` on `1792 x 5120`, `rows > 512` on
     `3200 x 6144`), while retaining `PreparationSession` across the runner
     lifetime so autotuned plans and compiled caches are never evicted.
   - Leaves 3D BMM `.wo_a` on standard `ModelOptLinearMethod`.
3. **Fused BF16 Router GEMM + `sqrtsoftplus` Top-6 Gate (`sm12x_mega_gate`, `40 layers`)**:
   - Uses `flashinfer.experimental.sm12x_mega_gate` (`enable_pdl=False`, pre-compiled
     CuTe-DSL TVM-FFI schedules + shared static scratch/self-rearming barriers) for
     `M >= ZSTEP_MEGA_GATE_MIN_TOKENS` (default `1`), accelerating both CUDAGraph
     decode (`M >= 1`) and eager prefill (`1.44x-3.27x` faster across `M = 256..3072`).
4. **Fused Paged DSA Indexer (`sm120_dsa_indexer`, `8 layers`)**:
   - Uses `flashinfer.experimental.sm120_dsa_indexer._api.dsa_indexer_det_topk_paged`
     for paged decode (`3.0x-4.3x` faster across all 8 indexer layers in CUDAGraph
     decode), exploiting exact two-level candidate block identity whenever
     compressed context length `<= candidate_topk_blocks * candidate_block_size`
     (`2048 * 8 == 16384 == kRowSelKeys`) and falling back to two-level `candidate_blocks`
     above `16384` tokens on `compress_ratio == 1` layers. Keeps ragged prefill on
     stock `fp8_fp4_mqa_logits + top_k_per_row_prefill`.
"""

from __future__ import annotations

import ctypes
from dataclasses import dataclass
import hashlib
import logging
import math
import os
from pathlib import Path
import sys
import time
import types
from typing import Any
import weakref

import torch
import torch.distributed as dist

from zstep_worker.env_config import WorkerRuntimeConfig
from zstep_worker.kernels_sm120.topk_canon import canonicalize_prefill_topk

_LOG = logging.getLogger(__name__)

_DEFAULT_COMPAT_CANDIDATES: tuple[str, ...] = (
    "/usr/local/lib/python3.12/dist-packages/flashinfer/experimental/b12x/_compat",
)

SM120_DSA_CANDIDATE_COVER_TOKENS: int = 16384
DEFAULT_MEGA_GATE_MIN_TOKENS: int = 1

# Per-shape eager Cutlass crossover thresholds measured on RTX 6000D (`sm_120`,
# 156 SMs) with retained `PreparationSession` + `autotune=True`:
# - `fused_wqa_wkv` (`1792 x 5120`): `b12x` fast-bind wins for `M <= 2048`
#   (`125-131 us` vs Cutlass `165-176 us`), Cutlass wins at `M >= 3072` (`219 vs 299 us`).
# - `engram.wkv` (`3200 x 6144`): `b12x` fast-bind wins for `M <= 512`
#   (`127-143 us` vs `160 us`), Cutlass wins at `M >= 1024` (`170 vs 189 us`).
_CUTLASS_LARGE_M_THRESHOLDS: dict[tuple[int, int], int] = {
    (1792, 5120): 2048,
    (3200, 6144): 512,
}


@dataclass(frozen=True)
class B12xPrepareSummary:
    enabled: bool
    moe_layers: int
    linear_layers: int
    linear_shapes: int
    moe_scratch_bytes: int
    linear_scratch_bytes: int
    prepare_ms: float
    mega_gate_layers: int = 0
    mega_gate_schedules: int = 0
    mega_gate_min_tokens: int = 0
    dsa_decode_enabled: bool = False


@dataclass
class _Sm120MegaGateState:
    scratch_flat: torch.Tensor
    score_barriers_u64: torch.Tensor
    barriers_i32_flat: torch.Tensor
    ids_i64_buf: torch.Tensor
    img_mask_bool: torch.Tensor
    img_lt_bool: torch.Tensor
    img_zero_u8: torch.Tensor
    sentinel_logits: torch.Tensor
    dummy_f32: torch.Tensor
    dummy_u8: torch.Tensor
    dummy_i32: torch.Tensor
    dummy_i64_by_topk: dict[int, torch.Tensor]
    min_tokens: int
    max_tokens: int
    num_schedules: int
    patched_layers: int


def _compute_b12x_source_fingerprint() -> str:
    """Compute a short deterministic hash over loaded `b12x` sources to isolate disk caches."""
    b12x_mod = sys.modules.get("b12x")
    b12x_file = getattr(b12x_mod, "__file__", None) if b12x_mod is not None else None
    if not b12x_file:
        return "default"
    pkg_dir = Path(b12x_file).resolve().parent
    rel_files = (
        "__init__.py",
        "preparation/session.py",
        "gemm/block_fp8_linear.py",
        "gemm/_shared/block_fp8.py",
        "moe/fused_moe/__init__.py",
        "moe/fused_moe/api.py",
    )
    digest = hashlib.sha256()
    matched = 0
    for rel in rel_files:
        p = pkg_dir / rel
        if p.is_file():
            digest.update(rel.encode("utf-8"))
            digest.update(p.read_bytes())
            matched += 1
    return digest.hexdigest()[:10] if matched > 0 else "default"


def is_sm120_dsa_decode_available() -> bool:
    """Return True when SM120 fused paged DSA indexer decode is enabled and importable."""
    cfg = WorkerRuntimeConfig.from_env()
    require_exp = cfg.require_sm120_exp
    if not cfg.enable_sm120_dsa_decode:
        if require_exp:
            raise RuntimeError(
                "ZSTEP_REQUIRE_SM120_EXP=1 conflicts with ZSTEP_ENABLE_SM120_DSA_DECODE=0"
            )
        return False
    if not torch.cuda.is_available():
        if require_exp:
            raise RuntimeError("ZSTEP_REQUIRE_SM120_EXP=1 set, but CUDA is unavailable")
        return False
    major, minor = torch.cuda.get_device_capability()
    if (major, minor) != (12, 0):
        if require_exp:
            raise RuntimeError(
                f"ZSTEP_REQUIRE_SM120_EXP=1 requires sm_120 GPU, got sm_{major}{minor}"
            )
        return False
    try:
        from flashinfer.experimental.sm120_dsa_indexer._api import (  # noqa: F401
            dsa_indexer_det_topk_paged,
        )
        from flashinfer.utils import is_sm120f_supported

        dev = torch.device("cuda", torch.cuda.current_device())
        supported = bool(is_sm120f_supported(dev))
    except (ImportError, RuntimeError, AttributeError) as exc:
        if require_exp:
            raise RuntimeError(
                "ZSTEP_REQUIRE_SM120_EXP=1 set, but `flashinfer.experimental.sm120_dsa_indexer` "
                f"is unavailable: {exc}. Run `./scripts/prepare_sm120_overlay.sh` and mount "
                "the unpacked SM120 overlay into the container."
            ) from exc
        return False
    if not supported and require_exp:
        raise RuntimeError(
            "ZSTEP_REQUIRE_SM120_EXP=1 set, but `is_sm120f_supported` returned False"
        )
    return supported


def _prepend_pythonpath(entry: str) -> None:
    if not entry:
        return
    if entry not in sys.path:
        sys.path.insert(0, entry)
    cur = os.environ.get("PYTHONPATH", "")
    parts = [p for p in cur.split(os.pathsep) if p]
    if entry not in parts:
        os.environ["PYTHONPATH"] = (
            f"{entry}{os.pathsep}{cur}" if cur else entry
        )


def maybe_enable_b12x_for_dsv41() -> bool:
    """Configure environment & import paths for `b12x` on SM120 if available."""
    cfg = WorkerRuntimeConfig.from_env()
    require_b12x = cfg.require_b12x or cfg.require_sm120_exp
    explicit_enable = cfg.explicit_enable_b12x
    if not cfg.enable_b12x:
        if require_b12x:
            raise RuntimeError(
                "ZSTEP_REQUIRE_B12X=1 conflicts with ZSTEP_ENABLE_B12X=0"
            )
        return False
    if not torch.cuda.is_available():
        if require_b12x:
            raise RuntimeError("ZSTEP_REQUIRE_B12X=1 set, but CUDA is unavailable")
        return False
    major, minor = torch.cuda.get_device_capability()
    if (major, minor) != (12, 0):
        if require_b12x:
            raise RuntimeError(
                f"ZSTEP_REQUIRE_B12X=1 requires sm_120 GPU, got sm_{major}{minor}"
            )
        return False

    # Force `dynamic` tactic over `micro` for `M = 1, 2, 4` (`36.9 us` vs `117 us`).
    os.environ.setdefault("B12X_W4A8_TINY_DECODE", "0")

    custom_compat = cfg.b12x_compat_dir
    candidates = (
        (custom_compat, *_DEFAULT_COMPAT_CANDIDATES)
        if custom_compat
        else _DEFAULT_COMPAT_CANDIDATES
    )
    for cand in candidates:
        if cand and os.path.isdir(cand):
            _prepend_pythonpath(cand)

    try:
        import b12x  # noqa: F401
        from b12x.gemm import block_fp8_linear  # noqa: F401
        from b12x.moe import fused_moe  # noqa: F401
        from b12x.preparation import PreparationSession  # noqa: F401
    except ImportError as exc:
        msg = (
            "SM120 `b12x` overlay package could not be imported "
            f"(checked ZSTEP_B12X_COMPAT_DIR and {_DEFAULT_COMPAT_CANDIDATES}): {exc}. "
            "Mount the bundled SM120 overlay via "
            "`./scripts/prepare_sm120_overlay.sh --print-docker-args` "
            "(`worker/third_party/flashinfer_sm120_overlay_d0cbb476.tar.xz`), "
            "or unset ZSTEP_REQUIRE_B12X / ZSTEP_REQUIRE_SM120_EXP (or set ZSTEP_ENABLE_B12X=0)."
        )
        if require_b12x:
            raise RuntimeError(msg) from exc
        if explicit_enable and os.environ.get("RANK", "0") == "0":
            _LOG.warning("%s Falling back to standard ModelOpt kernels (b12x=off).", msg)
        return False

    # Ensure spawn-mode compiler workers launched by `PreparationSession` inherit
    # the exact parent directory of `b12x` via `PYTHONPATH`.
    b12x_file = getattr(sys.modules.get("b12x"), "__file__", None)
    if b12x_file:
        pkg_parent = str(Path(b12x_file).resolve().parent.parent)
        _prepend_pythonpath(pkg_parent)
    return True


def _align_up(val: int, align: int = 256) -> int:
    return ((val + align - 1) // align) * align


def _specs_nbytes(specs: Any) -> int:
    total = 0
    for sp in specs:
        elem_size = torch.empty((), dtype=sp.dtype).element_size()
        nbytes = int(math.prod(sp.shape)) * elem_size
        total = _align_up(total, 1024) + nbytes
    return total


def _slice_arena_for_specs(
    arena_u8: torch.Tensor, specs: Any
) -> list[torch.Tensor]:
    views: list[torch.Tensor] = []
    offset = 0
    for sp in specs:
        offset = _align_up(offset, 1024)
        elem_size = torch.empty((), dtype=sp.dtype).element_size()
        nbytes = int(math.prod(sp.shape)) * elem_size
        chunk = arena_u8[offset : offset + nbytes]
        views.append(chunk.view(sp.dtype).reshape(sp.shape))
        offset += nbytes
    return views


_get_raw_cuda_stream = getattr(torch._C, "_cuda_getCurrentRawStream", lambda d: torch.cuda.current_stream(d).cuda_stream)


def _build_moe_capi_launcher(prog: Any, ca: Any, oidx: int) -> tuple[Any, ...] | None:
    try:
        import cuda.bindings.driver as cuda
        from b12x._lib.utils import make_ptr
        res = prog._load() if hasattr(prog, "_load") else prog
        jf = getattr(res, "_executable", res)
        ea, nca = getattr(jf, "execution_args", None), len(ca)
        if ea is None or not hasattr(jf, "capi_func"):
            return None
        if getattr(jf, "_default_executor", None) is None:
            with jf._executor_lock:
                if jf._default_executor is None:
                    jf._default_executor = weakref.proxy(jf).to(None)
        ex = jf._default_executor
        if ex is None or getattr(ex, "_has_profiler", False):
            return None
        meta = ea._meta
        rect = list(ca if nca == meta.arg_count else ea.get_rectified_args(tuple(ca), {}))
        p_idx, i_idx = (0, 1, 2, oidx), (nca - 8, nca - 6, nca - 5, nca - 2)
        dyn_s = {*p_idx, *i_idx, nca - 1}
        for idx in range(meta.arg_count):
            if idx not in dyn_s and meta.numeric_flags[idx] and not hasattr(rect[idx], "__c_pointers__"):
                rect[idx] = meta.annotated_types[idx](rect[idx])
        aa, ab, u64_p, i32_p = list(rect), list(rect), [], []
        for k, ai in enumerate(p_idx):
            va, vb, p = 0x11000000 + (k + 1) * 0x10000, 0x55000000 + (k + 1) * 0x10000, ca[ai]
            aa[ai] = make_ptr(p._dtype, va, p._addr_space, assumed_align=p._assumed_align)
            ab[ai] = make_ptr(p._dtype, vb, p._addr_space, assumed_align=p._assumed_align)
            u64_p.append((va, vb))
        aa[nca - 1], ab[nca - 1] = cuda.CUstream(0x33001000), cuda.CUstream(0x77002000)
        u64_p.append((0x33001000, 0x77002000))
        for k, ai in enumerate(i_idx):
            va, vb, ann = 0x135700 + k + 1, 0x246800 + k + 1, meta.annotated_types[ai]
            aa[ai], ab[ai] = ann(va), ann(vb)
            i32_p.append((va, vb))
        (exe_a, ad_a), (exe_b, _) = ea.generate_execution_args(tuple(aa), {}), ea.generate_execution_args(tuple(ab), {})
        pk_a, pk_b = list(ex._get_invoke_packed_args(exe_a)), list(ex._get_invoke_packed_args(exe_b))
        if len(exe_a) != len(exe_b) or len(pk_a) != len(pk_b):
            return None
        packed = (ctypes.c_void_p * len(pk_a))(*[0 if x is None else x for x in pk_a])
        u64_buf, i32_buf, used, nb = (ctypes.c_uint64 * 5)(), (ctypes.c_int32 * 4)(), set(), len(pk_a)
        for pairs, ctp, base, stride in ((u64_p, ctypes.c_uint64, ctypes.addressof(u64_buf), 8), (i32_p, ctypes.c_int32, ctypes.addressof(i32_buf), 4)):
            for k, (va, vb) in enumerate(pairs):
                h = [s for s in range(nb) if s not in used and pk_a[s] and pk_b[s] and ctp.from_address(pk_a[s]).value == va and ctp.from_address(pk_b[s]).value == vb]
                if len(h) != 1:
                    return None
                used.add(h[0])
                packed[h[0]] = base + k * stride
        err = ex.cuda_result if getattr(ex, "_has_cuda_result", False) else None
        return u64_buf, i32_buf, packed, ex.capi_func, err, (res, jf, ex, tuple(aa), tuple(ad_a), tuple(exe_a))
    except Exception:
        return None


def install_b12x_dsv41_quant_patches() -> bool:
    """Patch `DeepseekV4FP8Config.get_quant_method` to dispatch MoE & Block32 FP8 to `b12x`."""
    if not maybe_enable_b12x_for_dsv41():
        return False

    from b12x.gemm import block_fp8_linear
    from b12x.gemm._shared.block_fp8 import (
        BlockFP8LinearBinding,
        _block_fp8_linear_x_q_from_scratch,
    )
    from b12x.moe import fused_moe
    from vllm.config.quantization import QuantSpec
    from vllm.model_executor.layers.fused_moe import (
        RoutedExperts,
        UnquantizedFusedMoEMethod,
    )
    from vllm.model_executor.layers.fused_moe.config import FusedMoEParallelConfig
    from vllm.model_executor.layers.linear import (
        LinearBase,
        UnquantizedLinearMethod,
        WEIGHT_LOADER_V2_SUPPORTED,
    )
    import vllm.models.deepseek_v41.quant_config as dsv41_qcfg
    from vllm.model_executor.layers.quantization.modelopt import (
        CkptCtx,
        ModelOptLinearMethod,
        ModelOptNvFp4FusedMoE,
    )
    from vllm.model_executor.layers.quantization.mxfp4 import Mxfp4MoEMethod
    from vllm.model_executor.layers.quantization.utils.quant_utils import (
        is_layer_skipped,
        kMxfp8Dynamic,
        kMxfp8Static,
    )
    from vllm.model_executor.utils import replace_parameter

    cfg_cls = dsv41_qcfg.DeepseekV4FP8Config
    if getattr(cfg_cls, "_zstep_b12x_patched", False):
        return True

    class ZstepB12xMxfp4MoEMethod(Mxfp4MoEMethod):
        """SM120 `b12x` unpadded (`I=288`) W4A8 MXFP4 RoutedExperts method."""

        def __init__(self, moe):
            super().__init__(moe)
            self._b12x_enabled: bool = False
            self._fallback_to_vllm: bool = False
            self._b12x_eplan: Any = None
            self._b12x_moe_state: Any = None
            self._b12x_scratch: tuple[torch.Tensor, ...] | None = None
            self._b12x_moe_arena: torch.Tensor | None = None
            self._b12x_moe_by_m: list[Any] = []
            self._b12x_moe_capi_by_kern: dict[tuple[int, int], Any] = {}
            self._b12x_dev_idx: int = 0

        @property
        def is_monolithic(self) -> bool:
            if not self._fallback_to_vllm:
                return False
            return super().is_monolithic

        @property
        def supports_internal_mk(self) -> bool:
            if not self._fallback_to_vllm:
                return False
            return super().supports_internal_mk

        @property
        def mk_can_overlap_shared_experts(self) -> bool:
            if not self._fallback_to_vllm:
                return False
            return super().mk_can_overlap_shared_experts

        @property
        def topk_indices_dtype(self) -> torch.dtype | None:
            if not self._fallback_to_vllm:
                return torch.int32
            return super().topk_indices_dtype

        @property
        def skip_forward_padding(self) -> bool:
            if not self._fallback_to_vllm:
                return False
            return super().skip_forward_padding

        @property
        def has_unpadded_output(self) -> bool:
            if not self._fallback_to_vllm:
                return False
            return super().has_unpadded_output

        def maybe_roundup_sizes(
            self,
            hidden_size: int,
            intermediate_size_per_partition: int,
            act_dtype: torch.dtype,
            moe_parallel_config: FusedMoEParallelConfig,
        ) -> tuple[int, int]:
            if (
                not self.moe.has_bias
                and not moe_parallel_config.use_ep
                and hidden_size % 256 == 0
                and intermediate_size_per_partition % 32 == 0
            ):
                return hidden_size, intermediate_size_per_partition
            return super().maybe_roundup_sizes(
                hidden_size,
                intermediate_size_per_partition,
                act_dtype,
                moe_parallel_config,
            )

        def process_weights_after_loading(self, layer: RoutedExperts) -> None:
            act_raw = getattr(layer, "activation", "silu")
            act_val = str(getattr(act_raw, "value", act_raw)).lower()
            nonlinearity = "silu" if "silu" in act_val else act_val
            if (
                self.moe.has_bias
                or layer.use_ep
                or self.hidden_size % 256 != 0
                or self.intermediate_size % 32 != 0
                or nonlinearity != "silu"
            ):
                self._fallback_to_vllm = True
                super().process_weights_after_loading(layer)
                return

            device = layer.w13_weight.device
            swiglu_limit = getattr(layer, "swiglu_limit", None)
            i0 = self.intermediate_size
            eff_i = (i0 & ~127) + 64 if 0 < (i0 & 127) < 64 else i0
            w13, w2 = layer.w13_weight.data.contiguous(), layer.w2_weight.data.contiguous()
            s13, s2 = layer.w13_weight_scale.data.contiguous(), layer.w2_weight_scale.data.contiguous()
            if eff_i != i0:
                def _pad_w13(t: torch.Tensor, cols: int) -> torch.Tensor:
                    u, o = t.view(torch.uint8), torch.zeros((self.num_experts, 2 * eff_i, cols), dtype=torch.uint8, device=device)
                    o[:, :i0, :], o[:, eff_i : eff_i + i0, :] = u[:, :i0, :], u[:, i0:, :]
                    return o.view(t.dtype)
                def _pad_w2(t: torch.Tensor, c0: int, c1: int) -> torch.Tensor:
                    u, o = t.view(torch.uint8), torch.zeros((self.num_experts, self.hidden_size, c1), dtype=torch.uint8, device=device)
                    o[:, :, :c0] = u
                    return o.view(t.dtype)
                w13, s13 = _pad_w13(w13, self.hidden_size // 2), _pad_w13(s13, self.hidden_size // 32)
                w2, s2 = _pad_w2(w2, i0 // 2, eff_i // 2), _pad_w2(s2, i0 // 32, eff_i // 32)
            wplan = fused_moe.plan_weights(
                source=fused_moe.PackedSource(
                    format=fused_moe.PackedSourceFormat("fp4_e8m0_k32"),
                    w13_layout=fused_moe.W13Layout("w31"),
                ),
                activation=fused_moe.ActivationSpec(
                    mode=fused_moe.ActivationMode.A8,
                    nonlinearity="silu",
                    io_dtype=torch.bfloat16,
                    swiglu_limit=float(swiglu_limit)
                    if swiglu_limit is not None
                    else None,
                ),
                geometry=fused_moe.MoEGeometry(
                    num_experts=self.num_experts,
                    hidden_size=self.hidden_size,
                    intermediate_size=eff_i,
                ),
            )
            unit_scale = torch.ones(
                self.num_experts, dtype=torch.float32, device=device
            )
            prepared = fused_moe.prepare_weights(
                plan=wplan,
                weights=fused_moe.PackedWeights(
                    w13=w13,
                    w2=w2,
                    w13_block_scales=s13,
                    w2_block_scales=s2,
                    w13_global_scales=unit_scale,
                    w2_global_scales=unit_scale,
                    input_scale=unit_scale,
                    intermediate_scale=unit_scale,
                    immutable_input_scales=True,
                ),
            )
            del w13, w2, s13, s2
            layer._b12x_prepared_experts = prepared
            self._b12x_enabled = True
            self.moe_kernel = None
            # Mark non-None so `RoutedExperts._ensure_moe_quant_config_init` is a no-op.
            self.moe_quant_config = True  # type: ignore[assignment]

            # Release source checkpoint tensors (~10 GiB saved across 40 layers per GPU).
            for pname in (
                "w13_weight",
                "w2_weight",
                "w13_weight_scale",
                "w2_weight_scale",
            ):
                param = getattr(layer, pname, None)
                if param is not None:
                    replace_parameter(
                        layer,
                        pname,
                        torch.empty((0,), dtype=param.dtype, device=device),
                    )
            torch.cuda.empty_cache()

        def apply(
            self,
            layer: RoutedExperts,
            x: torch.Tensor,
            topk_weights: torch.Tensor,
            topk_ids: torch.Tensor,
            shared_experts: Any,
            shared_experts_input: torch.Tensor | None,
        ) -> torch.Tensor:
            if not self._b12x_enabled:
                return super().apply(
                    layer=layer,
                    x=x,
                    topk_weights=topk_weights,
                    topk_ids=topk_ids,
                    shared_experts=shared_experts,
                    shared_experts_input=shared_experts_input,
                )
            orig_shape = x.shape
            x_2d = x.reshape(-1, orig_shape[-1])
            if x_2d.shape[0] == 0:
                return torch.empty_like(x)
            if x_2d.dtype != torch.bfloat16 or not x_2d.is_contiguous():
                x_2d = x_2d.to(dtype=torch.bfloat16).contiguous()
            wts = (
                topk_weights
                if topk_weights.dtype == torch.float32
                and topk_weights.is_contiguous()
                else topk_weights.to(dtype=torch.float32).contiguous()
            )
            ids = (
                topk_ids
                if topk_ids.dtype == torch.int32 and topk_ids.is_contiguous()
                else topk_ids.to(dtype=torch.int32).contiguous()
            )
            out_2d = torch.empty_like(x_2d)
            m, by_m = int(x_2d.shape[0]), self._b12x_moe_by_m
            entry = by_m[m] if m < len(by_m) else False
            if entry is None and not torch.cuda.is_current_stream_capturing():
                self._capture_fast_moe_m(layer, x_2d, wts, ids, out_2d, m)
                return out_2d.view(orig_shape)
            if entry:
                b_span, b_ep, pad_rows, mac, dev_i, m_u64, m_i32, m_pk, m_capi, m_err, _kp = entry
                b_span.zero_()
                if b_ep is not None:
                    b_ep.zero_()
                m_u64[0], m_u64[1], m_u64[2], m_u64[3], m_u64[4] = x_2d.data_ptr(), ids.data_ptr(), wts.data_ptr(), out_2d.data_ptr(), _get_raw_cuda_stream(dev_i)
                m_i32[0], m_i32[1], m_i32[2], m_i32[3] = m, m, pad_rows, mac
                m_capi(m_pk)
                if m_err is not None and m_err.value != 0:
                    raise RuntimeError(f"b12x dynamic MoE CUDA error {m_err.value}")
                return out_2d.view(orig_shape)
            if self._b12x_moe_state is not None:
                binding = self._b12x_moe_state.bind(
                    scratch=self._b12x_scratch,
                    a=x_2d,
                    experts=layer._b12x_prepared_experts,
                    topk_weights=wts,
                    topk_ids=ids,
                    output=out_2d,
                    input_scales_static=True,
                )
            else:
                binding = fused_moe.bind(
                    self._b12x_eplan,
                    scratch=self._b12x_scratch,
                    a=x_2d,
                    experts=layer._b12x_prepared_experts,
                    topk_weights=wts,
                    topk_ids=ids,
                    output=out_2d,
                    input_scales_static=True,
                )
            binding.run()
            return out_2d.view(orig_shape)

        def _capture_fast_moe_m(self, layer: RoutedExperts, x_2d: torch.Tensor, wts: torch.Tensor, ids: torch.Tensor, out_2d: torch.Tensor, m: int) -> Any:
            import b12x.moe.fused_moe._impl as moe_impl
            st, rec, orig_gdk = self._b12x_moe_state, [], moe_impl._get_dynamic_kernel
            vm, a16 = st.variants, st.a16_max_tokens
            vcap = (
                a16
                if (0 < m <= a16 and m not in vm)
                else min((v for v in vm if v >= m), default=max(vm))
            )
            binding = vm[vcap].bind(scratch=self._b12x_scratch, a=x_2d, experts=layer._b12x_prepared_experts, topk_weights=wts, topk_ids=ids, output=out_2d, input_scales_static=True)
            def _spy_gdk(*a: Any, **kw: Any) -> Any:
                comp, mac_v = orig_gdk(*a, **kw)
                ext_r = bool(kw.get("external_route_plan", False) or kw.get("deterministic_output", False))
                return (lambda *ca: (rec.append((comp, mac_v, ca, ext_r)), comp(*ca))[1]), mac_v
            moe_impl._get_dynamic_kernel = _spy_gdk
            try:
                binding.run()
            finally:
                moe_impl._get_dynamic_kernel = orig_gdk
            entry, ei = False, layer._b12x_prepared_experts._impl
            if len(rec) == 1 and binding.implementation == "dynamic" and ei.mxfp4_csf is None and ei.nvfp4_csf is None and not (ei.mxfp4_csf_inline is not None and not binding.w4a8_csf_inline):
                comp, mac_v, ca, ext_r = rec[0]
                nca, oidx = len(ca), len(ca) - 11
                if not ext_r and nca >= 20 and [getattr(ca[i], "_pointer", None) for i in (0, 1, 2, oidx)] == [x_2d.data_ptr(), ids.data_ptr(), wts.data_ptr(), out_2d.data_ptr()] and ca[nca - 8] == m and ca[nca - 6] == m:
                    ckey = (vcap, id(comp))
                    if ckey not in self._b12x_moe_capi_by_kern:
                        self._b12x_moe_capi_by_kern[ckey] = _build_moe_capi_launcher(comp, ca, oidx)
                    capi = self._b12x_moe_capi_by_kern[ckey]
                    if capi is not None:
                        bc, be, ar = binding.barrier_count, binding.barrier_epoch, self._b12x_moe_arena
                        b_span, b_ep = bc, be
                        if ar is not None and bc is not None and be is not None:
                            off0, end1 = bc.data_ptr() - ar.data_ptr(), (be.data_ptr() - ar.data_ptr()) + be.numel() * 4
                            if 0 <= off0 < end1 <= ar.numel() and 0 <= be.data_ptr() - (bc.data_ptr() + bc.numel() * 4) <= 16:
                                b_span, b_ep = ar.narrow(0, off0, end1 - off0), None
                        entry = (b_span, b_ep, int(ca[nca - 5]), int(mac_v), self._b12x_dev_idx, *capi)
            self._b12x_moe_by_m[m] = entry
            return entry

    class ZstepModelOptNvFp4B12xMoEMethod(ModelOptNvFp4FusedMoE):
        """SM120 `b12x` unpadded (`I=576` on TP=4) hybrid W4A16/W4A4 ModelOpt NVFP4 MoE method.

        Class name intentionally includes `'ModelOpt'` because `vllm`'s
        `RoutedExperts.weight_loader` dispatches dual-shard `w13_input_scale`
        and `uses_weight_scale_2_pattern()` via `'ModelOpt' in quant_method_name`.
        """

        def __init__(self, quant_config: Any, moe_config: Any):
            super().__init__(quant_config=quant_config, moe_config=moe_config)
            self._b12x_enabled: bool = False
            self._fallback_to_vllm: bool = False
            self._b12x_eplan: Any = None
            self._b12x_moe_state: Any = None
            self._b12x_scratch: tuple[torch.Tensor, ...] | None = None
            self._b12x_moe_arena: torch.Tensor | None = None
            self._b12x_moe_by_m: list[Any] = []
            self._b12x_moe_capi_by_kern: dict[tuple[int, int], Any] = {}
            self._b12x_dev_idx: int = 0

        @property
        def is_monolithic(self) -> bool:
            if not self._fallback_to_vllm:
                return False
            return super().is_monolithic

        @property
        def supports_internal_mk(self) -> bool:
            if not self._fallback_to_vllm:
                return False
            return super().supports_internal_mk

        @property
        def mk_can_overlap_shared_experts(self) -> bool:
            if not self._fallback_to_vllm:
                return False
            return super().mk_can_overlap_shared_experts

        @property
        def topk_indices_dtype(self) -> torch.dtype | None:
            if not self._fallback_to_vllm:
                return torch.int32
            return super().topk_indices_dtype

        @property
        def skip_forward_padding(self) -> bool:
            if not self._fallback_to_vllm:
                return False
            return super().skip_forward_padding

        @property
        def has_unpadded_output(self) -> bool:
            if not self._fallback_to_vllm:
                return False
            return super().has_unpadded_output

        def maybe_roundup_sizes(
            self,
            hidden_size: int,
            intermediate_size_per_partition: int,
            act_dtype: torch.dtype,
            moe_parallel_config: FusedMoEParallelConfig,
        ) -> tuple[int, int]:
            if (
                not self.moe.has_bias
                and not moe_parallel_config.use_ep
                and hidden_size % 256 == 0
                and intermediate_size_per_partition % 32 == 0
            ):
                return hidden_size, intermediate_size_per_partition
            return super().maybe_roundup_sizes(
                hidden_size,
                intermediate_size_per_partition,
                act_dtype,
                moe_parallel_config,
            )

        def create_weights(
            self,
            layer: RoutedExperts,
            num_experts: int,
            hidden_size: int,
            intermediate_size_per_partition: int,
            params_dtype: torch.dtype,
            **extra_weight_attrs: Any,
        ) -> None:
            from vllm.model_executor.layers.fused_moe.routed_experts import (
                FusedMoeWeightScaleSupported,
            )

            self.num_experts = int(num_experts)
            self.hidden_size = int(hidden_size)
            self.intermediate_size = int(intermediate_size_per_partition)
            super().create_weights(
                layer,
                num_experts,
                hidden_size,
                intermediate_size_per_partition,
                params_dtype,
                **extra_weight_attrs,
            )
            for pname in ("w13_weight_scale", "w2_weight_scale"):
                p = getattr(layer, pname, None)
                if p is not None:
                    p.quant_method = FusedMoeWeightScaleSupported.BLOCK.value
            for pname in (
                "w13_weight_scale_2",
                "w2_weight_scale_2",
                "w13_input_scale",
                "w2_input_scale",
            ):
                p = getattr(layer, pname, None)
                if p is not None:
                    p.quant_method = FusedMoeWeightScaleSupported.TENSOR.value

        def process_weights_after_loading(self, layer: RoutedExperts) -> None:
            cfg = WorkerRuntimeConfig.from_env()
            if cfg.nvfp4_act_mode == "a8":
                raise RuntimeError(
                    "ZSTEP_NVFP4_ACT_MODE='a8' (`w4a8_nvfp4`) is unsupported on SM120: "
                    "FlashInfer 0.7.6 `b12x` raises CUDA_ERROR_ILLEGAL_ADDRESS (700) in "
                    "dynamic m64 grouped preparation. Use ZSTEP_NVFP4_ACT_MODE='a4'."
                )
            act_raw = getattr(layer, "activation", "silu")
            act_val = str(getattr(act_raw, "value", act_raw)).lower()
            nonlinearity = "silu" if "silu" in act_val else act_val
            if (
                self.moe.has_bias
                or layer.use_ep
                or self.hidden_size % 256 != 0
                or self.intermediate_size % 64 != 0
                or nonlinearity != "silu"
            ):
                self._fallback_to_vllm = True
                super().process_weights_after_loading(layer)
                return

            device = layer.w13_weight.device
            swiglu_limit = getattr(layer, "swiglu_limit", None)
            w13 = layer.w13_weight.data.contiguous()
            w2 = layer.w2_weight.data.contiguous()
            s13 = layer.w13_weight_scale.data.contiguous()
            s2 = layer.w2_weight_scale.data.contiguous()
            g13 = layer.w13_weight_scale_2.data[:, 0].to(torch.float32).contiguous()
            g2 = layer.w2_weight_scale_2.data.reshape(-1).to(torch.float32).contiguous()
            a13_raw = layer.w13_input_scale.data
            if a13_raw.ndim == 2:
                a13_raw = a13_raw.max(dim=-1).values
            a13_recip = (1.0 / a13_raw.reshape(-1).to(torch.float32).clamp_min(1e-12)).contiguous()
            a2_recip = (
                1.0 / layer.w2_input_scale.data.reshape(-1).to(torch.float32).clamp_min(1e-12)
            ).contiguous()
            if a13_recip.numel() == 1:
                a13_recip = a13_recip.expand(self.num_experts).contiguous()
            if a2_recip.numel() == 1:
                a2_recip = a2_recip.expand(self.num_experts).contiguous()

            # Hybrid W4A16 (M <= 16 decode: avoids 2.6x-6.7x W4A4 small-M slowdown) +
            # W4A4 NVFP4 Tensor Cores (M > 16 prefill: 1.345x faster at M=4096).
            wplan = fused_moe.plan_weights(
                source=fused_moe.PackedSource(
                    format=fused_moe.PackedSourceFormat("modelopt_nvfp4"),
                    w13_layout=fused_moe.W13Layout("w31"),
                ),
                activation=fused_moe.ActivationSpec(
                    mode=fused_moe.ActivationMode.A4,
                    a16_max_tokens=16,
                    nonlinearity="silu",
                    io_dtype=torch.bfloat16,
                    swiglu_limit=float(swiglu_limit)
                    if swiglu_limit is not None
                    else None,
                ),
                geometry=fused_moe.MoEGeometry(
                    num_experts=self.num_experts,
                    hidden_size=self.hidden_size,
                    intermediate_size=self.intermediate_size,
                ),
            )
            prepared = fused_moe.prepare_weights(
                plan=wplan,
                weights=fused_moe.PackedWeights(
                    w13=w13,
                    w2=w2,
                    w13_block_scales=s13,
                    w2_block_scales=s2,
                    w13_global_scales=g13,
                    w2_global_scales=g2,
                    input_scale=a13_recip,
                    intermediate_scale=a2_recip,
                    immutable_input_scales=True,
                ),
            )
            del w13, w2, s13, s2, g13, g2, a13_recip, a2_recip
            layer._b12x_prepared_experts = prepared
            self._b12x_enabled = True
            self.moe_kernel = None
            self.moe_quant_config = True  # type: ignore[assignment]

            for pname in (
                "w13_weight",
                "w2_weight",
                "w13_weight_scale",
                "w2_weight_scale",
                "w13_weight_scale_2",
                "w2_weight_scale_2",
                "w13_input_scale",
                "w2_input_scale",
            ):
                param = getattr(layer, pname, None)
                if param is not None:
                    replace_parameter(
                        layer,
                        pname,
                        torch.empty((0,), dtype=param.dtype, device=device),
                    )
            torch.cuda.empty_cache()

        apply = ZstepB12xMxfp4MoEMethod.apply
        _capture_fast_moe_m = ZstepB12xMxfp4MoEMethod._capture_fast_moe_m

    ZstepB12xNvfp4MoEMethod = ZstepModelOptNvFp4B12xMoEMethod

    class ZstepB12xBlock32LinearMethod(ModelOptLinearMethod):
        """SM120 `b12x` `[32, 32] UE8M0` 2D FP8 linear method."""

        def __init__(
            self,
            spec: QuantSpec,
            ctx: CkptCtx,
            *,
            layer_prefix: str = "",
            format_scheme: Any = None,
        ) -> None:
            super().__init__(spec, ctx, format_scheme=format_scheme)
            self._layer_prefix: str = layer_prefix
            self._is_shared_expert: bool = ".shared_experts." in layer_prefix
            self._orig_kernel: Any = None
            self._cutlass_fallback_kernel: Any = None
            self._cutlass_large_m_threshold: int = 0
            self._b12x_enabled: bool = False
            self._b12x_caps: tuple[int, ...] = ()
            self._b12x_plans: tuple[Any, ...] = ()
            self._b12x_scratches: tuple[list[torch.Tensor], ...] = ()
            self._b12x_fast_plans: tuple[tuple[Any, ...], ...] = ()
            self._out_features: int = 0
            self._in_features: int = 0

        def create_weights(
            self,
            layer: torch.nn.Module,
            input_size_per_partition: int,
            output_partition_sizes: list[int],
            input_size: int,
            output_size: int,
            params_dtype: torch.dtype,
            **extra_weight_attrs: Any,
        ) -> None:
            super().create_weights(
                layer,
                input_size_per_partition,
                output_partition_sizes,
                input_size,
                output_size,
                params_dtype,
                **extra_weight_attrs,
            )
            # Disable `can_fuse_query_quant` (`F8_128x4` swizzled QuantizedActivation)
            # so `wq_b` and `indexer.wq_b` always receive contiguous BF16 tensors.
            self._orig_kernel = self.kernel
            self.kernel = None
            if hasattr(layer, "_input_quant_key"):
                delattr(layer, "_input_quant_key")

        def process_weights_after_loading(self, layer: torch.nn.Module) -> None:
            weight = getattr(layer, "weight", None)
            wscale = getattr(layer, "weight_scale", None)
            if (
                getattr(layer, "is_bmm", False)
                or weight is None
                or wscale is None
                or weight.ndim != 2
                or wscale.ndim != 2
            ):
                self.kernel = self._orig_kernel
                super().process_weights_after_loading(layer)
                return

            out_features = int(weight.shape[0])
            in_features = int(weight.shape[1])
            if (
                out_features % 32 != 0
                or in_features % 32 != 0
                or wscale.shape[1] != in_features // 32
                or wscale.shape[0] not in (out_features, out_features // 32)
            ):
                self.kernel = self._orig_kernel
                super().process_weights_after_loading(layer)
                return

            cutlass_threshold = _CUTLASS_LARGE_M_THRESHOLDS.get(
                (out_features, in_features), 0
            )
            keep_cutlass_fallback = cutlass_threshold > 0

            try:
                raw_scale_u8 = (
                    wscale.data[::32, :].contiguous()
                    if wscale.shape[0] == out_features
                    else wscale.data.contiguous()
                )
                w_src = (
                    weight.data.detach().clone().contiguous()
                    if keep_cutlass_fallback
                    else weight.data.contiguous()
                )
                packed = block_fp8_linear.pack_weight(
                    w_src,
                    raw_scale_u8.view(torch.float8_e8m0fnu),
                    block_size=(32, 32),
                )
            except (RuntimeError, ValueError, TypeError) as exc:
                _LOG.warning(
                    "b12x pack_weight fallback to ModelOpt on %s (%dx%d): %s",
                    self._layer_prefix,
                    out_features,
                    in_features,
                    exc,
                )
                self.kernel = self._orig_kernel
                super().process_weights_after_loading(layer)
                return

            if keep_cutlass_fallback:
                self.kernel = self._orig_kernel
                super().process_weights_after_loading(layer)
                self._cutlass_fallback_kernel = self.kernel
                self._cutlass_large_m_threshold = cutlass_threshold
                self.kernel = None
            else:
                device = weight.device
                replace_parameter(
                    layer,
                    "weight",
                    torch.empty((0,), dtype=weight.dtype, device=device),
                )
                replace_parameter(
                    layer,
                    "weight_scale",
                    torch.empty((0,), dtype=wscale.dtype, device=device),
                )

            layer._b12x_packed_weight = packed
            self._out_features = out_features
            self._in_features = in_features
            self._b12x_enabled = True

        def _select_plan_idx(self, rows: int) -> int:
            caps = self._b12x_caps
            for idx, cap in enumerate(caps):
                if cap >= rows:
                    return idx
            return len(caps) - 1

        def apply(
            self,
            layer: torch.nn.Module,
            x: torch.Tensor,
            bias: torch.Tensor | None = None,
        ) -> torch.Tensor:
            if not self._b12x_enabled:
                return super().apply(layer, x, bias)

            orig_shape = x.shape
            x_2d = x.reshape(-1, orig_shape[-1])
            rows = int(x_2d.shape[0])
            out_shape = (*orig_shape[:-1], self._out_features)
            if rows == 0:
                return torch.empty(
                    out_shape, dtype=torch.bfloat16, device=x.device
                )

            if (
                self._cutlass_fallback_kernel is not None
                and rows > self._cutlass_large_m_threshold
                and not torch.cuda.is_current_stream_capturing()
            ):
                out_fb = self._cutlass_fallback_kernel.apply_weights(
                    layer=layer, x=x_2d, bias=bias
                )
                return out_fb.view(out_shape)

            if x_2d.dtype != torch.bfloat16 or not x_2d.is_contiguous():
                x_2d = x_2d.to(dtype=torch.bfloat16).contiguous()

            pidx = self._select_plan_idx(rows)
            out_2d = torch.empty(
                (rows, self._out_features),
                dtype=torch.bfloat16,
                device=x_2d.device,
            )
            fast = self._b12x_fast_plans
            if fast:
                plan, state, scr_u8, ws, cap, mma_tiler_mn, exact_x_q = fast[pidx]
                x_q = (
                    exact_x_q
                    if rows == cap
                    else _block_fp8_linear_x_q_from_scratch(
                        scr_u8,
                        tokens=rows,
                        in_features=self._in_features,
                        output_dtype=torch.bfloat16,
                    )
                )
                binding = BlockFP8LinearBinding(
                    plan=plan,
                    source=x_2d,
                    packed_weight=layer._b12x_packed_weight,
                    x_q=x_q,
                    output=out_2d.view(rows, self._out_features, 1),
                    workspace=ws,
                    bias=None,
                    expected_m=cap,
                    mma_tiler_mn=mma_tiler_mn,
                )
                state.run_binding(binding)
            else:
                binding = block_fp8_linear.bind(
                    self._b12x_plans[pidx],
                    scratch=self._b12x_scratches[pidx],
                    source=x_2d,
                    packed_weight=layer._b12x_packed_weight,
                    output=out_2d.view(rows, self._out_features, 1),
                )
                block_fp8_linear.run(binding=binding)
            if bias is not None:
                out_2d.add_(bias)
            return out_2d.view(out_shape)

    # Register subclass for vLLM's `weight_loader_v2` dispatch.
    if ZstepB12xBlock32LinearMethod.__name__ not in WEIGHT_LOADER_V2_SUPPORTED:
        if hasattr(WEIGHT_LOADER_V2_SUPPORTED, "append"):
            WEIGHT_LOADER_V2_SUPPORTED.append(
                ZstepB12xBlock32LinearMethod.__name__
            )
        else:
            WEIGHT_LOADER_V2_SUPPORTED.add(
                ZstepB12xBlock32LinearMethod.__name__
            )

    orig_get_quant_method = cfg_cls.get_quant_method

    def _patched_get_quant_method(self, layer: torch.nn.Module, prefix: str):
        if (
            isinstance(layer, LinearBase)
            and self.weight_block_size == [32, 32]
            and self.is_scale_e8m0
        ):
            if is_layer_skipped(
                prefix=prefix,
                ignored_layers=self.ignored_layers,
                fused_mapping=self.packed_modules_mapping,
                match_mode=self.ignored_layers_match_mode,
            ):
                return UnquantizedLinearMethod()
            # Keep 3D batched matmul `.wo_a` on standard `ModelOptLinearMethod`.
            if prefix.endswith(".wo_a") or getattr(layer, "is_bmm", False):
                qm_bmm = orig_get_quant_method(self, layer, prefix)
                if prefix.endswith(".wo_a") and not getattr(qm_bmm, "_zstep_wo_a_wrapped", False):

                    def _wo_a_pwal(
                        lyr: torch.nn.Module,
                        _orig: Any = qm_bmm.process_weights_after_loading,
                    ) -> None:
                        w = getattr(lyr, "weight", None)
                        ws = getattr(lyr, "weight_scale", None)
                        if (
                            getattr(lyr, "bmm_batch_size", 0) == 1
                            and w is not None
                            and ws is not None
                            and w.shape == (1024, 4096)
                            and ws.ndim == 2
                        ):
                            ws_blk = (
                                ws.data[::32, :].contiguous()
                                if ws.shape[0] == 1024
                                else ws.data.contiguous()
                            )
                            lyr._b12x_wo_a_packed = block_fp8_linear.pack_weight(
                                w.data.contiguous(),
                                ws_blk.view(torch.float8_e8m0fnu),
                                block_size=(32, 32),
                            )
                        _orig(lyr)

                    qm_bmm.process_weights_after_loading = _wo_a_pwal
                    qm_bmm._zstep_wo_a_wrapped = True
                return qm_bmm
            rows, cols = self.weight_block_size
            return ZstepB12xBlock32LinearMethod(
                QuantSpec(weight=kMxfp8Static, activation=kMxfp8Dynamic),
                CkptCtx(scale_block_size=(rows, cols)),
                layer_prefix=prefix,
            )

        if isinstance(layer, RoutedExperts):
            if is_layer_skipped(
                prefix=prefix,
                ignored_layers=self.ignored_layers,
                fused_mapping=self.packed_modules_mapping,
            ):
                return UnquantizedFusedMoEMethod(layer.moe_config)
            if self.expert_dtype == "fp4":
                if self.moe_quant_algo == "NVFP4":
                    return ZstepB12xNvfp4MoEMethod(
                        quant_config=self._get_nvfp4_config(),
                        moe_config=layer.moe_config,
                    )
                return ZstepB12xMxfp4MoEMethod(layer.moe_config)

        return orig_get_quant_method(self, layer, prefix)

    cfg_cls.get_quant_method = _patched_get_quant_method
    cfg_cls._zstep_b12x_patched = True
    return True


def install_mxfp4_engram_patches() -> bool:
    """Idempotently patch `ParallelEngramEmbedding` to support `engram_embed_quant == 'mxfp4'`."""
    try:
        from vllm.config import get_current_vllm_config
        from vllm.models.deepseek_v41.common import engram as c_engram
        from vllm.models.deepseek_v41.nvidia import engram as nv_engram
        from vllm.triton_utils import tl, triton
    except ImportError:
        return False

    if getattr(c_engram, "_zstep_mxfp4_engram_patched", False):
        return True
    c_engram._zstep_mxfp4_engram_patched = True  # type: ignore[attr-defined]

    @triton.jit(
        do_not_specialize=[
            "vocab_start",
            "vocab_end",
            "num_rows",
            "ids_stride_t",
            "ids_stride_h",
            "GRID",
        ]
    )
    def _engram_mxfp4_lookup_kernel(
        weight_u8,
        scales_u8,
        ids,
        out,
        vocab_start,
        vocab_end,
        num_rows,
        ids_stride_t,
        ids_stride_h,
        HEAD_START: tl.constexpr,
        LOCAL_HEADS: tl.constexpr,
        TOTAL_HEADS: tl.constexpr,
        HALF_DIM: tl.constexpr,
        QUANT_BLOCK: tl.constexpr,
        BLOCK_R: tl.constexpr,
        GRID,
    ):
        half_cols = tl.arange(0, HALF_DIM)
        even_cols = half_cols * 2
        odd_cols = even_cols + 1
        scale_cols = even_cols // QUANT_BLOCK
        for base in tl.range(tl.program_id(0) * BLOCK_R, num_rows, GRID * BLOCK_R):
            rows = base + tl.arange(0, BLOCK_R)
            valid = rows < num_rows
            head = HEAD_START + rows % LOCAL_HEADS
            token = (rows // LOCAL_HEADS).to(tl.int64)
            index = tl.load(
                ids + token * ids_stride_t + head * ids_stride_h,
                mask=valid & (head < TOTAL_HEADS),
                other=-1,
            ).to(tl.int64)
            owned = valid & (head < TOTAL_HEADS) & (index >= vocab_start) & (index < vocab_end)
            local = tl.where(owned, index - vocab_start, 0)
            packed = tl.load(
                weight_u8 + local[:, None] * HALF_DIM + half_cols[None, :],
                mask=owned[:, None],
                other=0,
            ).to(tl.int32)
            sc_byte = tl.load(
                scales_u8 + local[:, None] * ((HALF_DIM * 2) // QUANT_BLOCK) + scale_cols[None, :],
                mask=owned[:, None],
                other=0,
            ).to(tl.int32)
            scale = (sc_byte << 23).to(tl.float32, bitcast=True)

            # Decode OCP FP4 E2M1 nibbles: mag_idx in 0..7 -> {0, 0.5, 1, 1.5, 2, 3, 4, 6}
            lo = packed & 0xF
            hi = (packed >> 4) & 0xF
            lo_m = lo & 0x7
            hi_m = hi & 0x7
            lo_abs = tl.where(
                lo_m <= 3,
                lo_m.to(tl.float32) * 0.5,
                tl.where(
                    lo_m <= 5,
                    (lo_m - 2).to(tl.float32),
                    tl.where(lo_m == 6, 4.0, 6.0),
                ),
            )
            hi_abs = tl.where(
                hi_m <= 3,
                hi_m.to(tl.float32) * 0.5,
                tl.where(
                    hi_m <= 5,
                    (hi_m - 2).to(tl.float32),
                    tl.where(hi_m == 6, 4.0, 6.0),
                ),
            )
            lo_val = tl.where((lo & 0x8) != 0, -lo_abs, lo_abs) * scale
            hi_val = tl.where((hi & 0x8) != 0, -hi_abs, hi_abs) * scale

            out_row_base = out + rows[:, None] * (HALF_DIM * 2)
            tl.store(out_row_base + even_cols[None, :], lo_val.to(tl.bfloat16), mask=valid[:, None])
            tl.store(out_row_base + odd_cols[None, :], hi_val.to(tl.bfloat16), mask=valid[:, None])

    def _is_mxfp4_engram_active() -> bool:
        try:
            hf_cfg = get_current_vllm_config().model_config.hf_config
            qcfg = getattr(hf_cfg, "quantization_config", None) or {}
            return str(qcfg.get("engram_embed_quant", "")).lower() == "mxfp4"
        except Exception:
            return False

    orig_base_alloc = c_engram.ParallelEngramEmbedding._allocate_weights
    orig_base_lookup = c_engram.ParallelEngramEmbedding.lookup
    orig_nv_alloc = nv_engram.ParallelEngramEmbedding._allocate_weights

    def _patched_base_alloc(self: Any) -> tuple[torch.Tensor, torch.Tensor]:
        if not _is_mxfp4_engram_active():
            return orig_base_alloc(self)
        self._zstep_is_mxfp4 = True
        return (
            torch.empty(self.part_num_embeddings, self.dim // 2, dtype=torch.uint8),
            torch.empty(
                self.part_num_embeddings, self.dim // self.block_size, dtype=torch.uint8
            ),
        )

    def _patched_nv_alloc(self: Any) -> tuple[torch.Tensor, torch.Tensor]:
        if not _is_mxfp4_engram_active():
            return orig_nv_alloc(self)
        self._zstep_is_mxfp4 = True
        if not self.cpu_offload:
            return _patched_base_alloc(self)
        half_dim = self.dim // 2
        if self.use_thp:
            weight_bytes = self.part_num_embeddings * half_dim
            scale_bytes = self.part_num_embeddings * (self.dim // self.block_size)
            packed = nv_engram._allocate_huge_page_storage(weight_bytes + scale_bytes)
            if packed is not None:
                self._packed = packed
                return (
                    packed[:weight_bytes].view(torch.uint8).view(-1, half_dim),
                    packed[weight_bytes:].view(torch.uint8).view(-1, self.dim // self.block_size),
                )
        return (
            torch.empty(
                self.part_num_embeddings,
                half_dim,
                dtype=torch.uint8,
                device="cpu",
                pin_memory=True,
            ),
            torch.empty(
                self.part_num_embeddings,
                self.dim // self.block_size,
                dtype=torch.uint8,
                device="cpu",
                pin_memory=True,
            ),
        )

    def _patched_base_lookup(
        self: Any, indices: torch.Tensor, out: torch.Tensor, background: bool = False
    ) -> None:
        if not getattr(self, "_zstep_is_mxfp4", False):
            return orig_base_lookup(self, indices, out, background=background)
        rows = indices.shape[0] * self.part_n_hash_cols
        if not rows:
            return
        weight, scales = self._storage()
        tiles = triton.cdiv(rows, 16)
        grid = min(tiles, self._num_sms // 2 if background else self._num_sms)
        _engram_mxfp4_lookup_kernel[(grid,)](
            weight,
            scales,
            indices,
            out,
            self.vocab_start_idx,
            self.vocab_end_idx,
            rows,
            indices.stride(0),
            indices.stride(1),
            HEAD_START=self.head_start,
            LOCAL_HEADS=self.part_n_hash_cols,
            TOTAL_HEADS=self.n_hash_cols,
            HALF_DIM=self.dim // 2,
            QUANT_BLOCK=self.block_size,
            BLOCK_R=16,
            GRID=grid,
        )

    c_engram.ParallelEngramEmbedding._allocate_weights = _patched_base_alloc
    c_engram.ParallelEngramEmbedding.lookup = _patched_base_lookup
    nv_engram.ParallelEngramEmbedding._allocate_weights = _patched_nv_alloc
    return True


def install_sm120_dsa_indexer_patches() -> bool:
    """Idempotently patch `SparseAttnIndexer.forward_cuda` to fuse paged decode on SM120."""
    if not is_sm120_dsa_decode_available():
        return False

    from flashinfer.experimental.sm120_dsa_indexer._api import (
        dsa_indexer_det_topk_paged,
    )
    from vllm import _custom_ops as ops
    from vllm.config import CUDAGraphMode
    from vllm.forward_context import get_forward_context
    from vllm.model_executor.kernels.attention.dsa.candidate_blocks import (
        apply_candidate_mask as _apply_candidate_mask,
    )
    from vllm.model_executor.kernels.attention.dsa.candidate_blocks import (
        select_candidate_blocks as _select_candidate_blocks,
    )
    import vllm.model_executor.layers.sparse_attn_indexer as sai
    from vllm.platforms import current_platform
    from vllm.utils.deep_gemm import fp8_fp4_mqa_logits
    from vllm.utils.torch_utils import _resolve_layer_name
    from vllm.v1.attention.ops.common import pack_seq_triton, unpack_seq_triton
    from vllm.v1.attention.ops.pcp import maybe_gather_indexer_k
    from vllm.v1.worker.workspace import current_workspace_manager

    indexer_cls = sai.SparseAttnIndexer
    if getattr(indexer_cls, "_zstep_sm120_dsa_patched", False):
        return True

    if WorkerRuntimeConfig.from_env().enable_topk_canon and not getattr(
        ops.top_k_per_row_prefill, "_zstep_topk_canon_patched", False
    ):
        orig_top_k_per_row_prefill = ops.top_k_per_row_prefill

        def _canonical_top_k_per_row_prefill(
            logits: torch.Tensor,
            row_starts: torch.Tensor,
            row_ends: torch.Tensor,
            indices: torch.Tensor,
            num_rows: int,
            stride0: int,
            stride1: int,
            top_k: int,
        ) -> None:
            orig_top_k_per_row_prefill(
                logits, row_starts, row_ends, indices, num_rows, stride0, stride1, top_k
            )
            canonicalize_prefill_topk(logits, row_starts, row_ends, indices, top_k)

        _canonical_top_k_per_row_prefill._zstep_topk_canon_patched = True  # type: ignore[attr-defined]
        ops.top_k_per_row_prefill = _canonical_top_k_per_row_prefill

    orig_forward_cuda = indexer_cls.forward_cuda
    gather_ws_shapes = sai._gather_workspace_shapes

    def _patched_forward_cuda(
        self,
        hidden_states: torch.Tensor,
        q_quant: torch.Tensor | tuple[torch.Tensor, torch.Tensor],
        k: torch.Tensor | None,
        weights: torch.Tensor,
    ) -> torch.Tensor:
        if (
            self.use_pcp
            or self.dcp_world_size > 1
            or self.head_dim != 128
            or not (1 <= self.topk_tokens <= 4096)
        ):
            return orig_forward_cuda(self, hidden_states, q_quant, k, weights)

        forward_context = get_forward_context()
        attn_metadata = forward_context.attn_metadata
        if not isinstance(attn_metadata, dict):
            return orig_forward_cuda(self, hidden_states, q_quant, k, weights)

        attn_metadata_narrowed = attn_metadata.get(self.k_cache.prefix)
        if attn_metadata_narrowed is None or attn_metadata_narrowed.num_decodes <= 0:
            return orig_forward_cuda(self, hidden_states, q_quant, k, weights)

        decode_metadata = attn_metadata_narrowed.decode
        if (
            decode_metadata is None
            or decode_metadata.global_seq_lens is not None
            or decode_metadata.indices is not None
        ):
            return orig_forward_cuda(self, hidden_states, q_quant, k, weights)

        compressed_max_seq = (
            int(attn_metadata_narrowed.max_seq_len) + self.compress_ratio - 1
        ) // self.compress_ratio
        candidate_blocks = self.candidate_blocks
        use_long_candidates = False
        if candidate_blocks is not None:
            cand_cover = int(candidate_blocks.shape[1]) * int(self.candidate_block_size)
            if compressed_max_seq > cand_cover:
                if (
                    int(candidate_blocks.shape[1]) != 2048
                    or int(self.candidate_block_size) not in (8, 16)
                    or compressed_max_seq > 65536 * int(self.candidate_block_size)
                ):
                    return orig_forward_cuda(self, hidden_states, q_quant, k, weights)
                use_long_candidates = True

        if isinstance(q_quant, tuple):
            q_values, q_scale = q_quant
        else:
            q_values, q_scale = q_quant, None

        if q_values.shape[-2] not in (32, 64):
            return orig_forward_cuda(self, hidden_states, q_quant, k, weights)

        num_decode_tokens = attn_metadata_narrowed.num_decode_tokens
        decode_lens = decode_metadata.decode_lens
        needs_padded_path = decode_metadata.requires_padding or (
            num_decode_tokens % decode_lens.shape[0] != 0
        )
        if num_decode_tokens == 0:
            est_next_n = 1
        elif needs_padded_path:
            # Check max decode length via metadata upper bound before mutating kv_cache.
            est_next_n = int(getattr(decode_metadata, "write_max_decode_len", 1))
        else:
            est_next_n = num_decode_tokens // decode_lens.shape[0]
        if not (1 <= est_next_n <= 16):
            return orig_forward_cuda(self, hidden_states, q_quant, k, weights)
        if use_long_candidates and (
            needs_padded_path
            or candidate_blocks is None
            or int(decode_lens.shape[0]) * est_next_n > int(candidate_blocks.shape[0])
        ):
            return orig_forward_cuda(self, hidden_states, q_quant, k, weights)

        raw_kv_cache = self.k_cache.kv_cache
        slot_mapping = attn_metadata_narrowed.slot_mapping
        has_prefill = attn_metadata_narrowed.num_prefills > 0
        topk_tokens = self.topk_tokens
        topk_indices_buffer = self.topk_indices_buffer

        num_tokens = slot_mapping.shape[0]
        k_local = k[:num_tokens] if k is not None else None
        if not self.skip_k_cache_insert:
            assert k_local is not None and self.scale_fmt is not None
            k_ins, slot_mapping_for_cache = maybe_gather_indexer_k(
                k_local, slot_mapping, num_decode_tokens, False
            )
            ops.indexer_k_quant_and_cache(
                k_ins,
                raw_kv_cache,
                slot_mapping_for_cache,
                self.quant_block_size,
                self.scale_fmt,
            )

        if forward_context.cudagraph_runtime_mode != CUDAGraphMode.FULL:
            dense_mha_layer = _resolve_layer_name(self.dense_mha_metadata_layer_name)
            if dense_mha_layer:
                mla_metadata = attn_metadata.get(dense_mha_layer)
                mla_prefill = getattr(mla_metadata, "prefill", None)
                if (
                    getattr(mla_prefill, "use_dense_mha", False)
                    and getattr(mla_metadata, "num_decode_tokens", -1) == 0
                    and not torch.cuda.is_current_stream_capturing()
                ):
                    return topk_indices_buffer

        if num_decode_tokens == 0:
            padded_q = q_values[:1].reshape(1, 1, *q_values.shape[1:])
            padded_q_scale = (
                q_scale[:1].reshape(1, 1, *q_scale.shape[1:])
                if q_scale is not None
                else None
            )
        elif needs_padded_path:
            if q_scale is not None:
                padded_q = pack_seq_triton(
                    q_values[:num_decode_tokens], decode_lens, pad_value=0
                )
                padded_q_scale = pack_seq_triton(
                    q_scale[:num_decode_tokens], decode_lens, pad_value=0
                )
            else:
                padded_q = pack_seq_triton(q_values[:num_decode_tokens], decode_lens)
                padded_q_scale = None
        else:
            padded_q = q_values[:num_decode_tokens].reshape(
                decode_lens.shape[0], -1, *q_values.shape[1:]
            )
            padded_q_scale = (
                q_scale[:num_decode_tokens].reshape(
                    decode_lens.shape[0], -1, *q_scale.shape[1:]
                )
                if q_scale is not None
                else None
            )

        batch_size = padded_q.shape[0]
        next_n = padded_q.shape[1]
        num_padded_tokens = batch_size * next_n

        # `dsa_indexer_det_topk_paged` writes every `[num_padded_tokens, topk_tokens]`
        # slot directly (filling `-1` for `i >= seq_len`, including `seq_len == 0`
        # padded CUDAGraph rows), so pure non-padded decode needs no prior `-1` fill.
        if (
            has_prefill
            or needs_padded_path
            or num_padded_tokens != hidden_states.shape[0]
            or topk_indices_buffer.shape[1] != topk_tokens
        ):
            topk_indices_buffer[: hidden_states.shape[0]] = -1

        if has_prefill:
            prefill_metadata = attn_metadata_narrowed.prefill
            assert prefill_metadata is not None
            fp8_dtype = current_platform.fp8_dtype()
            values_spec, scales_spec = gather_ws_shapes(
                self.max_total_seq_len, self.head_dim, fp8_dtype, self.use_fp4_cache
            )
            k_quant_full, k_scale_full = current_workspace_manager().get_simultaneous(
                values_spec, scales_spec
            )
            for chunk in prefill_metadata.chunks:
                cu_seqlen_ks = chunk.cu_seqlen_ks
                cu_seqlen_ke = chunk.cu_seqlen_ke
                k_quant = k_quant_full[: chunk.max_local_total_seq_lens]
                k_scale = k_scale_full[: chunk.max_local_total_seq_lens]
                if not chunk.skip_kv_gather and chunk.local_total_seq_lens > 0:
                    ops.cp_gather_indexer_k_quant_cache(
                        raw_kv_cache,
                        k_quant,
                        k_scale,
                        chunk.block_table,
                        chunk.local_cu_seq_lens,
                    )
                q_slice = q_values[chunk.token_start : chunk.token_end]
                q_scale_slice = (
                    q_scale[chunk.token_start : chunk.token_end]
                    if q_scale is not None
                    else None
                )
                prefill_topk = topk_indices_buffer[
                    chunk.token_start : chunk.token_end, :topk_tokens
                ]
                if chunk.local_total_seq_lens == 0 or q_slice.shape[0] == 0:
                    prefill_topk.fill_(-1)
                else:
                    if (
                        not self.use_fp4_cache
                        and candidate_blocks is not None
                        and not self.candidate_write
                    ):
                        from zstep_worker.kernels_sm120.candidate_mqa import (
                            run_candidate_sparse_prefill_topk,
                        )
                        chunk_candidates = candidate_blocks[
                            chunk.token_start : chunk.token_end
                        ]
                        k_scale_f32 = k_scale.view(torch.float32).squeeze(-1)
                        if run_candidate_sparse_prefill_topk(
                            q_slice=q_slice,
                            q_scale=q_scale_slice,
                            k_quant=k_quant,
                            k_scale=k_scale_f32,
                            weights=weights[chunk.token_start : chunk.token_end],
                            chunk_candidates=chunk_candidates,
                            cu_seqlen_ks=cu_seqlen_ks,
                            cu_seqlen_ke=cu_seqlen_ke,
                            prefill_topk=prefill_topk,
                            topk_tokens=topk_tokens,
                            candidate_block_size=self.candidate_block_size,
                            local_total_seq_lens=int(chunk.local_total_seq_lens),
                        ):
                            continue
                    if self.use_fp4_cache:
                        q_slice_cast = q_slice.view(torch.int8)
                        k_quant_cast = k_quant.view(torch.int8)
                        k_scale_cast = k_scale.view(torch.int32).squeeze(-1)
                    else:
                        q_slice_cast = q_slice
                        k_quant_cast = k_quant
                        k_scale_cast = k_scale.view(torch.float32).squeeze(-1)
                    logits = fp8_fp4_mqa_logits(
                        (q_slice_cast, q_scale_slice),
                        (k_quant_cast, k_scale_cast),
                        weights[chunk.token_start : chunk.token_end],
                        cu_seqlen_ks,
                        cu_seqlen_ke,
                        clean_logits=False,
                    )
                    num_rows = logits.shape[0]
                    if candidate_blocks is not None:
                        chunk_candidates = candidate_blocks[
                            chunk.token_start : chunk.token_end
                        ]
                        if self.candidate_write:
                            _select_candidate_blocks(
                                logits,
                                cu_seqlen_ks,
                                cu_seqlen_ke,
                                chunk_candidates.shape[1],
                                self.candidate_block_size,
                                chunk_candidates,
                            )
                        else:
                            _apply_candidate_mask(
                                logits,
                                cu_seqlen_ks,
                                cu_seqlen_ke,
                                chunk_candidates,
                                self.candidate_block_size,
                            )
                    ops.top_k_per_row_prefill(
                        logits,
                        cu_seqlen_ks,
                        cu_seqlen_ke,
                        prefill_topk,
                        num_rows,
                        logits.stride(0),
                        logits.stride(1),
                        topk_tokens,
                    )

        seq_lens = decode_metadata.seq_lens[:batch_size]
        block_table = decode_metadata.block_table[:batch_size]
        page_rows = int(raw_kv_cache.shape[1])
        bt_cap = int(block_table.shape[1]) * page_rows
        eff_max_ctx = min(bt_cap, max(compressed_max_seq, 1))
        decode_topk = topk_indices_buffer[:num_padded_tokens, :topk_tokens]

        cand_slice = candidate_blocks[:num_padded_tokens] if (use_long_candidates and candidate_blocks is not None) else None
        dsa_indexer_det_topk_paged(
            padded_q,
            raw_kv_cache,
            weights[:num_padded_tokens],
            seq_lens,
            block_table,
            top_k=topk_tokens,
            max_context_len=eff_max_ctx,
            page_rows=page_rows,
            q_scale=padded_q_scale,
            indices=decode_topk,
            return_scores=False,
            candidate_blocks=(None if self.candidate_write else cand_slice),
            candidate_out=(cand_slice if self.candidate_write else None),
            candidate_block_size=int(self.candidate_block_size),
        )

        if needs_padded_path:
            unpacked = unpack_seq_triton(
                decode_topk.reshape(batch_size, -1, decode_topk.shape[-1]),
                decode_lens,
            )
            topk_indices_buffer[: unpacked.shape[0], : unpacked.shape[-1]] = unpacked

        return topk_indices_buffer

    indexer_cls.forward_cuda = _patched_forward_cuda
    indexer_cls._zstep_sm120_dsa_patched = True
    return True


def _prepare_sm120_dsa_indexer(
    *,
    rank: int,
    device: torch.device,
    decode_buckets: tuple[int, ...],
    exchange_group: dist.ProcessGroup | None,
) -> bool:
    """Compile `sm120_dsa_indexer` on rank 0 first and pre-size decode workspace."""
    if not is_sm120_dsa_decode_available():
        return False
    install_sm120_dsa_indexer_patches()

    from flashinfer.experimental.sm120_dsa_indexer._api import (
        _decode_logits,
        _decode_long_ws,
        _module,
        _row_sel_keys,
    )

    if exchange_group is not None and dist.is_initialized():
        if rank == 0:
            _module()
        dist.barrier(group=exchange_group)
        if rank != 0:
            _module()
        dist.barrier(group=exchange_group)
    else:
        _module()

    max_rows = max((int(b) for b in decode_buckets if int(b) > 0), default=128)
    max_rows = max(max_rows, 256)
    _decode_logits(
        device, max_rows, max(int(_row_sel_keys()), SM120_DSA_CANDIDATE_COVER_TOKENS, 65536)
    )
    _decode_long_ws(device, max_rows, slot=0)
    _decode_long_ws(device, max_rows, slot=1)
    return True


def _prepare_sm12x_mega_gate(
    model: torch.nn.Module,
    *,
    rank: int,
    device: torch.device,
    max_tokens: int,
    exchange_group: dist.ProcessGroup | None,
) -> tuple[int, int, int]:
    """Pre-compile SM120 CuTe-DSL MegaGate schedules (rank 0 first) and patch eligible MoERunners."""
    cfg_env = WorkerRuntimeConfig.from_env()
    require_exp = cfg_env.require_sm120_exp
    if not cfg_env.enable_sm12x_mega_gate:
        if require_exp:
            raise RuntimeError(
                "ZSTEP_REQUIRE_SM120_EXP=1 conflicts with ZSTEP_ENABLE_SM12X_MEGA_GATE=0"
            )
        return 0, 0, 0
    min_tokens = cfg_env.mega_gate_min_tokens
    if min_tokens <= 0 or min_tokens > int(max_tokens):
        if require_exp:
            raise RuntimeError(
                f"ZSTEP_REQUIRE_SM120_EXP=1 set, but ZSTEP_MEGA_GATE_MIN_TOKENS={min_tokens} "
                f"is outside (0, {int(max_tokens)}]"
            )
        return 0, 0, 0
    if not torch.cuda.is_available() or torch.cuda.get_device_capability() != (12, 0):
        if require_exp:
            raise RuntimeError("ZSTEP_REQUIRE_SM120_EXP=1 requires an sm_120 CUDA device")
        return 0, 0, 0

    try:
        from flashinfer.experimental.sm12x_mega_gate.heuristics import (
            BLOCK_K,
            MAX_EXPERTS,
            MegaGateConfig,
            select_config,
        )
        from flashinfer.experimental.sm12x_mega_gate.mega_gate import (
            EXPERTS_PER_LANE,
            MAX_SLOTS,
            SCORE_BARRIER_WORDS,
            KernelSpec,
            device_facts,
            load_kernel,
        )
        import vllm.envs as vllm_envs
        from vllm.forward_context import (
            get_forward_context,
            is_forward_context_available,
        )

        arch, sms = device_facts(rank)
    except (ImportError, RuntimeError, AttributeError) as exc:
        if require_exp:
            raise RuntimeError(
                "ZSTEP_REQUIRE_SM120_EXP=1 set, but `flashinfer.experimental.sm12x_mega_gate` "
                f"is unavailable: {exc}. Run `./scripts/prepare_sm120_overlay.sh` and mount "
                "the unpacked SM120 overlay into the container."
            ) from exc
        if rank == 0:
            _LOG.warning("SM12x MegaGate unavailable on rank %d: %s", rank, exc)
        return 0, 0, 0

    check_skip_pad = bool(getattr(vllm_envs, "VLLM_MOE_SKIP_PADDING", False))

    eligible: list[
        tuple[
            Any,
            torch.Tensor,
            torch.Tensor | None,
            torch.Tensor | None,
            int,
            int,
            float,
        ]
    ] = []
    unique_geom: set[tuple[int, int, int, bool, bool]] = set()
    max_top_k = 0
    max_tok_cap = int(max_tokens)

    for mod in model.modules():
        if mod.__class__.__name__ != "MoERunner":
            continue
        if getattr(mod, "_zstep_mega_gate_patched", False):
            continue
        gate = getattr(mod, "gate", None)
        router = getattr(mod, "router", None)
        moe_cfg = getattr(mod, "moe_config", None)
        routed_exp = getattr(mod, "routed_experts", None)
        if gate is None or router is None or moe_cfg is None or routed_exp is None:
            continue
        qm = getattr(routed_exp, "quant_method", None)
        if qm is None or bool(getattr(qm, "is_monolithic", False)):
            continue
        eplb_state = getattr(router, "eplb_state", None)
        if (
            bool(getattr(mod, "_fse_fuse_gate", False))
            or bool(getattr(mod, "do_naive_dispatch_combine", False))
            or int(getattr(moe_cfg, "pcp_size", 1)) > 1
            or getattr(gate, "tid2eid", None) is not None
            or (
                eplb_state is not None
                and getattr(eplb_state, "logical_to_physical_map", None) is not None
            )
            or getattr(router, "_hash_indices_table", None) is not None
            or int(getattr(router, "num_fused_shared_experts", 0)) != 0
            or getattr(router, "scoring_func", None) != "sqrtsoftplus"
            or not bool(getattr(router, "renormalize", False))
        ):
            continue

        weight = getattr(gate, "weight", None)
        if (
            not isinstance(weight, torch.Tensor)
            or weight.ndim != 2
            or weight.dtype != torch.bfloat16
            or weight.device != device
            or not weight.is_contiguous()
            or weight.data_ptr() % 16 != 0
        ):
            continue
        num_experts, hidden = int(weight.shape[0]), int(weight.shape[1])
        if (
            hidden <= 0
            or hidden % BLOCK_K != 0
            or not (0 < num_experts <= MAX_EXPERTS)
            or num_experts % EXPERTS_PER_LANE != 0
        ):
            continue

        top_k = int(getattr(router, "top_k", 0))
        if not (0 < top_k <= min(num_experts, MAX_SLOTS)):
            continue

        routed_scale = float(getattr(router, "routed_scaling_factor", 1.0))
        if not math.isfinite(routed_scale):
            continue

        bias_param = getattr(router, "e_score_correction_bias", None)
        bias_t: torch.Tensor | None = None
        if bias_param is not None:
            bias_t = bias_param.data
            if (
                bias_t.ndim != 1
                or int(bias_t.shape[0]) != num_experts
                or bias_t.dtype != torch.float32
                or bias_t.device != device
                or not bias_t.is_contiguous()
                or bias_t.data_ptr() % 16 != 0
            ):
                continue

        img_sentinel_lo = int(getattr(router, "image_sentinel_lo", 0))
        bias_vl_param = getattr(router, "bias_vl", None)
        if bias_vl_param is None:
            bias_vl_param = getattr(gate, "bias_vl", None)
        bias_vl_t: torch.Tensor | None = None
        if bias_vl_param is not None and img_sentinel_lo > 0:
            bias_vl_t = bias_vl_param.data
            if (
                bias_vl_t.ndim != 1
                or int(bias_vl_t.shape[0]) != num_experts
                or bias_vl_t.dtype != torch.float32
                or bias_vl_t.device != device
                or not bias_vl_t.is_contiguous()
                or bias_vl_t.data_ptr() % 16 != 0
            ):
                continue

        eligible.append(
            (mod, weight, bias_t, bias_vl_t, img_sentinel_lo, top_k, routed_scale)
        )
        unique_geom.add(
            (hidden, num_experts, top_k, bias_t is not None, bias_vl_t is not None)
        )
        max_top_k = max(max_top_k, top_k)

    if not eligible:
        if require_exp:
            raise RuntimeError(
                "ZSTEP_REQUIRE_SM120_EXP=1 set, but SM12x MegaGate found 0 eligible MoERunner layers"
            )
        if rank == 0:
            _LOG.warning("SM12x MegaGate found 0 eligible MoERunner layers")
        return 0, 0, 0

    enable_pdl = bool(cfg_env.mega_gate_enable_pdl or str(arch).startswith("sm_120"))
    specs_to_compile: list[KernelSpec] = []
    geom_cfg_by_m: dict[
        tuple[int, int, int, bool, bool], list[MegaGateConfig | None]
    ] = {}
    max_scratch_elems = 16
    max_barrier_rows = 1

    for hidden, num_experts, top_k, has_bias, has_image_bias in sorted(unique_geom):
        cfgs_m: list[MegaGateConfig | None] = [None] * (max_tok_cap + 1)
        seen_cfgs: dict[MegaGateConfig, KernelSpec] = {}
        for m in range(min_tokens, max_tok_cap + 1):
            cfg = select_config(m, hidden, num_experts, sms, False)
            if (
                cfg.num_logical_ctas > sms
                or (hidden // BLOCK_K) % cfg.num_split_k != 0
            ) and sms > 110:
                cfg = select_config(m, hidden, num_experts, 110, False)
            if cfg.num_logical_ctas > sms or (hidden // BLOCK_K) % cfg.num_split_k != 0:
                continue
            cfgs_m[m] = cfg
            max_scratch_elems = max(
                max_scratch_elems, int(math.prod(cfg.scratch_shape(m, num_experts)))
            )
            max_barrier_rows = max(max_barrier_rows, int(cfg.num_token_blocks(m)))
            if cfg not in seen_cfgs:
                spec = KernelSpec(
                    config=cfg,
                    hidden=hidden,
                    num_experts=num_experts,
                    num_topk=top_k,
                    num_shared=0,
                    has_bias=has_bias,
                    has_image_bias=has_image_bias,
                    has_mask=False,
                    has_physical_map=False,
                    has_unmapped=False,
                    enable_pdl=enable_pdl,
                    has_fixed_routing=False,
                )
                seen_cfgs[cfg] = spec
                specs_to_compile.append(spec)
        geom_cfg_by_m[(hidden, num_experts, top_k, has_bias, has_image_bias)] = cfgs_m

    if not specs_to_compile:
        if require_exp:
            raise RuntimeError(
                "ZSTEP_REQUIRE_SM120_EXP=1 set, but SM12x MegaGate compiled 0 schedules"
            )
        return 0, 0, 0

    compiled_by_spec: dict[KernelSpec, Any] = {}

    def _compile_all() -> None:
        for sp in specs_to_compile:
            compiled_by_spec[sp] = load_kernel(arch, rank, sp)

    if exchange_group is not None and dist.is_initialized():
        if rank == 0:
            _compile_all()
        dist.barrier(group=exchange_group)
        if rank != 0:
            _compile_all()
        dist.barrier(group=exchange_group)
    else:
        _compile_all()

    scratch_flat = torch.empty(max_scratch_elems, dtype=torch.float32, device=device)
    score_barriers_u64 = torch.zeros(
        (max_barrier_rows, SCORE_BARRIER_WORDS), dtype=torch.uint64, device=device
    )
    barriers_i32_flat = score_barriers_u64.view(torch.int32).view(-1)
    ids_i64_buf = torch.empty(
        (max_tok_cap, max_top_k), dtype=torch.int64, device=device
    )
    img_mask_bool = torch.zeros(max_tok_cap, dtype=torch.bool, device=device)
    img_lt_bool = torch.zeros(max_tok_cap, dtype=torch.bool, device=device)
    img_mask_u8 = img_mask_bool.view(torch.uint8)
    img_zero_u8 = torch.zeros(max_tok_cap, dtype=torch.uint8, device=device)
    sentinel_logits = torch.empty((0,), dtype=torch.float32, device=device)
    dummy_f32 = torch.zeros(1, dtype=torch.float32, device=device)
    dummy_u8 = torch.zeros(1, dtype=torch.uint8, device=device)
    dummy_i32 = torch.zeros(1, dtype=torch.int32, device=device)
    dummy_i64_by_topk: dict[int, torch.Tensor] = {
        tk: torch.zeros((1, tk), dtype=torch.int64, device=device)
        for _h, _e, tk, _hb, _hib in unique_geom
    }

    # Build O(1) runtime dispatch tables indexed by `M` (`0..max_tok_cap`):
    # `(compiled_kernel, num_worker_groups, scratch_len, barrier_i32_len)` or `None`.
    geom_tables: dict[
        tuple[int, int, int, bool, bool], tuple[tuple[Any, int, int, int] | None, ...]
    ] = {}
    for gkey, cfgs_m in geom_cfg_by_m.items():
        hidden, num_experts, top_k, has_bias, has_image_bias = gkey
        entries: list[tuple[Any, int, int, int] | None] = [None] * (max_tok_cap + 1)
        for m in range(min_tokens, max_tok_cap + 1):
            cfg = cfgs_m[m]
            if cfg is None:
                continue
            spec = KernelSpec(
                config=cfg,
                hidden=hidden,
                num_experts=num_experts,
                num_topk=top_k,
                num_shared=0,
                has_bias=has_bias,
                has_image_bias=has_image_bias,
                has_mask=False,
                has_physical_map=False,
                has_unmapped=False,
                enable_pdl=enable_pdl,
                has_fixed_routing=False,
            )
            kern = compiled_by_spec[spec]
            nwg = int(cfg.num_worker_groups(m, sms))
            scr_len = int(math.prod(cfg.scratch_shape(m, num_experts)))
            bar_len = int(cfg.num_token_blocks(m)) * (SCORE_BARRIER_WORDS * 2)
            entries[m] = (kern, nwg, scr_len, bar_len)
        geom_tables[gkey] = tuple(entries)

    seen_img_lo: set[int] = set()
    for (
        runner,
        weight,
        bias_t,
        bias_vl_t,
        img_sentinel_lo,
        top_k,
        routed_scale,
    ) in eligible:
        hidden = int(weight.shape[1])
        num_experts = int(weight.shape[0])
        has_img_bias = bias_vl_t is not None
        gkey = (hidden, num_experts, top_k, bias_t is not None, has_img_bias)
        sched_table = geom_tables[gkey]
        bias_arg = bias_t if bias_t is not None else dummy_f32
        img_bias_arg = bias_vl_t if bias_vl_t is not None else dummy_f32
        is_mask_producer = False
        if has_img_bias and img_sentinel_lo not in seen_img_lo:
            seen_img_lo.add(img_sentinel_lo)
            is_mask_producer = True
        dummy_i64 = dummy_i64_by_topk[top_k]
        orig_gate_forward = runner.gate.forward
        orig_select_experts = runner.router.select_experts

        def _make_hooks(
            *,
            _orig_gate_fwd: Any = orig_gate_forward,
            _orig_sel: Any = orig_select_experts,
            _router: Any = runner.router,
            _weight: torch.Tensor = weight,
            _bias_arg: torch.Tensor = bias_arg,
            _img_bias_arg: torch.Tensor = img_bias_arg,
            _has_img_bias: bool = has_img_bias,
            _img_lo: int = img_sentinel_lo,
            _produce_mask: bool = is_mask_producer,
            _dummy_i64: torch.Tensor = dummy_i64,
            _table: tuple[tuple[Any, int, int, int] | None, ...] = sched_table,
            _hidden: int = hidden,
            _top_k: int = top_k,
            _scale: float = routed_scale,
        ):
            def _gate_fwd(self_gate: Any, x: torch.Tensor):
                x_2d = x if x.ndim == 2 else x.reshape(-1, x.shape[-1])
                m = int(x_2d.shape[0])
                if (
                    min_tokens <= m <= max_tok_cap
                    and x_2d.shape[1] == _hidden
                    and x_2d.dtype == torch.bfloat16
                    and x_2d.is_contiguous()
                    and x_2d.data_ptr() % 16 == 0
                    and _table[m] is not None
                    and not (
                        check_skip_pad
                        and is_forward_context_available()
                        and get_forward_context().is_padding is not None
                    )
                ):
                    return sentinel_logits, None
                return _orig_gate_fwd(x)

            def _select_experts(
                hidden_states: torch.Tensor,
                router_logits: torch.Tensor,
                topk_indices_dtype: torch.dtype | None = None,
                *,
                input_ids: torch.Tensor | None = None,
            ) -> tuple[torch.Tensor, torch.Tensor]:
                if router_logits is sentinel_logits:
                    x_2d = (
                        hidden_states
                        if hidden_states.ndim == 2
                        else hidden_states.reshape(-1, _hidden)
                    )
                    m = int(x_2d.shape[0])
                    entry = _table[m]
                    assert entry is not None
                    kern, nwg, scr_len, bar_len = entry
                    if _has_img_bias:
                        if input_ids is not None:
                            if _produce_mask:
                                ids_flat = input_ids.reshape(-1)[:m]
                                torch.ge(ids_flat, _img_lo, out=img_mask_bool[:m])
                                torch.lt(ids_flat, _img_lo + 5, out=img_lt_bool[:m])
                                img_mask_bool[:m].logical_and_(img_lt_bool[:m])
                            img_mask_arg = img_mask_u8[:m]
                        else:
                            img_mask_arg = img_zero_u8[:m]
                    else:
                        img_mask_arg = dummy_u8
                    ids_i64 = (
                        ids_i64_buf[:m]
                        if _top_k == max_top_k
                        else ids_i64_buf[:m, :_top_k].contiguous()
                    )
                    out_wts = torch.empty(
                        (m, _top_k), dtype=torch.float32, device=device
                    )
                    kern(
                        x_2d,
                        _weight,
                        _bias_arg,
                        _img_bias_arg,
                        img_mask_arg,
                        dummy_u8,
                        dummy_u8,
                        dummy_i32,
                        dummy_i32,
                        ids_i64,
                        out_wts,
                        _dummy_i64,
                        scratch_flat[:scr_len],
                        barriers_i32_flat[:bar_len],
                        m,
                        0,
                        _scale,
                        1,
                        nwg,
                    )
                    target_dtype = (
                        torch.int32
                        if topk_indices_dtype is None
                        else topk_indices_dtype
                    )
                    out_ids = (
                        ids_i64.clone()
                        if target_dtype == torch.int64
                        else ids_i64.to(dtype=target_dtype)
                    )
                    replay_out = getattr(_router, "_routing_replay_out", None)
                    if replay_out is not None:
                        replay_out[:m].copy_(out_ids.to(torch.int16))
                    return out_wts, out_ids
                return _orig_sel(
                    hidden_states=hidden_states,
                    router_logits=router_logits,
                    topk_indices_dtype=topk_indices_dtype,
                    input_ids=input_ids,
                )

            return _gate_fwd, _select_experts

        gate_fwd_fn, sel_fn = _make_hooks()
        runner.gate.forward = types.MethodType(gate_fwd_fn, runner.gate)
        runner.router.select_experts = sel_fn
        runner._zstep_mega_gate_patched = True

    model._sm120_mega_gate_state = _Sm120MegaGateState(  # type: ignore[attr-defined]
        scratch_flat=scratch_flat,
        score_barriers_u64=score_barriers_u64,
        barriers_i32_flat=barriers_i32_flat,
        ids_i64_buf=ids_i64_buf,
        img_mask_bool=img_mask_bool,
        img_lt_bool=img_lt_bool,
        img_zero_u8=img_zero_u8,
        sentinel_logits=sentinel_logits,
        dummy_f32=dummy_f32,
        dummy_u8=dummy_u8,
        dummy_i32=dummy_i32,
        dummy_i64_by_topk=dummy_i64_by_topk,
        min_tokens=min_tokens,
        max_tokens=max_tok_cap,
        num_schedules=len(specs_to_compile),
        patched_layers=len(eligible),
    )
    return len(eligible), len(specs_to_compile), min_tokens


_MHC_STATE_BY_DEV: dict[int, Any] = {}
_CSMLA_STATE_BY_DEV: dict[
    int,
    dict[tuple[int, int, int, bool, str], tuple[tuple[int, Any, Any, Any, int], ...]],
] = {}


def _evict_csmla_owner(dev_idx: int, owner_id: int) -> None:
    dev_tbl = _CSMLA_STATE_BY_DEV.get(dev_idx)
    if not dev_tbl:
        return
    empty_sigs: list[tuple[int, int, int, bool, str]] = []
    for sig, entries in dev_tbl.items():
        kept = tuple(e for e in entries if e[4] != owner_id)
        if kept:
            dev_tbl[sig] = kept
        else:
            empty_sigs.append(sig)
    for sig in empty_sigs:
        dev_tbl.pop(sig, None)
    if not dev_tbl:
        _CSMLA_STATE_BY_DEV.pop(dev_idx, None)


def _compute_mhc_split_cap(compute_splits_fn: Any, cap: int) -> int:
    """Compute the exact upper bound of `splits * n` across `1 <= n <= cap`."""
    mcap = max(32768, int(cap))
    for n in range(1, int(cap) + 1):
        prod = int(compute_splits_fn(20480, n)) * n
        if prod > mcap:
            mcap = prod
    return mcap


def _sync_mhc_state_in(s: Any, main_s: torch.cuda.Stream | None) -> None:
    if getattr(s, "deferred", None) is not None:
        s.flush(main_s)
    if getattr(s, "pending", False):
        if main_s is not None:
            main_s.wait_stream(s.mhc_stream)
        s.pending = False


def _wrap_mhc_forward_join(target_mod: torch.nn.Module, dev_idx: int) -> None:
    """Ensure `mhc_stream` always joins `main_s` before `forward` exits or unwinds."""
    if getattr(target_mod, "_zstep_sm120_mhc_fwd_wrapped", False):
        return
    orig_fwd = target_mod.forward

    def _wrapped_forward(*args: Any, **kwargs: Any) -> Any:
        try:
            return orig_fwd(*args, **kwargs)
        except BaseException:
            s = _MHC_STATE_BY_DEV.get(dev_idx)
            if s is not None:
                s.deferred = None
                if getattr(s, "pending", False):
                    if (
                        getattr(s, "dev", None) is not None
                        and s.dev.type == "cuda"
                        and torch.cuda.is_available()
                        and not torch.cuda.is_current_stream_capturing()
                    ):
                        torch.cuda.current_stream(s.dev).wait_stream(s.mhc_stream)
                    s.pending = False
            raise
        finally:
            s = _MHC_STATE_BY_DEV.get(dev_idx)
            if s is not None and (
                getattr(s, "deferred", None) is not None or getattr(s, "pending", False)
            ):
                main_s = (
                    torch.cuda.current_stream(s.dev)
                    if getattr(s, "dev", None) is not None
                    and s.dev.type == "cuda"
                    and torch.cuda.is_available()
                    else None
                )
                _sync_mhc_state_in(s, main_s)

    target_mod._zstep_sm120_orig_forward = orig_fwd  # type: ignore[attr-defined]
    target_mod._zstep_sm120_mhc_fwd_wrapped = True  # type: ignore[attr-defined]
    target_mod.forward = _wrapped_forward  # type: ignore[method-assign]


def flush_sm120_mhc_before_ar(device_idx: int | None) -> None:
    """Flush any deferred `n > 256` mHC Sinkhorn projection into the `shm_ar` window."""
    s = _MHC_STATE_BY_DEV.get(0 if device_idx is None else device_idx)
    if s is not None and getattr(s, "deferred", None) is not None:
        stream = (
            torch.cuda.current_stream(s.dev)
            if getattr(s, "dev", None) is not None
            and s.dev.type == "cuda"
            and torch.cuda.is_available()
            else None
        )
        s.flush(stream)


def attach_sm120_mhc_shm_ar(
    device_idx: int | None,
    shm_ar: Any | None,
    runner: Any | None = None,
    shm_ar_decode: Any | None = None,
) -> None:
    """Bind weak references to `HostMappedShmAllReduce` and runner for `M in {1, 2, 4}` cluster8 fusion."""
    s = _MHC_STATE_BY_DEV.get(0 if device_idx is None else int(device_idx))
    if s is not None:
        s.shm_ar_ref = weakref.ref(shm_ar) if shm_ar is not None else None
        s.shm_ar_dec_ref = weakref.ref(shm_ar_decode) if shm_ar_decode is not None else None
        s.runner_ref = weakref.ref(runner) if runner is not None else None


def _prepare_sm120_mhc_overlap(
    model: torch.nn.Module, *, device: torch.device, max_tokens: int
) -> None:
    """Install 1-pass in-place SM120 mHC post+collapse+RMSNorm and side-stream Sinkhorn overlap."""
    cfg = WorkerRuntimeConfig.from_env()
    if not (
        cfg.enable_sm120_mhc_overlap
        and torch.cuda.is_available()
        and torch.cuda.get_device_capability() == (12, 0)
    ):
        return
    try:
        from vllm.distributed import tensor_model_parallel_all_reduce
        from vllm.forward_context import in_piecewise_cudagraph
        from vllm.model_executor.kernels.mhc.warmup import compute_mhc_pre_num_splits
        from vllm.models.deepseek_v41.nvidia import model as dsv41_model
        from vllm.models.deepseek_v41.nvidia.ops import mhc as dsv41_mhc
        from vllm.utils.deep_gemm import is_deep_gemm_supported, tf32_hc_prenorm_gemm
        from zstep_worker.kernels_sm120.shm_allreduce import (
            get_host_mapped_allreduce_extension,
        )
    except ImportError:
        return
    if not is_deep_gemm_supported():
        return

    cap = max(int(max_tokens), 8192)
    dev_idx = 0 if device.index is None else int(device.index)
    mcap = _compute_mhc_split_cap(compute_mhc_pre_num_splits, cap)
    f32 = {"dtype": torch.float32, "device": device}

    st = _MHC_STATE_BY_DEV.get(dev_idx)
    if st is None:
        st = types.SimpleNamespace(
            dev=device,
            mhc_stream=torch.cuda.Stream(device=device),
            mix_flat=torch.empty((mcap * 24,), **f32),
            sqr_flat=torch.empty((mcap,), **f32),
            empty_aux=torch.empty((0, 5120), dtype=torch.bfloat16, device=device),
            post_buf=tuple(torch.empty((cap, 4, 1), **f32) for _ in range(4)),
            comb_buf=tuple(torch.empty((cap, 4, 4), **f32) for _ in range(4)),
            pre_buf=tuple(torch.empty((cap, 4), **f32) for _ in range(4)),
            smem_mul=tuple(torch.empty((32 * 16 * 24,), **f32) for _ in range(4)),
            smem_sqr=tuple(torch.empty((32 * 16,), **f32) for _ in range(4)),
            smem_done_cnt=torch.zeros((64,), dtype=torch.int32, device=device),
            ar_cta_done=torch.zeros((64,), dtype=torch.int64, device=device),
            ar_cta_wait=torch.zeros((128,), dtype=torch.int64, device=device),
            cap=cap,
            slot=0,
            pending=False,
            deferred=None,
            ext=None,
            shm_ar_ref=None,
            shm_ar_dec_ref=None,
            runner_ref=None,
            owners={id(model)},
        )

        def _flush(main_s: torch.cuda.Stream) -> None:
            if st.deferred is not None:
                (
                    n,
                    res,
                    fn,
                    hc_scale,
                    hc_base,
                    np_,
                    nc,
                    npr,
                    r_eps,
                    pre_eps,
                    sk_eps,
                    p_mult,
                    sk_rep,
                    ring_slot,
                ) = st.deferred
                st.deferred = None
                if 1 <= n <= 12 and fn.dtype == torch.float32 and fn.is_contiguous() and (fn.data_ptr() & 15) == 0:
                    st.ext = st.ext or get_host_mapped_allreduce_extension()
                    st.mhc_stream.wait_stream(main_s)
                    with torch.cuda.stream(st.mhc_stream):
                        st.ext.launch_mhc_prenorm_sinkhorn_smem(
                            res, fn, hc_scale, hc_base, st.smem_mul[ring_slot], st.smem_sqr[ring_slot],
                            st.smem_done_cnt, np_.view(n, 4), nc.view(n, 16), npr, int(ring_slot),
                            r_eps, pre_eps, sk_eps, p_mult, sk_rep, 1, st.mhc_stream.cuda_stream,
                        )
                    st.pending = True
                    return
                splits = int(compute_mhc_pre_num_splits(20480, n))
                needed = splits * n
                if needed > st.sqr_flat.numel():
                    if torch.cuda.is_current_stream_capturing():
                        raise RuntimeError(
                            f"mHC split workspace {needed} exceeds {st.sqr_flat.numel()} in capture"
                        )
                    st.mix_flat = torch.empty((needed * 24,), dtype=torch.float32, device=st.dev)
                    st.sqr_flat = torch.empty((needed,), dtype=torch.float32, device=st.dev)
                mix_v = st.mix_flat[: needed * 24].view(splits, n, 24)
                sqr_v = st.sqr_flat[: needed].view(splits, n)
                st.mhc_stream.wait_stream(main_s)
                with torch.cuda.stream(st.mhc_stream):
                    tf32_hc_prenorm_gemm(res.view(n, 20480), fn, mix_v, sqr_v, splits)
                    st.ext.launch_mhc_sinkhorn_stats(
                        mix_v,
                        sqr_v,
                        hc_scale,
                        hc_base,
                        np_.view(n, 4),
                        nc.view(n, 16),
                        npr,
                        20480,
                        r_eps,
                        pre_eps,
                        sk_eps,
                        p_mult,
                        sk_rep,
                        1,
                        st.mhc_stream.cuda_stream,
                    )
                st.pending = True

        st.flush = _flush
        _MHC_STATE_BY_DEV[dev_idx] = st
    else:
        owners = getattr(st, "owners", None)
        if not isinstance(owners, set):
            owners = set()
            st.owners = owners
        owners.add(id(model))
        if not torch.cuda.is_current_stream_capturing():
            if cap > st.cap:
                st.post_buf = tuple(torch.empty((cap, 4, 1), **f32) for _ in range(4))
                st.comb_buf = tuple(torch.empty((cap, 4, 4), **f32) for _ in range(4))
                st.pre_buf = tuple(torch.empty((cap, 4), **f32) for _ in range(4))
                st.cap = cap
            if mcap > st.sqr_flat.numel():
                st.mix_flat = torch.empty((mcap * 24,), **f32)
                st.sqr_flat = torch.empty((mcap,), **f32)

    model._sm120_mhc_dev = dev_idx  # type: ignore[attr-defined]
    _wrap_mhc_forward_join(model, dev_idx)
    inner_model = getattr(model, "model", None)
    if isinstance(inner_model, torch.nn.Module):
        _wrap_mhc_forward_join(inner_model, dev_idx)

    lang_mod = getattr(model, "language_model", model)
    dsv4_core = getattr(lang_mod, "model", lang_mod)
    if (
        cfg.enable_sm120_fuse_mhc_ar
        and isinstance(dsv4_core, dsv41_model.DeepseekV4Model)
        and not getattr(dsv4_core, "use_sequence_parallel", False)
    ):
        ok_layers = [
            lyr
            for lyr in dsv4_core.layers
            if isinstance(lyr, dsv41_model.DeepseekV4DecoderLayer)
            and hasattr(getattr(lyr, "attn", None), "wo_b")
            and hasattr(getattr(getattr(lyr, "ffn", None), "experts", None), "moe_config")
            and not getattr(lyr.ffn.experts, "_fused_output_is_reduced", False)
        ]
        if ok_layers and len(ok_layers) == len(dsv4_core.layers):
            dsv4_core.fuse_mhc_all_reduce = True
            for lyr in ok_layers:
                lyr.fuse_mhc_all_reduce = True
                lyr.attn.wo_b.reduce_results = False
                lyr.ffn.reduce_results = False
                lyr.ffn.experts.moe_config.skip_final_all_reduce = True

    if not getattr(dsv41_model, "_zstep_sm120_mhc_patched", False):
        orig_post = dsv41_model.mhc_post_tilelang
        orig_shifted = dsv41_model.mhc_shifted_post_pre

        def _patched_post(x, residual, post_layer_mix, comb_res_mix):
            if isinstance(residual, torch.Tensor) and residual.is_cuda:
                r_idx = 0 if residual.device.index is None else int(residual.device.index)
                s = _MHC_STATE_BY_DEV.get(r_idx)
            else:
                s = None
            if s is not None:
                main_s = torch.cuda.current_stream(residual.device)
                _sync_mhc_state_in(s, main_s)
                n = int(residual.shape[0]) if residual.ndim == 3 else -1
                if (
                    residual.ndim == 3
                    and residual.shape[1:] == (4, 5120)
                    and isinstance(x, torch.Tensor)
                    and x.is_cuda
                    and x.shape == (n, 5120)
                    and x.dtype == torch.bfloat16
                    and residual.dtype == torch.bfloat16
                    and isinstance(post_layer_mix, torch.Tensor)
                    and post_layer_mix.is_cuda
                    and post_layer_mix.dtype == torch.float32
                    and post_layer_mix.numel() == n * 4
                    and isinstance(comb_res_mix, torch.Tensor)
                    and comb_res_mix.is_cuda
                    and comb_res_mix.dtype == torch.float32
                    and comb_res_mix.numel() == n * 16
                    and all(
                        t.is_contiguous()
                        for t in (x, residual, post_layer_mix, comb_res_mix)
                    )
                    and (x.data_ptr() & 15) == 0
                    and (residual.data_ptr() & 15) == 0
                ):
                    if n > 0:
                        s.ext = s.ext or get_host_mapped_allreduce_extension()
                        s.ext.launch_mhc_post_collapse_rmsnorm(
                            x,
                            residual,
                            post_layer_mix,
                            comb_res_mix,
                            post_layer_mix,
                            s.empty_aux,
                            s.empty_aux,
                            s.empty_aux,
                            1e-6,
                            0,
                            1,
                            0,
                            0,
                            0,
                            main_s.cuda_stream,
                        )
                    return residual
            return orig_post(x, residual, post_layer_mix, comb_res_mix)

        def _patched_shifted(
            x,
            residual,
            post_layer_mix,
            comb_res_mix,
            fn,
            hc_scale,
            hc_base,
            rms_eps,
            hc_pre_eps,
            hc_sinkhorn_eps,
            hc_post_mult_value,
            sinkhorn_repeat,
            pre_mix=None,
            norm_weight=None,
            norm_eps=1e-6,
            capture_aux=False,
            *,
            stream=None,
            reduce_results=False,
        ):
            if isinstance(residual, torch.Tensor) and residual.is_cuda:
                r_idx = 0 if residual.device.index is None else int(residual.device.index)
                s = _MHC_STATE_BY_DEV.get(r_idx)
            else:
                s = None
            n = int(residual.shape[0]) if (isinstance(residual, torch.Tensor) and residual.ndim == 3) else -1
            if (
                s is not None
                and isinstance(x, torch.Tensor)
                and x.is_cuda
                and stream is None
                and pre_mix is not None
                and norm_weight is not None
                and isinstance(fn, torch.Tensor)
                and fn.is_cuda
                and fn.shape == (24, 20480)
                and residual.shape[1:] == (4, 5120)
                and x.shape == (n, 5120)
                and 0 < n <= s.cap
                and x.dtype == torch.bfloat16
                and residual.dtype == torch.bfloat16
                and isinstance(norm_weight, torch.Tensor)
                and norm_weight.is_cuda
                and norm_weight.dtype == torch.bfloat16
                and norm_weight.shape == (5120,)
                and isinstance(post_layer_mix, torch.Tensor)
                and post_layer_mix.is_cuda
                and post_layer_mix.dtype == torch.float32
                and post_layer_mix.numel() == n * 4
                and isinstance(comb_res_mix, torch.Tensor)
                and comb_res_mix.is_cuda
                and comb_res_mix.dtype == torch.float32
                and comb_res_mix.numel() == n * 16
                and isinstance(pre_mix, torch.Tensor)
                and pre_mix.is_cuda
                and pre_mix.dtype == torch.float32
                and pre_mix.numel() == n * 4
                and isinstance(hc_scale, torch.Tensor)
                and hc_scale.is_cuda
                and hc_scale.dtype == torch.float32
                and hc_scale.is_contiguous()
                and hc_scale.numel() >= 3
                and isinstance(hc_base, torch.Tensor)
                and hc_base.is_cuda
                and hc_base.dtype == torch.float32
                and hc_base.is_contiguous()
                and hc_base.numel() >= 24
                and int(sinkhorn_repeat) >= 1
                and all(
                    t.is_contiguous()
                    for t in (residual, post_layer_mix, comb_res_mix, pre_mix, norm_weight)
                )
                and (residual.data_ptr() & 15) == 0
                and (norm_weight.data_ptr() & 15) == 0
            ):
                s.ext = s.ext or get_host_mapped_allreduce_extension()
                main_s = torch.cuda.current_stream(residual.device)
                x = x if x.is_contiguous() else x.contiguous()
                ar_ref = getattr(s, "shm_ar_ref", None)
                ar = ar_ref() if callable(ar_ref) else None
                use_c8 = bool(
                    reduce_results
                    and n in (1, 2, 4)
                    and ar is not None
                    and not getattr(ar, "_closed", True)
                    and getattr(ar, "_use_shm_v5", False)
                    and ar.world_size == 8
                    and (ar.blocks == 8 or (ar._pxb_enabled and 32 <= ar.blocks <= 56))
                    and (x.data_ptr() & 15) == 0
                )
                ar_dec = (
                    s.shm_ar_dec_ref()
                    if callable(getattr(s, "shm_ar_dec_ref", None))
                    else None
                )
                use_dec_ar = bool(
                    reduce_results
                    and not use_c8
                    and 5 <= n <= 128
                    and ar_dec is not None
                    and not getattr(ar_dec, "_closed", True)
                    and getattr(ar_dec, "_use_shm_v5", False)
                    and ar_dec.world_size == 8
                    and ar_dec._pxb_enabled
                    and (x.data_ptr() & 15) == 0
                )
                use_wave_mhc = bool(
                    reduce_results
                    and not use_c8
                    and not use_dec_ar
                    and ar is not None
                    and not getattr(ar, "_closed", True)
                    and getattr(ar, "_use_shm_v5", False)
                    and ar.world_size == 8
                    and ar._pxb_enabled
                    and ar._pxb_bfp == 1
                    and n >= 256
                    and n * 640 <= ar.pxb_max_vecs
                    and (x.data_ptr() & 15) == 0
                )
                if reduce_results and not use_c8 and not use_dec_ar and not use_wave_mhc:
                    x = tensor_model_parallel_all_reduce(x)
                    x = x if x.is_contiguous() else x.contiguous()
                if (x.data_ptr() & 15) == 0:
                    layer_in = torch.empty(
                        (n, 5120), dtype=torch.bfloat16, device=residual.device
                    )
                    aux_buf = torch.empty_like(layer_in) if capture_aux else s.empty_aux
                    if use_wave_mhc:
                        if getattr(s, "deferred", None) is not None:
                            s.flush(main_s)
                        s.mhc_stream.wait_stream(main_s)
                        s.ext.launch_shm_v5_pxb_allreduce(
                            x, x, ar.control, ar._rank_shm_ptrs,
                            ar._my_hbm_ptr, ar._pt_hbm_ptr, ar.slot_stride_bytes,
                            ar.pxb_slot_stride_bytes, ar.rank, ar.world_size,
                            ar.blocks, ar.threads, ar.one_shot_max_vecs,
                            ar.pxb_max_vecs, ar._pxb_bfp,
                            s.ar_cta_done.data_ptr(), main_s.cuda_stream,
                        )
                        s.ext.launch_mhc_post_collapse_rmsnorm(
                            x, residual, post_layer_mix, comb_res_mix, pre_mix,
                            norm_weight, layer_in, aux_buf, float(norm_eps),
                            1 if capture_aux else 0, 0, ar.blocks,
                            s.ar_cta_done.data_ptr(), s.ar_cta_wait.data_ptr(),
                            s.mhc_stream.cuda_stream,
                        )
                        main_s.wait_stream(s.mhc_stream)
                        s.pending = False
                        runner_ref = getattr(s, "runner_ref", None)
                        runner_obj = runner_ref() if callable(runner_ref) else None
                        if runner_obj is not None:
                            runner_obj.shm_ar_calls += 1
                    elif use_dec_ar:
                        _sync_mhc_state_in(s, main_s)
                        s.ext.launch_ar_mhc_decode(
                            x,
                            residual,
                            post_layer_mix,
                            comb_res_mix,
                            pre_mix,
                            norm_weight,
                            layer_in,
                            aux_buf,
                            ar_dec.control,
                            ar_dec._rank_shm_ptrs,
                            ar_dec._my_hbm_ptr,
                            ar_dec._pt_hbm_ptr,
                            ar_dec.slot_stride_bytes,
                            ar_dec.pxb_slot_stride_bytes,
                            float(norm_eps),
                            1 if capture_aux else 0,
                            ar_dec.rank,
                            main_s.cuda_stream,
                        )
                        runner_obj = (
                            s.runner_ref()
                            if callable(getattr(s, "runner_ref", None))
                            else None
                        )
                        if runner_obj is not None:
                            runner_obj.shm_ar_calls += 1
                    elif use_c8:
                        _sync_mhc_state_in(s, main_s)
                        flag_base = ar.blocks if (ar._pxb_enabled and 32 <= ar.blocks <= 56) else 0
                        s.ext.launch_ar_mhc_cluster8(
                            x,
                            residual,
                            post_layer_mix,
                            comb_res_mix,
                            pre_mix,
                            norm_weight,
                            layer_in,
                            aux_buf,
                            ar.control,
                            ar._rank_shm_ptrs,
                            ar._my_hbm_ptr if ar._pxb_enabled else 0,
                            ar._pt_hbm_ptr if ar._pxb_enabled else 0,
                            ar.slot_stride_bytes,
                            ar.pxb_slot_stride_bytes if ar._pxb_enabled else 0,
                            float(norm_eps),
                            1 if capture_aux else 0,
                            ar.rank,
                            flag_base,
                            main_s.cuda_stream,
                        )
                        runner_ref = getattr(s, "runner_ref", None)
                        runner_obj = runner_ref() if callable(runner_ref) else None
                        if runner_obj is not None:
                            runner_obj.shm_ar_calls += 1
                    else:
                        _sync_mhc_state_in(s, main_s)
                        s.ext.launch_mhc_post_collapse_rmsnorm(
                            x,
                            residual,
                            post_layer_mix,
                            comb_res_mix,
                            pre_mix,
                            norm_weight,
                            layer_in,
                            aux_buf,
                            float(norm_eps),
                            1 if capture_aux else 0,
                            0,
                            0,
                            0,
                            0,
                            main_s.cuda_stream,
                        )
                    p, s.slot = s.slot, (s.slot + 1) & 3
                    next_post = s.post_buf[p][:n]
                    next_comb = s.comb_buf[p][:n]
                    next_pre = s.pre_buf[p][:n]
                    if in_piecewise_cudagraph():
                        splits = int(compute_mhc_pre_num_splits(20480, n))
                        needed = splits * n
                        mix_v = s.mix_flat[: needed * 24].view(splits, n, 24)
                        sqr_v = s.sqr_flat[: needed].view(splits, n)
                        tf32_hc_prenorm_gemm(residual.view(n, 20480), fn, mix_v, sqr_v, splits)
                        s.ext.launch_mhc_sinkhorn_stats(
                            mix_v, sqr_v, hc_scale, hc_base, next_post.view(n, 4),
                            next_comb.view(n, 16), next_pre, 20480, float(rms_eps),
                            float(hc_pre_eps), float(hc_sinkhorn_eps),
                            float(hc_post_mult_value), int(sinkhorn_repeat), 1, main_s.cuda_stream,
                        )
                    else:
                        s.deferred = (
                            n, residual, fn, hc_scale, hc_base, next_post, next_comb, next_pre,
                            float(rms_eps), float(hc_pre_eps), float(hc_sinkhorn_eps),
                            float(hc_post_mult_value), int(sinkhorn_repeat), int(p),
                        )
                        if n <= 256:
                            s.flush(main_s)
                    return (
                        residual,
                        next_post,
                        next_comb,
                        layer_in,
                        next_pre,
                        aux_buf if capture_aux else None,
                    )
                # If `(x.data_ptr() & 15) != 0` after external AR (`not use_c8`), `x` was
                # already reduced above; pass `reduce_results=False` to `orig_shifted`.
                if reduce_results and not use_c8 and not use_dec_ar and not use_wave_mhc:
                    reduce_results = False
            if s is not None:
                _sync_mhc_state_in(s, torch.cuda.current_stream(residual.device))
            return orig_shifted(
                x,
                residual,
                post_layer_mix,
                comb_res_mix,
                fn,
                hc_scale,
                hc_base,
                rms_eps,
                hc_pre_eps,
                hc_sinkhorn_eps,
                hc_post_mult_value,
                sinkhorn_repeat,
                pre_mix=pre_mix,
                norm_weight=norm_weight,
                norm_eps=norm_eps,
                capture_aux=capture_aux,
                stream=stream,
                reduce_results=reduce_results,
            )

        dsv41_model.mhc_post_tilelang = _patched_post
        dsv41_model.mhc_shifted_post_pre = _patched_shifted
        dsv41_mhc.mhc_shifted_post_pre = _patched_shifted
        if hasattr(dsv41_mhc, "mhc_post_tilelang"):
            dsv41_mhc.mhc_post_tilelang = _patched_post
        for mod_name in (
            "vllm.model_executor.kernels.mhc.tilelang",
            "vllm.model_executor.kernels.mhc",
        ):
            src_mod = sys.modules.get(mod_name)
            if src_mod is not None and getattr(src_mod, "mhc_post_tilelang", None) is orig_post:
                src_mod.mhc_post_tilelang = _patched_post  # type: ignore[attr-defined]
        dsv41_model._zstep_sm120_mhc_fns = (  # type: ignore[attr-defined]
            orig_post,
            orig_shifted,
            _patched_post,
            _patched_shifted,
        )
        dsv41_model._zstep_sm120_mhc_patched = True

    patched_fns = getattr(dsv41_model, "_zstep_sm120_mhc_fns", None)
    if patched_fns is not None:
        orig_p, orig_s, patch_p, patch_s = patched_fns
        visited_mods = {
            sys.modules.get(type(sub).__module__) for sub in model.modules()
        }
        for mod in visited_mods:
            if mod is None:
                continue
            if getattr(mod, "mhc_post_tilelang", None) is orig_p:
                mod.mhc_post_tilelang = patch_p  # type: ignore[attr-defined]
            if getattr(mod, "mhc_shifted_post_pre", None) is orig_s:
                mod.mhc_shifted_post_pre = patch_s  # type: ignore[attr-defined]


def prepare_b12x_dsv41_model(
    model: torch.nn.Module,
    *,
    rank: int,
    device: torch.device,
    max_tokens: int,
    decode_buckets: tuple[int, ...],
    exchange_group: dist.ProcessGroup | None = None,
    enable_4k_linear_cap: bool = False,
) -> B12xPrepareSummary:
    """Compile/autotune all SM120 `b12x`, `sm12x_mega_gate` & `sm120_dsa_indexer` plans (rank 0 first)."""
    t0 = time.perf_counter()
    _prepare_sm120_mhc_overlap(model, device=device, max_tokens=max_tokens)
    dsa_enabled = _prepare_sm120_dsa_indexer(
        rank=rank,
        device=device,
        decode_buckets=decode_buckets,
        exchange_group=exchange_group,
    )
    mg_layers, mg_schedules, mg_min_tokens = _prepare_sm12x_mega_gate(
        model,
        rank=rank,
        device=device,
        max_tokens=max_tokens,
        exchange_group=exchange_group,
    )

    if not maybe_enable_b12x_for_dsv41():
        prep_ms = (time.perf_counter() - t0) * 1000.0
        return B12xPrepareSummary(
            enabled=False,
            moe_layers=0,
            linear_layers=0,
            linear_shapes=0,
            moe_scratch_bytes=0,
            linear_scratch_bytes=0,
            prepare_ms=prep_ms,
            mega_gate_layers=mg_layers,
            mega_gate_schedules=mg_schedules,
            mega_gate_min_tokens=mg_min_tokens,
            dsa_decode_enabled=dsa_enabled,
        )

    from b12x.gemm import block_fp8_linear
    from b12x.gemm._shared.block_fp8 import (
        _SCRATCH_ALIGN_BYTES,
        _block_fp8_linear_scratch_layout,
        _block_fp8_linear_x_q_from_scratch,
    )
    from b12x.moe import fused_moe
    from b12x.moe.fused_moe.workloads import make_tuning_routes
    from b12x.preparation import PreparationSession, PreparedCall
    from b12x.preparation.types import require_prepared

    moe_layers: list[tuple[str, torch.nn.Module, Any]] = []
    linear_layers: list[tuple[str, torch.nn.Module, Any]] = []
    for mod_name, mod in model.named_modules():
        qm = getattr(mod, "quant_method", None)
        if qm is None or not getattr(qm, "_b12x_enabled", False):
            continue
        if hasattr(mod, "_b12x_prepared_experts"):
            moe_layers.append((mod_name, mod, qm))
        elif hasattr(mod, "_b12x_packed_weight"):
            linear_layers.append((mod_name, mod, qm))

    if not moe_layers and not linear_layers:
        prep_ms = (time.perf_counter() - t0) * 1000.0
        return B12xPrepareSummary(
            enabled=False,
            moe_layers=0,
            linear_layers=0,
            linear_shapes=0,
            moe_scratch_bytes=0,
            linear_scratch_bytes=0,
            prepare_ms=prep_ms,
            mega_gate_layers=mg_layers,
            mega_gate_schedules=mg_schedules,
            mega_gate_min_tokens=mg_min_tokens,
            dsa_decode_enabled=dsa_enabled,
        )

    caps_set = set(int(b) for b in decode_buckets if int(b) > 0)
    extra_lin_caps = (
        (256, 512, 1024, 2048, 3072, 4096, int(max_tokens))
        if enable_4k_linear_cap
        else (256, 512, 1024, 2048, 3072, int(max_tokens))
    )
    for extra_c in extra_lin_caps:
        if 0 < extra_c <= int(max_tokens):
            caps_set.add(extra_c)
    caps_set.add(int(max_tokens))
    caps_tuple: tuple[int, ...] = tuple(sorted(caps_set))

    moe_warmup_counts = tuple(
        sorted(
            {
                *(int(b) for b in decode_buckets if int(b) > 0),
                1,
                2,
                4,
                8,
                16,
                32,
                40,
                64,
                80,
                128,
                *(c for c in (24, 48, 96, 256, 512, 1024, 2048) if c <= int(max_tokens)),
                min(3072, int(max_tokens)),
                int(max_tokens),
            }
        )
    )

    requests: list[Any] = []

    # 1. Build per-layer MoE execution plans (all 40 share identical query key in `PreparationSession`).
    for idx, (_mname, layer, qm) in enumerate(moe_layers):
        prepared = layer._b12x_prepared_experts
        top_k = int(layer.moe_config.experts_per_token)
        num_experts = int(qm.num_experts)
        hidden_size = int(qm.hidden_size)
        eplan = fused_moe.plan_execution(
            experts=prepared,
            capacity=fused_moe.ExecutionCapacity(
                max_tokens=int(max_tokens),
                top_k=top_k,
                warmup_token_counts=moe_warmup_counts,
                route_num_experts=0,
            ),
            routing=fused_moe.RoutingSpec(
                apply_router_weight_on_input=bool(
                    layer.apply_router_weight_on_input
                )
            ),
        )
        qm._b12x_eplan = eplan

        def _make_moe_call(
            cnt: int,
            *,
            _prep: Any = prepared,
            _tk: int = top_k,
            _ne: int = num_experts,
            _hs: int = hidden_size,
        ):
            def _factory(state: Any) -> PreparedCall:
                scr = tuple(
                    torch.empty(s.shape, dtype=s.dtype, device=device)
                    for s in state.scratch.scratch_specs()
                )
                hid = (
                    torch.randn((cnt, _hs), dtype=torch.bfloat16, device=device)
                    * 0.125
                )
                out = torch.empty_like(hid)
                ids = make_tuning_routes(cnt, _tk, _ne, device=device)[0].clone()
                wts = torch.full(
                    (cnt, _tk), 1.0 / _tk, dtype=torch.float32, device=device
                )
                b = state.bind(
                    scratch=scr,
                    a=hid,
                    experts=_prep,
                    topk_weights=wts,
                    topk_ids=ids,
                    output=out,
                    input_scales_static=True,
                )
                return PreparedCall(
                    run=b.run,
                    output=out,
                    produce=lambda: hid.normal_(std=0.125),
                    reset=lambda: out.zero_(),
                )

            return _factory

        calls = {c: _make_moe_call(int(c)) for c in eplan.token_counts}
        requests.append(
            eplan.request(
                name=f"dsv41_moe_l{idx}",
                prepare_calls=calls,
                benchmark_calls=calls,
            )
        )

    # 2. Build shared Block32 FP8 linear plans keyed by `(out_features, in_features)`.
    shape_rep_weight: dict[tuple[int, int], Any] = {}
    shape_streams: dict[tuple[int, int], set[int]] = {}
    for _lname, layer, qm in linear_layers:
        skey = (int(qm._out_features), int(qm._in_features))
        if skey not in shape_rep_weight:
            shape_rep_weight[skey] = layer._b12x_packed_weight
        stream_slot = 1 if qm._is_shared_expert else 0
        shape_streams.setdefault(skey, set()).add(stream_slot)

    wo_attn_mods = [
        m
        for m in model.modules()
        if getattr(getattr(m, "wo_a", None), "_b12x_wo_a_packed", None) is not None
        and getattr(getattr(m, "wo_b", None), "_b12x_packed_weight", None) is not None
        and (
            int(getattr(m, "n_local_groups", 0)),
            int(getattr(m, "n_local_heads", 0)),
            int(getattr(m, "nope_head_dim", 0)),
            int(getattr(m, "rope_head_dim", 0)),
            int(getattr(m, "o_lora_rank", 0)),
        )
        == (1, 8, 448, 64, 1024)
    ]
    pwo_dyn: Any = None
    if wo_attn_mods:
        from b12x.gemm import wo_projection
        from b12x.preparation import FrozenMapping

        shape_rep_weight[(1024, 4096)] = wo_attn_mods[0].wo_a._b12x_wo_a_packed
        shape_streams.setdefault((1024, 4096), set()).add(0)
        pwo_dyn = wo_projection.plan(
            wo_projection.Caps(
                device=device,
                max_tokens=int(max_tokens),
                groups=1,
                group_width=4096,
                rank=1024,
                hidden=5120,
            ),
            invocation=FrozenMapping(
                {
                    "operation": "inv_rope",
                    "heads_per_group": 8,
                    "nope_dim": 448,
                    "rope_dim": 64,
                    "positions_dtype": "int64",
                    "cos_sin_dtype": "float32",
                    "dynamic_tokens": True,
                }
            ),
        )

        def _mk_wodyn(st_w: Any, *, _p: Any = pwo_dyn) -> PreparedCall:
            n_w = min(64, int(max_tokens))
            ot = torch.randn((n_w, 8, 512), dtype=torch.bfloat16, device=device) * 0.1
            pt = torch.arange(n_w, dtype=torch.int64, device=device)
            cs = torch.ones(
                (max(64, int(max_tokens)), 64), dtype=torch.float32, device=device
            )
            dw = wo_projection.pack_weights(
                torch.zeros((1024, 4096), dtype=torch.float8_e4m3fn, device=device),
                torch.full((32, 128), 127, dtype=torch.uint8, device=device).view(
                    torch.float8_e8m0fnu
                ),
                torch.zeros((5120, 1024), dtype=torch.float8_e4m3fn, device=device),
                torch.full((160, 32), 127, dtype=torch.uint8, device=device).view(
                    torch.float8_e8m0fnu
                ),
                groups=1,
                group_width=4096,
                rank=1024,
                hidden=5120,
                block_size=(32, 32),
            )
            scr = [
                torch.empty(sp.shape, dtype=sp.dtype, device=device)
                for sp in _p.scratch_specs()
            ]
            bd = st_w.bind_inv_rope(
                scratch=scr,
                o=ot,
                positions=pt,
                cos_sin_cache=cs,
                weights=dw,
                heads_per_group=8,
                nope_dim=448,
                rope_dim=64,
            )
            return PreparedCall(
                run=lambda: st_w.run_inv_rope(bd),
                produce=lambda: ot.normal_(std=0.1),
            )

        requests.append(
            pwo_dyn.request(
                name="dsv41_wo_inv_rope_dyn",
                prepare_call=_mk_wodyn,
                benchmark_call=_mk_wodyn,
            )
        )

    shape_plans: dict[tuple[int, int], tuple[Any, ...]] = {}
    for (out_f, in_f), rep_w in sorted(shape_rep_weight.items()):
        plans_for_shape: list[Any] = []
        for cap in caps_tuple:
            p = block_fp8_linear.plan(
                block_fp8_linear.Caps(
                    device=device,
                    max_tokens=cap,
                    in_features=in_f,
                    out_features=out_f,
                    block_size=(32, 32),
                )
            )
            plans_for_shape.append(p)

            def _make_lin_call(state: Any, *, _w: Any = rep_w) -> PreparedCall:
                q = state.query
                src = (
                    torch.randn(
                        (q.max_tokens, q.in_features),
                        dtype=torch.bfloat16,
                        device=device,
                    )
                    * 0.25
                )
                out = torch.empty(
                    (q.max_tokens, q.out_features, 1),
                    dtype=torch.bfloat16,
                    device=device,
                )
                scr = [
                    torch.empty(sp.shape, dtype=sp.dtype, device=device)
                    for sp in state.scratch.scratch_specs()
                ]
                b = state.bind(
                    scratch=scr, source=src, packed_weight=_w, output=out
                )
                return PreparedCall(
                    run=lambda: state.run_binding(b),
                    produce=lambda: src.normal_(std=0.25),
                )

            requests.append(
                p.request(
                    name=f"dsv41_fp8_{out_f}x{in_f}_m{cap}",
                    prepare_call=_make_lin_call,
                    benchmark_call=_make_lin_call,
                )
            )
        shape_plans[(out_f, in_f)] = tuple(plans_for_shape)

    # 2b. Build shared `b12x.attention.compressed_sparse_mla` plans (decode CG + long-prefill extend).
    csmla_caps: tuple[int, ...] = tuple(sorted(int(b) for b in decode_buckets if int(b) > 0))
    csmla_plans: dict[tuple[int, int, int, bool, str], tuple[Any, ...]] = {}
    csmla_requests: list[Any] = []
    ext_cap = min(8192, int(max_tokens))
    try:
        from b12x.attention import compressed_sparse_mla as _csmla

        csmla_sigs: set[tuple[int, int, int, bool]] = set()
        for _mod in model.modules():
            if hasattr(_mod, "swa_cache_layer") and getattr(_mod, "swa_bytes_per_token", 0) == 584:
                h_n = int(getattr(_mod, "padded_heads", getattr(_mod, "n_local_heads", 0)))
                sw_n = int(getattr(_mod, "window_size", 0))
                tk_buf = getattr(_mod, "topk_indices_buffer", None)
                iw_n = int(tk_buf.shape[-1]) if (int(getattr(_mod, "compress_ratio", 0)) > 0 and tk_buf is not None) else 0
                if h_n > 0 and sw_n > 0 and (iw_n > 0 or sw_n == 128):
                    csmla_sigs.add((h_n, sw_n, iw_n, getattr(_mod, "attn_sink", None) is not None))
        if csmla_sigs and csmla_caps:
            dummy_kv = torch.zeros((16, 64 * 584), dtype=torch.uint8, device=device)
            for h_n, sw_n, iw_n, has_sink in sorted(csmla_sigs):
                d_sink = torch.zeros((h_n,), dtype=torch.float32, device=device) if has_sink else None
                modes = (("decode", csmla_caps), ("extend", (ext_cap,))) if (iw_n >= 2048 and ext_cap >= 2048) else (("decode", csmla_caps),)
                for mode_s, m_caps in modes:
                    sig_plans: list[Any] = []
                    for cap in m_caps:
                        dq = torch.empty((cap, h_n, 512), dtype=torch.bfloat16, device=device)
                        inv = _csmla.invocation_from_tensors(q=dq, swa_k_cache=dummy_kv, indexed_k_cache=dummy_kv if iw_n > 0 else None, attn_sink=d_sink, return_lse=False, out=torch.empty_like(dq))
                        ckw: dict[str, Any] = dict(device=device, num_q_heads=h_n, max_q_rows=cap, max_width=sw_n + iw_n, swa_width=sw_n, swa_page_size=64, indexed_width=iw_n, page_size=64, mode=mode_s)
                        if iw_n > 0:
                            ckw["indexed_page_size"] = 64
                        cp = _csmla.plan(_csmla.Caps(**ckw), invocation=inv)
                        sig_plans.append(cp)

                        def _mk_csmla(state: Any, *, _p: Any = cp, _c: int = cap, _h: int = h_n, _sw: int = sw_n, _iw: int = iw_n, _hs: bool = has_sink) -> PreparedCall:
                            q_t = torch.randn((_c, _h, 512), dtype=torch.bfloat16, device=device) * 0.1
                            o_t, kv_t = torch.empty_like(q_t), torch.zeros((16, 64 * 584), dtype=torch.uint8, device=device)
                            si = torch.arange(_sw, dtype=torch.int32, device=device).view(1, _sw).expand(_c, _sw).contiguous()
                            sl = torch.full((_c,), _sw, dtype=torch.int32, device=device)
                            ii = torch.arange(_iw, dtype=torch.int32, device=device).view(1, _iw).expand(_c, _iw).contiguous() if _iw > 0 else None
                            il = torch.full((_c,), _iw, dtype=torch.int32, device=device) if _iw > 0 else None
                            sk = torch.zeros((_h,), dtype=torch.float32, device=device) if _hs else None
                            sc = [torch.empty(sp.shape, dtype=sp.dtype, device=device) for sp in _p.scratch_specs()]
                            bd = state.bind(_p, scratch=sc, q=q_t, swa_indices=si, swa_lengths=sl, indexed_indices=ii, indexed_lengths=il)
                            return PreparedCall(run=lambda: state.run(bd, swa_k_cache=kv_t, indexed_k_cache=kv_t if _iw > 0 else None, sm_scale=512**-0.5, attn_sink=sk, out=o_t), produce=lambda: q_t.normal_(std=0.1))

                        csmla_requests.append(cp.request(name=f"dsv41_csmla_{mode_s}_h{h_n}_s{sw_n}_i{iw_n}_m{cap}", prepare_call=_mk_csmla, benchmark_call=_mk_csmla))
                    csmla_plans[(h_n, sw_n, iw_n, has_sink, mode_s)] = tuple(sig_plans)
            requests.extend(csmla_requests)
    except Exception as exc:
        _LOG.warning("b12x compressed_sparse_mla preparation skipped: %s", exc)
        csmla_plans.clear()

    # 3. Run `PreparationSession` on rank 0 FIRST so compilation & GPU autotuning
    # happen free of multi-rank filesystem races or GPU power throttling, then
    # let ranks 1..7 load the warm persistent cache in <1s.
    # IMPORTANT: Do NOT call `sess.close()` here — `sess.prepare()` already drains
    # and closes its `CompilePool`, whereas `sess.close()` clears `plan._prepared`
    # and evicts unretained compiled programs before CUDAGraph capture & serving!
    b12x_fp = _compute_b12x_source_fingerprint()
    default_cache_name = (
        "b12x_zstep_dsv41_v2"
        if b12x_fp == "default"
        else f"b12x_zstep_dsv41_v2_{b12x_fp}"
    )
    cfg_env = WorkerRuntimeConfig.from_env()
    cache_dir = cfg_env.b12x_cache_dir or str(
        Path.home() / ".cache" / default_cache_name
    )
    autotune = cfg_env.b12x_autotune

    def _run_session() -> Any:
        sess = PreparationSession(
            device=rank,
            autotune=autotune,
            cache_dir=cache_dir,
        )
        try:
            sess.prepare(tuple(requests))
            return sess
        except BaseException:
            sess.close()
            raise

    active_sess: Any = None
    dev_idx = 0 if device.index is None else int(device.index)
    try:
        if exchange_group is not None and dist.is_initialized():
            if rank == 0:
                active_sess = _run_session()
            dist.barrier(group=exchange_group)
            if rank != 0:
                active_sess = _run_session()
            dist.barrier(group=exchange_group)
        else:
            active_sess = _run_session()

        torch.cuda.empty_cache()

        # 4. Pre-allocate static 1024B-aligned scratch arenas (zero-alloc CUDAGraph capture & replay).
        moe_scratch_bytes = 0
        if moe_layers:
            for _mname, _layer, qm in moe_layers:
                moe_scratch_bytes = max(
                    moe_scratch_bytes, _specs_nbytes(qm._b12x_eplan.scratch_specs())
                )
            moe_arena = torch.empty(
                max(1024, moe_scratch_bytes), dtype=torch.uint8, device=device
            )
            d_hs = int(moe_layers[0][2].hidden_size)
            d_tk = int(moe_layers[0][1].moe_config.experts_per_token)
            d_ne = int(moe_layers[0][2].num_experts)
            d_x = torch.zeros((int(max_tokens), d_hs), dtype=torch.bfloat16, device=device)
            d_out = torch.empty((int(max_tokens), d_hs), dtype=torch.bfloat16, device=device)
            d_ids = (
                (torch.arange(int(max_tokens) * d_tk, device=device, dtype=torch.int32) % d_ne)
                .view(int(max_tokens), d_tk)
                .contiguous()
            )
            d_wts = torch.full(
                (int(max_tokens), d_tk), 1.0 / d_tk, dtype=torch.float32, device=device
            )
            for _mname, _layer, qm in moe_layers:
                qm._b12x_moe_state = require_prepared(qm._b12x_eplan, "moe.decode")
                qm._b12x_scratch = tuple(
                    _slice_arena_for_specs(moe_arena, qm._b12x_eplan.scratch_specs())
                )
                qm._b12x_moe_arena = moe_arena
                qm._b12x_moe_by_m = [None] * (int(max_tokens) + 1)
                qm._b12x_moe_capi_by_kern = {}
                qm._b12x_dev_idx = dev_idx
                vm = qm._b12x_moe_state.variants
                a16 = int(qm._b12x_moe_state.a16_max_tokens)
                vm_s = sorted(v for v in vm if 0 < v <= int(max_tokens))
                for vc in vm_s:
                    qm._capture_fast_moe_m(
                        _layer, d_x[:vc], d_wts[:vc], d_ids[:vc], d_out[:vc], vc
                    )
                for m_i in range(1, int(max_tokens) + 1) if vm_s else ():
                    if qm._b12x_moe_by_m[m_i] is None:
                        src_vc = (
                            a16
                            if 0 < m_i <= a16 <= int(max_tokens) and a16 in vm
                            else min((v for v in vm_s if v >= m_i), default=vm_s[-1])
                        )
                        qm._b12x_moe_by_m[m_i] = qm._b12x_moe_by_m[src_vc]
            del d_x, d_out, d_ids, d_wts

        # Separate linear scratch arenas for main stream (slot 0) vs `SharedExperts` aux stream (slot 1).
        slot_max_bytes: dict[int, int] = {0: 0, 1: 0}
        for skey, plans in shape_plans.items():
            max_for_shape = max(_specs_nbytes(p.scratch_specs()) for p in plans)
            for slot in shape_streams[skey]:
                slot_max_bytes[slot] = max(slot_max_bytes[slot], max_for_shape)

        slot_arenas: dict[int, torch.Tensor] = {}
        linear_scratch_bytes = 0
        for slot, nbytes in slot_max_bytes.items():
            if nbytes > 0 or any(slot in s for s in shape_streams.values()):
                alloc_bytes = max(1024, nbytes)
                slot_arenas[slot] = torch.empty(
                    alloc_bytes, dtype=torch.uint8, device=device
                )
                linear_scratch_bytes += alloc_bytes

        slot_shape_scratches: dict[
            tuple[int, tuple[int, int]], tuple[list[torch.Tensor], ...]
        ] = {}
        slot_shape_fast_plans: dict[
            tuple[int, tuple[int, int]], tuple[tuple[Any, ...], ...]
        ] = {}
        for skey, plans in shape_plans.items():
            for slot in shape_streams[skey]:
                arena = slot_arenas[slot]
                scratches_for_shape = tuple(
                    _slice_arena_for_specs(arena, p.scratch_specs()) for p in plans
                )
                slot_shape_scratches[(slot, skey)] = scratches_for_shape
                fast_for_shape: list[tuple[Any, ...]] = []
                for cap, p, scr_list in zip(
                    caps_tuple, plans, scratches_for_shape, strict=True
                ):
                    state = require_prepared(p, "gemm.block_fp8_linear")
                    splan = state.scratch
                    scr_u8 = scr_list[0]
                    ws: torch.Tensor | None = None
                    if splan.workspace_nbytes:
                        ws_off = _align_up(
                            _block_fp8_linear_scratch_layout(
                                tokens=splan.caps.max_tokens,
                                in_features=splan.caps.in_features,
                                out_features=splan.caps.out_features,
                                output_dtype=splan.caps.output_dtype,
                            ).nbytes,
                            _SCRATCH_ALIGN_BYTES,
                        )
                        ws = scr_u8.narrow(0, ws_off, splan.workspace_nbytes).view(
                            torch.float32
                        )
                    exact_x_q = _block_fp8_linear_x_q_from_scratch(
                        scr_u8,
                        tokens=int(cap),
                        in_features=int(splan.caps.in_features),
                        output_dtype=splan.caps.output_dtype,
                    )
                    fast_for_shape.append(
                        (
                            p,
                            state,
                            scr_u8,
                            ws,
                            int(cap),
                            splan.mma_tiler_mn,
                            exact_x_q,
                        )
                    )
                slot_shape_fast_plans[(slot, skey)] = tuple(fast_for_shape)

        for _lname, _layer, qm in linear_layers:
            skey = (int(qm._out_features), int(qm._in_features))
            slot = 1 if qm._is_shared_expert else 0
            qm._b12x_caps = caps_tuple
            qm._b12x_plans = shape_plans[skey]
            qm._b12x_scratches = slot_shape_scratches[(slot, skey)]
            qm._b12x_fast_plans = slot_shape_fast_plans[(slot, skey)]

        if pwo_dyn is not None and wo_attn_mods and (0, (1024, 4096)) in slot_shape_fast_plans:
            st_wodyn = require_prepared(pwo_dyn, "gemm.wo_projection")
            wo_fast_a = slot_shape_fast_plans[(0, (1024, 4096))]
            wo_max_tok = int(max_tokens)
            wo_z_buf = torch.empty((wo_max_tok, 1024), dtype=torch.bfloat16, device=device)
            for attn_m in wo_attn_mods:
                pk_a = attn_m.wo_a._b12x_wo_a_packed
                orig_op = attn_m._o_proj
                rhs_a = (pk_a.weight.values.view(1024, 4096, 1), pk_a.weight.scale_mma)

                def _fast_o_proj(
                    attn_out: torch.Tensor,
                    positions: torch.Tensor,
                    *,
                    _am: Any = attn_m,
                    _rhs: Any = rhs_a,
                    _orig: Any = orig_op,
                ) -> torch.Tensor:
                    o = attn_out[:, :8, :]
                    m_r = int(attn_out.shape[0])
                    if (
                        0 < m_r <= wo_max_tok
                        and o.dtype == torch.bfloat16
                        and positions.dtype == torch.int64
                        and positions.is_contiguous()
                    ):
                        cap_idx = next(
                            (i_c for i_c, c_v in enumerate(caps_tuple) if c_v >= m_r),
                            len(caps_tuple) - 1,
                        )
                        _pl, st_a, scr_a, ws_a, cap_a, _tm, ex_xq = wo_fast_a[cap_idx]
                        xq_a = (
                            ex_xq
                            if m_r == cap_a
                            else _block_fp8_linear_x_q_from_scratch(
                                scr_a,
                                tokens=m_r,
                                in_features=4096,
                                output_dtype=torch.bfloat16,
                            )
                        )
                        z_m = wo_z_buf[:m_r]
                        st_wodyn.quantize_a_inv_rope(
                            o if o.is_contiguous() else o.contiguous(),
                            positions,
                            _am.rotary_emb.cos_sin_cache,
                            groups=1,
                            heads_per_group=8,
                            nope_dim=448,
                            rope_dim=64,
                            out=xq_a,
                        )
                        st_a.dense.run(
                            (xq_a.values.view(m_r, 4096, 1), xq_a.scale_mma),
                            _rhs,
                            out=z_m.view(m_r, 1024, 1),
                            split_k_workspace=ws_a,
                        )
                        return _am._wo_b_proj(z_m)
                    return _orig(attn_out, positions)

                attn_m._zstep_orig_o_proj = orig_op
                attn_m._o_proj = _fast_o_proj
            model._b12x_wo_z_buf = wo_z_buf  # type: ignore[attr-defined]

        csmla_arena: torch.Tensor | None = None
        if csmla_plans:
            owner_id = id(model)
            csmla_bytes = max(_specs_nbytes(p.scratch_specs()) for plans in csmla_plans.values() for p in plans)
            csmla_arena = torch.empty(max(1024, csmla_bytes), dtype=torch.uint8, device=device)
            dev_tbl_new: dict[tuple[int, int, int, bool, str], tuple[tuple[int, Any, Any, Any, int], ...]] = {}
            for sig, plans in csmla_plans.items():
                m_caps = (ext_cap,) if sig[4] == "extend" else csmla_caps
                entries_list: list[tuple[int, Any, Any, Any, int]] = []
                for cap, p in zip(m_caps, plans, strict=True):
                    st_p = require_prepared(p, "attention.compressed_sparse_mla")
                    sc_l = _slice_arena_for_specs(csmla_arena, p.scratch_specs())
                    if sig[4] == "extend":
                        dq = torch.empty((cap, sig[0], 512), dtype=torch.bfloat16, device=device)
                        si_d = torch.zeros((cap, sig[1]), dtype=torch.int32, device=device)
                        sl_d = torch.zeros((cap,), dtype=torch.int32, device=device)
                        ii_d = torch.zeros((cap, sig[2]), dtype=torch.int32, device=device)
                        sc_l = st_p.bind(p, scratch=sc_l, q=dq, swa_indices=si_d, swa_lengths=sl_d, indexed_indices=ii_d, indexed_lengths=sl_d).scratch
                    entries_list.append((int(cap), p, st_p, sc_l, owner_id))
                dev_tbl_new[sig] = tuple(entries_list)
            try:
                from b12x.attention._shared.mla.compressed_api import _run_sm120_compressed_prefill as _b12x_prefill
            except ImportError:
                _b12x_prefill = None
            from vllm.forward_context import get_forward_context as _get_fc
            from vllm.models.deepseek_v41.nvidia import model as _dsv41_m

            _smla_mod = sys.modules.get(_dsv41_m.DeepseekV4FlashInferSM120Attention.__module__, _dsv41_m)
            if not getattr(_smla_mod, "_zstep_b12x_csmla_patched", False):
                orig_smla = _smla_mod.flashinfer_trtllm_batch_decode_sparse_mla_dsv4

                def _patched_smla(**kw: Any) -> Any:
                    q, swa_kv, out = kw.get("query"), kw.get("swa_kv_cache"), kw.get("out")
                    comp_kv, swa_lens = kw.get("compressed_kv_cache"), kw.get("swa_topk_lens")
                    swa_i = kw.get("sparse_indices")
                    ex_i = kw.get("extra_sparse_indices")
                    if (
                        q is not None and out is not None and swa_kv is not None and swa_lens is not None and swa_i is not None
                        and (q.ndim == 3 or (q.ndim == 4 and int(q.shape[1]) == 1))
                        and q.shape == out.shape
                        and kw.get("kv_layout", "NHD") == "NHD"
                        and q.dtype == torch.bfloat16 and out.dtype == torch.bfloat16
                        and q.is_contiguous() and out.is_contiguous()
                        and q.shape[-1] == 512 and out.shape[-1] == 512
                        and swa_i.is_contiguous() and (ex_i is None or ex_i.is_contiguous())
                        and swa_kv.is_contiguous() and swa_kv.shape[-1] == 584 and swa_kv.numel() % (64 * 584) == 0
                        and (comp_kv is None or (comp_kv.is_contiguous() and comp_kv.shape[-1] == 584 and comp_kv.numel() % (64 * 584) == 0))
                    ):
                        dev_tbl = _CSMLA_STATE_BY_DEV.get(0 if q.device.index is None else int(q.device.index))
                        if dev_tbl is not None:
                            rows, h_n = int(q.shape[0]), int(q.shape[-2])
                            swa_i2 = swa_i.view(rows, -1)
                            idx_i2 = ex_i.view(rows, -1) if ex_i is not None else None
                            iw_n = int(idx_i2.shape[1]) if idx_i2 is not None else 0
                            sinks = kw.get("sinks")
                            if torch.cuda.is_current_stream_capturing():
                                entries = dev_tbl.get((h_n, int(swa_i2.shape[1]), iw_n, sinks is not None, "decode"))
                                if entries is not None and (iw_n == 0 or comp_kv is not None) and (iw_n > 0 or rows <= 64):
                                    for cap_n, pl, st_p, sc_l, _own in entries:
                                        if cap_n >= rows:
                                            bd = st_p.bind(pl, scratch=sc_l, q=q.view(rows, h_n, 512), swa_indices=swa_i2, swa_lengths=swa_lens, indexed_indices=idx_i2, indexed_lengths=kw.get("extra_sparse_topk_lens"))
                                            return st_p.run(bd, swa_k_cache=swa_kv.view(-1, 64 * 584), indexed_k_cache=comp_kv.view(-1, 64 * 584) if (iw_n > 0 and comp_kv is not None) else None, sm_scale=float(kw.get("bmm1_scale", 1.0)), attn_sink=sinks, out=out.view(rows, h_n, 512))
                            elif _b12x_prefill is not None and rows >= 2048 and iw_n >= 2048 and comp_kv is not None:
                                entries = dev_tbl.get((h_n, int(swa_i2.shape[1]), iw_n, sinks is not None, "extend"))
                                if entries is not None and entries[0][0] >= rows:
                                    am = _get_fc().attn_metadata
                                    msl = max((int(getattr(v, "max_seq_len", 0) or 0) for v in am.values()), default=0) if isinstance(am, dict) else 0
                                    if msl >= 2048:
                                        _, _, st_p, ext_ws, _own = entries[0]
                                        return _b12x_prefill(q3=q.view(rows, h_n, 512), swa_k_cache=swa_kv.view(-1, 64 * 584), swa_indices=swa_i2, swa_topk_lengths=swa_lens, workspace=ext_ws, sm_scale=float(kw.get("bmm1_scale", 1.0)), swa_page_size=64, indexed_k_cache=comp_kv.view(-1, 64 * 584), indexed_indices=idx_i2, indexed_topk_lengths=kw.get("extra_sparse_topk_lens"), indexed_page_size=64, attn_sink=sinks, return_lse=False, lse_scale="base2", out=out.view(rows, h_n, 512), prepared=st_p.launch)
                    return orig_smla(**kw)

                _smla_mod.flashinfer_trtllm_batch_decode_sparse_mla_dsv4 = _patched_smla
                _smla_mod._zstep_b12x_csmla_patched = True  # type: ignore[attr-defined]

            merged_tbl = _CSMLA_STATE_BY_DEV.setdefault(dev_idx, {})
            for sig, new_entries in dev_tbl_new.items():
                existing = tuple(e for e in merged_tbl.get(sig, ()) if e[4] != owner_id)
                combined = existing + new_entries
                reverse_order = sig[4] == "extend"
                merged_tbl[sig] = tuple(sorted(combined, key=lambda e: e[0], reverse=reverse_order))
            model._b12x_csmla_dev = dev_idx  # type: ignore[attr-defined]

        model._b12x_session = active_sess  # type: ignore[attr-defined]
        model._b12x_scratch_arenas = (  # type: ignore[attr-defined]
            moe_arena if moe_layers else None,
            slot_arenas,
            csmla_arena,
        )
    except BaseException:
        _evict_csmla_owner(dev_idx, id(model))
        model._b12x_csmla_dev = None  # type: ignore[attr-defined]
        if active_sess is not None:
            active_sess.close()
        raise

    prepare_ms = (time.perf_counter() - t0) * 1000.0
    return B12xPrepareSummary(
        enabled=True,
        moe_layers=len(moe_layers),
        linear_layers=len(linear_layers),
        linear_shapes=len(shape_plans),
        moe_scratch_bytes=moe_scratch_bytes,
        linear_scratch_bytes=linear_scratch_bytes,
        prepare_ms=prepare_ms,
        mega_gate_layers=mg_layers,
        mega_gate_schedules=mg_schedules,
        mega_gate_min_tokens=mg_min_tokens,
        dsa_decode_enabled=dsa_enabled,
    )


def close_b12x_dsv41_model(model: torch.nn.Module | None) -> None:
    """Release retained `b12x`, `sm12x_mega_gate`, and `sm120_dsa_indexer` state during teardown."""
    if model is not None:
        sess = getattr(model, "_b12x_session", None)
        if sess is not None:
            model._b12x_session = None  # type: ignore[attr-defined]
            sess.close()
        if hasattr(model, "_b12x_scratch_arenas"):
            model._b12x_scratch_arenas = None  # type: ignore[attr-defined]
        if hasattr(model, "_b12x_wo_z_buf"):
            model._b12x_wo_z_buf = None  # type: ignore[attr-defined]
        for sub_m in model.modules():
            if hasattr(sub_m, "_zstep_orig_o_proj"):
                sub_m._o_proj = sub_m._zstep_orig_o_proj  # type: ignore[attr-defined]
                delattr(sub_m, "_zstep_orig_o_proj")
        if hasattr(model, "_sm120_mega_gate_state"):
            model._sm120_mega_gate_state = None  # type: ignore[attr-defined]
        for target_mod in (model, getattr(model, "model", None)):
            if isinstance(target_mod, torch.nn.Module) and hasattr(
                target_mod, "_zstep_sm120_orig_forward"
            ):
                target_mod.forward = target_mod._zstep_sm120_orig_forward  # type: ignore[method-assign]
                delattr(target_mod, "_zstep_sm120_orig_forward")
                target_mod._zstep_sm120_mhc_fwd_wrapped = False  # type: ignore[attr-defined]
        mhc_dev = getattr(model, "_sm120_mhc_dev", None)
        if mhc_dev is not None:
            st = _MHC_STATE_BY_DEV.get(mhc_dev)
            if st is not None:
                owners = getattr(st, "owners", None)
                if isinstance(owners, set):
                    owners.discard(id(model))
                    if not owners:
                        _MHC_STATE_BY_DEV.pop(mhc_dev, None)
                else:
                    _MHC_STATE_BY_DEV.pop(mhc_dev, None)
            model._sm120_mhc_dev = None  # type: ignore[attr-defined]
        csmla_dev = getattr(model, "_b12x_csmla_dev", None)
        if csmla_dev is not None:
            _evict_csmla_owner(csmla_dev, id(model))
            model._b12x_csmla_dev = None  # type: ignore[attr-defined]

    dsa_api = sys.modules.get("flashinfer.experimental.sm120_dsa_indexer._api")
    if dsa_api is not None:
        for attr in ("_LOGITS", "_LONG_WS", "_RETIRED"):
            container = getattr(dsa_api, attr, None)
            if container is not None and hasattr(container, "clear"):
                container.clear()
