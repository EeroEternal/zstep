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

Integrates the measured winning SM120 operator families into `zstep`'s Mode B
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
     `M >= ZSTEP_MEGA_GATE_MIN_TOKENS` (default `32`), keeping small `M < 32` decode
     batches on `GateLinear` + `dsv4_topk` and accelerating `M >= 32` decode and
     eager prefill (`1.44x-3.27x` faster across `M = 256..3072`).
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

import torch
import torch.distributed as dist

_LOG = logging.getLogger(__name__)

_DEFAULT_COMPAT_CANDIDATES: tuple[str, ...] = (
    "/usr/local/lib/python3.12/dist-packages/flashinfer/experimental/b12x/_compat",
)

SM120_DSA_CANDIDATE_COVER_TOKENS: int = 16384
DEFAULT_MEGA_GATE_MIN_TOKENS: int = 32

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


def _env_flag(name: str, default: bool = True) -> bool:
    raw = os.environ.get(name)
    if raw is None:
        return default
    return raw.strip().lower() not in ("0", "false", "no", "off")


def _env_int(name: str, default: int) -> int:
    raw = os.environ.get(name)
    if raw is None or not raw.strip():
        return default
    return int(raw.strip())


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
    if not _env_flag("ZSTEP_ENABLE_SM120_DSA_DECODE", default=True):
        return False
    if not torch.cuda.is_available():
        return False
    if torch.cuda.get_device_capability() != (12, 0):
        return False
    try:
        from flashinfer.experimental.sm120_dsa_indexer._api import (  # noqa: F401
            dsa_indexer_det_topk_paged,
        )
        from flashinfer.utils import is_sm120f_supported

        dev = torch.device("cuda", torch.cuda.current_device())
        return bool(is_sm120f_supported(dev))
    except (ImportError, RuntimeError, AttributeError):
        return False


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
    require_b12x = _env_flag("ZSTEP_REQUIRE_B12X", default=False)
    explicit_enable = os.environ.get("ZSTEP_ENABLE_B12X") is not None
    if not _env_flag("ZSTEP_ENABLE_B12X", default=True):
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

    custom_compat = os.environ.get("ZSTEP_B12X_COMPAT_DIR", "").strip()
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
            "Mount `flashinfer/experimental/b12x` from "
            "local-inference-lab/flashinfer@79819b112f8bda7d09709c24bb31de16d7ce59c6 "
            "or set ZSTEP_ENABLE_B12X=0."
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
                    intermediate_size=self.intermediate_size,
                ),
            )
            unit_scale = torch.ones(
                self.num_experts, dtype=torch.float32, device=device
            )
            prepared = fused_moe.prepare_weights(
                plan=wplan,
                weights=fused_moe.PackedWeights(
                    w13=layer.w13_weight.data.contiguous(),
                    w2=layer.w2_weight.data.contiguous(),
                    w13_block_scales=layer.w13_weight_scale.data.contiguous(),
                    w2_block_scales=layer.w2_weight_scale.data.contiguous(),
                    w13_global_scales=unit_scale,
                    w2_global_scales=unit_scale,
                    input_scale=unit_scale,
                    intermediate_scale=unit_scale,
                    immutable_input_scales=True,
                ),
            )
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
                return orig_get_quant_method(self, layer, prefix)
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
            if self.expert_dtype == "fp4" and self.moe_quant_algo != "NVFP4":
                return ZstepB12xMxfp4MoEMethod(layer.moe_config)

        return orig_get_quant_method(self, layer, prefix)

    cfg_cls.get_quant_method = _patched_get_quant_method
    cfg_cls._zstep_b12x_patched = True
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
        if candidate_blocks is not None:
            cand_cover = int(candidate_blocks.shape[1]) * int(self.candidate_block_size)
            if compressed_max_seq > cand_cover:
                return orig_forward_cuda(self, hidden_states, q_quant, k, weights)

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
        device, max_rows, min(int(_row_sel_keys()), SM120_DSA_CANDIDATE_COVER_TOKENS)
    )
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
    if not _env_flag("ZSTEP_ENABLE_SM12X_MEGA_GATE", default=True):
        return 0, 0, 0
    min_tokens = _env_int("ZSTEP_MEGA_GATE_MIN_TOKENS", DEFAULT_MEGA_GATE_MIN_TOKENS)
    if min_tokens <= 0 or min_tokens > int(max_tokens):
        return 0, 0, 0
    if not torch.cuda.is_available() or torch.cuda.get_device_capability() != (12, 0):
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
        if rank == 0:
            _LOG.warning("SM12x MegaGate found 0 eligible MoERunner layers")
        return 0, 0, 0

    enable_pdl = _env_flag("ZSTEP_MEGA_GATE_ENABLE_PDL", default=False)
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


def prepare_b12x_dsv41_model(
    model: torch.nn.Module,
    *,
    rank: int,
    device: torch.device,
    max_tokens: int,
    decode_buckets: tuple[int, ...],
    exchange_group: dist.ProcessGroup | None = None,
) -> B12xPrepareSummary:
    """Compile/autotune all SM120 `b12x`, `sm12x_mega_gate` & `sm120_dsa_indexer` plans (rank 0 first)."""
    t0 = time.perf_counter()
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
    for extra_c in (256, 512, 1024, 2048, 3072, int(max_tokens)):
        if 0 < extra_c <= int(max_tokens):
            caps_set.add(extra_c)
    caps_set.add(int(max_tokens))
    caps_tuple: tuple[int, ...] = tuple(sorted(caps_set))

    moe_warmup_counts = tuple(
        sorted(
            {
                1,
                2,
                4,
                8,
                16,
                32,
                64,
                128,
                512,
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
    cache_dir = os.environ.get(
        "ZSTEP_B12X_CACHE_DIR",
        str(Path.home() / ".cache" / default_cache_name),
    )
    autotune = _env_flag("ZSTEP_B12X_AUTOTUNE", default=True)

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
            for _mname, _layer, qm in moe_layers:
                qm._b12x_moe_state = require_prepared(qm._b12x_eplan, "moe.decode")
                qm._b12x_scratch = tuple(
                    _slice_arena_for_specs(moe_arena, qm._b12x_eplan.scratch_specs())
                )

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

        model._b12x_session = active_sess  # type: ignore[attr-defined]
        model._b12x_scratch_arenas = (  # type: ignore[attr-defined]
            moe_arena if moe_layers else None,
            slot_arenas,
        )
    except BaseException:
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
        if hasattr(model, "_sm120_mega_gate_state"):
            model._sm120_mega_gate_state = None  # type: ignore[attr-defined]

    dsa_api = sys.modules.get("flashinfer.experimental.sm120_dsa_indexer._api")
    if dsa_api is not None:
        for attr in ("_LOGITS", "_LONG_WS", "_RETIRED"):
            container = getattr(dsa_api, attr, None)
            if container is not None and hasattr(container, "clear"):
                container.clear()
