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

"""Selective SM120 (`sm_120`) `b12x` compute integration for DeepSeek-V4.1-Flash.

Integrates the two winning SM120 `b12x` operator families into `zstep`'s Mode B
`DeepSeekV41StepRunner` while keeping `zstep`'s Rust SHM control plane, `shm_v5`
NUMA-local collectives (`M <= 256` + 100% CUDAGraph), `NCCL_LOC` (`M > 256`),
and vLLM 0.31.0's `mhc_shifted_post_pre` / `GateLinear` / `ParallelLMHead` /
FlashInfer sparse MLA+SWA untouched:

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
"""

from __future__ import annotations

from dataclasses import dataclass
import logging
import math
import os
from pathlib import Path
import sys
import time
from typing import Any

import torch
import torch.distributed as dist

_LOG = logging.getLogger(__name__)

_DEFAULT_COMPAT_CANDIDATES: tuple[str, ...] = (
    "/usr/local/lib/python3.12/dist-packages/flashinfer/experimental/b12x/_compat",
)

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


def _env_flag(name: str, default: bool = True) -> bool:
    raw = os.environ.get(name)
    if raw is None:
        return default
    return raw.strip().lower() not in ("0", "false", "no", "off")


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


def prepare_b12x_dsv41_model(
    model: torch.nn.Module,
    *,
    rank: int,
    device: torch.device,
    max_tokens: int,
    decode_buckets: tuple[int, ...],
    exchange_group: dist.ProcessGroup | None = None,
) -> B12xPrepareSummary:
    """Compile/autotune all `b12x` MoE & Block32 FP8 plans (rank 0 first) and bind static scratch."""
    t0 = time.perf_counter()
    if not maybe_enable_b12x_for_dsv41():
        return B12xPrepareSummary(False, 0, 0, 0, 0, 0, 0.0)

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
        return B12xPrepareSummary(False, 0, 0, 0, 0, 0, 0.0)

    caps_set = set(int(b) for b in decode_buckets if int(b) > 0)
    for extra_c in (256, 512, 1024, 2048, 3072, int(max_tokens)):
        if 0 < extra_c <= int(max_tokens):
            caps_set.add(extra_c)
    caps_set.add(int(max_tokens))
    caps_tuple: tuple[int, ...] = tuple(sorted(caps_set))

    moe_warmup_counts = tuple(
        sorted({1, 2, 4, 8, 16, 32, 64, 128, 512, min(3072, int(max_tokens)), int(max_tokens)})
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
    cache_dir = os.environ.get(
        "ZSTEP_B12X_CACHE_DIR",
        str(Path.home() / ".cache" / "b12x_zstep_dsv41_v2"),
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
    )


def close_b12x_dsv41_model(model: torch.nn.Module | None) -> None:
    """Close any retained `b12x` `PreparationSession` attached to `model` during runner teardown."""
    if model is None:
        return
    sess = getattr(model, "_b12x_session", None)
    if sess is not None:
        model._b12x_session = None  # type: ignore[attr-defined]
        sess.close()
