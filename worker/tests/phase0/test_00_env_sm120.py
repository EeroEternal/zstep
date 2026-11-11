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

"""Phase 0 Step 0.1: Verify 8x sm_120 (RTX 5090 / RTX 6000D) toolchain, P2P, and b12x."""

from __future__ import annotations

import importlib.util
import json


def check_sm120_cluster(expected_gpus: int) -> dict[str, object]:
    """Inspect visible GPUs, compute capability (12, 0), VRAM, P2P, and sm_120 libs."""
    import torch
    import triton

    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is not available on this host")

    count = torch.cuda.device_count()
    if count != expected_gpus:
        raise RuntimeError(f"Expected {expected_gpus} GPUs, found {count}")

    devices: list[dict[str, object]] = []
    p2p_matrix: list[list[bool]] = []
    any_cross_gpu_p2p = False

    for idx in range(count):
        props = torch.cuda.get_device_properties(idx)
        cap = (props.major, props.minor)
        if cap != (12, 0):
            raise RuntimeError(
                f"GPU {idx} ({props.name}) has capability {cap}, expected (12, 0)"
            )
        devices.append(
            {
                "index": idx,
                "name": props.name,
                "capability": f"{cap[0]}.{cap[1]}",
                "sm_count": props.multi_processor_count,
                "total_vram_gb": round(props.total_memory / (1024**3), 2),
            }
        )
        row: list[bool] = []
        for peer in range(count):
            if idx == peer:
                row.append(True)
            else:
                ok = bool(torch.cuda.can_device_access_peer(idx, peer))
                if ok:
                    any_cross_gpu_p2p = True
                row.append(ok)
        p2p_matrix.append(row)

    has_flashinfer_b12x = (
        importlib.util.find_spec("flashinfer.experimental.b12x") is not None
    )
    has_cutlass_dsl = importlib.util.find_spec("cutlass") is not None

    flashinfer_version = None
    if importlib.util.find_spec("flashinfer") is not None:
        import flashinfer

        flashinfer_version = flashinfer.__version__

    return {
        "torch_version": torch.__version__,
        "cuda_version": torch.version.cuda,
        "triton_version": triton.__version__,
        "flashinfer_version": flashinfer_version,
        "has_flashinfer_b12x": has_flashinfer_b12x,
        "has_cutlass_dsl": has_cutlass_dsl,
        "direct_bar1_p2p_available": any_cross_gpu_p2p,
        "recommended_allreduce_backend": (
            "pcie_bar1_p2p_twoshot"
            if any_cross_gpu_p2p
            else "numa_host_mapped_ddio_allreduce"
        ),
        "devices": devices,
        "p2p_matrix": p2p_matrix,
    }


if __name__ == "__main__":
    report = check_sm120_cluster(expected_gpus=8)
    print(json.dumps(report, indent=2))
