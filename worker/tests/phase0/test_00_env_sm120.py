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

"""Phase 0 Step 0.1: Verify 8x RTX 6000D (sm_120) toolchain, P2P matrix, and b12x."""

from __future__ import annotations

import importlib.util


def check_sm120_cluster(expected_gpus: int) -> dict[str, object]:
    """Inspect visible GPUs, compute capability (12, 0), VRAM, and P2P access."""
    import torch

    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is not available on this host")

    count = torch.cuda.device_count()
    if count != expected_gpus:
        raise RuntimeError(f"Expected {expected_gpus} GPUs, found {count}")

    devices: list[dict[str, object]] = []
    p2p_matrix: list[list[bool]] = []

    for idx in range(count):
        props = torch.cuda.get_device_properties(idx)
        cap = (props.major, props.minor)
        if cap != (12, 0):
            raise RuntimeError(
                f"GPU {idx} ({props.name}) has capability {cap}, expected (12, 0) for RTX 6000D"
            )
        devices.append(
            {
                "index": idx,
                "name": props.name,
                "capability": cap,
                "sm_count": props.multi_processor_count,
                "total_vram_gb": round(props.total_memory / (1024**3), 2),
            }
        )
        row: list[bool] = []
        for peer in range(count):
            if idx == peer:
                row.append(True)
            else:
                row.append(bool(torch.cuda.can_device_access_peer(idx, peer)))
        p2p_matrix.append(row)

    has_flashinfer_b12x = (
        importlib.util.find_spec("flashinfer.experimental.b12x") is not None
    )
    has_cutlass_dsl = importlib.util.find_spec("cutlass") is not None

    return {
        "devices": devices,
        "p2p_matrix": p2p_matrix,
        "has_flashinfer_b12x": has_flashinfer_b12x,
        "has_cutlass_dsl": has_cutlass_dsl,
    }


if __name__ == "__main__":
    report = check_sm120_cluster(expected_gpus=8)
    for dev in report["devices"]:  # type: ignore[union-attr]
        print(f"GPU {dev}")
    print(f"P2P matrix: {report['p2p_matrix']}")
    print(
        f"flashinfer.experimental.b12x={report['has_flashinfer_b12x']}, "
        f"cutlass_dsl={report['has_cutlass_dsl']}"
    )
