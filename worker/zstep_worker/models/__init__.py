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

"""Explicit real-weight model runners for zstep Mode B workers."""

from __future__ import annotations

__all__ = ["DeepSeekV41StepRunner", "Qwen35MoeStepRunner"]


def __getattr__(name: str):
    if name == "DeepSeekV41StepRunner":
        from zstep_worker.models.deepseek_v41 import DeepSeekV41StepRunner

        return DeepSeekV41StepRunner
    if name == "Qwen35MoeStepRunner":
        from zstep_worker.models.qwen35_moe import Qwen35MoeStepRunner

        return Qwen35MoeStepRunner
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
