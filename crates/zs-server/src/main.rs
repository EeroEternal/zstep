// Copyright (c) 2026 zstep Contributors
//
// Permission is hereby granted, free of charge, to any person obtaining a copy
// of this software and associated documentation files (the "Software"), to deal
// in the Software without restriction, including without limitation the rights
// to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
// copies of the Software, and to permit persons to whom the Software is
// furnished to do so, subject to the following conditions:
//
// The above copyright notice and this permission notice shall be included in all
// copies or substantial portions of the Software.
//
// THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
// IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
// FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
// AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
// LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
// OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
// SOFTWARE.

use zs_wire::{ShmGeometry, StepPlanLayout, ZSTEP_PROTOCOL_VERSION, ZSTEP_SHM_MAGIC};

fn main() {
    let geometry = ShmGeometry {
        world_size: 8,
        max_seqs: 64,
        max_step_tokens: 16384,
        max_decoder_rows: 8192,
        max_pages_per_seq: 2048,
        page_block_tokens: 64,
        spec_verify_width: 6,
    };
    let plan_layout = StepPlanLayout::compute(geometry);
    println!(
        "zstep master (wire magic=0x{ZSTEP_SHM_MAGIC:016X}, v{ZSTEP_PROTOCOL_VERSION}): \
         8x RTX 6000D StepPlanSlot = {} KiB",
        plan_layout.total_slot_bytes / 1024
    );
}
