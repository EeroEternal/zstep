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

use std::path::Path;
use std::sync::atomic::{AtomicU64, Ordering};

use zs_shm::{SharedMemoryArena, ShmLayout};
use zs_wire::{SeqStepOutput, ShmGeometry, StepHeader, StepOpcode, StepResultHeader};

fn geometry() -> ShmGeometry {
    ShmGeometry {
        world_size: 1,
        max_seqs: 2,
        max_step_tokens: 8,
        max_decoder_rows: 8,
        max_pages_per_seq: 4,
        page_block_tokens: 64,
        spec_verify_width: 6,
    }
}

#[test]
fn typed_views_publish_and_creator_unlinks_on_drop() {
    let name = format!("zstep_test_{}_raii", std::process::id());
    let object_path = Path::new("/dev/shm").join(&name);
    let layout = ShmLayout::compute(geometry()).expect("layout");
    {
        let mut arena = SharedMemoryArena::create(&name, layout.total_bytes).expect("create arena");
        arena.initialize(&layout).expect("initialize arena");
        assert!(object_path.exists());
        assert_eq!(arena.superblock().expect("superblock").world_size, 1);
        {
            let mut plan = arena.plan_slot_mut(&layout, 1).expect("mutable plan");
            *plan.header_mut().expect("plan header") = StepHeader {
                step_id: AtomicU64::new(0),
                opcode: StepOpcode::NormalStep as u32,
                num_seqs: 1,
                num_encoder_tokens: 3,
                encoder_graph_bucket: 0,
                num_decoder_rows: 3,
                decoder_graph_bucket: 0,
                num_sample_rows: 1,
                is_dspark_verify: 0,
                flags: 0,
                _reserved: [0; 3],
            };
            plan.token_ids_mut().expect("token payload")[..3].copy_from_slice(&[7, 11, 13]);
            plan.publish(41).expect("publish plan");
        }
        let plan = arena.plan_slot(&layout, 1).expect("read plan");
        assert_eq!(plan.published_step_id().expect("acquire plan"), 41);
        assert_eq!(plan.header().expect("plan header").num_encoder_tokens, 3);

        {
            let mut result = arena.result_slot_mut(&layout, 1).expect("mutable result");
            *result.header_mut().expect("result header") = StepResultHeader {
                completed_step_id: AtomicU64::new(0),
                gpu_exec_us: 17,
                status_code: 0,
                num_seq_results: 1,
                _pad: [0; 11],
            };
            result.outputs_mut().expect("result outputs")[0] = SeqStepOutput {
                req_slot: 0,
                num_accepted: 1,
                accepted_tokens: [23, -1, -1, -1, -1, -1, -1, -1],
                next_draft_tokens: [-1; 8],
                accepted_logprobs_bits: [0; 8],
            };
            result.publish(41).expect("publish result");
        }
        let result = arena.result_slot(&layout, 1).expect("read result");
        assert_eq!(result.completed_step_id().expect("acquire result"), 41);
        assert_eq!(result.outputs().expect("outputs")[0].accepted_tokens[0], 23);
        assert_eq!(
            arena
                .worker_state(0)
                .expect("worker state")
                .status_code
                .load(Ordering::Relaxed),
            0
        );
    }
    assert!(
        !object_path.exists(),
        "creator must unlink the POSIX object"
    );
}
