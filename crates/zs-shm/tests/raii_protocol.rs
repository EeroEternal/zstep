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
use zs_wire::{
    SeqStepOutput, ShmGeometry, StepHeader, StepOpcode, StepResultHeader, VramGeometrySpec,
};

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

#[test]
fn unlink_now_removes_shm_entry_while_existing_mappings_remain_live() {
    let name = format!("zstep_test_{}_unlink_now", std::process::id());
    let object_path = Path::new("/dev/shm").join(&name);
    let layout = ShmLayout::compute(geometry()).expect("layout");

    let mut creator = SharedMemoryArena::create(&name, layout.total_bytes).expect("create");
    creator.initialize(&layout).expect("initialize");
    assert!(object_path.exists());

    let reader = SharedMemoryArena::open(&name, layout.total_bytes, false).expect("open readonly");
    creator.unlink_now().expect("unlink_now");
    assert!(
        !object_path.exists(),
        "unlink_now must remove /dev/shm entry immediately after workers attach"
    );

    // Existing mappings on both creator and attached worker remain readable/writable.
    {
        let mut plan = creator.plan_slot_mut(&layout, 0).expect("mutable plan");
        plan.header_mut().expect("plan header").num_encoder_tokens = 7;
        plan.publish(2).expect("publish");
    }
    let observed = reader.plan_slot(&layout, 0).expect("reader plan");
    assert_eq!(observed.published_step_id().expect("step id"), 2);
    assert_eq!(
        observed.header().expect("plan header").num_encoder_tokens,
        7
    );
}

#[test]
fn readonly_mapping_rejects_mutable_views_and_abort_flag_propagates() {
    let name = format!("zstep_test_{}_ro_guard", std::process::id());
    let layout = ShmLayout::compute(geometry()).expect("layout");

    let mut creator = SharedMemoryArena::create(&name, layout.total_bytes).expect("create");
    creator.initialize(&layout).expect("initialize");
    let mut readonly =
        SharedMemoryArena::open(&name, layout.total_bytes, false).expect("open readonly");
    creator.unlink_now().expect("unlink_now");

    assert!(
        readonly.try_bytes_mut().is_err(),
        "PROT_READ mapping must reject mutable byte slices instead of risking SIGSEGV"
    );
    assert!(readonly.plan_slot_mut(&layout, 0).is_err());
    assert!(readonly.result_slot_mut(&layout, 0).is_err());
    assert!(readonly.signal_abort().is_err());

    assert!(!readonly.is_aborted().expect("is_aborted"));
    creator.signal_abort().expect("signal_abort");
    assert!(readonly.is_aborted().expect("is_aborted"));
}

#[test]
fn cross_language_golden_layout_and_hash_parity() {
    // Must match worker/tests/fixtures/shm_layout_golden.json byte-for-byte.
    struct GoldenCase {
        name: &'static str,
        geom: ShmGeometry,
        expected_offsets: [usize; 14],
        total_slot_bytes: usize,
        result_slot_stride: usize,
        layout_hash: u64,
    }

    let qwen_geom = ShmGeometry::from_vram_budget(VramGeometrySpec {
        world_size: 8,
        per_gpu_vram_mb: 32_607,
        reserved_weights_and_workspace_mb: 12_288,
        kv_bytes_per_token_per_gpu: 10_240,
        max_seqs: 128,
        max_step_tokens: 8_192,
        max_decoder_rows: 8_192,
        page_block_tokens: 64,
        spec_verify_width: 4,
    })
    .expect("qwen35 32g vram geometry");

    let cases = [
        GoldenCase {
            name: "dsv41_tp8_default",
            geom: ShmGeometry {
                world_size: 8,
                max_seqs: 128,
                max_step_tokens: 4096,
                max_decoder_rows: 4096,
                max_pages_per_seq: 256,
                page_block_tokens: 128,
                spec_verify_width: 1,
            },
            expected_offsets: [
                0, 256, 16640, 33024, 49408, 53504, 102656, 119040, 135424, 135936, 136704, 267776,
                269824, 270336,
            ],
            total_slot_bytes: 274432,
            result_slot_stride: 16384,
            layout_hash: 0x74f0_b51b_0370_8051,
        },
        GoldenCase {
            name: "qwen35_tp8_vram32g",
            geom: qwen_geom,
            expected_offsets: [
                0, 256, 33024, 65792, 98560, 106752, 205056, 237824, 270592, 271104, 271872,
                2369024, 2371072, 2371584,
            ],
            total_slot_bytes: 2375680,
            result_slot_stride: 16384,
            layout_hash: 0x85ff_7508_8e32_5cd2,
        },
        GoldenCase {
            name: "odd_prime_bounds",
            geom: ShmGeometry {
                world_size: 1,
                max_seqs: 3,
                max_step_tokens: 19,
                max_decoder_rows: 7,
                max_pages_per_seq: 11,
                page_block_tokens: 64,
                spec_verify_width: 1,
            },
            expected_offsets: [
                0, 256, 512, 768, 1024, 1280, 1536, 1792, 2048, 2304, 2560, 2816, 3072, 3328,
            ],
            total_slot_bytes: 4096,
            result_slot_stride: 4096,
            layout_hash: 0xc582_f8c7_0110_8dc7,
        },
    ];

    for c in cases {
        let shm = ShmLayout::compute(c.geom).expect(c.name);
        let p = shm.plan;
        let actual = [
            p.header_offset,
            p.token_ids_offset,
            p.positions_offset,
            p.req_slot_ids_offset,
            p.replay_mask_offset,
            p.engram_prev_window_offset,
            p.narrow_keep_indices_offset,
            p.sample_row_indices_offset,
            p.seq_lens_offset,
            p.query_start_loc_offset,
            p.global_kv_page_table_offset,
            p.swa_page_table_offset,
            p.compressor_slot_ids_offset,
            p.temperatures_offset,
        ];
        assert_eq!(actual, c.expected_offsets, "{} offsets mismatch", c.name);
        assert_eq!(
            p.total_slot_bytes, c.total_slot_bytes,
            "{} slot bytes",
            c.name
        );
        assert_eq!(
            shm.result_slot_stride, c.result_slot_stride,
            "{} result stride",
            c.name
        );
        assert_eq!(
            p.layout_hash(shm.result_slot_stride),
            c.layout_hash,
            "{} layout_hash mismatch",
            c.name
        );
    }
}
