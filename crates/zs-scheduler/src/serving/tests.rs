// Copyright (c) 2026 zstep Contributors
// SPDX-License-Identifier: MIT

use super::*;

#[test]
fn slot_pool_raii_returns_slots_on_drop_and_unwind() {
    let pool = SlotPool::new(4);
    assert_eq!(pool.available(), 4);
    let s0 = pool.allocate().expect("slot 0");
    let s1 = pool.allocate().expect("slot 1");
    assert_eq!(s0.id(), 0);
    assert_eq!(s1.id(), 1);
    assert_eq!(pool.available(), 2);
    drop(s0);
    assert_eq!(pool.available(), 3);

    let pool_clone = pool.clone();
    let unwind_res = std::panic::catch_unwind(move || {
        let _s_panic = pool_clone.allocate().expect("panic slot");
        assert_eq!(pool_clone.available(), 2);
        panic!("simulated panic while holding SlotLease");
    });
    assert!(unwind_res.is_err());
    assert_eq!(pool.available(), 3);
    drop(s1);
    assert_eq!(pool.available(), 4);
}

#[test]
fn active_sequence_kv_page_growth_and_raii_release() {
    let slot_pool = SlotPool::new(2);
    let kv_pool = BlockPool::new(CacheGroupKind::GlobalAndIndex, 8);
    let mut shadow_pt = vec![-1i32; 2 * 4];

    {
        let mut seq = ActiveSequence {
            request_id: 0,
            slot_lease: slot_pool.allocate().expect("slot"),
            prefix_lease: None,
            kv_leases: Vec::new(),
            page_ids: Vec::new(),
            prompt_tokens: vec![42],
            prefilled_tokens: 1,
            computed_prompt_tokens: 1,
            cached_prefix_tokens: 0,
            restore_checkpoint_id: -1,
            pending_checkpoints: Vec::new(),
            checkpoint_alloc_stopped: false,
            current_seq_len: 0,
            last_token: 42,
            generated_tokens: Vec::new(),
            generated_top2: Vec::new(),
            pending_draft_tokens: Vec::new(),
            staged_verify: None,
            target_output_len: 10,
            reserved_pages: 0,
            permit_time: Instant::now(),
            submit_time: Instant::now(),
            first_token_time: None,
            last_token_time: None,
        };
        let mut dirty_pt_slots = [0u128; 2];
        seq.ensure_kv_pages(17, 16, 4, &kv_pool, &mut shadow_pt, &mut dirty_pt_slots)
            .expect("allocate 2 pages");
        assert_eq!(seq.page_ids.len(), 2);
        assert_eq!(kv_pool.available(), 6);
        assert_eq!(shadow_pt[0], seq.page_ids[0]);
        assert_eq!(shadow_pt[1], seq.page_ids[1]);
        assert_eq!(dirty_pt_slots, [1, 1]);

        dirty_pt_slots = [0, 0];
        // Growing from 17 -> 32 stays within the existing 2 pages (2 * 16 = 32)
        seq.ensure_kv_pages(32, 16, 4, &kv_pool, &mut shadow_pt, &mut dirty_pt_slots)
            .expect("still 2 pages");
        assert_eq!(kv_pool.available(), 6);
        assert_eq!(dirty_pt_slots, [0, 0]);

        // Token 33 crosses boundary and allocates page 3
        seq.ensure_kv_pages(33, 16, 4, &kv_pool, &mut shadow_pt, &mut dirty_pt_slots)
            .expect("allocate 3rd page");
        assert_eq!(seq.page_ids.len(), 3);
        assert_eq!(kv_pool.available(), 5);
        assert_eq!(dirty_pt_slots, [1, 1]);
    }

    // Dropping `seq` releases all 3 KV blocks and the sequence slot
    assert_eq!(kv_pool.available(), 8);
    assert_eq!(slot_pool.available(), 2);
}

#[test]
fn max_seqs_above_decode_graph_limit_is_rejected() {
    assert!(validate_serving_max_seqs(0).is_err());
    assert!(validate_serving_max_seqs(1).is_ok());
    assert!(validate_serving_max_seqs(128).is_ok());
    assert!(validate_serving_max_seqs(129).is_err());
    assert!(validate_serving_max_seqs(256).is_err());
}

#[test]
fn json_string_escapes_rfc8259_and_preserves_utf8() {
    assert_eq!(
        json_string("DeepSeek-V4.1 \"中文\" \\ \n\t\x01\x1f"),
        "\"DeepSeek-V4.1 \\\"中文\\\" \\\\ \\n\\t\\u0001\\u001f\""
    );
}

#[test]
fn overlapped_decode_lookback_includes_token_from_inflight_step() {
    let slot_pool = SlotPool::new(1);
    let now = Instant::now();
    let mut seq = ActiveSequence {
        request_id: 0,
        slot_lease: slot_pool.allocate().expect("slot"),
        prefix_lease: None,
        kv_leases: Vec::new(),
        page_ids: Vec::new(),
        prompt_tokens: vec![10, 11, 12, 13],
        prefilled_tokens: 4,
        computed_prompt_tokens: 4,
        cached_prefix_tokens: 0,
        restore_checkpoint_id: -1,
        pending_checkpoints: Vec::new(),
        checkpoint_alloc_stopped: false,
        current_seq_len: 4,
        last_token: 13,
        generated_tokens: Vec::new(),
        generated_top2: Vec::new(),
        pending_draft_tokens: Vec::new(),
        staged_verify: None,
        target_output_len: 4,
        reserved_pages: 0,
        permit_time: now,
        submit_time: now,
        first_token_time: None,
        last_token_time: None,
    };

    // Commit-true staging of the first decode, matching `stage_decode_step`.
    let inflight_pos = seq.current_seq_len;
    assert_eq!(seq.lookback_window(inflight_pos), [11, 12, 13]);
    seq.current_seq_len = inflight_pos + 1;

    // Pre-stage of the next step runs before harvest, so index 4 is absent.
    let stale = seq.lookback_window(seq.current_seq_len);
    assert_eq!(stale, [12, 13, -1]);

    seq.last_token = 99;
    seq.generated_tokens.push(99);
    let (token, repaired) = seq.revise_overlapped_decode();
    assert_eq!(token, 99);
    assert_eq!(repaired, [12, 13, 99]);
    assert_eq!(seq.current_seq_len, 6);
}

#[test]
fn chunked_prefill_progresses_to_exact_prompt_end_and_samples_only_final_chunk() {
    let prompt_len = 10;
    let mut prefilled_tokens = 0;
    let mut ranges = Vec::new();
    while let Some((start, end)) = prefill_chunk_range(prompt_len, prefilled_tokens, 6, 4) {
        assert!(end - start <= 4);
        ranges.push((start, end));
        prefilled_tokens = end;
    }
    assert_eq!(ranges, vec![(0, 4), (4, 8), (8, 10)]);
    assert_eq!(prefilled_tokens, prompt_len);
    let sample_chunks = ranges.iter().filter(|(_, end)| *end == prompt_len).count();
    assert_eq!(sample_chunks, 1);
    assert_eq!(prefill_chunk_range(10, 4, 2, 4), Some((4, 6)));
    assert_eq!(prefill_chunk_range(10, 10, 6, 4), None);
}

#[test]
fn decode_sequences_force_single_prefill_interleaving() {
    let now = Instant::now();
    let waiting: VecDeque<PendingRequest> = (0..4)
        .map(|request_id| PendingRequest {
            request_id,
            prompt_tokens: vec![256; 1024],
            target_output_len: 8,
            permit_time: now,
            submit_time: now,
        })
        .collect();
    assert!(should_schedule_prefill(
        1,
        1,
        false,
        8,
        4096,
        2048,
        &waiting,
        0,
        DEFAULT_PREFILL_COALESCE_WAIT_MS,
    ));
    assert!(!should_schedule_prefill(
        1,
        1,
        false,
        8,
        4096,
        2048,
        &waiting,
        1,
        DEFAULT_PREFILL_COALESCE_WAIT_MS,
    ));
    assert!(should_schedule_prefill(
        1,
        2,
        true,
        8,
        4096,
        2048,
        &waiting,
        0,
        DEFAULT_PREFILL_COALESCE_WAIT_MS,
    ));
    assert!(!should_schedule_prefill(
        1,
        2,
        true,
        8,
        4096,
        2048,
        &waiting,
        1,
        DEFAULT_PREFILL_COALESCE_WAIT_MS,
    ));
    assert!(should_schedule_prefill(
        0,
        1,
        false,
        8,
        4096,
        2048,
        &waiting,
        4,
        DEFAULT_PREFILL_COALESCE_WAIT_MS,
    ));
}

#[test]
fn ttft_and_submit_ttft_use_their_respective_start_times() {
    let submit_time = Instant::now();
    let permit_time = submit_time + std::time::Duration::from_millis(40);
    let first_token_time = submit_time + std::time::Duration::from_millis(100);
    let (ttft_us, submit_ttft_us) =
        first_token_latency_us(permit_time, submit_time, first_token_time);
    assert_eq!(ttft_us, 60_000);
    assert_eq!(submit_ttft_us, 100_000);
}

#[test]
fn tier_aggregates_include_permit_and_submit_ttft_population_stats() {
    let results = (0..3)
        .map(|repeat| BenchmarkTierResult {
            label: "tier".to_owned(),
            repeat,
            seed: BENCHMARK_RANDOM_SEED + repeat as u64,
            concurrency: 4,
            num_prompts: 12,
            input_len: 512,
            output_len: 256,
            steps: 100,
            overlapped_decode_steps: 90,
            wall_time_ms: 1_000.0,
            output_throughput_tok_s: 100.0,
            total_throughput_tok_s: 300.0,
            mean_ttft_ms: 40.0,
            median_ttft_ms: 40.0,
            p99_ttft_ms: 45.0,
            mean_submit_ttft_ms: 100.0 * (repeat + 1) as f64,
            median_submit_ttft_ms: 100.0 * (repeat + 1) as f64,
            p99_submit_ttft_ms: 110.0 * (repeat + 1) as f64,
            mean_tpot_ms: 4.0,
            median_tpot_ms: 4.0,
            p99_tpot_ms: 5.0,
            mean_itl_ms: 4.0,
            median_itl_ms: 4.0,
            p99_itl_ms: 6.0,
        })
        .collect::<Vec<_>>();
    let aggregate = TierAggregateStats::from_results("tier".to_owned(), &results)
        .expect("three repeat samples");
    assert_eq!(aggregate.repeats, 3);
    assert_eq!(aggregate.mean_ttft_ms.mean, 40.0);
    assert_eq!(aggregate.mean_submit_ttft_ms.mean, 200.0);
    assert!((aggregate.mean_submit_ttft_ms.std - 81.649_658).abs() < 1e-6);
    assert_eq!(aggregate.mean_submit_ttft_ms.min, 100.0);
    assert_eq!(aggregate.mean_submit_ttft_ms.max, 300.0);
    let encoded = aggregate.to_json();
    assert!(encoded.contains("\"median_submit_ttft_ms\""));
    assert!(encoded.contains("\"p99_submit_ttft_ms\""));
}

#[test]
fn random_prompts_are_distinct_and_in_valid_vocab_range() {
    let prompts = build_deterministic_random_prompts(16, 64);
    assert_eq!(prompts.len(), 16);
    assert_ne!(prompts[0], prompts[1]);
    for p in &prompts {
        assert_eq!(p.len(), 64);
        assert!(p.iter().all(|&tok| (256..65_256).contains(&tok)));
    }

    let round0 = build_deterministic_random_prompts_seeded(8, 64, BENCHMARK_RANDOM_SEED);
    let round1 = build_deterministic_random_prompts_seeded(8, 64, BENCHMARK_RANDOM_SEED + 1_000);
    let round0_again = build_deterministic_random_prompts_seeded(8, 64, BENCHMARK_RANDOM_SEED);
    assert_eq!(round0, round0_again);
    for (p0, p1) in round0.iter().zip(&round1) {
        assert_ne!(p0, p1);
    }
}

#[test]
fn engram_lookback_window_spans_prompt_boundary_and_generated_tokens() {
    let slot_pool = SlotPool::new(1);
    let mut seq = ActiveSequence {
        request_id: 0,
        slot_lease: slot_pool.allocate().expect("slot"),
        prefix_lease: None,
        kv_leases: Vec::new(),
        page_ids: Vec::new(),
        prompt_tokens: vec![101, 102],
        prefilled_tokens: 2,
        computed_prompt_tokens: 2,
        cached_prefix_tokens: 0,
        restore_checkpoint_id: -1,
        pending_checkpoints: Vec::new(),
        checkpoint_alloc_stopped: false,
        current_seq_len: 2,
        last_token: 201,
        generated_tokens: vec![201, 202],
        generated_top2: Vec::new(),
        pending_draft_tokens: Vec::new(),
        staged_verify: None,
        target_output_len: 8,
        reserved_pages: 0,
        permit_time: Instant::now(),
        submit_time: Instant::now(),
        first_token_time: None,
        last_token_time: None,
    };
    // Initial prefill start (`chunk_start = 0`)
    assert_eq!(seq.lookback_window(0), [-1, -1, -1]);
    // Chunked prefill at offset 1 (`chunk_start = 1`)
    assert_eq!(seq.lookback_window(1), [-1, -1, 101]);
    // Decode step 1 (`pos = 2`, feeding `201`)
    assert_eq!(seq.lookback_window(2), [-1, 101, 102]);
    // Overlapped decode step 2 (`pos = 3`, feeding `202`)
    assert_eq!(seq.lookback_window(3), [101, 102, 201]);
    // Steady-state decode step 3 (`pos = 4`)
    seq.generated_tokens.push(203);
    assert_eq!(seq.lookback_window(4), [102, 201, 202]);
    assert_eq!(seq.lookback_window(5), [201, 202, 203]);
}

#[test]
fn ngram_drafter_uses_latest_longest_suffix_and_respects_width() {
    let config = SpeculativeConfig {
        method: SpeculativeMethod::Ngram,
        num_speculative_tokens: 3,
        prompt_lookup_min: 2,
        prompt_lookup_max: 4,
    };
    let history = [1, 2, 3, 4, 9, 1, 2, 3, 4, 8, 7, 1, 2, 3, 4];
    assert_eq!(ngram_draft_tokens(&history, &config), vec![8, 7, 1]);

    let no_match = [1, 2, 3, 4, 5, 6, 7, 8];
    assert!(ngram_draft_tokens(&no_match, &config).is_empty());
}

#[test]
fn decode_continues_for_all_respects_inflight_prefill_and_waiting() {
    let slot_pool = SlotPool::new(2);
    let now = Instant::now();
    let make_seq =
        |slot_pool: &SlotPool, prefilled: usize, generated: usize, target: usize| ActiveSequence {
            request_id: 0,
            slot_lease: slot_pool.allocate().expect("slot"),
            prefix_lease: None,
            kv_leases: Vec::new(),
            page_ids: Vec::new(),
            prompt_tokens: vec![10, 20, 30, 40],
            prefilled_tokens: prefilled,
            computed_prompt_tokens: prefilled,
            cached_prefix_tokens: 0,
            restore_checkpoint_id: -1,
            pending_checkpoints: Vec::new(),
            checkpoint_alloc_stopped: false,
            current_seq_len: prefilled + generated.saturating_sub(1),
            last_token: 99,
            generated_tokens: vec![99; generated],
            generated_top2: Vec::new(),
            pending_draft_tokens: Vec::new(),
            staged_verify: None,
            target_output_len: target,
            reserved_pages: 0,
            permit_time: now,
            submit_time: now,
            first_token_time: None,
            last_token_time: None,
        };

    assert!(!decode_continues_for_all(&[], false, 0, false));

    // 1 generated out of 4: in_flight=0 -> 1+0+1=2 < 4 (true); in_flight=1 -> 1+1+1=3 < 4 (true);
    // in_flight=2 -> 1+2+1=4 < 4 (false, retires right after planned step).
    let mut active = vec![make_seq(&slot_pool, 4, 1, 4)];
    assert!(decode_continues_for_all(&active, false, 0, false));
    assert!(decode_continues_for_all(&active, false, 1, false));
    assert!(!decode_continues_for_all(&active, false, 2, false));

    // While prefill step is still in flight (`computed_prompt_tokens < prompt_len`),
    // APC blocks pre-promotion decode overlap, then re-enables steady-state overlap afterward.
    active[0].computed_prompt_tokens = 2;
    assert!(decode_continues_for_all(&active, false, 0, false));
    assert!(!decode_continues_for_all(&active, false, 0, true));
    active[0].computed_prompt_tokens = 4;
    assert!(decode_continues_for_all(&active, false, 0, true));

    // An admissible waiting request blocks overlap so prefill interleaving is not starved.
    assert!(!decode_continues_for_all(&active, true, 0, false));
    drop(active);

    // Partial prefill sequence in `active` blocks decode-only pre-publishing.
    let partial = vec![make_seq(&slot_pool, 2, 0, 4)];
    assert!(!decode_continues_for_all(&partial, false, 0, false));
}

#[test]
fn whole_request_kv_demand_excludes_unwritten_final_token() {
    // 512 prompt + 256 output: the 256th token is sampled but never written to KV.
    assert_eq!(request_kv_page_demand(512, 256, 16), 48);
    assert_eq!(request_kv_page_demand(512, 257, 16), 48);
    assert_eq!(request_kv_page_demand(512, 258, 16), 49);
    // Single-token output only needs the prompt pages.
    assert_eq!(request_kv_page_demand(17, 1, 16), 2);
    assert_eq!(request_kv_page_demand(16, 1, 16), 1);
    assert_eq!(request_kv_page_demand(512, 128, 128), 5);
    // Saturating arithmetic never wraps extreme usize lengths to zero.
    assert!(request_kv_page_demand(2, usize::MAX, 16) > 0);
}

#[test]
fn admission_reserves_whole_request_against_earlier_promises() {
    let slot_pool = SlotPool::new(2);
    let kv_pool = BlockPool::new(CacheGroupKind::GlobalAndIndex, 10);
    let mut shadow_pt = vec![-1i32; 2 * 8];
    let mut dirty = [0u128; 2];
    let now = Instant::now();
    // Admitted with a 6-page promise; step through 2 prefill chunks (16 -> 32) then decode.
    let mut seq = ActiveSequence {
        request_id: 0,
        slot_lease: slot_pool.allocate().expect("slot"),
        prefix_lease: None,
        kv_leases: Vec::new(),
        page_ids: Vec::new(),
        prompt_tokens: vec![1; 32],
        prefilled_tokens: 16,
        computed_prompt_tokens: 16,
        cached_prefix_tokens: 0,
        restore_checkpoint_id: -1,
        pending_checkpoints: Vec::new(),
        checkpoint_alloc_stopped: false,
        current_seq_len: 16,
        last_token: 1,
        generated_tokens: Vec::new(),
        generated_top2: Vec::new(),
        pending_draft_tokens: Vec::new(),
        staged_verify: None,
        target_output_len: 65,
        reserved_pages: request_kv_page_demand(32, 65, 16),
        permit_time: now,
        submit_time: now,
        first_token_time: None,
        last_token_time: None,
    };
    assert_eq!(seq.reserved_pages, 6);
    seq.ensure_kv_pages(16, 16, 8, &kv_pool, &mut shadow_pt, &mut dirty)
        .expect("first prefill chunk");
    let mut active = vec![seq];
    assert_eq!(kv_pool.available(), 9);
    assert_eq!(outstanding_reserved_pages(&active), 5);
    assert!(kv_demand_fits(kv_pool.available(), 5, 4));
    assert!(!kv_demand_fits(kv_pool.available(), 5, 5));

    active[0].prefilled_tokens = 32;
    active[0].current_seq_len = 32;
    active[0]
        .ensure_kv_pages(32, 16, 8, &kv_pool, &mut shadow_pt, &mut dirty)
        .expect("second prefill chunk");
    assert_eq!(kv_pool.available(), 8);
    assert_eq!(outstanding_reserved_pages(&active), 4);
    // 8 free pages minus 4 promised: a 4-page request fits, a 5-page one waits even
    // though its first chunk alone would fit.
    assert!(kv_demand_fits(kv_pool.available(), 4, 4));
    assert!(!kv_demand_fits(kv_pool.available(), 4, 5));

    // Growing within the promise never changes the unreserved headroom.
    active[0]
        .ensure_kv_pages(96, 16, 8, &kv_pool, &mut shadow_pt, &mut dirty)
        .expect("decode growth inside reservation");
    assert_eq!(kv_pool.available(), 4);
    assert_eq!(outstanding_reserved_pages(&active), 0);
    assert!(kv_demand_fits(kv_pool.available(), 0, 4));
    drop(active);
    assert_eq!(kv_pool.available(), 10);
}

#[test]
fn validate_request_kv_demand_rejects_empty_overflow_and_over_capacity() {
    // Exact boundary equality (8 pages == max_pages_per_seq == kv_capacity) is accepted.
    validate_request_kv_demand_with_limits([(64, 65)].into_iter(), 16, 8, 8)
        .expect("exact max_pages_per_seq and kv_capacity boundary fits");

    // Demand exceeding max_pages_per_seq is rejected even when kv_capacity is larger.
    let err_seq = validate_request_kv_demand_with_limits([(64, 66)].into_iter(), 16, 8, 64)
        .expect_err("exceeds max_pages_per_seq");
    assert!(err_seq.to_string().contains("max_pages_per_seq=8"));

    // Demand exceeding kv_capacity is rejected even when max_pages_per_seq is larger.
    let err_cap = validate_request_kv_demand_with_limits([(64, 66)].into_iter(), 16, 64, 8)
        .expect_err("exceeds num_kv_pages");
    assert!(err_cap.to_string().contains("num_kv_pages=8"));

    // Empty prompt anywhere in the batch is rejected pre-flight before poisoning the engine.
    let err_empty =
        validate_request_kv_demand_with_limits([(16, 4), (0, 4)].into_iter(), 16, 64, 64)
            .expect_err("empty prompt rejected");
    assert!(err_empty.to_string().contains("positive"));

    // Overflowing token sum is rejected instead of wrapping in release mode.
    let err_ovf = validate_request_kv_demand_with_limits([(2, usize::MAX)].into_iter(), 16, 64, 64)
        .expect_err("usize overflow rejected");
    assert!(err_ovf.to_string().contains("overflows"));
}

#[test]
fn inadmissible_waiting_request_keeps_overlap_open_until_retirement() {
    let slot_pool = SlotPool::new(2);
    let kv_pool = BlockPool::new(CacheGroupKind::GlobalAndIndex, 8);
    let mut shadow_pt = vec![-1i32; 2 * 8];
    let mut dirty = [0u128; 2];
    let now = Instant::now();

    let mut make_active = |request_id: usize| {
        let mut seq = ActiveSequence {
            request_id,
            slot_lease: slot_pool.allocate().expect("slot"),
            prefix_lease: None,
            kv_leases: Vec::new(),
            page_ids: Vec::new(),
            prompt_tokens: vec![1; 16],
            prefilled_tokens: 16,
            computed_prompt_tokens: 16,
            cached_prefix_tokens: 0,
            restore_checkpoint_id: -1,
            pending_checkpoints: Vec::new(),
            checkpoint_alloc_stopped: false,
            current_seq_len: 16,
            last_token: 1,
            generated_tokens: vec![99],
            generated_top2: Vec::new(),
            pending_draft_tokens: Vec::new(),
            staged_verify: None,
            target_output_len: 33, // demand = ceil((16 + 32) / 16) = 3 pages
            reserved_pages: request_kv_page_demand(16, 33, 16),
            permit_time: now,
            submit_time: now,
            first_token_time: None,
            last_token_time: None,
        };
        seq.ensure_kv_pages(17, 16, 8, &kv_pool, &mut shadow_pt, &mut dirty)
            .expect("initial kv pages");
        seq
    };

    // Active seq 0 promises 3 pages (2 allocated, 1 outstanding); 5 unreserved pages remain.
    let mut active = vec![make_active(0)];
    let mut waiting = VecDeque::new();
    waiting.push_back(PendingRequest {
        request_id: 1,
        prompt_tokens: vec![1; 32],
        target_output_len: 65, // demand = 6 pages > 5 unreserved pages
        permit_time: now,
        submit_time: now,
    });

    // Waiting request cannot fit yet -> waiting_admissible is false -> decode overlap stays on.
    let admissible_under_pressure =
        front_request_admissible_in_pools(&active, &waiting, &slot_pool, &kv_pool, None, 2, 16);
    assert!(!admissible_under_pressure);
    assert!(decode_continues_for_all(
        &active,
        admissible_under_pressure,
        0,
        false
    ));

    // Slot exhaustion also keeps waiting_admissible false even when KV pages fit.
    waiting[0].target_output_len = 17; // demand = ceil((32 + 16) / 16) = 3 pages <= 5
    active.push(make_active(2)); // consumes the 2nd slot; 8 - 6 = 2 unreserved pages
    assert_eq!(slot_pool.available(), 0);
    assert!(!front_request_admissible_in_pools(
        &active, &waiting, &slot_pool, &kv_pool, None, 2, 16
    ));

    // Retiring seq 2 frees both its slot and its 3-page reservation (5 unreserved >= 3 demand),
    // flipping waiting_admissible to true and halting decode overlap so prefill can interleave.
    active.pop();
    let admissible_after_retire =
        front_request_admissible_in_pools(&active, &waiting, &slot_pool, &kv_pool, None, 2, 16);
    assert!(admissible_after_retire);
    assert!(!decode_continues_for_all(
        &active,
        admissible_after_retire,
        0,
        false
    ));
}

#[test]
fn apc_admission_reservation_protects_evictable_headroom_and_shares_pinned_prefixes() {
    let slot_pool = SlotPool::new(4);
    let kv_pool = BlockPool::new(CacheGroupKind::GlobalAndIndex, 6);
    let mut tree = PrefixTree::with_pages_per_node(16, 1).expect("tree");
    let mut shadow_pt = vec![-1i32; 4 * 8];
    let mut dirty = [0u128; 2];
    let now = Instant::now();

    // Populate a 2-page cached prefix [7; 32] (unpinned, evictable_kv_pages = 2, free = 4).
    let cached_prompt = vec![7i32; 32];
    let p0 = kv_pool.allocate(1).expect("p0");
    let p1 = kv_pool.allocate(1).expect("p1");
    let warm_lease = tree
        .extend_prefix(
            tree.match_prefix_bounded(&[], 0),
            &cached_prompt,
            vec![p0, p1],
        )
        .expect("insert warm prefix");
    drop(warm_lease);
    assert_eq!(kv_pool.available(), 4);
    assert_eq!(tree.evictable_kv_pages(), 2);

    // Seq A (disjoint prompt, 16 tok + 65 out => demand 5 pages) allocates 1 page so far,
    // leaving 4 outstanding reserved pages. Reclaimable = 4 free + 2 evictable = 6 >= 5!
    let mut queries = 0u64;
    let mut hits = 0u64;
    let mut seq_a = admit_request_into_sequence(
        PendingRequest {
            request_id: 1,
            prompt_tokens: vec![9; 16],
            target_output_len: 65,
            permit_time: now,
            submit_time: now,
        },
        slot_pool.allocate().expect("slot a"),
        5,
        8,
        Some(&mut tree),
        false,
        &mut queries,
        &mut hits,
        &mut shadow_pt,
        &mut dirty,
    )
    .expect("admit seq a");
    let mut evictions = 0u64;
    seq_a
        .ensure_kv_pages_with_cache(
            16,
            16,
            8,
            &kv_pool,
            Some(&mut tree),
            &mut evictions,
            &mut shadow_pt,
            &mut dirty,
        )
        .expect("seq a first page");
    let mut active = vec![seq_a];
    assert_eq!(kv_pool.available(), 3);
    assert_eq!(tree.evictable_kv_pages(), 2);
    assert_eq!(outstanding_reserved_pages(&active), 4);

    // Candidate B hits the 2 unpinned cached pages (32 + 16 + 1 out => demand 3 pages,
    // matched 2 pages, extra_demand 1 page). Pinning those 2 unpinned pages would drop
    // reclaimable headroom from 5 to 3 < 4 outstanding reserved + 1 extra = 5 -> MUST WAIT!
    let mut prompt_b = cached_prompt.clone();
    prompt_b.extend_from_slice(&[8; 16]);
    let mut waiting = VecDeque::new();
    waiting.push_back(PendingRequest {
        request_id: 2,
        prompt_tokens: prompt_b.clone(),
        target_output_len: 1,
        permit_time: now,
        submit_time: now,
    });
    assert!(!front_request_admissible_in_pools(
        &active,
        &waiting,
        &slot_pool,
        &kv_pool,
        Some(&tree),
        4,
        16,
    ));

    // Seq A grows across its remaining 4 reserved pages (total 5 pages = 80 tokens), evicting
    // 1 of the 2 unpinned cached nodes on demand while leaving the root node cached; then a
    // 6th page (96 tokens) evicts the remaining cached root node!
    active[0]
        .ensure_kv_pages_with_cache(
            80,
            16,
            8,
            &kv_pool,
            Some(&mut tree),
            &mut evictions,
            &mut shadow_pt,
            &mut dirty,
        )
        .expect("seq a full decode growth");
    assert_eq!(evictions, 1);
    assert_eq!(kv_pool.available(), 0);
    assert_eq!(tree.cached_kv_pages(), 1);
    active[0]
        .ensure_kv_pages_with_cache(
            96,
            16,
            8,
            &kv_pool,
            Some(&mut tree),
            &mut evictions,
            &mut shadow_pt,
            &mut dirty,
        )
        .expect("seq a sixth page evicts remaining root");
    assert_eq!(evictions, 2);
    assert_eq!(kv_pool.available(), 0);
    assert_eq!(tree.cached_kv_pages(), 0);
    active.clear();
    assert_eq!(kv_pool.available(), 6);

    // Now admit two concurrent requests sharing a pinned prefix: the 2nd sees newly_pinned = 0!
    let p0 = kv_pool.allocate(1).expect("p0");
    let p1 = kv_pool.allocate(1).expect("p1");
    let warm2 = tree
        .extend_prefix(
            tree.match_prefix_bounded(&[], 0),
            &cached_prompt,
            vec![p0, p1],
        )
        .expect("reinsert prefix");
    drop(warm2);
    let req_b1 = waiting.pop_front().expect("b1");
    let (demand_b1, cached_b1, extra_b1, reclaimable_b1) =
        peek_request_admission(&req_b1.prompt_tokens, 1, 16, &kv_pool, Some(&tree));
    assert_eq!(
        (demand_b1, cached_b1, extra_b1, reclaimable_b1),
        (3, 32, 1, 4)
    );
    let mut seq_b1 = admit_request_into_sequence(
        req_b1,
        slot_pool.allocate().expect("slot b1"),
        demand_b1,
        8,
        Some(&mut tree),
        false,
        &mut queries,
        &mut hits,
        &mut shadow_pt,
        &mut dirty,
    )
    .expect("admit b1");
    seq_b1
        .ensure_kv_pages_with_cache(
            48,
            16,
            8,
            &kv_pool,
            Some(&mut tree),
            &mut evictions,
            &mut shadow_pt,
            &mut dirty,
        )
        .expect("b1 suffix page");
    active.push(seq_b1);
    assert_eq!(outstanding_reserved_pages(&active), 0);
    // Cached 2 pages are already pinned by b1 (`newly_pinned_pages == 0`), 3 pages remain free,
    // and b2 only needs 1 extra suffix page -> admissible concurrently!
    waiting.push_back(PendingRequest {
        request_id: 3,
        prompt_tokens: prompt_b,
        target_output_len: 17, // demand = ceil((48 + 16)/16) = 4, matched = 2, extra = 2 <= 3
        permit_time: now,
        submit_time: now,
    });
    assert!(front_request_admissible_in_pools(
        &active,
        &waiting,
        &slot_pool,
        &kv_pool,
        Some(&tree),
        4,
        16,
    ));
}

#[test]
fn duplicate_cold_promotion_canonicalizes_pages_and_hybrid_checkpoints_roundtrip() {
    // 1) Duplicate cold promotion (across two incremental chunks 0..16 then 16..40)
    //    canonicalizes the 2nd sequence's page_ids + shadow_pt and returns duplicate pages.
    let slot_pool = SlotPool::new(4);
    let kv_pool = BlockPool::new(CacheGroupKind::GlobalAndIndex, 16);
    let mut tree = PrefixTree::with_pages_per_node(16, 1).expect("tree");
    let mut shadow_pt = vec![-1i32; 4 * 8];
    let mut dirty = [0u128; 2];
    let now = Instant::now();
    let mut queries = 0u64;
    let mut hits = 0u64;
    let mut evictions = 0u64;

    let shared_prompt = vec![42i32; 40]; // 2 full 16-token nodes + 8-token tail (3 pages)
    let mut active = Vec::new();
    for request_id in 0..2 {
        let mut seq = admit_request_into_sequence(
            PendingRequest {
                request_id,
                prompt_tokens: shared_prompt.clone(),
                target_output_len: 8,
                permit_time: now,
                submit_time: now,
            },
            slot_pool.allocate().expect("slot"),
            3,
            8,
            Some(&mut tree),
            false,
            &mut queries,
            &mut hits,
            &mut shadow_pt,
            &mut dirty,
        )
        .expect("admit cold");
        seq.prefilled_tokens = 16;
        seq.current_seq_len = 16;
        seq.ensure_kv_pages_with_cache(
            16,
            16,
            8,
            &kv_pool,
            Some(&mut tree),
            &mut evictions,
            &mut shadow_pt,
            &mut dirty,
        )
        .expect("alloc chunk 1 page");
        active.push(seq);
    }
    promote_completed_prefixes_in_batch(
        &mut active,
        Some(&mut tree),
        false,
        false,
        1,
        8,
        &mut shadow_pt,
        &mut dirty,
    )
    .expect("promote chunk 1");
    assert_eq!(kv_pool.available(), 15); // 1 canonical page shared across both sequences!

    for seq in &mut active {
        seq.prefilled_tokens = 40;
        seq.current_seq_len = 40;
        seq.ensure_kv_pages_with_cache(
            40,
            16,
            8,
            &kv_pool,
            Some(&mut tree),
            &mut evictions,
            &mut shadow_pt,
            &mut dirty,
        )
        .expect("alloc chunk 2 pages");
    }
    assert_eq!(kv_pool.available(), 11); // 15 - 4 = 11
    dirty = [0, 0];
    promote_completed_prefixes_in_batch(
        &mut active,
        Some(&mut tree),
        false,
        false,
        1,
        8,
        &mut shadow_pt,
        &mut dirty,
    )
    .expect("promote duplicate batch chunk 2");
    // Duplicate 2nd prefix page from active[1] was freed immediately back to kv_pool!
    assert_eq!(kv_pool.available(), 12);
    assert_eq!(active[0].page_ids[..2], active[1].page_ids[..2]);
    let slot1 = active[1].slot_lease.id() as usize;
    assert_eq!(
        &shadow_pt[slot1 * 8..slot1 * 8 + 2],
        &active[0].page_ids[..2]
    );
    assert_ne!(dirty[0] & (1u128 << slot1), 0);
    active.clear();
    assert_eq!(kv_pool.available(), 14); // 2 prefix pages stay cached in `tree`
    assert_eq!(tree.evictable_kv_pages(), 2);

    // 2) Hybrid recurrent checkpointing (`pages_per_node = 4`, `16` tok/page -> `64` tok/node):
    let kv_hybrid = BlockPool::new(CacheGroupKind::GlobalAndIndex, 16);
    let ckpt_pool = BlockPool::new(CacheGroupKind::RecurrentState, 4);
    let mut hybrid_tree = PrefixTree::with_pages_per_node(64, 4).expect("hybrid tree");
    let mut hybrid_pt = vec![-1i32; 4 * 16];
    let mut hybrid_dirty = [0u128; 2];
    let hybrid_prompt: Vec<i32> = (0..140).map(|i| 300 + i).collect(); // 2 full 64-tok nodes + 12 tail
    let mut h_seq = admit_request_into_sequence(
        PendingRequest {
            request_id: 10,
            prompt_tokens: hybrid_prompt.clone(),
            target_output_len: 4,
            permit_time: now,
            submit_time: now,
        },
        slot_pool.allocate().expect("h slot"),
        9,
        16,
        Some(&mut hybrid_tree),
        true,
        &mut queries,
        &mut hits,
        &mut hybrid_pt,
        &mut hybrid_dirty,
    )
    .expect("admit hybrid cold");
    assert_eq!(h_seq.restore_checkpoint_id, -1);
    h_seq.prefilled_tokens = 140;
    h_seq.current_seq_len = 140;
    h_seq
        .ensure_kv_pages_with_cache(
            140,
            16,
            16,
            &kv_hybrid,
            Some(&mut hybrid_tree),
            &mut evictions,
            &mut hybrid_pt,
            &mut hybrid_dirty,
        )
        .expect("alloc 9 hybrid pages");
    let ckpt_64 = ckpt_pool.allocate(1).expect("ckpt 64");
    let ckpt_128 = ckpt_pool.allocate(1).expect("ckpt 128");
    let expected_ckpt_128_id = ckpt_128.ids()[0];
    h_seq.pending_checkpoints.push((64, ckpt_64));
    h_seq.pending_checkpoints.push((128, ckpt_128));
    let mut h_active = vec![h_seq];
    promote_completed_prefixes_in_batch(
        &mut h_active,
        Some(&mut hybrid_tree),
        true,
        false,
        4,
        16,
        &mut hybrid_pt,
        &mut hybrid_dirty,
    )
    .expect("promote hybrid");
    assert!(h_active[0].pending_checkpoints.is_empty());
    assert_eq!(h_active[0].kv_leases.len(), 1); // 9 - 8 promoted = 1 tail page
    h_active.clear();
    assert_eq!(kv_hybrid.available(), 8); // 8 pages cached in 2 nodes
    assert_eq!(ckpt_pool.available(), 2); // 2 recurrent checkpoints cached in 2 nodes

    // Warm hit on the same 140-token prompt reuses 128 tokens (8 KV pages) and restores
    // `expected_ckpt_128_id` for the remaining 12-token suffix chunk!
    let h_warm = admit_request_into_sequence(
        PendingRequest {
            request_id: 11,
            prompt_tokens: hybrid_prompt,
            target_output_len: 4,
            permit_time: now,
            submit_time: now,
        },
        slot_pool.allocate().expect("h warm slot"),
        9,
        16,
        Some(&mut hybrid_tree),
        true,
        &mut queries,
        &mut hits,
        &mut hybrid_pt,
        &mut hybrid_dirty,
    )
    .expect("admit hybrid warm");
    assert_eq!(h_warm.cached_prefix_tokens, 128);
    assert_eq!(h_warm.prefilled_tokens, 128);
    assert_eq!(h_warm.page_ids.len(), 8);
    assert_eq!(h_warm.restore_checkpoint_id, expected_ckpt_128_id);
    drop(h_warm);
    assert_eq!(hybrid_tree.evict_lru(2), 2);
    assert_eq!(kv_hybrid.available(), 16);
    assert_eq!(ckpt_pool.available(), 4);
}

#[test]
fn rewind_rejected_suffix_pages_reclaims_exclusive_pages_after_prefix_promotion() {
    let slot_pool = SlotPool::new(2);
    let kv_pool = BlockPool::new(CacheGroupKind::GlobalAndIndex, 8);
    let mut tree = PrefixTree::with_pages_per_node(16, 1).expect("prefix tree");
    let mut shadow_pt = vec![99i32; 2 * 8];
    let mut dirty = [0u128; 2];
    let mut queries = 0u64;
    let mut hits = 0u64;
    let mut evictions = 0u64;
    let now = Instant::now();

    // Cold 30-token prompt: 1 full 16-token prefix node + 14-token tail (2 KV pages total).
    let prompt: Vec<i32> = (0..30).map(|i| 500 + i).collect();
    let mut seq = admit_request_into_sequence(
        PendingRequest {
            request_id: 21,
            prompt_tokens: prompt,
            target_output_len: 16,
            permit_time: now,
            submit_time: now,
        },
        slot_pool.allocate().expect("slot"),
        4,
        8,
        Some(&mut tree),
        false,
        &mut queries,
        &mut hits,
        &mut shadow_pt,
        &mut dirty,
    )
    .expect("admit cold sequence");
    assert_eq!(seq.cached_prefix_tokens, 0);

    seq.prefilled_tokens = 30;
    seq.current_seq_len = 30;
    seq.ensure_kv_pages_with_cache(
        30,
        16,
        8,
        &kv_pool,
        Some(&mut tree),
        &mut evictions,
        &mut shadow_pt,
        &mut dirty,
    )
    .expect("alloc 2 prompt pages");
    assert_eq!(kv_pool.available(), 6);

    let mut active = vec![seq];
    promote_completed_prefixes_in_batch(
        &mut active,
        Some(&mut tree),
        false,
        false,
        1,
        8,
        &mut shadow_pt,
        &mut dirty,
    )
    .expect("promote 16-token prefix node");
    // Node 0 moved from `kv_leases` into `prefix_lease`, while `cached_prefix_tokens` stays 0.
    assert_eq!(active[0].cached_prefix_tokens, 0);
    assert_eq!(
        active[0].prefix_lease.as_ref().unwrap().block_ids().len(),
        1
    );
    assert_eq!(active[0].kv_leases.len(), 1);
    assert_eq!(active[0].page_ids.len(), 2);

    // Speculative verify stages 5 tokens (1 anchor + 4 drafts) up to seq_len = 35 -> 3 pages.
    active[0]
        .ensure_kv_pages_with_cache(
            35,
            16,
            8,
            &kv_pool,
            Some(&mut tree),
            &mut evictions,
            &mut shadow_pt,
            &mut dirty,
        )
        .expect("alloc 3rd page for cross-boundary draft");
    assert_eq!(kv_pool.available(), 5);
    assert_eq!(active[0].kv_leases.len(), 2);
    assert_eq!(active[0].page_ids.len(), 3);

    // Reject drafts back to verified_len = 31 (still within 2 pages: `31.div_ceil(16) == 2`).
    dirty = [0, 0];
    rewind_rejected_suffix_pages(&mut active[0], 31, 16, 8, 99, &mut shadow_pt, &mut dirty);
    let prefix_pages = active[0].prefix_lease.as_ref().unwrap().block_ids().len();
    assert_eq!(active[0].page_ids.len(), 2);
    assert_eq!(active[0].kv_leases.len(), 1);
    assert_eq!(
        active[0].page_ids.len(),
        prefix_pages + active[0].kv_leases.len()
    );
    // The rejected 3rd page must be returned to `kv_pool` immediately, not leaked until EOS.
    assert_eq!(kv_pool.available(), 6);
    let slot = active[0].slot_lease.id() as usize;
    assert_eq!(shadow_pt[slot * 8 + 2], 99);
    assert_ne!(dirty[0] & (1u128 << slot), 0);
}

#[test]
fn promote_completed_prefixes_holds_incomplete_mtp_chunk_boundary_until_next_chunk() {
    let slot_pool = SlotPool::new(2);
    let kv_pool = BlockPool::new(CacheGroupKind::GlobalAndIndex, 16);
    let ckpt_pool = BlockPool::new(CacheGroupKind::RecurrentState, 4);
    let mut tree = PrefixTree::with_pages_per_node(64, 4).expect("hybrid tree");
    let mut shadow_pt = vec![-1i32; 2 * 16];
    let mut dirty = [0u128; 2];
    let mut queries = 0u64;
    let mut hits = 0u64;
    let mut evictions = 0u64;
    let now = Instant::now();

    let prompt: Vec<i32> = (0..128).map(|i| 900 + i).collect();
    let mut seq = admit_request_into_sequence(
        PendingRequest {
            request_id: 31,
            prompt_tokens: prompt,
            target_output_len: 4,
            permit_time: now,
            submit_time: now,
        },
        slot_pool.allocate().expect("slot"),
        8,
        16,
        Some(&mut tree),
        true,
        &mut queries,
        &mut hits,
        &mut shadow_pt,
        &mut dirty,
    )
    .expect("admit mtp sequence");

    // Chunk 1 computes `0..64` (`prefilled_tokens < prompt_tokens.len()`): MTP KV at pos 63
    // is deferred until Chunk 2 provides `prompt[64]`, so node 0 must NOT be promoted yet.
    seq.prefilled_tokens = 64;
    seq.current_seq_len = 64;
    seq.ensure_kv_pages_with_cache(
        64,
        16,
        16,
        &kv_pool,
        Some(&mut tree),
        &mut evictions,
        &mut shadow_pt,
        &mut dirty,
    )
    .expect("alloc chunk 1 pages");
    seq.pending_checkpoints
        .push((64, ckpt_pool.allocate(1).expect("ckpt 64")));
    let mut active = vec![seq];
    promote_completed_prefixes_in_batch(
        &mut active,
        Some(&mut tree),
        true,
        true,
        4,
        16,
        &mut shadow_pt,
        &mut dirty,
    )
    .expect("hold incomplete mtp chunk");
    assert_eq!(active[0].prefix_lease.as_ref().unwrap().matched_nodes(), 0);
    assert_eq!(active[0].kv_leases.len(), 4);
    assert_eq!(active[0].pending_checkpoints.len(), 1);

    // Chunk 2 completes `64..128` (`prefilled_tokens == prompt_tokens.len()`): both nodes
    // (`0..64` and `64..128`) now have complete MTP KV and promote cleanly.
    active[0].prefilled_tokens = 128;
    active[0].current_seq_len = 128;
    active[0]
        .ensure_kv_pages_with_cache(
            128,
            16,
            16,
            &kv_pool,
            Some(&mut tree),
            &mut evictions,
            &mut shadow_pt,
            &mut dirty,
        )
        .expect("alloc chunk 2 pages");
    active[0]
        .pending_checkpoints
        .push((128, ckpt_pool.allocate(1).expect("ckpt 128")));
    promote_completed_prefixes_in_batch(
        &mut active,
        Some(&mut tree),
        true,
        true,
        4,
        16,
        &mut shadow_pt,
        &mut dirty,
    )
    .expect("promote completed mtp prompt");
    assert_eq!(active[0].prefix_lease.as_ref().unwrap().matched_nodes(), 2);
    assert!(active[0].kv_leases.is_empty());
    assert!(active[0].pending_checkpoints.is_empty());
}

#[test]
fn provenance_collects_zstep_env_overrides_and_redacts_secrets() {
    let overrides = collect_zstep_env_overrides([
        ("PATH", "/usr/bin"),
        ("ZSTEP_P2P_MODE", "auto"),
        ("ZSTEP_NUM_SPEC_TOKENS", "5"),
        ("ZSTEP_REQUIRE_SM120_EXP", "1"),
        ("ZSTEP_API_KEY", "sk-live-secret"),
        ("ZSTEP_SECRET_KEY", "sk-secondary"),
        ("ZSTEP_PASSWORD", "pw-123"),
        ("ZSTEP_CUSTOM_TOKEN", "tok-123"),
    ]);
    assert_eq!(overrides.len(), 7);
    assert_eq!(
        overrides.get("ZSTEP_P2P_MODE").map(String::as_str),
        Some("auto")
    );
    assert_eq!(
        overrides.get("ZSTEP_NUM_SPEC_TOKENS").map(String::as_str),
        Some("5")
    );
    for secret_key in [
        "ZSTEP_API_KEY",
        "ZSTEP_SECRET_KEY",
        "ZSTEP_PASSWORD",
        "ZSTEP_CUSTOM_TOKEN",
    ] {
        assert_eq!(
            overrides.get(secret_key).map(String::as_str),
            Some("[REDACTED]"),
            "{secret_key} must be redacted"
        );
    }

    let prov = BenchmarkProvenance {
        git_commit: "deadbeef".to_owned(),
        nvidia_driver_version: "595.99.02".to_owned(),
        gpu_graphics_clocks_mhz: vec![2400],
        gpu_mem_clocks_mhz: vec![14000],
        seed: 42,
        repeats: 3,
        warmup_prompts: 16,
        max_step_tokens: 4096,
        max_prefill_chunk: 2048,
        env_overrides: overrides,
    };
    let json = prov.to_json();
    assert!(json.contains("\"env_overrides\": {"));
    assert!(json.contains("\"ZSTEP_P2P_MODE\": \"auto\""));
    assert!(json.contains("\"ZSTEP_API_KEY\": \"[REDACTED]\""));
    assert!(!json.contains("sk-live-secret"));
}
