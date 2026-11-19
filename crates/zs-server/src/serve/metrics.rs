// Copyright (c) 2026 zstep Contributors
// SPDX-License-Identifier: MIT

use super::*;

fn append_histogram<const N: usize>(
    out: &mut String,
    name: &str,
    help: &str,
    bounds: &[f64; N],
    buckets: &[AtomicU64; N],
    count: u64,
    sum_secs: f64,
) {
    out.push_str(&format!("# HELP {name} {help}\n# TYPE {name} histogram\n"));
    for (bound, counter) in bounds.iter().zip(buckets) {
        let cumulative = counter.load(Ordering::Relaxed);
        out.push_str(&format!("{name}_bucket{{le=\"{bound}\"}} {cumulative}\n"));
    }
    out.push_str(&format!("{name}_bucket{{le=\"+Inf\"}} {count}\n"));
    out.push_str(&format!("{name}_sum {sum_secs:.6}\n"));
    out.push_str(&format!("{name}_count {count}\n"));
}

fn append_prefix_cache_counters(out: &mut String, queries: u64, hits: u64, evictions: u64) {
    out.push_str(
        "# HELP zstep:prefix_cache_queries_total Prompt tokens queried against Automatic Prefix Caching (APC)\n# TYPE zstep:prefix_cache_queries_total counter\n",
    );
    out.push_str(&format!("zstep:prefix_cache_queries_total {queries}\n"));
    out.push_str(
        "# HELP zstep:prefix_cache_hits_total Prompt tokens served from Automatic Prefix Caching (APC)\n# TYPE zstep:prefix_cache_hits_total counter\n",
    );
    out.push_str(&format!("zstep:prefix_cache_hits_total {hits}\n"));
    out.push_str(
        "# HELP zstep:prefix_cache_evictions_total Cached KV pages evicted from Automatic Prefix Caching (APC)\n# TYPE zstep:prefix_cache_evictions_total counter\n",
    );
    out.push_str(&format!("zstep:prefix_cache_evictions_total {evictions}\n"));
}

fn append_spec_decode_metrics(out: &mut String, metrics: &OnlineMetrics) {
    let drafts = metrics.spec_decode_num_drafts.load(Ordering::Relaxed);
    let draft_tokens = metrics.spec_decode_num_draft_tokens.load(Ordering::Relaxed);
    let accepted = metrics
        .spec_decode_num_accepted_tokens
        .load(Ordering::Relaxed);
    out.push_str("# HELP zstep:spec_decode_num_drafts_total Speculative verification steps\n# TYPE zstep:spec_decode_num_drafts_total counter\n");
    out.push_str(&format!("zstep:spec_decode_num_drafts_total {drafts}\n"));
    out.push_str("# HELP zstep:spec_decode_num_draft_tokens_total Proposed draft tokens\n# TYPE zstep:spec_decode_num_draft_tokens_total counter\n");
    out.push_str(&format!(
        "zstep:spec_decode_num_draft_tokens_total {draft_tokens}\n"
    ));
    out.push_str("# HELP zstep:spec_decode_num_accepted_tokens_total Accepted draft tokens excluding bonus tokens\n# TYPE zstep:spec_decode_num_accepted_tokens_total counter\n");
    out.push_str(&format!(
        "zstep:spec_decode_num_accepted_tokens_total {accepted}\n"
    ));
    if drafts > 0 {
        let mean_acceptance = 1.0 + accepted as f64 / drafts as f64;
        out.push_str("# HELP zstep:spec_decode_mean_acceptance_length Mean accepted tokens per verification step\n# TYPE zstep:spec_decode_mean_acceptance_length gauge\n");
        out.push_str(&format!(
            "zstep:spec_decode_mean_acceptance_length {mean_acceptance}\n"
        ));
    }
}

pub(super) fn render_metrics(state: &AppState) -> String {
    let mut out = String::new();
    let Some(metrics) = state.metrics() else {
        out.push_str(
            "# HELP zstep:num_requests_waiting Requests accepted by HTTP and waiting for a slot lease\n# TYPE zstep:num_requests_waiting gauge\nzstep:num_requests_waiting 0\n",
        );
        out.push_str(
            "# HELP zstep:num_requests_running Active sequences holding a slot lease (prefill + decode)\n# TYPE zstep:num_requests_running gauge\nzstep:num_requests_running 0\n",
        );
        out.push_str(
            "# HELP zstep:kv_cache_usage_perc Physical KV page occupancy in [0,1]\n# TYPE zstep:kv_cache_usage_perc gauge\nzstep:kv_cache_usage_perc 0\n",
        );
        out.push_str(
            "# HELP zstep:num_preemptions_total Sequence preemptions\n# TYPE zstep:num_preemptions_total counter\nzstep:num_preemptions_total 0\n",
        );
        if state.enable_prefix_caching {
            append_prefix_cache_counters(&mut out, 0, 0, 0);
        }
        if state.speculative_enabled {
            append_spec_decode_metrics(&mut out, &OnlineMetrics::default());
        }
        out.push_str(
            "# HELP zstep:triton_recompilations_total Runtime Triton JIT recompilations\n# TYPE zstep:triton_recompilations_total gauge\nzstep:triton_recompilations_total 0\n",
        );
        return out;
    };

    let gauges = [
        (
            "zstep:num_requests_waiting",
            metrics.waiting.load(Ordering::Relaxed) as f64,
            "Requests accepted by HTTP and waiting for a slot lease",
        ),
        (
            "zstep:num_requests_running",
            metrics.running.load(Ordering::Relaxed) as f64,
            "Active sequences holding a slot lease (prefill + decode)",
        ),
        (
            "zstep:kv_cache_usage_perc",
            metrics.kv_cache_usage(),
            "Physical KV page occupancy in [0,1]",
        ),
    ];
    for (name, value, help) in gauges {
        out.push_str(&format!("# HELP {name} {help}\n# TYPE {name} gauge\n"));
        out.push_str(&format!("{name} {value}\n"));
    }
    out.push_str("# HELP zstep:num_preemptions_total Sequence preemptions\n# TYPE zstep:num_preemptions_total counter\n");
    out.push_str(&format!(
        "zstep:num_preemptions_total {}\n",
        metrics.preemptions.load(Ordering::Relaxed)
    ));
    if state.enable_prefix_caching {
        append_prefix_cache_counters(
            &mut out,
            metrics.prefix_cache_queries.load(Ordering::Relaxed),
            metrics.prefix_cache_hits.load(Ordering::Relaxed),
            metrics.prefix_cache_evictions.load(Ordering::Relaxed),
        );
    }
    if state.speculative_enabled {
        append_spec_decode_metrics(&mut out, &metrics);
    }
    let outcomes = [
        ("finished", metrics.finished.load(Ordering::Relaxed)),
        ("cancelled", metrics.cancelled.load(Ordering::Relaxed)),
        (
            "rejected_overloaded",
            metrics.rejected_overloaded.load(Ordering::Relaxed),
        ),
        ("failed", metrics.failed.load(Ordering::Relaxed)),
    ];
    out.push_str("# HELP zstep:request_outcomes_total Request terminal outcomes\n# TYPE zstep:request_outcomes_total counter\n");
    for (outcome, value) in outcomes {
        out.push_str(&format!(
            "zstep:request_outcomes_total{{outcome=\"{outcome}\"}} {value}\n"
        ));
    }

    let ttft_count = metrics.ttft_count.load(Ordering::Relaxed);
    let ttft_sum = metrics.ttft_micros.load(Ordering::Relaxed) as f64 / 1e6;
    append_histogram(
        &mut out,
        "zstep:time_to_first_token_seconds",
        "Time to first token in seconds",
        &TTFT_BUCKET_BOUNDS_SECS,
        &metrics.ttft_buckets,
        ttft_count,
        ttft_sum,
    );

    let tpot_count = metrics.tpot_count.load(Ordering::Relaxed);
    let tpot_sum = metrics.tpot_micros.load(Ordering::Relaxed) as f64 / 1e6;
    append_histogram(
        &mut out,
        "zstep:time_per_output_token_seconds",
        "Inter-token latency in seconds",
        &TPOT_BUCKET_BOUNDS_SECS,
        &metrics.tpot_buckets,
        tpot_count,
        tpot_sum,
    );

    out.push_str("# HELP zstep:triton_recompilations_total Runtime Triton JIT recompilations\n# TYPE zstep:triton_recompilations_total gauge\n");
    out.push_str(&format!(
        "zstep:triton_recompilations_total {}\n",
        metrics.triton_recompilations.load(Ordering::Relaxed)
    ));
    out.push_str(&format!(
        "# zstep uptime_seconds {:.3}\n",
        state.started.elapsed().as_secs_f64()
    ));
    out
}
