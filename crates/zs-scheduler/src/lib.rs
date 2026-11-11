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

//! Single-master continuous batching scheduler for `zstep`.
//!
//! Computes both `encoder_rows` (Layers 0..19) and narrowed `decoder_rows`
//! (Layers 20..39) plus CUDA Graph bucket targets for all 8 RTX 6000D workers.

mod mode_b;
mod serving;

pub use mode_b::{ModeBConfig, ModeBCoordinator, ModeBError, ModeBMetrics, WorkerLaunchConfig};
pub use serving::{
    AggregateMetric, BENCHMARK_RANDOM_SEED, BenchmarkProvenance, BenchmarkTierResult,
    BenchmarkTierSpec, GenerationScheduleStats, ModeBServingEngine, ServingBenchmarkReport,
    SlotLease, SlotPool, TierAggregateStats, build_deterministic_random_prompts,
};

/// Select the smallest captured CUDA Graph bucket `>= active_rows`, or `0` (eager)
/// if `active_rows` exceeds the largest captured graph bucket.
pub fn select_graph_bucket(active_rows: u32, sorted_buckets: &[u32]) -> u32 {
    if active_rows == 0 {
        return 0;
    }
    for &bucket in sorted_buckets {
        if bucket >= active_rows {
            return bucket;
        }
    }
    0
}

/// Compute how many rows survive `DeepSeek-V4.1`'s Layer-20 CED narrowing for one
/// scheduled sequence:
/// * Incomplete prefill chunk: 1 row
/// * Final prefill chunk: `min(chunk_tokens, ced_tail_window)` rows (`ced_tail_window = 128`)
/// * Decode / DSpark verify: all `scheduled_tokens` rows (`1` or `6`)
pub fn compute_ced_decoder_rows(
    scheduled_tokens: u32,
    is_prefill: bool,
    completes_prompt: bool,
    ced_tail_window: u32,
) -> u32 {
    if !is_prefill {
        return scheduled_tokens;
    }
    if completes_prompt {
        scheduled_tokens.min(ced_tail_window)
    } else {
        1
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn ced_narrowing_matches_v41_contract() {
        // Non-final 8192-token prefill chunk keeps 1 row
        assert_eq!(compute_ced_decoder_rows(8192, true, false, 128), 1);
        // Final 4096-token prefill chunk keeps last 128 rows
        assert_eq!(compute_ced_decoder_rows(4096, true, true, 128), 128);
        // DSpark verify step keeps all 6 rows
        assert_eq!(compute_ced_decoder_rows(6, false, true, 128), 6);
    }

    #[test]
    fn bucket_selection_picks_tightest_fit_or_eager() {
        let ladder = [6, 12, 24, 48, 96, 192];
        assert_eq!(select_graph_bucket(48, &ladder), 48);
        assert_eq!(select_graph_bucket(49, &ladder), 96);
        assert_eq!(select_graph_bucket(512, &ladder), 0);
    }
}
