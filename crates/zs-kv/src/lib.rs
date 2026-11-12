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

//! Four-group FlatKV block allocator with RAII ownership (`BlockLease`),
//! Radix prefix cache (`PrefixTree`, `RadixLease`, `KvCacheManager`), and
//! SWA 128-token bounded replay planning.

mod radix;

use std::sync::{Arc, Mutex};

pub use radix::{
    KvCacheManager, KvError, PrefillSchedule, PrefixTree, RadixLease, RequestKvHandle,
};

/// Physical cache groups used by `DeepSeek-V4.1-Flash` FlatKV attention.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum CacheGroupKind {
    /// 512D NVFP4 (`E2M1/E4M3`, 288B/row) + 128D MXFP4 Index (`E2M1/E8M0`, 68B/row),
    /// co-allocated 1:1 and eligible for Radix prefix caching.
    GlobalAndIndex,
    /// 512D MXFP8 (`E4M3/E8M0`, 528B/row) sliding window; private per request,
    /// regenerated on prefix hit via 128-token bounded replay.
    SwaWindow,
    /// Private compressor tail state slot; never prefix-cached.
    CompressorTail,
}

struct PoolInner {
    free_ids: Vec<i32>,
    capacity: usize,
}

/// Free-list physical block pool whose allocated IDs are returned automatically
/// when their owning [`BlockLease`] drops.
#[derive(Clone)]
pub struct BlockPool {
    kind: CacheGroupKind,
    inner: Arc<Mutex<PoolInner>>,
}

impl BlockPool {
    pub fn new(kind: CacheGroupKind, num_blocks: usize) -> Self {
        let free_ids = (0..num_blocks as i32).rev().collect();
        Self {
            kind,
            inner: Arc::new(Mutex::new(PoolInner {
                free_ids,
                capacity: num_blocks,
            })),
        }
    }

    pub fn kind(&self) -> CacheGroupKind {
        self.kind
    }

    pub fn available(&self) -> usize {
        self.inner
            .lock()
            .unwrap_or_else(|poisoned| poisoned.into_inner())
            .free_ids
            .len()
    }

    pub fn capacity(&self) -> usize {
        self.inner
            .lock()
            .unwrap_or_else(|poisoned| poisoned.into_inner())
            .capacity
    }

    pub fn allocate(&self, count: usize) -> Option<BlockLease> {
        let mut guard = self
            .inner
            .lock()
            .unwrap_or_else(|poisoned| poisoned.into_inner());
        if guard.free_ids.len() < count {
            return None;
        }
        let split_at = guard.free_ids.len() - count;
        let ids = guard.free_ids.split_off(split_at);
        Some(BlockLease {
            pool: Arc::clone(&self.inner),
            ids,
        })
    }
}

/// RAII handle owning physical cache blocks; returns them to [`BlockPool`] on `Drop`
/// without panicking even during unwind.
pub struct BlockLease {
    pool: Arc<Mutex<PoolInner>>,
    ids: Vec<i32>,
}

impl BlockLease {
    pub fn ids(&self) -> &[i32] {
        &self.ids
    }

    pub fn len(&self) -> usize {
        self.ids.len()
    }

    pub fn is_empty(&self) -> bool {
        self.ids.is_empty()
    }

    /// Split a multi-block lease into single-block [`BlockLease`] handles without releasing
    /// or duplicating ownership of any physical block ID.
    pub fn split_individual(mut self) -> Vec<Self> {
        let owned_ids = std::mem::take(&mut self.ids);
        owned_ids
            .into_iter()
            .map(|id| Self {
                pool: Arc::clone(&self.pool),
                ids: vec![id],
            })
            .collect()
    }
}

impl Drop for BlockLease {
    fn drop(&mut self) {
        if self.ids.is_empty() {
            return;
        }
        let mut guard = self
            .pool
            .lock()
            .unwrap_or_else(|poisoned| poisoned.into_inner());
        guard.free_ids.append(&mut self.ids);
    }
}

/// Compute the effective cached prefix length and replay span for a `DeepSeek-V4.1`
/// SWA bounded-replay hit (`swa_window = 128`).
pub fn plan_bounded_replay(matched_prefix_tokens: u32, swa_window: u32) -> (u32, u32) {
    if matched_prefix_tokens == 0 {
        return (0, 0);
    }
    let replay_tokens = matched_prefix_tokens.min(swa_window);
    let skipped_prefix = matched_prefix_tokens - replay_tokens;
    (skipped_prefix, replay_tokens)
}

#[cfg(test)]
mod tests {
    use std::panic::{AssertUnwindSafe, catch_unwind};

    use super::*;

    #[test]
    fn block_lease_raii_returns_blocks_on_drop() {
        let pool = BlockPool::new(CacheGroupKind::GlobalAndIndex, 16);
        assert_eq!(pool.available(), 16);
        {
            let lease = pool.allocate(6).expect("enough blocks");
            assert_eq!(lease.ids().len(), 6);
            assert_eq!(pool.available(), 10);
        }
        assert_eq!(pool.available(), 16);
    }

    #[test]
    fn swa_bounded_replay_rewinds_at_most_window() {
        assert_eq!(plan_bounded_replay(0, 128), (0, 0));
        assert_eq!(plan_bounded_replay(64, 128), (0, 64));
        assert_eq!(plan_bounded_replay(1024, 128), (896, 128));
    }

    #[test]
    fn radix_prefix_hit_plans_128_token_bounded_replay_and_shares_pages() {
        let manager = KvCacheManager::new(32, 16, 8, 64, 128, 4).expect("valid manager");
        let prompt_a: Vec<i32> = (0..350).map(|idx| idx + 1000).collect();

        // Request A misses completely: allocates ceil(350 / 64) = 6 GlobalAndIndex blocks,
        // 4 SwaWindow blocks, and 1 CompressorTail slot.
        let mut handle_a = manager.admit_request(&prompt_a).expect("admit A");
        assert_eq!(handle_a.schedule().skipped_tokens, 0);
        assert_eq!(handle_a.schedule().replay_tokens, 0);
        assert_eq!(handle_a.schedule().fresh_tokens, 350);
        assert_eq!(handle_a.schedule().replay_mask, vec![0u8; 350]);
        assert_eq!(manager.global_pool().available(), 26);
        assert_eq!(manager.swa_pool().available(), 12);
        assert_eq!(manager.compressor_pool().available(), 7);

        // Commit completed full 64-token blocks (5 full blocks = 320 tokens).
        let promoted = handle_a
            .commit_completed_blocks(&prompt_a)
            .expect("commit A");
        assert_eq!(promoted, 5);
        assert_eq!(manager.prefix_tree().cached_blocks(), 5);
        let pages_a = handle_a.global_page_table();
        assert_eq!(pages_a.len(), 6);

        // Request B shares the first 320 tokens (5 blocks) + 80 new suffix tokens (total 400 tokens -> 7 blocks).
        let mut prompt_b = prompt_a[..320].to_vec();
        prompt_b.extend((0..80).map(|idx| idx + 9000));
        let handle_b = manager.admit_request(&prompt_b).expect("admit B");

        // 320 matched tokens with swa_window = 128 -> skip 192 tokens, replay 128 tokens, 80 fresh tokens!
        let sched_b = handle_b.schedule();
        assert_eq!(sched_b.skipped_tokens, 192);
        assert_eq!(sched_b.replay_tokens, 128);
        assert_eq!(sched_b.fresh_tokens, 80);
        assert_eq!(sched_b.scheduled_tokens.len(), 208);
        assert_eq!(sched_b.scheduled_positions[0], 192);
        assert_eq!(*sched_b.scheduled_positions.last().expect("nonempty"), 399);
        assert_eq!(sched_b.replay_mask[..128], vec![1u8; 128]);
        assert_eq!(sched_b.replay_mask[128..], vec![0u8; 80]);

        // First 5 GlobalAndIndex pages are shared 1:1 with Request A; SWA and Compressor stay private!
        let pages_b = handle_b.global_page_table();
        assert_eq!(&pages_b[..5], &pages_a[..5]);
        assert_ne!(handle_a.swa_page_table(), handle_b.swa_page_table());
        assert_ne!(handle_a.compressor_slot_id(), handle_b.compressor_slot_id());
    }

    /// Witness of the current contract clash, not the desired end state.
    /// An exact block-aligned hit keeps `replay_mask = 0` on the last token
    /// (a GlobalAndIndex write) while that token's page stays shared with the
    /// live owner of the cached block. A private page id without a byte copy
    /// would drop the other rows of that block, so this is not "fixed" here.
    #[test]
    fn shared_exact_prefix_schedules_global_write_on_aliased_page() {
        let manager = KvCacheManager::new(16, 8, 4, 64, 128, 4).expect("valid manager");
        let prompt: Vec<i32> = (0..256).collect();
        let mut first = manager.admit_request(&prompt).expect("admit first");
        assert_eq!(first.commit_completed_blocks(&prompt).expect("commit"), 4);
        let repeat = manager.admit_request(&prompt).expect("admit repeat");
        let shared = first.global_page_table();
        let pages = repeat.global_page_table();
        assert_eq!(
            pages, shared,
            "exact hit currently aliases every cached page"
        );
        let block_tokens = 64usize;
        let mut aliased_writes = 0usize;
        for (offset, &mask) in repeat.schedule().replay_mask.iter().enumerate() {
            if mask != 0 {
                continue;
            }
            let pos = repeat.schedule().scheduled_positions[offset] as usize;
            let page = pages[pos / block_tokens];
            if shared.contains(&page) {
                aliased_writes += 1;
            }
        }
        assert_eq!(aliased_writes, 1);
    }

    #[test]
    fn block_ids_stay_unique_across_admit_commit_and_drop() {
        let manager = KvCacheManager::new(24, 32, 16, 64, 128, 4).expect("valid manager");
        let mut live = Vec::new();
        for turn in 0..12u32 {
            let len = 64 * ((turn % 5) + 1) as usize + (turn as usize % 17);
            let prompt: Vec<i32> = (0..len).map(|i| (i as i32) + (turn as i32) * 3).collect();
            if let Ok(mut handle) = manager.admit_request(&prompt) {
                let _ = handle.commit_completed_blocks(&prompt);
                live.push(handle);
            }
            if live.len() > 4 {
                live.remove(0);
            }
            let tail_blocks: usize = live
                .iter()
                .map(|handle| {
                    let prefix_blocks = handle.matched_prefix_tokens() as usize / 64;
                    handle.global_page_table().len() - prefix_blocks
                })
                .sum();
            assert_eq!(
                manager.global_pool().available()
                    + manager.prefix_tree().cached_blocks()
                    + tail_blocks,
                manager.global_pool().capacity()
            );
            assert_eq!(manager.swa_pool().available() + live.len() * 4, 32);
            assert_eq!(
                manager.compressor_pool().available() + live.len(),
                manager.compressor_pool().capacity()
            );
        }
        drop(live);
        let cached = manager.prefix_tree().cached_blocks();
        assert_eq!(
            manager.global_pool().available() + cached,
            manager.global_pool().capacity()
        );
        assert_eq!(manager.prefix_tree().evict_lru(cached), cached);
        assert_eq!(
            manager.global_pool().available(),
            manager.global_pool().capacity()
        );
        assert_eq!(manager.swa_pool().available(), 32);
        assert_eq!(
            manager.compressor_pool().available(),
            manager.compressor_pool().capacity()
        );
    }

    #[test]
    fn exact_block_aligned_full_prefix_hit_keeps_final_token_fresh() {
        let manager = KvCacheManager::new(16, 8, 4, 64, 128, 4).expect("valid manager");
        let prompt: Vec<i32> = (0..256).collect();
        {
            let mut first = manager.admit_request(&prompt).expect("admit first");
            assert_eq!(first.commit_completed_blocks(&prompt).expect("commit"), 4);
        }
        // Re-sending the exact 256-token prompt matches 256 tokens in the Radix tree,
        // clamps effective_matched to 255 -> skips 127, replays 128, keeps final token fresh!
        let repeat = manager.admit_request(&prompt).expect("admit repeat");
        assert_eq!(repeat.schedule().skipped_tokens, 127);
        assert_eq!(repeat.schedule().replay_tokens, 128);
        assert_eq!(repeat.schedule().fresh_tokens, 1);
        assert_eq!(repeat.schedule().scheduled_tokens.len(), 129);
        assert_eq!(*repeat.schedule().replay_mask.last().expect("nonempty"), 0);
    }

    #[test]
    fn lru_eviction_respects_active_pins_and_cascades_unpinned_chains() {
        // Pool has exactly 6 GlobalAndIndex blocks.
        let manager = KvCacheManager::new(6, 16, 8, 64, 128, 4).expect("valid manager");
        let branch_old: Vec<i32> = (0..192).map(|x| x + 100).collect(); // 3 blocks
        let branch_pinned: Vec<i32> = (0..128).map(|x| x + 500).collect(); // 2 blocks

        {
            let mut h_old = manager.admit_request(&branch_old).expect("admit old");
            assert_eq!(
                h_old.commit_completed_blocks(&branch_old).expect("commit"),
                3
            );
        }
        // `branch_old` dropped -> its 3 blocks remain cached in PrefixTree (unpinned).
        assert_eq!(manager.prefix_tree().cached_blocks(), 3);
        assert_eq!(manager.prefix_tree().evictable_blocks(), 3);
        assert_eq!(manager.global_pool().available(), 3);

        // `h_pinned` stays alive -> pins 2 blocks. Only 1 free block left in `global_pool`,
        // plus 3 evictable blocks in `branch_old` = 4 allocatable blocks max.
        let mut h_pinned = manager.admit_request(&branch_pinned).expect("admit pinned");
        assert_eq!(
            h_pinned
                .commit_completed_blocks(&branch_pinned)
                .expect("commit"),
            2
        );
        assert_eq!(manager.global_pool().available(), 1);
        assert_eq!(manager.prefix_tree().evictable_blocks(), 3);

        // Requesting 5 fresh blocks (> 1 free + 3 evictable) fails without corrupting pins!
        let too_big: Vec<i32> = (0..320).map(|x| x + 900).collect();
        let err = manager
            .admit_request(&too_big)
            .err()
            .expect("must fail OOM");
        assert!(matches!(
            err,
            KvError::OutOfBlocks {
                kind: CacheGroupKind::GlobalAndIndex,
                ..
            }
        ));

        // Requesting 4 fresh blocks (256 tokens) succeeds by evicting `branch_old` while keeping `h_pinned` intact!
        let fits: Vec<i32> = (0..256).map(|x| x + 2000).collect();
        let h_fits = manager
            .admit_request(&fits)
            .expect("admit fits after LRU eviction");
        assert_eq!(h_fits.global_page_table().len(), 4);
        assert_eq!(manager.global_pool().available(), 0);
        // `h_pinned`'s 2 blocks are still in the tree and pinned!
        assert_eq!(manager.prefix_tree().cached_blocks(), 2);
        assert_eq!(manager.prefix_tree().evictable_blocks(), 0);

        drop(h_fits);
        drop(h_pinned);
        assert_eq!(manager.prefix_tree().evictable_blocks(), 2);
        assert_eq!(manager.prefix_tree().evict_lru(2), 2);
        assert_eq!(manager.global_pool().available(), 6);
    }

    #[test]
    fn concurrent_duplicate_promotion_returns_redundant_blocks_to_pool() {
        let manager = KvCacheManager::new(10, 8, 4, 64, 128, 4).expect("valid manager");
        let shared_prompt: Vec<i32> = (0..128).map(|x| x + 42).collect();

        // Both requests arrive before either commits -> both allocate 2 fresh blocks (4 blocks used).
        let mut h1 = manager.admit_request(&shared_prompt).expect("admit h1");
        let mut h2 = manager.admit_request(&shared_prompt).expect("admit h2");
        assert_eq!(manager.global_pool().available(), 6);

        // h1 commits its 2 blocks -> tree caches 2 blocks.
        assert_eq!(
            h1.commit_completed_blocks(&shared_prompt)
                .expect("commit h1"),
            2
        );
        assert_eq!(manager.global_pool().available(), 6);

        // h2 commits the same 2 blocks -> deduplicated against h1's nodes, returning h2's 2 redundant blocks immediately!
        assert_eq!(
            h2.commit_completed_blocks(&shared_prompt)
                .expect("commit h2"),
            2
        );
        assert_eq!(manager.global_pool().available(), 8);
        assert_eq!(h1.global_page_table(), h2.global_page_table());
    }

    #[test]
    fn unwind_panic_releases_all_leases_and_unpins_tree() {
        let manager = KvCacheManager::new(12, 8, 4, 64, 128, 4).expect("valid manager");
        let prompt: Vec<i32> = (0..192).collect();

        let panic_res = catch_unwind(AssertUnwindSafe(|| {
            let mut handle = manager.admit_request(&prompt).expect("admit");
            handle.commit_completed_blocks(&prompt).expect("commit");
            panic!("simulated worker/request unwind");
        }));
        assert!(panic_res.is_err());

        // SWA and Compressor leases were returned on unwind, and the 3 promoted tree blocks are unpinned!
        assert_eq!(manager.swa_pool().available(), 8);
        assert_eq!(manager.compressor_pool().available(), 4);
        assert_eq!(manager.prefix_tree().evictable_blocks(), 3);
        assert_eq!(manager.prefix_tree().evict_lru(3), 3);
        assert_eq!(manager.global_pool().available(), 12);
    }
}
