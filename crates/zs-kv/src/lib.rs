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

//! Physical cache block allocator with RAII ownership (`BlockLease`),
//! Automatic Prefix Cache (`PrefixTree`, `PrefixLease`, `KvCacheManager`), and
//! SWA 128-token bounded replay planning.

mod dram_tier;
mod hybrid_alias;
pub mod kv_hash;
mod prefix_cache;

use std::sync::{Arc, Mutex};

pub use dram_tier::{
    DEFAULT_LOADING_AREA_RATIO, DramPrefixCache, DramTierConfig, DramTierStats, KvTransferOp,
};
pub use hybrid_alias::{
    GDN_FINE_GRAINED_TOKENS, HybridBlockAliasEntry, HybridBlockAliasTable, SubPageAllocator,
    SubPageLease,
};
pub use prefix_cache::{
    KvCacheEvent, KvCacheEventSink, KvCacheManager, KvError, PrefillSchedule, PrefixLease,
    PrefixTree, PrefixVictim, RequestKvHandle,
};

/// Physical cache groups managed by `zs-kv`.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum CacheGroupKind {
    /// Paged attention KV blocks (`GlobalAndIndex` on `DeepSeek-V4.1-Flash` or paged KV on
    /// `Qwen3.5`), eligible for Automatic Prefix Caching.
    GlobalAndIndex,
    /// 512D MXFP8 (`E4M3/E8M0`, 528B/row) sliding window; private per request when using
    /// SWA bounded replay.
    SwaWindow,
    /// Private compressor tail state slot; never prefix-cached.
    CompressorTail,
    /// Recurrent state (`conv_state` + `ssm_state`) checkpoint slots for hybrid models
    /// (such as `Qwen3.5-35B-A3B` `GatedDeltaNet` layers), co-owned with prefix tree nodes.
    RecurrentState,
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
    pub fn split_individual(self) -> Vec<Self> {
        self.split_chunks(1)
    }

    /// Split a lease into chunks of up to `chunk_size` physical block IDs without releasing
    /// or duplicating ownership.
    pub fn split_chunks(mut self, chunk_size: usize) -> Vec<Self> {
        assert!(chunk_size > 0, "chunk_size must be positive");
        let owned_ids = std::mem::take(&mut self.ids);
        owned_ids
            .chunks(chunk_size)
            .map(|chunk| Self {
                pool: Arc::clone(&self.pool),
                ids: chunk.to_vec(),
            })
            .collect()
    }

    /// Merge multiple [`BlockLease`] handles from the same [`BlockPool`] into a single
    /// [`BlockLease`] preserving block ID order.
    ///
    /// Returns `None` if `leases` is empty or if the leases belong to different pools.
    pub fn merge(mut leases: Vec<Self>) -> Option<Self> {
        let first_pool = Arc::clone(&leases.first()?.pool);
        if !leases
            .iter()
            .all(|lease| Arc::ptr_eq(&first_pool, &lease.pool))
        {
            return None;
        }
        let total_ids: usize = leases.iter().map(|lease| lease.ids.len()).sum();
        let mut merged_ids = Vec::with_capacity(total_ids);
        for lease in &mut leases {
            merged_ids.append(&mut lease.ids);
        }
        Some(Self {
            pool: first_pool,
            ids: merged_ids,
        })
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
    fn block_lease_split_and_merge_roundtrip_preserves_order_and_ownership() {
        let pool = BlockPool::new(CacheGroupKind::GlobalAndIndex, 12);
        let original = pool.allocate(8).expect("allocate 8");
        let expected_ids = original.ids().to_vec();
        assert_eq!(pool.available(), 4);

        let singles = original.split_individual();
        assert_eq!(singles.len(), 8);
        assert_eq!(pool.available(), 4);

        let mut quad_leases = Vec::new();
        let mut iter = singles.into_iter();
        for _ in 0..2 {
            let group: Vec<BlockLease> = iter.by_ref().take(4).collect();
            quad_leases.push(BlockLease::merge(group).expect("same pool merge"));
        }
        assert_eq!(quad_leases[0].ids(), &expected_ids[..4]);
        assert_eq!(quad_leases[1].ids(), &expected_ids[4..]);
        drop(quad_leases);
        assert_eq!(pool.available(), 12);
    }

    #[test]
    fn swa_bounded_replay_rewinds_at_most_window() {
        assert_eq!(plan_bounded_replay(0, 128), (0, 0));
        assert_eq!(plan_bounded_replay(64, 128), (0, 64));
        assert_eq!(plan_bounded_replay(1024, 128), (896, 128));
    }

    #[test]
    fn prefix_hit_plans_128_token_bounded_replay_and_shares_pages() {
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

    /// An exact block-aligned prefix hit (`matched_tokens == prompt_len`) must replay only
    /// the trailing `min(matched_tokens, swa_window)` tokens with `replay_mask = 1` (rebuilding
    /// private `SwaWindow` + `CompressorTail` and computing next-token logits at `prompt_len - 1`)
    /// while performing **zero** `replay_mask = 0` writes to any shared `GlobalAndIndex` page.
    #[test]
    fn shared_exact_prefix_performs_zero_writes_on_aliased_pages() {
        let manager = KvCacheManager::new(16, 8, 4, 64, 128, 4).expect("valid manager");
        let prompt: Vec<i32> = (0..256).collect();
        let mut first = manager.admit_request(&prompt).expect("admit first");
        assert_eq!(first.commit_completed_blocks(&prompt).expect("commit"), 4);
        let repeat = manager.admit_request(&prompt).expect("admit repeat");
        let shared = first.global_page_table();
        let pages = repeat.global_page_table();
        assert_eq!(
            pages, shared,
            "exact hit shares all 4 immutable cached GlobalAndIndex pages"
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
        assert_eq!(
            aliased_writes, 0,
            "shared cached pages must never be scheduled with replay_mask = 0"
        );
        assert_eq!(
            repeat.schedule().scheduled_positions.last().copied(),
            Some(255),
            "final prompt token must still execute to produce next-token logits"
        );
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
    fn exact_block_aligned_full_prefix_hit_replays_swa_tail_for_final_logits() {
        let manager = KvCacheManager::new(16, 8, 4, 64, 128, 4).expect("valid manager");
        let prompt: Vec<i32> = (0..256).collect();
        {
            let mut first = manager.admit_request(&prompt).expect("admit first");
            assert_eq!(first.commit_completed_blocks(&prompt).expect("commit"), 4);
        }
        // Re-sending the exact 256-token prompt matches all 256 tokens in the prefix tree:
        // skips 128, replays the trailing 128 tokens (`128..256`) with `replay_mask = 1`
        // so `pos = 255` produces next-token logits without writing shared GlobalAndIndex pages.
        let repeat = manager.admit_request(&prompt).expect("admit repeat");
        assert_eq!(repeat.schedule().skipped_tokens, 128);
        assert_eq!(repeat.schedule().replay_tokens, 128);
        assert_eq!(repeat.schedule().fresh_tokens, 0);
        assert_eq!(repeat.schedule().scheduled_tokens.len(), 128);
        assert_eq!(repeat.schedule().scheduled_positions[0], 128);
        assert_eq!(repeat.schedule().scheduled_positions[127], 255);
        assert_eq!(repeat.schedule().replay_mask, vec![1u8; 128]);
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
    fn hybrid_multi_page_prefix_tree_manages_aux_checkpoints_bounded_match_and_eviction() {
        let kv_pool = BlockPool::new(CacheGroupKind::GlobalAndIndex, 24);
        let aux_pool = BlockPool::new(CacheGroupKind::RecurrentState, 6);
        let tree = PrefixTree::with_pages_per_node(64, 4).expect("64-token node, 4 pages/node");
        let tokens: Vec<i32> = (0..192).map(|i| i + 700).collect(); // 3 nodes * 64 tokens

        let base = tree.match_prefix_bounded(&tokens, 2);
        assert_eq!(base.matched_nodes(), 0);
        assert_eq!(base.matched_tokens(), 0);
        assert_eq!(base.leaf_aux_id(), None);

        // Promote 3 nodes (12 KV pages + 3 recurrent checkpoint slots) for request 1,
        // and simulate concurrent duplicate promotion by request 2.
        let kv_1 = kv_pool.allocate(12).expect("kv_1").split_chunks(4);
        let aux_1 = aux_pool.allocate(3).expect("aux_1").split_individual();
        let kv_2 = kv_pool.allocate(12).expect("kv_2").split_chunks(4);
        let aux_2 = aux_pool.allocate(3).expect("aux_2").split_individual();
        assert_eq!(kv_pool.available(), 0);
        assert_eq!(aux_pool.available(), 0);

        let lease_1 = tree
            .extend_prefix_with_aux(base, &tokens, kv_1, Some(aux_1))
            .expect("extend 1");
        assert_eq!(lease_1.matched_nodes(), 3);
        assert_eq!(lease_1.matched_tokens(), 192);
        assert_eq!(lease_1.block_ids().len(), 12);
        assert!(lease_1.leaf_aux_id().is_some());
        assert_eq!(tree.cached_blocks(), 3);
        assert_eq!(tree.cached_kv_pages(), 12);

        // Concurrent duplicate promotion must return BOTH kv_2 (12 pages) and aux_2 (3 slots)!
        let base_2 = tree.match_prefix_bounded(&[], 0);
        let lease_2 = tree
            .extend_prefix_with_aux(base_2, &tokens, kv_2, Some(aux_2))
            .expect("dedup extend 2");
        assert_eq!(kv_pool.available(), 12);
        assert_eq!(aux_pool.available(), 3);
        assert_eq!(lease_1.block_ids(), lease_2.block_ids());
        assert_eq!(lease_1.leaf_aux_id(), lease_2.leaf_aux_id());

        // Bounded match with `max_nodes = 2` leaves the final 64-token block uncached so the
        // caller can still execute suffix prefill to produce logits.
        let bounded = tree.match_prefix_bounded(&tokens, 2);
        assert_eq!(bounded.matched_nodes(), 2);
        assert_eq!(bounded.matched_tokens(), 128);
        assert_eq!(bounded.block_ids(), &lease_1.block_ids()[..8]);
        assert_ne!(bounded.leaf_aux_id(), lease_1.leaf_aux_id());

        drop(bounded);
        drop(lease_1);
        drop(lease_2);
        assert_eq!(tree.evictable_blocks(), 3);
        assert_eq!(tree.evictable_kv_pages(), 12);
        assert_eq!(tree.evict_lru(3), 3);
        assert_eq!(kv_pool.available(), 24);
        assert_eq!(aux_pool.available(), 6);
        assert_eq!(tree.alias_count(), 0);
    }

    #[test]
    fn hybrid_128_token_gdn_prefix_tree_maintains_unified_block_id_aliases() {
        let kv_pool = BlockPool::new(CacheGroupKind::GlobalAndIndex, 32);
        let aux_pool = BlockPool::new(CacheGroupKind::RecurrentState, 4);
        // Qwen3.5 128-token fine-grained GDN node = 8 * 16-token KV pages + 1 recurrent slot.
        let tree = PrefixTree::with_pages_per_node(GDN_FINE_GRAINED_TOKENS, 8)
            .expect("128-token node, 8 pages/node");
        let tokens: Vec<i32> = (0..256).map(|i| i + 1_000).collect(); // 2 nodes * 128 tokens

        let kv = kv_pool.allocate(16).expect("kv").split_chunks(8);
        let aux = aux_pool.allocate(2).expect("aux").split_individual();
        let lease = tree
            .extend_prefix_with_aux(tree.match_prefix(&[]), &tokens, kv, Some(aux))
            .expect("promote 2 128-token GDN nodes");

        assert_eq!(tree.alias_count(), 2);
        assert_eq!(lease.matched_tokens(), 256);
        let leaf_anchor = lease.leaf_anchor_block_id().expect("leaf anchor block id");
        let leaf_aux = lease.leaf_aux_id().expect("leaf aux slot");
        assert_eq!(leaf_anchor, lease.block_ids()[8]);
        assert_eq!(tree.resolve_aux_for_kv_block(leaf_anchor), Some(leaf_aux));
        assert_eq!(
            tree.resolve_kv_pages_for_aux_slot(leaf_aux),
            Some(lease.block_ids()[8..16].to_vec())
        );

        // Matched prefix via `match_prefix_bounded` resolves identically through the unified alias.
        let hit = tree.match_prefix_bounded(&tokens, 1);
        let hit_anchor = hit.leaf_anchor_block_id().expect("first node anchor");
        let hit_aux = hit.leaf_aux_id().expect("first node aux");
        assert_eq!(hit_anchor, lease.block_ids()[0]);
        assert_eq!(tree.resolve_aux_for_kv_block(hit_anchor), Some(hit_aux));
        assert_ne!(hit_aux, leaf_aux);

        drop(hit);
        drop(lease);
        let victims = tree.evict_lru_victims(1);
        assert_eq!(victims.len(), 1);
        assert_eq!(victims[0].anchor_block_id(), Some(leaf_anchor));
        assert_eq!(tree.alias_count(), 1);
        assert_eq!(tree.resolve_aux_for_kv_block(leaf_anchor), None);
        assert_eq!(tree.resolve_kv_pages_for_aux_slot(leaf_aux), None);
        drop(victims);
        assert_eq!(tree.evict_lru(1), 1);
        assert_eq!(tree.alias_count(), 0);
        assert_eq!(kv_pool.available(), 32);
        assert_eq!(aux_pool.available(), 4);
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
