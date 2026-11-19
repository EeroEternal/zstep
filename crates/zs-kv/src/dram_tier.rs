// Copyright (c) 2026 zstep Contributors
// SPDX-License-Identifier: MIT

//! Exclusive Host DRAM KV tier with Victim Chain demotion and 10% HBM Loading Area
//! (Issue #40 Proposals 1 & 2).
//!
//! Architecture:
//! - **Exclusive DRAM**: a cached prefix block resides in either HBM [`crate::PrefixTree`]
//!   *or* [`DramPrefixCache`], never duplicated in both.
//! - **Victim Chain**: HBM LRU victims are demoted into Host DRAM via D2H `KvTransfer`
//!   while their HBM [`crate::BlockLease`] remains pinned; the HBM lease is returned to the
//!   HBM [`crate::BlockPool`] only after the worker completes the D2H transfer.
//! - **10% HBM Loading Area**: reserves `loading_area_pages` of HBM capacity exclusively for
//!   DRAM → HBM reloads (`pending → loading → ready`), preventing reload/eviction deadlocks.

use std::collections::HashMap;

use crate::KvError;

/// Default fraction of total HBM KV pages reserved as the DRAM → HBM Loading Area.
pub const DEFAULT_LOADING_AREA_RATIO: f64 = 0.10;

/// Configuration for the Host DRAM KV cache tier and HBM Loading Area.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct DramTierConfig {
    /// Total logical KV blocks allocated in pinned Host DRAM (`0` disables DRAM tier).
    pub num_dram_blocks: usize,
    /// Fraction of total HBM KV pages reserved exclusively for DRAM → HBM reloads
    /// (`(0.0, 0.5]`, default `0.10`). Ignored when `num_dram_blocks == 0`.
    pub loading_area_ratio: f64,
}

impl Default for DramTierConfig {
    fn default() -> Self {
        Self {
            num_dram_blocks: 0,
            loading_area_ratio: DEFAULT_LOADING_AREA_RATIO,
        }
    }
}

impl DramTierConfig {
    pub fn disabled() -> Self {
        Self::default()
    }

    pub fn enabled(num_dram_blocks: usize, loading_area_ratio: f64) -> Result<Self, KvError> {
        if num_dram_blocks > 0
            && (!loading_area_ratio.is_finite()
                || loading_area_ratio <= 0.0
                || loading_area_ratio > 0.5)
        {
            return Err(KvError::InvalidGeometry(format!(
                "loading_area_ratio={loading_area_ratio} must be in (0.0, 0.5]"
            )));
        }
        Ok(Self {
            num_dram_blocks,
            loading_area_ratio,
        })
    }

    #[inline]
    pub fn is_enabled(self) -> bool {
        self.num_dram_blocks > 0
    }

    /// Number of HBM KV pages reserved exclusively for DRAM → HBM reloads.
    /// Returns `0` when the DRAM tier is disabled or `total_hbm_pages <= 1`.
    pub fn loading_area_pages(self, total_hbm_pages: usize) -> usize {
        if !self.is_enabled() || total_hbm_pages <= 1 {
            return 0;
        }
        let raw = ((total_hbm_pages as f64) * self.loading_area_ratio).ceil() as usize;
        raw.clamp(1, (total_hbm_pages / 2).max(1))
    }
}

/// Single physical page transfer between HBM and Host DRAM executed by `StepOpcode::KvTransfer`.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct KvTransferOp {
    pub hbm_page_id: i32,
    pub dram_slot_id: i32,
    /// `false` = D2H (HBM → Host DRAM offload), `true` = H2D (Host DRAM → HBM reload).
    pub is_h2d: bool,
}

/// Cumulative observability counters for the Host DRAM KV tier.
#[derive(Debug, Clone, Copy, Default, PartialEq, Eq)]
pub struct DramTierStats {
    pub dram_capacity_blocks: usize,
    pub dram_occupied_blocks: usize,
    pub loading_area_pages: usize,
    pub dram_offloads_blocks: u64,
    pub dram_reloads_blocks: u64,
    pub dram_evictions_blocks: u64,
    pub dram_hits_tokens: u64,
}

#[derive(Debug, Clone)]
struct DramEntry {
    dram_slot_id: i32,
    last_access_tick: u64,
}

/// Exclusive Host DRAM prefix block index with strict LRU eviction and slot recycling.
pub struct DramPrefixCache {
    block_tokens: usize,
    capacity_blocks: usize,
    loading_area_pages: usize,
    entries: HashMap<Vec<i32>, DramEntry>,
    free_slots: Vec<i32>,
    access_clock: u64,
    offloads_blocks: u64,
    reloads_blocks: u64,
    evictions_blocks: u64,
    hits_tokens: u64,
}

impl DramPrefixCache {
    pub fn new(
        block_tokens: u32,
        total_hbm_pages: usize,
        config: DramTierConfig,
    ) -> Result<Self, KvError> {
        if block_tokens == 0 {
            return Err(KvError::InvalidGeometry(
                "block_tokens must be positive".to_owned(),
            ));
        }
        if config.num_dram_blocks == 0 {
            return Err(KvError::InvalidGeometry(
                "num_dram_blocks must be positive when constructing DramPrefixCache".to_owned(),
            ));
        }
        let validated = DramTierConfig::enabled(config.num_dram_blocks, config.loading_area_ratio)?;
        let loading_area_pages = validated.loading_area_pages(total_hbm_pages);
        let free_slots = (0..validated.num_dram_blocks as i32).rev().collect();
        Ok(Self {
            block_tokens: block_tokens as usize,
            capacity_blocks: validated.num_dram_blocks,
            loading_area_pages,
            entries: HashMap::with_capacity(validated.num_dram_blocks),
            free_slots,
            access_clock: 0,
            offloads_blocks: 0,
            reloads_blocks: 0,
            evictions_blocks: 0,
            hits_tokens: 0,
        })
    }

    #[inline]
    pub fn block_tokens(&self) -> usize {
        self.block_tokens
    }

    #[inline]
    pub fn loading_area_pages(&self) -> usize {
        self.loading_area_pages
    }

    #[inline]
    pub fn occupied_blocks(&self) -> usize {
        self.entries.len()
    }

    #[inline]
    pub fn capacity_blocks(&self) -> usize {
        self.capacity_blocks
    }

    pub fn stats(&self) -> DramTierStats {
        DramTierStats {
            dram_capacity_blocks: self.capacity_blocks,
            dram_occupied_blocks: self.entries.len(),
            loading_area_pages: self.loading_area_pages,
            dram_offloads_blocks: self.offloads_blocks,
            dram_reloads_blocks: self.reloads_blocks,
            dram_evictions_blocks: self.evictions_blocks,
            dram_hits_tokens: self.hits_tokens,
        }
    }

    fn next_tick(&mut self) -> u64 {
        self.access_clock = self.access_clock.wrapping_add(1);
        self.access_clock
    }

    /// Read-only probe: count contiguous DRAM-resident blocks immediately following
    /// `hbm_matched_tokens`, always leaving at least 1 trailing prompt token uncached
    /// so the model computes final logits.
    pub fn peek_continuation_blocks(
        &self,
        prompt_tokens: &[i32],
        hbm_matched_tokens: usize,
    ) -> usize {
        if prompt_tokens.len() <= hbm_matched_tokens + 1 {
            return 0;
        }
        let max_prefix_len = prompt_tokens.len() - 1;
        let mut blocks = 0usize;
        let mut end = hbm_matched_tokens + self.block_tokens;
        while end <= max_prefix_len {
            if self.entries.contains_key(&prompt_tokens[..end]) {
                blocks += 1;
                end += self.block_tokens;
            } else {
                break;
            }
        }
        blocks
    }

    /// Demote an HBM victim block into Exclusive Host DRAM.
    ///
    /// Returns `Some(KvTransferOp { is_h2d: false, .. })` when a D2H copy is required, or
    /// `None` if the exact prefix path is already resident in DRAM (refreshes LRU tick).
    pub fn admit_victim(&mut self, token_path: Vec<i32>, hbm_page_id: i32) -> Option<KvTransferOp> {
        let tick = self.next_tick();
        if let Some(existing) = self.entries.get_mut(&token_path) {
            existing.last_access_tick = tick;
            return None;
        }
        let dram_slot_id = if let Some(slot) = self.free_slots.pop() {
            slot
        } else {
            let lru_key = self
                .entries
                .iter()
                .min_by_key(|(_, entry)| entry.last_access_tick)
                .map(|(k, _)| k.clone())?;
            let evicted = self.entries.remove(&lru_key)?;
            self.evictions_blocks += 1;
            evicted.dram_slot_id
        };
        self.entries.insert(
            token_path,
            DramEntry {
                dram_slot_id,
                last_access_tick: tick,
            },
        );
        self.offloads_blocks += 1;
        Some(KvTransferOp {
            hbm_page_id,
            dram_slot_id,
            is_h2d: false,
        })
    }

    /// Roll back a DRAM admission if the worker's D2H `KvTransfer` fails before completion.
    pub fn rollback_failed_offload(&mut self, token_path: &[i32]) {
        if let Some(removed) = self.entries.remove(token_path) {
            self.free_slots.push(removed.dram_slot_id);
        }
    }

    /// Pop up to `max_blocks` contiguous DRAM blocks following `hbm_matched_tokens` for
    /// exclusive reload into HBM (removing them from the DRAM index immediately).
    ///
    /// Caller MUST call [`Self::release_dram_slots`] once the H2D `KvTransfer` completes
    /// (or fails) so the DRAM slot IDs return to the free pool.
    pub fn take_continuation_for_reload(
        &mut self,
        prompt_tokens: &[i32],
        hbm_matched_tokens: usize,
        max_blocks: usize,
    ) -> Vec<(Vec<i32>, i32)> {
        if max_blocks == 0 || prompt_tokens.len() <= hbm_matched_tokens + 1 {
            return Vec::new();
        }
        let max_prefix_len = prompt_tokens.len() - 1;
        let mut taken = Vec::new();
        let mut end = hbm_matched_tokens + self.block_tokens;
        while taken.len() < max_blocks && end <= max_prefix_len {
            let key = &prompt_tokens[..end];
            if let Some(entry) = self.entries.remove(key) {
                self.reloads_blocks += 1;
                self.hits_tokens += self.block_tokens as u64;
                taken.push((key.to_vec(), entry.dram_slot_id));
                end += self.block_tokens;
            } else {
                break;
            }
        }
        taken
    }

    /// Return DRAM slot IDs to the free list after their H2D reload transfer completes.
    pub fn release_dram_slots(&mut self, slots: &[i32]) {
        self.free_slots.extend_from_slice(slots);
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn loading_area_reserves_ten_percent_and_zero_when_disabled() {
        let disabled = DramTierConfig::disabled();
        assert_eq!(disabled.loading_area_pages(2560), 0);

        let cfg = DramTierConfig::enabled(512, 0.10).expect("valid config");
        assert_eq!(cfg.loading_area_pages(2560), 256);
        assert_eq!(cfg.loading_area_pages(10), 1);
        assert_eq!(cfg.loading_area_pages(15), 2);

        assert!(DramTierConfig::enabled(64, 0.0).is_err());
        assert!(DramTierConfig::enabled(64, 0.6).is_err());
    }

    #[test]
    fn exclusive_dram_victim_demotion_lru_eviction_and_reload() {
        let cfg = DramTierConfig::enabled(2, 0.10).expect("valid");
        let mut dram = DramPrefixCache::new(4, 20, cfg).expect("dram cache");
        assert_eq!(dram.loading_area_pages(), 2);

        // Demote block A ([1,2,3,4]) and block A->B ([1..8]) into 2-slot DRAM.
        let p_a = vec![1, 2, 3, 4];
        let p_ab = vec![1, 2, 3, 4, 5, 6, 7, 8];
        let op1 = dram.admit_victim(p_a.clone(), 10).expect("op1");
        assert!(!op1.is_h2d);
        assert_eq!(op1.hbm_page_id, 10);
        let op2 = dram.admit_victim(p_ab.clone(), 11).expect("op2");
        assert_ne!(op1.dram_slot_id, op2.dram_slot_id);
        assert_eq!(dram.occupied_blocks(), 2);

        // Duplicate demotion of p_a refreshes LRU tick without triggering D2H.
        assert!(dram.admit_victim(p_a.clone(), 12).is_none());

        // Prompt [1..10] with 0 HBM match sees 2 contiguous DRAM blocks ([1..4], [1..8]).
        let prompt: Vec<i32> = (1..=10).collect();
        assert_eq!(dram.peek_continuation_blocks(&prompt, 0), 2);

        // Exclusive reload takes both blocks out of DRAM and recycles slots on release.
        let taken = dram.take_continuation_for_reload(&prompt, 0, 2);
        assert_eq!(taken.len(), 2);
        assert_eq!(dram.occupied_blocks(), 0);
        let slots: Vec<i32> = taken.iter().map(|(_, s)| *s).collect();
        dram.release_dram_slots(&slots);

        // Re-fill 2 slots, then insert a 3rd block -> must evict oldest LRU block.
        assert!(dram.admit_victim(vec![10, 11, 12, 13], 20).is_some());
        assert!(dram.admit_victim(vec![20, 21, 22, 23], 21).is_some());
        let op3 = dram
            .admit_victim(vec![30, 31, 32, 33], 22)
            .expect("evict lru");
        assert_eq!(dram.occupied_blocks(), 2);
        assert_eq!(dram.stats().dram_evictions_blocks, 1);
        assert_eq!(dram.peek_continuation_blocks(&[10, 11, 12, 13, 99], 0), 0);
        assert_eq!(dram.peek_continuation_blocks(&[30, 31, 32, 33, 99], 0), 1);
        assert!(op3.dram_slot_id >= 0);
    }

    #[test]
    fn partial_hbm_match_plus_dram_continuation_promotes_suffix_into_prefix_tree() {
        use crate::{BlockPool, CacheGroupKind, PrefixTree};

        let hbm_pool = BlockPool::new(CacheGroupKind::GlobalAndIndex, 16);
        let tree = PrefixTree::with_pages_per_node(4, 1).expect("prefix tree");
        let cfg = DramTierConfig::enabled(4, 0.10).expect("cfg");
        let mut dram = DramPrefixCache::new(4, 16, cfg).expect("dram");

        // Promote 3 blocks (12 tokens) into HBM PrefixTree, then unpin.
        let prompt: Vec<i32> = (1..=13).collect();
        let init_leases = (0..3)
            .map(|_| hbm_pool.allocate(1).expect("page"))
            .collect();
        let base = tree.match_prefix(&prompt[..12]);
        let promoted = tree
            .extend_prefix(base, &prompt[..12], init_leases)
            .expect("initial promote");
        assert_eq!(promoted.matched_tokens(), 12);
        drop(promoted);

        // Evict the leaf block (tokens 8..12) from HBM into DRAM, leaving blocks 0..8 in HBM.
        let victims = tree.evict_lru_victims(1);
        assert_eq!(victims.len(), 1);
        assert_eq!(victims[0].token_path, prompt[..12]);
        let op = dram
            .admit_victim(victims[0].token_path.clone(), victims[0].lease.ids()[0])
            .expect("d2h op");
        assert!(!op.is_h2d);
        drop(victims);

        // New request matches 8 tokens in HBM and 1 continuation block (4 tokens) in DRAM.
        let hbm_matched_tokens = 8usize;
        assert_eq!(
            dram.peek_continuation_blocks(&prompt, hbm_matched_tokens),
            1
        );
        let taken = dram.take_continuation_for_reload(&prompt, hbm_matched_tokens, 1);
        assert_eq!(taken.len(), 1);
        let reload_leases = vec![hbm_pool.allocate(1).expect("reload page")];
        let full_len = hbm_matched_tokens + reload_leases.len() * 4;
        let base_lease = tree.match_prefix_bounded(&prompt, hbm_matched_tokens / 4);
        assert_eq!(base_lease.matched_tokens(), 8);
        let merged = tree
            .extend_prefix(
                base_lease,
                &prompt[hbm_matched_tokens..full_len],
                reload_leases,
            )
            .expect("suffix promotion must succeed");
        assert_eq!(merged.matched_tokens(), 12);
        dram.release_dram_slots(&[taken[0].1]);
    }
}
