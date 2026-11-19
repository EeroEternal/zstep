// Copyright (c) 2026 zstep Contributors
// SPDX-License-Identifier: MIT

//! Unified KV / recurrent-state Block-ID aliasing and 128-token fine-grained sub-page allocation
//! for hybrid attention + GatedDeltaNet (Qwen3.5) models.
//!
//! 1. [`HybridBlockAliasTable`] maps each promoted hybrid prefix node's canonical anchor KV
//!    block ID (`kv_page_ids[0]`) bidirectionally to its co-owned GDN recurrent checkpoint slot,
//!    providing a single unified Block-ID namespace across attention KV and recurrent state.
//! 2. [`SubPageAllocator`] packs fine-grained 128-token slices into larger backing [`BlockPool`]
//!    blocks with best-fit packing and RAII [`SubPageLease`] reclamation, compressing internal
//!    fragmentation for long-session GDN checkpoints.

use std::collections::HashMap;
use std::sync::{Arc, Mutex};

use crate::{BlockLease, BlockPool, KvError};

/// Default fine-grained GDN checkpoint / sub-page token granularity (aligned with DeepSeek-V4.1
/// and SSEngine's 128-token recurrent checkpoint spacing).
pub const GDN_FINE_GRAINED_TOKENS: u32 = 128;

/// Bidirectional alias entry binding one hybrid prefix node's physical KV pages to its co-owned
/// recurrent checkpoint slot under a single canonical Block-ID (`anchor_block_id`).
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct HybridBlockAliasEntry {
    pub anchor_block_id: i32,
    pub kv_page_ids: Vec<i32>,
    pub recurrent_slot_id: i32,
}

/// O(1) bidirectional alias map between KV page groups and GDN recurrent checkpoint slots.
///
/// Because every live prefix node in [`crate::PrefixTree`] owns disjoint physical KV pages,
/// `kv_page_ids[0]` serves as the canonical unified Block-ID without allocating a third ID space.
#[derive(Debug, Default)]
pub struct HybridBlockAliasTable {
    by_anchor: HashMap<i32, HybridBlockAliasEntry>,
    slot_to_anchor: HashMap<i32, i32>,
}

impl HybridBlockAliasTable {
    pub fn new() -> Self {
        Self::default()
    }

    /// Register a unified alias binding `kv_page_ids` (anchored at `kv_page_ids[0]`) to
    /// `recurrent_slot_id`. Returns the canonical `anchor_block_id`.
    pub fn insert(&mut self, kv_page_ids: &[i32], recurrent_slot_id: i32) -> Result<i32, KvError> {
        let Some(&anchor_block_id) = kv_page_ids.first() else {
            return Err(KvError::InvalidGeometry(
                "hybrid alias requires at least one KV page ID".to_owned(),
            ));
        };
        if self.by_anchor.contains_key(&anchor_block_id) {
            return Err(KvError::InvalidGeometry(format!(
                "duplicate hybrid anchor_block_id={anchor_block_id}"
            )));
        }
        if self.slot_to_anchor.contains_key(&recurrent_slot_id) {
            return Err(KvError::InvalidGeometry(format!(
                "duplicate hybrid recurrent_slot_id={recurrent_slot_id}"
            )));
        }
        self.by_anchor.insert(
            anchor_block_id,
            HybridBlockAliasEntry {
                anchor_block_id,
                kv_page_ids: kv_page_ids.to_vec(),
                recurrent_slot_id,
            },
        );
        self.slot_to_anchor
            .insert(recurrent_slot_id, anchor_block_id);
        Ok(anchor_block_id)
    }

    /// Remove the alias entry anchored at `anchor_block_id` (e.g. on LRU eviction).
    pub fn remove_by_anchor(&mut self, anchor_block_id: i32) -> Option<HybridBlockAliasEntry> {
        let entry = self.by_anchor.remove(&anchor_block_id)?;
        self.slot_to_anchor.remove(&entry.recurrent_slot_id);
        Some(entry)
    }

    /// Resolve the GDN recurrent checkpoint slot aliased to `anchor_block_id`.
    pub fn recurrent_slot_for_block(&self, anchor_block_id: i32) -> Option<i32> {
        self.by_anchor
            .get(&anchor_block_id)
            .map(|entry| entry.recurrent_slot_id)
    }

    /// Resolve the canonical anchor Block-ID aliased to `recurrent_slot_id`.
    pub fn anchor_for_recurrent_slot(&self, recurrent_slot_id: i32) -> Option<i32> {
        self.slot_to_anchor.get(&recurrent_slot_id).copied()
    }

    /// Resolve the full KV page list aliased to `recurrent_slot_id`.
    pub fn kv_pages_for_recurrent_slot(&self, recurrent_slot_id: i32) -> Option<&[i32]> {
        let anchor = self.slot_to_anchor.get(&recurrent_slot_id)?;
        self.by_anchor
            .get(anchor)
            .map(|entry| entry.kv_page_ids.as_slice())
    }

    pub fn len(&self) -> usize {
        self.by_anchor.len()
    }

    pub fn is_empty(&self) -> bool {
        self.by_anchor.is_empty()
    }
}

struct BackingPageState {
    _lease: BlockLease,
    /// Bit `i` == 1 means sub-page slice `i` is currently allocated.
    used_mask: u64,
}

struct SubPageAllocatorInner {
    pool: BlockPool,
    backing_page_tokens: u32,
    slice_tokens: u32,
    slices_per_page: u32,
    pages: HashMap<i32, BackingPageState>,
    active_slices: usize,
}

/// Fine-grained sub-page allocator packing `slice_tokens` (e.g. 128-token) slices into larger
/// physical [`BlockPool`] pages with best-fit packing and automatic RAII backing-page release.
#[derive(Clone)]
pub struct SubPageAllocator {
    inner: Arc<Mutex<SubPageAllocatorInner>>,
}

impl SubPageAllocator {
    pub fn new(
        pool: BlockPool,
        backing_page_tokens: u32,
        slice_tokens: u32,
    ) -> Result<Self, KvError> {
        if backing_page_tokens == 0 || slice_tokens == 0 {
            return Err(KvError::InvalidGeometry(
                "backing_page_tokens and slice_tokens must be positive".to_owned(),
            ));
        }
        if backing_page_tokens % slice_tokens != 0 {
            return Err(KvError::InvalidGeometry(format!(
                "backing_page_tokens ({backing_page_tokens}) must be a multiple of slice_tokens ({slice_tokens})"
            )));
        }
        let slices_per_page = backing_page_tokens / slice_tokens;
        if slices_per_page == 0 || slices_per_page > 64 {
            return Err(KvError::InvalidGeometry(format!(
                "slices_per_page ({slices_per_page}) must be in 1..=64"
            )));
        }
        Ok(Self {
            inner: Arc::new(Mutex::new(SubPageAllocatorInner {
                pool,
                backing_page_tokens,
                slice_tokens,
                slices_per_page,
                pages: HashMap::new(),
                active_slices: 0,
            })),
        })
    }

    pub fn slice_tokens(&self) -> u32 {
        self.inner
            .lock()
            .unwrap_or_else(|poisoned| poisoned.into_inner())
            .slice_tokens
    }

    pub fn backing_page_tokens(&self) -> u32 {
        self.inner
            .lock()
            .unwrap_or_else(|poisoned| poisoned.into_inner())
            .backing_page_tokens
    }

    pub fn slices_per_page(&self) -> u32 {
        self.inner
            .lock()
            .unwrap_or_else(|poisoned| poisoned.into_inner())
            .slices_per_page
    }

    pub fn active_slices(&self) -> usize {
        self.inner
            .lock()
            .unwrap_or_else(|poisoned| poisoned.into_inner())
            .active_slices
    }

    pub fn backing_pages_in_use(&self) -> usize {
        self.inner
            .lock()
            .unwrap_or_else(|poisoned| poisoned.into_inner())
            .pages
            .len()
    }

    /// Fraction of allocated backing-page slice capacity currently unused (`0.0` = zero waste).
    pub fn fragmentation_ratio(&self) -> f64 {
        let guard = self
            .inner
            .lock()
            .unwrap_or_else(|poisoned| poisoned.into_inner());
        let total_slots = guard.pages.len() * guard.slices_per_page as usize;
        if total_slots == 0 {
            return 0.0;
        }
        total_slots.saturating_sub(guard.active_slices) as f64 / total_slots as f64
    }

    /// Allocate one `slice_tokens` sub-page slice, packing into the fullest partially-used
    /// backing page first (best-fit) before requesting a new backing page from [`BlockPool`].
    pub fn allocate_slice(&self) -> Option<SubPageLease> {
        let mut guard = self
            .inner
            .lock()
            .unwrap_or_else(|poisoned| poisoned.into_inner());
        let full_mask: u64 = if guard.slices_per_page == 64 {
            u64::MAX
        } else {
            (1u64 << guard.slices_per_page) - 1
        };

        // Best-fit: pick the partial page with the most used bits (lowest free count),
        // breaking ties deterministically by smallest backing_block_id.
        let best_partial = guard
            .pages
            .iter()
            .filter(|(_, state)| state.used_mask != full_mask)
            .max_by(|(id_a, a), (id_b, b)| {
                a.used_mask
                    .count_ones()
                    .cmp(&b.used_mask.count_ones())
                    .then_with(|| id_b.cmp(id_a))
            })
            .map(|(&block_id, _)| block_id);

        let (backing_block_id, slice_index) = if let Some(block_id) = best_partial {
            let state = guard
                .pages
                .get_mut(&block_id)
                .expect("selected partial backing page must exist");
            let free_bits = (!state.used_mask) & full_mask;
            let idx = free_bits.trailing_zeros();
            state.used_mask |= 1u64 << idx;
            (block_id, idx)
        } else {
            let lease = guard.pool.allocate(1)?;
            let block_id = lease.ids()[0];
            guard.pages.insert(
                block_id,
                BackingPageState {
                    _lease: lease,
                    used_mask: 1u64,
                },
            );
            (block_id, 0)
        };

        guard.active_slices += 1;
        let slice_tokens = guard.slice_tokens;
        Some(SubPageLease {
            allocator: Arc::clone(&self.inner),
            backing_block_id,
            slice_index,
            slice_tokens,
        })
    }
}

/// RAII handle owning one fine-grained sub-page slice inside a backing [`BlockPool`] page.
///
/// Dropping the last live `SubPageLease` on a backing page immediately returns that physical
/// backing page to [`BlockPool`].
pub struct SubPageLease {
    allocator: Arc<Mutex<SubPageAllocatorInner>>,
    backing_block_id: i32,
    slice_index: u32,
    slice_tokens: u32,
}

impl SubPageLease {
    pub fn backing_block_id(&self) -> i32 {
        self.backing_block_id
    }

    pub fn slice_index(&self) -> u32 {
        self.slice_index
    }

    pub fn token_offset(&self) -> u32 {
        self.slice_index * self.slice_tokens
    }

    pub fn slice_tokens(&self) -> u32 {
        self.slice_tokens
    }
}

impl Drop for SubPageLease {
    fn drop(&mut self) {
        let mut guard = self
            .allocator
            .lock()
            .unwrap_or_else(|poisoned| poisoned.into_inner());
        guard.active_slices = guard.active_slices.saturating_sub(1);
        let remove_page = if let Some(state) = guard.pages.get_mut(&self.backing_block_id) {
            state.used_mask &= !(1u64 << self.slice_index);
            state.used_mask == 0
        } else {
            false
        };
        if remove_page {
            // Dropping `BackingPageState` releases its `BlockLease` back to `BlockPool`.
            guard.pages.remove(&self.backing_block_id);
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::CacheGroupKind;

    #[test]
    fn hybrid_block_alias_table_bidirectional_and_collision_guards() {
        let mut table = HybridBlockAliasTable::new();
        assert!(table.is_empty());

        let anchor = table.insert(&[10, 11, 12, 13, 14, 15, 16, 17], 42).unwrap();
        assert_eq!(anchor, 10);
        assert_eq!(table.len(), 1);
        assert_eq!(table.recurrent_slot_for_block(10), Some(42));
        assert_eq!(table.anchor_for_recurrent_slot(42), Some(10));
        assert_eq!(
            table.kv_pages_for_recurrent_slot(42),
            Some(&[10, 11, 12, 13, 14, 15, 16, 17][..])
        );

        // Duplicate anchor or slot is rejected.
        assert!(table.insert(&[10, 99], 43).is_err());
        assert!(table.insert(&[20, 21], 42).is_err());
        assert!(table.insert(&[], 44).is_err());

        let removed = table.remove_by_anchor(10).unwrap();
        assert_eq!(removed.recurrent_slot_id, 42);
        assert!(table.is_empty());
        assert_eq!(table.recurrent_slot_for_block(10), None);
        assert_eq!(table.anchor_for_recurrent_slot(42), None);
    }

    #[test]
    fn sub_page_allocator_packs_128_token_slices_and_reclaims_backing_pages() {
        let pool = BlockPool::new(CacheGroupKind::GlobalAndIndex, 2);
        let alloc = SubPageAllocator::new(pool.clone(), 512, GDN_FINE_GRAINED_TOKENS).unwrap();
        assert_eq!(alloc.slices_per_page(), 4);
        assert_eq!(alloc.backing_pages_in_use(), 0);

        // First 4 slices pack into 1 single backing page (zero fragmentation at 4/4).
        let s0 = alloc.allocate_slice().unwrap();
        let s1 = alloc.allocate_slice().unwrap();
        let s2 = alloc.allocate_slice().unwrap();
        let s3 = alloc.allocate_slice().unwrap();
        assert_eq!(alloc.backing_pages_in_use(), 1);
        assert_eq!(pool.available(), 1);
        assert_eq!(s0.backing_block_id(), s3.backing_block_id());
        assert_eq!(
            [
                s0.token_offset(),
                s1.token_offset(),
                s2.token_offset(),
                s3.token_offset()
            ],
            [0, 128, 256, 384]
        );
        assert_eq!(alloc.fragmentation_ratio(), 0.0);

        // 5th slice allocates the 2nd backing page.
        let s4 = alloc.allocate_slice().unwrap();
        assert_eq!(alloc.backing_pages_in_use(), 2);
        assert_eq!(pool.available(), 0);

        // Dropping s1 frees a slot in backing page 0; best-fit reuses page 0 (3/4 full > 1/4 full).
        let page0 = s0.backing_block_id();
        drop(s1);
        let s5 = alloc.allocate_slice().unwrap();
        assert_eq!(s5.backing_block_id(), page0);
        assert_eq!(s5.slice_index(), 1);

        // Dropping s4 (only slice on page 1) immediately returns page 1 to BlockPool.
        drop(s4);
        assert_eq!(alloc.backing_pages_in_use(), 1);
        assert_eq!(pool.available(), 1);

        drop((s0, s2, s3, s5));
        assert_eq!(alloc.active_slices(), 0);
        assert_eq!(alloc.backing_pages_in_use(), 0);
        assert_eq!(pool.available(), 2);
    }

    #[test]
    fn sub_page_allocator_handles_64_slices_invalid_geometry_and_exhaustion() {
        let pool = BlockPool::new(CacheGroupKind::GlobalAndIndex, 1);
        assert!(SubPageAllocator::new(pool.clone(), 0, 128).is_err());
        assert!(SubPageAllocator::new(pool.clone(), 512, 0).is_err());
        assert!(SubPageAllocator::new(pool.clone(), 300, 128).is_err());
        assert!(SubPageAllocator::new(pool.clone(), 128 * 65, 128).is_err());

        let alloc = SubPageAllocator::new(pool.clone(), 128 * 64, 128).unwrap();
        assert_eq!(alloc.slices_per_page(), 64);
        let slices: Vec<_> = (0..64)
            .map(|_| alloc.allocate_slice().expect("slice within 64-bit mask"))
            .collect();
        assert_eq!(alloc.active_slices(), 64);
        assert_eq!(alloc.fragmentation_ratio(), 0.0);
        assert!(alloc.allocate_slice().is_none());

        drop(slices);
        assert_eq!(alloc.active_slices(), 0);
        assert_eq!(alloc.backing_pages_in_use(), 0);
        assert_eq!(pool.available(), 1);
    }
}
