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

//! Deterministic block-granular Radix prefix cache with RAII pin leases (`RadixLease`),
//! cascading LRU leaf eviction, and four-group FlatKV request admission (`RequestKvHandle`).

use std::collections::HashMap;
use std::fmt;
use std::sync::{Arc, Mutex};

use crate::{BlockLease, BlockPool, CacheGroupKind, plan_bounded_replay};

/// Errors produced during Radix prefix admission or block promotion.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum KvError {
    /// Block token granularity or SWA window parameter is zero or misaligned.
    InvalidGeometry(String),
    /// Physical cache group ran out of free blocks even after evicting unpinned cached leaves.
    OutOfBlocks {
        kind: CacheGroupKind,
        requested: usize,
        available: usize,
    },
    /// Token slice or lease count does not align with the tree's `block_tokens`.
    MisalignedPromotion(String),
}

impl fmt::Display for KvError {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::InvalidGeometry(msg) => write!(formatter, "invalid KV geometry: {msg}"),
            Self::OutOfBlocks {
                kind,
                requested,
                available,
            } => write!(
                formatter,
                "out of {kind:?} blocks: requested {requested}, available {available}"
            ),
            Self::MisalignedPromotion(msg) => {
                write!(formatter, "misaligned block promotion: {msg}")
            }
        }
    }
}

impl std::error::Error for KvError {}

struct RadixNode {
    parent: usize,
    token_key: Box<[i32]>,
    children: HashMap<Box<[i32]>, usize>,
    lease: Option<BlockLease>,
    pin_count: usize,
    last_access_tick: u64,
    evicted: bool,
}

struct PrefixTreeInner {
    block_tokens: usize,
    nodes: Vec<RadixNode>,
    free_node_slots: Vec<usize>,
    cached_blocks: usize,
    access_clock: u64,
}

impl PrefixTreeInner {
    fn next_tick(&mut self) -> u64 {
        self.access_clock = self.access_clock.wrapping_add(1);
        self.access_clock
    }

    fn alloc_node(&mut self, node: RadixNode) -> usize {
        if let Some(slot) = self.free_node_slots.pop() {
            self.nodes[slot] = node;
            slot
        } else {
            let slot = self.nodes.len();
            self.nodes.push(node);
            slot
        }
    }

    fn evict_lru_leaves(&mut self, target_blocks: usize) -> usize {
        let mut evicted_count = 0;
        while evicted_count < target_blocks {
            let candidate = self
                .nodes
                .iter()
                .enumerate()
                .skip(1)
                .filter(|(_, node)| {
                    !node.evicted && node.pin_count == 0 && node.children.is_empty()
                })
                .min_by_key(|(_, node)| node.last_access_tick)
                .map(|(idx, _)| idx);

            let Some(victim_idx) = candidate else {
                break;
            };

            let parent_idx = self.nodes[victim_idx].parent;
            let key = self.nodes[victim_idx].token_key.clone();
            self.nodes[parent_idx].children.remove(&key);
            // Dropping `lease` returns the physical block ID directly to `BlockPool`.
            let dropped_lease = self.nodes[victim_idx].lease.take();
            drop(dropped_lease);
            self.nodes[victim_idx].evicted = true;
            self.free_node_slots.push(victim_idx);
            self.cached_blocks = self.cached_blocks.saturating_sub(1);
            evicted_count += 1;
        }
        evicted_count
    }
}

/// Thread-safe Radix prefix tree over `CacheGroupKind::GlobalAndIndex` blocks.
#[derive(Clone)]
pub struct PrefixTree {
    inner: Arc<Mutex<PrefixTreeInner>>,
}

impl PrefixTree {
    pub fn new(block_tokens: u32) -> Result<Self, KvError> {
        if block_tokens == 0 {
            return Err(KvError::InvalidGeometry(
                "block_tokens must be positive".to_owned(),
            ));
        }
        let root = RadixNode {
            parent: 0,
            token_key: Box::new([]),
            children: HashMap::new(),
            lease: None,
            pin_count: 1,
            last_access_tick: 0,
            evicted: false,
        };
        Ok(Self {
            inner: Arc::new(Mutex::new(PrefixTreeInner {
                block_tokens: block_tokens as usize,
                nodes: vec![root],
                free_node_slots: Vec::new(),
                cached_blocks: 0,
                access_clock: 0,
            })),
        })
    }

    pub fn block_tokens(&self) -> u32 {
        let guard = self
            .inner
            .lock()
            .unwrap_or_else(|poisoned| poisoned.into_inner());
        guard.block_tokens as u32
    }

    /// Total physical blocks currently retained in the Radix tree.
    pub fn cached_blocks(&self) -> usize {
        let guard = self
            .inner
            .lock()
            .unwrap_or_else(|poisoned| poisoned.into_inner());
        guard.cached_blocks
    }

    /// Number of currently unpinned blocks eligible for LRU eviction.
    pub fn evictable_blocks(&self) -> usize {
        let guard = self
            .inner
            .lock()
            .unwrap_or_else(|poisoned| poisoned.into_inner());
        guard
            .nodes
            .iter()
            .skip(1)
            .filter(|node| !node.evicted && node.pin_count == 0)
            .count()
    }

    /// Match as many full `block_tokens` chunks from the start of `tokens` as exist in the tree,
    /// pinning every matched ancestor via the returned RAII [`RadixLease`].
    pub fn match_prefix(&self, tokens: &[i32]) -> RadixLease {
        let mut guard = self
            .inner
            .lock()
            .unwrap_or_else(|poisoned| poisoned.into_inner());
        let block_tokens = guard.block_tokens;
        let mut current = 0usize;
        let mut block_ids = Vec::new();

        for chunk in tokens.chunks_exact(block_tokens) {
            let next_idx = guard.nodes[current].children.get(chunk).copied();
            let Some(child_idx) = next_idx else {
                break;
            };
            let tick = guard.next_tick();
            let child = &mut guard.nodes[child_idx];
            child.pin_count += 1;
            child.last_access_tick = tick;
            let block_id = child
                .lease
                .as_ref()
                .expect("live radix node owns a physical block")
                .ids()[0];
            block_ids.push(block_id);
            current = child_idx;
        }

        let matched_tokens = (block_ids.len() * block_tokens) as u32;
        RadixLease {
            tree: Arc::clone(&self.inner),
            leaf_node: current,
            block_ids,
            matched_tokens,
        }
    }

    /// Insert or deduplicate full `block_tokens` chunks starting after `existing_lease`,
    /// consuming `new_single_blocks` (one single-block [`BlockLease`] per new chunk) and
    /// returning an extended [`RadixLease`] covering the full prefix.
    ///
    /// If another request concurrently inserted any of the same blocks, the duplicate
    /// [`BlockLease`] in `new_single_blocks` drops immediately and returns its physical ID
    /// to [`BlockPool`].
    pub fn extend_prefix(
        &self,
        existing_lease: RadixLease,
        extension_tokens: &[i32],
        new_single_blocks: Vec<BlockLease>,
    ) -> Result<RadixLease, KvError> {
        let mut guard = self
            .inner
            .lock()
            .unwrap_or_else(|poisoned| poisoned.into_inner());
        let block_tokens = guard.block_tokens;
        if extension_tokens.len() != new_single_blocks.len() * block_tokens {
            return Err(KvError::MisalignedPromotion(format!(
                "extension_tokens len {} != {} blocks * {}",
                extension_tokens.len(),
                new_single_blocks.len(),
                block_tokens
            )));
        }
        for lease in &new_single_blocks {
            if lease.ids().len() != 1 {
                return Err(KvError::MisalignedPromotion(
                    "each promoted BlockLease must own exactly 1 block".to_owned(),
                ));
            }
        }

        let mut current = existing_lease.leaf_node;
        let mut block_ids = existing_lease.block_ids.clone();
        // Consume `existing_lease` without decrementing the ancestor pins we are extending.
        let mut transferred = existing_lease;
        transferred.leaf_node = 0;
        transferred.block_ids.clear();
        drop(transferred);

        for (chunk, block_lease) in extension_tokens
            .chunks_exact(block_tokens)
            .zip(new_single_blocks)
        {
            let existing_child = guard.nodes[current].children.get(chunk).copied();
            if let Some(child_idx) = existing_child {
                // Another request already cached this exact block: drop `block_lease` so its
                // physical block returns immediately to `BlockPool`, and pin the canonical node.
                drop(block_lease);
                let tick = guard.next_tick();
                let child = &mut guard.nodes[child_idx];
                child.pin_count += 1;
                child.last_access_tick = tick;
                let block_id = child
                    .lease
                    .as_ref()
                    .expect("live radix node owns a physical block")
                    .ids()[0];
                block_ids.push(block_id);
                current = child_idx;
            } else {
                let block_id = block_lease.ids()[0];
                let tick = guard.next_tick();
                let key: Box<[i32]> = chunk.into();
                let node = RadixNode {
                    parent: current,
                    token_key: key.clone(),
                    children: HashMap::new(),
                    lease: Some(block_lease),
                    pin_count: 1,
                    last_access_tick: tick,
                    evicted: false,
                };
                let child_idx = guard.alloc_node(node);
                guard.nodes[current].children.insert(key, child_idx);
                guard.cached_blocks += 1;
                block_ids.push(block_id);
                current = child_idx;
            }
        }

        let matched_tokens = (block_ids.len() * block_tokens) as u32;
        Ok(RadixLease {
            tree: Arc::clone(&self.inner),
            leaf_node: current,
            block_ids,
            matched_tokens,
        })
    }

    /// Evict up to `target_blocks` unpinned leaf blocks in strict LRU order.
    pub fn evict_lru(&self, target_blocks: usize) -> usize {
        let mut guard = self
            .inner
            .lock()
            .unwrap_or_else(|poisoned| poisoned.into_inner());
        guard.evict_lru_leaves(target_blocks)
    }
}

/// RAII handle pinning a matched chain of blocks inside [`PrefixTree`].
///
/// Dropping `RadixLease` decrements `pin_count` along the ancestor chain without panicking
/// even if a thread unwinds.
pub struct RadixLease {
    tree: Arc<Mutex<PrefixTreeInner>>,
    leaf_node: usize,
    block_ids: Vec<i32>,
    matched_tokens: u32,
}

impl RadixLease {
    pub fn block_ids(&self) -> &[i32] {
        &self.block_ids
    }

    pub fn matched_tokens(&self) -> u32 {
        self.matched_tokens
    }
}

impl Drop for RadixLease {
    fn drop(&mut self) {
        if self.leaf_node == 0 {
            return;
        }
        let mut guard = self
            .tree
            .lock()
            .unwrap_or_else(|poisoned| poisoned.into_inner());
        let mut cursor = self.leaf_node;
        while cursor != 0 {
            let tick = guard.next_tick();
            let parent = guard.nodes[cursor].parent;
            let node = &mut guard.nodes[cursor];
            node.pin_count = node.pin_count.saturating_sub(1);
            node.last_access_tick = tick;
            cursor = parent;
        }
        self.leaf_node = 0;
    }
}

/// Prefill replay schedule computed when admitting a prompt against [`KvCacheManager`].
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct PrefillSchedule {
    /// Prefix tokens completely skipped (`0..skipped_tokens` are not executed).
    pub skipped_tokens: u32,
    /// Matched prefix tail tokens (`skipped_tokens..matched_tokens`) re-executed with
    /// `replay_mask = 1` to rebuild private `SwaWindow` and `CompressorTail` state.
    pub replay_tokens: u32,
    /// Uncached prompt suffix tokens (`matched_tokens..prompt_len`) executed with `replay_mask = 0`.
    pub fresh_tokens: u32,
    /// Token IDs scheduled for GPU execution (`prompt[skipped_tokens..]`).
    pub scheduled_tokens: Vec<i32>,
    /// Absolute sequence positions for each entry in `scheduled_tokens`.
    pub scheduled_positions: Vec<i32>,
    /// Per-token replay mask (`1` = SWA/Compressor replay only; `0` = write GlobalAndIndex + SWA).
    pub replay_mask: Vec<u8>,
}

/// Composite RAII handle owning every cache resource for one active request:
/// pinned Radix prefix blocks, uncached `GlobalAndIndex` tail blocks, private `SwaWindow`
/// ring pages, and private `CompressorTail` state slot.
pub struct RequestKvHandle {
    prefix_tree: PrefixTree,
    global_pool: BlockPool,
    radix_lease: Option<RadixLease>,
    tail_global_blocks: Vec<BlockLease>,
    swa_lease: BlockLease,
    compressor_lease: BlockLease,
    block_tokens: u32,
    schedule: PrefillSchedule,
}

impl RequestKvHandle {
    pub fn schedule(&self) -> &PrefillSchedule {
        &self.schedule
    }

    pub fn matched_prefix_tokens(&self) -> u32 {
        self.radix_lease
            .as_ref()
            .map_or(0, RadixLease::matched_tokens)
    }

    /// Assemble the complete ordered `GlobalAndIndex` page table (pinned prefix + tail blocks).
    pub fn global_page_table(&self) -> Vec<i32> {
        let prefix_ids = self
            .radix_lease
            .as_ref()
            .map_or(&[][..], RadixLease::block_ids);
        let mut table = Vec::with_capacity(prefix_ids.len() + self.tail_global_blocks.len());
        table.extend_from_slice(prefix_ids);
        for block in &self.tail_global_blocks {
            table.extend_from_slice(block.ids());
        }
        table
    }

    /// Physical `SwaWindow` ring page IDs owned exclusively by this request.
    pub fn swa_page_table(&self) -> &[i32] {
        self.swa_lease.ids()
    }

    /// Exclusive `CompressorTail` slot ID for this request.
    pub fn compressor_slot_id(&self) -> i32 {
        self.compressor_lease.ids()[0]
    }

    /// Allocate additional `GlobalAndIndex` tail blocks as decode grows past the current page
    /// boundary, evicting unpinned Radix leaves on demand if needed.
    pub fn ensure_capacity_for_seq_len(&mut self, seq_len_tokens: usize) -> Result<(), KvError> {
        let needed_total = seq_len_tokens.div_ceil(self.block_tokens as usize);
        let current_total = self
            .radix_lease
            .as_ref()
            .map_or(0, |lease| lease.block_ids().len())
            + self.tail_global_blocks.len();
        if needed_total <= current_total {
            return Ok(());
        }
        let missing = needed_total - current_total;
        let allocated = allocate_with_eviction(&self.global_pool, &self.prefix_tree, missing)?;
        self.tail_global_blocks.extend(allocated.split_individual());
        Ok(())
    }

    /// Promote all newly completed full `block_tokens` chunks from `tail_global_blocks` into the
    /// shared [`PrefixTree`] so subsequent requests can match them.
    pub fn commit_completed_blocks(
        &mut self,
        full_sequence_tokens: &[i32],
    ) -> Result<usize, KvError> {
        let block_tokens = self.block_tokens as usize;
        let prefix_blocks = self
            .radix_lease
            .as_ref()
            .map_or(0, |lease| lease.block_ids().len());
        let total_full_blocks = full_sequence_tokens.len() / block_tokens;
        if total_full_blocks <= prefix_blocks {
            return Ok(0);
        }
        let promotable = (total_full_blocks - prefix_blocks).min(self.tail_global_blocks.len());
        if promotable == 0 {
            return Ok(0);
        }
        let token_start = prefix_blocks * block_tokens;
        let token_end = (prefix_blocks + promotable) * block_tokens;
        let promoted_blocks: Vec<BlockLease> =
            self.tail_global_blocks.drain(..promotable).collect();
        let previous_lease = self
            .radix_lease
            .take()
            .expect("RequestKvHandle always holds a RadixLease");
        match self.prefix_tree.extend_prefix(
            previous_lease,
            &full_sequence_tokens[token_start..token_end],
            promoted_blocks,
        ) {
            Ok(updated_lease) => {
                self.radix_lease = Some(updated_lease);
                Ok(promotable)
            }
            Err(err) => {
                self.radix_lease = Some(
                    self.prefix_tree
                        .match_prefix(&full_sequence_tokens[..token_start]),
                );
                Err(err)
            }
        }
    }
}

fn allocate_with_eviction(
    global_pool: &BlockPool,
    prefix_tree: &PrefixTree,
    count: usize,
) -> Result<BlockLease, KvError> {
    if let Some(lease) = global_pool.allocate(count) {
        return Ok(lease);
    }
    let available_before = global_pool.available();
    let deficit = count.saturating_sub(available_before);
    prefix_tree.evict_lru(deficit);
    global_pool
        .allocate(count)
        .ok_or_else(|| KvError::OutOfBlocks {
            kind: CacheGroupKind::GlobalAndIndex,
            requested: count,
            available: global_pool.available(),
        })
}

/// Unified four-group FlatKV allocator + Radix prefix cache manager.
#[derive(Clone)]
pub struct KvCacheManager {
    global_pool: BlockPool,
    swa_pool: BlockPool,
    compressor_pool: BlockPool,
    prefix_tree: PrefixTree,
    block_tokens: u32,
    swa_window: u32,
    swa_pages_per_seq: usize,
}

impl KvCacheManager {
    pub fn new(
        global_blocks: usize,
        swa_blocks: usize,
        compressor_slots: usize,
        block_tokens: u32,
        swa_window: u32,
        swa_pages_per_seq: usize,
    ) -> Result<Self, KvError> {
        if block_tokens == 0 || swa_window == 0 || swa_pages_per_seq == 0 {
            return Err(KvError::InvalidGeometry(
                "block_tokens, swa_window, and swa_pages_per_seq must all be positive".to_owned(),
            ));
        }
        let min_swa_pages = (swa_window as usize).div_ceil(block_tokens as usize);
        if swa_pages_per_seq < min_swa_pages {
            return Err(KvError::InvalidGeometry(format!(
                "swa_pages_per_seq ({swa_pages_per_seq}) < ceil(swa_window / block_tokens) ({min_swa_pages})"
            )));
        }
        Ok(Self {
            global_pool: BlockPool::new(CacheGroupKind::GlobalAndIndex, global_blocks),
            swa_pool: BlockPool::new(CacheGroupKind::SwaWindow, swa_blocks),
            compressor_pool: BlockPool::new(CacheGroupKind::CompressorTail, compressor_slots),
            prefix_tree: PrefixTree::new(block_tokens)?,
            block_tokens,
            swa_window,
            swa_pages_per_seq,
        })
    }

    pub fn global_pool(&self) -> &BlockPool {
        &self.global_pool
    }

    pub fn swa_pool(&self) -> &BlockPool {
        &self.swa_pool
    }

    pub fn compressor_pool(&self) -> &BlockPool {
        &self.compressor_pool
    }

    pub fn prefix_tree(&self) -> &PrefixTree {
        &self.prefix_tree
    }

    /// Admit a prompt, matching any cached `GlobalAndIndex` prefix, planning the 128-token
    /// SWA/Compressor bounded replay, and acquiring RAII leases across all four cache groups.
    pub fn admit_request(&self, prompt_tokens: &[i32]) -> Result<RequestKvHandle, KvError> {
        if prompt_tokens.is_empty() {
            return Err(KvError::InvalidGeometry(
                "prompt_tokens must not be empty".to_owned(),
            ));
        }
        let radix_lease = self.prefix_tree.match_prefix(prompt_tokens);
        let matched_blocks = radix_lease.block_ids().len();
        let total_blocks = prompt_tokens.len().div_ceil(self.block_tokens as usize);
        let uncached_blocks = total_blocks.saturating_sub(matched_blocks);

        // Ensure at least 1 fresh token is executed to produce next-token logits when the entire
        // prompt is an exact multiple of `block_tokens` and matched 100% in the Radix tree.
        let matched_tokens_raw = radix_lease.matched_tokens();
        let effective_matched =
            matched_tokens_raw.min(prompt_tokens.len().saturating_sub(1) as u32);
        let (skipped_tokens, replay_tokens) =
            plan_bounded_replay(effective_matched, self.swa_window);
        let fresh_tokens = prompt_tokens.len() as u32 - effective_matched;

        let tail_lease = if uncached_blocks > 0 {
            Some(allocate_with_eviction(
                &self.global_pool,
                &self.prefix_tree,
                uncached_blocks,
            )?)
        } else {
            None
        };

        let swa_lease = self
            .swa_pool
            .allocate(self.swa_pages_per_seq)
            .ok_or_else(|| KvError::OutOfBlocks {
                kind: CacheGroupKind::SwaWindow,
                requested: self.swa_pages_per_seq,
                available: self.swa_pool.available(),
            })?;

        let compressor_lease =
            self.compressor_pool
                .allocate(1)
                .ok_or_else(|| KvError::OutOfBlocks {
                    kind: CacheGroupKind::CompressorTail,
                    requested: 1,
                    available: self.compressor_pool.available(),
                })?;

        let scheduled_start = skipped_tokens as usize;
        let scheduled_tokens = prompt_tokens[scheduled_start..].to_vec();
        let scheduled_positions: Vec<i32> =
            (skipped_tokens as i32..prompt_tokens.len() as i32).collect();
        let mut replay_mask = Vec::with_capacity(scheduled_tokens.len());
        replay_mask.resize(replay_tokens as usize, 1u8);
        replay_mask.resize(scheduled_tokens.len(), 0u8);

        let schedule = PrefillSchedule {
            skipped_tokens,
            replay_tokens,
            fresh_tokens,
            scheduled_tokens,
            scheduled_positions,
            replay_mask,
        };

        let tail_global_blocks = tail_lease
            .map(BlockLease::split_individual)
            .unwrap_or_default();

        Ok(RequestKvHandle {
            prefix_tree: self.prefix_tree.clone(),
            global_pool: self.global_pool.clone(),
            radix_lease: Some(radix_lease),
            tail_global_blocks,
            swa_lease,
            compressor_lease,
            block_tokens: self.block_tokens,
            schedule,
        })
    }
}
