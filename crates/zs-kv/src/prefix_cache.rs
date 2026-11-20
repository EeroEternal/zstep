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

//! Deterministic block-granular Automatic Prefix Cache (APC) with RAII pin leases (`PrefixLease`),
//! optional co-owned recurrent checkpoint slots (`aux_lease`), cascading LRU leaf eviction, and
//! four-group FlatKV request admission (`RequestKvHandle`).

use std::collections::{HashMap, HashSet};
use std::fmt;
use std::sync::{Arc, Mutex};

use crate::{
    BlockLease, BlockPool, CacheGroupKind, HybridBlockAliasTable, kv_hash, plan_bounded_replay,
};

/// External KV cache event emitted by [`PrefixTree`] for cross-replica prefix affinity
/// (`NEBULA_KV_HASH_SEED` / `nebula-indexd`).
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum KvCacheEvent {
    Stored { block_hashes: Vec<[u8; 8]> },
    Removed { block_hashes: Vec<[u8; 8]> },
    Cleared,
}

/// Non-blocking callback receiving [`KvCacheEvent`] batches outside the [`PrefixTree`] lock.
pub type KvCacheEventSink = Arc<dyn Fn(KvCacheEvent) + Send + Sync>;

/// Errors produced during Automatic Prefix Cache admission or block promotion.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum KvError {
    /// Block token granularity, pages-per-node, or SWA window parameter is zero or misaligned.
    InvalidGeometry(String),
    /// Physical cache group ran out of free blocks even after evicting unpinned cached leaves.
    OutOfBlocks {
        kind: CacheGroupKind,
        requested: usize,
        available: usize,
    },
    /// Token slice or lease count does not align with the tree's `block_tokens` / `pages_per_node`.
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

struct PrefixNode {
    parent: usize,
    token_key: Box<[i32]>,
    block_hash: [u8; 32],
    children: HashMap<Box<[i32]>, usize>,
    lease: Option<BlockLease>,
    aux_lease: Option<BlockLease>,
    pin_count: usize,
    last_access_tick: u64,
    evicted: bool,
}

struct PrefixTreeInner {
    block_tokens: usize,
    pages_per_node: usize,
    nodes: Vec<PrefixNode>,
    free_node_slots: Vec<usize>,
    cached_blocks: usize,
    access_clock: u64,
    alias_table: HybridBlockAliasTable,
    event_sink: Option<KvCacheEventSink>,
}

impl PrefixTreeInner {
    fn next_tick(&mut self) -> u64 {
        self.access_clock = self.access_clock.wrapping_add(1);
        self.access_clock
    }

    fn alloc_node(&mut self, node: PrefixNode) -> usize {
        if let Some(slot) = self.free_node_slots.pop() {
            self.nodes[slot] = node;
            slot
        } else {
            let slot = self.nodes.len();
            self.nodes.push(node);
            slot
        }
    }

    fn reconstruct_token_path(&self, leaf_idx: usize) -> Vec<i32> {
        let mut chain = Vec::new();
        let mut cur = leaf_idx;
        while cur != 0 {
            chain.push(cur);
            cur = self.nodes[cur].parent;
        }
        chain.reverse();
        let mut path = Vec::with_capacity(chain.len() * self.block_tokens);
        for idx in chain {
            path.extend_from_slice(&self.nodes[idx].token_key);
        }
        path
    }

    fn evict_lru_victims_inner(
        &mut self,
        target_blocks: usize,
    ) -> (Vec<PrefixVictim>, Vec<[u8; 8]>) {
        let mut victims = Vec::with_capacity(target_blocks.min(self.cached_blocks));
        let mut removed_hashes = Vec::new();
        let emit_hashes = self.event_sink.is_some();
        while victims.len() < target_blocks {
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

            let token_path = self.reconstruct_token_path(victim_idx);
            let parent_idx = self.nodes[victim_idx].parent;
            let key = self.nodes[victim_idx].token_key.clone();
            let victim_hash = self.nodes[victim_idx].block_hash;
            self.nodes[parent_idx].children.remove(&key);
            let lease = self.nodes[victim_idx].lease.take();
            let aux_lease = self.nodes[victim_idx].aux_lease.take();
            if aux_lease.is_some() {
                if let Some(&anchor_block_id) = lease.as_ref().and_then(|kv| kv.ids().first()) {
                    self.alias_table.remove_by_anchor(anchor_block_id);
                }
            }
            self.nodes[victim_idx].evicted = true;
            self.free_node_slots.push(victim_idx);
            self.cached_blocks = self.cached_blocks.saturating_sub(1);
            if let Some(lease) = lease {
                if emit_hashes {
                    removed_hashes.push(kv_hash::external_hash(&victim_hash));
                }
                victims.push(PrefixVictim {
                    token_path,
                    lease,
                    aux_lease,
                });
            } else {
                drop(aux_lease);
            }
        }
        (victims, removed_hashes)
    }
}

/// An unpinned leaf node evicted from [`PrefixTree`], keeping its HBM [`BlockLease`] alive
/// until the caller finishes any D2H offload to Host DRAM (RAII overwrite-WAR protection).
pub struct PrefixVictim {
    pub token_path: Vec<i32>,
    pub lease: BlockLease,
    pub aux_lease: Option<BlockLease>,
}

impl PrefixVictim {
    /// Canonical unified Block-ID (the victim node's anchor KV page ID).
    pub fn anchor_block_id(&self) -> Option<i32> {
        self.lease.ids().first().copied()
    }
}

/// Thread-safe block-aligned prefix cache tree over physical KV pages and optional recurrent
/// checkpoint slots.
#[derive(Clone)]
pub struct PrefixTree {
    inner: Arc<Mutex<PrefixTreeInner>>,
}

impl PrefixTree {
    pub fn new(block_tokens: u32) -> Result<Self, KvError> {
        Self::with_pages_per_node(block_tokens, 1)
    }

    pub fn with_pages_per_node(block_tokens: u32, pages_per_node: usize) -> Result<Self, KvError> {
        Self::with_pages_per_node_and_seed(
            block_tokens,
            pages_per_node,
            &kv_hash::env_kv_hash_seed(),
        )
    }

    pub fn with_pages_per_node_and_seed(
        block_tokens: u32,
        pages_per_node: usize,
        hash_seed: &str,
    ) -> Result<Self, KvError> {
        if block_tokens == 0 {
            return Err(KvError::InvalidGeometry(
                "block_tokens must be positive".to_owned(),
            ));
        }
        if pages_per_node == 0 {
            return Err(KvError::InvalidGeometry(
                "pages_per_node must be positive".to_owned(),
            ));
        }
        if (block_tokens as usize) % pages_per_node != 0 {
            return Err(KvError::InvalidGeometry(format!(
                "block_tokens ({block_tokens}) must be a multiple of pages_per_node ({pages_per_node})"
            )));
        }
        let root = PrefixNode {
            parent: 0,
            token_key: Box::new([]),
            block_hash: kv_hash::none_hash_for_seed(hash_seed),
            children: HashMap::new(),
            lease: None,
            aux_lease: None,
            pin_count: 1,
            last_access_tick: 0,
            evicted: false,
        };
        Ok(Self {
            inner: Arc::new(Mutex::new(PrefixTreeInner {
                block_tokens: block_tokens as usize,
                pages_per_node,
                nodes: vec![root],
                free_node_slots: Vec::new(),
                cached_blocks: 0,
                access_clock: 0,
                alias_table: HybridBlockAliasTable::new(),
                event_sink: None,
            })),
        })
    }

    /// Attach a non-blocking [`KvCacheEventSink`] for `BlockStored` / `BlockRemoved` notifications.
    pub fn set_event_sink(&self, sink: Option<KvCacheEventSink>) {
        let mut guard = self
            .inner
            .lock()
            .unwrap_or_else(|poisoned| poisoned.into_inner());
        guard.event_sink = sink;
    }

    pub fn block_tokens(&self) -> u32 {
        let guard = self
            .inner
            .lock()
            .unwrap_or_else(|poisoned| poisoned.into_inner());
        guard.block_tokens as u32
    }

    pub fn pages_per_node(&self) -> usize {
        let guard = self
            .inner
            .lock()
            .unwrap_or_else(|poisoned| poisoned.into_inner());
        guard.pages_per_node
    }

    /// Total cached prefix nodes currently retained in the tree.
    pub fn cached_blocks(&self) -> usize {
        let guard = self
            .inner
            .lock()
            .unwrap_or_else(|poisoned| poisoned.into_inner());
        guard.cached_blocks
    }

    /// Total physical KV pages currently retained in the tree (`cached_blocks * pages_per_node`).
    pub fn cached_kv_pages(&self) -> usize {
        let guard = self
            .inner
            .lock()
            .unwrap_or_else(|poisoned| poisoned.into_inner());
        guard.cached_blocks * guard.pages_per_node
    }

    /// Number of currently unpinned nodes eligible for LRU eviction.
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

    /// Number of physical KV pages held by currently unpinned nodes (`evictable_blocks * pages_per_node`).
    pub fn evictable_kv_pages(&self) -> usize {
        let guard = self
            .inner
            .lock()
            .unwrap_or_else(|poisoned| poisoned.into_inner());
        let unpinned_nodes = guard
            .nodes
            .iter()
            .skip(1)
            .filter(|node| !node.evicted && node.pin_count == 0)
            .count();
        unpinned_nodes * guard.pages_per_node
    }

    /// Inspect how many nodes and currently-unpinned nodes match up to `max_nodes` full
    /// `block_tokens` chunks without mutating `pin_count` or `last_access_tick`.
    pub fn peek_prefix_bounded(&self, tokens: &[i32], max_nodes: usize) -> (usize, usize) {
        let guard = self
            .inner
            .lock()
            .unwrap_or_else(|poisoned| poisoned.into_inner());
        let block_tokens = guard.block_tokens;
        let mut current = 0usize;
        let mut matched_nodes = 0usize;
        let mut unpinned_matched_nodes = 0usize;
        for chunk in tokens.chunks_exact(block_tokens).take(max_nodes) {
            let Some(&child_idx) = guard.nodes[current].children.get(chunk) else {
                break;
            };
            matched_nodes += 1;
            if guard.nodes[child_idx].pin_count == 0 {
                unpinned_matched_nodes += 1;
            }
            current = child_idx;
        }
        (matched_nodes, unpinned_matched_nodes)
    }

    /// Match as many full `block_tokens` chunks from the start of `tokens` as exist in the tree,
    /// pinning every matched ancestor via the returned RAII [`PrefixLease`].
    pub fn match_prefix(&self, tokens: &[i32]) -> PrefixLease {
        self.match_prefix_bounded(tokens, usize::MAX)
    }

    /// Match up to `max_nodes` full `block_tokens` chunks from the start of `tokens`,
    /// pinning every matched ancestor via the returned RAII [`PrefixLease`].
    pub fn match_prefix_bounded(&self, tokens: &[i32], max_nodes: usize) -> PrefixLease {
        let mut guard = self
            .inner
            .lock()
            .unwrap_or_else(|poisoned| poisoned.into_inner());
        let block_tokens = guard.block_tokens;
        let mut current = 0usize;
        let mut block_ids = Vec::new();
        let mut matched_nodes = 0usize;
        let mut leaf_aux_id = None;
        let mut leaf_anchor_block_id = None;

        for chunk in tokens.chunks_exact(block_tokens).take(max_nodes) {
            let next_idx = guard.nodes[current].children.get(chunk).copied();
            let Some(child_idx) = next_idx else {
                break;
            };
            let tick = guard.next_tick();
            let child = &mut guard.nodes[child_idx];
            child.pin_count += 1;
            child.last_access_tick = tick;
            let ids = child
                .lease
                .as_ref()
                .expect("live prefix node owns physical KV pages")
                .ids();
            block_ids.extend_from_slice(ids);
            leaf_anchor_block_id = ids.first().copied();
            leaf_aux_id = child.aux_lease.as_ref().map(|aux| aux.ids()[0]);
            matched_nodes += 1;
            current = child_idx;
        }

        let matched_tokens = (matched_nodes * block_tokens) as u32;
        PrefixLease {
            tree: Arc::clone(&self.inner),
            leaf_node: current,
            block_ids,
            matched_nodes,
            matched_tokens,
            leaf_aux_id,
            leaf_anchor_block_id,
        }
    }

    /// Insert or deduplicate full `block_tokens` chunks starting after `existing_lease`,
    /// consuming `new_node_blocks` (one [`BlockLease`] of `pages_per_node` pages per new chunk)
    /// and returning an extended [`PrefixLease`] covering the full prefix.
    ///
    /// If another request concurrently inserted any of the same nodes, the duplicate
    /// [`BlockLease`] in `new_node_blocks` drops immediately and returns its physical IDs
    /// to [`BlockPool`].
    pub fn extend_prefix(
        &self,
        existing_lease: PrefixLease,
        extension_tokens: &[i32],
        new_node_blocks: Vec<BlockLease>,
    ) -> Result<PrefixLease, KvError> {
        self.extend_prefix_with_aux(existing_lease, extension_tokens, new_node_blocks, None)
    }

    /// Insert or deduplicate full `block_tokens` chunks with optional 1:1 auxiliary recurrent
    /// checkpoint leases (`new_aux_leases`).
    pub fn extend_prefix_with_aux(
        &self,
        existing_lease: PrefixLease,
        extension_tokens: &[i32],
        new_node_blocks: Vec<BlockLease>,
        new_aux_leases: Option<Vec<BlockLease>>,
    ) -> Result<PrefixLease, KvError> {
        let mut guard = self
            .inner
            .lock()
            .unwrap_or_else(|poisoned| poisoned.into_inner());
        let block_tokens = guard.block_tokens;
        let pages_per_node = guard.pages_per_node;
        if extension_tokens.len() != new_node_blocks.len() * block_tokens {
            return Err(KvError::MisalignedPromotion(format!(
                "extension_tokens len {} != {} nodes * {}",
                extension_tokens.len(),
                new_node_blocks.len(),
                block_tokens
            )));
        }
        for lease in &new_node_blocks {
            if lease.ids().len() != pages_per_node {
                return Err(KvError::MisalignedPromotion(format!(
                    "each promoted BlockLease must own {pages_per_node} pages, got {}",
                    lease.ids().len()
                )));
            }
        }
        if let Some(ref aux_vec) = new_aux_leases {
            if aux_vec.len() != new_node_blocks.len() {
                return Err(KvError::MisalignedPromotion(format!(
                    "new_aux_leases len {} != new_node_blocks len {}",
                    aux_vec.len(),
                    new_node_blocks.len()
                )));
            }
            let mut seen_anchors = HashSet::with_capacity(new_node_blocks.len());
            let mut seen_slots = HashSet::with_capacity(aux_vec.len());
            for (block_lease, aux) in new_node_blocks.iter().zip(aux_vec.iter()) {
                if aux.ids().len() != 1 {
                    return Err(KvError::MisalignedPromotion(
                        "each promoted aux BlockLease must own exactly 1 slot".to_owned(),
                    ));
                }
                let anchor = block_lease.ids()[0];
                let slot = aux.ids()[0];
                if !seen_anchors.insert(anchor)
                    || !seen_slots.insert(slot)
                    || guard.alias_table.recurrent_slot_for_block(anchor).is_some()
                    || guard.alias_table.anchor_for_recurrent_slot(slot).is_some()
                {
                    return Err(KvError::InvalidGeometry(format!(
                        "hybrid alias collision for anchor_block_id={anchor} / aux_slot={slot}"
                    )));
                }
            }
        }

        let mut current = existing_lease.leaf_node;
        let mut block_ids = existing_lease.block_ids.clone();
        let mut matched_nodes = existing_lease.matched_nodes;
        let mut leaf_aux_id = existing_lease.leaf_aux_id;
        let mut leaf_anchor_block_id = existing_lease.leaf_anchor_block_id;
        // Consume `existing_lease` without decrementing the ancestor pins we are extending.
        let mut transferred = existing_lease;
        transferred.leaf_node = 0;
        transferred.block_ids.clear();
        drop(transferred);

        let aux_items: Vec<Option<BlockLease>> = match new_aux_leases {
            Some(vec) => vec.into_iter().map(Some).collect(),
            None => (0..new_node_blocks.len()).map(|_| None).collect(),
        };

        let mut stored_hashes = Vec::new();
        let emit_hashes = guard.event_sink.is_some();

        for ((chunk, block_lease), aux_lease) in extension_tokens
            .chunks_exact(block_tokens)
            .zip(new_node_blocks)
            .zip(aux_items)
        {
            let existing_child = guard.nodes[current].children.get(chunk).copied();
            if let Some(child_idx) = existing_child {
                // Another request already cached this exact node: drop redundant KV and aux
                // leases so their physical IDs return immediately to their `BlockPool`s.
                drop(block_lease);
                drop(aux_lease);
                let tick = guard.next_tick();
                let child = &mut guard.nodes[child_idx];
                child.pin_count += 1;
                child.last_access_tick = tick;
                let ids = child
                    .lease
                    .as_ref()
                    .expect("live prefix node owns physical KV pages")
                    .ids();
                block_ids.extend_from_slice(ids);
                leaf_anchor_block_id = ids.first().copied();
                leaf_aux_id = child.aux_lease.as_ref().map(|aux| aux.ids()[0]);
                matched_nodes += 1;
                current = child_idx;
            } else {
                block_ids.extend_from_slice(block_lease.ids());
                leaf_anchor_block_id = block_lease.ids().first().copied();
                leaf_aux_id = aux_lease.as_ref().map(|aux| aux.ids()[0]);
                if let Some(aux_slot_id) = leaf_aux_id {
                    guard
                        .alias_table
                        .insert(block_lease.ids(), aux_slot_id)
                        .expect("pre-flight validated unique hybrid alias");
                }
                let node_hash = if emit_hashes {
                    let h = kv_hash::block_hash(&guard.nodes[current].block_hash, chunk);
                    stored_hashes.push(kv_hash::external_hash(&h));
                    h
                } else {
                    [0u8; 32]
                };
                let tick = guard.next_tick();
                let key: Box<[i32]> = chunk.into();
                let node = PrefixNode {
                    parent: current,
                    token_key: key.clone(),
                    block_hash: node_hash,
                    children: HashMap::new(),
                    lease: Some(block_lease),
                    aux_lease,
                    pin_count: 1,
                    last_access_tick: tick,
                    evicted: false,
                };
                let child_idx = guard.alloc_node(node);
                guard.nodes[current].children.insert(key, child_idx);
                guard.cached_blocks += 1;
                matched_nodes += 1;
                current = child_idx;
            }
        }

        let sink = guard.event_sink.clone();
        drop(guard);
        if !stored_hashes.is_empty() {
            if let Some(sink) = sink {
                sink(KvCacheEvent::Stored {
                    block_hashes: stored_hashes,
                });
            }
        }

        let matched_tokens = (matched_nodes * block_tokens) as u32;
        Ok(PrefixLease {
            tree: Arc::clone(&self.inner),
            leaf_node: current,
            block_ids,
            matched_nodes,
            matched_tokens,
            leaf_aux_id,
            leaf_anchor_block_id,
        })
    }

    /// Resolve the GDN recurrent checkpoint slot aliased to a node's anchor KV Block-ID.
    pub fn resolve_aux_for_kv_block(&self, anchor_kv_block_id: i32) -> Option<i32> {
        let guard = self
            .inner
            .lock()
            .unwrap_or_else(|poisoned| poisoned.into_inner());
        guard
            .alias_table
            .recurrent_slot_for_block(anchor_kv_block_id)
    }

    /// Resolve the physical KV page IDs aliased to a GDN recurrent checkpoint slot.
    pub fn resolve_kv_pages_for_aux_slot(&self, aux_slot_id: i32) -> Option<Vec<i32>> {
        let guard = self
            .inner
            .lock()
            .unwrap_or_else(|poisoned| poisoned.into_inner());
        guard
            .alias_table
            .kv_pages_for_recurrent_slot(aux_slot_id)
            .map(|ids| ids.to_vec())
    }

    /// Number of active KV ↔ recurrent-state unified Block-ID aliases in this tree.
    pub fn alias_count(&self) -> usize {
        let guard = self
            .inner
            .lock()
            .unwrap_or_else(|poisoned| poisoned.into_inner());
        guard.alias_table.len()
    }

    /// Evict up to `target_blocks` unpinned leaf nodes in strict LRU order.
    pub fn evict_lru(&self, target_blocks: usize) -> usize {
        self.evict_lru_victims(target_blocks).len()
    }

    /// Evict up to `target_blocks` unpinned leaf nodes in strict LRU order, returning each
    /// victim's full token path and live HBM [`BlockLease`] so callers can safely offload
    /// KV bytes to Host DRAM before dropping the lease back to the HBM pool.
    pub fn evict_lru_victims(&self, target_blocks: usize) -> Vec<PrefixVictim> {
        let mut guard = self
            .inner
            .lock()
            .unwrap_or_else(|poisoned| poisoned.into_inner());
        let (victims, removed_hashes) = guard.evict_lru_victims_inner(target_blocks);
        let sink = guard.event_sink.clone();
        drop(guard);
        if !removed_hashes.is_empty() {
            if let Some(sink) = sink {
                sink(KvCacheEvent::Removed {
                    block_hashes: removed_hashes,
                });
            }
        }
        victims
    }
}

/// RAII handle pinning a matched chain of prefix nodes inside [`PrefixTree`].
///
/// Dropping `PrefixLease` decrements `pin_count` along the ancestor chain without panicking
/// even if a thread unwinds.
pub struct PrefixLease {
    tree: Arc<Mutex<PrefixTreeInner>>,
    leaf_node: usize,
    block_ids: Vec<i32>,
    matched_nodes: usize,
    matched_tokens: u32,
    leaf_aux_id: Option<i32>,
    leaf_anchor_block_id: Option<i32>,
}

impl PrefixLease {
    pub fn block_ids(&self) -> &[i32] {
        &self.block_ids
    }

    pub fn matched_nodes(&self) -> usize {
        self.matched_nodes
    }

    pub fn matched_tokens(&self) -> u32 {
        self.matched_tokens
    }

    pub fn leaf_aux_id(&self) -> Option<i32> {
        self.leaf_aux_id
    }

    /// Canonical unified Block-ID of the matched leaf node (`kv_page_ids[0]` of the leaf),
    /// directly resolving to `leaf_aux_id()` in [`PrefixTree::resolve_aux_for_kv_block`].
    pub fn leaf_anchor_block_id(&self) -> Option<i32> {
        self.leaf_anchor_block_id
    }
}

impl Drop for PrefixLease {
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
/// pinned APC prefix blocks, uncached `GlobalAndIndex` tail blocks, private `SwaWindow`
/// ring pages, and private `CompressorTail` state slot.
pub struct RequestKvHandle {
    prefix_tree: PrefixTree,
    global_pool: BlockPool,
    prefix_lease: Option<PrefixLease>,
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
        self.prefix_lease
            .as_ref()
            .map_or(0, PrefixLease::matched_tokens)
    }

    /// Assemble the complete ordered `GlobalAndIndex` page table (pinned prefix + tail blocks).
    pub fn global_page_table(&self) -> Vec<i32> {
        let prefix_ids = self
            .prefix_lease
            .as_ref()
            .map_or(&[][..], PrefixLease::block_ids);
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
    /// boundary, evicting unpinned prefix cache leaves on demand if needed.
    pub fn ensure_capacity_for_seq_len(&mut self, seq_len_tokens: usize) -> Result<(), KvError> {
        let needed_total = seq_len_tokens.div_ceil(self.block_tokens as usize);
        let current_total = self
            .prefix_lease
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
            .prefix_lease
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
            .prefix_lease
            .take()
            .expect("RequestKvHandle always holds a PrefixLease");
        match self.prefix_tree.extend_prefix(
            previous_lease,
            &full_sequence_tokens[token_start..token_end],
            promoted_blocks,
        ) {
            Ok(updated_lease) => {
                self.prefix_lease = Some(updated_lease);
                Ok(promotable)
            }
            Err(err) => {
                self.prefix_lease = Some(
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
    let deficit_pages = count.saturating_sub(available_before);
    let pages_per_node = prefix_tree.pages_per_node();
    let target_nodes = deficit_pages.div_ceil(pages_per_node);
    prefix_tree.evict_lru(target_nodes);
    global_pool
        .allocate(count)
        .ok_or_else(|| KvError::OutOfBlocks {
            kind: CacheGroupKind::GlobalAndIndex,
            requested: count,
            available: global_pool.available(),
        })
}

/// Unified four-group FlatKV allocator + Automatic Prefix Cache manager.
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
        let prefix_lease = self.prefix_tree.match_prefix(prompt_tokens);
        let matched_blocks = prefix_lease.block_ids().len();
        let total_blocks = prompt_tokens.len().div_ceil(self.block_tokens as usize);
        let uncached_blocks = total_blocks.saturating_sub(matched_blocks);

        // When the entire prompt is an exact multiple of `block_tokens` and matched 100% in the
        // prefix tree (`matched_tokens_raw == prompt_tokens.len()`), `swa_window >= 1` guarantees
        // `replay_tokens = matched_tokens_raw.min(swa_window) >= 1` ending at `prompt_len - 1`,
        // so next-token logits and private SWA/Compressor state are produced with `replay_mask = 1`
        // without ever overwriting any shared `GlobalAndIndex` cached page (`fresh_tokens = 0`).
        let matched_tokens_raw = prefix_lease.matched_tokens();
        let (skipped_tokens, replay_tokens) =
            plan_bounded_replay(matched_tokens_raw, self.swa_window);
        let fresh_tokens = prompt_tokens.len() as u32 - matched_tokens_raw;

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
            prefix_lease: Some(prefix_lease),
            tail_global_blocks,
            swa_lease,
            compressor_lease,
            block_tokens: self.block_tokens,
            schedule,
        })
    }
}
