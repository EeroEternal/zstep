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

//! Four-group FlatKV block allocator with RAII ownership (`BlockLease`) and
//! SWA 128-token bounded replay planning.

use std::sync::{Arc, Mutex};

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
}
