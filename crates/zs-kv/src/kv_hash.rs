// Copyright (c) 2026 zstep Contributors
// SPDX-License-Identifier: MIT

//! `sha256_cbor` prefix block hashing compatible with Nebula (`NEBULA_KV_HASH_SEED`).
//!
//! Replicates `sha256(cbor2.dumps((parent_hash, tuple(token_ids), None), canonical=True))`
//! chained per full prefix block, with root `NONE_HASH = sha256_cbor_text(seed)`.
//! External 8-byte keys use the trailing 8 big-endian bytes (`hash[24..32]`), matching
//! `nebula_router::kv_hash::external`.

use sha2::{Digest, Sha256};

/// Default hash seed when `NEBULA_KV_HASH_SEED` is unset or empty.
pub const DEFAULT_KV_HASH_SEED: &str = "42";

/// Resolve `NEBULA_KV_HASH_SEED` from the process environment (falling back to `"42"`).
pub fn env_kv_hash_seed() -> String {
    std::env::var("NEBULA_KV_HASH_SEED")
        .ok()
        .map(|s| s.trim().to_owned())
        .filter(|s| !s.is_empty())
        .unwrap_or_else(|| DEFAULT_KV_HASH_SEED.to_owned())
}

fn put_uint(out: &mut Vec<u8>, n: u64) {
    match n {
        0..=23 => out.push(n as u8),
        24..=255 => {
            out.push(24);
            out.push(n as u8);
        }
        256..=65535 => {
            out.push(25);
            out.extend_from_slice(&(n as u16).to_be_bytes());
        }
        65536..=4_294_967_295 => {
            out.push(26);
            out.extend_from_slice(&(n as u32).to_be_bytes());
        }
        _ => {
            out.push(27);
            out.extend_from_slice(&n.to_be_bytes());
        }
    }
}

fn put_bytes(out: &mut Vec<u8>, b: &[u8]) {
    match b.len() {
        0..=23 => out.push(0x40 | b.len() as u8),
        24..=255 => {
            out.push(0x58);
            out.push(b.len() as u8);
        }
        _ => {
            out.push(0x59);
            out.extend_from_slice(&(b.len() as u16).to_be_bytes());
        }
    }
    out.extend_from_slice(b);
}

fn put_array_header(out: &mut Vec<u8>, n: usize) {
    match n {
        0..=23 => out.push(0x80 | n as u8),
        24..=255 => {
            out.push(0x98);
            out.push(n as u8);
        }
        _ => {
            out.push(0x99);
            out.extend_from_slice(&(n as u16).to_be_bytes());
        }
    }
}

/// Root `NONE_HASH = sha256(cbor_canonical_text(seed))`.
pub fn none_hash_for_seed(seed: &str) -> [u8; 32] {
    let b = seed.as_bytes();
    let mut buf = Vec::with_capacity(b.len() + 3);
    match b.len() {
        0..=23 => buf.push(0x60 | b.len() as u8),
        24..=255 => {
            buf.push(0x78);
            buf.push(b.len() as u8);
        }
        _ => {
            buf.push(0x79);
            buf.extend_from_slice(&(b.len() as u16).to_be_bytes());
        }
    }
    buf.extend_from_slice(b);
    let mut h = Sha256::new();
    h.update(&buf);
    let mut out = [0u8; 32];
    out.copy_from_slice(&h.finalize());
    out
}

/// Chain one block: `sha256(cbor_canonical([parent_32B, [tok_0..tok_{B-1}], null]))`.
pub fn block_hash(parent: &[u8; 32], tokens: &[i32]) -> [u8; 32] {
    let mut buf = Vec::with_capacity(8 + 32 + tokens.len() * 3);
    put_array_header(&mut buf, 3);
    put_bytes(&mut buf, parent);
    put_array_header(&mut buf, tokens.len());
    for &t in tokens {
        put_uint(&mut buf, t.max(0) as u64);
    }
    buf.push(0xf6); // extra_keys = null
    let mut h = Sha256::new();
    h.update(&buf);
    let mut out = [0u8; 32];
    out.copy_from_slice(&h.finalize());
    out
}

/// Canonical 8-byte big-endian external key (`hash[24..32]`).
pub fn external_hash(hash: &[u8; 32]) -> [u8; 8] {
    let mut out = [0u8; 8];
    out.copy_from_slice(&hash[24..]);
    out
}

/// Compute chained 32-byte block hashes for all full `block_size` chunks in `tokens`.
pub fn compute_block_hashes(tokens: &[i32], block_size: usize, seed: &str) -> Vec<[u8; 32]> {
    let block_size = block_size.max(1);
    let mut parent = none_hash_for_seed(seed);
    let mut out = Vec::with_capacity(tokens.len() / block_size);
    for chunk in tokens.chunks_exact(block_size) {
        let h = block_hash(&parent, chunk);
        parent = h;
        out.push(h);
    }
    out
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn golden_sha256_cbor_matches_nebula_router_and_vllm() {
        // Matches `nebula-router/src/kv_hash.rs:248` (`PYTHONHASHSEED=42`).
        let tokens = [
            37, 57610, 37611, 9934, 11, 7722, 1467, 1549, 11, 24132, 279, 4734, 1852, 15870, 17, 20,
        ];
        let hashes = compute_block_hashes(&tokens, 16, "42");
        assert_eq!(hashes.len(), 1);
        let ext = external_hash(&hashes[0]);
        assert_eq!(u64::from_be_bytes(ext), 1_015_703_622_044_383_654_u64);
    }

    #[test]
    fn seed_changes_root_and_partial_tail_is_ignored() {
        let tokens: Vec<i32> = (1..=20).collect();
        let h42 = compute_block_hashes(&tokens, 16, "42");
        let h99 = compute_block_hashes(&tokens, 16, "99");
        assert_eq!(h42.len(), 1);
        assert_ne!(h42[0], h99[0]);
    }
}
