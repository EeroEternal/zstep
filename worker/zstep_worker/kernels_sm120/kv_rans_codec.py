# Copyright (c) 2026 zstep Contributors
# SPDX-License-Identifier: MIT

"""SM120 & Host Lossless Multi-Plane KV-Cache Codec (`Latent576` / `DSv41-584`, `0 ULP`).

Splits compound KV pages (`584 B/row` `DSv41` `fp8_ds_mla` or `576 B/row` `Latent576`
or generic BF16 2-byte lanes) into homogeneous entropy planes (FP8 latent, BF16 RoPE
sign+exponent, BF16 RoPE mantissa, and block scales), encodes each plane losslessly
with 32-way warp-interleaved 12-bit static rANS on SM120 GPUs (or IEEE CRC32-guarded
zlib/deflate per-plane codec on CPU for unit tests), and verifies IEEE 802.3 CRC32 on
every decoded block (`0 ULP`).
"""

from __future__ import annotations

from dataclasses import dataclass
import os
from pathlib import Path
import struct
from typing import Any
import zlib

import torch
from torch.utils.cpp_extension import load_inline

_CSRC_DIR = Path(__file__).resolve().parent / "csrc"
_CPP_SOURCE: str = (_CSRC_DIR / "kv_rans_codec.cpp").read_text(encoding="utf-8")
_CUDA_SOURCE: str = (_CSRC_DIR / "kv_rans_codec_kernels.cu").read_text(encoding="utf-8")

_EXTENSION: Any = None
KV_PKT_MAGIC: int = 0x31564B5A  # "ZKV1" LE


def get_kv_rans_codec_extension() -> Any:
    """Build or load the SM120 GPU lossless rANS + Page-Split CUDA extension."""
    global _EXTENSION
    if _EXTENSION is None:
        os.environ.setdefault("TORCH_CUDA_ARCH_LIST", "12.0")
        _EXTENSION = load_inline(
            name="zstep_kv_rans_codec_sm120_v1_ext",
            cpp_sources=[_CPP_SOURCE],
            cuda_sources=[_CUDA_SOURCE],
            functions=[
                "kv_rans_slot_stride",
                "launch_kv_page_split",
                "launch_kv_rans32_encode",
                "launch_kv_rans32_decode",
            ],
            extra_cflags=["-O3"],
            extra_cuda_cflags=["-O3", "-gencode=arch=compute_120,code=sm_120"],
            with_cuda=True,
            verbose=False,
        )
    return _EXTENSION


def split_kv_planes_host(raw_bytes: bytes, rows: int = 128, row_stride: int = 0) -> tuple[bytes, ...]:
    """Split a compound KV page into homogeneous byte planes (`0 ULP` reversible)."""
    n = len(raw_bytes)
    if rows > 0 and row_stride in (576, 584) and n == rows * row_stride:
        t = torch.frombuffer(bytearray(raw_bytes), dtype=torch.uint8).view(rows, row_stride)
        fp8_p = bytes(t[:, :512].contiguous().numpy().tobytes())
        if row_stride == 584:
            sc_p = bytes(t[:, 512:520].contiguous().numpy().tobytes())
            rope = t[:, 520:584].view(rows, 32, 2)
            return (
                fp8_p,
                bytes(rope[:, :, 1].contiguous().numpy().tobytes()),
                bytes(rope[:, :, 0].contiguous().numpy().tobytes()),
                sc_p,
            )
        rope = t[:, 512:576].view(rows, 32, 2)
        return (
            fp8_p,
            bytes(rope[:, :, 1].contiguous().numpy().tobytes()),
            bytes(rope[:, :, 0].contiguous().numpy().tobytes()),
        )
    if (n & 1) == 0 and n > 0:
        t2 = torch.frombuffer(bytearray(raw_bytes), dtype=torch.uint8).view(-1, 2)
        return (
            bytes(t2[:, 1].contiguous().numpy().tobytes()),
            bytes(t2[:, 0].contiguous().numpy().tobytes()),
        )
    return (raw_bytes,)


def unsplit_kv_planes_host(
    planes: tuple[bytes, ...], raw_len: int, rows: int = 128, row_stride: int = 0
) -> bytes:
    """Reconstruct exact original KV page bytes from split planes (`0 ULP`)."""
    if rows > 0 and row_stride in (576, 584) and raw_len == rows * row_stride:
        out = torch.empty((rows, row_stride), dtype=torch.uint8)
        out[:, :512] = torch.frombuffer(bytearray(planes[0]), dtype=torch.uint8).view(rows, 512)
        if row_stride == 584:
            if len(planes) != 4:
                raise ValueError(f"Expected 4 planes for stride=584, got {len(planes)}")
            out[:, 512:520] = torch.frombuffer(bytearray(planes[3]), dtype=torch.uint8).view(rows, 8)
            rope = out[:, 520:584].view(rows, 32, 2)
        else:
            if len(planes) != 3:
                raise ValueError(f"Expected 3 planes for stride=576, got {len(planes)}")
            rope = out[:, 512:576].view(rows, 32, 2)
        rope[:, :, 1] = torch.frombuffer(bytearray(planes[1]), dtype=torch.uint8).view(rows, 32)
        rope[:, :, 0] = torch.frombuffer(bytearray(planes[2]), dtype=torch.uint8).view(rows, 32)
        return bytes(out.numpy().tobytes())
    if (raw_len & 1) == 0 and raw_len > 0 and len(planes) == 2:
        out2 = torch.empty((raw_len // 2, 2), dtype=torch.uint8)
        out2[:, 1] = torch.frombuffer(bytearray(planes[0]), dtype=torch.uint8)
        out2[:, 0] = torch.frombuffer(bytearray(planes[1]), dtype=torch.uint8)
        return bytes(out2.numpy().tobytes())
    if len(planes) == 1 and len(planes[0]) == raw_len:
        return planes[0]
    raise ValueError("Invalid plane layout for KV page reconstruction")


@dataclass(frozen=True)
class KvCodecStats:
    raw_bytes: int
    compressed_bytes: int
    crc32: int

    @property
    def compression_ratio(self) -> float:
        return self.raw_bytes / max(1, self.compressed_bytes)


def encode_lossless_kv_packet(
    raw_bytes: bytes, *, rows: int = 128, row_stride: int = 0
) -> tuple[bytes, KvCodecStats]:
    """Encode a KV block into a CRC32-authenticated `ZKV1` multi-plane lossless packet."""
    orig_len = len(raw_bytes)
    crc = zlib.crc32(raw_bytes) & 0xFFFFFFFF
    planes = split_kv_planes_host(raw_bytes, rows=rows, row_stride=row_stride)
    comp_planes: list[bytes] = []
    flags = 0
    for idx, p in enumerate(planes):
        c = zlib.compress(p, level=1)
        if len(c) < len(p):
            comp_planes.append(c)
            flags |= 1 << idx
        else:
            comp_planes.append(p)
    hdr = bytearray(struct.pack("<IIIHH", KV_PKT_MAGIC, orig_len, crc, len(planes), flags))
    for p, cp in zip(planes, comp_planes, strict=True):
        hdr.extend(struct.pack("<II", len(p), len(cp)))
    for cp in comp_planes:
        hdr.extend(cp)
    if len(hdr) >= orig_len:
        # Store compact 16-byte literal header + raw bytes if incompressible
        lit = struct.pack("<IIIHH", KV_PKT_MAGIC, orig_len, crc, 0, 0) + raw_bytes
        return lit, KvCodecStats(raw_bytes=orig_len, compressed_bytes=len(lit), crc32=crc)
    pkt = bytes(hdr)
    return pkt, KvCodecStats(raw_bytes=orig_len, compressed_bytes=len(pkt), crc32=crc)


def decode_lossless_kv_packet(
    packet: bytes, *, rows: int = 128, row_stride: int = 0
) -> bytes:
    """Decode and CRC32-verify a `ZKV1` multi-plane lossless KV packet (`0 ULP` or raise)."""
    if len(packet) < 16:
        raise ValueError("Truncated ZKV1 packet header")
    magic, orig_len, expected_crc, num_planes, flags = struct.unpack_from("<IIIHH", packet, 0)
    if magic != KV_PKT_MAGIC:
        raise ValueError(f"Invalid ZKV1 packet magic {magic:#x}")
    if num_planes == 0:
        raw = packet[16 : 16 + orig_len]
        if len(raw) != orig_len:
            raise ValueError("Truncated literal ZKV1 payload")
    else:
        pos = 16
        meta: list[tuple[int, int]] = []
        for _ in range(num_planes):
            if pos + 8 > len(packet):
                raise ValueError("Truncated ZKV1 plane directory")
            meta.append(struct.unpack_from("<II", packet, pos))
            pos += 8
        planes: list[bytes] = []
        for idx, (raw_plen, comp_plen) in enumerate(meta):
            if pos + comp_plen > len(packet):
                raise ValueError("Truncated ZKV1 plane payload")
            seg = packet[pos : pos + comp_plen]
            pos += comp_plen
            if (flags >> idx) & 1:
                try:
                    dec = zlib.decompress(seg)
                except zlib.error as exc:
                    raise ValueError(f"Corrupted ZKV1 compressed plane: {exc}") from exc
            else:
                dec = seg
            if len(dec) != raw_plen:
                raise ValueError("Decoded plane length mismatch")
            planes.append(dec)
        raw = unsplit_kv_planes_host(tuple(planes), orig_len, rows=rows, row_stride=row_stride)
    actual_crc = zlib.crc32(raw) & 0xFFFFFFFF
    if actual_crc != expected_crc:
        raise ValueError(
            f"KV lossless codec CRC32 mismatch: expected {expected_crc:#010x}, got {actual_crc:#010x}"
        )
    return raw
