"""Quantized Frequent Directions (Q-FD) for memory-constrained streaming SVD.

Implements:
  - FrequentDirections: Liberty (2013) baseline, FP32 sketch buffer.
  - QuantizedFD: persistent sketch buffer stored in INT8 or INT4 with per-row
    scaling. Buffer is dequantized for the shrink step (SVD), then recompressed.

Persistent (idle) memory savings: 4x for INT8, 8x for INT4. Peak transient
memory during shrink remains ~FP32-equivalent. Target deployment: mobile/edge
devices that must retain the sketch across many ingestion cycles between
queries, where idle footprint dominates.

The theoretical motivation for tolerating aggressive quantization is that FD's
shrink step subtracts the squared (l/2)-th singular value sigma_{l/2}^2 from
every squared singular value of the buffer. Per-row quantization noise of
relative magnitude eps contributes O(eps^2 * ||B_i||^2) Frobenius error, which
is in turn absorbed into the 'tail' that shrink discards as long as it is
small compared to sigma_{l/2}^2. We test this empirically here.
"""

from __future__ import annotations

import numpy as np


# -----------------------------------------------------------------------------
# Quantization primitives
# -----------------------------------------------------------------------------

def quantize_int8_rowwise(B: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Symmetric per-row INT8 quantization. Returns (codes int8, scales f32)."""
    abs_max = np.max(np.abs(B), axis=1)
    scales = np.where(abs_max > 0, abs_max / 127.0, 1.0).astype(np.float32)
    safe_scales = np.where(scales > 0, scales, 1.0)
    codes = np.clip(np.round(B / safe_scales[:, None]), -127, 127).astype(np.int8)
    return codes, scales


def dequantize_int8_rowwise(codes: np.ndarray, scales: np.ndarray) -> np.ndarray:
    return codes.astype(np.float32) * scales[:, None]


def quantize_int4_blockwise(
    B: np.ndarray, group_size: int = 32
) -> tuple[np.ndarray, np.ndarray]:
    """Block-wise (group-quantized) symmetric INT4. Each row is split into
    contiguous groups of `group_size` columns; each group has its own scale.
    Returns (packed uint8 of shape (l, ceil(d/2)), scales f32 of shape
    (l, n_groups))."""
    l, d = B.shape
    n_groups = (d + group_size - 1) // group_size
    pad = n_groups * group_size - d
    if pad > 0:
        Bp = np.concatenate([B, np.zeros((l, pad), dtype=B.dtype)], axis=1)
    else:
        Bp = B
    Bp = Bp.reshape(l, n_groups, group_size)
    abs_max = np.max(np.abs(Bp), axis=2)
    scales = np.where(abs_max > 0, abs_max / 7.0, 1.0).astype(np.float32)
    safe_scales = np.where(scales > 0, scales, 1.0)
    q = np.clip(np.round(Bp / safe_scales[:, :, None]), -7, 7).astype(np.int8)
    q = q.reshape(l, n_groups * group_size)
    qu = (q + 8).astype(np.uint8) & 0x0F
    high = qu[:, 0::2]
    low = qu[:, 1::2]
    packed = (high << 4) | low
    return packed, scales


def dequantize_int4_blockwise(
    packed: np.ndarray, scales: np.ndarray, d: int, group_size: int = 32
) -> np.ndarray:
    l = packed.shape[0]
    high = (packed >> 4) & 0x0F
    low = packed & 0x0F
    out = np.empty((l, high.shape[1] + low.shape[1]), dtype=np.int8)
    out[:, 0::2] = high.astype(np.int8) - 8
    out[:, 1::2] = low.astype(np.int8) - 8
    n_groups = scales.shape[1]
    padded = n_groups * group_size
    out_f = out[:, :padded].astype(np.float32).reshape(l, n_groups, group_size)
    out_f = out_f * scales[:, :, None]
    return out_f.reshape(l, padded)[:, :d]


# -----------------------------------------------------------------------------
# NF4: 4-bit Normal Float (Dettmers et al., QLoRA 2023)
# -----------------------------------------------------------------------------
# 16-level codebook matching quantiles of N(0, 1) over [-1, 1]. Information-
# theoretically optimal for normally-distributed inputs, which exactly matches
# our setting: after each FD shrink the kept rows are sigma_i * V_i where V_i
# are nearly-orthogonal random-like vectors whose entries are approximately
# Gaussian (especially for large d).

_NF4_LEVELS = np.array([
    -1.0, -0.6961928, -0.5250730, -0.39491748, -0.28444138, -0.18477343,
    -0.09105004, 0.0, 0.07958029, 0.16093020, 0.24611230, 0.33791524,
    0.44070983, 0.56261438, 0.72295684, 1.0,
], dtype=np.float32)
_NF4_BOUNDARIES = ((_NF4_LEVELS[:-1] + _NF4_LEVELS[1:]) / 2).astype(np.float32)


def quantize_nf4_blockwise(
    B: np.ndarray, group_size: int = 32, scale_mode: str = "absmax"
) -> tuple[np.ndarray, np.ndarray]:
    """NF4 block quantization. `scale_mode` is "absmax" (peak) or "sigma3"
    (3 * std, robust to outliers)."""
    l, d = B.shape
    n_groups = (d + group_size - 1) // group_size
    pad = n_groups * group_size - d
    if pad > 0:
        Bp = np.concatenate([B, np.zeros((l, pad), dtype=B.dtype)], axis=1)
    else:
        Bp = B
    Bp = Bp.reshape(l, n_groups, group_size)
    if scale_mode == "absmax":
        scales = np.max(np.abs(Bp), axis=2)
    else:
        scales = 3.0 * np.std(Bp, axis=2) + 1e-12
    scales = np.where(scales > 0, scales, 1.0).astype(np.float32)
    norm = (Bp / scales[:, :, None]).astype(np.float32)
    norm = np.clip(norm, -1.0, 1.0)
    # searchsorted gives the index of the codebook level >= norm; we want the
    # nearest level, which is determined by the midpoint boundaries.
    flat = norm.reshape(-1)
    idx = np.searchsorted(_NF4_BOUNDARIES, flat).astype(np.uint8)
    idx = idx.reshape(l, n_groups * group_size)
    # Pack two 4-bit indices into one uint8.
    high = idx[:, 0::2]
    low = idx[:, 1::2]
    packed = (high << 4) | low
    return packed, scales


def dequantize_nf4_blockwise(
    packed: np.ndarray, scales: np.ndarray, d: int, group_size: int = 32
) -> np.ndarray:
    l = packed.shape[0]
    high = (packed >> 4) & 0x0F
    low = packed & 0x0F
    idx = np.empty((l, high.shape[1] + low.shape[1]), dtype=np.uint8)
    idx[:, 0::2] = high
    idx[:, 1::2] = low
    n_groups = scales.shape[1]
    padded = n_groups * group_size
    vals = _NF4_LEVELS[idx[:, :padded]].reshape(l, n_groups, group_size)
    vals = vals * scales[:, :, None]
    return vals.reshape(l, padded)[:, :d]


# -----------------------------------------------------------------------------
# 5-bit packing: 8 INT5 codes -> 40 bits -> 5 bytes (uint64 intermediate).
# -----------------------------------------------------------------------------
# Storage layout: each row is split into groups of `group_size` values; each
# group has its own FP32 scale. group_size must be a multiple of 8 so that
# the 5-bit codes pack evenly into bytes (8 codes per 5 bytes).

def _pack_5bit(codes_uint5: np.ndarray) -> np.ndarray:
    """codes_uint5 has shape (..., 8) with values in [0, 31]. Returns shape
    (..., 5) uint8."""
    c = codes_uint5.astype(np.uint64)
    p = (
        c[..., 0]
        | (c[..., 1] << 5)
        | (c[..., 2] << 10)
        | (c[..., 3] << 15)
        | (c[..., 4] << 20)
        | (c[..., 5] << 25)
        | (c[..., 6] << 30)
        | (c[..., 7] << 35)
    )
    out = np.empty(c.shape[:-1] + (5,), dtype=np.uint8)
    out[..., 0] = p & 0xFF
    out[..., 1] = (p >> 8) & 0xFF
    out[..., 2] = (p >> 16) & 0xFF
    out[..., 3] = (p >> 24) & 0xFF
    out[..., 4] = (p >> 32) & 0xFF
    return out


def _unpack_5bit(packed: np.ndarray) -> np.ndarray:
    """Inverse of _pack_5bit. Input shape (..., 5) uint8 -> (..., 8) uint8."""
    p = (
        packed[..., 0].astype(np.uint64)
        | (packed[..., 1].astype(np.uint64) << 8)
        | (packed[..., 2].astype(np.uint64) << 16)
        | (packed[..., 3].astype(np.uint64) << 24)
        | (packed[..., 4].astype(np.uint64) << 32)
    )
    out = np.empty(packed.shape[:-1] + (8,), dtype=np.uint8)
    out[..., 0] = (p >>  0) & 0x1F
    out[..., 1] = (p >>  5) & 0x1F
    out[..., 2] = (p >> 10) & 0x1F
    out[..., 3] = (p >> 15) & 0x1F
    out[..., 4] = (p >> 20) & 0x1F
    out[..., 5] = (p >> 25) & 0x1F
    out[..., 6] = (p >> 30) & 0x1F
    out[..., 7] = (p >> 35) & 0x1F
    return out


def quantize_int5_blockwise(
    B: np.ndarray, group_size: int = 32
) -> tuple[np.ndarray, np.ndarray]:
    """Block-wise symmetric INT5 quantization. 32 levels in [-15, 15]
    (one level reserved for symmetry; effective range [-15, 15] with step
    abs_max / 15).

    Returns (packed bytes shape (l, n_groups * group_size * 5 // 8), scales
    (l, n_groups)). group_size must be a multiple of 8.
    """
    assert group_size % 8 == 0, "group_size must be multiple of 8 for 5-bit packing"
    l, d = B.shape
    n_groups = (d + group_size - 1) // group_size
    pad = n_groups * group_size - d
    if pad > 0:
        Bp = np.concatenate([B, np.zeros((l, pad), dtype=B.dtype)], axis=1)
    else:
        Bp = B
    Bp = Bp.reshape(l, n_groups, group_size)
    abs_max = np.max(np.abs(Bp), axis=2)
    scales = np.where(abs_max > 0, abs_max / 15.0, 1.0).astype(np.float32)
    safe_scales = np.where(scales > 0, scales, 1.0)
    q = np.clip(np.round(Bp / safe_scales[:, :, None]), -15, 15).astype(np.int8)
    qu = (q + 16).astype(np.uint8) & 0x1F  # shift to [1, 31]; -16 unused
    qu = qu.reshape(l, n_groups, group_size // 8, 8)
    packed = _pack_5bit(qu).reshape(l, n_groups * (group_size // 8) * 5)
    return packed, scales


def dequantize_int5_blockwise(
    packed: np.ndarray, scales: np.ndarray, d: int, group_size: int = 32
) -> np.ndarray:
    l = packed.shape[0]
    n_groups = scales.shape[1]
    sub = group_size // 8
    packed = packed.reshape(l, n_groups, sub, 5)
    qu = _unpack_5bit(packed)              # (l, n_groups, sub, 8)
    qu = qu.reshape(l, n_groups, group_size).astype(np.int8) - 16
    vals = qu.astype(np.float32) * scales[:, :, None]
    padded = n_groups * group_size
    return vals.reshape(l, padded)[:, :d]


# -----------------------------------------------------------------------------
# NF5: 32-level nonuniform codebook matching N(0, 1) quantiles.
# -----------------------------------------------------------------------------

def _build_nf5_levels() -> np.ndarray:
    """32 levels at evenly-spaced quantiles of N(0, 1), normalised to [-1, 1].
    Uses the inverse CDF (norm.ppf) directly rather than the QLoRA-paper
    asymmetric construction; for 32 levels the symmetric quantile spacing
    is essentially identical and avoids the ad-hoc offset choice."""
    from scipy.stats import norm  # local import: optional dependency
    n = 32
    qs = np.linspace(0.5 / n, 1.0 - 0.5 / n, n)
    levels = norm.ppf(qs).astype(np.float32)
    levels = levels / float(np.max(np.abs(levels)))
    levels.sort()
    return levels


_NF5_LEVELS = _build_nf5_levels()
_NF5_BOUNDARIES = ((_NF5_LEVELS[:-1] + _NF5_LEVELS[1:]) / 2).astype(np.float32)


def quantize_nf5_blockwise(
    B: np.ndarray, group_size: int = 32
) -> tuple[np.ndarray, np.ndarray]:
    assert group_size % 8 == 0
    l, d = B.shape
    n_groups = (d + group_size - 1) // group_size
    pad = n_groups * group_size - d
    if pad > 0:
        Bp = np.concatenate([B, np.zeros((l, pad), dtype=B.dtype)], axis=1)
    else:
        Bp = B
    Bp = Bp.reshape(l, n_groups, group_size)
    abs_max = np.max(np.abs(Bp), axis=2)
    scales = np.where(abs_max > 0, abs_max, 1.0).astype(np.float32)
    safe = np.where(scales > 0, scales, 1.0)
    norm = np.clip(Bp / safe[:, :, None], -1.0, 1.0)
    flat = norm.reshape(-1)
    idx = np.searchsorted(_NF5_BOUNDARIES, flat).astype(np.uint8)
    idx = idx.reshape(l, n_groups, group_size // 8, 8)
    packed = _pack_5bit(idx).reshape(l, n_groups * (group_size // 8) * 5)
    return packed, scales


def dequantize_nf5_blockwise(
    packed: np.ndarray, scales: np.ndarray, d: int, group_size: int = 32
) -> np.ndarray:
    l = packed.shape[0]
    n_groups = scales.shape[1]
    sub = group_size // 8
    packed = packed.reshape(l, n_groups, sub, 5)
    idx = _unpack_5bit(packed).reshape(l, n_groups, group_size)
    vals = _NF5_LEVELS[idx]
    vals = vals * scales[:, :, None]
    padded = n_groups * group_size
    return vals.reshape(l, padded)[:, :d]


# -----------------------------------------------------------------------------
# 6-bit packing: 4 INT6 codes -> 24 bits -> 3 bytes (uint32 intermediate).
# group_size must be a multiple of 4.
# -----------------------------------------------------------------------------

def _pack_6bit(codes_uint6: np.ndarray) -> np.ndarray:
    """codes_uint6 shape (..., 4) values in [0, 63] -> shape (..., 3) uint8."""
    c = codes_uint6.astype(np.uint32)
    p = c[..., 0] | (c[..., 1] << 6) | (c[..., 2] << 12) | (c[..., 3] << 18)
    out = np.empty(c.shape[:-1] + (3,), dtype=np.uint8)
    out[..., 0] = p & 0xFF
    out[..., 1] = (p >> 8) & 0xFF
    out[..., 2] = (p >> 16) & 0xFF
    return out


def _unpack_6bit(packed: np.ndarray) -> np.ndarray:
    p = (
        packed[..., 0].astype(np.uint32)
        | (packed[..., 1].astype(np.uint32) << 8)
        | (packed[..., 2].astype(np.uint32) << 16)
    )
    out = np.empty(packed.shape[:-1] + (4,), dtype=np.uint8)
    out[..., 0] = (p >> 0) & 0x3F
    out[..., 1] = (p >> 6) & 0x3F
    out[..., 2] = (p >> 12) & 0x3F
    out[..., 3] = (p >> 18) & 0x3F
    return out


def quantize_int6_blockwise(
    B: np.ndarray, group_size: int = 32
) -> tuple[np.ndarray, np.ndarray]:
    """Block-wise symmetric INT6: 64 levels, range [-31, 31], step abs_max/31."""
    assert group_size % 4 == 0
    l, d = B.shape
    n_groups = (d + group_size - 1) // group_size
    pad = n_groups * group_size - d
    if pad > 0:
        Bp = np.concatenate([B, np.zeros((l, pad), dtype=B.dtype)], axis=1)
    else:
        Bp = B
    Bp = Bp.reshape(l, n_groups, group_size)
    abs_max = np.max(np.abs(Bp), axis=2)
    scales = np.where(abs_max > 0, abs_max / 31.0, 1.0).astype(np.float32)
    safe = np.where(scales > 0, scales, 1.0)
    q = np.clip(np.round(Bp / safe[:, :, None]), -31, 31).astype(np.int8)
    qu = (q + 32).astype(np.uint8) & 0x3F
    qu = qu.reshape(l, n_groups, group_size // 4, 4)
    packed = _pack_6bit(qu).reshape(l, n_groups * (group_size // 4) * 3)
    return packed, scales


def dequantize_int6_blockwise(
    packed: np.ndarray, scales: np.ndarray, d: int, group_size: int = 32
) -> np.ndarray:
    l = packed.shape[0]
    n_groups = scales.shape[1]
    sub = group_size // 4
    packed = packed.reshape(l, n_groups, sub, 3)
    qu = _unpack_6bit(packed).reshape(l, n_groups, group_size).astype(np.int8) - 32
    vals = qu.astype(np.float32) * scales[:, :, None]
    padded = n_groups * group_size
    return vals.reshape(l, padded)[:, :d]


# -----------------------------------------------------------------------------
# NF6: 64-level nonuniform codebook matching N(0, 1) quantiles.
# -----------------------------------------------------------------------------

def _build_nf6_levels() -> np.ndarray:
    from scipy.stats import norm
    n = 64
    qs = np.linspace(0.5 / n, 1.0 - 0.5 / n, n)
    levels = norm.ppf(qs).astype(np.float32)
    levels = levels / float(np.max(np.abs(levels)))
    levels.sort()
    return levels


_NF6_LEVELS = _build_nf6_levels()
_NF6_BOUNDARIES = ((_NF6_LEVELS[:-1] + _NF6_LEVELS[1:]) / 2).astype(np.float32)


def quantize_nf6_blockwise(
    B: np.ndarray, group_size: int = 32
) -> tuple[np.ndarray, np.ndarray]:
    assert group_size % 4 == 0
    l, d = B.shape
    n_groups = (d + group_size - 1) // group_size
    pad = n_groups * group_size - d
    if pad > 0:
        Bp = np.concatenate([B, np.zeros((l, pad), dtype=B.dtype)], axis=1)
    else:
        Bp = B
    Bp = Bp.reshape(l, n_groups, group_size)
    abs_max = np.max(np.abs(Bp), axis=2)
    scales = np.where(abs_max > 0, abs_max, 1.0).astype(np.float32)
    safe = np.where(scales > 0, scales, 1.0)
    norm_v = np.clip(Bp / safe[:, :, None], -1.0, 1.0)
    flat = norm_v.reshape(-1)
    idx = np.searchsorted(_NF6_BOUNDARIES, flat).astype(np.uint8)
    idx = idx.reshape(l, n_groups, group_size // 4, 4)
    packed = _pack_6bit(idx).reshape(l, n_groups * (group_size // 4) * 3)
    return packed, scales


def dequantize_nf6_blockwise(
    packed: np.ndarray, scales: np.ndarray, d: int, group_size: int = 32
) -> np.ndarray:
    l = packed.shape[0]
    n_groups = scales.shape[1]
    sub = group_size // 4
    packed = packed.reshape(l, n_groups, sub, 3)
    idx = _unpack_6bit(packed).reshape(l, n_groups, group_size)
    vals = _NF6_LEVELS[idx]
    vals = vals * scales[:, :, None]
    padded = n_groups * group_size
    return vals.reshape(l, padded)[:, :d]


# -----------------------------------------------------------------------------
# Baseline Frequent Directions (Liberty 2013)
# -----------------------------------------------------------------------------

class FrequentDirections:
    """FP32 Frequent Directions sketch.

    Maintains an l x d buffer. After processing all rows, top-k singular triplets
    of B approximate those of A with error bound:

        || A^T A - B^T B ||_2 <= || A - A_k ||_F^2 / (l - k)
    """

    def __init__(self, d: int, ell: int):
        assert ell >= 2 and ell % 2 == 0, "ell must be even and >= 2"
        self.d = d
        self.ell = ell
        self.B = np.zeros((ell, d), dtype=np.float32)
        self.next_row = 0
        self.shrink_count = 0

    def append(self, row: np.ndarray) -> None:
        if self.next_row >= self.ell:
            self._shrink()
        self.B[self.next_row] = row.astype(np.float32, copy=False)
        self.next_row += 1

    def append_batch(self, rows: np.ndarray) -> None:
        for r in rows:
            self.append(r)

    def _shrink(self) -> None:
        # Full FP SVD of the l x d buffer.
        _, s, Vt = np.linalg.svd(self.B, full_matrices=False)
        half = self.ell // 2
        delta = s[half - 1] ** 2 if len(s) >= half else 0.0
        s_new = np.sqrt(np.maximum(s ** 2 - delta, 0.0))
        # Keep top half rows; pad bottom with zeros.
        self.B[:half] = (s_new[:half, None] * Vt[:half]).astype(np.float32)
        self.B[half:] = 0.0
        self.next_row = half
        self.shrink_count += 1

    def topk(self, k: int) -> tuple[np.ndarray, np.ndarray]:
        _, s, Vt = np.linalg.svd(self.B, full_matrices=False)
        return s[:k], Vt[:k]

    def persistent_bytes(self) -> int:
        return self.B.nbytes


# -----------------------------------------------------------------------------
# Quantized Frequent Directions
# -----------------------------------------------------------------------------

class QuantizedFD:
    """FD with the persistent sketch buffer stored in INT8 or INT4.

    The buffer is dequantized to FP32 only during the shrink SVD step and during
    `topk` queries. Between shrinks, only `next_row` (a small FP32 staging row)
    plus the quantized archive is held. For ell rows of dimension d the steady-
    state memory is ~ ell * d * (1 byte INT8 / 0.5 byte INT4) versus 4 bytes for
    vanilla FD.

    To make the staging zone also tiny, ingestion alternates: incoming rows are
    written into a small FP32 staging buffer of size `stage_rows`. Once the
    staging buffer is full it is quantized in-place into the archive. When the
    archive itself fills to ell rows, a shrink is triggered.
    """

    def __init__(
        self,
        d: int,
        ell: int,
        mode: str = "int8",
        stage_rows: int = 8,
        group_size: int = 32,
        nf4_scale_mode: str = "absmax",
    ):
        assert ell >= 2 and ell % 2 == 0
        assert mode in ("int8", "int4", "nf4", "int5", "nf5", "int6", "nf6")
        if mode in ("int5", "nf5"):
            assert group_size % 8 == 0, "5-bit modes require group_size multiple of 8"
        if mode in ("int6", "nf6"):
            assert group_size % 4 == 0, "6-bit modes require group_size multiple of 4"
        self.d = d
        self.ell = ell
        self.mode = mode
        self.group_size = group_size
        self.nf4_scale_mode = nf4_scale_mode
        self.stage_rows = min(stage_rows, ell)
        self.shrink_count = 0
        self.next_archive = 0
        self.stage = np.zeros((self.stage_rows, d), dtype=np.float32)
        self.stage_fill = 0
        if mode == "int8":
            self.codes = np.zeros((ell, d), dtype=np.int8)
            self.scales = np.zeros((ell,), dtype=np.float32)
        elif mode in ("int4", "nf4"):
            self.n_groups = (d + group_size - 1) // group_size
            packed_cols = (self.n_groups * group_size + 1) // 2
            self.codes = np.zeros((ell, packed_cols), dtype=np.uint8)
            self.scales = np.zeros((ell, self.n_groups), dtype=np.float32)
        elif mode in ("int5", "nf5"):
            self.n_groups = (d + group_size - 1) // group_size
            packed_cols = self.n_groups * (group_size // 8) * 5
            self.codes = np.zeros((ell, packed_cols), dtype=np.uint8)
            self.scales = np.zeros((ell, self.n_groups), dtype=np.float32)
        else:  # int6 / nf6
            self.n_groups = (d + group_size - 1) // group_size
            packed_cols = self.n_groups * (group_size // 4) * 3
            self.codes = np.zeros((ell, packed_cols), dtype=np.uint8)
            self.scales = np.zeros((ell, self.n_groups), dtype=np.float32)

    # -- ingestion ------------------------------------------------------------

    def append(self, row: np.ndarray) -> None:
        self.stage[self.stage_fill] = row.astype(np.float32, copy=False)
        self.stage_fill += 1
        if self.stage_fill == self.stage_rows:
            self._flush_stage()

    def append_batch(self, rows: np.ndarray) -> None:
        for r in rows:
            self.append(r)

    def _flush_stage(self) -> None:
        if self.stage_fill == 0:
            return
        slab = self.stage[: self.stage_fill]
        space = self.ell - self.next_archive
        if self.stage_fill > space:
            # Flush what fits, shrink, then flush the rest.
            self._archive(slab[:space])
            self._shrink()
            self._archive(slab[space:])
        else:
            self._archive(slab)
            if self.next_archive == self.ell:
                self._shrink()
        self.stage_fill = 0

    def _archive(self, slab: np.ndarray) -> None:
        n = slab.shape[0]
        if n == 0:
            return
        if self.mode == "int8":
            codes, scales = quantize_int8_rowwise(slab)
        elif self.mode == "int4":
            codes, scales = quantize_int4_blockwise(slab, self.group_size)
        elif self.mode == "nf4":
            codes, scales = quantize_nf4_blockwise(
                slab, self.group_size, self.nf4_scale_mode
            )
        elif self.mode == "int5":
            codes, scales = quantize_int5_blockwise(slab, self.group_size)
        elif self.mode == "nf5":
            codes, scales = quantize_nf5_blockwise(slab, self.group_size)
        elif self.mode == "int6":
            codes, scales = quantize_int6_blockwise(slab, self.group_size)
        else:  # nf6
            codes, scales = quantize_nf6_blockwise(slab, self.group_size)
        self.codes[self.next_archive : self.next_archive + n] = codes
        self.scales[self.next_archive : self.next_archive + n] = scales
        self.next_archive += n

    # -- shrink ---------------------------------------------------------------

    def _dequantize_archive(self) -> np.ndarray:
        n = self.next_archive
        if n == 0:
            return np.zeros((0, self.d), dtype=np.float32)
        if self.mode == "int8":
            return dequantize_int8_rowwise(self.codes[:n], self.scales[:n])
        if self.mode == "int4":
            return dequantize_int4_blockwise(
                self.codes[:n], self.scales[:n], self.d, self.group_size
            )
        if self.mode == "nf4":
            return dequantize_nf4_blockwise(
                self.codes[:n], self.scales[:n], self.d, self.group_size
            )
        if self.mode == "int5":
            return dequantize_int5_blockwise(
                self.codes[:n], self.scales[:n], self.d, self.group_size
            )
        if self.mode == "nf5":
            return dequantize_nf5_blockwise(
                self.codes[:n], self.scales[:n], self.d, self.group_size
            )
        if self.mode == "int6":
            return dequantize_int6_blockwise(
                self.codes[:n], self.scales[:n], self.d, self.group_size
            )
        return dequantize_nf6_blockwise(
            self.codes[:n], self.scales[:n], self.d, self.group_size
        )

    def _shrink(self) -> None:
        B = self._dequantize_archive()  # ell x d FP32 transient
        _, s, Vt = np.linalg.svd(B, full_matrices=False)
        half = self.ell // 2
        delta = s[half - 1] ** 2 if len(s) >= half else 0.0
        s_new = np.sqrt(np.maximum(s ** 2 - delta, 0.0))
        kept = (s_new[:half, None] * Vt[:half]).astype(np.float32)
        # Recompress kept rows into the archive; clear remainder.
        if self.mode == "int8":
            self.codes[:] = 0
        else:
            self.codes[:] = 0
        self.scales[:] = 0.0
        self.next_archive = 0
        self._archive(kept)
        self.shrink_count += 1

    # -- query ----------------------------------------------------------------

    def topk(self, k: int) -> tuple[np.ndarray, np.ndarray]:
        # Make sure staging is included.
        self._flush_stage()
        B = self._dequantize_archive()
        if B.shape[0] == 0:
            return np.zeros(k), np.zeros((k, self.d))
        _, s, Vt = np.linalg.svd(B, full_matrices=False)
        return s[:k], Vt[:k]

    def persistent_bytes(self) -> int:
        # Idle: archive (codes + scales) plus the small FP32 staging buffer.
        return self.codes.nbytes + self.scales.nbytes + self.stage.nbytes


# -----------------------------------------------------------------------------
# Mixed-Precision FD (MP-FD): top half in INT8, bottom half in NF4.
# -----------------------------------------------------------------------------

class MixedPrecisionFD:
    """Rank-aware Q-FD: rows kept after each shrink (the 'important' top-half)
    are stored in INT8; freshly-ingested rows during the current cycle (the
    bottom-half slots) are stored in NF4.

    Rationale (validated empirically below): kept rows encode integrated
    singular-component information whose subspace identity is sensitive to
    quantization noise; raw incoming rows are individual data samples that can
    tolerate coarser quantization. Per-element memory averages
    (1 + 0.5) / 2 = 0.75 bytes (plus scale overhead), about 5x lower than FP32
    and 1.5x higher than full NF4.
    """

    def __init__(self, d: int, ell: int, group_size: int = 32, stage_rows: int = 8):
        assert ell >= 2 and ell % 2 == 0
        self.d = d
        self.ell = ell
        self.half = ell // 2
        self.group_size = group_size
        self.stage_rows = min(stage_rows, ell)
        self.shrink_count = 0
        self.next_archive = 0
        self.stage = np.zeros((self.stage_rows, d), dtype=np.float32)
        self.stage_fill = 0

        # Top half: INT8 (high precision, kept rows).
        self.top_codes = np.zeros((self.half, d), dtype=np.int8)
        self.top_scales = np.zeros((self.half,), dtype=np.float32)
        # Bottom half: NF4 block (low precision, fresh ingest).
        self.n_groups = (d + group_size - 1) // group_size
        packed_cols = (self.n_groups * group_size + 1) // 2
        self.bot_codes = np.zeros((self.half, packed_cols), dtype=np.uint8)
        self.bot_scales = np.zeros((self.half, self.n_groups), dtype=np.float32)

    def append(self, row: np.ndarray) -> None:
        self.stage[self.stage_fill] = row.astype(np.float32, copy=False)
        self.stage_fill += 1
        if self.stage_fill == self.stage_rows:
            self._flush_stage()

    def append_batch(self, rows: np.ndarray) -> None:
        for r in rows:
            self.append(r)

    def _flush_stage(self) -> None:
        if self.stage_fill == 0:
            return
        slab = self.stage[: self.stage_fill]
        space = self.ell - self.next_archive
        if self.stage_fill > space:
            self._archive(slab[:space])
            self._shrink()
            self._archive(slab[space:])
        else:
            self._archive(slab)
            if self.next_archive == self.ell:
                self._shrink()
        self.stage_fill = 0

    def _archive(self, slab: np.ndarray) -> None:
        n = slab.shape[0]
        if n == 0:
            return
        # Slabs go to slots [next_archive, next_archive + n). Slots in [0, half)
        # are top (INT8); slots in [half, ell) are bottom (NF4).
        start = self.next_archive
        end = start + n
        if end <= self.half:
            codes, scales = quantize_int8_rowwise(slab)
            self.top_codes[start:end] = codes
            self.top_scales[start:end] = scales
        elif start >= self.half:
            codes, scales = quantize_nf4_blockwise(slab, self.group_size)
            self.bot_codes[start - self.half : end - self.half] = codes
            self.bot_scales[start - self.half : end - self.half] = scales
        else:
            split = self.half - start
            codes, scales = quantize_int8_rowwise(slab[:split])
            self.top_codes[start : start + split] = codes
            self.top_scales[start : start + split] = scales
            codes2, scales2 = quantize_nf4_blockwise(slab[split:], self.group_size)
            self.bot_codes[: n - split] = codes2
            self.bot_scales[: n - split] = scales2
        self.next_archive = end

    def _dequantize_full(self) -> np.ndarray:
        n_top = min(self.next_archive, self.half)
        n_bot = max(0, self.next_archive - self.half)
        out = np.zeros((self.next_archive, self.d), dtype=np.float32)
        if n_top > 0:
            out[:n_top] = dequantize_int8_rowwise(
                self.top_codes[:n_top], self.top_scales[:n_top]
            )
        if n_bot > 0:
            out[self.half : self.half + n_bot] = dequantize_nf4_blockwise(
                self.bot_codes[:n_bot], self.bot_scales[:n_bot], self.d, self.group_size
            )
        return out

    def _shrink(self) -> None:
        B = self._dequantize_full()
        _, s, Vt = np.linalg.svd(B, full_matrices=False)
        delta = s[self.half - 1] ** 2 if len(s) >= self.half else 0.0
        s_new = np.sqrt(np.maximum(s ** 2 - delta, 0.0))
        kept = (s_new[: self.half, None] * Vt[: self.half]).astype(np.float32)
        # Re-archive kept rows into the TOP (INT8) buffer; bottom is empty now.
        self.top_codes[:] = 0
        self.top_scales[:] = 0
        self.bot_codes[:] = 0
        self.bot_scales[:] = 0
        codes, scales = quantize_int8_rowwise(kept)
        self.top_codes[:] = codes
        self.top_scales[:] = scales
        self.next_archive = self.half
        self.shrink_count += 1

    def topk(self, k: int) -> tuple[np.ndarray, np.ndarray]:
        self._flush_stage()
        B = self._dequantize_full()
        if B.shape[0] == 0:
            return np.zeros(k), np.zeros((k, self.d))
        _, s, Vt = np.linalg.svd(B, full_matrices=False)
        return s[:k], Vt[:k]

    def persistent_bytes(self) -> int:
        return (
            self.top_codes.nbytes + self.top_scales.nbytes
            + self.bot_codes.nbytes + self.bot_scales.nbytes
            + self.stage.nbytes
        )


# -----------------------------------------------------------------------------
# Dynamic MP-FD: rank chosen per-shrink from the singular-value gap.
# -----------------------------------------------------------------------------

def _pick_m_from_sigma_gap(
    sigmas: np.ndarray, m_min: int, m_max: int
) -> int:
    """Choose m so that gap_at_m = log s[m-1] - log s[m] is maximised over
    m in [m_min, m_max - 1]. Falls back to m_max when the spectrum is
    essentially flat (all gaps within 1.5x of the mean gap).

    Two structural exclusions are applied to avoid FD-shrink artifacts:

      (a) m = m_max - 1 is excluded because s[m_max - 1] is often
          shrink-induced near-zero (the median-shrink delta zeroes it),
          which produces a spurious 'infinite' gap that is not a real
          rank signal.
      (b) Indices where s[m] is below 1e-3 of the largest singular value
          are excluded from the search; gaps into the noise floor are not
          informative for the natural rank.
    """
    s = np.maximum(sigmas, 1e-12)
    log_s = np.log(s)
    # Mask out indices where s[m] is below the noise floor (relative to max).
    s_max = float(s.max())
    floor = 1e-3 * s_max
    # m ranges over [m_min, m_max - 2] so we never pick the last kept row.
    upper = min(m_max - 2, len(s) - 1)
    if upper < m_min:
        return m_max
    candidates = np.arange(m_min, upper + 1)
    gap_at_m = log_s[candidates - 1] - log_s[candidates]
    valid = s[candidates] >= floor
    if not np.any(valid):
        return m_max
    gap_at_m = np.where(valid, gap_at_m, -np.inf)
    finite = gap_at_m[np.isfinite(gap_at_m)]
    if finite.size == 0:
        return m_max
    if finite.max() < 1.5 * finite.mean() + 1e-6:
        return m_max
    best = int(np.argmax(gap_at_m))
    return int(candidates[best])


class DynamicMPFD:
    """Dynamic-rank Mixed-Precision FD.

    Same memory budget as MixedPrecisionFD (ell/2 INT8 + ell/2 NF4), but the
    INT8/NF4 partition is chosen per-shrink from the singular-value gap of
    the kept top-half rows. Concretely, after each shrink:

      * m_t = pick_m(s_new[:ell/2])  -- in [m_min, ell/2]
      * top INT8 bank holds m_t kept rows (slots [0, m_t))
      * bottom NF4 bank holds the remaining ell/2 - m_t kept rows (slots
        [0, ell/2 - m_t)); new ingested rows fill slots [ell/2 - m_t, ell/2),
        so the next shrink fires after exactly m_t new rows.

    Predictions:
      * On low-effective-rank streams, m_t < ell/2 -> more frequent shrinks
        but each shrink discards quantization error before it accumulates.
      * On flat-spectrum streams, m_t = ell/2 (the gap detector falls back),
        recovering plain MP-FD behaviour.
    """

    def __init__(
        self,
        d: int,
        ell: int,
        m_min: int = 4,
        group_size: int = 32,
        stage_rows: int = 8,
    ):
        assert ell >= 4 and ell % 2 == 0
        self.d = d
        self.ell = ell
        self.half = ell // 2
        self.m_min = max(1, m_min)
        self.m_max = self.half
        self.group_size = group_size
        self.stage_rows = min(stage_rows, ell)
        self.shrink_count = 0
        self.stage = np.zeros((self.stage_rows, d), dtype=np.float32)
        self.stage_fill = 0

        # State.
        self.m_current = self.half  # initial: behave like fixed MP-FD.
        self.n_top = 0       # filled INT8 slots (== m_current after shrink, fixed during ingestion)
        self.n_bot_kept = 0  # NF4 slots holding kept-from-shrink rows: [0, n_bot_kept)
        self.n_bot = 0       # NF4 slots populated overall (kept + new): [0, n_bot)

        # Storage banks.
        self.top_codes = np.zeros((self.half, d), dtype=np.int8)
        self.top_scales = np.zeros((self.half,), dtype=np.float32)
        self.n_groups = (d + group_size - 1) // group_size
        packed_cols = (self.n_groups * group_size + 1) // 2
        self.bot_codes = np.zeros((self.half, packed_cols), dtype=np.uint8)
        self.bot_scales = np.zeros((self.half, self.n_groups), dtype=np.float32)

    # -- ingestion --

    def append(self, row: np.ndarray) -> None:
        self.stage[self.stage_fill] = row.astype(np.float32, copy=False)
        self.stage_fill += 1
        if self.stage_fill == self.stage_rows:
            self._flush_stage()

    def append_batch(self, rows: np.ndarray) -> None:
        for r in rows:
            self.append(r)

    def _flush_stage(self) -> None:
        if self.stage_fill == 0:
            return
        slab = self.stage[: self.stage_fill]
        # Repeatedly: write what fits in the NF4 bank, shrink if the bank fills.
        offset = 0
        while offset < slab.shape[0]:
            free = self.half - self.n_bot
            if free == 0:
                self._shrink()
                continue
            take = min(free, slab.shape[0] - offset)
            chunk = slab[offset : offset + take]
            codes, scales = quantize_nf4_blockwise(chunk, self.group_size)
            self.bot_codes[self.n_bot : self.n_bot + take] = codes
            self.bot_scales[self.n_bot : self.n_bot + take] = scales
            self.n_bot += take
            offset += take
            if self.n_bot == self.half:
                self._shrink()
        self.stage_fill = 0

    # -- shrink --

    def _dequantize_full(self) -> np.ndarray:
        rows = self.n_top + self.n_bot
        out = np.zeros((rows, self.d), dtype=np.float32)
        if self.n_top > 0:
            out[: self.n_top] = dequantize_int8_rowwise(
                self.top_codes[: self.n_top], self.top_scales[: self.n_top]
            )
        if self.n_bot > 0:
            out[self.n_top :] = dequantize_nf4_blockwise(
                self.bot_codes[: self.n_bot],
                self.bot_scales[: self.n_bot],
                self.d,
                self.group_size,
            )
        return out

    def _shrink(self) -> None:
        B = self._dequantize_full()
        if B.shape[0] < self.half:
            return
        _, s, Vt = np.linalg.svd(B, full_matrices=False)
        delta = s[self.half - 1] ** 2 if len(s) >= self.half else 0.0
        s_new = np.sqrt(np.maximum(s ** 2 - delta, 0.0))
        kept = (s_new[: self.half, None] * Vt[: self.half]).astype(np.float32)

        # Choose m_t from the σ-gap of the kept singular values.
        m_t = _pick_m_from_sigma_gap(s_new[: self.half], self.m_min, self.m_max)
        self.m_current = m_t

        # Reset banks and re-archive the kept rows.
        self.top_codes[:] = 0
        self.top_scales[:] = 0
        self.bot_codes[:] = 0
        self.bot_scales[:] = 0
        codes, scales = quantize_int8_rowwise(kept[:m_t])
        self.top_codes[:m_t] = codes
        self.top_scales[:m_t] = scales
        self.n_top = m_t

        rest = kept[m_t : self.half]
        if rest.shape[0] > 0:
            codes_b, scales_b = quantize_nf4_blockwise(rest, self.group_size)
            self.bot_codes[: rest.shape[0]] = codes_b
            self.bot_scales[: rest.shape[0]] = scales_b
        self.n_bot_kept = rest.shape[0]
        self.n_bot = self.n_bot_kept
        self.shrink_count += 1

    # -- query --

    def topk(self, k: int) -> tuple[np.ndarray, np.ndarray]:
        self._flush_stage()
        B = self._dequantize_full()
        if B.shape[0] == 0:
            return np.zeros(k), np.zeros((k, self.d))
        _, s, Vt = np.linalg.svd(B, full_matrices=False)
        return s[:k], Vt[:k]

    def persistent_bytes(self) -> int:
        return (
            self.top_codes.nbytes + self.top_scales.nbytes
            + self.bot_codes.nbytes + self.bot_scales.nbytes
            + self.stage.nbytes
        )
class DecoupledMPFD:
    """The corrected dynamic MP-FD design (RESEARCH.md, iteration 4).

    Like DynamicMPFD it picks m_t at each shrink from the sigma-gap of the
    kept top-half rows. Unlike DynamicMPFD, the shrink cadence is held fixed
    at ell/2 freshly-ingested rows per shrink, regardless of m_t. The
    corresponding cost is an oversized NF4 bank: c_nf4 = ell rather than
    ell/2, so it can hold (ell/2 - m_t) overflow kept-rows plus ell/2 new
    incoming rows simultaneously.

    Memory per element: 0.5 (INT8) + 0.5 (NF4) = 1.0 byte. That is ~33% more
    than fixed MP-FD (0.75 byte/elem) but identical to plain Q-FD INT8 - so
    we are paying with bytes that vanilla INT8 already spends, in exchange
    for the ability to spend them adaptively.

    Hypothesis tested: this version should match or beat fixed MP-FD on
    quality across all regimes, validating the iteration-3 lesson that the
    sigma-gap rank signal is real but must be decoupled from shrink cadence.
    """

    def __init__(
        self,
        d: int,
        ell: int,
        m_min: int = 4,
        group_size: int = 32,
        stage_rows: int = 8,
    ):
        assert ell >= 4 and ell % 2 == 0
        self.d = d
        self.ell = ell
        self.half = ell // 2
        self.m_min = max(1, m_min)
        self.m_max = self.half
        self.group_size = group_size
        self.stage_rows = min(stage_rows, ell)
        self.shrink_count = 0
        self.stage = np.zeros((self.stage_rows, d), dtype=np.float32)
        self.stage_fill = 0

        self.m_current = self.half
        self.n_top = 0
        self.n_bot_kept = 0
        self.n_bot = 0

        self.top_codes = np.zeros((self.half, d), dtype=np.int8)
        self.top_scales = np.zeros((self.half,), dtype=np.float32)
        self.n_groups = (d + group_size - 1) // group_size
        packed_cols = (self.n_groups * group_size + 1) // 2
        self.bot_codes = np.zeros((self.ell, packed_cols), dtype=np.uint8)
        self.bot_scales = np.zeros((self.ell, self.n_groups), dtype=np.float32)

    def append(self, row):
        self.stage[self.stage_fill] = row.astype(np.float32, copy=False)
        self.stage_fill += 1
        if self.stage_fill == self.stage_rows:
            self._flush_stage()

    def append_batch(self, rows):
        for r in rows:
            self.append(r)

    def _flush_stage(self):
        if self.stage_fill == 0:
            return
        slab = self.stage[: self.stage_fill]
        offset = 0
        while offset < slab.shape[0]:
            new_rows_so_far = self.n_bot - self.n_bot_kept
            free = self.half - new_rows_so_far
            if free == 0:
                self._shrink()
                continue
            take = min(free, slab.shape[0] - offset)
            chunk = slab[offset : offset + take]
            codes, scales = quantize_nf4_blockwise(chunk, self.group_size)
            self.bot_codes[self.n_bot : self.n_bot + take] = codes
            self.bot_scales[self.n_bot : self.n_bot + take] = scales
            self.n_bot += take
            offset += take
            if (self.n_bot - self.n_bot_kept) == self.half:
                self._shrink()
        self.stage_fill = 0

    def _dequantize_full(self):
        rows = self.n_top + self.n_bot
        out = np.zeros((rows, self.d), dtype=np.float32)
        if self.n_top > 0:
            out[: self.n_top] = dequantize_int8_rowwise(
                self.top_codes[: self.n_top], self.top_scales[: self.n_top]
            )
        if self.n_bot > 0:
            out[self.n_top :] = dequantize_nf4_blockwise(
                self.bot_codes[: self.n_bot],
                self.bot_scales[: self.n_bot],
                self.d,
                self.group_size,
            )
        return out

    def _shrink(self):
        B = self._dequantize_full()
        if B.shape[0] < self.half:
            return
        _, s, Vt = np.linalg.svd(B, full_matrices=False)
        delta = s[self.half - 1] ** 2 if len(s) >= self.half else 0.0
        s_new = np.sqrt(np.maximum(s ** 2 - delta, 0.0))
        kept = (s_new[: self.half, None] * Vt[: self.half]).astype(np.float32)

        m_t = _pick_m_from_sigma_gap(s_new[: self.half], self.m_min, self.m_max)
        self.m_current = m_t

        self.top_codes[:] = 0
        self.top_scales[:] = 0
        self.bot_codes[:] = 0
        self.bot_scales[:] = 0
        codes, scales = quantize_int8_rowwise(kept[:m_t])
        self.top_codes[:m_t] = codes
        self.top_scales[:m_t] = scales
        self.n_top = m_t

        rest = kept[m_t : self.half]
        if rest.shape[0] > 0:
            codes_b, scales_b = quantize_nf4_blockwise(rest, self.group_size)
            self.bot_codes[: rest.shape[0]] = codes_b
            self.bot_scales[: rest.shape[0]] = scales_b
        self.n_bot_kept = rest.shape[0]
        self.n_bot = self.n_bot_kept
        self.shrink_count += 1

    def topk(self, k):
        self._flush_stage()
        B = self._dequantize_full()
        if B.shape[0] == 0:
            return np.zeros(k), np.zeros((k, self.d))
        _, s, Vt = np.linalg.svd(B, full_matrices=False)
        return s[:k], Vt[:k]

    def persistent_bytes(self):
        return (
            self.top_codes.nbytes + self.top_scales.nbytes
            + self.bot_codes.nbytes + self.bot_scales.nbytes
            + self.stage.nbytes
        )
