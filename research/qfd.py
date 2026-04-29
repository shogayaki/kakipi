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
    ):
        assert ell >= 2 and ell % 2 == 0
        assert mode in ("int8", "int4")
        self.d = d
        self.ell = ell
        self.mode = mode
        self.group_size = group_size
        self.stage_rows = min(stage_rows, ell)
        self.shrink_count = 0
        self.next_archive = 0
        self.stage = np.zeros((self.stage_rows, d), dtype=np.float32)
        self.stage_fill = 0
        if mode == "int8":
            self.codes = np.zeros((ell, d), dtype=np.int8)
            self.scales = np.zeros((ell,), dtype=np.float32)
        else:
            self.n_groups = (d + group_size - 1) // group_size
            # Round packed columns to even number of nibbles by padding within group_size.
            packed_cols = (self.n_groups * group_size + 1) // 2
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
        else:
            codes, scales = quantize_int4_blockwise(slab, self.group_size)
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
        return dequantize_int4_blockwise(
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
