"""Synthetic and real-world matrix generators for benchmarking."""

from __future__ import annotations

import numpy as np


def low_rank_plus_noise(
    n: int, d: int, rank: int, noise: float = 0.01, seed: int = 0
) -> np.ndarray:
    """Rank-r matrix plus Gaussian noise. Standard streaming-PCA testbed."""
    rng = np.random.default_rng(seed)
    U, _ = np.linalg.qr(rng.standard_normal((n, rank)).astype(np.float32))
    V, _ = np.linalg.qr(rng.standard_normal((d, rank)).astype(np.float32))
    # Heavy-tailed singular values, but bounded to avoid FP overflow on tall A.
    s = np.linspace(rank, 1.0, rank).astype(np.float32) ** 1.5
    A = (U * s) @ V.T
    A += noise * rng.standard_normal((n, d)).astype(np.float32)
    return A.astype(np.float32)


def power_law_spectrum(
    n: int, d: int, alpha: float = 1.0, seed: int = 0
) -> np.ndarray:
    """Matrix with power-law decaying singular values. Stresses the
    'slowly-decaying spectrum' regime where randomized SVD struggles."""
    rng = np.random.default_rng(seed)
    r = min(n, d)
    s = (1.0 / (1.0 + np.arange(r))) ** alpha
    U, _ = np.linalg.qr(rng.standard_normal((n, r)).astype(np.float32))
    V, _ = np.linalg.qr(rng.standard_normal((d, r)).astype(np.float32))
    return ((U * s.astype(np.float32)) @ V.T).astype(np.float32)
