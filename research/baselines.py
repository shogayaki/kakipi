"""Reference baselines for evaluating Q-FD."""

from __future__ import annotations

import numpy as np


def truncated_svd(A: np.ndarray, k: int) -> tuple[np.ndarray, np.ndarray]:
    """Exact top-k singular triplets via dense SVD. Memory: O(n*d)."""
    _, s, Vt = np.linalg.svd(A, full_matrices=False)
    return s[:k], Vt[:k]


def randomized_svd(
    A: np.ndarray, k: int, oversample: int = 10, n_iter: int = 4, seed: int = 0
) -> tuple[np.ndarray, np.ndarray]:
    """Halko-Martinsson-Tropp randomized SVD. Memory: O(n * (k + p))."""
    rng = np.random.default_rng(seed)
    n, d = A.shape
    p = k + oversample
    Omega = rng.standard_normal((d, p)).astype(np.float32)
    Y = A @ Omega
    for _ in range(n_iter):
        Q, _ = np.linalg.qr(Y)
        Z = A.T @ Q
        Q2, _ = np.linalg.qr(Z)
        Y = A @ Q2
    Q, _ = np.linalg.qr(Y)
    B = Q.T @ A
    _, s, Vt = np.linalg.svd(B, full_matrices=False)
    return s[:k], Vt[:k]
