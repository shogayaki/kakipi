"""Benchmark Q-FD vs baselines on streaming PCA / SVD tasks.

Metrics:
  - covariance_err: || A^T A - B^T B ||_2  (FD's native bound)
  - topk_sigma_err: max relative error of the top-k singular values
  - subspace_err:   sin(theta) distance between leading-k right subspaces
  - persistent_bytes: idle memory footprint of the persistent state
"""

from __future__ import annotations

import time
from dataclasses import dataclass, asdict
import json
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).parent))
from qfd import (  # noqa: E402
    FrequentDirections,
    QuantizedFD,
    MixedPrecisionFD,
    DynamicMPFD,
    DecoupledMPFD,
)
from baselines import truncated_svd, randomized_svd  # noqa: E402
from datasets import low_rank_plus_noise, power_law_spectrum  # noqa: E402


@dataclass
class Result:
    method: str
    ell: int
    covariance_err: float
    topk_sigma_err: float
    subspace_err: float
    persistent_bytes: int
    wall_seconds: float
    note: str = ""


def subspace_distance(V_true: np.ndarray, V_est: np.ndarray) -> float:
    """sin(theta_max) between the row spans of two k x d matrices."""
    # Make orthonormal bases.
    Q_true, _ = np.linalg.qr(V_true.T)
    Q_est, _ = np.linalg.qr(V_est.T)
    s = np.linalg.svd(Q_true.T @ Q_est, compute_uv=False)
    s = np.clip(s, 0.0, 1.0)
    return float(np.sqrt(max(0.0, 1.0 - s.min() ** 2)))


def evaluate_sketch(
    name: str, sketch, A: np.ndarray, k: int, sigma_true, V_true
) -> Result:
    t0 = time.perf_counter()
    sketch.append_batch(A)
    s_est, V_est = sketch.topk(k)
    wall = time.perf_counter() - t0

    # Reconstruct B^T B for the covariance error metric.
    s_full, V_full = sketch.topk(min(sketch.ell, sketch.d))
    BtB_est = V_full.T @ np.diag(s_full ** 2) @ V_full
    AtA = A.T @ A
    cov_err = float(np.linalg.norm(AtA - BtB_est, ord=2))

    sigma_err = float(np.max(np.abs(s_est - sigma_true) / np.maximum(sigma_true, 1e-12)))
    sub_err = subspace_distance(V_true, V_est)

    note = ""
    if isinstance(sketch, (DynamicMPFD, DecoupledMPFD)):
        note = f"shrinks={sketch.shrink_count} m={sketch.m_current}"
    elif hasattr(sketch, "shrink_count"):
        note = f"shrinks={sketch.shrink_count}"

    return Result(
        method=name,
        ell=getattr(sketch, "ell", 0),
        covariance_err=cov_err,
        topk_sigma_err=sigma_err,
        subspace_err=sub_err,
        persistent_bytes=sketch.persistent_bytes(),
        wall_seconds=wall,
        note=note,
    )


def run_dataset(name: str, A: np.ndarray, k: int, ells: list[int]) -> list[dict]:
    n, d = A.shape
    print(f"\n=== {name} (n={n}, d={d}, k={k}) ===")
    sigma_true, V_true = truncated_svd(A, k)
    out: list[Result] = []

    # Reference: dense full SVD (top-k)
    out.append(Result(
        method="truncated_svd",
        ell=0,
        covariance_err=0.0,
        topk_sigma_err=0.0,
        subspace_err=0.0,
        persistent_bytes=A.nbytes,
        wall_seconds=0.0,
    ))

    # Reference: randomized SVD on full matrix (memory ~ O(n*(k+p)))
    t0 = time.perf_counter()
    s_rsvd, V_rsvd = randomized_svd(A, k, oversample=10, n_iter=4)
    out.append(Result(
        method="randomized_svd",
        ell=0,
        covariance_err=float(np.linalg.norm(
            A.T @ A - V_rsvd.T @ np.diag(s_rsvd ** 2) @ V_rsvd, ord=2
        )),
        topk_sigma_err=float(np.max(np.abs(s_rsvd - sigma_true) / np.maximum(sigma_true, 1e-12))),
        subspace_err=subspace_distance(V_true, V_rsvd),
        persistent_bytes=A.nbytes + (k + 10) * d * 4,  # holds A + sketch
        wall_seconds=time.perf_counter() - t0,
    ))

    for ell in ells:
        out.append(evaluate_sketch("FD-fp32", FrequentDirections(d, ell), A, k, sigma_true, V_true))
        out.append(evaluate_sketch("QFD-int8", QuantizedFD(d, ell, "int8"), A, k, sigma_true, V_true))
        out.append(evaluate_sketch("QFD-int6-g32", QuantizedFD(d, ell, "int6", group_size=32), A, k, sigma_true, V_true))
        out.append(evaluate_sketch("QFD-nf6-g32", QuantizedFD(d, ell, "nf6", group_size=32), A, k, sigma_true, V_true))
        out.append(evaluate_sketch("QFD-int5-g32", QuantizedFD(d, ell, "int5", group_size=32), A, k, sigma_true, V_true))
        out.append(evaluate_sketch("QFD-nf5-g32", QuantizedFD(d, ell, "nf5", group_size=32), A, k, sigma_true, V_true))
        out.append(evaluate_sketch("QFD-int4-g32", QuantizedFD(d, ell, "int4", group_size=32), A, k, sigma_true, V_true))
        out.append(evaluate_sketch("QFD-nf4-g32", QuantizedFD(d, ell, "nf4", group_size=32), A, k, sigma_true, V_true))
        out.append(evaluate_sketch("MP-FD-int8/nf4", MixedPrecisionFD(d, ell, group_size=32), A, k, sigma_true, V_true))
        out.append(evaluate_sketch("Dyn-MP-FD", DynamicMPFD(d, ell, m_min=k, group_size=32), A, k, sigma_true, V_true))
        out.append(evaluate_sketch("Decoupled-MP-FD", DecoupledMPFD(d, ell, m_min=k, group_size=32), A, k, sigma_true, V_true))

    rows = [asdict(r) for r in out]
    for r in rows:
        print(
            f"  {r['method']:14s} ell={r['ell']:>4d}  "
            f"cov_err={r['covariance_err']:.3e}  "
            f"sigma_err={r['topk_sigma_err']:.3e}  "
            f"subspace_err={r['subspace_err']:.3e}  "
            f"mem={r['persistent_bytes']/1024:.1f} KB  "
            f"t={r['wall_seconds']:.2f}s  "
            f"{r.get('note', '')}"
        )
    return rows


def main():
    out: dict[str, list[dict]] = {}

    # 1) Synthetic low-rank-plus-noise with healthy SNR. Classic streaming-PCA
    # regime where FD provably works.
    A1 = low_rank_plus_noise(n=10000, d=300, rank=15, noise=0.005, seed=0)
    out["low_rank_plus_noise"] = run_dataset(
        "low_rank+noise (SNR healthy)", A1, k=10, ells=[32, 64, 128]
    )

    # 2) Power-law spectrum, slow decay alpha=0.5. Slowly-decaying spectrum
    # regime that breaks classical randomized SVD assumptions.
    A2 = power_law_spectrum(n=4000, d=300, alpha=0.5, seed=1)
    out["power_law_slow_decay"] = run_dataset(
        "power-law alpha=0.5 (slow decay)", A2, k=10, ells=[64, 128, 256]
    )

    # 3) Power-law spectrum, fast decay alpha=2.0. The "no noise floor" case
    # which we expect to be the failure mode of aggressive quantization.
    A3 = power_law_spectrum(n=4000, d=300, alpha=2.0, seed=2)
    out["power_law_fast_decay"] = run_dataset(
        "power-law alpha=2.0 (fast decay)", A3, k=10, ells=[32, 64, 128]
    )

    # 4) Tall, mobile-relevant: many rows, moderate d. Healthy SNR.
    A4 = low_rank_plus_noise(n=50000, d=200, rank=15, noise=0.005, seed=3)
    out["tall_low_rank"] = run_dataset(
        "tall low-rank (mobile-shape)", A4, k=10, ells=[32, 64, 128]
    )

    Path(__file__).parent.joinpath("results.json").write_text(json.dumps(out, indent=2))
    print("\nWrote results to research/results.json")


if __name__ == "__main__":
    main()
