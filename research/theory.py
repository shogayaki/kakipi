"""Iteration 7: empirical formal-bound analysis for Q-FD.

Goal: derive and validate a parameterised bound of the form

    cov_err(b-bit Q-FD)  ~  tail_bound(FD-fp32)  +  C(dataset) * eps_q^p

where eps_q is the per-block quantization rel-err of b-bit symmetric block
quantization on i.i.d. Gaussian blocks (eps_q ~ 2^-b), and (C, p) is fitted
from the (4, 5, 6, 8)-bit benchmark data per (dataset, ell).

The §6.3 sketch predicts two regimes:
  * Sub-critical (eps_q < sigma_{ell/2} / ||A||_F): the per-cycle quantization
    noise is absorbed by FD's shrink and the excess covariance error is
    linear in eps_q.
  * Super-critical (eps_q > sigma_{ell/2} / ||A||_F): the noise floor of
    the sketch is the quantization noise itself; subspace identity is
    destroyed and the error grows super-linearly.

This module reads research/results.json, fits the linear-regime model
(log-log line) per (dataset, ell), and reports the slope/intercept/R^2.
A slope close to 1 confirms the linear scaling; large slopes (>2) flag the
super-critical regime where the bound fails.
"""

from __future__ import annotations

import json
import math
from dataclasses import dataclass
from pathlib import Path

import numpy as np


# Per-block roundtrip rel-err on i.i.d. Gaussian, group_size = 32.
# Empirically measured in research/test_qfd.py; theory predicts ~ 2^-b.
EPS_Q = {
    "QFD-int4-g32": 0.1034,   # 2^-4 = 0.0625, plus tail-clipping inflation
    "QFD-int5-g32": 0.0466,   # 2^-5 = 0.0312
    "QFD-int6-g32": 0.0229,   # 2^-6 = 0.0156
    "QFD-int8":     0.0060,   # per-row INT8 (no group), 2^-8 = 0.0039
}


@dataclass
class FitResult:
    dataset: str
    ell: int
    n_points: int
    slope: float          # exponent p in cov_err - baseline ~ C * eps^p
    intercept: float      # log C
    r2: float
    base_cov: float       # FD-fp32 cov_err (the "tail" term)
    points: list          # [(method, eps_q, cov_err)]

    @property
    def C(self) -> float:
        return math.exp(self.intercept)

    def predict(self, eps_q: float) -> float:
        """Predicted cov_err at given quantization noise level."""
        return self.base_cov + self.C * (eps_q ** self.slope)


def fit_dataset(rows: list[dict]) -> list[FitResult]:
    """Fit the model per `ell` for one dataset's rows."""
    by_ell: dict[int, list[tuple[str, float, float]]] = {}
    fd_baseline: dict[int, float] = {}
    for r in rows:
        if r["method"] == "FD-fp32":
            fd_baseline[r["ell"]] = r["covariance_err"]
        if r["method"] in EPS_Q:
            by_ell.setdefault(r["ell"], []).append(
                (r["method"], EPS_Q[r["method"]], r["covariance_err"])
            )

    fits: list[FitResult] = []
    for ell, points in sorted(by_ell.items()):
        base = fd_baseline.get(ell, 0.0)
        clean = [(m, eps, ce) for (m, eps, ce) in points if ce - base > 0]
        if len(clean) < 3:
            continue
        x = np.log(np.array([eps for _, eps, _ in clean]))
        y = np.log(np.array([ce - base for _, _, ce in clean]))
        slope, intercept = np.polyfit(x, y, 1)
        ypred = slope * x + intercept
        ss_res = np.sum((y - ypred) ** 2)
        ss_tot = np.sum((y - y.mean()) ** 2)
        r2 = 1.0 - ss_res / ss_tot if ss_tot > 0 else float("nan")
        fits.append(
            FitResult(
                dataset="",  # filled by caller
                ell=ell,
                n_points=len(clean),
                slope=float(slope),
                intercept=float(intercept),
                r2=float(r2),
                base_cov=base,
                points=clean,
            )
        )
    return fits


def regime_classify(slope: float, r2: float) -> str:
    """Classify a fit by which regime it is in.

    * 'linear-clean'  — slope in [0.7, 1.3], R^2 >= 0.95: §6.3 sub-critical
    * 'linear-noisy'  — slope in [0.5, 1.5], R^2 >= 0.85
    * 'super-critical'— slope > 1.5, signals quantization-noise blowup
    * 'saturated'     — slope < 0.5, low-bit cases already saturated by other
                       errors (e.g. fast-decay regime where FD itself is
                       near-perfect and any quantization noise dominates).
    """
    if slope > 1.5:
        return "super-critical"
    if slope < 0.5:
        return "saturated"
    if 0.7 <= slope <= 1.3 and r2 >= 0.95:
        return "linear-clean"
    return "linear-noisy"


def main():
    here = Path(__file__).parent
    with open(here / "results.json") as f:
        data = json.load(f)

    print("Iteration 7: empirical fit of   cov_err - FD_baseline  ~  C * eps_q^p\n")
    print(f"{'Dataset':<28s} {'ell':>4s} {'pts':>4s} {'slope':>7s} "
          f"{'R^2':>7s} {'C':>10s} {'regime':>16s}")
    print("-" * 86)

    all_fits: list[FitResult] = []
    for ds_name, rows in data.items():
        fits = fit_dataset(rows)
        for f in fits:
            f.dataset = ds_name
            regime = regime_classify(f.slope, f.r2)
            print(
                f"{ds_name:<28s} {f.ell:>4d} {f.n_points:>4d} "
                f"{f.slope:>7.3f} {f.r2:>7.3f} {f.C:>10.2e} {regime:>16s}"
            )
            all_fits.append(f)

    # Summary statistics across well-behaved fits.
    clean = [f for f in all_fits
             if regime_classify(f.slope, f.r2) == "linear-clean"]
    if clean:
        print()
        print(f"Linear-clean (sub-critical) fits: n={len(clean)}")
        slopes = np.array([f.slope for f in clean])
        print(f"  slope  mean={slopes.mean():.3f}  median={np.median(slopes):.3f}  "
              f"std={slopes.std():.3f}")
        print("  -> Empirical scaling cov_err - tail ~ C * eps_q^p with p ~ "
              f"{np.median(slopes):.2f} (median).")
        print("     This is consistent with the perturbation argument that the")
        print("     dominant cross-term ||B'^T E + E^T B'||_2 scales linearly in")
        print("     ||E||_F = eps_q ||B'||_F.")

    # Cross-validation: leave-one-out predict each held-out bit width from the others.
    print()
    print("Leave-one-out cross-validation (predict held-out bit-width from rest):")
    print(f"{'Dataset':<28s} {'ell':>4s} {'held-out':>10s} "
          f"{'observed':>10s} {'predicted':>10s} {'rel-err':>9s}")
    print("-" * 86)
    for f in all_fits:
        if f.n_points < 4:
            continue
        for held_method, held_eps, held_cov in f.points:
            others = [(m, eps, ce) for (m, eps, ce) in f.points if m != held_method]
            x = np.log(np.array([eps for _, eps, _ in others]))
            y = np.log(np.array([max(ce - f.base_cov, 1e-30)
                                 for _, _, ce in others]))
            slope, intercept = np.polyfit(x, y, 1)
            pred_excess = math.exp(slope * math.log(held_eps) + intercept)
            pred = f.base_cov + pred_excess
            obs_excess = held_cov - f.base_cov
            if obs_excess > 0:
                rel = abs(pred_excess - obs_excess) / obs_excess
            else:
                rel = float("nan")
            if regime_classify(f.slope, f.r2) == "linear-clean":
                print(f"{f.dataset:<28s} {f.ell:>4d} {held_method[-4:]:>10s} "
                      f"{held_cov:>10.2e} {pred:>10.2e} {rel:>9.2%}")


if __name__ == "__main__":
    main()
