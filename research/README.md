# Q-FD: Quantized Frequent Directions for Mobile-Memory Streaming SVD

Research probe (April 2026) on streaming top-k SVD for memory-constrained
devices via quantizing the Frequent Directions (Liberty 2013) sketch buffer.

See [RESEARCH.md](RESEARCH.md) for the full write-up, including hypotheses,
experiments, results, and an honest discussion of failure modes.

## Quick start

```bash
pip install numpy scipy
python3 research/test_qfd.py     # sanity tests (~5 s)
python3 research/benchmark.py    # full benchmark (~30 s)
```

## Files

| File           | Purpose                                                     |
|----------------|-------------------------------------------------------------|
| `qfd.py`       | `FrequentDirections` (FP32 baseline) + `QuantizedFD` (INT8 / INT4 / NF4 / INT5 / NF5 / INT6 / NF6) + `MixedPrecisionFD` (MP-FD: rank-aware INT8/NF4) + `DynamicMPFD` (iteration 3, broken) + `DecoupledMPFD` (iteration 4, σ-gap with fixed cadence — confirmed positive) |
| `theory.py`    | Iteration 7: empirical fit of `cov_err ≤ tail + C·ε_q^p` from `results.json`; classifies each (dataset, ℓ) cell into linear-clean / saturated / super-critical; leave-one-out cross-validation. Run: `python3 research/theory.py`. |
| `scaling_law.png` | Iteration 7 figure: log-log plots of excess cov_err vs ε_q across the four datasets, showing the three regimes visually. |
| `baselines.py` | Reference truncated SVD and randomized SVD                  |
| `datasets.py`  | Synthetic generators (low-rank+noise, power-law spectra)    |
| `benchmark.py` | Cross-method comparison; dumps `results.json`               |
| `test_qfd.py`  | Sanity tests                                                |
| `RESEARCH.md`  | Full research write-up                                       |
| `results.json` | Latest benchmark numbers                                    |

## TL;DR

* INT8 quantization of the FD buffer is essentially free **on slow-decay
  spectra** (the regime that matters for real big-data) — ~3.5× persistent-
  memory reduction with matching top-k singular values.
* **Naive 4-bit quantization (INT4 or NF4) is structurally insufficient** for
  FD — the problem is the bit budget, not the codebook.
* **MP-FD** (rank-aware mixed precision: top rows INT8, bottom rows NF4)
  matches Q-FD INT8 quality at lower memory and was the key positive result
  of iteration 2. Headline: top-k SVD of a 50,000 × 200 matrix in **28 KB**.
* **Dynamic MP-FD** (iteration 3): the σ-gap rank picker correctly identifies
  the natural rank, but coupling `m_t` to shrink frequency *increases*
  cumulative quantization noise. Net negative result with a clean lesson —
  see RESEARCH.md §5.4.
* **Decoupled-MP-FD** (iteration 4, *positive*): the corrected design with
  the σ-gap picker but a fixed shrink cadence. Beats both Q-FD INT8 AND
  fixed MP-FD on subspace recovery in the mobile-shape benchmark, at the
  same memory class as Q-FD INT8.
* **INT5 / NF5** (iteration 5, *partial-positive*): 5-bit beats INT4 by 2×
  but still 4-10× worse than INT8.
* **INT6 / NF6** (iteration 6, *positive — the §6.3 sweet spot*): 6-bit
  uniform reaches **2.7× of INT8 subspace quality at 86% of its memory**
  on slow-decay spectra, fully confirming §6.3's `ε_q ∝ 2⁻ᵇ` analysis.
  NF6 again *underperforms* uniform INT6 — the nonuniform-codebook
  miscalibration is now a two-data-point negative result.
* **Empirically validated formal bound** (iteration 7, *positive*): with
  4 bit widths (4/5/6/8) on each of 12 (dataset, ℓ) cells, fit
  `cov_err ≤ tail + C(A) · ε_q^p` and obtain `p ≈ 0.83 ± 0.07` with
  R² > 0.99 in the linear-clean regime. Leave-one-out predicts held-out
  bit widths to within 0.3-10%. Three regimes (linear / saturated /
  super-critical) match §6.3's analytical prediction. See
  `research/scaling_law.png`.
* INT8 still fails on fast-decay (geometric) spectra; characterised as a
  fundamental no-noise-floor limit.

This is exploratory research, not a published paper. See RESEARCH.md §10.
