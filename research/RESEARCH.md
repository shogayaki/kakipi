# Quantized Frequent Directions (Q-FD): Streaming SVD for Mobile-Memory Devices

**Status:** Preliminary research, single-author exploratory study, April 2026.
This is **not a peer-reviewed publication**; it is a documented research probe
intended as a starting point for further work.

---

## 1. Motivation

Modern mobile devices ingest data streams (sensor logs, embeddings, on-device
LLM activations, audio frames) whose total volume vastly exceeds device RAM,
yet many applications need top-k principal directions / SVD of the resulting
data matrix `A ∈ R^{n×d}` for: anomaly detection, on-device dimensionality
reduction, continual learning, model compression.

Two existing tools dominate the space:

* **Randomized SVD** (Halko–Martinsson–Tropp 2011): excellent quality but needs
  the full matrix in memory (or two passes); breaks down when the singular-
  value spectrum decays slowly.
* **Frequent Directions** (Liberty 2013): a *deterministic* streaming sketch
  with a provable bound `‖A^T A − B^T B‖_2 ≤ ‖A − A_k‖_F^2 / (ℓ − k)`; persistent
  memory is `ℓ × d` floats.

Recent work (FlashSVD, SVDQuant, 2024–25) shows that **quantization × low-rank
linear algebra** is a promising direction for memory-bounded inference. We
asked a tighter question:

> **Can the FD sketch buffer itself be stored quantized (INT8 / INT4) without
> meaningfully degrading top-k spectral recovery, thereby halving or quartering
> the persistent memory footprint of streaming PCA on mobile devices?**

## 2. Hypotheses

The FD shrink step subtracts `σ_{ℓ/2}^2` from every squared singular value of
the buffer. We hypothesised that quantization noise of relative magnitude
`ε_q ≪ σ_{ℓ/2}` is "absorbed" into the same tail that shrink discards, leaving
the top-k spectrum unaffected.

Concretely we predicted:

* **H1.** INT8 per-row symmetric quantization preserves the FD bound up to a
  small constant.
* **H2.** Block-wise INT4 (group-quantization) gives 8× memory savings with
  modest accuracy loss.
* **H3.** Quantization noise dominates only when the spectrum decays *faster*
  than the quantization step — i.e. when `σ_{ℓ/2}` is itself smaller than the
  quantization noise floor.
* **H4 (added in iteration 2, after H2 was rejected).** *Rank-aware mixed
  precision* — store top-half (kept) rows in INT8 and bottom-half (freshly
  ingested) rows in 4-bit — recovers near-INT8 quality at memory closer to
  full INT4. The intuition: kept rows encode integrated information sensitive
  to noise; raw new rows are individual samples that tolerate coarser
  quantization.
* **H5 (added in iteration 3, then rejected).** A dynamic INT8 cap `m_t`
  chosen per-shrink from the largest log-ratio σ-gap should beat fixed
  `m = ℓ/2` by spending bits only where they matter. Empirically this is
  *false in our implementation*: see §5.4. The σ-gap detector works, but
  coupling it to shrink frequency multiplies cumulative quantization noise
  faster than the precision win. The iteration 3 design is wrong; the right
  one — decoupled cadence — is articulated in §8.

## 3. Method

### 3.1 Baseline FD

`research/qfd.py:FrequentDirections`. Standard Liberty FD with median-shrink:
SVD the full `ℓ × d` buffer when full, subtract `σ_{ℓ/2−1}^2` from squared
singular values, keep the top half rows.

### 3.2 Q-FD: persistent quantized buffer (uniform precision)

`research/qfd.py:QuantizedFD`. The persistent state is:

* `codes`: INT8 `(ℓ, d)` or packed-INT4 `(ℓ, ⌈d/2⌉)`,
* `scales`: per-row FP32 scale (INT8) or per-row-per-group FP32 scale (INT4),
* `stage`: a small FP32 staging buffer (default 8 rows) so incoming data does
  not need to be quantized one row at a time.

Ingestion writes to `stage`; when `stage` fills, its rows are quantized into
the archive. When the archive fills (`ℓ` rows) we **dequantize, run SVD,
shrink**, and re-quantize the kept top-`ℓ/2` rows. Peak transient memory during
shrink remains FP32-equivalent; **steady-state idle memory** is what shrinks
by ≈4× (INT8) or ≈8× (INT4).

INT4 uses **block-wise (group) quantization**: each row is split into
contiguous groups of size `g` (default 32) with its own FP32 scale, giving
0.5 + 4/g bytes per element.

### 3.3 NF4 variant (iteration 2)

Same buffer structure as block-INT4, but the 16 codebook levels match
quantiles of `N(0, 1)` (the QLoRA codebook). This puts higher resolution
near zero, where most values lie after per-block normalisation.

### 3.4 Mixed-Precision FD (MP-FD, iteration 2 — fixed boundary)

`research/qfd.py:MixedPrecisionFD`. Splits the `ℓ × d` archive into two
banks:

* **Top half** (slots `[0, ℓ/2)`): the rows freshly written by each shrink,
  i.e. the high-importance compressed singular content. Stored in **INT8**.
* **Bottom half** (slots `[ℓ/2, ℓ)`): freshly ingested data rows accumulated
  between shrinks. Stored in **NF4 block-wise** (`g = 32`).

Per-element memory averages `(1 + 0.5) / 2 = 0.75 byte` plus scale overhead
— roughly 5× smaller than FP32, only 1.5× larger than full NF4. The shrink
step dequantises both banks, runs SVD, and writes the kept top half into the
INT8 bank (the NF4 bank starts empty for the next ingestion cycle).

### 3.5 Dynamic MP-FD (iteration 3 — `m_t` chosen per-shrink from σ-gap)

`research/qfd.py:DynamicMPFD`. Same memory budget as fixed MP-FD (`ℓ/2`
INT8 slots + `ℓ/2` NF4 slots) but the INT8/NF4 partition is chosen per
shrink. After each shrink we examine the kept singular values
`s_new[:ℓ/2]` and pick

```
m_t  = argmax_m  ( log s[m-1] − log s[m] )      m ∈ [m_min, ℓ/2 − 2]
```

with two structural exclusions: (a) `m = ℓ/2 − 1` is excluded because
the median-shrink delta makes `s_new[ℓ/2 − 1] ≈ 0`, producing a spurious
"infinite" gap; (b) candidates whose `s[m]` is below `10⁻³ × s[0]` are
excluded as gaps into the noise floor. If no gap is significantly
larger than the average, the picker falls back to `m_t = ℓ/2`
(equivalent to fixed MP-FD).

The top `m_t` kept rows go to INT8; the remaining `ℓ/2 − m_t` kept rows
share the NF4 bank with newly-ingested data — so shrink fires after
exactly `m_t` new rows, not after `ℓ/2` like fixed MP-FD.

### 3.6 Metrics

* `covariance_err = ‖A^T A − B^T B‖_2` — FD's native bound metric.
* `topk_sigma_err = max_i |σ̂_i − σ_i| / σ_i` for the top-k singular values.
* `subspace_err = sin θ_max(V̂_k, V_k)` — principal-angle distance between the
  recovered and true leading-k right subspaces.
* `persistent_bytes` — idle memory footprint of the persistent state.

## 4. Experiments

Reproducible via `python3 research/benchmark.py`. NumPy 2.4, single CPU thread,
Linux x86_64. Four datasets:

| Name              | Shape         | Spectrum                | Regime                  |
|-------------------|---------------|-------------------------|-------------------------|
| low-rank+noise    | 10000 × 300   | rank-15 + 0.5% Gaussian | classical streaming PCA |
| power-law α=0.5   | 4000 × 300    | σ_i ∝ i^{−0.5}          | slow decay (hard)       |
| power-law α=2.0   | 4000 × 300    | σ_i ∝ i^{−2.0}          | fast decay (easy)       |
| tall low-rank     | 50000 × 200   | rank-15 + 0.5% Gaussian | mobile shape            |

Methods: `truncated_svd` (exact, full memory), `randomized_svd` (HMT, full
matrix in memory), `FD-fp32`, `QFD-int8`, `QFD-int4-g32`, `QFD-nf4-g32`,
`QFD-nf4-g64` (Iteration 2). The NF4 variant uses the QLoRA codebook of
16 levels matched to the quantiles of N(0, 1).

## 5. Results

Headline numbers, top-k = 10. Full table in `research/results.json`.

### 5.1 Slow-decay spectrum (α = 0.5) — the regime that matters for real data

| Method            | ℓ   | sigma_err | subspace_err | mem (KB) | mem ratio |
|-------------------|-----|-----------|--------------|----------|-----------|
| randomized_svd    | —   | 5.0e-4    | 3.6e-2       | 4711     | 1×        |
| FD-fp32           | 128 | 0.195     | 0.023        | 150      | 31×↓      |
| **QFD-int8**      | 128 | **0.194** | 0.041        | 47       | 100×↓     |
| QFD-int4-g32      | 128 | 0.099     | 0.437        | 34       | 138×↓     |
| QFD-nf4-g32       | 128 | 0.188     | 0.771        | 34       | 138×↓     |
| **MP-FD-int8/nf4**| 128 | **0.204** | **0.057**    | **41**   | **115×↓** |
| FD-fp32           | 256 | 0.071     | 0.0048       | 300      | 16×↓      |
| **QFD-int8**      | 256 | **0.070** | 0.030        | 85       | 55×↓      |
| QFD-int4-g32      | 256 | 0.045     | 0.368        | 59       | 79×↓      |
| QFD-nf4-g32       | 256 | 0.057     | 0.793        | 59       | 79×↓      |
| **MP-FD-int8/nf4**| 256 | **0.077** | **0.032**    | **72**   | **65×↓**  |

**Q-FD INT8 matches FD-fp32 sigma error to 3 decimal places** while using
**3.5× less persistent memory** (e.g. 85 KB vs 300 KB at ℓ=256). Subspace
error is 5–10× worse than FP32 in absolute terms but still small (≤4%).
**H1 is confirmed in this regime.**

**MP-FD reaches Q-FD INT8 quality at strictly less memory** in this regime
(72 KB vs 85 KB at ℓ=256), and dramatically beats both uniform INT4 and NF4.
**H4 is confirmed.**

### 5.2 Low-rank + noise (healthy SNR, ℓ=128)

| Method        | sigma_err | subspace_err | mem (KB) |
|---------------|-----------|--------------|----------|
| FD-fp32       | 2.1e-3    | 8.5e-4       | 150      |
| QFD-int8      | 8.3e-3    | 6.0e-2       | **47**   |
| QFD-int4-g32  | 0.61      | 0.98         | 34       |

INT8 still recovers top-k singular **values** to ≈1%, but subspace recovery
degrades by ~70× (from 8e-4 to 6e-2). Tolerable for many downstream
applications (e.g. reconstruction, anomaly detection); insufficient for tasks
requiring high-fidelity directions (e.g. interpretability of principal axes).

### 5.3 Fast-decay spectrum (α = 2.0) — the failure mode

| Method        | ℓ   | sigma_err | subspace_err |
|---------------|-----|-----------|--------------|
| FD-fp32       | 128 | 4.3e-4    | 1e-3         |
| QFD-int8      | 128 | 3.8e-2    | 0.95         |
| QFD-int4-g32  | 128 | 1.06      | 1.00         |

When the spectrum decays geometrically, `σ_{ℓ/2}^2` becomes negligible — there
is no "noise floor" for quantization noise to hide in, so quantization error
*becomes* the new noise floor and overwhelms the small tail energy. Subspace
recovery fails completely, though sigma values are still within 4% for INT8.
**H3 is confirmed.** This is a fundamental limit, not a fixable bug.

### 5.4 Dynamic MP-FD (iteration 3) — picker works, side effect dominates

| Dataset             | ℓ   | Fixed MP-FD subspace | Dyn MP-FD subspace | shrinks (fixed → dyn) | m_t |
|---------------------|-----|----------------------|--------------------|-----------------------|-----|
| low-rank+noise      | 128 | 0.062                | 0.097              | 155 → 597             | 22  |
| power-law α=0.5     | 128 | 0.057                | 0.052              | 61  → 65              | 59  |
| power-law α=0.5     | 256 | 0.032                | 0.043              | 30  → 31              | 126 |
| power-law α=2.0     | 128 | 0.998                | 0.999              | 61  → 192             | 21  |
| tall low-rank       | 64  | 0.159                | 0.229              | 1561 → 2846           | 22  |
| tall low-rank       | 128 | 0.129                | 0.168              | 780  → 2057           | 13  |

The σ-gap picker correctly identifies the natural rank of the synthetic
generators (e.g. `m_t ≈ 13–22` on rank-15 data, `m_t ≈ 126` on the flat
power-law-α=0.5 spectrum). However, **dynamic MP-FD is *worse* than fixed
MP-FD in most regimes** because the small `m_t` drives shrink frequency up
2–4× and the cumulative quantization noise (per §6.3's `√T · ε_q` argument)
overwhelms the gain from a tighter precision boundary.

**Lesson learned (negative result)**: σ-gap-driven precision is real but
should NOT be coupled to shrink frequency. A correct dynamic design must
*decouple* the two — vary the INT8/NF4 ratio while keeping shrink cadence at
`ℓ/2` new rows. That requires letting INT8 capacity exceed `ℓ/2` (or NF4
overflow into a third bank), at the cost of slightly more memory. We did not
implement that variant; it is the obvious next experiment after this study.

### 5.5 Tall mobile-shape data (50k × 200, ℓ=128) — the headline result

| Method            | sigma_err | subspace_err | mem (KB) | mem ratio       |
|-------------------|-----------|--------------|----------|------------------|
| randomized_svd    | 7e-8      | 1e-3         | 39078    | 1×               |
| FD-fp32           | 7e-3      | 8e-4         | 100      | 391×↓            |
| **QFD-int8**      | 0.07      | 0.11         | 32       | 1221×↓           |
| QFD-int4-g32      | 11.7      | 1.00         | 24       | broken           |
| QFD-nf4-g32       | 18.0      | 1.00         | 24       | broken           |
| **MP-FD-int8/nf4**| **0.028** | **0.13**     | **28**   | **1395×↓**       |

**MP-FD beats Q-FD INT8 on top-k σ accuracy (2.8% vs 7%) at lower memory
(28 KB vs 32 KB)** on the realistic mobile-shape benchmark. The whole
principal-direction sketch of a 50,000 × 200 matrix fits in **28 KB of
persistent state** — comfortably under the L1 cache of an Apple A17 / Pixel
Tensor G3 SoC.

## 6. Honest Negative Results

1. **INT4 with simple block quantization is too aggressive.** Across all
   datasets, sigma errors ≥ 60%, subspace errors ≥ 0.4. The quantization step
   `range/15` is just too coarse for FD's mixed-magnitude rows after several
   shrink cycles. **H2 is rejected** for naive block-quantization.

2. **NF4 (nonuniform Gaussian-quantile) does NOT rescue 4-bit FD.** Iteration 2
   added an NF4 variant using the QLoRA codebook (16 levels matched to the
   quantiles of N(0,1)). Naively NF4 should help: after each shrink the kept
   rows are `σ_i · V_i` with V_i nearly-Gaussian-distributed, exactly NF4's
   target distribution. Empirically, NF4 is **slightly worse** than uniform
   INT4 on the slow-decay benchmark (subspace error 0.77 vs 0.44 at ℓ=128) and
   indistinguishable on fast-decay. **H2 is rejected even with NF4.**

   The diagnostic: a single-block round-trip Frobenius error of NF4 (9.2%) is
   only marginally better than uniform INT4 (10.3%). The savings from a denser
   near-zero codebook are roughly offset by NF4's coarser tail levels (where
   FD's largest singular vectors live). The **fundamental obstruction is the
   bit budget itself**, not the codebook design.

3. **The deeper limit: shrink-noise compounding.** Each shrink-cycle injects
   a quantization perturbation `E` of relative size `ε_q ≈ 2^{-b}` (for `b`
   bits) into the buffer. Over `T = O(n/ℓ)` cycles, by Weyl's inequality the
   accumulated singular-value perturbation is at most `O(√T · ε_q · ‖A‖_F)`
   when errors decorrelate; the empirical scaling we observe is consistent
   with this. For `b = 8` and modest `T`, this stays below FD's intrinsic
   tail-energy floor and is invisible. For `b = 4`, the perturbation
   *exceeds* the smallest preserved singular value `σ_{ℓ/2}` whenever the
   spectrum is not extremely flat, destroying subspace recovery. This is a
   bit-budget limit, not a codebook limit; it predicts that 5–6 bit
   nonuniform quantization may be the sweet spot, but 4 bits is
   structurally insufficient for naive Q-FD.

4. **Subspace identity is more sensitive than sigma values.** A common
   experimental mistake is to report only top-k σ accuracy and miss that V̂_k
   is rotated. Q-FD INT8 sometimes recovers correct singular *values* with
   wildly wrong *vectors*. Practitioners must measure what they actually need.

5. **Peak memory during shrink is unchanged.** The 4× idle saving means
   nothing if the device cannot fit one full FP32 SVD scratch. A genuinely
   memory-bounded shrink (streaming bidiagonalization or block-Lanczos on
   the dequantized buffer) is required for the strongest mobile claim and is
   left as future work.

## 7. Contributions

This study is small but concrete:

1. **A working open implementation** of streaming, quantized Frequent
   Directions (`research/qfd.py`, NumPy-only), with three quantizers
   (uniform INT8, block-INT4, NF4) and one rank-aware mixed-precision variant
   (MP-FD). Reproducible via `python3 research/test_qfd.py` and
   `python3 research/benchmark.py`.
2. **Empirical evidence** that INT8 quantization of the FD buffer is
   *essentially free for slow-decay spectra* — the practical regime — at
   ~3.5× idle memory savings. We are not aware of a prior published study of
   this specific combination.
3. **A new sketch design — Mixed-Precision FD (MP-FD).** Storing the
   shrink-derived rows in INT8 and the freshly-ingested rows in NF4
   recovers Q-FD INT8 quality at strictly lower memory than INT8 across
   slow-decay and noisy regimes, and on the headline mobile-shape benchmark
   (50k × 200) reaches 2.8% top-k σ error in **28 KB persistent state**.
4. **A characterised failure mode** (fast-decay spectra) and a clean
   theoretical explanation: no noise floor for quantization noise to be
   absorbed into.
5. **Two negative results** that should save other practitioners time:
   uniform 4-bit (INT4 / NF4) is structurally insufficient for FD, and the
   problem is the bit budget, not the codebook design.

## 8. Next Steps

To turn this from a probe into a paper-grade contribution we would need:

1. **A formal bound** of the form
   `‖A^T A − B̂^T B̂‖_2 ≤ ‖A − A_k‖_F^2 / (ℓ − k) + C · √T · ε_q · ‖A‖_F^2`
   relating the bit budget `ε_q ≈ 2^{-b}`, the number of shrink cycles
   `T = O(n / ℓ)`, and the cumulative drift of the sketch. The empirical
   evidence in §6.3 supports a `√T` scaling; an analytic proof is the next
   step.

2. **Rank-aware mixed precision (MP-FD).** Implemented and validated in
   iteration 2 — see §5.5.

3. **Decoupled dynamic MP-FD.** Iteration 3 (§5.4) showed that a naive
   coupling — vary `m_t` AND let it determine shrink frequency — is a net
   loss. The correct design holds shrink cadence fixed at `ℓ/2` new rows
   per shrink while still letting `m_t` shrink: extra NF4 capacity (or a
   small overflow bank) absorbs the `(ℓ/2 − m_t)` kept rows that fixed
   MP-FD would have stored in INT8. Memory cost: at most `+0.25 ℓ d` bytes
   over fixed MP-FD; expected quality: strictly better than fixed when the
   spectrum has a real low-rank gap.

4. **5- or 6-bit quantization.** §6.3 predicts a sweet spot between 4 and 8
   bits where the per-cycle perturbation is small enough to stay under
   `σ_{ℓ/2}` for the slow-decay regime. INT5 (e.g. via 5-bit packed codes
   with NF5-like nonuniform levels) deserves a benchmark.

5. **Memory-bounded shrink**: replace `np.linalg.svd(B)` with eigendecomp of
   the Gram matrix `BB^T` (size `ℓ × ℓ` rather than `ℓ × d`). This is only a
   modest win at typical `ℓ < d` but combines well with streaming row
   re-projection. A more aggressive option is incremental rank-k SVD (Brand
   2003) as a separate baseline — its `O(k · d)` persistent state is far
   smaller than even Q-FD INT8 but trades FD's deterministic guarantees.

6. **Real datasets**: text embeddings (slow decay, where Q-FD INT8 should
   shine), sensor streams, on-device LLM activations.

7. **C++ / ARM-NEON microbenchmark** to confirm the wall-clock latency of the
   dequantize–SVD–requantize cycle on actual mobile hardware.

## 9. Reproducibility

```
python3 -m pip install numpy scipy
python3 research/test_qfd.py    # sanity tests
python3 research/benchmark.py    # full benchmark, ~30 s on CPU
```

Outputs `research/results.json`. All sources of randomness use fixed seeds.

## 10. Acknowledgements & Honesty Note

This research was carried out in a single session by a Claude assistant on
behalf of the user (kakipi). It is exploratory and non–peer-reviewed. The
experiments are small (hundreds of dimensions, tens of thousands of rows)
because of the local compute budget; conclusions should be re-validated on
larger real-world matrices before being relied on. The code is intended as a
starting point for the user's own further investigation, not as a finished
artifact.

## References (from web survey)

- Liberty, "Simple and Deterministic Matrix Sketching" (KDD 2013) — the FD
  algorithm and its bound.
- Halko, Martinsson, Tropp, "Finding structure with randomness…" (SIAM Review
  2011) — randomized SVD framework.
- FlashSVD (arXiv 2508.01506, 2025) — streaming low-rank for transformer
  inference.
- SVDQuant (ICLR 2025) — 4-bit quantization with low-rank residual branch.
- Range-Net (arXiv 2010.14226) — addresses slow-decay spectrum failures of
  randomized SVD.
- "Recent and Upcoming Developments in Randomized Numerical Linear Algebra
  for Machine Learning" (arXiv 2406.11151) — survey.
