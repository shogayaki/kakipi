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
| `qfd.py`       | `FrequentDirections` (FP32 baseline) + `QuantizedFD` (INT8 / INT4 / NF4) + `MixedPrecisionFD` (MP-FD: rank-aware INT8/NF4) + `DynamicMPFD` (per-shrink σ-gap m_t — iteration 3, mostly negative) |
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
  see RESEARCH.md §5.4 and §8.3 for the corrected design.
* INT8 still fails on fast-decay (geometric) spectra; characterised as a
  fundamental no-noise-floor limit.

This is exploratory research, not a published paper. See RESEARCH.md §10.
