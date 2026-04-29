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
| `qfd.py`       | `FrequentDirections` (FP32 baseline) + `QuantizedFD` (INT8 / block-INT4) |
| `baselines.py` | Reference truncated SVD and randomized SVD                  |
| `datasets.py`  | Synthetic generators (low-rank+noise, power-law spectra)    |
| `benchmark.py` | Cross-method comparison; dumps `results.json`               |
| `test_qfd.py`  | Sanity tests                                                |
| `RESEARCH.md`  | Full research write-up                                       |
| `results.json` | Latest benchmark numbers                                    |

## TL;DR

* INT8 quantization of the FD buffer is essentially free **on slow-decay
  spectra** (the regime that matters for real big-data) — 3.5× persistent-
  memory reduction with matching top-k singular values.
* INT8 fails on fast-decay (geometric) spectra because there is no noise
  floor to absorb quantization noise.
* Naive block-INT4 is too aggressive; needs NF4-style nonuniform quantization.

This is exploratory research, not a published paper. See RESEARCH.md §10.
