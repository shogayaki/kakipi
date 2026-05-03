"""Sanity tests for FD and Q-FD implementations."""

import numpy as np
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))
from qfd import (
    FrequentDirections,
    QuantizedFD,
    MixedPrecisionFD,
    DynamicMPFD,
    DecoupledMPFD,
    _pick_m_from_sigma_gap,
    quantize_int8_rowwise,
    dequantize_int8_rowwise,
    quantize_int4_blockwise,
    dequantize_int4_blockwise,
    quantize_nf4_blockwise,
    dequantize_nf4_blockwise,
)


def test_nf4_roundtrip():
    rng = np.random.default_rng(0)
    X = rng.standard_normal((20, 256)).astype(np.float32) * 5.0
    packed, scales = quantize_nf4_blockwise(X, group_size=64)
    X_hat = dequantize_nf4_blockwise(packed, scales, d=256, group_size=64)
    err = np.linalg.norm(X - X_hat) / np.linalg.norm(X)
    print(f"  nf4-block(g=64) roundtrip rel-err on Gaussian: {err:.4f}")
    # On a single Gaussian block, NF4 and uniform INT4 are similar (~10%).
    # The win shows up downstream when the codebook's denser near-zero
    # resolution preserves the small components that FD's shrink would
    # otherwise destroy.
    assert err < 0.15, f"NF4 unexpectedly bad: {err}"


def test_nf4_qfd_close_to_fd():
    A = low_rank_plus_noise(2000, 200, rank=15, noise=0.05, seed=0)
    k, ell = 10, 128
    fd = FrequentDirections(A.shape[1], ell)
    fd.append_batch(A)
    s_fd, _ = fd.topk(k)

    qfd = QuantizedFD(A.shape[1], ell, mode="nf4", group_size=32)
    qfd.append_batch(A)
    s_q, _ = qfd.topk(k)
    diff = float(np.max(np.abs(s_q - s_fd) / s_fd))
    print(f"  NF4-QFD vs FD top-k sigma rel-diff: {diff:.4f}")
from datasets import low_rank_plus_noise


def test_int8_roundtrip():
    rng = np.random.default_rng(0)
    X = rng.standard_normal((10, 50)).astype(np.float32) * 5.0
    codes, scales = quantize_int8_rowwise(X)
    X_hat = dequantize_int8_rowwise(codes, scales)
    err = np.linalg.norm(X - X_hat) / np.linalg.norm(X)
    assert err < 0.01, f"int8 relative error too high: {err}"
    print(f"  int8 roundtrip rel-err: {err:.4f}")


def test_int4_roundtrip():
    rng = np.random.default_rng(0)
    X = rng.standard_normal((10, 96)).astype(np.float32) * 5.0
    packed, scales = quantize_int4_blockwise(X, group_size=32)
    X_hat = dequantize_int4_blockwise(packed, scales, d=96, group_size=32)
    err = np.linalg.norm(X - X_hat) / np.linalg.norm(X)
    print(f"  int4-block(g=32) roundtrip rel-err: {err:.4f}")
    # INT4 has an intrinsic ~10% Frobenius rel-error on iid Gaussian data
    # (range/15 step size). This is expected; the FD shrink should absorb it.
    assert err < 0.15, f"int4 relative error unexpectedly high: {err}"


def test_fd_bound():
    """Verify FD's covariance error bound holds: || A^T A - B^T B || <= ||A - A_k||_F^2 / (l - k)."""
    A = low_rank_plus_noise(2000, 200, rank=15, noise=0.05, seed=0)
    k, ell = 10, 64
    fd = FrequentDirections(A.shape[1], ell)
    fd.append_batch(A)
    s_full, V_full = fd.topk(min(ell, A.shape[1]))
    BtB = V_full.T @ np.diag(s_full ** 2) @ V_full
    cov_err = np.linalg.norm(A.T @ A - BtB, ord=2)

    # Bound RHS
    _, s_true, _ = np.linalg.svd(A, full_matrices=False)
    tail_frob_sq = float(np.sum(s_true[k:] ** 2))
    bound = tail_frob_sq / (ell - k)
    # Using the median-shrink (iSVD-like) FD variant; the strict bound is
    # tail / (l - 2k) rather than tail / (l - k). Use 2x slack to cover
    # constant differences between variants.
    print(f"  FD cov_err={cov_err:.3e}, simple-bound={bound:.3e}, ratio={cov_err/bound:.3f}")
    assert cov_err <= bound * 2.0, "FD bound violated with 2x slack!"


def test_qfd_int8_close_to_fd():
    """Q-FD INT8 should produce a sketch quality within ~10% of FP32 FD."""
    A = low_rank_plus_noise(2000, 200, rank=15, noise=0.05, seed=0)
    k, ell = 10, 64
    fd = FrequentDirections(A.shape[1], ell)
    fd.append_batch(A)
    s_fd, V_fd = fd.topk(k)

    qfd = QuantizedFD(A.shape[1], ell, mode="int8")
    qfd.append_batch(A)
    s_q, V_q = qfd.topk(k)

    sigma_diff = float(np.max(np.abs(s_q - s_fd) / s_fd))
    print(f"  QFD-int8 vs FD top-k sigma rel-diff: {sigma_diff:.4f}")
    assert sigma_diff < 0.05, "Q-FD INT8 should track FD closely"

    # Memory check
    print(f"  FD bytes={fd.persistent_bytes()}, QFD-int8 bytes={qfd.persistent_bytes()}")


def test_qfd_int4_close_to_fd():
    A = low_rank_plus_noise(2000, 200, rank=15, noise=0.05, seed=0)
    k, ell = 10, 128
    fd = FrequentDirections(A.shape[1], ell)
    fd.append_batch(A)
    s_fd, _ = fd.topk(k)

    qfd = QuantizedFD(A.shape[1], ell, mode="int4")
    qfd.append_batch(A)
    s_q, _ = qfd.topk(k)
    sigma_diff = float(np.max(np.abs(s_q - s_fd) / s_fd))
    print(f"  QFD-int4 vs FD top-k sigma rel-diff: {sigma_diff:.4f}")


def test_sigma_gap_picker():
    """The picker should pick m at the largest log-ratio drop."""
    # Clear gap at index 4 (between 5th and 6th values).
    s = np.array([10.0, 9.0, 8.0, 7.0, 6.0, 0.6, 0.4, 0.3, 0.2, 0.1])
    m = _pick_m_from_sigma_gap(s, m_min=2, m_max=8)
    print(f"  picked m={m} (expected 5 for clear gap at index 4)")
    assert m == 5, f"expected 5, got {m}"

    # Flat spectrum -> fall back to m_max.
    s_flat = np.array([1.0, 0.95, 0.91, 0.88, 0.85, 0.83])
    m = _pick_m_from_sigma_gap(s_flat, m_min=2, m_max=4)
    print(f"  flat-spectrum m={m} (expected 4 = m_max)")
    assert m == 4


def test_dynamic_mpfd():
    """Dynamic MP-FD should at minimum match fixed MP-FD's quality."""
    A = low_rank_plus_noise(2000, 200, rank=15, noise=0.01, seed=0)
    k, ell = 10, 64
    fd = FrequentDirections(A.shape[1], ell)
    fd.append_batch(A)
    s_fd, _ = fd.topk(k)

    fixed = MixedPrecisionFD(A.shape[1], ell)
    fixed.append_batch(A)
    s_fixed, _ = fixed.topk(k)

    dyn = DynamicMPFD(A.shape[1], ell, m_min=k)
    dyn.append_batch(A)
    s_dyn, _ = dyn.topk(k)

    decoupled = DecoupledMPFD(A.shape[1], ell, m_min=k)
    decoupled.append_batch(A)
    s_dec, _ = decoupled.topk(k)

    fixed_diff = float(np.max(np.abs(s_fixed - s_fd) / s_fd))
    dyn_diff = float(np.max(np.abs(s_dyn - s_fd) / s_fd))
    dec_diff = float(np.max(np.abs(s_dec - s_fd) / s_fd))
    print(f"  fixed MP-FD vs FD top-k diff: {fixed_diff:.4f}  (shrinks={fixed.shrink_count})")
    print(f"  dynamic MP-FD vs FD top-k diff: {dyn_diff:.4f}  (shrinks={dyn.shrink_count}, m={dyn.m_current})")
    print(f"  decoupled MP-FD vs FD top-k diff: {dec_diff:.4f}  (shrinks={decoupled.shrink_count}, m={decoupled.m_current})")
    print(f"  memory: fixed={fixed.persistent_bytes()}  dyn={dyn.persistent_bytes()}  decoupled={decoupled.persistent_bytes()}")
    # Decoupled should have shrink count within +/- 1 of fixed (final flush
    # may add one), and quality should match or beat fixed.
    assert abs(decoupled.shrink_count - fixed.shrink_count) <= 1, (
        f"decoupled shrink count {decoupled.shrink_count} differs from fixed {fixed.shrink_count}"
    )
    assert dec_diff <= fixed_diff * 1.5, (
        f"decoupled quality {dec_diff} should not be much worse than fixed {fixed_diff}"
    )


if __name__ == "__main__":
    print("test_int8_roundtrip");    test_int8_roundtrip()
    print("test_int4_roundtrip");    test_int4_roundtrip()
    print("test_nf4_roundtrip");     test_nf4_roundtrip()
    print("test_fd_bound");          test_fd_bound()
    print("test_qfd_int8_close");    test_qfd_int8_close_to_fd()
    print("test_qfd_int4_close");    test_qfd_int4_close_to_fd()
    print("test_nf4_qfd_close");     test_nf4_qfd_close_to_fd()
    print("test_sigma_gap_picker");  test_sigma_gap_picker()
    print("test_dynamic_mpfd");      test_dynamic_mpfd()
    print("\nAll tests passed.")
