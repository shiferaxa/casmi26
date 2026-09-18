"""Spectrum preprocessing and binned vectorisation."""
from __future__ import annotations

import numpy as np
import scipy.sparse as sp

MAX_MZ = 1500.0
BIN_WIDTH = 1.0  # Da; coarse on purpose so 15 ppm instrument shifts still align
N_BINS = int(MAX_MZ / BIN_WIDTH)


def clean_peaks(
    mzs: np.ndarray,
    intensities: np.ndarray,
    precursor_mz: float | None,
    rel_floor: float = 0.01,
    top_n: int = 64,
    drop_above_precursor: bool = True,
    precursor_tol: float = 0.5,
) -> tuple[np.ndarray, np.ndarray]:
    mzs = np.asarray(mzs, dtype=np.float64)
    inten = np.asarray(intensities, dtype=np.float64)
    if mzs.size == 0 or inten.max() <= 0:
        return mzs[:0], inten[:0]
    inten = inten / inten.max()
    keep = inten >= rel_floor
    if drop_above_precursor and precursor_mz is not None and np.isfinite(precursor_mz):
        keep &= mzs < (precursor_mz - precursor_tol)
    mzs, inten = mzs[keep], inten[keep]
    if mzs.size > top_n:
        idx = np.argsort(inten)[-top_n:]
        mzs, inten = mzs[idx], inten[idx]
    order = np.argsort(mzs)
    return mzs[order], inten[order]


def vectorize(mzs: np.ndarray, intensities: np.ndarray, power: float = 0.5) -> tuple[np.ndarray, np.ndarray]:
    """Return (bin_indices, values) for a sqrt-intensity, L2-normalised binned spectrum."""
    if mzs.size == 0:
        return np.zeros(0, dtype=np.int32), np.zeros(0, dtype=np.float32)
    bins = np.floor(mzs / BIN_WIDTH).astype(np.int32)
    ok = (bins >= 0) & (bins < N_BINS)
    bins, vals = bins[ok], np.power(intensities[ok], power)
    # merge duplicate bins
    uniq, inv = np.unique(bins, return_inverse=True)
    merged = np.zeros(uniq.size, dtype=np.float64)
    np.add.at(merged, inv, vals)
    norm = np.linalg.norm(merged)
    if norm > 0:
        merged /= norm
    return uniq, merged.astype(np.float32)


def build_sparse_matrix(rows: list[tuple[np.ndarray, np.ndarray]]) -> sp.csr_matrix:
    indptr = [0]
    indices, data = [], []
    for bins, vals in rows:
        indices.append(bins)
        data.append(vals)
        indptr.append(indptr[-1] + bins.size)
    indices = np.concatenate(indices) if indices else np.zeros(0, dtype=np.int32)
    data = np.concatenate(data) if data else np.zeros(0, dtype=np.float32)
    return sp.csr_matrix((data, indices, np.asarray(indptr)), shape=(len(rows), N_BINS))


def spectral_entropy_similarity(
    mz_a: np.ndarray, int_a: np.ndarray, mz_b: np.ndarray, int_b: np.ndarray, tol: float = 0.02
) -> float:
    """Li et al. 2021 entropy similarity on already-cleaned peak lists (unweighted variant)."""
    def norm(x):
        s = x.sum()
        return x / s if s > 0 else x

    a, b = norm(int_a), norm(int_b)
    # greedy match by m/z tolerance
    i = j = 0
    ab = []
    while i < mz_a.size and j < mz_b.size:
        d = mz_a[i] - mz_b[j]
        if abs(d) <= tol:
            ab.append(a[i] + b[j])
            i += 1
            j += 1
        elif d < 0:
            ab.append(a[i]); i += 1
        else:
            ab.append(b[j]); j += 1
    ab.extend(a[i:]); ab.extend(b[j:])
    ab = np.asarray(ab) / 2.0

    def H(p):
        p = p[p > 0]
        return float(-(p * np.log(p)).sum())

    return max(0.0, 1.0 - (2 * H(ab) - H(a) - H(b)) / np.log(4))
