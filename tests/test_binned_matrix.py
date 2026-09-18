"""Vectorised library build must equal the per-row clean_peaks + vectorize path."""
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pyarrow as pa

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from casmi.library_search import REL_FLOOR, TOP_N, binned_matrix  # noqa: E402
from casmi.spectra import N_BINS, clean_peaks, vectorize  # noqa: E402


def test_vectorised_matches_row_path():
    rng = np.random.default_rng(1)
    rows = []
    for _ in range(50):
        n = int(rng.integers(0, 200))
        mz = np.sort(rng.uniform(30, 500, n))
        it = rng.uniform(0, 1, n)
        if n:
            it[rng.integers(n)] = 1.0
        rows.append(dict(mzs=mz, intensities=it, precursor_mz=float(rng.uniform(200, 480))))
    table = pa.Table.from_pandas(pd.DataFrame(rows), preserve_index=False)
    prec = np.array([r["precursor_mz"] for r in rows])
    mat = binned_matrix(table, prec)
    assert mat.shape == (50, N_BINS)
    for i, r in enumerate(rows):
        mz, it = clean_peaks(r["mzs"], r["intensities"], r["precursor_mz"], rel_floor=REL_FLOOR, top_n=TOP_N)
        bins, vals = vectorize(mz, it)
        dense = np.zeros(N_BINS, dtype=np.float32)
        dense[bins] = vals
        np.testing.assert_allclose(mat[i].toarray().ravel(), dense, atol=1e-5)
