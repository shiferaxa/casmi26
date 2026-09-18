"""Library search baseline: precursor-window filtered cosine on binned spectra,
aggregated to structures, fused across the spectra of one molecule_id.

Library = every training spectrum, labelled by InChIKey14 (the provided column
when present, else the metric's tautomer-canonical key computed from SMILES).
This directly covers competition Class 1 (library knowns) and partly Class 2
(known structure, spectrum from a different instrument or energy).

Building the library is fully vectorised over pyarrow list offsets so the
2.5M-row training set takes seconds, not a Python loop over rows.
"""
from __future__ import annotations

from collections import defaultdict
from dataclasses import dataclass

import numpy as np
import pandas as pd
import pyarrow as pa
import pyarrow.compute as pc
import scipy.sparse as sp
from tqdm import tqdm

from .metric import key14
from .spectra import N_BINS, BIN_WIDTH, MAX_MZ, clean_peaks, vectorize

REL_FLOOR = 0.01
TOP_N = 64
PRECURSOR_TOL = 0.5


def _list_column(table: pa.Table, name: str) -> tuple[np.ndarray, np.ndarray]:
    """Return (offsets, flat values) for a list<double> column."""
    col = table[name]
    arr = col.combine_chunks() if isinstance(col, pa.ChunkedArray) else col
    offsets = arr.offsets.to_numpy().astype(np.int64)
    values = arr.values.to_numpy(zero_copy_only=False).astype(np.float64)
    # combine_chunks can leave a non-zero start offset
    values = values[offsets[0]:offsets[-1]]
    offsets = offsets - offsets[0]
    return offsets, values


def ensure_keys(table: pa.Table) -> np.ndarray:
    """InChIKey14 per row: provided column if present, else computed from smiles."""
    if "inchikey14" in table.column_names:
        return np.asarray(table["inchikey14"].to_pylist(), dtype=object)
    return np.asarray([key14(s) for s in table["smiles"].to_pylist()], dtype=object)


def binned_matrix(table: pa.Table, precursor: np.ndarray) -> sp.csr_matrix:
    """Vectorised clean_peaks + vectorize for every row of the table."""
    offsets, mz = _list_column(table, "mzs")
    _, inten = _list_column(table, "intensities")
    n = len(table)
    lengths = np.diff(offsets)
    row = np.repeat(np.arange(n, dtype=np.int64), lengths)

    # normalise to base peak = 1 per row (train intensities should already be, but be safe)
    nonempty = lengths > 0
    row_max = np.zeros(n)
    if nonempty.any():
        row_max[nonempty] = np.maximum.reduceat(inten, offsets[:-1][nonempty])
    inten = inten / np.where(row_max[row] > 0, row_max[row], 1.0)

    keep = (inten >= REL_FLOOR) & (mz > 0) & (mz < MAX_MZ) & (mz < precursor[row] - PRECURSOR_TOL)
    row, mz, inten = row[keep], mz[keep], inten[keep]

    # top-N most intense peaks per row
    order = np.lexsort((-inten, row))
    row, mz, inten = row[order], mz[order], inten[order]
    first = np.r_[0, np.flatnonzero(np.diff(row)) + 1]
    starts = np.repeat(first, np.diff(np.r_[first, row.size]))
    rank = np.arange(row.size) - starts
    keep = rank < TOP_N
    row, mz, inten = row[keep], mz[keep], inten[keep]

    bins = np.floor(mz / BIN_WIDTH).astype(np.int32)
    vals = np.sqrt(inten).astype(np.float32)
    mat = sp.csr_matrix((vals, (row, bins)), shape=(n, N_BINS))
    mat.sum_duplicates()
    norms = np.sqrt(np.asarray(mat.multiply(mat).sum(axis=1)).ravel())
    norms[norms == 0] = 1.0
    mat = sp.diags(1.0 / norms).dot(mat).tocsr()
    return mat


@dataclass
class Library:
    matrix: sp.csr_matrix          # n_spectra x N_BINS, rows L2-normalised
    precursor_mz: np.ndarray       # n_spectra
    key: np.ndarray                # n_spectra, InChIKey14 (object)
    key_to_smiles: dict[str, str]  # one representative SMILES per key
    order: np.ndarray              # argsort of precursor_mz for window lookup

    @classmethod
    def from_table(cls, table: pa.Table) -> "Library":
        """table needs canonical columns: mzs, intensities, precursor_mz, smiles, optional inchikey14."""
        keys = ensure_keys(table)
        smiles = table["smiles"].to_pylist()
        prec = np.asarray(table["precursor_mz"].to_numpy(zero_copy_only=False), dtype=np.float64)
        valid = np.array([k is not None for k in keys]) & np.isfinite(prec)
        if not valid.all():
            table = table.filter(pa.array(valid))
            keys, prec = keys[valid], prec[valid]
            smiles = [s for s, v in zip(smiles, valid) if v]
        key_to_smiles: dict[str, str] = {}
        for k, s in zip(keys, smiles):
            key_to_smiles.setdefault(k, s)
        return cls(
            matrix=binned_matrix(table, prec),
            precursor_mz=prec,
            key=keys,
            key_to_smiles=key_to_smiles,
            order=np.argsort(prec, kind="stable"),
        )

    @classmethod
    def from_frame(cls, df: pd.DataFrame, show_progress: bool = False) -> "Library":
        """Convenience for small pandas frames (tests, notebooks)."""
        return cls.from_table(pa.Table.from_pandas(df, preserve_index=False))

    def window(self, mz: float, ppm: float) -> np.ndarray:
        tol = mz * ppm * 1e-6
        sorted_mz = self.precursor_mz[self.order]
        lo = np.searchsorted(sorted_mz, mz - tol, side="left")
        hi = np.searchsorted(sorted_mz, mz + tol, side="right")
        return self.order[lo:hi]

    def nearest_mass_keys(self, mz: float, n: int) -> list[str]:
        """Fallback fill: distinct structures with precursor closest to mz."""
        sorted_mz = self.precursor_mz[self.order]
        pos = int(np.searchsorted(sorted_mz, mz))
        out: list[str] = []
        seen: set[str] = set()
        lo, hi = pos - 1, pos
        while len(out) < n and (lo >= 0 or hi < sorted_mz.size):
            take_lo = hi >= sorted_mz.size or (lo >= 0 and mz - sorted_mz[lo] <= sorted_mz[hi] - mz)
            idx = self.order[lo] if take_lo else self.order[hi]
            if take_lo:
                lo -= 1
            else:
                hi += 1
            k = self.key[idx]
            if k not in seen:
                seen.add(k)
                out.append(k)
        return out


def score_query(lib: Library, mzs, intensities, precursor_mz: float, ppm: float = 20.0) -> dict[str, float]:
    """Return structure key -> best cosine over library spectra within the precursor window."""
    cand = lib.window(precursor_mz, ppm)
    if cand.size == 0:
        return {}
    mz, inten = clean_peaks(mzs, intensities, precursor_mz, rel_floor=REL_FLOOR, top_n=TOP_N)
    bins, vals = vectorize(mz, inten)
    if bins.size == 0:
        return {}
    q = np.zeros(lib.matrix.shape[1], dtype=np.float32)
    q[bins] = vals
    sims = lib.matrix[cand].dot(q)
    best: dict[str, float] = {}
    for idx, s in zip(cand, sims):
        k = lib.key[idx]
        if s > best.get(k, -1.0):
            best[k] = float(s)
    return best


def rrf(rankings: list[list[str]], k: int = 60) -> list[str]:
    """Reciprocal rank fusion of several ranked key lists."""
    score: dict[str, float] = defaultdict(float)
    for ranking in rankings:
        for r, key in enumerate(ranking, start=1):
            score[key] += 1.0 / (k + r)
    return [key for key, _ in sorted(score.items(), key=lambda kv: -kv[1])]


def predict(
    lib: Library,
    test: pd.DataFrame,
    ppm: float = 20.0,
    top_k: int = 25,
    show_progress: bool = True,
) -> dict[str, list[str]]:
    """test: pandas frame with molecule_id, mzs, intensities, precursor_mz. Returns molecule_id -> ranked SMILES."""
    out: dict[str, list[str]] = {}
    groups = test.groupby("molecule_id", sort=False)
    it = tqdm(groups, total=groups.ngroups, desc="searching") if show_progress else groups
    for mid, g in it:
        per_spectrum: list[list[str]] = []
        for r in g.itertuples(index=False):
            scores = score_query(lib, np.asarray(r.mzs), np.asarray(r.intensities), float(r.precursor_mz), ppm)
            per_spectrum.append([k for k, _ in sorted(scores.items(), key=lambda kv: -kv[1])[:200]])
        fused = rrf(per_spectrum)[:top_k]
        if len(fused) < top_k:
            mz = float(np.median(g["precursor_mz"]))
            for k in lib.nearest_mass_keys(mz, top_k * 2):
                if k not in fused:
                    fused.append(k)
                if len(fused) >= top_k:
                    break
        out[mid] = [lib.key_to_smiles[k] for k in fused[:top_k]]
    return out
