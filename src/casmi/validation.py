"""Leakage-safe pseudo validation splits mirroring the three test classes.

Class 1  library known:      query spectrum is itself in the library (trivial, sanity check).
Class 2  known structure:    structure stays in the library via OTHER spectra, query spectrum held out.
Class 3  novel structure:    every spectrum of the structure is removed from the library.

Splits operate on a metadata frame (one row per spectrum, no peak lists) and
return row index arrays into the full pyarrow table. Grouping is by InChIKey14
so tautomers and stereoisomers of a held-out structure cannot leak back in.

Test set is all timsTOF with precursor m/z 245 to 460, so queries default to
timsTOF spectra in that range to reflect the real domain.
"""
from __future__ import annotations

import numpy as np
import pandas as pd
import pyarrow as pa

from .library_search import ensure_keys

TEST_MZ_RANGE = (240.0, 465.0)


def metadata(table: pa.Table) -> pd.DataFrame:
    """One row per spectrum: key14, smiles, precursor_mz, instrument, adduct."""
    cols = [c for c in ["smiles", "precursor_mz", "instrument", "adduct", "ingest_lib"] if c in table.column_names]
    meta = table.select(cols).to_pandas()
    meta["key14"] = ensure_keys(table)
    meta = meta[meta["key14"].notna()]
    return meta


def _query_pool(meta: pd.DataFrame, instrument_regex: str | None, mz_range: tuple[float, float] | None) -> pd.DataFrame:
    pool = meta
    if instrument_regex and "instrument" in meta.columns:
        pool = pool[pool["instrument"].astype(str).str.contains(instrument_regex, case=False, regex=True)]
    if mz_range:
        pool = pool[(pool["precursor_mz"] >= mz_range[0]) & (pool["precursor_mz"] <= mz_range[1])]
    return pool


def split_class2(
    meta: pd.DataFrame,
    n_queries: int = 400,
    seed: int = 0,
    instrument_regex: str | None = "timsTOF",
    mz_range: tuple[float, float] | None = TEST_MZ_RANGE,
) -> tuple[np.ndarray, np.ndarray]:
    """Return (library_rows, query_rows). Each query structure keeps at least one other spectrum in the library."""
    rng = np.random.default_rng(seed)
    counts = meta["key14"].value_counts()
    pool = _query_pool(meta, instrument_regex, mz_range)
    pool = pool[pool["key14"].isin(counts[counts >= 2].index)]
    one_per_key = pool.groupby("key14").sample(n=1, random_state=int(rng.integers(1 << 31))).index.to_numpy()
    query_rows = rng.choice(one_per_key, size=min(n_queries, one_per_key.size), replace=False)
    library_rows = np.setdiff1d(meta.index.to_numpy(), query_rows)
    return library_rows, query_rows


def split_class3(
    meta: pd.DataFrame,
    n_structures: int = 400,
    seed: int = 0,
    instrument_regex: str | None = "timsTOF",
    mz_range: tuple[float, float] | None = TEST_MZ_RANGE,
) -> tuple[np.ndarray, np.ndarray]:
    """Return (library_rows, query_rows). Every spectrum of a query structure leaves the library."""
    rng = np.random.default_rng(seed)
    pool = _query_pool(meta, instrument_regex, mz_range)
    keys = pool["key14"].unique()
    held = rng.choice(keys, size=min(n_structures, keys.size), replace=False)
    is_held = meta["key14"].isin(set(held))
    query_rows = meta.index[is_held & meta.index.isin(pool.index)].to_numpy()
    library_rows = meta.index[~is_held].to_numpy()
    return library_rows, query_rows


def split_cross_instrument(
    meta: pd.DataFrame,
    n_structures: int = 400,
    seed: int = 0,
    instrument_regex: str = "timsTOF",
    mz_range: tuple[float, float] | None = TEST_MZ_RANGE,
) -> tuple[np.ndarray, np.ndarray]:
    """Harsh Class 2: every timsTOF spectrum of a query structure is held out; the structure stays in the
    library only through spectra from OTHER instruments. Mirrors "structure known from public libraries,
    spectrum newly measured on the test instrument"."""
    rng = np.random.default_rng(seed)
    is_inst = meta["instrument"].astype(str).str.contains(instrument_regex, case=False, regex=True)
    with_other = set(meta.loc[~is_inst, "key14"])
    pool = _query_pool(meta, instrument_regex, mz_range)
    keys = np.array(sorted(set(pool["key14"]) & with_other))
    held = set(rng.choice(keys, size=min(n_structures, keys.size), replace=False))
    is_held = meta["key14"].isin(held)
    query_rows = meta.index[is_held & is_inst].to_numpy()
    library_rows = meta.index[~(is_held & is_inst)].to_numpy()
    return library_rows, query_rows


def query_frame(table: pa.Table, meta: pd.DataFrame, query_rows: np.ndarray) -> pd.DataFrame:
    """Pandas frame for predict(): molecule_id is the structure key, so multi-spectrum fusion mirrors the test."""
    q = table.take(pa.array(query_rows)).select(["mzs", "intensities", "precursor_mz"]).to_pandas()
    q["molecule_id"] = meta.loc[query_rows, "key14"].to_numpy()
    return q


def truth(meta: pd.DataFrame, query_rows: np.ndarray) -> dict[str, str]:
    q = meta.loc[query_rows]
    return q.groupby("key14")["smiles"].first().to_dict()
