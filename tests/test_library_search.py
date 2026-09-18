"""Synthetic end-to-end: a small library, a query equal to one structure plus a same-mass decoy."""
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from casmi.library_search import Library, predict  # noqa: E402
from casmi.metric import mrr_at_k  # noqa: E402

CAFFEINE = "CN1C=NC2=C1C(=O)N(C)C(=O)N2C"


def row(smiles, precursor_mz, peaks, molecule_id=None):
    d = dict(
        smiles=smiles,
        precursor_mz=precursor_mz,
        mzs=np.array([p[0] for p in peaks], dtype=float),
        intensities=np.array([p[1] for p in peaks], dtype=float),
    )
    if molecule_id:
        d["molecule_id"] = molecule_id
    return d


def test_exact_hit_ranks_first():
    lib_df = pd.DataFrame([
        row(CAFFEINE, 195.0877, [(138.066, 1.0), (110.071, 0.4), (83.06, 0.1)]),
        row("OC(=O)c1ccccc1O", 139.039, [(121.03, 1.0), (93.03, 0.5)]),
        row("CC1=CC(=O)C=CC1=O", 123.044, [(95.05, 1.0), (67.05, 0.3)]),
        # decoy at the same precursor mass with different fragments
        row("Cn1cnc2c1c(=O)[nH]c(=O)n2C", 195.0877, [(150.0, 1.0), (60.0, 0.9)]),
    ])
    lib = Library.from_frame(lib_df, show_progress=False)
    test = pd.DataFrame([
        row(None, 195.0879, [(138.07, 1.0), (110.07, 0.35), (83.06, 0.12)], molecule_id="m1"),
    ]).drop(columns=["smiles"])
    preds = predict(lib, test, ppm=20, top_k=25, show_progress=False)
    assert mrr_at_k({"m1": CAFFEINE}, preds) == 1.0
    assert 1 < len(preds["m1"]) <= 25  # padded with nearest-mass structures
