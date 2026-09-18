"""Write and validate submission.csv in the competition format."""
from __future__ import annotations

import pandas as pd

from .metric import key14


def write_submission(preds: dict[str, list[str]], sample_path: str, out_path: str, k: int = 25) -> pd.DataFrame:
    sample = pd.read_csv(sample_path)
    id_col, smi_col = sample.columns[0], sample.columns[1]
    rows = []
    for mid in sample[id_col]:
        cands = preds.get(mid, [])
        uniq, seen = [], set()
        for s in cands:
            kk = key14(s)
            if kk is None or kk in seen:
                continue
            seen.add(kk)
            uniq.append(s)
            if len(uniq) == k:
                break
        if not uniq:
            uniq = ["C"]  # never leave a row empty
        rows.append({id_col: mid, smi_col: ";".join(uniq)})
    sub = pd.DataFrame(rows)
    sub.to_csv(out_path, index=False)
    return sub


def validate_submission(sub: pd.DataFrame, sample: pd.DataFrame, k: int = 25) -> None:
    assert list(sub.columns) == list(sample.columns), "column mismatch"
    assert len(sub) == len(sample), "row count mismatch"
    assert (sub.iloc[:, 0].values == sample.iloc[:, 0].values).all(), "row order mismatch"
    for cell in sub.iloc[:, 1]:
        parts = [p for p in str(cell).split(";") if p]
        assert 1 <= len(parts) <= k, f"bad candidate count: {len(parts)}"
