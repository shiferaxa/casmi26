"""First thing to run after downloading: prints schema, resolved columns, and basic stats.

    python scripts/inspect_data.py data/
"""
import sys
from pathlib import Path

import pyarrow.parquet as pq
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from casmi.schema import resolve_columns  # noqa: E402

root = Path(sys.argv[1] if len(sys.argv) > 1 else "data")
for name in ["train.parquet", "test.parquet"]:
    p = root / name
    if not p.exists():
        print(f"missing {p}")
        continue
    try:
        pf = pq.ParquetFile(p)
    except Exception as e:
        print(f"cannot open {p}: {e}")
        continue
    print(f"\n=== {name}: {pf.metadata.num_rows:,} rows, {pf.metadata.num_row_groups} row groups")
    print(pf.schema_arrow)
    print("resolved:", resolve_columns(pf.schema_arrow.names))
    head = pf.read_row_group(0).to_pandas().head(3)
    with pd.option_context("display.max_columns", None, "display.width", 200):
        print(head)
    # cheap categorical summaries on the first row group
    rg = pf.read_row_group(0).to_pandas()
    for c in rg.columns:
        if rg[c].dtype == object and rg[c].map(type).eq(str).all() and rg[c].nunique() < 50:
            print(f"\n{c} value counts (row group 0):\n{rg[c].value_counts().head(20)}")

s = root / "sample_submission.csv"
if s.exists():
    ss = pd.read_csv(s)
    print(f"\n=== sample_submission.csv: {len(ss)} rows, columns {list(ss.columns)}")
    print(ss.head(3))
