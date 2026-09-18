"""Library-search baseline end to end.

    python scripts/run_baseline.py --mode val      # pseudo Class-2 validation (timsTOF queries in test mass range)
    python scripts/run_baseline.py --mode val2x    # harsh Class-2: timsTOF spectra held out, structure kept via other instruments
    python scripts/run_baseline.py --mode val3     # pseudo Class-3 validation (expect near 0, documents the gap)
    python scripts/run_baseline.py --mode submit   # writes artifacts/submission.csv

Use --limit to work on the first N training rows while iterating.
"""
import argparse
import sys
import time
from pathlib import Path

import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
from casmi.library_search import Library, predict  # noqa: E402
from casmi.metric import mrr_at_k  # noqa: E402
from casmi.schema import resolve_columns  # noqa: E402
from casmi.submission import validate_submission, write_submission  # noqa: E402
from casmi.validation import metadata, query_frame, split_class2, split_class3, split_cross_instrument, truth  # noqa: E402

NEEDED = ["smiles", "inchikey14", "precursor_mz", "mzs", "intensities", "instrument", "adduct", "ingest_lib", "molecule_id"]


def load_canonical(path: Path, limit: int | None = None) -> pa.Table:
    """Read only the needed columns and rename them to canonical names."""
    schema = pq.read_schema(path)
    resolved = resolve_columns(schema.names)
    cols = {canon: real for canon, real in resolved.items() if canon in NEEDED and real}
    table = pq.read_table(path, columns=list(cols.values()))
    table = table.rename_columns([next(c for c, r in cols.items() if r == name) for name in table.column_names])
    return table.slice(0, limit) if limit else table


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--mode", choices=["val", "val2x", "val3", "submit"], default="val")
    ap.add_argument("--data", default=str(ROOT / "data"))
    ap.add_argument("--limit", type=int, default=None, help="use only the first N training rows")
    ap.add_argument("--ppm", type=float, default=20.0)
    ap.add_argument("--n-queries", type=int, default=400)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--instrument-regex", default="timsTOF")
    args = ap.parse_args()

    t0 = time.time()
    train = load_canonical(Path(args.data) / "train.parquet", args.limit)
    print(f"train rows: {len(train):,}  cols: {train.column_names}  ({time.time()-t0:.0f}s)")

    if args.mode in ("val", "val2x", "val3"):
        meta = metadata(train)
        print(f"metadata ready ({time.time()-t0:.0f}s)")
        if args.mode == "val":
            lib_rows, q_rows = split_class2(meta, args.n_queries, args.seed, args.instrument_regex)
        elif args.mode == "val2x":
            lib_rows, q_rows = split_cross_instrument(meta, args.n_queries, args.seed, args.instrument_regex)
        else:
            lib_rows, q_rows = split_class3(meta, args.n_queries, args.seed, args.instrument_regex)
        lib = Library.from_table(train.take(pa.array(lib_rows)))
        print(f"library built: {lib.matrix.shape[0]:,} spectra, {len(lib.key_to_smiles):,} structures ({time.time()-t0:.0f}s)")
        queries = query_frame(train, meta, q_rows)
        preds = predict(lib, queries, ppm=args.ppm)
        score = mrr_at_k(truth(meta, q_rows), preds)
        print(f"\n{args.mode} MRR@25 = {score:.4f}  (queries={queries['molecule_id'].nunique()}, "
              f"query spectra={len(queries)}, library={lib.matrix.shape[0]:,})  total {time.time()-t0:.0f}s")
    else:
        test = load_canonical(Path(args.data) / "test.parquet").to_pandas()
        lib = Library.from_table(train)
        print(f"library built: {lib.matrix.shape[0]:,} spectra ({time.time()-t0:.0f}s)")
        preds = predict(lib, test, ppm=args.ppm)
        sample = Path(args.data) / "sample_submission.csv"
        out = ROOT / "artifacts" / "submission.csv"
        sub = write_submission(preds, str(sample), str(out))
        validate_submission(sub, pd.read_csv(sample))
        print(f"wrote {out} ({len(sub)} rows) in {time.time()-t0:.0f}s")


if __name__ == "__main__":
    main()
