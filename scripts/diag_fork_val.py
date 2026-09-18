"""Failure breakdown for the public pipeline on np-examples, classes 2 and 3.

For every structure: is the truth in COCONUT, in the candidate pool, inside the 8.5 ppm window,
still present after the 80-candidate prune, and at what rank. Tells whether misses come from
mass windows, pruning, or ranking.

    python scripts/diag_fork_val.py [--limit N]
"""
import argparse
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
import pyarrow.compute as pc
import pyarrow.parquet as pq

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", default=str(ROOT / "data"))
    ap.add_argument("--public", default=str(ROOT / "data" / "public"))
    ap.add_argument("--cache", default=str(ROOT / "data" / "cache"))
    ap.add_argument("--limit", type=int, default=None)
    ap.add_argument("--out", default=str(ROOT / "artifacts" / "diag_np_examples.csv"))
    args = ap.parse_args()

    from casmi.fork import core, harness
    from casmi.fork.core import CFG
    from casmi.metric import key14, reciprocal_rank

    t0 = time.time()
    st = harness.init(args.data, args.public, args.cache)
    pool = st.pool
    cols = ["inchikey14", "normalized_smiles", "ingest_lib", "precursor_mz", "adduct", "ionization_mode",
            "instrument_type", "collision_energy_ev", "ms2_mzs", "ms2_normalized_intensities"]
    tb = pq.read_table(Path(args.data) / "train.parquet", columns=cols)
    npx = tb.filter(pc.equal(tb["ingest_lib"], "enveda-np-examples")).to_pandas()
    keys = sorted(npx.inchikey14.unique())
    if args.limit:
        keys = keys[: args.limit]
    pool_keys = set(pool.keys)
    rows = []
    for n, key in enumerate(keys):
        sub = npx[npx.inchikey14 == key].reset_index(drop=True)
        truth = sub.normalized_smiles.iloc[0]
        nm = core.neutral_mass(sub.precursor_mz.values.astype(np.float64), sub.adduct.values)
        nms = nm[np.isfinite(nm)]
        target = float(np.median(nms)) if len(nms) else np.nan
        win = pool.window(target, CFG.PPM_WIN) if len(nms) else np.zeros(0, int)
        in_window = bool(len(win)) and key in set(pool.keys[win])
        r = dict(key=key, n_spectra=len(sub), adducts=",".join(sorted(set(sub.adduct))), target=target,
                 in_coconut=key in pool.coconut_keys, in_pool=key in pool_keys, n_window=int(len(win)),
                 in_window=in_window)
        if in_window:
            i = pool.k2i[key]
            r["truth_ppm"] = float(abs(pool.mass[i] - target) / target * 1e6)
        for c in (2, 3):
            ev = harness.evidence(st, sub, hide_key=key, drop_from_pool=(c == 3))
            in_cand = ev is not None and len(ev.cand) > 0 and key in set(pool.keys[ev.cand])
            smis = harness.rank(st, ev)
            rr = reciprocal_rank(truth, smis)
            r[f"c{c}_in_cand"] = in_cand
            r[f"c{c}_rank"] = int(round(1 / rr)) if rr > 0 else 0
            r[f"c{c}_lv_max"] = float(ev.lv.max()) if ev is not None and len(ev.lv) else 0.0
            r[f"c{c}_analog_top"] = float(ev.sims[0]) if ev is not None and len(ev.sims) else 0.0
        rows.append(r)
        if n % 25 == 0 or n == len(keys) - 1:
            print(f"  {n+1}/{len(keys)}  {time.time()-t0:.0f}s", flush=True)
            harness.save_frag_cache(st)

    d = pd.DataFrame(rows)
    d.to_csv(args.out, index=False)
    N = len(d)
    print(f"\nstructures {N}")
    print(f"in COCONUT {d.in_coconut.sum()}   in pool {d.in_pool.sum()}   in 8.5ppm window {d.in_window.sum()}")
    print(f"window size: median {d.n_window.median():.0f}  max {d.n_window.max()}  over 80: {(d.n_window > 80).sum()}")
    for c in (2, 3):
        ic = d[f"c{c}_in_cand"]
        rk = d[f"c{c}_rank"]
        print(f"class{c}: truth in pruned candidates {ic.sum()}/{N}; of those rank1 {(rk == 1).sum()}, "
              f"rank2-5 {((rk >= 2) & (rk <= 5)).sum()}, rank6-25 {((rk >= 6) & (rk <= 25)).sum()}, "
              f"outside top25 {((rk == 0) & ic).sum()}; lost by window {(~d.in_window).sum()}, "
              f"lost by prune {(d.in_window & ~ic).sum()}")
    lost = d[d.in_window & ~d.c2_in_cand]
    if len(lost):
        print("\nclass2 lost by prune, window sizes:", lost.n_window.describe()[["min", "50%", "max"]].to_dict())
    miss_w = d[~d.in_window]
    if len(miss_w):
        print("\nnot in window: adducts", miss_w.adducts.value_counts().head(8).to_dict(), " in_pool", miss_w.in_pool.sum())
    print(f"total {time.time()-t0:.0f}s")


if __name__ == "__main__":
    main()
