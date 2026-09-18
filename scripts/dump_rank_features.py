"""Build ranker training rows from the harness: per-candidate features with truth labels.

Structures are sampled from the training set (timsTOF spectra, test mass range), with the
np-examples structures held out for evaluation. Each structure yields rows under class 1
(everything visible) and class 2 (own spectra hidden). With --overlap, the 545 structures that
COCONUT also holds yield class 3 rows (training copy removed from the pool).

Output: artifacts/rank_rows_<tag>.npz with X (float32, 31 features), y (int8), group (structure
index), cls (1, 2 or 3), n_cand, and the structure keys. Checkpointed every 50 structures.

    python scripts/dump_rank_features.py --n 1500 --prune 400 --tag c12
    python scripts/dump_rank_features.py --overlap --prune 400 --tag c3
"""
import argparse
import os
import pickle
import sys
import time
from pathlib import Path

import numpy as np
import pyarrow as pa
import pyarrow.compute as pc
import pyarrow.parquet as pq

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

COLS = ["inchikey14", "normalized_smiles", "ingest_lib", "precursor_mz", "adduct", "ionization_mode",
        "instrument_type", "collision_energy_ev", "ms2_mzs", "ms2_normalized_intensities"]


def pool_canonical_keys(cache_root: Path, public_root: Path):
    """Canonical key per pool row, aligned with harness.init's pool (COCONUT rows, then train rows, finite mass only)."""
    z = np.load(cache_root / "pool_key14.npz", allow_pickle=True)
    k14 = z["key14"]
    co_mass = np.load(public_root / "coconut-casmi26-candidates" / "coco_mass.npy")
    tr_mass = np.load(cache_root / "train_fp.npz", allow_pickle=True)["mass"]
    mass = np.concatenate([co_mass, tr_mass])
    assert len(mass) == len(k14), (len(mass), len(k14))
    return k14[np.isfinite(mass)]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", default=str(ROOT / "data"))
    ap.add_argument("--public", default=str(ROOT / "data" / "public"))
    ap.add_argument("--cache", default=str(ROOT / "data" / "cache"))
    ap.add_argument("--ranker", default=str(ROOT / "artifacts" / "ranker_versions" / "v4" / "rank_train.npz"))
    ap.add_argument("--n", type=int, default=1500)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--prune", type=int, default=400)
    ap.add_argument("--prune-mode", default="lib")
    ap.add_argument("--overlap", action="store_true", help="class 3 rows from the COCONUT-overlap structures")
    ap.add_argument("--overlap-classes", default="3", help="classes to dump for the overlap structures")
    ap.add_argument("--tag", default="c12")
    ap.add_argument("--workers", type=int, default=None)
    args = ap.parse_args()

    from casmi.fork import harness, core
    from casmi.metric import key14

    t0 = time.time()
    st = harness.init(args.data, args.public, args.cache, workers=args.workers, ranker_path=args.ranker)
    pool_k14 = pool_canonical_keys(Path(args.cache), Path(args.public))
    assert len(pool_k14) == len(st.pool.mass)
    # harness pool is sorted by mass; recover the permutation to align canonical keys
    co_mass = np.load(Path(args.public) / "coconut-casmi26-candidates" / "coco_mass.npy")
    tr_mass = np.load(Path(args.cache) / "train_fp.npz", allow_pickle=True)["mass"]
    mass = np.concatenate([co_mass, tr_mass])
    mass = mass[np.isfinite(mass)]
    order = np.argsort(mass)
    pool_k14 = pool_k14[order]
    assert np.allclose(mass[order], st.pool.mass)

    tb = pq.read_table(Path(args.data) / "train.parquet", columns=COLS)
    npx_keys = set(tb.filter(pc.equal(tb["ingest_lib"], "enveda-np-examples"))["inchikey14"].to_pylist())
    rng = np.random.default_rng(args.seed)

    if args.overlap:
        z = np.load(Path(args.cache) / "pool_key14.npz", allow_pickle=True)
        k14, src = z["key14"], z["source"]
        co = set(x for x in k14[src == 0] if x)
        tr_keys = np.load(Path(args.cache) / "train_fp.npz", allow_pickle=True)["keys"]
        chosen = sorted(set(tr_keys[i] for i, x in enumerate(k14[src == 1]) if x and x in co) - npx_keys)
        classes = [int(c) for c in args.overlap_classes.split(",")]
    else:
        meta = tb.select(["inchikey14", "instrument_type", "precursor_mz", "adduct"]).to_pandas()
        meta["nm"] = core.neutral_mass(meta.precursor_mz.values.astype(float), meta.adduct.values)
        ok = (meta.instrument_type == "timsTOF") & meta.nm.between(240, 465) & ~meta.inchikey14.isin(npx_keys)
        cand_keys = np.array(sorted(meta.loc[ok, "inchikey14"].unique()))
        chosen = list(rng.choice(cand_keys, size=min(args.n, len(cand_keys)), replace=False))
        classes = [1, 2]
    print(f"[INFO] {len(chosen)} structures, classes {classes}, prune {args.prune}:{args.prune_mode}", flush=True)

    sub_tb = tb.filter(pc.is_in(tb["inchikey14"], value_set=pa.array(chosen)))
    df = sub_tb.to_pandas()
    df["_tims"] = (df.instrument_type == "timsTOF").astype(int)
    df = df.sort_values(["inchikey14", "_tims"], ascending=[True, False])
    df = df.groupby("inchikey14", sort=False).head(6).drop(columns="_tims").reset_index(drop=True)
    groups = {k: g.reset_index(drop=True) for k, g in df.groupby("inchikey14")}

    out = ROOT / "artifacts" / f"rank_rows_{args.tag}.npz"
    X_all, y_all, g_all, c_all, n_all, keys_done = [], [], [], [], [], []

    def checkpoint():
        if not X_all:
            return
        np.savez(out, X=np.vstack(X_all).astype(np.float32), y=np.concatenate(y_all).astype(np.int8),
                 group=np.concatenate(g_all).astype(np.int32), cls=np.concatenate(c_all).astype(np.int8),
                 n_cand=np.concatenate(n_all).astype(np.int16), keys=np.asarray(keys_done, dtype=object))

    for gi, key in enumerate(chosen):
        sub = groups.get(key)
        if sub is None or not len(sub):
            continue
        truth_k = key14(sub.normalized_smiles.iloc[0])
        for c in classes:
            ch = harness.channels(st, sub, hide_key=None if c == 1 else key)
            ev = harness.evidence(st, ch, drop_key=key if c == 3 else None, prune_n=args.prune, prune_mode=args.prune_mode)
            if ev is None or not len(ev.cand):
                continue
            X = core.rank_features(ev.cfp, ev.lv, ev.afp, ev.sims, ev.zlog, ev.fsc)
            y = (pool_k14[ev.cand] == truth_k).astype(np.int8)
            X_all.append(X); y_all.append(y)
            g_all.append(np.full(len(y), gi)); c_all.append(np.full(len(y), c)); n_all.append(np.full(len(y), len(y)))
        keys_done.append(key)
        if gi % 50 == 0 or gi == len(chosen) - 1:
            checkpoint()
            harness.save_frag_cache(st)
            pos = int(np.concatenate(y_all).sum()) if y_all else 0
            print(f"  {gi+1}/{len(chosen)}  rows {sum(len(y) for y in y_all):,}  positives {pos}  {time.time()-t0:.0f}s", flush=True)
    checkpoint()
    print(f"wrote {out}  total {time.time()-t0:.0f}s")


if __name__ == "__main__":
    main()
