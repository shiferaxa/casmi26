"""Validate the transplanted public pipeline on the enveda-np-examples structures.

These 250 structures were measured by the host on the test instrument (timsTOF), so they are the
closest proxy to the test set. Three masks per structure:
  class1  everything visible (library known)
  class2  its own spectra hidden from library and analog channels, structure stays in the pool
  class3  as class2, and the structure key is removed from the candidate pool

Channels are computed once per structure and mask; prune variants recompute candidates and
fragmentation; ranking variants are scored from each evidence.

    python scripts/run_fork_val.py
    python scripts/run_fork_val.py --prunes 80:lib,400:lib,200:model --classes 2,3 --gates none --lib-hards none
"""
import argparse
import sys
import time
from collections import Counter
from itertools import product
from pathlib import Path

import numpy as np
import pyarrow as pa
import pyarrow.compute as pc
import pyarrow.parquet as pq

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))


def _parse_floats(s: str):
    return [None if t.strip().lower() == "none" else float(t) for t in s.split(",")]


def _parse_prunes(s: str):
    out = []
    for t in s.split(","):
        n, mode = t.split(":")
        out.append((int(n), mode))
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", default=str(ROOT / "data"))
    ap.add_argument("--public", default=str(ROOT / "data" / "public"))
    ap.add_argument("--cache", default=str(ROOT / "data" / "cache"))
    ap.add_argument("--limit", type=int, default=None)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--prunes", default="80:lib")
    ap.add_argument("--gates", default="none")
    ap.add_argument("--lib-hards", default="none")
    ap.add_argument("--no-frag", action="store_true")
    ap.add_argument("--classes", default="1,2,3")
    ap.add_argument("--split", choices=["npx", "overlap"], default="npx")
    ap.add_argument("--workers", type=int, default=None)
    ap.add_argument("--bio", action="store_true", help="add the ChEBI/LIPID MAPS block to the candidate pool")
    ap.add_argument("--sim-powers", default="4", help="analog similarity powers to rank with, comma separated (public code 4.0, public ranker rows built at 3.0)")
    ap.add_argument("--models", default=None, help="folder with the fp_*.pt checkpoints for channel 4 (default: public v2)")
    ap.add_argument("--ranker", default=None, help="rank_train.npz to fit the ranker from (default: the one under data/public)")
    args = ap.parse_args()

    from casmi.fork import core, harness
    from casmi.metric import mrr_at_k, reciprocal_rank

    t0 = time.time()
    st = harness.init(args.data, args.public, args.cache, workers=args.workers, ranker_path=args.ranker, use_bio=args.bio, models_dir=args.models)

    cols = ["inchikey14", "normalized_smiles", "ingest_lib", "precursor_mz", "adduct", "ionization_mode",
            "instrument_type", "collision_energy_ev", "ms2_mzs", "ms2_normalized_intensities"]
    tb = pq.read_table(Path(args.data) / "train.parquet", columns=cols)
    if args.split == "npx":
        npx = tb.filter(pc.equal(tb["ingest_lib"], "enveda-np-examples")).to_pandas()
    else:
        # overlap split: training structures that COCONUT also holds (by canonical key), so removing the
        # training copy from the pool leaves the COCONUT copy as the only route to the truth (real class 3)
        z = np.load(Path(args.cache) / "pool_key14.npz", allow_pickle=True)
        k14, src = z["key14"], z["source"]
        co = set(x for x in k14[src == 0] if x)
        tr_fp = np.load(Path(args.cache) / "train_fp.npz", allow_pickle=True)
        tr_keys = tr_fp["keys"]
        overlap_train_keys = set(tr_keys[i] for i, x in enumerate(k14[src == 1]) if x and x in co)
        sub_tb = tb.filter(pc.is_in(tb["inchikey14"], value_set=pa.array(sorted(overlap_train_keys))))
        df = sub_tb.to_pandas()
        # prefer timsTOF spectra and cap the spectra per structure like the test set
        df["_tims"] = (df.instrument_type == "timsTOF").astype(int)
        df = df.sort_values(["inchikey14", "_tims"], ascending=[True, False])
        npx = df.groupby("inchikey14", sort=False).head(6).drop(columns="_tims").reset_index(drop=True)
        print(f"[INFO] overlap split: {npx.inchikey14.nunique()} structures, {len(npx)} spectra, "
              f"{int((npx.instrument_type == 'timsTOF').sum())} timsTOF", flush=True)
    keys = sorted(npx.inchikey14.unique())
    rng = np.random.default_rng(args.seed)
    if args.limit:
        keys = list(rng.choice(keys, size=min(args.limit, len(keys)), replace=False))
    print(f"[INFO] {args.split}: {len(npx)} spectra, {len(keys)} structures evaluated", flush=True)

    classes = [int(c) for c in args.classes.split(",")]
    prunes = _parse_prunes(args.prunes)
    rvars = list(product(_parse_floats(args.gates), _parse_floats(args.lib_hards), _parse_floats(args.sim_powers)))
    combos = [(c, p, v) for c in classes for p in prunes for v in rvars]
    results = {k: {} for k in combos}
    ranks = {k: Counter() for k in combos}
    truth = {}
    for n, key in enumerate(keys):
        sub = npx[npx.inchikey14 == key].reset_index(drop=True)
        truth[key] = sub.normalized_smiles.iloc[0]
        for c in classes:
            ch = harness.channels(st, sub, hide_key=None if c == 1 else key)
            for p in prunes:
                ev = harness.evidence(st, ch, drop_key=key if c == 3 else None, prune_n=p[0], prune_mode=p[1],
                                      use_frag=not args.no_frag)
                for v in rvars:
                    core.P_SIM = v[2]
                    core.CFG.SIM_POWER = v[2]
                    smis = harness.rank(st, ev, lib_gate=v[0], lib_hard=v[1])
                    results[(c, p, v)][key] = smis
                    rr = reciprocal_rank(truth[key], smis)
                    ranks[(c, p, v)][int(round(1 / rr)) if rr > 0 else 0] += 1
        if n % 25 == 0 or n == len(keys) - 1:
            print(f"  {n+1}/{len(keys)}  {time.time()-t0:.0f}s", flush=True)
            harness.save_frag_cache(st)

    print(f"\nstructures: {len(keys)}  seed {args.seed}  frag {'off' if args.no_frag else 'on'}  ranker {args.ranker or 'default'}  bio {args.bio}  models {args.models or 'public v2'}")
    print(f"{'class':<6}{'prune':<11}{'gate':<7}{'lib_hard':<10}{'p_sim':<7}{'MRR@25':<9}{'r1':<5}{'r2':<5}{'r3':<5}{'r4-5':<6}{'top25':<7}{'miss':<5}")
    for k in combos:
        c, p, v = k
        score = mrr_at_k(truth, results[k])
        h = ranks[k]
        top25 = sum(x for r, x in h.items() if r > 0)
        print(f"{c:<6}{f'{p[0]}:{p[1]}':<11}{str(v[0]):<7}{str(v[1]):<10}{str(v[2]):<7}{score:<9.4f}{h[1]:<5}{h[2]:<5}{h[3]:<5}"
              f"{h[4]+h[5]:<6}{top25:<7}{h[0]:<5}")
    print(f"total {time.time() - t0:.0f}s")


if __name__ == "__main__":
    main()
