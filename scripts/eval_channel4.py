"""Channel 4 alone: rank the mass-window candidates by fingerprint model score, no ranker.

The bar an own fingerprint model has to clear. Scores candidates inside the 8.5 ppm window by
the public pipeline's normalised dot product (bits dot logits over sqrt of bit count) and by the
Bernoulli log-likelihood, on np-examples under class 2 (own spectra hidden, which changes nothing
for channel 4 but keeps the pool identical) and class 3 (training copy removed from the pool).

    python scripts/eval_channel4.py [--models path/to/dir_with_fp_*.pt]
"""
import argparse
import sys
import time
from collections import Counter
from pathlib import Path

import numpy as np
import pyarrow.compute as pc
import pyarrow.parquet as pq

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", default=str(ROOT / "data"))
    ap.add_argument("--public", default=str(ROOT / "data" / "public"))
    ap.add_argument("--cache", default=str(ROOT / "data" / "cache"))
    ap.add_argument("--models", default=None, help="folder with fp_*.pt to evaluate instead of the public ones")
    ap.add_argument("--limit", type=int, default=None)
    args = ap.parse_args()

    from casmi.fork import core, harness
    from casmi.fork.core import CFG
    from casmi.metric import key14

    t0 = time.time()
    st = harness.init(args.data, args.public, args.cache, frag_pool=False)
    if args.models:
        core.SEARCH_ROOTS = [args.models]
        core.load_neural_models()
        core.SEARCH_ROOTS = [args.data, args.public]
    pool = st.pool
    # canonical key per pool row, aligned with the mass-sorted pool
    z = np.load(Path(args.cache) / "pool_key14.npz", allow_pickle=True)
    co_mass = np.load(Path(args.public) / "coconut-casmi26-candidates" / "coco_mass.npy")
    tr_mass = np.load(Path(args.cache) / "train_fp.npz", allow_pickle=True)["mass"]
    mass = np.concatenate([co_mass, tr_mass])
    good = np.isfinite(mass)
    pool_k14 = z["key14"][good][np.argsort(mass[good])]
    assert len(pool_k14) == len(pool.mass)

    cols = ["inchikey14", "normalized_smiles", "ingest_lib", "precursor_mz", "adduct", "ionization_mode",
            "instrument_type", "collision_energy_ev", "ms2_mzs", "ms2_normalized_intensities"]
    tb = pq.read_table(Path(args.data) / "train.parquet", columns=cols)
    npx = tb.filter(pc.equal(tb["ingest_lib"], "enveda-np-examples")).to_pandas()
    keys = sorted(npx.inchikey14.unique())[: args.limit] if args.limit else sorted(npx.inchikey14.unique())

    scorers = {
        "dot_norm": lambda F, zl: (F @ zl) / np.sqrt(np.maximum(F.sum(1), 1.0)),
        # log-likelihood of the candidate bits under independent Bernoulli bits with p = sigmoid(logit):
        # sum_i [b_i log p_i + (1 - b_i) log(1 - p_i)] = bits dot logits + sum log(1 - p)
        "bernoulli": lambda F, zl: F @ zl + np.log(1.0 - 1.0 / (1.0 + np.exp(-np.clip(zl, -30, 30))) + 1e-7).sum(),
    }
    ranks = {(c, s): Counter() for c in (2, 3) for s in scorers}
    n_eval = 0
    for n, key in enumerate(keys):
        sub = npx[npx.inchikey14 == key].reset_index(drop=True)
        truth_k = key14(sub.normalized_smiles.iloc[0])
        nm = core.neutral_mass(sub.precursor_mz.values.astype(np.float64), sub.adduct.values)
        nms = nm[np.isfinite(nm)]
        if not len(nms):
            continue
        target = float(np.median(nms))
        zl = core.model_logits(sub)
        if zl is None:
            continue
        zl = np.asarray(zl, np.float32)
        n_eval += 1
        for c in (2, 3):
            cand = pool.window(target, CFG.PPM_WIN)
            if len(cand) == 0:
                cand = pool.window(target, CFG.PPM_FALLBACK)
            if c == 3 and len(cand):
                cand = cand[pool.keys[cand] != key]
            if not len(cand):
                for s in scorers:
                    ranks[(c, s)][0] += 1
                continue
            F = pool.fps(cand).astype(np.float32)
            is_truth = pool_k14[cand] == truth_k
            for s, fn in scorers.items():
                sc = fn(F, zl)
                order = np.argsort(-sc)
                pos = np.flatnonzero(is_truth[order])
                r = int(pos[0]) + 1 if len(pos) else 0
                ranks[(c, s)][r if 0 < r <= 25 else 0] += 1
        if n % 50 == 0:
            print(f"  {n+1}/{len(keys)}  {time.time()-t0:.0f}s", flush=True)

    print(f"\nchannel 4 alone on np-examples, {n_eval} structures, models: {args.models or 'public'}")
    print(f"{'class':<6}{'scorer':<11}{'MRR@25':<9}{'r1':<5}{'r2-5':<6}{'top25':<7}{'miss':<5}")
    for (c, s), h in ranks.items():
        mrr = sum(v / r for r, v in h.items() if r > 0) / max(n_eval, 1)
        top25 = sum(v for r, v in h.items() if r > 0)
        print(f"{c:<6}{s:<11}{mrr:<9.4f}{h[1]:<5}{sum(h[r] for r in (2,3,4,5)):<6}{top25:<7}{h[0]:<5}")
    print(f"total {time.time()-t0:.0f}s")


if __name__ == "__main__":
    main()
