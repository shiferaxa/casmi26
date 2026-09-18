"""Train our own ranker on rows from dump_rank_features.py and compare with the public one.

Rows are grouped by structure; a fraction of structures is held out. The model is the same
HistGradientBoosting ensemble the kernel fits (so the output file drops into the kernel as
rank_train.npz), evaluated by MRR@25 over held-out structures, next to the public v4 rows
fitted the same way and scored on the same held-out rows.

    python scripts/train_ranker.py --rows artifacts/rank_rows_c12.npz artifacts/rank_rows_c3.npz
    python scripts/train_ranker.py --rows ... --out artifacts/rank_train_ours.npz   # write the kernel file
"""
import argparse
import sys
import time
from pathlib import Path

import numpy as np
from sklearn.ensemble import HistGradientBoostingClassifier

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

GBM = dict(max_depth=6, max_iter=500, learning_rate=0.03, min_samples_leaf=80, l2_regularization=1.0)
W1_PRIORS = (0.30, 0.60)
SEEDS = (0, 1, 2, 3)


def load_rows(paths):
    Xs, ys, gs, cs = [], [], [], []
    offset = 0
    for p in paths:
        z = np.load(p, allow_pickle=True)
        Xs.append(z["X"]); ys.append(z["y"]); gs.append(z["group"] + offset); cs.append(z["cls"])
        offset += int(z["group"].max()) + 1
        print(f"{p}: rows {len(z['y']):,} positives {int(z['y'].sum())} structures {len(np.unique(z['group']))}")
    return np.vstack(Xs), np.concatenate(ys), np.concatenate(gs), np.concatenate(cs)


def fit_ensemble(X, y, M, seeds=SEEDS, priors=W1_PRIORS):
    models = []
    for w1 in priors:
        W = np.where(M == 0, w1, 1.0 - w1)
        for sd in seeds:
            m = HistGradientBoostingClassifier(random_state=sd, **GBM)
            m.fit(X, y, sample_weight=W)
            models.append(m)
    return models


def predict(models, X):
    return np.mean([m.predict_proba(X)[:, 1] for m in models], axis=0)


def mrr_by_group(p, y, g, cls, k=25):
    """MRR@25 per class over groups: rank of the first positive within the group by descending p."""
    out = {}
    for c in np.unique(cls):
        rr = []
        for gid in np.unique(g[cls == c]):
            m = (g == gid) & (cls == c)
            order = np.argsort(-p[m])
            yy = y[m][order]
            pos = np.flatnonzero(yy)
            rr.append(1.0 / (pos[0] + 1) if len(pos) and pos[0] < k else 0.0)
        out[int(c)] = (float(np.mean(rr)), len(rr))
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--rows", nargs="+", required=True)
    ap.add_argument("--public", default=str(ROOT / "artifacts" / "ranker_versions" / "v4" / "rank_train.npz"))
    ap.add_argument("--holdout", type=float, default=0.2)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--out", default=None, help="write all rows as a kernel-compatible rank_train.npz")
    args = ap.parse_args()

    t0 = time.time()
    X, y, g, cls = load_rows(args.rows)
    M = (cls != 1).astype(np.int8)  # public convention: M == 0 gets the class-1 prior weight
    rng = np.random.default_rng(args.seed)
    structures = np.unique(g)
    held = set(rng.choice(structures, size=int(len(structures) * args.holdout), replace=False))
    te = np.array([x in held for x in g])
    tr = ~te
    print(f"train rows {tr.sum():,} ({len(np.unique(g[tr]))} structures)  held-out rows {te.sum():,} ({len(held)} structures)")

    ours = fit_ensemble(X[tr], y[tr], M[tr])
    p_ours = predict(ours, X[te])
    print(f"ours   : {mrr_by_group(p_ours, y[te], g[te], cls[te])}   ({time.time()-t0:.0f}s)")

    z = np.load(args.public)
    nfeat = z["X"].shape[1]
    pub = fit_ensemble(z["X"], z["Y"], z["M"])
    p_pub = predict(pub, X[te][:, :nfeat])
    print(f"public : {mrr_by_group(p_pub, y[te], g[te], cls[te])}")

    both = fit_ensemble(np.vstack([X[tr][:, :nfeat], z["X"]]), np.concatenate([y[tr], z["Y"]]), np.concatenate([M[tr], z["M"]]))
    p_both = predict(both, X[te][:, :nfeat])
    print(f"ours+public rows: {mrr_by_group(p_both, y[te], g[te], cls[te])}")

    # seed variance of ours on the held-out rows
    per_seed = []
    for sd in SEEDS:
        m = fit_ensemble(X[tr], y[tr], M[tr], seeds=(sd,), priors=(0.5,))
        per_seed.append(mrr_by_group(predict(m, X[te]), y[te], g[te], cls[te]))
    for c in sorted(per_seed[0]):
        vals = [s[c][0] for s in per_seed]
        print(f"class {c} single-seed MRR: mean {np.mean(vals):.4f} sd {np.std(vals):.4f}")

    if args.out:
        np.savez(args.out, X=X.astype(np.float32), Y=y.astype(np.int8), M=M.astype(np.int8))
        print(f"wrote {args.out}: {len(y):,} rows, {X.shape[1]} features")
    print(f"total {time.time()-t0:.0f}s")


if __name__ == "__main__":
    main()
