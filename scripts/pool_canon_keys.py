"""Canonical (metric) InChIKey14 for every candidate pool structure, cached to data/cache/pool_key14.npz.

COCONUT and the training set standardise SMILES differently, so the same structure can carry two
key strings in the pool. This gives one key per pool row using the metric's own canonicalisation,
so duplicates can be merged and true COCONUT coverage of a structure set can be measured.

    python scripts/pool_canon_keys.py [--workers 4]
"""
import argparse
import os
import pickle
import sys
import time
from multiprocessing import Pool
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))


def _key(smi):
    from casmi.metric import key14
    return key14(smi)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--public", default=str(ROOT / "data" / "public"))
    ap.add_argument("--cache", default=str(ROOT / "data" / "cache"))
    ap.add_argument("--workers", type=int, default=4)
    args = ap.parse_args()
    t0 = time.time()
    d = Path(args.public) / "coconut-casmi26-candidates"
    cm = pickle.load(open(d / "coco_meta.pkl", "rb"))
    co_smis = list(cm["smiles"])
    z = np.load(Path(args.cache) / "train_fp.npz", allow_pickle=True)
    tr_smis = list(z["smiles"])
    smis = co_smis + tr_smis
    print(f"{len(co_smis):,} COCONUT + {len(tr_smis):,} train structures", flush=True)
    with Pool(args.workers) as p:
        keys = p.map(_key, smis, chunksize=2000)
    keys = np.asarray(keys, dtype=object)
    src = np.array([0] * len(co_smis) + [1] * len(tr_smis), dtype=np.int8)
    os.makedirs(args.cache, exist_ok=True)
    np.savez(Path(args.cache) / "pool_key14.npz", key14=keys, source=src, smiles=np.asarray(smis, dtype=object))
    bad = sum(k is None for k in keys)
    co = set(k for k in keys[: len(co_smis)] if k)
    tr = set(k for k in keys[len(co_smis):] if k)
    print(f"unparsable {bad}; distinct COCONUT {len(co):,}, distinct train {len(tr):,}, overlap {len(co & tr):,}  ({time.time()-t0:.0f}s)")


if __name__ == "__main__":
    main()
