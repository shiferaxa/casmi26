"""Local driver for the transplanted public pipeline (core.py).

init() loads every asset from local folders, caching the expensive train-structure
fingerprints and the fragment enumerations. Scoring is staged so variants share work:
  channels()   library search, analog search and model logits for one molecule (expensive,
               independent of candidate selection)
  evidence()   candidate window, prune, fingerprints, fragmentation scores for one prune config
  rank()       features and ranker under one ranking variant, returns 25 SMILES

Masks for validation:
  hide_key        structure key whose spectra are hidden from the library and analog channels (class 2)
  drop_from_pool  also remove that structure key from the candidate pool (class 3)
Prune variants (the public pipeline uses 80 and mode "lib"):
  prune_n         candidates kept when the mass window is larger than this
  prune_mode      "lib"   library similarity times 100 minus mass difference (public pipeline)
                  "model" fingerprint dot model logits, library hits always kept
Ranking variants:
  lib_gate        if the best library similarity is below this, library evidence is zeroed before ranking
  lib_hard        library hits with similarity at or above this are forced to the top
"""
from __future__ import annotations

import os
import pickle
import time
from dataclasses import dataclass
from multiprocessing import Pool as MPool

import numpy as np
import pandas as pd
import pyarrow.parquet as pq

from . import core
from .core import CFG


@dataclass
class State:
    L: dict
    rep: np.ndarray
    rep_key: np.ndarray
    rep_nm: np.ndarray
    pool: core.CandidatePool
    train_path: str
    cache_root: str


@dataclass
class Channels:
    target: float
    specs: list
    lib_hits: dict
    analogs: list
    zlog: np.ndarray | None
    ion_mode: float


@dataclass
class Evidence:
    target: float
    cand: np.ndarray          # pool indices after prune
    cfp: np.ndarray | None
    lv: np.ndarray
    afp: np.ndarray | None
    sims: np.ndarray
    fsc: np.ndarray
    zlog: np.ndarray | None
    n_window: int


def _worker_init(bits):
    """Windows spawns fresh worker processes, so module globals set in the parent must be passed in."""
    core.BITS = bits


def _cached_train_fps(train_path: str, cache_path: str, coco_keys: set, workers: int):
    if os.path.exists(cache_path):
        z = np.load(cache_path, allow_pickle=True)
        return z["fp"], z["mass"], z["keys"], z["smiles"]
    tr = pq.read_table(train_path, columns=["inchikey14", "normalized_smiles"]).to_pandas()
    tr = tr.dropna().drop_duplicates("inchikey14")
    tr = tr[~tr.inchikey14.isin(coco_keys)]
    print(f"[INFO] fingerprinting {len(tr):,} training structures with {workers} workers (cached afterwards)", flush=True)
    t0 = time.time()
    with MPool(workers, initializer=_worker_init, initargs=(core.BITS,)) as mp:
        res = mp.map(core.fp_and_mass, list(tr.normalized_smiles), chunksize=500)
    ok = [i for i, r in enumerate(res) if r is not None]
    fp = np.packbits(np.stack([res[i][0] for i in ok]), axis=1)
    assert fp.ndim == 2 and fp.shape[1] == (len(core.BITS) + 7) // 8, f"bad fingerprint shape {fp.shape}"
    mass = np.array([res[i][1] for i in ok])
    keys = tr.inchikey14.values[ok].astype(object)
    smiles = tr.normalized_smiles.values[ok].astype(object)
    os.makedirs(os.path.dirname(cache_path), exist_ok=True)
    np.savez(cache_path, fp=fp, mass=mass, keys=keys, smiles=smiles)
    print(f"[INFO] fingerprints done in {time.time()-t0:.0f}s", flush=True)
    return fp, mass, keys, smiles


def _frag_cache_path(cache_root: str) -> str:
    return os.path.join(cache_root, "frag_cache.pkl")


def save_frag_cache(st: State) -> None:
    os.makedirs(st.cache_root, exist_ok=True)
    with open(_frag_cache_path(st.cache_root), "wb") as f:
        pickle.dump(core.FRAG_CACHE, f, protocol=pickle.HIGHEST_PROTOCOL)


def init(data_root: str, public_root: str, cache_root: str, workers: int | None = None, frag_pool: bool = True,
         ranker_path: str | None = None, use_bio: bool = False) -> State:
    t0 = time.time()
    workers = workers or max(2, (os.cpu_count() or 4) - 2)
    core.SEARCH_ROOTS = [data_root, public_root]
    train_path = core.find_file("train.parquet")
    core.BITS = np.load(core.find_file("fp_bits.npy"))
    ranker_path = ranker_path or core.find_file("rank_train.npz")
    print(f"[INFO] ranker training file: {ranker_path}", flush=True)
    core.fit_rankers(ranker_path)
    core.load_neural_models()

    d = os.path.dirname(core.find_file("coco_fp.npy"))
    cm = pickle.load(open(os.path.join(d, "coco_meta.pkl"), "rb"))
    co_fp = np.load(os.path.join(d, "coco_fp.npy"))
    co_mass = np.load(os.path.join(d, "coco_mass.npy"))
    co_keys = np.asarray(cm["keys"], dtype=object)
    co_smis = np.asarray(cm["smiles"], dtype=object)
    coco_keys_set = set(co_keys)
    if use_bio:
        bd = os.path.dirname(core.find_file("bio_fp.npy"))
        bm = pickle.load(open(os.path.join(bd, "bio_meta.pkl"), "rb"))
        bi_keys = np.asarray(bm["keys"], dtype=object)
        new = np.array([k not in coco_keys_set for k in bi_keys])
        co_fp = np.vstack([co_fp, np.load(os.path.join(bd, "bio_fp.npy"))[new]])
        co_mass = np.concatenate([co_mass, np.load(os.path.join(bd, "bio_mass.npy"))[new]])
        co_keys = np.concatenate([co_keys, bi_keys[new]])
        co_smis = np.concatenate([co_smis, np.asarray(bm["smiles"], dtype=object)[new]])
        coco_keys_set = set(co_keys)
        print(f"[INFO] ChEBI/LIPID MAPS block: {int(new.sum()):,} new structures", flush=True)
    tr_fp, tr_mass, tr_keys, tr_smi = _cached_train_fps(train_path, os.path.join(cache_root, "train_fp.npz"), coco_keys_set, workers)
    fp = np.vstack([co_fp, tr_fp])
    mass = np.concatenate([co_mass, tr_mass])
    keys = np.concatenate([co_keys, tr_keys])
    smis = np.concatenate([co_smis, tr_smi])
    good = np.isfinite(mass)
    pool = core.CandidatePool(fp[good], mass[good], keys[good], smis[good], cm["nbits"])
    pool.coconut_keys = coco_keys_set
    print(f"[INFO] candidate pool: {len(pool.mass):,} structures ({time.time()-t0:.0f}s)", flush=True)

    L = core.load_library(train_path)
    rep, rep_key, rep_nm = core.build_rep(L)
    fc = _frag_cache_path(cache_root)
    if os.path.exists(fc):
        with open(fc, "rb") as f:
            core.FRAG_CACHE = pickle.load(f)
        print(f"[INFO] fragment cache: {len(core.FRAG_CACHE):,} structures", flush=True)
    if frag_pool:
        core.FRAG_POOL = MPool(workers)
    print(f"[INFO] init done in {time.time()-t0:.0f}s", flush=True)
    return State(L, rep, rep_key, rep_nm, pool, train_path, cache_root)


def channels(st: State, sub: pd.DataFrame, hide_key: str | None = None) -> Channels | None:
    """Library search, analog search and model logits for one molecule."""
    nm = core.neutral_mass(sub.precursor_mz.values.astype(np.float64), sub.adduct.values)
    nms = nm[np.isfinite(nm)]
    if not len(nms):
        return None
    target = float(np.median(nms))
    specs = [(r.ms2_mzs, r.ms2_normalized_intensities) for r in sub.itertuples()]
    lib_hits = core.lib_sim(st.L, specs, target, hide_key=hide_key)
    analogs = core.analog_sim(st.L, specs, target, st.rep, st.rep_key, st.rep_nm, hide_key=hide_key)
    zlog = core.model_logits(sub)
    ion_mode = float(np.mean([1.0 if m == "positive" else -1.0 for m in sub.ionization_mode]))
    return Channels(target, specs, lib_hits, analogs, zlog, ion_mode)


def evidence(st: State, ch: Channels | None, drop_key: str | None = None, prune_n: int = 80,
             prune_mode: str = "lib", use_frag: bool = True) -> Evidence | None:
    """Candidate window, prune, fingerprints and fragmentation for one prune config."""
    if ch is None:
        return None
    pool = st.pool
    target = ch.target
    cand = pool.window(target, CFG.PPM_WIN)
    if len(cand) == 0:
        cand = pool.window(target, CFG.PPM_FALLBACK)
    if drop_key is not None and len(cand):
        cand = cand[pool.keys[cand] != drop_key]
    n_window = int(len(cand))
    if not len(cand):
        return Evidence(target, cand, None, np.zeros(0, np.float32), None, np.zeros(0, np.float32),
                        np.zeros(0, np.float32), ch.zlog, 0)
    lv_all = np.array([ch.lib_hits.get(pool.keys[c], 0.0) for c in cand], np.float32)
    if len(cand) > prune_n:
        if prune_mode == "lib":
            mass_diff = np.abs(pool.mass[cand] - target)
            cand = cand[np.argsort(-(lv_all * 100.0 - mass_diff))[:prune_n]]
        elif prune_mode == "model":
            if ch.zlog is None:
                mass_diff = np.abs(pool.mass[cand] - target)
                score = -mass_diff
            else:
                fps_all = pool.fps(cand).astype(np.float32)
                score = fps_all @ np.asarray(ch.zlog, np.float32)
                score = score / np.sqrt(np.maximum(fps_all.sum(1), 1.0))
            score = score + np.where(lv_all > 0, 1e6, 0.0)  # library hits are always kept
            cand = cand[np.argsort(-score)[:prune_n]]
        else:
            raise ValueError(prune_mode)
    cfp = pool.fps(cand)
    lv = np.array([ch.lib_hits.get(pool.keys[c], 0.0) for c in cand], np.float32)
    ids, sims = [], []
    for k, s in ch.analogs:
        i = pool.k2i.get(k, -1)
        if i >= 0:
            ids.append(i)
            sims.append(s)
    afp = pool.fps(np.array(ids)) if ids else None
    if use_frag:
        fsc = core.frag_scores([pool.smiles[c] for c in cand], ch.specs, ch.ion_mode)
    else:
        fsc = np.zeros(len(cand), np.float32)
    return Evidence(target, cand, cfp, lv, afp, np.array(sims, np.float32), fsc, ch.zlog, n_window)


def rank(st: State, ev: Evidence | None, lib_gate: float | None = None, lib_hard: float | None = None) -> list[str]:
    """Rank the evidence under one variant and return 25 deduplicated SMILES."""
    if ev is None:
        return ["CCO"]
    pool = st.pool
    smis: list[str] = []
    if len(ev.cand):
        lv = ev.lv
        if lib_gate is not None and lv.max() < lib_gate:
            lv = np.zeros_like(lv)
        X = core.rank_features(ev.cfp, lv, ev.afp, ev.sims, ev.zlog, ev.fsc)[:, :core.NFEAT]
        score = core.rank_proba(X)
        if lib_hard is not None:
            score = score + np.where(lv >= lib_hard, 10.0 + lv, 0.0)
        order = np.argsort(-score)
        smis = core.dedupe_fill([pool.smiles[ev.cand[i]] for i in order], pool, ev.target, CFG.TOPN)
    if len(smis) < CFG.TOPN:
        smis = core.dedupe_fill(smis, pool, ev.target, CFG.TOPN)
    return smis or ["CCO"]


def score_molecule(st: State, sub: pd.DataFrame, hide_key: str | None = None, drop_from_pool: bool = False,
                   lib_gate: float | None = None, lib_hard: float | None = None, use_frag: bool = True,
                   prune_n: int = 80, prune_mode: str = "lib"):
    """Single-variant convenience wrapper."""
    ch = channels(st, sub, hide_key)
    ev = evidence(st, ch, drop_key=hide_key if drop_from_pool else None, prune_n=prune_n,
                  prune_mode=prune_mode, use_frag=use_frag)
    diag = dict(n_cand=0 if ev is None else int(len(ev.cand)),
                lv_max=0.0 if ev is None or not len(ev.lv) else float(ev.lv.max()))
    return rank(st, ev, lib_gate, lib_hard), diag
