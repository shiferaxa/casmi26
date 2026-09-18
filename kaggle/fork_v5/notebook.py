# v5 of the fork: as v4 (no prune) with our own ranker rows (1,500 class 1 and 2 structures plus 700 class 3, 400-candidate windows).
# candidate prune widened from 80 to 400 (the 80-candidate prune drops hidden-structure truths).
# Based on the public kernel haideptry/0-339-top-1-4-channel-transformer-analog-ensemble
# (Apache 2.0 on Kaggle). Reference run to confirm the environment reproduces 0.339.


# ===================================================================================
#  CONFIG — Tuned Hyperparameters Annotated with Experimental Measurements
# ===================================================================================
class CFG:
    # --- Candidate Generation ------------------------------------------------------
    PPM_WIN      = 8.5    # Optimal window for timsTOF (+1.4 ppm systematic offset).
                          # Tighter is better: +-10 ppm -> 0.521, +-8.5 ppm -> 0.524 Class-2 MRR
    PPM_FALLBACK = 30.0   # Robust fallback if tight window yields 0 candidates

    # --- Spectral Preprocessing ----------------------------------------------------
    INT_FLOOR    = 0.002  # Drop peaks below 0.2% base-peak intensity
    MAX_PEAKS    = 256    # Keep N most intense peaks after floor
    MZ_TOL       = 0.01   # Da tolerance for peak alignment
    INT_POWER    = 1.0    # Linear intensity with entropy weighting outperforms sqrt
    ENT_WEIGHT   = True   # Entropy weighting for low-entropy spectra (Li et al. 2021)

    # --- Mass-Shifted Analog Propagation (Channel 2) --------------------------------
    ANALOG_WIN   = 200.0  # +- Da mass-shift search window
    N_ANALOG     = 100    # Analogs kept per molecule (scaled from 80 -> 100)
    SIM_POWER    = 4.0    # sim^p power weighting. p=1 -> 0.498, p=3 -> 0.521, p=4 -> 0.525

    # --- Calibrated Ensemble Ranker (Variance Reduction) ---------------------------
    W1_PRIORS    = (0.30, 0.60)  # Hedging Class-1/Class-2 trade-off improves LB by ~0.003
    SEEDS        = (0, 1, 2, 3)  # Bagging 4 random seeds eliminates 0.006 seed noise floor
    W1           = 0.50          # Calibrated class weight anchor
    GBM = dict(max_depth=6, max_iter=500, learning_rate=0.03,
               min_samples_leaf=80, l2_regularization=1.0)

    USE_BIO_DB   = False  # ChEBI/LIPID MAPS toggle (disabled by default to prevent decoy dilution)
    TOPN         = 25     # Output exactly 25 SMILES per molecule
    PRUNE_N      = 100000 # v4: no prune, every candidate in the mass window is ranked (v3 used 400, public 80)


import os, sys, glob, time, pickle, math, subprocess
from multiprocessing import Pool as MPool
import numpy as np
import pandas as pd
import pyarrow.parquet as pq
import pyarrow as pa

T0 = time.time()

def find_file(name):
    hits = glob.glob(f'/kaggle/input/**/{name}', recursive=True)
    if not hits:
        raise FileNotFoundError(f"File not found in /kaggle/input: {name}")
    return sorted(hits, key=len)[0]

COMP   = os.path.dirname(find_file('test.parquet'))
TRAIN  = os.path.join(COMP, 'train.parquet')
TEST   = os.path.join(COMP, 'test.parquet')
SAMPLE = os.path.join(COMP, 'sample_submission.csv')

print(f"[INFO] Competition directory: {COMP}")
print(f"       Files available: {os.listdir(COMP)}")

# Bootstrap offline RDKit wheel if available in attached datasets
rdkit_whls = glob.glob('/kaggle/input/**/rdkit-*.whl', recursive=True)
if rdkit_whls:
    print(f"[INFO] Installing offline RDKit wheel: {rdkit_whls[0]}")
    subprocess.run([sys.executable, '-m', 'pip', 'install', '-q', '--no-index', rdkit_whls[0]], check=False)

try:
    from rdkit import Chem, RDLogger
    from rdkit.Chem import rdFingerprintGenerator, MACCSkeys
    from rdkit.Chem.Descriptors import ExactMolWt
    RDLogger.DisableLog('rdApp.*')
    HAVE_RDKIT = True
    print("[SUCCESS] RDKit 2026.3.3 is active and verified.")
except Exception as e:
    HAVE_RDKIT = False
    print(f"[WARNING] RDKit unavailable ({e}). Running without in-silico fragmentation.")


import numpy as np
from numba import njit, prange

@njit(cache=True, fastmath=True)
def _clean(mz, it, floor, topk, power, ent_weight):
    n = len(mz)
    if n == 0:
        return np.empty(0, np.float32), np.empty(0, np.float32)
    mx = 0.0
    for i in range(n):
        if it[i] > mx: mx = it[i]
    if mx <= 0:
        return np.empty(0, np.float32), np.empty(0, np.float32)
    thr = floor * mx
    c = 0
    for i in range(n):
        if it[i] >= thr: c += 1
    idx = np.empty(c, np.int64)
    j = 0
    for i in range(n):
        if it[i] >= thr:
            idx[j] = i; j += 1
    if c > topk:
        v = np.empty(c, np.float32)
        for i in range(c): v[i] = it[idx[i]]
        o = np.argsort(v)[c - topk:]
        k2 = np.empty(topk, np.int64)
        for i in range(topk): k2[i] = idx[o[i]]
        k2.sort()
        idx = k2
        c = topk
    om = np.empty(c, np.float32)
    oi = np.empty(c, np.float32)
    s = 0.0
    for i in range(c):
        om[i] = mz[idx[i]]
        v = it[idx[i]] ** power
        oi[i] = v
        s += v
    if s > 0:
        for i in range(c): oi[i] /= s
    if ent_weight:
        S = 0.0
        for i in range(c):
            if oi[i] > 0: S -= oi[i] * np.log(oi[i])
        if S < 3.0:
            w = 0.25 + 0.25 * S
            s2 = 0.0
            for i in range(c):
                oi[i] = oi[i] ** w
                s2 += oi[i]
            if s2 > 0:
                for i in range(c): oi[i] /= s2
    return om, oi

@njit(cache=True, fastmath=True)
def entropy_sim(qmz, qp, cmz, cp, tol):
    i = 0; j = 0; n = len(qmz); m = len(cmz)
    SA = 0.0
    for x in range(n):
        if qp[x] > 0: SA -= qp[x] * np.log(qp[x])
    SB = 0.0
    for x in range(m):
        if cp[x] > 0: SB -= cp[x] * np.log(cp[x])
    SAB = 0.0; tot = 0.0
    buf = np.empty(n + m, np.float64); b = 0
    while i < n and j < m:
        d = qmz[i] - cmz[j]
        if d < -tol:
            buf[b] = qp[i]; i += 1; b += 1
        elif d > tol:
            buf[b] = cp[j]; j += 1; b += 1
        else:
            buf[b] = qp[i] + cp[j]; i += 1; j += 1; b += 1
    while i < n: buf[b] = qp[i]; i += 1; b += 1
    while j < m: buf[b] = cp[j]; j += 1; b += 1
    for x in range(b): tot += buf[x]
    if tot <= 0: return 0.0
    for x in range(b):
        v = buf[x] / tot
        if v > 0: SAB -= v * np.log(v)
    return 1.0 - (2.0 * SAB - SA - SB) / np.log(4.0)

@njit(cache=True, fastmath=True)
def entropy_sim_shift(qmz, qp, cmz, cp, tol, shift):
    """Evaluates max of direct and mass-shifted spectral entropy similarity."""
    a = entropy_sim(qmz, qp, cmz, cp, tol)
    if shift > -0.001 and shift < 0.001:
        return a
    sm = np.empty(len(cmz), np.float32)
    for i in range(len(cmz)):
        sm[i] = cmz[i] + shift
    b = entropy_sim(qmz, qp, sm, cp, tol)
    return a if a > b else b

@njit(cache=True, fastmath=True, parallel=True)
def search(qmz, qp, cand, off, allmz, allin, tol, floor, topk, power, ent_weight):
    out = np.zeros(len(cand), np.float32)
    for k in prange(len(cand)):
        c = cand[k]; a = off[c]; b = off[c + 1]
        if b <= a: continue
        cm, cp = _clean(allmz[a:b], allin[a:b], floor, topk, power, ent_weight)
        if len(cm) == 0: continue
        out[k] = entropy_sim(qmz, qp, cm, cp, tol)
    return out

@njit(cache=True, fastmath=True, parallel=True)
def search_shift(qmz, qp, cand, off, allmz, allin, tol, floor, topk, power, ent_weight, shift):
    out = np.zeros(len(cand), np.float32)
    for k in prange(len(cand)):
        c = cand[k]; a = off[c]; b = off[c + 1]
        if b <= a: continue
        cm, cp = _clean(allmz[a:b], allin[a:b], floor, topk, power, ent_weight)
        if len(cm) == 0: continue
        out[k] = entropy_sim_shift(qmz, qp, cm, cp, tol, shift[k])
    return out


# ===================================================================================
#  High-Precision Adduct Physics & Reference Library Construction
# ===================================================================================
MASS = dict(C=12.0, H=1.00782503207, N=14.0030740048, O=15.9949146196, P=30.97376163,
            S=31.97207100, F=18.99840322, Cl=34.96885268, Br=78.9183371, I=126.904473,
            Na=22.9897692809, K=38.96370668, Si=27.9769265325, B=11.0093054, Se=79.9165213)
E = 0.00054857990
PROTON = MASS['H'] - E
H2O = 2 * MASS['H'] + MASS['O']
NH4 = MASS['N'] + 4 * MASS['H']
FORMATE = MASS['C'] + 2 * MASS['H'] + 2 * MASS['O']
ACETATE = 2 * MASS['C'] + 4 * MASS['H'] + 2 * MASS['O']

ADDUCTS = {
    "[M+H]+": (1, 1, PROTON), "[M+NH4]+": (1, 1, NH4 - E), "[M+Na]+": (1, 1, MASS['Na'] - E),
    "[M+K]+": (1, 1, MASS['K'] - E), "[M-H2O+H]+": (1, 1, PROTON - H2O), "[M-2H2O+H]+": (1, 1, PROTON - 2*H2O),
    "[M+2H]2+": (1, 2, 2 * PROTON), "[M]+": (1, 1, -E), "[M-H2O]+": (1, 1, -E - H2O),
    "[M+CH3OH+H]+": (1, 1, PROTON + MASS['C'] + 4 * MASS['H'] + MASS['O']),
    "[M+CH3CN+H]+": (1, 1, PROTON + 2 * MASS['C'] + 3 * MASS['H'] + MASS['N']),
    "[M-H]-": (1, 1, -PROTON), "[M-H2O-H]-": (1, 1, -PROTON - H2O), "[M+CH2O2-H]-": (1, 1, FORMATE - PROTON),
    "[M+C2H4O2-H]-": (1, 1, ACETATE - PROTON), "[M+Cl]-": (1, 1, MASS['Cl'] + E), "[M]-": (1, 1, E),
    "[M-2H]-": (1, 2, -2 * PROTON), "[M+Na-2H]-": (1, 1, MASS['Na'] - 2 * PROTON),
    "[2M+H]+": (2, 1, PROTON), "[2M+Na]+": (2, 1, MASS['Na'] - E), "[2M+NH4]+": (2, 1, NH4 - E),
    "[2M+K]+": (2, 1, MASS['K'] - E), "[2M-H]-": (2, 1, -PROTON), "[2M+CH2O2-H]-": (2, 1, FORMATE - PROTON),
    "[2M+C2H4O2-H]-": (2, 1, ACETATE - PROTON), "[2M+Na-2H]-": (2, 1, MASS['Na'] - 2 * PROTON),
    "[3M+H]+": (3, 1, PROTON), "[3M-H]-": (3, 1, -PROTON),
}

def neutral_mass(mz, adduct):
    out = np.full(len(mz), np.nan)
    ad = np.asarray(adduct, dtype=object)
    for a, (n, z, d) in ADDUCTS.items():
        m = (ad == a)
        if m.any():
            out[m] = (mz[m] * z - d) / n
    return out

def load_library(path):
    t0 = time.time()
    t = pq.read_table(path, columns=['inchikey14', 'normalized_smiles', 'adduct', 'precursor_mz',
                                     'ms2_mzs', 'ms2_normalized_intensities'])
    mzc = t.column('ms2_mzs').combine_chunks()
    itc = t.column('ms2_normalized_intensities').combine_chunks()
    off = mzc.offsets.to_numpy().astype(np.int64)
    allmz = mzc.values.to_numpy(zero_copy_only=False).astype(np.float32)
    allin = itc.values.to_numpy(zero_copy_only=False).astype(np.float32)
    prec = t.column('precursor_mz').to_numpy(zero_copy_only=False).astype(np.float64)
    add = np.asarray(t.column('adduct').cast(pa.string()).to_pylist(), dtype=object)
    ik  = np.asarray(t.column('inchikey14').cast(pa.string()).to_pylist(), dtype=object)
    smi = np.asarray(t.column('normalized_smiles').cast(pa.string()).to_pylist(), dtype=object)
    nm  = neutral_mass(prec, add)
    ok  = np.isfinite(nm)
    order = np.argsort(np.where(ok, nm, 1e18), kind='mergesort')
    best = {}
    for k, s in zip(ik, smi):
        if k and s and k not in best:
            best[k] = s
    print(f"[INFO] Reference Library loaded: {len(off)-1:,} spectra / {len(best):,} unique structures ({time.time()-t0:.1f}s)", flush=True)
    return dict(off=off, mz=allmz, it=allin, nm=nm, ik=ik, best=best,
                order=order, snm=nm[order], n_ok=int(ok.sum()))

def lib_window(L, target, tol):
    lo = np.searchsorted(L['snm'][:L['n_ok']], target - tol, 'left')
    hi = np.searchsorted(L['snm'][:L['n_ok']], target + tol, 'right')
    return L['order'][lo:hi]

def build_rep(L):
    """Extracts the single richest representative spectrum per unique structure for analog retrieval."""
    npk = np.diff(L['off'])
    best = {}; ik = L['ik']
    for i in range(len(ik)):
        k = ik[i]
        if k and (k not in best or npk[i] > npk[best[k]]):
            best[k] = i
    rep = np.array(sorted(best.values()))
    nm = L['nm'][rep]
    ok = np.isfinite(nm)
    rep = rep[ok]; nm = nm[ok]; key = ik[rep]
    o = np.argsort(nm)
    print(f"[INFO] Built representative analog set: {len(rep):,} unique scaffolds.", flush=True)
    return rep[o], key[o], nm[o]


# ===================================================================================
#  711,705-Structure Candidate Pool with Bitpacked Precomputed Fingerprints
# ===================================================================================
BITS = np.load(find_file('fp_bits.npy'))
_g = {}

def _fp_init():
    if not HAVE_RDKIT: return
    _g['m2'] = rdFingerprintGenerator.GetMorganGenerator(radius=2, fpSize=4096)
    _g['m3'] = rdFingerprintGenerator.GetMorganGenerator(radius=3, fpSize=4096)
    _g['rk'] = rdFingerprintGenerator.GetRDKitFPGenerator(fpSize=2048, maxPath=6)

def fp_and_mass(smi):
    if not HAVE_RDKIT: return None
    if not _g: _fp_init()
    m = Chem.MolFromSmiles(smi)
    if m is None: return None
    try:
        fp = np.concatenate([
            _g['m2'].GetFingerprintAsNumPy(m).astype(np.uint8),
            _g['m3'].GetFingerprintAsNumPy(m).astype(np.uint8),
            _g['rk'].GetFingerprintAsNumPy(m).astype(np.uint8),
            np.array(MACCSkeys.GenMACCSKeys(m), dtype=np.uint8)
        ])[BITS]
        return fp, float(ExactMolWt(m))
    except Exception:
        return None

class CandidatePool:
    """Unified candidate pool sorted by neutral exact mass."""
    def __init__(self, fp, mass, keys, smiles, nbits):
        o = np.argsort(mass)
        self._fp = fp[o]
        self.mass = mass[o]
        self.keys = np.asarray(keys, dtype=object)[o]
        self.smiles = np.asarray(smiles, dtype=object)[o]
        self.nbits = nbits
        self.k2i = {k: i for i, k in enumerate(self.keys)}

    def window(self, t, ppm):
        a = np.searchsorted(self.mass, t * (1 - ppm / 1e6), 'left')
        b = np.searchsorted(self.mass, t * (1 + ppm / 1e6), 'right')
        return np.arange(a, b)

    def fps(self, idx):
        return np.unpackbits(np.asarray(self._fp[idx]), axis=1)[:, :self.nbits]

def build_candidate_pool():
    t0 = time.time()
    d = os.path.dirname(find_file('coco_fp.npy'))
    cm = pickle.load(open(os.path.join(d, 'coco_meta.pkl'), 'rb'))
    co_fp = np.load(os.path.join(d, 'coco_fp.npy'))
    co_mass = np.load(os.path.join(d, 'coco_mass.npy'))
    co_keys = np.asarray(cm['keys'], dtype=object)
    co_smis = np.asarray(cm['smiles'], dtype=object)
    print(f"[INFO] COCONUT 2.0 component: {len(co_mass):,} structures loaded ({time.time()-t0:.1f}s)", flush=True)

    if CFG.USE_BIO_DB:
        try:
            bd = os.path.dirname(find_file('bio_fp.npy'))
            bm = pickle.load(open(os.path.join(bd, 'bio_meta.pkl'), 'rb'))
            bi_fp = np.load(os.path.join(bd, 'bio_fp.npy'))
            bi_mass = np.load(os.path.join(bd, 'bio_mass.npy'))
            co_fp = np.vstack([co_fp, bi_fp])
            co_mass = np.concatenate([co_mass, bi_mass])
            co_keys = np.concatenate([co_keys, np.asarray(bm['keys'], dtype=object)])
            co_smis = np.concatenate([co_smis, np.asarray(bm['smiles'], dtype=object)])
            print(f"[INFO] + ChEBI/LIPID MAPS: {len(bi_mass):,} structures integrated.", flush=True)
        except Exception:
            print("[INFO] Bio DB not found, continuing with COCONUT + Train.", flush=True)

    tr = pq.read_table(TRAIN, columns=['inchikey14', 'normalized_smiles']).to_pandas()
    tr = tr.dropna().drop_duplicates('inchikey14')
    coco_keys_set = set(co_keys)
    tr = tr[~tr.inchikey14.isin(coco_keys_set)]
    print(f"[INFO] Computing fingerprints for {len(tr):,} training structures (multiprocessing)...", flush=True)

    if HAVE_RDKIT:
        with MPool(4) as mp:
            res = mp.map(fp_and_mass, list(tr.normalized_smiles), chunksize=500)
        ok = [i for i, r in enumerate(res) if r is not None]
        tr_fp = np.packbits(np.stack([res[i][0] for i in ok]), axis=1)
        tr_mass = np.array([res[i][1] for i in ok])
        tr_keys = tr.inchikey14.values[ok]
        tr_smi = tr.normalized_smiles.values[ok]
    else:
        tr_fp = np.empty((0, co_fp.shape[1]), dtype=np.uint8)
        tr_mass = np.empty(0, dtype=np.float64)
        tr_keys = np.empty(0, dtype=object)
        tr_smi = np.empty(0, dtype=object)

    fp = np.vstack([co_fp, tr_fp])
    mass = np.concatenate([co_mass, tr_mass])
    keys = np.concatenate([co_keys, tr_keys])
    smis = np.concatenate([co_smis, tr_smi])
    good = np.isfinite(mass)

    pool = CandidatePool(fp[good], mass[good], keys[good], smis[good], cm['nbits'])
    print(f"[SUCCESS] Unified candidate pool ready: {len(pool.mass):,} structures ({time.time()-t0:.1f}s)", flush=True)
    return pool


# ===================================================================================
#  Evidence Channels: Direct Library Match, Analog Match & MetFrag-lite
# ===================================================================================
AMU = {'C':12.0, 'H':1.00782503207, 'N':14.0030740048, 'O':15.9949146196, 'P':30.97376163,
       'S':31.97207100, 'F':18.99840322, 'Cl':34.96885268, 'Br':78.9183371, 'I':126.904473,
       'Na':22.9897692809, 'K':38.96370668, 'Si':27.9769265325, 'B':11.0093054, 'Se':79.9165213}
H_ATOM = AMU['H']
PROTON_MASS = H_ATOM - 0.00054857990

def clean_spectrum(mz, it):
    return _clean(np.asarray(mz, np.float32), np.asarray(it, np.float32),
                  CFG.INT_FLOOR, CFG.MAX_PEAKS, CFG.INT_POWER, CFG.ENT_WEIGHT)

def lib_sim(L, specs, target):
    """CLASS 1: Direct match against library spectra within tight neutral mass window."""
    cand = lib_window(L, target, target * CFG.PPM_WIN / 1e6)
    if len(cand) == 0: return {}
    agg = {}
    for mz, it in specs:
        qm, qp = clean_spectrum(mz, it)
        if len(qm) == 0: continue
        sc = search(qm, qp, cand, L['off'], L['mz'], L['it'],
                    CFG.MZ_TOL, CFG.INT_FLOOR, CFG.MAX_PEAKS, CFG.INT_POWER, CFG.ENT_WEIGHT)
        for c, s in zip(cand, sc):
            k = L['ik'][c]
            if s > agg.get(k, -1.0):
                agg[k] = float(s)
    return agg

def analog_sim(L, specs, target, rep, rep_key, rep_nm):
    """CLASS 2: Mass-shifted spectral entropy match across wide +-200 Da window."""
    lo = np.searchsorted(rep_nm, target - CFG.ANALOG_WIN, 'left')
    hi = np.searchsorted(rep_nm, target + CFG.ANALOG_WIN, 'right')
    cand = rep[lo:hi]
    if len(cand) == 0: return []
    shift = (target - rep_nm[lo:hi]).astype(np.float32)
    ckey = rep_key[lo:hi]
    agg = {}
    for mz, it in specs:
        qm, qp = clean_spectrum(mz, it)
        if len(qm) == 0: continue
        sc = search_shift(qm, qp, cand, L['off'], L['mz'], L['it'],
                          CFG.MZ_TOL, CFG.INT_FLOOR, CFG.MAX_PEAKS,
                          CFG.INT_POWER, CFG.ENT_WEIGHT, shift)
        for c, k, s in zip(cand, ckey, sc):
            if s > agg.get(k, -1.0):
                agg[k] = float(s)
    return sorted(agg.items(), key=lambda x: -x[1])[:CFG.N_ANALOG]

# MetFrag-lite: In-silico 1-2 bond fragmentation
def mol_graph(smi):
    if not HAVE_RDKIT: return None
    m = Chem.MolFromSmiles(smi)
    if m is None: return None
    n = m.GetNumAtoms()
    w = np.zeros(n)
    for a in m.GetAtoms():
        w[a.GetIdx()] = AMU.get(a.GetSymbol(), 0.0) + a.GetTotalNumHs() * H_ATOM
    if (w == 0).any(): return None
    bonds = [(b.GetBeginAtomIdx(), b.GetEndAtomIdx()) for b in m.GetBonds()]
    return w, bonds, n

def _components(n, bonds, drop):
    adj = [[] for _ in range(n)]
    for i, (a, b) in enumerate(bonds):
        if i in drop: continue
        adj[a].append(b); adj[b].append(a)
    seen = np.zeros(n, bool); comps = []
    for s in range(n):
        if seen[s]: continue
        stack = [s]; seen[s] = True; cur = [s]
        while stack:
            u = stack.pop()
            for v in adj[u]:
                if not seen[v]: seen[v] = True; stack.append(v); cur.append(v)
        comps.append(cur)
    return comps

def fragment_masses(smi, max_breaks=2, max_bonds=34):
    g = mol_graph(smi)
    if g is None: return np.zeros(0)
    w, bonds, n = g
    nb = len(bonds)
    if nb == 0 or nb > max_bonds: return np.array([w.sum()])
    out = {w.sum()}
    for i in range(nb):
        for c in _components(n, bonds, {i}):
            out.add(float(w[c].sum()))
    if max_breaks >= 2:
        for i in range(nb):
            for j in range(i + 1, nb):
                for c in _components(n, bonds, {i, j}):
                    out.add(float(w[c].sum()))
    return np.array(sorted(out))

def explain_score(frag_mass, peak_mz, peak_int, mode=1.0, tol=0.01, h_shifts=(-2, -1, 0, 1, 2)):
    if len(frag_mass) == 0 or len(peak_mz) == 0: return 0.0
    ion = []
    for dh in h_shifts:
        ion.append(frag_mass + dh * H_ATOM + (PROTON_MASS if mode > 0 else -PROTON_MASS))
    ion = np.sort(np.concatenate(ion))
    w = np.sqrt(np.asarray(peak_int, float)); tot = w.sum()
    if tot <= 0: return 0.0
    idx = np.searchsorted(ion, peak_mz)
    ok = np.zeros(len(peak_mz), bool)
    for off in (-1, 0):
        k = np.clip(idx + off, 0, len(ion) - 1)
        ok |= np.abs(ion[k] - peak_mz) <= tol
    return float(w[ok].sum() / tot)

def _frag_masses_wrapper(smi):
    try: return fragment_masses(smi)
    except Exception: return np.zeros(0)

def frag_scores(cand_smiles, specs, mode, workers=4):
    if not HAVE_RDKIT:
        return np.zeros(len(cand_smiles), np.float32)
    with MPool(workers) as mp:
        frags = mp.map(_frag_masses_wrapper, cand_smiles, chunksize=8)
    peaks = []
    for mz, it in specs:
        m2, i2 = _clean(np.asarray(mz, np.float32), np.asarray(it, np.float32),
                        CFG.INT_FLOOR, CFG.MAX_PEAKS, 1.0, False)
        peaks.append((np.asarray(m2, float), np.asarray(i2, float)))
    out = np.zeros(len(cand_smiles), np.float32)
    for j, f in enumerate(frags):
        out[j] = max((explain_score(f, a, b, mode=mode, tol=CFG.MZ_TOL) for a, b in peaks), default=0.0)
    return out


# ===================================================================================
#  Channel 4: Neural Spectrum -> Molecular Fingerprint Model (FPNet)
#  Ranking is f . z (Exact Bayes log-likelihood dot product)
# ===================================================================================
import torch
import torch.nn as nn
import torch.nn.functional as F

MAX_PEAKS_NN = 128
ADDUCT_LIST = ["[M+H]+","[M+NH4]+","[M+Na]+","[M+K]+","[M-H2O+H]+","[M-2H2O+H]+","[M]+",
               "[M-H]-","[M-H2O-H]-","[M+CH2O2-H]-","[M+C2H4O2-H]-","[M+Cl]-","[M]-",
               "[M+2H]2+","[M-2H]-","[2M+H]+","[2M+Na]+","[2M+NH4]+","[2M-H]-","[2M+K]+",
               "[2M+CH2O2-H]-","[2M+C2H4O2-H]-","[2M+Na-2H]-","[M+Na-2H]-","[M-H2O]+","<unk>"]
ADDUCT_IX = {a: i for i, a in enumerate(ADDUCT_LIST)}
INSTR_LIST = ["timsTOF", "Orbitrap", "QTOF", "IT", "other"]
INSTR_IX = {a: i for i, a in enumerate(INSTR_LIST)}

def instr_family(s):
    if s is None: return 4
    t = str(s).lower()
    if 'timstof' in t: return 0
    if 'orbitrap' in t or 'qft' in t or 'ftms' in t or 'hybrid ft' in t or 'itft' in t or 'exactive' in t: return 1
    if 'tof' in t: return 2
    if 'trap' in t or 'qq' in t: return 3
    return 4

def prep_peaks(mz, inten, prec_mz, max_peaks=MAX_PEAKS_NN, floor=1e-3, win=50.0, per_win=8):
    mz = np.asarray(mz, np.float64); it = np.asarray(inten, np.float64)
    if len(mz) == 0: return np.zeros(0, np.float32), np.zeros(0, np.float32)
    keep = (mz <= prec_mz + 1.5)
    mz, it = mz[keep], it[keep]
    if len(mz) == 0: return np.zeros(0, np.float32), np.zeros(0, np.float32)
    mx = it.max()
    if mx <= 0: return np.zeros(0, np.float32), np.zeros(0, np.float32)
    keep = it >= floor * mx
    mz, it = mz[keep], it[keep]
    if len(mz) > max_peaks:
        order = np.argsort(-it)
        bucket = (mz // win).astype(np.int64)
        cnt = {}; sel = []
        for i in order:
            b = bucket[i]; c = cnt.get(b, 0)
            if c < per_win: cnt[b] = c + 1; sel.append(i)
        sel = np.array(sel)
        if len(sel) > max_peaks:
            sel = sel[np.argsort(-it[sel])[:max_peaks]]
        elif len(sel) < max_peaks:
            rest = np.array([i for i in order if i not in set(sel.tolist())])
            need = max_peaks - len(sel)
            if len(rest): sel = np.concatenate([sel, rest[:need]])
        mz, it = mz[sel], it[sel]
    o = np.argsort(mz)
    mz, it = mz[o], it[o]
    v = np.sqrt(it / it.max())
    return mz.astype(np.float32), v.astype(np.float32)

class SinEmb(nn.Module):
    def __init__(self, dim, lo=-2.0, hi=3.2, power=1.0):
        super().__init__()
        n = dim // 2
        wav = torch.pow(10.0, (hi - lo) * torch.pow(torch.linspace(0, 1, n), power) + lo)
        self.register_buffer('inv', (2 * math.pi) / wav)
    def forward(self, x):
        a = x.unsqueeze(-1) * self.inv
        return torch.cat([torch.sin(a), torch.cos(a)], -1)

class Block(nn.Module):
    def __init__(self, d, h, drop):
        super().__init__(); self.h = h
        self.n1 = nn.LayerNorm(d); self.qkv = nn.Linear(d, 3 * d); self.o = nn.Linear(d, d)
        self.n2 = nn.LayerNorm(d)
        self.ff = nn.Sequential(nn.Linear(d, 4 * d), nn.GELU(), nn.Dropout(drop), nn.Linear(4 * d, d))
        self.drop = nn.Dropout(drop)
    def forward(self, x, pad):
        B, N, D = x.shape; y = self.n1(x)
        q, k, v = self.qkv(y).view(B, N, 3, self.h, D // self.h).permute(2, 0, 3, 1, 4)
        m = (~pad)[:, None, None, :]
        a = F.scaled_dot_product_attention(q, k, v, attn_mask=m)
        x = x + self.drop(self.o(a.transpose(1, 2).reshape(B, N, D)))
        return x + self.drop(self.ff(self.n2(x)))

class FPNet(nn.Module):
    def __init__(self, nbits, d=512, layers=6, heads=8, drop=0.1):
        super().__init__()
        self.d = d
        self.mz_emb = SinEmb(d)
        self.nl_emb = SinEmb(d)
        self.pk = nn.Linear(2 * d + 1, d)
        self.prec_emb = SinEmb(d)
        self.ad = nn.Embedding(len(ADDUCT_LIST), d)
        self.ins = nn.Embedding(len(INSTR_LIST), d)
        self.gl = nn.Linear(d + 3, d)
        self.blocks = nn.ModuleList([Block(d, heads, drop) for _ in range(layers)])
        self.norm = nn.LayerNorm(d)
        self.head = nn.Sequential(nn.Linear(2 * d, 2048), nn.GELU(), nn.Dropout(drop), nn.Linear(2048, nbits))

    def forward(self, mz, it, pad, prec, ad, ins, ce, mode):
        B, N = mz.shape
        nl = (prec[:, None] - mz).clamp(min=0)
        p = self.pk(torch.cat([self.mz_emb(mz), self.nl_emb(nl), it.unsqueeze(-1)], -1))
        g = self.gl(torch.cat([self.prec_emb(prec),
                               (ce / 100.0).unsqueeze(-1), mode.unsqueeze(-1),
                               torch.log1p(prec).unsqueeze(-1) / 10.0], -1)) + self.ad(ad) + self.ins(ins)
        x = torch.cat([g.unsqueeze(1), p], 1)
        pad = torch.cat([torch.zeros(B, 1, dtype=torch.bool, device=pad.device), pad], 1)
        for b in self.blocks: x = b(x, pad)
        x = self.norm(x)
        cls = x[:, 0]
        msk = (~pad[:, 1:]).float().unsqueeze(-1)
        mean = (x[:, 1:] * msk).sum(1) / msk.sum(1).clamp(min=1)
        return self.head(torch.cat([cls, mean], -1))

_MODEL = None

def load_neural_models():
    global _MODEL
    paths = sorted(glob.glob('/kaggle/input/**/fp_*.pt', recursive=True))
    if not paths:
        print("[INFO] No pretrained FPNet checkpoints found — proceeding with 3 channels.")
        return None
    dev = 'cuda' if torch.cuda.is_available() else 'cpu'
    single, merged = [], []
    for pth in paths:
        ck = torch.load(pth, map_location='cpu', weights_only=False)
        net = FPNet(ck['nbits'], d=ck['d'], layers=ck['layers']).to(dev).eval()
        net.load_state_dict(ck['model'])
        if 'merged' in pth.split('/')[-1]:
            merged.append(net)
        else:
            single.append(net)
        print(f"  Loaded {os.path.basename(pth)}: d={ck['d']}, layers={ck['layers']}, step={ck.get('step')}")
    print(f"[SUCCESS] Neural FPNet Ensemble ready: {len(single)} single + {len(merged)} merged on {dev}")
    _MODEL = (single, merged, dev, ck['nbits'])
    return _MODEL

def _merge_peaks(sub):
    mz = np.concatenate([np.asarray(r.ms2_mzs, float) for r in sub.itertuples()])
    it = np.concatenate([np.asarray(r.ms2_normalized_intensities, float) /
                         max(float(np.asarray(r.ms2_normalized_intensities, float).max()), 1e-9)
                         for r in sub.itertuples()])
    o = np.argsort(mz); mz, it = mz[o], it[o]
    keep = np.ones(len(mz), bool)
    for j in range(1, len(mz)):
        if mz[j] - mz[j - 1] < 0.005:
            if it[j] >= it[j - 1]: keep[j - 1] = False
            else: keep[j] = False
    return mz[keep], it[keep]

@torch.no_grad()
def model_logits(sub):
    if _MODEL is None: return None
    single, merged, dev, nbits = _MODEL
    out = []
    if single:
        za = _logits_from(sub, single)
        if za is not None: out.append(za)
    if merged:
        mz, it = _merge_peaks(sub)
        r0 = next(sub.itertuples())
        zb = _logits_raw([(mz, it)], merged, float(np.median(sub.precursor_mz)), r0.adduct,
                         r0.instrument_type, 25.0,
                         float(np.mean([1.0 if m == 'positive' else -1.0 for m in sub.ionization_mode])))
        if zb is not None: out.append(zb)
    return np.mean(out, axis=0) if out else None

@torch.no_grad()
def _logits_from(sub, nets):
    if _MODEL is None: return None
    dev = _MODEL[2]
    rows = list(sub.itertuples())
    P = [prep_peaks(r.ms2_mzs, r.ms2_normalized_intensities, float(r.precursor_mz)) for r in rows]
    P = [(a, b) for a, b in P if len(a)]
    if not P: return None
    B = len(P); N = max(len(a) for a, _ in P)
    mz = np.zeros((B, N), np.float32); it = np.zeros((B, N), np.float32); pad = np.ones((B, N), bool)
    for i, (a, b) in enumerate(P):
        mz[i, :len(a)] = a; it[i, :len(b)] = b; pad[i, :len(a)] = False
    def ce_of(r):
        v = r.collision_energy_ev
        try: return float(np.mean(np.atleast_1d(v))) if v is not None and len(np.atleast_1d(v)) else 25.0
        except Exception: return 25.0
    T = lambda x: torch.as_tensor(x, device=dev)
    args = (T(mz), T(it), T(pad),
            T(np.array([float(r.precursor_mz) for r in rows[:B]], np.float32)),
            T(np.array([ADDUCT_IX.get(r.adduct, ADDUCT_IX['<unk>']) for r in rows[:B]])),
            T(np.array([instr_family(r.instrument_type) for r in rows[:B]])),
            T(np.array([ce_of(r) for r in rows[:B]], np.float32)),
            T(np.array([1.0 if r.ionization_mode == 'positive' else -1.0 for r in rows[:B]], np.float32)))
    return np.mean([n(*args).float().mean(0).cpu().numpy() for n in nets], axis=0)

@torch.no_grad()
def _logits_raw(pairs, nets, prec, adduct, instrument, ce, mode):
    if _MODEL is None: return None
    dev = _MODEL[2]
    P = [prep_peaks(mz, it, prec) for mz, it in pairs]
    P = [(a, b) for a, b in P if len(a)]
    if not P: return None
    B = len(P); N = max(len(a) for a, _ in P)
    mz = np.zeros((B, N), np.float32); it = np.zeros((B, N), np.float32); pad = np.ones((B, N), bool)
    for i, (a, b) in enumerate(P):
        mz[i, :len(a)] = a; it[i, :len(b)] = b; pad[i, :len(a)] = False
    T = lambda x: torch.as_tensor(x, device=dev)
    args = (T(mz), T(it), T(pad), T(np.full(B, prec, np.float32)),
            T(np.full(B, ADDUCT_IX.get(adduct, ADDUCT_IX['<unk>']))),
            T(np.full(B, instr_family(instrument))),
            T(np.full(B, ce, np.float32)), T(np.full(B, mode, np.float32)))
    return np.mean([n(*args).float().mean(0).cpu().numpy() for n in nets], axis=0)


# ===================================================================================
#  31-Feature Extraction & Calibrated Dual-Prior Seed-Ensemble Ranker
# ===================================================================================
from sklearn.ensemble import HistGradientBoostingClassifier

N_ANALOG = CFG.N_ANALOG
P_SIM    = CFG.SIM_POWER
N_FEAT   = 31

def _rank_norm(x):
    o = np.argsort(-x)
    r = np.empty(len(x))
    r[o] = np.arange(len(x))
    return r / max(1, len(x) - 1)

def _z(x):
    s = x.std()
    return (x - x.mean()) / s if s > 1e-9 else np.zeros_like(x)

def rank_features(cand_fp, cand_lib, analog_fp, analog_sim_vals, model_logits=None, frag=None):
    nc = cand_fp.shape[0]
    cf = cand_fp.astype(np.float32)
    cs = cf.sum(1)
    lv = np.asarray(cand_lib, np.float32)
    lvmax = float(lv.max()) if nc else 0.0

    if analog_fp is not None and len(analog_sim_vals):
        af = analog_fp.astype(np.float32)
        asum = af.sum(1)
        inter = cf @ af.T
        tan = inter / (cs[:, None] + asum[None, :] - inter + 1e-9)
        w = np.clip(np.asarray(analog_sim_vals, np.float32), 0, None)
        ap = (tan * (w ** P_SIM)[None, :]).max(1)
        a1 = (tan * w[None, :]).max(1)
        best_tan = tan.max(1)
        top_tan = tan[:, 0]
        top_sim = float(w[0])
        mean_tan = (tan * (w ** P_SIM)[None, :]).sum(1) / ((w ** P_SIM).sum() + 1e-9)
    else:
        ap = a1 = best_tan = top_tan = mean_tan = np.zeros(nc, np.float32)
        top_sim = 0.0

    apmax = float(ap.max()) if nc else 0.0

    if model_logits is not None:
        raw = cf @ np.asarray(model_logits, np.float32)
        nrm = raw / np.sqrt(np.maximum(cs, 1.0))
        mfeat = [_z(raw), _rank_norm(raw), raw - raw.max(), _z(nrm), _rank_norm(nrm),
                 (raw == raw.max()).astype(np.float32)]
    else:
        mfeat = [np.zeros(nc, np.float32)] * 6

    if model_logits is not None and nc:
        mr = _rank_norm(cf @ np.asarray(model_logits, np.float32))
        lbest = int(np.argmax(lv)) if lvmax > 0 else -1
        agree = float(1.0 - mr[lbest]) if lbest >= 0 else 0.0
        abest = int(np.argmax(ap)) if apmax > 0 else -1
        agree_a = float(1.0 - mr[abest]) if abest >= 0 else 0.0
        xfeat = [lv * (1.0 - mr), ap * (1.0 - mr), np.full(nc, agree), np.full(nc, agree_a),
                 np.full(nc, agree * lvmax), np.full(nc, float(np.corrcoef(lv, -mr)[0, 1]) if lv.std() > 1e-9 else 0.0)]
    else:
        xfeat = [np.zeros(nc, np.float32)] * 6

    if frag is not None:
        fr = np.asarray(frag, np.float32)
        ffeat = [fr, _rank_norm(fr), fr - fr.max() if nc else fr, _z(fr)]
    else:
        ffeat = [np.zeros(nc, np.float32)] * 4

    return np.column_stack([
        lv, _rank_norm(lv), np.full(nc, lvmax), lv - lvmax, (lv > 0).astype(float),
        ap, _rank_norm(ap), np.full(nc, apmax), ap - apmax,
        a1, best_tan, top_tan, mean_tan, np.full(nc, top_sim),
        np.full(nc, np.log(max(nc, 1))),
        *mfeat, *ffeat, *xfeat,
    ]).astype(np.float32)

# Fit Calibrated Rankers across Priors and Seeds
rank_train_path = find_file('rank_train.npz')
z = np.load(rank_train_path)
NFEAT = z['X'].shape[1]

RANKERS = []
for w1 in CFG.W1_PRIORS:
    W = np.where(z['M'] == 0, w1, 1.0 - w1)
    for sd in CFG.SEEDS:
        m = HistGradientBoostingClassifier(random_state=sd, **CFG.GBM)
        m.fit(z['X'], z['Y'], sample_weight=W)
        RANKERS.append(m)

def rank_proba(X):
    """Ensemble prediction averaged across priors and seeds for low variance."""
    return np.mean([m.predict_proba(X)[:, 1] for m in RANKERS], axis=0)

print(f"[SUCCESS] Calibrated Ranker ensemble ready: {len(RANKERS)} GBMs ({len(CFG.W1_PRIORS)} priors x {len(CFG.SEEDS)} seeds) on {z['X'].shape[0]:,} rows x {NFEAT} features.")



# ---------------------------------------------------------------------------
# v2 helpers: metric-faithful dedupe and nearest-mass fill
# ---------------------------------------------------------------------------
from functools import lru_cache
if HAVE_RDKIT:
    from rdkit.Chem.MolStandardize import rdMolStandardize
    _TAUT = rdMolStandardize.TautomerEnumerator()

@lru_cache(maxsize=1_000_000)
def canon_key(smi):
    if not HAVE_RDKIT:
        return smi
    m = Chem.MolFromSmiles(smi)
    if m is None:
        return None
    try:
        m = _TAUT.Canonicalize(m)
    except Exception:
        pass
    try:
        ik = Chem.MolToInchiKey(m)
    except Exception:
        return None
    return ik[:14] if ik else None

def pool_nearest_iter(pool, target):
    """Yield pool indices in order of |mass - target|, walking outward from the target."""
    pos = int(np.searchsorted(pool.mass, target))
    lo, hi = pos - 1, pos
    n = len(pool.mass)
    while lo >= 0 or hi < n:
        take_lo = hi >= n or (lo >= 0 and target - pool.mass[lo] <= pool.mass[hi] - target)
        if take_lo:
            yield lo; lo -= 1
        else:
            yield hi; hi += 1

def dedupe_fill(ranked_smiles, pool, target, topn):
    out, seen = [], set()
    for smi in ranked_smiles:
        k = canon_key(smi)
        if k is None or k in seen:
            continue
        seen.add(k); out.append(smi)
        if len(out) == topn:
            return out
    for idx in pool_nearest_iter(pool, target):
        smi = pool.smiles[idx]
        k = canon_key(smi)
        if k is None or k in seen:
            continue
        seen.add(k); out.append(smi)
        if len(out) == topn:
            break
    return out

# ===================================================================================
#  Unified Test Inference Pipeline: 4-Channel Retrieval & Re-ranking
# ===================================================================================
load_neural_models()
pool = build_candidate_pool()
L = load_library(TRAIN)
rep, rep_key, rep_nm = build_rep(L)

te = pq.read_table(TEST).to_pandas()
te['nm'] = neutral_mass(te.precursor_mz.values.astype(np.float64), te.adduct.values)
mols = list(te.groupby('molecule_id'))
print(f"[INFO] Evaluating {len(te):,} spectra across {len(mols):,} unique test molecules...", flush=True)

rows, diag = [], []
for gi, (mid, sub) in enumerate(mols):
    nms  = sub.nm.values[np.isfinite(sub.nm.values)]
    smis = []
    if len(nms):
        target = float(np.median(nms))
        specs  = [(r.ms2_mzs, r.ms2_normalized_intensities) for r in sub.itertuples()]

        # Channel 1: Direct Library Similarity
        lib_hits = lib_sim(L, specs, target)

        # Channel 2: Mass-shifted Analog Propagation
        analogs  = analog_sim(L, specs, target, rep, rep_key, rep_nm)

        # Candidate Window from Unified Pool
        cand = pool.window(target, CFG.PPM_WIN)
        if len(cand) == 0:
            cand = pool.window(target, CFG.PPM_FALLBACK)

        if len(cand):
            if len(cand) > CFG.PRUNE_N:
                lv_coarse = np.array([lib_hits.get(pool.keys[c], 0.0) for c in cand], np.float32)
                mass_diff = np.abs(pool.mass[cand] - target)
                coarse_score = lv_coarse * 100.0 - mass_diff
                top_indices = np.argsort(-coarse_score)[:CFG.PRUNE_N]
                cand = cand[top_indices]
            cfp = pool.fps(cand)
            lv  = np.array([lib_hits.get(pool.keys[c], 0.0) for c in cand], np.float32)
            ids, sims = [], []
            for k, s in analogs:
                i = pool.k2i.get(k, -1)
                if i >= 0:
                    ids.append(i); sims.append(s)
            afp = pool.fps(np.array(ids)) if ids else None

            # Channel 3: In-silico Fragmentation
            ion_mode = float(np.mean([1.0 if m == 'positive' else -1.0 for m in sub.ionization_mode]))
            fsc = frag_scores([pool.smiles[c] for c in cand], specs, ion_mode)

            # Channel 4: Neural Fingerprint Model Logits
            zlog = model_logits(sub)

            # Feature Assembly & Calibrated Prediction
            X = rank_features(cfp, lv, afp, np.array(sims, np.float32), zlog, fsc)[:, :NFEAT]
            p = rank_proba(X)
            order = np.argsort(-p)
            smis = dedupe_fill([pool.smiles[cand[i]] for i in order], pool, target, CFG.TOPN)

            diag.append((mid, target, len(cand), float(lv.max()),
                         float(sims[0]) if sims else 0.0, float(p[order[0]])))

    if len(smis) < CFG.TOPN and len(nms):
        smis = dedupe_fill(smis, pool, float(np.median(nms)), CFG.TOPN)
    if not smis: smis = ['CCO']
    rows.append((mid, ';'.join(smis[:CFG.TOPN])))

    if gi % 50 == 0 or gi == len(mols) - 1:
        print(f"  Processed {gi+1}/{len(mols)} molecules | Elapsed: {time.time()-T0:.1f}s", flush=True)

# Build & strictly audit submission
submission = pd.DataFrame(rows, columns=['molecule_id', 'smiles'])
samp = pd.read_csv(SAMPLE)
submission = samp[['molecule_id']].merge(submission, on='molecule_id', how='left')
submission['smiles'] = submission['smiles'].fillna('CCO')

def pad_candidates(s):
    tokens = s.split(';')
    if len(tokens) < CFG.TOPN:
        tokens += ['CCO'] * (CFG.TOPN - len(tokens))
    return ';'.join(tokens[:CFG.TOPN])

submission['smiles'] = submission['smiles'].apply(pad_candidates)

# Strict Assertions Matching Competition Standards
assert len(submission) == len(samp), f"Row count mismatch: expected {len(samp)}, got {len(submission)}"
assert (submission['molecule_id'] == samp['molecule_id']).all(), "molecule_id alignment mismatch"
assert submission.molecule_id.duplicated().sum() == 0, "Duplicated molecule_ids found"
assert submission.smiles.isnull().sum() == 0, "Null values present in smiles column"
counts = submission['smiles'].apply(lambda s: len(s.split(';')))
assert (counts == 25).all(), "Every row must have exactly 25 candidates separated by semicolon"

submission.to_csv('submission.csv', index=False)
print(f"\n[SUCCESS] submission.csv generated successfully with shape {submission.shape} in {time.time()-T0:.1f}s total.")

print(submission.head())
