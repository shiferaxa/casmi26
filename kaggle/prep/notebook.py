# CASMI26: preprocess train.parquet into fixed-size peak tokens and fingerprint targets for model training.
# CPU kernel. Output (in /kaggle/working) is consumed by the training kernels through kernel_sources.
#   tokens_mz.npy   float32 (N, 128)   m/z of up to 128 peaks (0 padded)
#   tokens_it.npy   float16 (N, 128)   sqrt relative intensity
#   n_peaks.npy     int16   (N,)
#   prec.npy        float32 (N,)       precursor m/z
#   adduct.npy      int8    (N,)       index into ADDUCT_LIST
#   instr.npy       int8    (N,)       instrument family 0 timsTOF 1 orbitrap 2 tof 3 trap 4 other
#   ce.npy          float32 (N,)       mean collision energy in eV (25 if unknown)
#   mode.npy        int8    (N,)       +1 positive, -1 negative
#   fp_idx.npy      int32   (N,)       row into fp_packed.npy, -1 if the structure has no fingerprint
#   split.npy       int8    (N,)       0 train, 1 validation (2 percent of structures), 2 np-examples (held out)
#   fp_packed.npy   uint8   (S, 867)   6,930 selected fingerprint bits per structure, packed
#   fp_keys.npy     object  (S,)       inchikey14 per fingerprint row
import glob, os, time
import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq

T0 = time.time()
MAX_PEAKS = 128
ADDUCT_LIST = ["[M+H]+","[M+NH4]+","[M+Na]+","[M+K]+","[M-H2O+H]+","[M-2H2O+H]+","[M]+",
               "[M-H]-","[M-H2O-H]-","[M+CH2O2-H]-","[M+C2H4O2-H]-","[M+Cl]-","[M]-",
               "[M+2H]2+","[M-2H]-","[2M+H]+","[2M+Na]+","[2M+NH4]+","[2M-H]-","[2M+K]+",
               "[2M+CH2O2-H]-","[2M+C2H4O2-H]-","[2M+Na-2H]-","[M+Na-2H]-","[M-H2O]+","<unk>"]
ADDUCT_IX = {a: i for i, a in enumerate(ADDUCT_LIST)}


def instr_family(s):
    if s is None: return 4
    t = str(s).lower()
    if 'timstof' in t: return 0
    if 'orbitrap' in t or 'qft' in t or 'ftms' in t or 'hybrid ft' in t or 'itft' in t or 'exactive' in t: return 1
    if 'tof' in t: return 2
    if 'trap' in t or 'qq' in t: return 3
    return 4


def prep_peaks(mz, it, prec_mz, max_peaks=MAX_PEAKS, floor=1e-3, win=50.0, per_win=8):
    """Same selection as the public FPNet input: below precursor, floor, at most 8 peaks per 50 Da, top 128."""
    if len(mz) == 0:
        return np.zeros(0, np.float32), np.zeros(0, np.float32)
    keep = mz <= prec_mz + 1.5
    mz, it = mz[keep], it[keep]
    if len(mz) == 0:
        return np.zeros(0, np.float32), np.zeros(0, np.float32)
    mx = it.max()
    if mx <= 0:
        return np.zeros(0, np.float32), np.zeros(0, np.float32)
    keep = it >= floor * mx
    mz, it = mz[keep], it[keep]
    if len(mz) > max_peaks:
        order = np.argsort(-it)
        bucket = (mz // win).astype(np.int64)
        cnt = {}; sel = []
        for i in order:
            b = bucket[i]; c = cnt.get(b, 0)
            if c < per_win:
                cnt[b] = c + 1; sel.append(i)
        sel = np.array(sel)
        if len(sel) > max_peaks:
            sel = sel[np.argsort(-it[sel])[:max_peaks]]
        elif len(sel) < max_peaks:
            chosen = set(sel.tolist())
            rest = np.array([i for i in order if i not in chosen])
            need = max_peaks - len(sel)
            if len(rest): sel = np.concatenate([sel, rest[:need]])
        mz, it = mz[sel], it[sel]
    o = np.argsort(mz)
    mz, it = mz[o], it[o]
    return mz.astype(np.float32), np.sqrt(it / it.max()).astype(np.float32)


def find(name):
    hits = glob.glob(f"/kaggle/input/**/{name}", recursive=True)
    assert hits, name
    return sorted(hits, key=len)[0]


# fingerprints per structure
fpz = np.load(find("train_fp.npz"), allow_pickle=True)
fp_packed = fpz["fp"].astype(np.uint8)
fp_keys = fpz["keys"]
key_to_row = {k: i for i, k in enumerate(fp_keys)}
print(f"fingerprints: {fp_packed.shape} ({time.time()-T0:.0f}s)", flush=True)

cols = ["inchikey14", "ingest_lib", "precursor_mz", "adduct", "ionization_mode", "instrument_type",
        "collision_energy_ev", "ms2_mzs", "ms2_normalized_intensities"]
pf = pq.ParquetFile(find("train.parquet"))
N = pf.metadata.num_rows
tok_mz = np.zeros((N, MAX_PEAKS), np.float32)
tok_it = np.zeros((N, MAX_PEAKS), np.float16)
n_peaks = np.zeros(N, np.int16)
prec = np.zeros(N, np.float32)
adduct = np.zeros(N, np.int8)
instr = np.zeros(N, np.int8)
ce = np.full(N, 25.0, np.float32)
mode = np.zeros(N, np.int8)
fp_idx = np.full(N, -1, np.int32)
keys = np.empty(N, dtype=object)
libs = np.empty(N, dtype=object)

row = 0
for rg in range(pf.metadata.num_row_groups):
    t = pf.read_row_group(rg, columns=cols)
    mzc = t.column("ms2_mzs").combine_chunks()
    itc = t.column("ms2_normalized_intensities").combine_chunks()
    off = mzc.offsets.to_numpy().astype(np.int64)
    allmz = mzc.values.to_numpy(zero_copy_only=False).astype(np.float64)
    allit = itc.values.to_numpy(zero_copy_only=False).astype(np.float64)
    off = off - off[0]
    pmz = t.column("precursor_mz").to_numpy(zero_copy_only=False).astype(np.float64)
    ad = t.column("adduct").to_pylist()
    im = t.column("ionization_mode").to_pylist()
    it_ = t.column("instrument_type").to_pylist()
    cev = t.column("collision_energy_ev").to_pylist()
    ik = t.column("inchikey14").to_pylist()
    lb = t.column("ingest_lib").to_pylist()
    n = len(t)
    for i in range(n):
        a, b = off[i], off[i + 1]
        mz, it = prep_peaks(allmz[a:b], allit[a:b], pmz[i])
        k = len(mz)
        tok_mz[row, :k] = mz
        tok_it[row, :k] = it
        n_peaks[row] = k
        prec[row] = pmz[i]
        adduct[row] = ADDUCT_IX.get(ad[i], ADDUCT_IX["<unk>"])
        instr[row] = instr_family(it_[i])
        v = cev[i]
        if v is not None and len(v):
            ce[row] = float(np.mean(v))
        mode[row] = 1 if im[i] == "positive" else -1
        fp_idx[row] = key_to_row.get(ik[i], -1)
        keys[row] = ik[i]
        libs[row] = lb[i]
        row += 1
    print(f"row group {rg+1}/{pf.metadata.num_row_groups}  rows {row:,}  ({time.time()-T0:.0f}s)", flush=True)
assert row == N

# molecule-grouped split: np-examples structures held out entirely, 2 percent of the rest for validation
npx_keys = set(keys[libs == "enveda-np-examples"])
rng = np.random.default_rng(0)
all_keys = np.array(sorted(set(k for k in keys if k) - npx_keys))
val_keys = set(rng.choice(all_keys, size=int(0.02 * len(all_keys)), replace=False))
split = np.zeros(N, np.int8)
split[np.array([k in val_keys for k in keys])] = 1
split[np.array([k in npx_keys for k in keys])] = 2
print("split counts:", {int(s): int((split == s).sum()) for s in (0, 1, 2)}, " spectra without fingerprint:", int((fp_idx < 0).sum()))

out = "/kaggle/working"
np.save(f"{out}/tokens_mz.npy", tok_mz); np.save(f"{out}/tokens_it.npy", tok_it)
np.save(f"{out}/n_peaks.npy", n_peaks); np.save(f"{out}/prec.npy", prec)
np.save(f"{out}/adduct.npy", adduct); np.save(f"{out}/instr.npy", instr)
np.save(f"{out}/ce.npy", ce); np.save(f"{out}/mode.npy", mode)
np.save(f"{out}/fp_idx.npy", fp_idx); np.save(f"{out}/split.npy", split)
np.save(f"{out}/fp_packed.npy", fp_packed); np.save(f"{out}/fp_keys.npy", fp_keys, allow_pickle=True)
print(f"done {N:,} spectra in {time.time()-T0:.0f}s; peaks per spectrum median {np.median(n_peaks):.0f}")
