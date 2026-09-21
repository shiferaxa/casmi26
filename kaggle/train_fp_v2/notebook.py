# CASMI26: spectrum-to-fingerprint transformer, v2 recipe.
# Same architecture and checkpoint format as the public FPNet (drops into channel 4 unchanged).
# What v1 lacked and this adds, following the public author's own training script:
#   hard negatives   63 decoy structures per spectrum from the same 10 ppm mass window of the full
#                    candidate pool (COCONUT plus training structures), contrastive cross entropy on
#                    (bits dot logits) with the true structure at index 0, added to the per-bit BCE
#   merge views      with probability 0.6 the input is 2 to 4 spectra of the same structure merged,
#                    because test molecules come with several spectra
#   augmentation     peak dropout up to 30 percent, log-normal intensity jitter, 5 ppm m/z jitter
#   best checkpoint  kept by validation hard-negative top-1, not the last step
# GPU kernel. Reads the casmi26-prep output, the casmi26-train-fp dataset (structure masses) and the
# COCONUT candidate dataset. Checkpoints to /kaggle/working as fp_single_own2_s{step}.pt (best) and
# fp_single_own2_last.pt; stops before the weekly quota.
import glob, math, os, time
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

T0 = time.time()
CFG = dict(
    d=512, layers=6, heads=8, drop=0.1, nbits=6930,
    batch=256, K=63, ppm=10.0, lam=1.0, merge_p=0.6,
    lr=3e-4, wd=0.01, warmup=2000, steps=40000, lr_floor=0.02,
    val_every=1000, val_n=2048, time_budget_h=5.3, seed=7,
)
torch.manual_seed(CFG["seed"]); rng = np.random.default_rng(CFG["seed"])
dev = "cuda" if torch.cuda.is_available() else "cpu"
print("device", dev, torch.cuda.get_device_name(0) if dev == "cuda" else "", flush=True)


def find(name):
    hits = glob.glob(f"/kaggle/input/**/{name}", recursive=True)
    assert hits, name
    return sorted(hits, key=len)[0]


# ----------------------------------------------------------------------------- spectra (prep output)
D = os.path.dirname(find("tokens_mz.npy"))
tok_mz = np.ascontiguousarray(np.load(f"{D}/tokens_mz.npy", mmap_mode="r"))
tok_it = np.ascontiguousarray(np.load(f"{D}/tokens_it.npy", mmap_mode="r")).astype(np.float32)
n_peaks = np.load(f"{D}/n_peaks.npy").astype(np.int64); prec = np.load(f"{D}/prec.npy")
adduct = np.load(f"{D}/adduct.npy").astype(np.int64); instr = np.load(f"{D}/instr.npy").astype(np.int64)
ce = np.load(f"{D}/ce.npy"); mode = np.load(f"{D}/mode.npy").astype(np.float32)
fp_idx = np.load(f"{D}/fp_idx.npy"); split = np.load(f"{D}/split.npy")
fp_keys = np.load(f"{D}/fp_keys.npy", allow_pickle=True)
L = tok_mz.shape[1]
NB = CFG["nbits"]
print(f"spectra {len(prec):,} loaded ({time.time()-T0:.0f}s)", flush=True)

# ----------------------------------------------------------------------------- candidate pool for negatives
tfp = np.load(find("train_fp.npz"), allow_pickle=True)
assert (tfp["keys"] == fp_keys).all(), "train_fp.npz and prep fp_keys differ"
cdir = os.path.dirname(find("coco_fp.npy"))
co_fp = np.load(f"{cdir}/coco_fp.npy"); co_mass = np.load(f"{cdir}/coco_mass.npy")
pool_fp = np.vstack([co_fp, tfp["fp"]]).astype(np.uint8)
pool_mass = np.concatenate([co_mass, tfp["mass"]]).astype(np.float64)
good = np.isfinite(pool_mass)
order = np.argsort(pool_mass[good], kind="stable")
pool_fp = pool_fp[good][order]; pool_mass = pool_mass[good][order]
# position of every training structure in the sorted pool
train_rows = np.arange(len(co_fp), len(co_fp) + len(tfp["fp"]))
pos_of_struct = np.full(len(tfp["fp"]), -1, np.int64)
inv = np.empty(good.sum(), np.int64); inv[order] = np.arange(good.sum())
good_idx = np.flatnonzero(good)
gmap = {int(g): i for i, g in enumerate(good_idx)}
for s, r in enumerate(train_rows):
    j = gmap.get(int(r), -1)
    if j >= 0:
        pos_of_struct[s] = inv[j]
PFP = torch.from_numpy(np.ascontiguousarray(pool_fp)).to(dev)          # packed bits, about 620 MB
SHIFTS = torch.arange(7, -1, -1, device=dev, dtype=torch.uint8)
print(f"pool {len(pool_mass):,} structures, train structures with pool position {(pos_of_struct>=0).sum():,} ({time.time()-T0:.0f}s)", flush=True)


def unpack(idx):
    """(...,) pool indices -> (..., NB) float bits, unpacked on the GPU."""
    p = PFP[idx]                                                  # (..., 867) uint8
    bits = ((p.unsqueeze(-1) >> SHIFTS) & 1).reshape(*p.shape[:-1], -1)[..., :NB]
    return bits.float()


usable = (fp_idx >= 0) & (n_peaks > 0) & (pos_of_struct[np.maximum(fp_idx, 0)] >= 0)
train_ix = np.flatnonzero(usable & (split == 0)); val_ix = np.flatnonzero(usable & (split == 1))
spec_pos = pos_of_struct[np.maximum(fp_idx, 0)]                    # pool position per spectrum
print(f"train spectra {len(train_ix):,}  val spectra {len(val_ix):,}", flush=True)

# spectra grouped by structure for merge augmentation
g_order = np.argsort(fp_idx, kind="stable"); g_sorted = fp_idx[g_order]
g_bounds = np.searchsorted(g_sorted, np.arange(fp_idx.max() + 2))


def peers(i, kmax=4):
    s = fp_idx[i]; a, b = g_bounds[s], g_bounds[s + 1]
    if b - a <= 1:
        return [i]
    cand = g_order[a:b]
    k = int(rng.integers(2, min(kmax, b - a) + 1))
    pick = rng.choice(cand, size=min(k, len(cand)), replace=False)
    if i not in pick:
        pick = np.concatenate([[i], pick[:-1]])
    return list(pick)


def merged(idxs):
    mz = np.concatenate([tok_mz[j, :n_peaks[j]] for j in idxs]); it = np.concatenate([tok_it[j, :n_peaks[j]] for j in idxs])
    o = np.argsort(mz); mz, it = mz[o], it[o]
    keep = np.ones(len(mz), bool)
    for j in range(1, len(mz)):
        if mz[j] - mz[j - 1] < 0.005:
            if it[j] >= it[j - 1]: keep[j - 1] = False
            else: keep[j] = False
    mz, it = mz[keep], it[keep]
    if len(mz) > L:
        top = np.argsort(-it)[:L]; top.sort(); mz, it = mz[top], it[top]
    return mz, it


def sample_negs(pos, K):
    m = pool_mass[pos]; tol = m * CFG["ppm"] / 1e6
    lo = np.searchsorted(pool_mass, m - tol, "left"); hi = np.searchsorted(pool_mass, m + tol, "right")
    out = np.empty((len(pos), K), np.int64)
    for r, (a, b, p) in enumerate(zip(lo, hi, pos)):
        if b - a <= 1:
            out[r] = rng.integers(0, len(pool_mass), K)           # no isomers in the window: random decoys
        else:
            c = rng.integers(a, b, K); bad = c == p
            if bad.any(): c[bad] = np.where(c[bad] + 1 < b, c[bad] + 1, a)
            out[r] = c
    return out


def batch(ids, K, aug, merge_p):
    B = len(ids)
    mz = tok_mz[ids].copy(); it = tok_it[ids].copy()
    pad = np.arange(L)[None, :] >= n_peaks[ids][:, None]
    if merge_p > 0:
        for r, i in enumerate(ids):
            if rng.random() >= merge_p: continue
            pk = peers(i)
            if len(pk) < 2: continue
            m2, i2 = merged(pk); n2 = len(m2)
            mz[r] = 0; it[r] = 0; pad[r] = True
            mz[r, :n2] = m2; it[r, :n2] = i2; pad[r, :n2] = False
    if aug:
        keep = rng.random((B, L)) > rng.uniform(0.0, 0.30, size=(B, 1))
        pad = pad | (~keep)
        it = it * np.exp(rng.normal(0.0, 0.25, size=(B, L))).astype(np.float32)
        mz = mz * (1.0 + rng.normal(0.0, 5e-6, size=(B, L))).astype(np.float32)
        allpad = pad.all(1)
        if allpad.any(): pad[allpad, 0] = False
    T = lambda x: torch.as_tensor(x, device=dev)
    inp = (T(mz), T(it), T(pad), T(prec[ids]), T(adduct[ids]), T(instr[ids]), T(ce[ids]), T(mode[ids]))
    pos = spec_pos[ids]
    cand = torch.as_tensor(np.concatenate([pos[:, None], sample_negs(pos, K)], 1), device=dev)   # (B, K+1), col 0 true
    ypos = unpack(torch.as_tensor(pos, device=dev))
    return inp, ypos, cand


# ----------------------------------------------------------------------------- model (public FPNet layout)
ADDUCT_N, INSTR_N = 26, 5


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
        B, N, D_ = x.shape; y = self.n1(x)
        q, k, v = self.qkv(y).view(B, N, 3, self.h, D_ // self.h).permute(2, 0, 3, 1, 4)
        m = (~pad)[:, None, None, :]
        a = F.scaled_dot_product_attention(q, k, v, attn_mask=m)
        x = x + self.drop(self.o(a.transpose(1, 2).reshape(B, N, D_)))
        return x + self.drop(self.ff(self.n2(x)))


class FPNet(nn.Module):
    def __init__(self, nbits, d=512, layers=6, heads=8, drop=0.1):
        super().__init__()
        self.d = d
        self.mz_emb = SinEmb(d); self.nl_emb = SinEmb(d)
        self.pk = nn.Linear(2 * d + 1, d)
        self.prec_emb = SinEmb(d)
        self.ad = nn.Embedding(ADDUCT_N, d); self.ins = nn.Embedding(INSTR_N, d)
        self.gl = nn.Linear(d + 3, d)
        self.blocks = nn.ModuleList([Block(d, heads, drop) for _ in range(layers)])
        self.norm = nn.LayerNorm(d)
        self.head = nn.Sequential(nn.Linear(2 * d, 2048), nn.GELU(), nn.Dropout(drop), nn.Linear(2048, nbits))
    def forward(self, mz, it, pad, prec, ad, ins, ce, mode):
        B, N = mz.shape
        nl = (prec[:, None] - mz).clamp(min=0)
        p = self.pk(torch.cat([self.mz_emb(mz), self.nl_emb(nl), it.unsqueeze(-1)], -1))
        g = self.gl(torch.cat([self.prec_emb(prec), (ce / 100.0).unsqueeze(-1), mode.unsqueeze(-1),
                               torch.log1p(prec).unsqueeze(-1) / 10.0], -1)) + self.ad(ad) + self.ins(ins)
        x = torch.cat([g.unsqueeze(1), p], 1)
        pad = torch.cat([torch.zeros(B, 1, dtype=torch.bool, device=pad.device), pad], 1)
        for b in self.blocks: x = b(x, pad)
        x = self.norm(x)
        cls = x[:, 0]
        msk = (~pad[:, 1:]).float().unsqueeze(-1)
        mean = (x[:, 1:] * msk).sum(1) / msk.sum(1).clamp(min=1)
        return self.head(torch.cat([cls, mean], -1))


model = FPNet(NB, CFG["d"], CFG["layers"], CFG["heads"], CFG["drop"]).to(dev)
print(f"parameters {sum(p.numel() for p in model.parameters())/1e6:.1f}M", flush=True)
opt = torch.optim.AdamW(model.parameters(), lr=CFG["lr"], weight_decay=CFG["wd"], betas=(0.9, 0.98))
scaler = torch.cuda.amp.GradScaler(enabled=(dev == "cuda"))


def lr_at(s):
    if s < CFG["warmup"]:
        return CFG["lr"] * s / CFG["warmup"]
    p = (s - CFG["warmup"]) / max(1, CFG["steps"] - CFG["warmup"])
    return CFG["lr"] * (CFG["lr_floor"] + (1 - CFG["lr_floor"]) * 0.5 * (1 + math.cos(math.pi * min(p, 1.0))))


def scores(z, cand):
    raw = torch.bmm(unpack(cand), z.unsqueeze(-1)).squeeze(-1)             # (B, K+1) bits dot logits
    return (raw - raw.mean(dim=1, keepdim=True)) / math.sqrt(NB)


@torch.no_grad()
def validate():
    model.eval(); vb, vacc = [], []
    vids = rng.choice(val_ix, size=min(CFG["val_n"], len(val_ix)), replace=False)
    for s in range(0, len(vids), 128):
        inp, ypos, cand = batch(vids[s:s + 128], CFG["K"], aug=False, merge_p=CFG["merge_p"])
        with torch.autocast(device_type="cuda", dtype=torch.float16, enabled=(dev == "cuda")):
            z = model(*inp).float()
        vb.append(F.binary_cross_entropy_with_logits(z, ypos).item())
        vacc.append((scores(z, cand).argmax(1) == 0).float().mean().item())
    model.train()
    return float(np.mean(vb)), float(np.mean(vacc))


def save(path, step):
    torch.save({"model": model.state_dict(), "nbits": NB, "d": CFG["d"], "layers": CFG["layers"], "step": step, "cfg": CFG}, path)


# ----------------------------------------------------------------------------- training
best, best_path = -1.0, None
rb = rc = racc = 0.0; nr = 0
for step in range(CFG["steps"]):
    for g in opt.param_groups: g["lr"] = lr_at(step)
    ids = rng.choice(train_ix, size=CFG["batch"], replace=False)
    inp, ypos, cand = batch(ids, CFG["K"], aug=True, merge_p=CFG["merge_p"])
    with torch.autocast(device_type="cuda", dtype=torch.float16, enabled=(dev == "cuda")):
        z = model(*inp)
    z = z.float()
    lb = F.binary_cross_entropy_with_logits(z, ypos)
    sc = scores(z, cand)
    lc = F.cross_entropy(sc, torch.zeros(len(ids), dtype=torch.long, device=dev))
    loss = lb + CFG["lam"] * lc
    if not torch.isfinite(loss):
        print(f"non-finite loss at step {step}, skipped", flush=True); opt.zero_grad(set_to_none=True); continue
    opt.zero_grad(set_to_none=True)
    scaler.scale(loss).backward(); scaler.unscale_(opt)
    torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
    scaler.step(opt); scaler.update()
    rb += lb.item(); rc += lc.item(); racc += (sc.argmax(1) == 0).float().mean().item(); nr += 1
    if step % 200 == 0:
        el = time.time() - T0
        print(f"step {step} bce {rb/max(nr,1):.4f} ctr {rc/max(nr,1):.4f} top1 {racc/max(nr,1):.3f} lr {lr_at(step):.2e} {el:.0f}s {(step+1)*CFG['batch']/max(el,1):.0f} spec/s", flush=True)
        rb = rc = racc = 0.0; nr = 0
    if (step + 1) % CFG["val_every"] == 0:
        vb, va = validate()
        tag = ""
        if va > best:
            best = va
            if best_path and os.path.exists(best_path): os.remove(best_path)
            best_path = f"/kaggle/working/fp_single_own2_s{step+1}.pt"; save(best_path, step + 1); tag = "  <- best, saved"
        save("/kaggle/working/fp_single_own2_last.pt", step + 1)
        print(f"  VAL step {step+1} bce {vb:.4f} hardneg-top1 {va:.3f}{tag}  ({time.time()-T0:.0f}s)", flush=True)
    if (time.time() - T0) / 3600 > CFG["time_budget_h"]:
        print("time budget reached", flush=True); break
print(f"done, best hardneg-top1 {best:.3f} at {os.path.basename(best_path) if best_path else 'none'}, {time.time()-T0:.0f}s")
