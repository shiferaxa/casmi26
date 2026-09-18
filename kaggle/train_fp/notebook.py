# CASMI26: train a spectrum-to-fingerprint transformer (same architecture and checkpoint format as the
# public FPNet, so a checkpoint drops into channel 4 of the pipeline unchanged).
# GPU kernel. Reads the output of the casmi26-prep kernel. Checkpoints every epoch to /kaggle/working
# as fp_single_own_e{N}.pt (only the two most recent are kept), stops before the session limit.
import glob, math, os, time
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

T0 = time.time()
CFG = dict(
    d=512, layers=6, heads=8, drop=0.1, nbits=6930,
    batch=512, lr=3e-4, wd=0.01, warmup=1500, epochs=40,
    time_budget_h=11.3,           # Kaggle GPU session limit is 12 h
    val_spectra=4000, val_pool=4000,
    seed=0,
)
torch.manual_seed(CFG["seed"]); np.random.seed(CFG["seed"])
dev = "cuda" if torch.cuda.is_available() else "cpu"
print("device", dev, torch.cuda.get_device_name(0) if dev == "cuda" else "", flush=True)


def find(name):
    hits = glob.glob(f"/kaggle/input/**/{name}", recursive=True)
    assert hits, name
    return sorted(hits, key=len)[0]


# ----------------------------------------------------------------------------- data
D = os.path.dirname(find("tokens_mz.npy"))
tok_mz = np.load(f"{D}/tokens_mz.npy", mmap_mode="r")
tok_it = np.load(f"{D}/tokens_it.npy", mmap_mode="r")
n_peaks = np.load(f"{D}/n_peaks.npy"); prec = np.load(f"{D}/prec.npy"); adduct = np.load(f"{D}/adduct.npy")
instr = np.load(f"{D}/instr.npy"); ce = np.load(f"{D}/ce.npy"); mode = np.load(f"{D}/mode.npy")
fp_idx = np.load(f"{D}/fp_idx.npy"); split = np.load(f"{D}/split.npy")
fp_packed = np.load(f"{D}/fp_packed.npy")
NB = CFG["nbits"]
usable = (fp_idx >= 0) & (n_peaks > 0)
train_ix = np.flatnonzero(usable & (split == 0))
val_ix = np.flatnonzero(usable & (split == 1))
print(f"train spectra {len(train_ix):,}  val spectra {len(val_ix):,}  structures with fp {len(fp_packed):,}  ({time.time()-T0:.0f}s)", flush=True)
# load token arrays into RAM (about 2 GB) for fast random access
tok_mz = np.ascontiguousarray(tok_mz); tok_it = np.ascontiguousarray(tok_it)
print(f"tokens in RAM ({time.time()-T0:.0f}s)", flush=True)

bit_freq = np.unpackbits(fp_packed, axis=1)[:, :NB].mean(0)
print(f"bit frequency: mean {bit_freq.mean():.4f} min {bit_freq.min():.5f}", flush=True)


def batch_tensors(ix):
    mz = torch.from_numpy(tok_mz[ix]).to(dev, non_blocking=True)
    it = torch.from_numpy(tok_it[ix].astype(np.float32)).to(dev, non_blocking=True)
    npk = torch.from_numpy(n_peaks[ix].astype(np.int64)).to(dev)
    pad = torch.arange(mz.shape[1], device=dev)[None, :] >= npk[:, None]
    y = torch.from_numpy(np.unpackbits(fp_packed[fp_idx[ix]], axis=1)[:, :NB].astype(np.float32)).to(dev, non_blocking=True)
    return (mz, it, pad,
            torch.from_numpy(prec[ix]).to(dev), torch.from_numpy(adduct[ix].astype(np.int64)).to(dev),
            torch.from_numpy(instr[ix].astype(np.int64)).to(dev), torch.from_numpy(ce[ix]).to(dev),
            torch.from_numpy(mode[ix].astype(np.float32)).to(dev)), y


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
steps_per_epoch = len(train_ix) // CFG["batch"]
total_steps = steps_per_epoch * CFG["epochs"]
def lr_at(step):
    if step < CFG["warmup"]:
        return CFG["lr"] * step / CFG["warmup"]
    t = (step - CFG["warmup"]) / max(1, total_steps - CFG["warmup"])
    return CFG["lr"] * 0.5 * (1 + math.cos(math.pi * min(1.0, t)))
scaler = torch.cuda.amp.GradScaler(enabled=(dev == "cuda"))


# ----------------------------------------------------------------------------- validation
rng = np.random.default_rng(1)
val_spec = rng.choice(val_ix, size=min(CFG["val_spectra"], len(val_ix)), replace=False)
val_struct = np.unique(fp_idx[val_ix])
val_pool_struct = rng.choice(val_struct, size=min(CFG["val_pool"], len(val_struct)), replace=False)
val_pool_set = set(val_pool_struct.tolist())
val_spec = np.array([i for i in val_spec if fp_idx[i] in val_pool_set]) if len(val_pool_struct) < len(val_struct) else val_spec
pool_bits = torch.from_numpy(np.unpackbits(fp_packed[val_pool_struct], axis=1)[:, :NB].astype(np.float32)).to(dev)
pool_pos = {s: i for i, s in enumerate(val_pool_struct.tolist())}


@torch.no_grad()
def validate():
    """BCE on validation spectra and retrieval MRR of the true structure among the validation pool by Bernoulli score."""
    model.eval()
    bce, n, rr = 0.0, 0, []
    for s in range(0, len(val_spec), 1024):
        ix = val_spec[s:s + 1024]
        x, y = batch_tensors(ix)
        with torch.autocast(device_type="cuda", dtype=torch.float16, enabled=(dev == "cuda")):
            z = model(*x).float()
        bce += F.binary_cross_entropy_with_logits(z, y, reduction="sum").item(); n += y.numel()
        zc = z.clamp(-30, 30)
        score = pool_bits @ zc.T + torch.log1p(-torch.sigmoid(zc)).sum(1)[None, :]   # (pool, B)
        truth = torch.tensor([pool_pos[int(fp_idx[i])] for i in ix], device=dev)
        ts = score[truth, torch.arange(len(ix), device=dev)]
        rank = (score > ts[None, :]).sum(0) + 1
        rr.append(torch.where(rank <= 25, 1.0 / rank.float(), torch.zeros_like(rank, dtype=torch.float)).cpu())
    model.train()
    rr = torch.cat(rr)
    return bce / n, float(rr.mean()), float((rr == 1.0).float().mean())


def save(epoch, step):
    path = f"/kaggle/working/fp_single_own_e{epoch}.pt"
    torch.save({"model": model.state_dict(), "nbits": NB, "d": CFG["d"], "layers": CFG["layers"], "step": step,
                "epoch": epoch, "cfg": CFG}, path)
    old = sorted(glob.glob("/kaggle/working/fp_single_own_e*.pt"), key=os.path.getmtime)[:-2]
    for p in old:
        os.remove(p)
    return path


# ----------------------------------------------------------------------------- training
step = 0
bce, mrr, r1 = validate()
print(f"epoch 0  val bce {bce:.4f}  pool-mrr {mrr:.4f}  r1 {r1:.3f}  ({time.time()-T0:.0f}s)", flush=True)
for epoch in range(1, CFG["epochs"] + 1):
    perm = np.random.permutation(train_ix)
    run_loss, t_ep = 0.0, time.time()
    for b in range(steps_per_epoch):
        ix = np.sort(perm[b * CFG["batch"]:(b + 1) * CFG["batch"]])
        x, y = batch_tensors(ix)
        for g in opt.param_groups:
            g["lr"] = lr_at(step)
        with torch.autocast(device_type="cuda", dtype=torch.float16, enabled=(dev == "cuda")):
            z = model(*x)
        loss = F.binary_cross_entropy_with_logits(z.float(), y)
        opt.zero_grad(set_to_none=True)
        scaler.scale(loss).backward()
        scaler.unscale_(opt)
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        scaler.step(opt); scaler.update()
        run_loss += loss.item(); step += 1
        if b % 500 == 0:
            print(f"  epoch {epoch} step {b}/{steps_per_epoch} loss {run_loss/(b+1):.4f} lr {lr_at(step):.2e} ({time.time()-T0:.0f}s)", flush=True)
        if (time.time() - T0) / 3600 > CFG["time_budget_h"]:
            break
    bce, mrr, r1 = validate()
    path = save(epoch, step)
    print(f"epoch {epoch}  train loss {run_loss/max(1,b+1):.4f}  val bce {bce:.4f}  pool-mrr {mrr:.4f}  r1 {r1:.3f}  "
          f"epoch time {time.time()-t_ep:.0f}s  saved {os.path.basename(path)}  ({time.time()-T0:.0f}s)", flush=True)
    if (time.time() - T0) / 3600 > CFG["time_budget_h"]:
        print("time budget reached, stopping", flush=True)
        break
print(f"done in {time.time()-T0:.0f}s")
