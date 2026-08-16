"""Can ANYTHING in the stem see the oracle's choice, or is the bottleneck the signals?

The router picks an exit per tile from six hand-chosen scalars. Measured on the
warm-start checkpoint, none of them correlates past |r| = 0.12 with the oracle's
choice, while the oracle itself varies (std 0.348 exits). So the content contains
the distinction and the router cannot see it -- which is exactly why the first
regret sweep produced a constant router.

The obvious suspect is the bottleneck: six numbers to describe a 384-channel,
32x32 tile. The stem feature is already computed, so pooling it per tile costs
nothing. This fits a linear probe from each candidate representation to the
oracle's choice and reports how often it agrees, against the base rate of always
guessing the most common exit. A probe is the right instrument here: it is the
BEST a linear router could do given the representation, so a low number
convicts the representation rather than the training.
"""
import argparse, sys
from pathlib import Path
import torch
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
sys.path.insert(0, str(Path.home() / "DCVC"))
from torch.utils.data import DataLoader, SequentialSampler
from src.datasets.image_dataset import ImageFolder
from src.utils.common import get_training_lambdas
from flexuf.config import QP_LEVELS, FlexUFConfig
from flexuf.cost import exit_costs
from flexuf.model import FlexUFIntra, load_flexuf_state, DeterministicCrop
from flexuf.router.router import stem_signals

ap = argparse.ArgumentParser()
ap.add_argument("--ckpt", default="runs/warmstart/ckpt_warmstart.pth.tar")
ap.add_argument("--qp", type=int, default=32)
ap.add_argument("--patch", type=int, default=16)
ap.add_argument("--batches", type=int, default=40)
ap.add_argument("--device", default="cuda:4")
a = ap.parse_args()
dev = a.device

ck = torch.load(a.ckpt, map_location="cpu", weights_only=False)
cfg = FlexUFConfig(split_depth=2, latent_patch=a.patch, seam_repair="none")
net = FlexUFIntra(cfg).to(dev).eval(); load_flexuf_state(net, ck)

D = "/data10/shareddata/openimages/dcvc_train"
ds = ImageFolder(D, 512, 512, QP_LEVELS, get_training_lambdas([10., 2048.], QP_LEVELS))
import json as _j
ds.dataset = _j.loads((Path(D) / "description_val.json").read_text())[: a.batches * 4]
ds.dataset_length = len(ds.dataset)
ld = DataLoader(DeterministicCrop(ds), batch_size=4, num_workers=3,
                sampler=SequentialSampler(ds))

P = cfg.rgb_patch
S6, POOL, MSE = [], [], []
with torch.no_grad():
    for b in ld:
        x = b[0].to(dev); B = x.shape[0]
        qp = torch.full((B,), a.qp, dtype=torch.int32, device=dev)
        y, q, aux = net._encode_to_latent(x, qp)
        stem = net.dec.upsample(y)
        for g in range(cfg.split_depth):
            stem = net.dec.groups[g](stem)
        sc = aux["scales_hat"][:, : y.shape[1]]
        S6.append(stem_signals(stem, y, sc, cfg).cpu())
        # Per-tile mean AND std of the stem feature: 768 numbers describing the
        # tile instead of 6. Free -- the stem is already in memory.
        fp = cfg.feature_patch
        v = stem.view(B, stem.shape[1], stem.shape[2] // fp, fp, stem.shape[3] // fp, fp)
        v = v.permute(0, 2, 4, 1, 3, 5).reshape(-1, stem.shape[1], fp * fp)
        POOL.append(torch.cat([v.mean(-1), v.std(-1)], 1).cpu())
        outs = net.dec.forward_all_exits(y, q)
        nh, nw = x.shape[-2] // P, x.shape[-1] // P
        per = []
        for xh in outs:
            e = ((xh - x) ** 2).mean(1)
            per.append(e.view(B, nh, P, nw, P).permute(0, 1, 3, 2, 4)
                        .reshape(B * nh * nw, P * P).mean(1))
        MSE.append(torch.stack(per, 1).cpu())

S6 = torch.cat(S6); POOL = torch.cat(POOL); MSE = torch.cat(MSE)
C = exit_costs(cfg, "head").cpu()
print(f"  {MSE.shape[0]} tile, {cfg.num_exits} cikis, qp{a.qp}, {P}px tile\n")
print(f"  {'lambda':>9}{'oracle dagilimi':>34}{'6 skaler':>10}{'768 havuz':>11}{'taban':>8}")
for lam in (3e-4, 1e-3, 3e-3):
    k = (MSE + lam * C[None, :]).argmin(1)
    hist = torch.bincount(k, minlength=cfg.num_exits).tolist()
    base = max(hist) / len(k)
    row, realised = [], []
    for X in (S6, POOL):
        # HELD OUT. The pooled representation has 768 features; with a few
        # hundred tiles a linear probe can simply memorise them, and the first
        # version of this script duly reported 1.000 on data it had fitted.
        # A number that clean should be disbelieved before it is repeated.
        n = len(X); ntr = int(0.7 * n)
        g = torch.Generator().manual_seed(0)
        perm = torch.randperm(n, generator=g)
        tr, te = perm[:ntr], perm[ntr:]
        mu, sd = X[tr].mean(0), X[tr].std(0).clamp_min(1e-8)
        Xn = torch.cat([(X - mu) / sd, torch.ones(n, 1)], 1)
        Y = torch.zeros(n, cfg.num_exits); Y[torch.arange(n), k] = 1
        W = torch.linalg.solve(Xn[tr].T @ Xn[tr] + 1.0 * torch.eye(Xn.shape[1]),
                               Xn[tr].T @ Y[tr])
        pred = (Xn[te] @ W).argmax(1)
        row.append((pred == k[te]).float().mean().item())
        # Accuracy is not the deliverable. What matters is where the wrong tiles
        # went: a miss to the neighbouring exit costs almost nothing, a miss
        # across the ladder costs a lot. So the realised (saving, dB) of the
        # probe's OWN choices is computed and compared against the oracle's on
        # the same held-out tiles.
        mse_o = MSE[te].gather(1, k[te][:, None]).mean()
        mse_p = MSE[te].gather(1, pred[:, None]).mean()
        sv_o = 100 * (1 - C[k[te]].mean())
        sv_p = 100 * (1 - C[pred].mean())
        deep = MSE[te][:, -1].mean()
        db = lambda m: 10 * torch.log10(m / deep)
        realised.append((sv_o.item(), db(mse_o).item(), sv_p.item(), db(mse_p).item()))
    print(f"  {lam:>9.0e}{str(hist):>34}{row[0]:>10.3f}{row[1]:>11.3f}{base:>8.3f}")
    o = realised[0]
    print(f"           gerceklesen -> oracle {o[0]:5.1f}% / {o[1]:+.3f} dB   "
          f"probe(6 skaler) {o[2]:5.1f}% / {o[3]:+.3f} dB   "
          f"probe(768) {realised[1][2]:5.1f}% / {realised[1][3]:+.3f} dB")
print("\n  Probe %70 uzerinde fit edilip AYRILMIS %30 uzerinde raporlaniyor.")
print("  taban = hep en sik cikisi tahmin etmenin isabeti; probe onu gecemiyorsa temsil kor.")
