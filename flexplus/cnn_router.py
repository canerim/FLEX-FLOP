"""Is the missing 1.57 points inside the cell, where pooling threw it away?

At j=0 the deployable router reaches 31.89% against an oracle of 36.02, and a
PERFECT rank-1 predictor would already reach 33.46. So 1.57 points are lost
before shape is even considered: the current feature set cannot recover the
scale. Every one of those features is a pooled statistic of the cell -- a mean,
a std, a gradient magnitude -- and at j=0 the decisive question is whether a
cell survives on two blocks, which plausibly depends on structure inside the
cell that pooling destroys.

This tests that directly. Architecture: ClassSR's Class-Module, five
convolutions and a pool, reading the cell's stem patch rather than statistics
of it. Supervision: ours -- the measured per-exit error, regressed in log
space -- rather than ClassSR's two label-free losses, which the previous
experiment measured 10.3 points behind.

So the comparison isolates one thing. Same supervision as the gradient-boosted
router, same budget, same scoring; only the input representation changes.

Nothing under ~/FLEX-UF is modified.
"""
from __future__ import annotations

import argparse, json, sys, time
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

UF = Path.home() / "FLEX-UF"
HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(UF)); sys.path.insert(0, str(UF / "scripts"))
sys.path.insert(0, str(Path.home() / "DCVC")); sys.path.insert(0, str(HERE))

import ctc_intra as C                                          # noqa: E402
from flexuf.config import FlexUFConfig                         # noqa: E402
from flexuf.model import FlexUFIntra, load_flexuf_state        # noqa: E402
from flexuf.reference import reference_for                     # noqa: E402
from classsr_router import ClassModule, cells_of, cell_mse, frames_openimages  # noqa: E402
from adaptive_depth import dilate_depth                        # noqa: E402


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", default=str(UF / "runs/RECIPE512/ckpt_PAPER.pth.tar"))
    ap.add_argument("--device", default="cuda:0")
    ap.add_argument("--cell", type=int, default=64)
    ap.add_argument("--images", type=int, default=500)
    ap.add_argument("--crop", type=int, default=512)
    ap.add_argument("--epochs", type=int, default=6)
    ap.add_argument("--lr", type=float, default=1e-3)
    ap.add_argument("--split", type=int, default=0)
    ap.add_argument("--qps", type=int, nargs="+", default=[0, 16, 32, 48, 63])
    ap.add_argument("--out", default=str(HERE / "results/cnn_router_j0.json"))
    a = ap.parse_args()

    dev = torch.device(a.device)
    ck = torch.load(a.ckpt, map_location="cpu", weights_only=False)
    cfg = FlexUFConfig(**ck["config"]) if "config" in ck else FlexUFConfig()
    net = FlexUFIntra(cfg).to(dev).eval(); load_flexuf_state(net, ck)
    ref = FlexUFIntra(cfg).to(dev).eval()
    load_flexuf_state(ref, torch.load(reference_for(cfg, None),
                                      map_location="cpu", weights_only=False))
    for p in net.parameters():
        p.requires_grad_(False)
    K, j = cfg.num_exits, a.split
    cf = a.cell // (cfg.rgb_patch // cfg.feature_patch)
    cc = json.loads((HERE / "results/cost_constants.json").read_text())
    UPS, TRUNK, HEAD, NBLK = (cc["SHARE_UPSAMPLE"], cc["SHARE_TRUNK"],
                              cc["SHARE_HEAD"], cc["N_TRUNK_BLOCKS"])
    ADAPT = torch.tensor(cc["adapter"], dtype=torch.float32)
    cost_k = torch.tensor([(k + 1) * cfg.blocks_per_exit for k in range(K)],
                          dtype=torch.float32, device=dev)

    torch.manual_seed(0)
    train = frames_openimages(a.images, a.crop, dev)
    seqs, _ = C.discover([])
    test = [x[0:1] for x in
            (C.read_frames(s["path"], s["w"], s["h"], 1, 1)[0] for s in seqs)
            if x is not None] if False else []
    for s in seqs:
        x, _ = C.read_frames(s["path"], s["w"], s["h"], 1, 1)
        if x is not None:
            test.append(x[0:1])
    print(f"  {len(train)} egitim, {len(test)} test karesi, j={j}, "
          f"hucre {a.cell}px -> {cf}x{cf}", flush=True)

    @torch.no_grad()
    def tables(x, qp_v):
        x = x.to(dev)
        ph, pw = (-x.shape[-2]) % a.cell, (-x.shape[-1]) % a.cell
        xp = F.pad(x, (0, pw, 0, ph), mode="replicate") if (ph or pw) else x
        qp = torch.full((1,), qp_v, dtype=torch.int32, device=dev)
        y, q, _ = net._encode_to_latent(xp, qp)
        stem = net.dec.upsample(y)
        recs = net.dec.forward_all_exits(y, q)
        rr = ref.dec.forward_full(y, q)
        cells, nh, nw = cells_of(stem, cf)
        M = torch.stack([cell_mse(r, xp, a.cell) for r in recs], 1)
        R = cell_mse(rr, xp, a.cell)
        n = min(cells.shape[0], M.shape[0])
        return cells[:n], M[:n], R[:n], nh, nw

    ch = None
    for x in train[:1]:
        ch = tables(x, 32)[0].shape[1]
    # K outputs: log m_k for every exit, which is exactly what the Lagrangian
    # consumes, so this router plugs into the same sweep as the others.
    cm = ClassModule(ch, K, width=32, min_exit=0).to(dev)
    npar = sum(p.numel() for p in cm.parameters())
    opt = torch.optim.Adam(cm.parameters(), lr=a.lr)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(
        opt, T_max=max(1, a.epochs * len(train)))
    # log m sits near -9, so an output head initialised at zero starts nine
    # units away from every target and spends its first epochs learning a
    # constant. Standardise with fixed statistics estimated once, and undo
    # them at inference, so the network only has to learn the part that varies.
    with torch.no_grad():
        samp = torch.cat([torch.log(tables(x, 32)[1].clamp_min(1e-12))
                          for x in train[:24]])
        MU = samp.mean().item(); SD = samp.std().item()
    print(f"  CNN yonlendirici: {npar:,} parametre   hedef olcegi "
          f"mu={MU:.3f} sd={SD:.3f}", flush=True)

    t0 = time.time()
    for ep in range(a.epochs):
        tot = n = 0.0
        for x in train:
            qv = int(a.qps[torch.randint(0, len(a.qps), (1,)).item()])
            cells, M, R, _, _ = tables(x, qv)
            pred = cm(cells)
            tgt = (torch.log(M.clamp_min(1e-12)) - MU) / SD
            loss = F.mse_loss(pred[:, j:], tgt[:, j:])
            opt.zero_grad(set_to_none=True); loss.backward(); opt.step(); sched.step()
            tot += loss.item(); n += 1
        print(f"   epoch {ep}: log-MSE {tot/n:.4f}  ({time.time()-t0:.0f}s)",
              flush=True)

    cm.eval()
    out = {"params": npar, "split_depth": j, "cell_px": a.cell,
           "epochs": a.epochs, "images": len(train), "per_qp": {}}
    with torch.no_grad():
        for qv in a.qps:
            P, Mt, Rt, shapes = [], [], [], []
            for x in test:
                cells, M, R, nh, nw = tables(x, qv)
                P.append(cm(cells)); Mt.append(M); Rt.append(R)
                shapes.append((nh, nw))

            def at(lam):
                ks, dbs = [], []
                for p, M, R in zip(P, Mt, Rt):
                    mhat = (p[:, j:] * SD + MU).exp()
                    k = (mhat + lam * cost_k[None, j:]).argmin(1) + j
                    ks.append(k)
                    dbs.append((10 * torch.log10(
                        M.gather(1, k[:, None]).squeeze(1).mean() / R.mean())).item())
                return sum(dbs) / len(dbs), ks

            lo, hi = 0.0, 1e-8
            while at(hi)[0] <= 0.10 and hi < 1e6:
                hi *= 4
            for _ in range(40):
                mid = 0.5 * (lo + hi)
                if at(mid)[0] <= 0.10:
                    lo = mid
                else:
                    hi = mid
            db, ks = at(lo)
            tot_c = 0.0
            for k, (nh, nw) in zip(ks, shapes):
                e = k.reshape(nh, nw).float()
                e = e.repeat_interleave(cf, 0).repeat_interleave(cf, 1)
                D = dilate_depth((e + 1.0) * cfg.blocks_per_exit)
                ran = sum(int((D >= g * cfg.blocks_per_exit + 1 - 1e-6).sum())
                          * cfg.blocks_per_exit for g in range(K))
                tot_c += (UPS + TRUNK * (ran / D.numel()) / NBLK
                          + ADAPT[k.cpu()].mean().item() + HEAD)
            sv = 100 * (1 - tot_c / len(ks))
            out["per_qp"][qv] = {"saving_pct": sv, "db": db}
            print(f"   qp{qv:>3}: {sv:6.2f}%  dB {db:.4f}", flush=True)
    m = float(np.mean([v["saving_pct"] for v in out["per_qp"].values()]))
    out["mean_saving_pct"] = m
    print(f"\n  ortalama {m:.2f}%")
    Path(a.out).write_text(json.dumps(out, indent=2))
    print(f"  yazildi {a.out}")


if __name__ == "__main__":
    main()
