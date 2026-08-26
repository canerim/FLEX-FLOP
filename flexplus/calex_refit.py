"""Refit each adapter on the tiles that actually reach it, in closed form.

The ladder is trained under a uniform exit distribution and deployed under one
that is not uniform at all: at the 0.1 dB operating point, 71% of tiles leave
at e2, 21% at e3, and the two rungs below the split take 0%. Measured as a
divergence, KL(deployed || training) = 0.966 nat. Exit 2 carries seven tiles in
ten and receives one sixth of the auxiliary gradient.

This is the covariate shift CalexNet (arXiv:2509.08318) names for post-training
early-exit branches -- "branches train on samples they will never see at
inference" -- and its fix is importance weighting onto the distribution that
actually reaches each branch. The fix here is the same, but it needs no
training at all, because every adapter in this decoder ends in a pointwise
linear map:

    Adapter_k(f) = f + W_k h_k(f),      h_k = f            (Conv1x1Adapter)
                                        h_k = act(pw_in f) (FFNAdapter)

and the target is known: the deepest exit's raw feature, which is what the
shared head was fitted to consume and what makes e5 bit-exact stock UF. So
solving for W_k is ridge regression, in the manner of AdaRound and BRECQ
(arXiv:2102.05426), which reconstruct a frozen network layer by layer on a
small calibration set instead of retraining end to end.

    W_k* = argmin_W  sum_{x in S_k} || W h_k(x) - (f_{K-1}(x) - f_k(x)) ||^2
                     + mu ||W||^2

S_k is the whole point: only the tiles whose Lagrangian allocation actually
sends them to exit k, at the lambda the paper's own bisection chose for that
rate. Fit on all tiles instead and this reproduces the average the training
already optimised.

One thing that looks like it should matter and provably does not. The quantity
we care about is the head's output, not the feature, so the natural objective
carries the head's Jacobian as a metric, ||J(Wh - r)||^2. For an unregularised
least squares that metric cancels: min_W ||(HW - R) M||_F^2 has the same
solution as min_W ||HW - R||_F^2 for any full-rank M, because the minimiser
already zeroes the normal equations column by column. So the second-order
weighting BRECQ needs for quantisation is not needed here, and pretending
otherwise would be decoration.

A 256 px crop decoded whole IS a deployed tile -- same size, same replicate
padding at its border -- so the calibration features carry the seam the
adapters exist to absorb. Nothing under ~/FLEX-UF is written.
"""
from __future__ import annotations

import argparse, json, sys, time
from pathlib import Path

import numpy as np
import torch
from PIL import Image

UF = Path.home() / "FLEX-UF"
sys.path.insert(0, str(UF)); sys.path.insert(0, str(Path.home() / "DCVC"))
from src.utils.transforms import rgb2ycbcr_np                       # noqa: E402
from flexuf.config import FlexUFConfig                              # noqa: E402
from flexuf.cost import exit_costs                                  # noqa: E402
from flexuf.model import FlexUFIntra, load_flexuf_state             # noqa: E402

HERE = Path(__file__).resolve().parent
RES = HERE / "results"
OI = Path("/data10/shareddata/openimages/dcvc_train")


def crops(n, size, seed=0):
    """n square crops from OpenImages, in the decoder's own colour domain."""
    g = torch.Generator().manual_seed(seed)
    paths = []
    for sub in sorted(OI.glob("train_*")):
        paths += sorted(sub.glob("*.jpg"))[: max(1, n // 4)]
        if len(paths) > 4 * n:
            break
    idx = torch.randperm(len(paths), generator=g)[: 3 * n].tolist()
    out = []
    for i in idx:
        try:
            im = Image.open(paths[i]).convert("RGB")
        except Exception:
            continue
        a = np.asarray(im, dtype=np.float32) / 255.0
        if a.shape[0] < size or a.shape[1] < size:
            continue
        y = torch.randint(0, a.shape[0] - size + 1, (1,), generator=g).item()
        x = torch.randint(0, a.shape[1] - size + 1, (1,), generator=g).item()
        a = a[y:y + size, x:x + size]
        out.append(torch.as_tensor(rgb2ycbcr_np(a) - 0.5,
                                   dtype=torch.float32).permute(2, 0, 1)[None])
        if len(out) == n:
            break
    return out


def hidden(ad, f):
    """The vector the adapter's final linear map consumes."""
    if hasattr(ad, "pw_in"):
        return ad.act(ad.pw_in(f))
    return f


def set_linear(ad, W, b):
    lin = ad.pw_out if hasattr(ad, "pw_out") else ad.conv
    with torch.no_grad():
        lin.weight.copy_(W.t().reshape(lin.weight.shape).to(lin.weight.dtype))
        lin.bias.copy_(b.to(lin.bias.dtype))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", default=str(UF / "runs/RECIPE512/ckpt_PIN_e9.pth.tar"))
    ap.add_argument("--lams", default=str(UF / "results/signalled_RECIPE512_0825_2350.json"))
    ap.add_argument("--crops", type=int, default=1500)
    ap.add_argument("--size", type=int, default=256)
    ap.add_argument("--qps", type=int, nargs="+", default=[0, 16, 32, 48, 63])
    ap.add_argument("--mu", type=float, default=1e-3, help="ridge, relative to tr(G)/d")
    ap.add_argument("--target_usage", default="0,0,0.710,0.211,0.043,0.036",
                    help="the DEPLOYED exit usage to reproduce on the "
                         "calibration set. The operating point is a budget, "
                         "not a lambda: the lambda that delivers 0.1 dB on "
                         "1080p CTC frames sends 256 px OpenImages crops far "
                         "deeper, because the crops are much harder per pixel. "
                         "Holding lambda fixed would fit each adapter on the "
                         "wrong survivors. Empty keeps the paper's lambda.")
    ap.add_argument("--device", default="cuda:0")
    ap.add_argument("--out", default=str(Path.home() / "FLEX-PLUS/runs/CALEX_A/ckpt.pth.tar"))
    a = ap.parse_args()
    dev = torch.device(a.device)

    ck = torch.load(a.ckpt, map_location="cpu", weights_only=False)
    cfg = FlexUFConfig(**ck["config"])
    net = FlexUFIntra(cfg).to(dev).eval(); load_flexuf_state(net, ck)
    dec, K, j = net.dec, cfg.num_exits, cfg.split_depth
    cost = exit_costs(cfg, "head").to(dev)

    d = json.loads(Path(a.lams).read_text())
    lam = {r["qp"]: r["lam"] for r in d["rows"]
           if r.get("budget_reachable") and abs(r.get("budget_db", 0) - 0.1) < 1e-9}
    print(f"  epoch {ck.get('epoch')}, K={K} j={j}, {a.crops} kirpim {a.size}px, "
          f"lambda kaynagi epoch {d.get('ckpt_epoch')}", flush=True)

    targets = [k for k in range(j, K - 1)]       # e5 has no adapter
    # Two accumulators per exit. SURV is the experiment: only the tiles whose
    # allocation actually sends them to this exit. POOL is the control: every
    # tile, which is the average the training already optimised. Without the
    # control a gain cannot be attributed to the importance weighting rather
    # than to refitting at all.
    SURV = {k: None for k in targets}
    POOL = {k: None for k in targets}
    n_surv = {k: 0 for k in targets}
    n_pool = {k: 0 for k in targets}
    hist = np.zeros(K)

    xs = crops(a.crops, a.size)
    print(f"  {len(xs)} kirpim yuklendi", flush=True)

    # Pass 1: every crop's per-exit error, so lambda can be moved onto the
    # deployed usage before a single adapter is touched. One trunk pass and
    # six heads per crop; the features are not kept.
    lam_use = dict(lam)
    if a.target_usage:
        tgt = np.array([float(v) for v in a.target_usage.split(",")], dtype=float)
        tgt = tgt / tgt.sum()
        tgt_mean = float((np.arange(K) * tgt).sum())
        M = {q: [] for q in a.qps}
        with torch.no_grad():
            for x in xs:
                x = x.to(dev)
                for q in a.qps:
                    qp = torch.full((1,), q, dtype=torch.int32, device=dev)
                    y, qs, _ = net._encode_to_latent(x, qp)
                    feat = dec.upsample(y)
                    ms = []
                    for g in range(K):
                        feat = dec.groups[g](feat)
                        ms.append(float(((dec._apply_head(dec._at_exit(feat, g), qs)
                                          - x) ** 2).mean()))
                    M[q].append(ms)
        c = cost.cpu().numpy()
        print("  lambda'yi hedef kullanima oturtuyorum "
              f"(hedef ortalama cikis {tgt_mean:.3f}):", flush=True)
        for q in a.qps:
            A = np.asarray(M[q])
            def mean_exit(l):
                k = (A[:, j:] + l * c[None, j:]).argmin(1) + j
                return float(k.mean()), k
            lo, hi = 0.0, 1e-2
            while mean_exit(hi)[0] > tgt_mean and hi < 1e3:
                hi *= 4
            for _ in range(60):
                mid = 0.5 * (lo + hi)
                if mean_exit(mid)[0] > tgt_mean:
                    lo = mid
                else:
                    hi = mid
            lam_use[q] = hi
            _, k = mean_exit(hi)
            h = np.bincount(k, minlength=K) / len(k)
            print(f"   q{q:>3}: lambda {lam[q]:.3e} -> {hi:.3e}   kullanim "
                  + " ".join(f"e{g}:{100*h[g]:.0f}%" for g in range(K)),
                  flush=True)
    lam = lam_use
    t0 = time.time()
    with torch.no_grad():
        for i, x in enumerate(xs):
            x = x.to(dev)
            for q in a.qps:
                qp = torch.full((1,), q, dtype=torch.int32, device=dev)
                y, qs, _ = net._encode_to_latent(x, qp)
                feat = dec.upsample(y)
                fs = []
                for g in range(K):
                    feat = dec.groups[g](feat)
                    fs.append(feat)
                # the tile's own Lagrangian decision, at the paper's lambda
                m = torch.stack([
                    ((dec._apply_head(dec._at_exit(fs[g], g), qs) - x) ** 2).mean()
                    for g in range(K)])
                k = int((m[j:] + lam[q] * cost[j:]).argmin(0)) + j
                hist[k] += 1
                for t in targets:
                    h = hidden(dec.adapters[t], fs[t])[0]
                    r = (fs[K - 1] - fs[t])[0]
                    h = h.reshape(h.shape[0], -1).t().double()   # [N, C]
                    r = r.reshape(r.shape[0], -1).t().double()   # [N, C]
                    h1 = torch.cat([h, torch.ones_like(h[:, :1])], 1)
                    G = h1.t() @ h1; B = h1.t() @ r
                    R2 = float((r * r).sum())
                    if POOL[t] is None:
                        POOL[t] = [G.clone(), B.clone(), R2]
                    else:
                        POOL[t][0] += G; POOL[t][1] += B; POOL[t][2] += R2
                    n_pool[t] += 1
                    if t == k:
                        if SURV[t] is None:
                            SURV[t] = [G, B, R2]
                        else:
                            SURV[t][0] += G; SURV[t][1] += B; SURV[t][2] += R2
                        n_surv[t] += 1
            if (i + 1) % 100 == 0:
                print(f"   {i+1}/{len(xs)} kirpim, {time.time()-t0:.0f}s, "
                      "kalibrasyon kullanimi " +
                      " ".join(f"e{g}:{hist[g]/hist.sum()*100:.0f}%"
                               for g in range(K)), flush=True)

    print("\n  kapali form cozumu (SSE = ozn.-uzayi artik kare toplami):")
    report = {"ckpt": a.ckpt, "ckpt_epoch": ck.get("epoch"), "crops": len(xs),
              "qps": a.qps, "size": a.size, "mu": a.mu,
              "target_usage": a.target_usage,
              "lam_used": {str(q): float(lam[q]) for q in a.qps},
              "calib_usage_pct": (hist / max(hist.sum(), 1) * 100).tolist(),
              "exits": {}}
    solved = {"surv": {}, "pool": {}}

    def fit(acc):
        G, B, R2 = acc
        mu = a.mu * float(torch.diagonal(G).mean())
        W = torch.linalg.solve(
            G + mu * torch.eye(G.shape[0], dtype=G.dtype, device=G.device), B)
        sse = R2 + float((W.t() @ G @ W).diagonal().sum()) - 2 * float((W * B).sum())
        return W, sse

    def sse_of(W, acc):
        G, B, R2 = acc
        return R2 + float((W.t() @ G @ W).diagonal().sum()) - 2 * float((W * B).sum())

    for k in targets:
        ad = dec.adapters[k]
        lin = ad.pw_out if hasattr(ad, "pw_out") else ad.conv
        W_old = torch.cat([lin.weight.reshape(lin.weight.shape[0], -1).t().double(),
                           lin.bias[None].double()], 0).to(POOL[k][0].device)
        rec = {"tiles_survivor": n_surv[k], "tiles_pooled": n_pool[k]}
        if SURV[k] is not None and n_surv[k] >= 8:
            Ws, _ = fit(SURV[k])
            solved["surv"][k] = Ws
            rec["sse_trained_on_survivors"] = sse_of(W_old, SURV[k])
            rec["sse_survivor_fit"] = sse_of(Ws, SURV[k])
        Wp, _ = fit(POOL[k])
        solved["pool"][k] = Wp
        rec["sse_trained_on_all"] = sse_of(W_old, POOL[k])
        rec["sse_pooled_fit"] = sse_of(Wp, POOL[k])
        report["exits"][str(k)] = rec
        s0 = rec.get("sse_trained_on_survivors")
        s1 = rec.get("sse_survivor_fit")
        line = (f"   e{k}: {n_surv[k]:>5} hayatta-kalan / {n_pool[k]:>5} karo   "
                f"havuz {rec['sse_trained_on_all']:.4e} -> {rec['sse_pooled_fit']:.4e}")
        if s0 is not None:
            line += f"   hayatta-kalan {s0:.4e} -> {s1:.4e} ({100*(1-s1/s0):+.2f}%)"
        print(line, flush=True)

    # --out names the run, not a file inside somebody else's run. The first
    # version derived the directory from outdir.parent, so passing a NEW --out
    # wrote into the OLD run's folder and overwrote the checkpoint the
    # headline had been measured on.
    outdir = Path(a.out).parent
    stem = outdir.name
    for name, key in ((f"{stem}_survivor", "surv"), (f"{stem}_pooled", "pool")):
        st = {kk: vv.clone() for kk, vv in net.state_dict().items()}
        net2 = FlexUFIntra(cfg).to(dev).eval(); net2.load_state_dict(st)
        for k, W in solved[key].items():
            set_linear(net2.dec.adapters[k], W[:-1].float(), W[-1].float())
        p = outdir.parent / name / "ckpt.pth.tar"  # sibling of --out's dir
        p.parent.mkdir(parents=True, exist_ok=True)
        torch.save({"state_dict": net2.state_dict(), "config": ck["config"],
                    "epoch": ck.get("epoch"), "calex": report}, p)
        print(f"   yazildi {p}")

    (RES / "calex_refit.json").write_text(json.dumps(report, indent=2))
    print(f"\n  {time.time()-t0:.0f}s, results/calex_refit.json yazildi")


if __name__ == "__main__":
    main()
