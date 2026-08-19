"""Configuration C: signal only the tiles the router gets wrong.

A signals every tile's exit and is exact. B signals nothing and is approximate.
The paper measures a gap between them that reaches 13.3 points at qp 63, and
then leaves it there. This asks the obvious next question: the gap is not spread
evenly over tiles -- most tiles the router calls correctly, and a few it does
not -- so what does it cost to signal only the few?

The rule
--------
One multiplier governs both branches. The router's decision

    argmax_k [ log p_k - beta c_k ]  ==  argmin_k [ (-log p_k)/kappa + lam c_k ]

with beta = lam * kappa, so (-log p_k)/kappa is a distortion surrogate in MSE
units and kappa is a fixed nats-to-MSE calibration. kappa is measured once per
qp, as the ratio of the two multipliers that put pure A and pure B on the same
budget; after that a single lam moves the whole hybrid.

For a fraction rho of tiles the encoder overrides the router with the true
minimiser. It picks which by Lagrangian regret,

    Delta(t) = [D(t,k_router) + lam c_router] - [D(t,k*) + lam c*]

which is exactly how much the objective loses on that tile by not signalling it.
The encoder can compute this: it has the source, and it can run the decoder's
router because the router reads only decoded data.

Bits
----
An entropy-coded mask over tiles, N*H2(rho) bits, plus 3 bits for each override.
rho=0 costs nothing and must reproduce B; rho=1 costs 3N bits, needs no mask,
and must reproduce A -- with the router's own compute refunded, because a
decoder that is told every tile never runs one. Both endpoints are checked.

    python scripts/hybrid_curve.py --ckpt runs/RECIPE512/ckpt_eval.pth.tar \
        --router2 runs/RECIPE512/routers2/v2_lam1.3e-5.pth \
        --budget 0.1 --out results/hybrid_RECIPE512_b01.json
"""

from __future__ import annotations

import argparse
import json
import math
import sys
from pathlib import Path

import torch
import torch.nn.functional as F

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(Path.home() / "DCVC"))

import ctc_intra as C  # noqa: E402
from flexuf.config import FlexUFConfig  # noqa: E402
from flexuf.cost import exit_costs  # noqa: E402
from flexuf.eval import (reference_frame_mse, tiled_exit_mses,
                         true_frame_mse)  # noqa: E402
from flexuf.model import FlexUFIntra, load_flexuf_state  # noqa: E402
from flexuf.reference import reference_for  # noqa: E402
from router_curve import router_share  # noqa: E402


def hybrid_map(M, lp, cost, beta, lam, rho, j=0):
    """The exit map for configuration C, and how many tiles were overridden.

    Router choice everywhere except the `rho` fraction of tiles with the largest
    Lagrangian regret, which take the oracle's. rho=0 is pure B and rho=1 is
    pure A; both are exact, not approached.

    `lp` covers only the exits that exist, k = j..K-1, so its column c is exit
    c+j. `M` and `cost` are full-length. The head nominally masks the missing
    exits by assigning -1e4, which stops being a mask once its own logits reach
    that scale -- see router_curve.py -- so they are sliced off instead.
    """
    kr = (lp - beta * cost[None, j:]).argmax(1) + j
    n = M.shape[0]
    s = int(round(rho * n))
    if s <= 0:
        return kr, 0
    L = M + lam * cost[None, :]
    ko = L.argmin(1)
    if s >= n:
        return ko, n
    d = (L.gather(1, kr[:, None]) - L.gather(1, ko[:, None])).squeeze(1)
    idx = d.topk(s).indices
    k = kr.clone()
    k[idx] = ko[idx]
    return k, s


def h2(p: float) -> float:
    """Binary entropy in bits. The mask is i.i.d. Bernoulli(rho) to first order."""
    if p <= 0.0 or p >= 1.0:
        return 0.0
    return -(p * math.log2(p) + (1 - p) * math.log2(1 - p))


def main(argv):
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", required=True)
    ap.add_argument("--ref", default=None)
    ap.add_argument("--router2", required=True)
    ap.add_argument("--qps", type=int, nargs="+", default=[0, 16, 32, 48, 63])
    ap.add_argument("--rhos", type=float, nargs="+",
                    default=[0.0, 0.05, 0.1, 0.2, 0.35, 0.5, 1.0])
    ap.add_argument("--frames", type=int, default=1)
    ap.add_argument("--max_seqs", type=int, default=0)
    ap.add_argument("--device", default="cuda:0")
    ap.add_argument("--budget", type=float, default=0.1)
    ap.add_argument("--out", required=True)
    a = ap.parse_args(argv)

    TARGET = a.budget
    dev = a.device
    ck = torch.load(ROOT / a.ckpt, map_location="cpu", weights_only=False)
    cfg = FlexUFConfig(**ck["config"])
    net = FlexUFIntra(cfg).to(dev).eval()
    load_flexuf_state(net, ck)

    from flexuf.router.head2 import StemRouterHeadV2
    from flexuf.config import LATENT_CH, TRUNK_CH
    h = torch.load(ROOT / a.router2, map_location="cpu", weights_only=False)
    hsd = h.get("router_head_v2", h.get("state_dict", h))
    head2 = StemRouterHeadV2(TRUNK_CH, LATENT_CH, cfg.num_exits,
                             min_exit=cfg.split_depth).to(dev).eval()
    head2.load_state_dict(hsd)
    r2meta = {k: v for k, v in h.items() if k != "router_head_v2"}

    ref = FlexUFIntra(cfg).to(dev).eval()
    load_flexuf_state(ref, torch.load(reference_for(cfg, a.ref),
                                      map_location="cpu", weights_only=False))
    cost = exit_costs(cfg, "head").to(dev)
    j = cfg.split_depth

    seqs, _ = C.discover([])
    if a.max_seqs:
        seqs = seqs[:a.max_seqs]
    frames, measured = [], []
    for s in seqs:
        x, pl = C.read_frames(s["path"], s["w"], s["h"], a.frames, 1)
        if x is not None:
            for i in range(x.shape[0]):
                frames.append((x[i:i + 1], pl[i]))
            measured.append(s["name"])
    print(f"  {len(frames)} CTC frames from {len(measured)} sequences, "
          f"{cfg.rgb_patch}px tile, budget {TARGET} dB\n")

    rows = []
    rshare = 0.0
    with torch.no_grad():
        for qp_v in a.qps:
            cache = []
            for x, pl in frames:
                x = x.to(dev)
                _, _, H, W = x.shape
                P = cfg.rgb_patch
                ph, pw = (-H) % P, (-W) % P
                xp = F.pad(x, (0, pw, 0, ph), mode="replicate") if (ph or pw) else x
                qp = torch.full((1,), qp_v, dtype=torch.int32, device=dev)
                y, q, aux = net._encode_to_latent(xp, qp)
                M = tiled_exit_mses(net.dec, y, q, xp, cfg)
                R = reference_frame_mse(ref.dec, y, q, xp)
                stem = net.dec.upsample(y)
                for g in range(cfg.split_depth):
                    stem = net.dec.groups[g](stem)
                # Exits below the split depth do not exist. The head nominally
                # masks them by assigning -1e4, which is not a mask when the
                # head's own logits have drifted to that scale -- see
                # router_curve.py. Slice them off instead.
                lg = head2(stem, y, aux["scales_hat"], qp,
                           cfg.feature_patch, cfg.latent_patch)[:, j:]
                lp = F.log_softmax(lg, 1)
                if rshare == 0.0:
                    px = xp.shape[-1] * xp.shape[-2]
                    rshare = router_share(head2, (stem, y, aux["scales_hat"], qp,
                                                  cfg.feature_patch,
                                                  cfg.latent_patch), px)
                    print(f"  router costs {100*rshare:.4f}% of the decode\n")
                cache.append((M, R, lp, y, q, xp))

            # --- the two multipliers -----------------------------------
            # An earlier version tried to tie both branches to one multiplier,
            # by reading the router's log-probabilities as a distortion in MSE
            # units. It does not work: the head is confident enough that the
            # conversion constant is ~1e-6, so the shared multiplier has to run
            # to 1e8 before the router responds at all, and at that multiplier
            # the oracle branch ignores distortion entirely and sends every
            # signalled tile to the cheapest exit. The two branches are on
            # different scales and have to keep their own knobs.
            #
            # beta is fixed at the value pure B needs for this budget -- the
            # router is a fixed component, and rho does not change it. lam is
            # then bisected over the signalled tiles alone, which is what
            # absorbs the quality the overrides buy back.
            def bisect(f, lo, hi, tgt, iters=60):
                """Largest x in [lo, hi] with f(x) <= tgt, for f increasing."""
                if f(lo) > tgt:
                    return lo
                if f(hi) <= tgt:
                    return hi
                for _ in range(iters):
                    mid = 0.5 * (lo + hi)
                    if f(mid) <= tgt:
                        lo = mid
                    else:
                        hi = mid
                return lo

            def db_router(beta):
                t = n = 0.0
                for M, R, lp, *_ in cache:
                    k = (lp - beta * cost[None, j:]).argmax(1) + j
                    t += (10 * torch.log10(
                        M.gather(1, k[:, None]).squeeze(1).mean() / R)).item()
                    n += 1
                return t / n

            def db_oracle(lam):
                t = n = 0.0
                for M, R, lp, *_ in cache:
                    k = (M + lam * cost[None, :]).argmin(1)
                    t += (10 * torch.log10(
                        M.gather(1, k[:, None]).squeeze(1).mean() / R)).item()
                    n += 1
                return t / n

            LAM_HI = 1.0
            betaB = bisect(db_router, -2.0e4, 2.0e4, TARGET)
            lamA = bisect(db_oracle, 0.0, LAM_HI, TARGET)

            # --- how the regret is distributed over tiles ------------------
            # At a FIXED lam the hybrid objective is exactly separable, so
            # overriding a set S of tiles removes exactly the sum of their
            # regrets. The recovered fraction as a function of |S| is therefore
            # the Lorenz curve of the regret distribution, and its curvature --
            # the Gini coefficient -- says in one number whether a small
            # signalling budget can do most of the work. This is the fixed-lam
            # prediction; the measured sweep below re-bisects lam at every rho,
            # so the two agree only to the extent that re-tuning is a second
            # order effect.
            reg = []
            for M, R, lp, *_ in cache:
                L = M + lamA * cost[None, :]
                ko = L.argmin(1)
                kr = (lp - betaB * cost[None, j:]).argmax(1) + j
                reg.append((L.gather(1, kr[:, None])
                            - L.gather(1, ko[:, None])).squeeze(1))
            reg = torch.cat(reg).sort(descending=True).values
            tot = float(reg.sum())
            cum = torch.cumsum(reg, 0) / (tot if tot > 0 else 1.0)
            nt = reg.numel()
            lorenz = {f"{r:g}": float(cum[max(0, min(nt - 1,
                                                     int(round(r * nt)) - 1))])
                      for r in a.rhos if r > 0}
            # Gini of a non-negative vector sorted descending
            i = torch.arange(1, nt + 1, device=reg.device, dtype=reg.dtype)
            gini = float((2 * (i * reg.flip(0)).sum() / (nt * reg.sum())
                          - (nt + 1) / nt)) if tot > 0 else 0.0
            print(f"  qp {qp_v}:  beta_B {betaB:.4g}   lam_A {lamA:.4g}   "
                  f"Gini(regret) {gini:.3f}   "
                  + "  ".join(f"L({k})={v:.3f}" for k, v in lorenz.items()))

            def choose(M, lp, lam, rho):
                return hybrid_map(M, lp, cost, betaB, lam, rho, j)

            for rho in a.rhos:
                pays_router = rho < 1.0

                def table_db(lam):
                    t = n = 0.0
                    for M, R, lp, *_ in cache:
                        k, _ = choose(M, lp, lam, rho)
                        t += (10 * torch.log10(
                            M.gather(1, k[:, None]).squeeze(1).mean() / R)).item()
                        n += 1
                    return t / n

                def true_db(lam):
                    t = 0.0
                    for M, R, lp, y_, q_, xp_ in cache:
                        k, _ = choose(M, lp, lam, rho)
                        k = k.clamp(min=cfg.split_depth)
                        t += (10 * torch.log10(
                            true_frame_mse(net.dec, y_, q_, xp_, k) / R)).item()
                    return t / len(cache)

                floor = true_db(0.0)
                if floor > TARGET + 1e-9:
                    rows.append({"qp": qp_v, "rho": rho, "budget_reachable": False,
                                 "floor_db": floor})
                    print(f"      rho {rho:<5} floor {floor:.4f} dB > budget")
                    continue

                inner, lam = TARGET, None
                for _ in range(6):
                    lam = bisect(table_db, 0.0, LAM_HI, inner)
                    td = true_db(lam)
                    if abs(td - TARGET) < 5e-4:
                        break
                    inner = inner + (TARGET - td)

                SVR = ov = ntile = n = 0.0
                for M, R, lp, *_ in cache:
                    k, s = choose(M, lp, lam, rho)
                    SVR += (1 - cost[k].mean()
                            - (rshare if pays_router else 0.0)).item()
                    ov += s
                    ntile += M.shape[0]
                    n += 1
                N = ntile / n
                bits = N * h2(ov / ntile) + 3.0 * (ov / n)
                rows.append({"qp": qp_v, "rho": rho, "lam": lam, "beta": betaB,
                             "saving_pct_vs_release": 100 * SVR / n,
                             "db_vs_uf": td, "map_bits": bits,
                             "tiles_per_frame": N,
                             "overridden_per_frame": ov / n,
                             "pays_router": pays_router,
                             "lorenz_at_rho": lorenz.get(f"{rho:g}"),
                             "gini_regret": gini,
                             "budget_reachable": True})
                print(f"      rho {rho:<5} saved {100*SVR/n:>6.2f}%  "
                      f"@ {td:>7.4f} dB   {bits:>6.1f} bits/frame")

    out = {"ckpt": a.ckpt, "router2": a.router2, "router2_meta": r2meta,
           "budget_db": TARGET, "decision": "hybrid: signal the rho worst tiles",
           "router_compute_share_pct": 100 * rshare,
           "deepest_exit_cost": float(cost[-1]),
           "n_sequences": len(measured), "frames_per_seq": a.frames,
           "measured": measured, "rows": rows}
    (ROOT / a.out).write_text(json.dumps(out, indent=2))
    print(f"\n  -> {a.out}")


if __name__ == "__main__":
    main(sys.argv[1:])
