"""Are the two decoder-side signals complementary, or the same information?

Configuration B routes on a 144 K head. Section 5.6's baseline routes on the
bits the entropy model already spent, with no parameters at all, and beats the
head above q32. The head reads the entropy model's scales, so it has access to
something close to the bit count -- which raises the question of whether it is
failing to use it, or whether the two signals say the same thing and the head is
simply mis-calibrated away from its training multiplier.

One knob settles it. Put the head's log-probabilities into the rank-1 surrogate
as a multiplicative correction in the log-distortion domain,

    log D(t,k)  =  alpha log b(t) + c + log phi_k  -  gamma log p_k(t)

and run one Lagrangian on it. gamma = 0 is the parameter-free rule exactly.
gamma -> large is the head's own ordering. If the best gamma is interior and
beats both ends, the signals are complementary and a head that consumed the bit
count directly would be worth training. If it is at an end, one of them is
redundant.

gamma is one scalar per rate, chosen on the test set: read the sweep, not the
maximum, and the maximum as an upper bound.

    python scripts/combined_curve.py --ckpt runs/RECIPE512/ckpt_eval.pth.tar \
        --router2 runs/RECIPE512/routers2/v2_lam1.3e-5.pth \
        --budget 0.1 --out results/combined_RECIPE512_b01.json
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import torch
import torch.nn.functional as F

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(Path.home() / "DCVC"))

import ctc_intra as C  # noqa: E402
from flexuf.config import FlexUFConfig, LATENT_CH, TRUNK_CH  # noqa: E402
from flexuf.cost import exit_costs  # noqa: E402
from flexuf.eval import (reference_frame_mse, tiled_exit_mses,
                         true_frame_mse)  # noqa: E402
from flexuf.model import FlexUFIntra, load_flexuf_state  # noqa: E402
from flexuf.reference import reference_for  # noqa: E402
from router_curve import router_share  # noqa: E402


def main(argv):
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", required=True)
    ap.add_argument("--ref", default=None)
    ap.add_argument("--router2", required=True)
    ap.add_argument("--qps", type=int, nargs="+", default=[0, 16, 32, 48, 63])
    ap.add_argument("--gammas", type=float, nargs="+",
                    default=[0.0, 0.1, 0.25, 0.5, 0.75, 0.9, 1.0],
                    help="blend weight w: 0 = bits only, 1 = the head only")
    ap.add_argument("--frames", type=int, default=1)
    ap.add_argument("--max_seqs", type=int, default=0)
    ap.add_argument("--device", default="cuda:0")
    ap.add_argument("--budget", type=float, default=0.1)
    ap.add_argument("--out", required=True)
    a = ap.parse_args(argv)

    dev = a.device
    ck = torch.load(ROOT / a.ckpt, map_location="cpu", weights_only=False)
    cfg = FlexUFConfig(**ck["config"])
    net = FlexUFIntra(cfg).to(dev).eval()
    load_flexuf_state(net, ck)
    from flexuf.router.head2 import StemRouterHeadV2
    h = torch.load(ROOT / a.router2, map_location="cpu", weights_only=False)
    head = StemRouterHeadV2(TRUNK_CH, LATENT_CH, cfg.num_exits,
                            min_exit=cfg.split_depth).to(dev).eval()
    head.load_state_dict(h.get("router_head_v2", h.get("state_dict", h)))
    ref = FlexUFIntra(cfg).to(dev).eval()
    load_flexuf_state(ref, torch.load(reference_for(cfg, a.ref),
                                      map_location="cpu", weights_only=False))
    cost = exit_costs(cfg, "head").to(dev)
    j, K, P = cfg.split_depth, cfg.num_exits, cfg.rgb_patch

    seqs, _ = C.discover([])
    if a.max_seqs:
        seqs = seqs[:a.max_seqs]
    frames, measured = [], []
    for s in seqs:
        x, pl = C.read_frames(s["path"], s["w"], s["h"], a.frames, 1)
        if x is not None:
            for i in range(x.shape[0]):
                frames.append((x[i:i + 1], s["name"]))
            measured.append(s["name"])
    print(f"  {len(frames)} CTC frames from {len(measured)} sequences, "
          f"budget {a.budget} dB\n")

    rows, rshare = [], 0.0
    with torch.no_grad():
        for qp_v in a.qps:
            cache = []
            for x, name in frames:
                x = x.to(dev); _, _, H, W = x.shape
                ph, pw = (-H) % P, (-W) % P
                xp = F.pad(x, (0, pw, 0, ph), mode="replicate") if (ph or pw) else x
                qp = torch.full((1,), qp_v, dtype=torch.int32, device=dev)
                y, q, aux = net._encode_to_latent(xp, qp)
                bits = net.get_y_bits(net.add_noise(aux["y_res"]),
                                      aux["scales_hat"]).sum(1, keepdim=True)
                L = P // 16
                nh_, nw_ = bits.shape[-2] // L, bits.shape[-1] // L
                tb = (bits.view(1, 1, nh_, L, nw_, L).permute(0, 1, 2, 4, 3, 5)
                          .reshape(nh_ * nw_, L * L).sum(1))
                b = tb / tb.mean().clamp_min(1e-12)
                stem = net.dec.upsample(y)
                for g in range(j):
                    stem = net.dec.groups[g](stem)
                lp = F.log_softmax(head(stem, y, aux["scales_hat"], qp,
                                        cfg.feature_patch, cfg.latent_patch), 1)
                if rshare == 0.0:
                    rshare = router_share(head, (stem, y, aux["scales_hat"], qp,
                                                 cfg.feature_patch,
                                                 cfg.latent_patch),
                                          xp.shape[-1] * xp.shape[-2])
                    print(f"  router costs {100*rshare:.4f}% of the decode\n")
                cache.append((tiled_exit_mses(net.dec, y, q, xp, cfg),
                              reference_frame_mse(ref.dec, y, q, xp),
                              y, q, xp, b, lp, name))

            # rank-1 fit in log space, leave-one-sequence-out (as 5.6)
            lb = torch.cat([c[5].log() for c in cache]).double()
            LM = torch.cat([c[0][:, j:].clamp_min(1e-12).log().double()
                            for c in cache])
            sid = torch.cat([torch.full((c[0].shape[0],), i, device=dev)
                             for i, c in enumerate(cache)])
            u = LM.mean(1)

            def fit(mask):
                x_, y_ = lb[mask], u[mask]
                xm, ym = x_.mean(), y_.mean()
                al = (((x_ - xm) * (y_ - ym)).sum()
                      / ((x_ - xm) ** 2).sum().clamp_min(1e-30))
                cc = ym - al * xm
                return float(al), float(cc), \
                    (LM[mask] - (al * x_ + cc)[:, None]).mean(0).float()

            fits = {}
            for i, c in enumerate(cache):
                if c[7] not in fits:
                    fits[c[7]] = fit(sid != i)

            # Both signals normalised to unit mean per frame, then blended.
            # An earlier form put the router's log-probabilities inside the
            # rank-1 exponent as -gamma*log p_k. That breaks the floor: at
            # lam=0 the pure bit surrogate picks the deepest exit, because phi
            # is decreasing in k, but with the correction it picks whatever the
            # head likes, which is shallow -- so every gamma > 0 came back
            # "unreachable" rather than measuring anything.
            def blend(b_, lp_, name, w):
                al, c0, lphi = fits[name]
                A = torch.exp((al * b_.log() + c0)[:, None] + lphi[None, :])
                A = A / A.mean().clamp_min(1e-30)
                B = -lp_[:, j:]
                B = B / B.mean().clamp_min(1e-30)
                return (1 - w) * A + w * B

            LAM = 50.0
            for gamma in a.gammas:
                def maps_at(lam):
                    return [(blend(c[5], c[6], c[7], gamma)
                             + lam * cost[None, j:]).argmin(1) + j
                            for c in cache]

                def table_db(lam):
                    t = 0.0
                    for c, m in zip(cache, maps_at(lam)):
                        t += (10 * torch.log10(
                            c[0].gather(1, m[:, None]).squeeze(1).mean()
                            / c[1])).item()
                    return t / len(cache)

                def true_db(lam):
                    t = 0.0
                    for c, m in zip(cache, maps_at(lam)):
                        t += (10 * torch.log10(
                            true_frame_mse(net.dec, c[2], c[3], c[4], m)
                            / c[1])).item()
                    return t / len(cache)

                def bisect(f, lo, hi, tgt):
                    if f(lo) > tgt:
                        return None
                    if f(hi) <= tgt:
                        return hi
                    for _ in range(40):
                        mid = 0.5 * (lo + hi)
                        if f(mid) <= tgt:
                            lo = mid
                        else:
                            hi = mid
                    return lo

                inner, lam, td = a.budget, 0.0, None
                for _ in range(6):
                    v = bisect(table_db, -LAM, LAM, inner)
                    if v is None:
                        break
                    lam = v
                    td = true_db(lam)
                    if abs(td - a.budget) < 5e-4:
                        break
                    inner = max(1e-4, inner + (a.budget - td))
                if td is None:
                    rows.append({"qp": qp_v, "gamma": gamma,
                                 "budget_reachable": False})
                    print(f"      w {gamma:<5} unreachable")
                    continue
                maps = maps_at(lam)
                # gamma > 0 uses the head, so the head's compute is charged.
                pay = rshare if gamma > 0 else 0.0   # gamma = w, the blend
                sv = 100 * sum((1 - cost[m].mean() - pay).item()
                               for m in maps) / len(maps)
                rows.append({"qp": qp_v, "gamma": gamma, "lam": lam,
                             "saving_pct_vs_release": sv, "db_vs_uf": td,
                             "pays_router": gamma > 0, "map_bits": 0,
                             "bpp_added": 0.0, "budget_reachable": True})
                print(f"  qp {qp_v:>3}  w {gamma:<5} saved {sv:>6.2f}% @ "
                      f"{td:.4f} dB")

    (ROOT / a.out).write_text(json.dumps(
        {"ckpt": a.ckpt, "router2": a.router2, "budget_db": a.budget,
         "decision": "rank-1 bit surrogate corrected by the router's log-probs",
         "router_compute_share_pct": 100 * rshare,
         "deepest_exit_cost": float(cost[-1]),
         "n_sequences": len(measured), "frames_per_seq": a.frames,
         "measured": measured, "rows": rows}, indent=2))
    print(f"\n  -> {a.out}")


if __name__ == "__main__":
    main(sys.argv[1:])
