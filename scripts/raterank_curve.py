"""A zero-parameter rival to the learned router.

Configuration B spends a 144 K head and 0.163% of the decode to guess which
tiles can exit early. The entropy model has already produced a per-tile number
that is free, exact, and available at the decoder before the trunk runs: how
many bits that tile's latents cost. A tile the encoder had to spend bits on is
a tile whose content the model found hard.

`static_baseline.py` already showed that ranking by bits reproduces 68-79% of
the oracle's assignment -- but it did that by handing the heuristic the
oracle's own multiset of depths, which is encoder-side information. This is the
honest version: nothing but decoder-side data and one global multiplier.

The rule
--------
Model the tile's distortion at exit k as separable,

    D(t,k)  ~  b(t) * phi_k,        b(t) = tile bits / mean tile bits

so phi is a K-number profile that says what each exit costs on an average tile,
and b(t) says how far from average this tile is. Then run the same Lagrangian
the oracle runs, on the surrogate:

    k(t) = argmin_k [ b(t) * phi_k + lam c_k ]

lam is bisected once per qp against the real decode, exactly as beta is for the
router. phi is 6 numbers per qp -- an offline calibration constant, not a
per-frame quantity -- and it is fitted LEAVE-ONE-SEQUENCE-OUT so no sequence
contributes to the profile used to route it.

    python scripts/raterank_curve.py --ckpt runs/RECIPE512/ckpt_eval.pth.tar \
        --budget 0.1 --out results/raterank_RECIPE512_b01.json
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
sys.path.insert(0, str(Path(__file__).resolve().parent))
from gpu import pick as _gpu  # noqa: E402

import ctc_intra as C  # noqa: E402
from flexuf.config import FlexUFConfig  # noqa: E402
from flexuf.cost import exit_costs  # noqa: E402
from flexuf.eval import (reference_frame_mse, tiled_exit_mses,
                         true_frame_mse)  # noqa: E402
from flexuf.measure import measured_saving_pct  # noqa: E402
from flexuf.model import FlexUFIntra, load_flexuf_state  # noqa: E402
from flexuf.reference import reference_for  # noqa: E402


def spearman(a: torch.Tensor, b: torch.Tensor) -> float:
    def rank(v):
        o = v.argsort()
        r = torch.empty_like(o, dtype=torch.float64)
        r[o] = torch.arange(v.numel(), dtype=torch.float64, device=v.device)
        return r
    ra, rb = rank(a.double()), rank(b.double())
    ra = ra - ra.mean(); rb = rb - rb.mean()
    d = (ra.norm() * rb.norm()).clamp_min(1e-12)
    return float((ra * rb).sum() / d)


def main(argv):
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", required=True)
    ap.add_argument("--ref", default=None)
    ap.add_argument("--qps", type=int, nargs="+", default=[0, 16, 32, 48, 63])
    ap.add_argument("--frames", type=int, default=1)
    ap.add_argument("--max_seqs", type=int, default=0)
    ap.add_argument("--device", default=_gpu("cuda:0"))
    ap.add_argument("--budget", type=float, default=0.1)
    ap.add_argument("--out", required=True)
    a = ap.parse_args(argv)

    dev = a.device
    ck = torch.load(ROOT / a.ckpt, map_location="cpu", weights_only=False)
    cfg = FlexUFConfig(**ck["config"])
    net = FlexUFIntra(cfg).to(dev).eval()
    load_flexuf_state(net, ck)
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
          f"{P}px tile, budget {a.budget} dB, ZERO added bits and ZERO "
          f"added parameters\n")

    rows = []
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
                cache.append((tiled_exit_mses(net.dec, y, q, xp, cfg),
                              reference_frame_mse(ref.dec, y, q, xp),
                              y, q, xp, b, name))

            # A rank-1 model in LOG space, fitted leave-one-sequence-out.
            #
            #     log D(t,k)  ~  u(t) + log phi_k,    u(t) = alpha*log b(t) + c
            #
            # phi_k is the exit profile (K constants) and u(t) is the tile's
            # difficulty, predicted from its bit count alone. The first version
            # of this script assumed D ~ b * phi_k -- difficulty proportional to
            # bits -- and it is wrong: alpha comes out NEGATIVE. A tile the
            # entropy coder spent bits on is a tile whose latent already
            # describes it, so a shallow decoder reconstructs it well; a cheap
            # tile is one the decoder has to invent, and inventing takes depth.
            # Fitting the exponent rather than assuming it is what lets the
            # surrogate discover that.
            lb = torch.cat([c[5].log() for c in cache]).double()
            LM = torch.cat([c[0][:, j:].clamp_min(1e-12).log().double()
                            for c in cache])
            sid = torch.cat([torch.full((c[0].shape[0],), i, device=dev)
                             for i, c in enumerate(cache)])
            u_all = LM.mean(1)

            def fit(mask):
                x, y = lb[mask], u_all[mask]
                xm, ym = x.mean(), y.mean()
                al = ((x - xm) * (y - ym)).sum() / ((x - xm) ** 2).sum().clamp_min(1e-30)
                cc = ym - al * xm
                lphi = (LM[mask] - (al * x + cc)[:, None]).mean(0)
                return float(al), float(cc), lphi.float()

            alpha_all, c_all, lphi_all = fit(torch.ones_like(sid, dtype=torch.bool))
            fits = {}
            for i, cc_ in enumerate(cache):
                name = cc_[6]
                if name not in fits:
                    fits[name] = fit(sid != i) if len(cache) > 1 else (
                        alpha_all, c_all, lphi_all)

            def surrogate(b, name):
                al, c0, lphi = fits[name]
                return torch.exp(al * b.log() + c0)[:, None] * torch.exp(lphi)[None, :]

            def maps_at(lam):
                out = []
                for M, _R, _y, _q, _xp, b, name in cache:
                    S = surrogate(b, name)
                    k = (S + lam * cost[None, j:]).argmin(1) + j
                    out.append(k)
                return out

            def db_of(maps):
                return sum((10 * torch.log10(
                    true_frame_mse(net.dec, y_, q_, xp_, m) / R)).item()
                    for m, (_M, R, y_, q_, xp_, _b, _n) in zip(maps, cache)
                    ) / len(cache)

            def sv_of(maps):
                return 100 * sum((1 - cost[m].mean()).item()
                                 for m in maps) / len(maps)

            def sv_measured_of(maps):
                """The same saving, counted off the decode instead of modelled.

                The arithmetic model under-bills the shallow exits by a constant
                0.008 of a released decode (scripts/ceiling_measured.py), which
                is 0.4 to 0.8 saving points. Every table in the paper now
                reports the hook count, so this file has to carry it too or the
                rate-rank table and the main results table are on two different
                definitions of the same quantity.
                """
                tot = 0.0
                for (M, R, y_, q_, xp_, b_, nm_), m in zip(cache, maps):
                    tot += measured_saving_pct(net.dec, ref.dec, y_, q_,
                                               m.clamp(min=j))
                return tot / len(maps)

            # Bisect on the cheap per-tile TABLE, then correct against a real
            # decode of the resulting map -- the same two-level scheme as
            # signalled_curve.py. The first version of this script bisected
            # directly on true_db, which is 26 real decodes of 53 frames per
            # step and made a five-rate run take twenty minutes for what the
            # table settles in seconds.
            def table_db(lam):
                t = n = 0.0
                for (M, R, *_), m in zip(cache, maps_at(lam)):
                    t += (10 * torch.log10(
                        M.gather(1, m[:, None]).squeeze(1).mean() / R)).item()
                    n += 1
                return t / n

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

            floor = table_db(0.0)
            if floor > a.budget:
                rows.append({"qp": qp_v, "budget_reachable": False,
                             "floor_db": db_of(maps_at(0.0))})
                print(f"  qp {qp_v}: floor {floor:.4f} > budget")
                continue
            inner, lo = a.budget, 0.0
            for _ in range(6):
                v = bisect(table_db, 0.0, 1.0, inner)
                if v is None:
                    break
                lo = v
                td = db_of(maps_at(lo))
                if abs(td - a.budget) < 5e-4:
                    break
                inner = max(1e-4, inner + (a.budget - td))
            maps = maps_at(lo)

            # how well does the surrogate order the tiles at all
            def o_maps(lam):
                return [(M + lam * cost[None, :]).argmin(1).clamp(min=j)
                        for M, *_ in cache]

            def o_table_db(lam):
                t = n = 0.0
                for (M, R, *_), m in zip(cache, o_maps(lam)):
                    t += (10 * torch.log10(
                        M.gather(1, m[:, None]).squeeze(1).mean() / R)).item()
                    n += 1
                return t / n

            inner, lamO = a.budget, 0.0
            for _ in range(6):
                v = bisect(o_table_db, 0.0, 1.0, inner)
                if v is None:
                    break
                lamO = v
                td = db_of(o_maps(lamO))
                if abs(td - a.budget) < 5e-4:
                    break
                inner = max(1e-4, inner + (a.budget - td))
            om = [(M + lamO * cost[None, :]).argmin(1).clamp(min=j)
                  for M, *_ in cache]
            agree = sum((m == o).float().mean().item()
                        for m, o in zip(maps, om)) / len(maps)
            rho = sum(spearman(c[5], -o.float())
                      for c, o in zip(cache, om)) / len(cache)
            # What actually decides a tile is not its distortion LEVEL but the
            # SPREAD across the ladder: how much depth buys. A rank-1 model in
            # the level cannot see that, so measure whether bits carry it.
            rho_lvl = sum(spearman(c[5], c[0][:, j:].mean(1))
                          for c in cache) / len(cache)
            rho_spr = sum(spearman(c[5], c[0][:, j] - c[0][:, -1])
                          for c in cache) / len(cache)

            rows.append({"qp": qp_v, "lam": lo,
                         "saving_pct_vs_release": sv_of(maps),
                         "saving_pct_measured": sv_measured_of(maps),
                         "oracle_saving_pct_measured": sv_measured_of(om),
                         "db_vs_uf": db_of(maps),
                         "oracle_saving_pct_vs_release": sv_of(om),
                         "agreement": agree, "spearman_bits_vs_exit": rho,
                         "spearman_bits_vs_level": rho_lvl,
                         "spearman_bits_vs_spread": rho_spr,
                         "alpha": alpha_all, "phi": torch.exp(lphi_all).tolist(),
                         "bpp_added": 0.0, "map_bits": 0,
                         "budget_reachable": True})
            print(f"  qp {qp_v:>3}  rate-rank {sv_of(maps):>6.2f}% @ "
                  f"{db_of(maps):.4f} dB   oracle {sv_of(om):>6.2f}%   "
                  f"agree {agree:.3f}   spearman {rho:+.3f}   "
                  f"alpha {alpha_all:+.2f}   rho(level) {rho_lvl:+.2f}   "
                  f"rho(spread) {rho_spr:+.2f}")

    out = {"ckpt": a.ckpt, "budget_db": a.budget,
           "decision": "decoder-side rate-rank surrogate, no parameters",
           "router_compute_share_pct": 0.0,
           "deepest_exit_cost": float(cost[-1]),
           "n_sequences": len(measured), "frames_per_seq": a.frames,
           "measured": measured, "rows": rows}
    (ROOT / a.out).write_text(json.dumps(out, indent=2))
    print(f"\n  -> {a.out}")


if __name__ == "__main__":
    main(sys.argv[1:])
