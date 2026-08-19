"""The zero-added-bits configuration: the DECODER decides, nothing is signalled.

`signalled_curve.py` measures the shipped system, where the encoder picks the
exit assignment (it has the source frame, so its choice is exact) and sends it as
a ~94-bit map. That map is 0.008-0.020% of the bitrate -- small, but not zero,
so the file is not byte-identical to a stock stream.

This measures the other configuration, the one that adds **no bits at all**: the
decoder runs its own router head on the stem it has already computed, and the
bitstream is untouched. The compute it costs is 0.044% of the decode, already
inside the numbers below.

The gap between the two is the price of not signalling, and it is the number
that matters if the requirement is an unchanged bitstream rather than a small
one.

How a curve is obtained from a fixed router
-------------------------------------------
The router emits per-exit logits and picks `argmax`. That is one operating
point, not a curve, so there is no budget to bisect against. The knob added here
is a scalar tilt toward cheaper exits:

    k = argmax( log softmax(logits) - beta * cost )

`beta = 0` is the router's own trained choice; `beta -> +inf` drives every tile
to the cheapest exit; `beta < 0` buys quality back. Bisection on `beta` then hits
an exact dB budget, exactly as bisection on lambda does for the signalled path.

This is the honest analogue and not a second oracle: `log softmax(logits)` uses
only what the decoder can see, and `cost` is a constant table. Nothing here reads
the source frame -- which is precisely the information the signalled path has and
this one does not.

    python scripts/router_curve.py --ckpt runs/BEST/ckpt_eval.pth.tar \
        --out results/router_BEST.json
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
from flexuf.config import FlexUFConfig  # noqa: E402
from flexuf.cost import exit_costs
from flexuf.eval import (reference_frame_mse, tiled_exit_mses,
                         true_frame_mse)  # noqa: E402
from flexuf.model import FlexUFIntra, load_flexuf_state  # noqa: E402
from flexuf.reference import reference_for  # noqa: E402

TARGET = 0.1        # overridden by --budget

# `exit_costs()` prices the decoder. It does NOT include the router, because in
# the signalled configuration the decoder never runs one. Here it does, so its
# arithmetic is part of what the decode costs and has to come off the saving --
# leaving it out would be the same class of error as the denominator in
# DECISIONS 58: a real cost sitting outside the figure it belongs in.
# Measured per RGB pixel against the decode's 217,113 MAC/px (mac_audit.py).
DECODE_MACPX = 453540e6 / (1920 * 1088)


def router_share(head, args, pixels: int) -> float:
    """Fraction of one decode that this router head costs, measured not assumed."""
    macs = {}

    def hook(m, i, o):
        if isinstance(m, torch.nn.Conv2d):
            macs[id(m)] = (m.in_channels * m.out_channels * m.kernel_size[0]
                           * m.kernel_size[1] * o.shape[-1] * o.shape[-2]
                           / m.groups)
        elif isinstance(m, torch.nn.Linear):
            n = 1
            for d in o.shape[:-1]:
                n *= d
            macs[id(m)] = m.in_features * m.out_features * n

    hs = [m.register_forward_hook(hook) for m in head.modules()]
    with torch.no_grad():
        head(*args)
    for h in hs:
        h.remove()
    return sum(macs.values()) / pixels / DECODE_MACPX


def main(argv):
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", required=True)
    ap.add_argument("--ref", default=None)
    ap.add_argument("--router2", default=None,
                    help="a StemRouterHeadV2 trained against this FROZEN decoder "
                         "(scripts/train_router2.py). Without it the checkpoint's "
                         "own jointly-trained head is used, which in BEST has "
                         "collapsed to a qp-dependent constant.")
    ap.add_argument("--qps", type=int, nargs="+", default=[0, 16, 32, 48, 63])
    ap.add_argument("--frames", type=int, default=1)
    ap.add_argument("--max_seqs", type=int, default=0)
    ap.add_argument("--device", default="cuda:0")
    ap.add_argument("--at_lam", type=float, default=None,
                    help="Skip the budget bisection and report the oracle and "
                         "the router at THIS lambda instead. Both then sit at "
                         "the same operating point, so the difference is pure "
                         "prediction error -- the bisected comparison also "
                         "carries the mismatch between the router's single "
                         "training lambda and the per-rate lambda that lands "
                         "on 0.1 dB.")
    ap.add_argument("--budget", type=float, default=0.1,
                    help="quality budget in dB below the release; see "
                         "signalled_curve.py --budget.")
    ap.add_argument("--per_frame_scale", action="store_true",
                    help="divide each frame's log-probabilities by their own "
                         "mean magnitude before applying the tilt. The decoder "
                         "can do this -- it is a statistic of its own logits, "
                         "not of the source -- and it is what lets ONE global "
                         "beta serve frames the head is confident about and "
                         "frames it is not.")
    ap.add_argument("--out", required=True)
    a = ap.parse_args(argv)

    global TARGET
    TARGET = a.budget
    dev = a.device
    ck = torch.load(ROOT / a.ckpt, map_location="cpu", weights_only=False)
    cfg = FlexUFConfig(**ck["config"])
    net = FlexUFIntra(cfg).to(dev).eval()
    load_flexuf_state(net, ck)
    # A guard on the CONSTRUCTED model's parameters proves nothing -- the head
    # is randomly initialised, so it is nonzero whether or not the checkpoint
    # supplied it. Check the checkpoint's own keys instead.
    sd0 = ck.get("state_dict", ck)
    if not any(k.startswith("router_head.") for k in sd0):
        raise SystemExit(f"{a.ckpt} carries no router head; nothing to evaluate")

    head2, r2meta = None, None
    if a.router2:
        from flexuf.router.head2 import StemRouterHeadV2
        from flexuf.config import LATENT_CH, TRUNK_CH
        h = torch.load(ROOT / a.router2, map_location="cpu", weights_only=False)
        # train_router2.py wraps the weights beside its provenance rather than
        # saving a bare state_dict; take the tensors and keep the provenance,
        # because the lambda a router was trained at decides where on the
        # frontier its ordering is valid.
        hsd = h.get("router_head_v2", h.get("state_dict", h))
        head2 = StemRouterHeadV2(TRUNK_CH, LATENT_CH, cfg.num_exits,
                                 min_exit=cfg.split_depth).to(dev).eval()
        head2.load_state_dict(hsd)
        r2meta = {k: v for k, v in h.items() if k != "router_head_v2"}
        print(f"  router: V2 head from {a.router2}\n"
              f"          trained against a FROZEN decoder at lam="
              f"{r2meta.get('lam')}, held-out agreement "
              f"{r2meta.get('heldout_agree')}")
    else:
        print("  router: the checkpoint's own jointly-trained head")
    ref = FlexUFIntra(cfg).to(dev).eval()
    load_flexuf_state(ref, torch.load(reference_for(cfg, a.ref),
                                      map_location="cpu", weights_only=False))
    sa, sb = net.enc.state_dict(), ref.enc.state_dict()
    worst = max((sa[k] - sb[k]).abs().max().item() for k in sa)
    assert worst == 0.0, f"encoders differ by {worst}"
    cost = exit_costs(cfg, "head").to(dev)
    # The release costs exactly 1.0 in these units; cost[-1] is OUR deepest exit
    # at 1.0095 because it pays seam repair. Dividing by cost[-1] would quote a
    # saving against our own full-depth path while claiming it is against the
    # release -- see DECISIONS 58.
    D = float(cost[-1])

    seqs, missing = C.discover([])
    if a.max_seqs:
        missing = missing + [{"name": s["name"]} for s in seqs[a.max_seqs:]]
        seqs = seqs[:a.max_seqs]
    frames, measured = [], []
    for s in seqs:
        x, pl = C.read_frames(s["path"], s["w"], s["h"], a.frames, 1)
        if x is not None:
            for i in range(x.shape[0]):
                frames.append((x[i:i + 1], pl[i]))
            measured.append(s["name"])
    print(f"  encoder ayni (max|diff| = {worst}), {len(frames)} CTC karesi from "
          f"{len(measured)} sequences, {cfg.rgb_patch}px tile, ZERO added bits\n")

    rows = []
    rshare = 0.0            # set on the first frame, from its real shapes
    if a.at_lam is not None:
        print(f"  at lambda = {a.at_lam:g}, no bisection: oracle vs router at "
              f"the SAME operating point\n")
    else:
        print(f"  {'qp':>4}{'tasarruf':>10}{'vs release':>12}{'dB':>10}{'beta':>10}")
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
                # DEPLOYED path, one tiled decode per exit. The previous form
                # used dec.forward_all_exits, which runs full frame, so the
                # tiling penalty cancelled against the full-frame reference and
                # never appeared in the reported dB (flexuf/eval.py).
                M = tiled_exit_mses(net.dec, y, q, xp, cfg)
                R = reference_frame_mse(ref.dec, y, q, xp)
                # What the decoder can see: the stem it has already computed,
                # and (for V2) the latent and the entropy model's scales, both
                # of which it has decoded before the trunk runs. Still zero
                # added bits -- nothing here comes from the source frame.
                stem = net.dec.upsample(y)
                for g in range(cfg.split_depth):
                    stem = net.dec.groups[g](stem)
                if head2 is not None:
                    lg = head2(stem, y, aux["scales_hat"], qp,
                               cfg.feature_patch, cfg.latent_patch)
                else:
                    lg = net.router_head(stem, qp, cfg.feature_patch)
                # The head masks exits below the split depth by ASSIGNING
                # -1e4, which stops being a mask the moment the head's own
                # logits reach that scale -- and this one's have: its raw
                # outputs sit near -10000, so the "mask" is the LARGEST entry
                # in every row and log_softmax puts almost all the mass on the
                # two exits that do not exist. argmax then picks one and
                # clamp(min=j) turns it into the cheapest real exit, for
                # reasons that have nothing to do with the tile.
                #
                # Fixed at the decision site rather than in head2.py, because
                # the head files are imported by live training runs and a
                # crash-restart would pick up the edit mid-experiment. Every
                # rule below reads lg[:, j:] only.
                lg = lg[:, cfg.split_depth:]
                lp = F.log_softmax(lg, 1)
                if a.per_frame_scale:
                    # The tilt trades log-probability against cost, so its
                    # meaning depends on how large the log-probabilities are --
                    # and a 144 K head is far more confident on some frames
                    # than others. One global beta therefore over-tilts the
                    # confident frames and under-tilts the rest. Normalising by
                    # the frame's own mean magnitude removes that, and costs
                    # nothing: it is a statistic of the decoder's own logits.
                    m = (-lp).mean().clamp_min(1e-6)
                    lp = lp / m
                if rshare == 0.0:
                    px = xp.shape[-1] * xp.shape[-2]
                    rshare = (router_share(head2, (stem, y, aux["scales_hat"],
                                                   qp, cfg.feature_patch,
                                                   cfg.latent_patch), px)
                              if head2 is not None else
                              router_share(net.router_head,
                                           (stem, qp, cfg.feature_patch), px))
                    print(f"  router costs {100*rshare:.4f}% of the decode; "
                          f"charged against every saving below\n")
                cache.append((M, R, lp, y, q, xp))

            def at_beta(beta):
                SV = SVR = DB = n = 0.0
                for M, R, lp, _y, _q, _xp in cache:
                    k = (lp - beta * cost[None, cfg.split_depth:]).argmax(1) \
                        + cfg.split_depth
                    # rshare is added to the cost, i.e. subtracted from the
                    # saving: the decoder pays for the router here.
                    SV += (1 - (cost[k].mean() + rshare) / cost[-1]).item()
                    SVR += (1 - cost[k].mean() - rshare).item()
                    DB += (10 * torch.log10(
                        M.gather(1, k[:, None]).squeeze(1).mean() / R)).item()
                    n += 1
                return 100 * SV / n, DB / n, 100 * SVR / n

            def true_db(beta):
                """What a decode of the router's actual map delivers.

                The table measures each tile with its neighbours at the same
                exit; a routed frame is mixed. One real decode per frame settles
                it, which is affordable outside the bisection but not inside.
                """
                tot = 0.0
                for M, R, lp, y_, q_, xp_ in cache:
                    k = (lp - beta * cost[None, cfg.split_depth:]).argmax(1) \
                        + cfg.split_depth.clamp(
                        min=cfg.split_depth)
                    tot += (10 * torch.log10(
                        true_frame_mse(net.dec, y_, q_, xp_, k) / R)).item()
                return tot / len(cache)

            if a.at_lam is not None:
                # Same lambda for both, so nothing here depends on the tilt or
                # on which budget was chosen. Whatever separates them is the
                # router failing to predict what the search would have picked.
                lam = a.at_lam
                SO = SR = DO = DR = AG = RG = n = 0.0
                for M, R, lp, y_, q_, xp_ in cache:
                    ko = (M + lam * cost[None, :]).argmin(1)
                    kr = lp.argmax(1) + cfg.split_depth
                    SO += (1 - cost[ko].mean()).item()
                    SR += (1 - cost[kr].mean() - rshare).item()
                    DO += (10 * torch.log10(
                        M.gather(1, ko[:, None]).squeeze(1).mean() / R)).item()
                    DR += (10 * torch.log10(
                        M.gather(1, kr[:, None]).squeeze(1).mean() / R)).item()
                    AG += (ko == kr).float().mean().item()
                    L = M + lam * cost[None, :]
                    RG += (L.gather(1, kr[:, None]) - L.gather(1, ko[:, None])
                           ).mean().item()
                    n += 1
                rows.append({"qp": qp_v, "lam": lam,
                             "oracle_saving_pct_vs_release": 100 * SO / n,
                             "router_saving_pct_vs_release": 100 * SR / n,
                             "oracle_db": DO / n, "router_db": DR / n,
                             "agreement": AG / n, "regret": RG / n})
                print(f"  {qp_v:>4}  oracle {100*SO/n:>6.2f}% @ {DO/n:>7.4f} dB"
                      f"   router {100*SR/n:>6.2f}% @ {DR/n:>7.4f} dB"
                      f"   agree {AG/n:.3f}")
                continue

            # Bisect beta for an exact budget. Large beta = cheap exits = worse
            # dB, so dB is increasing in beta and the invariant is the same as
            # the lambda bisection's.
            # -50 did not reach the all-deepest allocation: a confident head
            # has log-probability gaps of hundreds, and the masked exits sit at
            # -1e4, so the tilt has to be able to outweigh those.
            LO, HI = -2.0e4, 2.0e4
            if true_db(LO) > TARGET:         # even the deepest choice overshoots
                sv, db, svr = at_beta(LO)
                rows.append({"qp": qp_v, "saving_pct": None,
                             "floor_db": true_db(LO), "budget_reachable": False})
                print(f"  {qp_v:>4}{'—':>10}{'—':>12}{true_db(LO):>10.4f}"
                      f"{'floor > budget':>16}")
                continue

            def bisect_to(t):
                lo, hi = LO, HI
                if at_beta(hi)[1] <= t:
                    return hi
                for _ in range(60):
                    mid = 0.5 * (lo + hi)
                    if at_beta(mid)[1] <= t:
                        lo = mid
                    else:
                        hi = mid
                return lo

            # Bisect on the cheap table, then correct against a real decode of
            # the router's actual map. Same two-level scheme as
            # signalled_curve.py, and for the same reason.
            inner, beta, td = TARGET, None, None
            for _ in range(6):
                beta = bisect_to(inner)
                td = true_db(beta)
                if abs(td - TARGET) < 5e-4:
                    break
                inner = inner + (TARGET - td)
            sv, db_table, svr = at_beta(beta)
            rows.append({"qp": qp_v, "saving_pct": sv,
                         "saving_pct_vs_release": svr, "db_vs_uf": td,
                         "db_vs_uf_table": db_table,
                         "beta": beta, "bpp_added": 0.0, "map_bits": 0,
                         "budget_reachable": True})
            print(f"  {qp_v:>4}{sv:>9.2f}%{svr:>11.2f}%{td:>10.4f}{beta:>10.3f}")

    (ROOT / a.out).write_text(json.dumps(
        {"ckpt": a.ckpt, "ckpt_epoch": ck.get("epoch"), "ckpt_step": ck.get("step"),
         "decision": "decoder-side router head, nothing signalled",
         "budget_db": TARGET,
         "router2": a.router2, "router2_meta": r2meta,
         "bits_added_per_frame": 0,
         "router_compute_share_pct": 100 * rshare,
         "deepest_exit_cost": D,
         "frames_per_seq": a.frames, "n_sequences": len(measured),
         "measured": measured, "not_measured": [m["name"] for m in missing],
         "rows": rows}, indent=2))
    print(f"\n  wrote {a.out}")
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
