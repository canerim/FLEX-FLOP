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

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(Path.home() / "DCVC"))

import ctc_intra as C  # noqa: E402
# The bisection and the cache it runs on live in flexuf/beta.py, because
# scripts/beta_calibration.py has to run the SAME search on held-out images and
# then apply its answer here. Two copies of a bisection is two bisections.
# router_share is re-exported: scripts/combined_curve.py and
# scripts/hybrid_curve.py import it from this module.
from flexuf.beta import (at_beta, bisect_beta, build_cache,  # noqa: E402,F401
                         floor_db, measured_saving, router_share)
from flexuf.config import FlexUFConfig  # noqa: E402
from flexuf.cost import exit_costs
from flexuf.model import FlexUFIntra, load_flexuf_state  # noqa: E402
from flexuf.reference import reference_for  # noqa: E402

TARGET = 0.1        # overridden by --budget


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
            def _say(r):
                print(f"  router costs {100*r:.4f}% of the decode; "
                      f"charged against every saving below\n")

            cache, rshare = build_cache(
                net, ref, head2, cfg, [x for x, _pl in frames], qp_v, dev,
                per_frame_scale=a.per_frame_scale, rshare=rshare,
                on_rshare=_say)

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

            # Even the deepest allocation the tilt can reach may overshoot the
            # budget; then there is no beta and the rate is reported as
            # unreachable rather than as a saving.
            fl = floor_db(net.dec, cache, cost, cfg.split_depth, dev)
            if fl > TARGET:
                rows.append({"qp": qp_v, "saving_pct": None,
                             "floor_db": fl, "budget_reachable": False})
                print(f"  {qp_v:>4}{'—':>10}{'—':>12}{fl:>10.4f}"
                      f"{'floor > budget':>16}")
                continue

            beta, td = bisect_beta(net.dec, cache, cost, cfg.split_depth,
                                   rshare, TARGET, dev)
            sv, db_table, svr = at_beta(cache, cost, cfg.split_depth, rshare,
                                        beta)
            # The same saving counted off the decode rather than modelled. The
            # arithmetic model under-bills the shallow exits by a constant
            # 0.008 of a released decode (scripts/ceiling_measured.py), and
            # every table in the paper now reports the hook count, so this file
            # has to carry it or configuration B is on a different definition
            # from configuration A in the table that compares them. The
            # router's own share is charged against it here exactly as it is
            # against the modelled figure.
            svr_meas = measured_saving(net.dec, ref.dec, cache, cost,
                                       cfg.split_depth, rshare, beta, dev)
            rows.append({"qp": qp_v, "saving_pct": sv,
                         "saving_pct_vs_release": svr,
                         "saving_pct_measured": svr_meas, "db_vs_uf": td,
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
