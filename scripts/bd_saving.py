"""A single number for the compute-quality trade-off, in the Bjontegaard style.

Why a single number
-------------------
The frontier is currently quoted at two points, 0.1 dB and 0.3 dB. Both are
arbitrary, and a claim resting on one sample of a curve is fragile in a way this
project has already been bitten by: reading the saving off the nearest sweep
sample rather than the curve made one test-set comparison look twice as large as
it was. Video coding solved the same problem for rate-quality with BD-rate --
integrate one axis over a stated interval of the other -- and the same
construction applies verbatim here with compute in place of rate.

Two numbers, because the trade-off has two directions:

    BD-saving   mean compute saved over a dB interval      "what does 0.3 dB buy"
    BD-quality  mean dB paid over a saving interval        "what does 25% cost"

Both are integrals of the MEASURED frontier, so neither depends on where the
lambda sweep happened to place a sample, and both are reported with the interval
they were taken over -- the number is meaningless without it.

Integration follows BD's own convention: the curve is integrated by the
trapezoid rule on a dense resampling of the piecewise-linear frontier through
the measured points. BD-rate fits a cubic in log-rate instead; that is a
smoothing choice made because rate curves are sampled at four points, and here
the frontier has 19, so interpolating between measurements adds nothing that
fitting would and cannot overshoot the way a fit can -- which is exactly the
failure that made the convexity test report violations that were not there.

The intervals are chosen once and stated, not tuned:

    dB in [floor, 0.3]  0.3 dB is where the project's 30-40% target is met.
                        The lower limit is the largest per-rate floor, because
                        the frontier cannot reach 0 dB: with every tile at the
                        deepest exit a small residual against the released
                        decoder remains, and it differs by rate. Integrating
                        each rate from its own floor would average five
                        different intervals and report them as one number.
    saving in [10, 30]% the band the deployed system actually operates in
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np

trapezoid = getattr(np, "trapezoid", np.trapz)

ROOT = Path(__file__).resolve().parent.parent


def frontier(rows, qp, key="db_vs_uf_per_frame"):
    """The measured (saving, dB) points at one QP, deduplicated and sorted.

    Kept as the upper-left staircase: for each saving the SMALLEST dB attained.
    A lambda sweep can land two allocations on the same saving with different
    distortion, and only the better one is on the frontier.
    """
    best: dict[float, float] = {}
    for r in rows:
        if r["qp"] != qp:
            continue
        s = round(r["saving_pct"], 6)
        v = r.get(key, r["db_vs_uf"])
        if s not in best or v < best[s]:
            best[s] = v
    pts = sorted(best.items())
    return np.array([p[0] for p in pts]), np.array([p[1] for p in pts])


def bd_saving(S, dB, lo, hi, n=2001):
    """Mean saving over dB in [lo, hi]. None if the frontier does not span it."""
    if dB.min() > lo or dB.max() < hi:
        return None
    xs = np.linspace(lo, hi, n)
    # dB is increasing in saving along the frontier, so it inverts directly.
    return float(trapezoid(np.interp(xs, dB, S), xs) / (hi - lo))


def bd_quality(S, dB, lo, hi, n=2001):
    """Mean dB over saving in [lo, hi]. None if the frontier does not span it."""
    if S.min() > lo or S.max() < hi:
        return None
    xs = np.linspace(lo, hi, n)
    return float(trapezoid(np.interp(xs, S, dB), xs) / (hi - lo))


def main(argv):
    ap = argparse.ArgumentParser()
    ap.add_argument("--curve", default="results/paper_curve_grid128.json")
    ap.add_argument("--convention", choices=["per_frame", "pooled"],
                    default="per_frame",
                    help="which decibel to integrate against. 'per_frame' is "
                         "what ~/DCVC/test_video.py computes and so what every "
                         "published DCVC-UF number means; 'pooled' puts every "
                         "tile into one MSE, the natural form for the "
                         "Lagrangian the theory is about. On an identical "
                         "allocation they differ by a quarter to a third of a "
                         "0.1 dB budget, pooled always the flattering one, so "
                         "the default is the one a reader will compare against.")
    ap.add_argument("--db_lo", type=float, default=None,
                    help="lower dB limit. Default: the largest floor across the "
                         "compared rates. The frontier cannot reach 0 dB -- with "
                         "every tile at the deepest exit there is still a small "
                         "residual against the released decoder -- and each rate "
                         "has a different floor, so integrating each from its own "
                         "would compare five different intervals and call the "
                         "result one number.")
    ap.add_argument("--db_hi", type=float, default=0.30)
    ap.add_argument("--sv_lo", type=float, default=10.0)
    ap.add_argument("--sv_hi", type=float, default=30.0)
    ap.add_argument("--out", default="results/bd_saving.json")
    a = ap.parse_args(argv)

    d = json.loads((ROOT / a.curve).read_text())
    rows = d["rows"]
    qps = sorted({r["qp"] for r in rows})

    key = ("db_vs_uf_per_frame" if a.convention == "per_frame" else "db_vs_uf")
    if a.convention == "per_frame" and "db_vs_uf_per_frame" not in rows[0]:
        raise SystemExit(
            f"{a.curve} predates the per-frame convention; regenerate it with "
            f"the current paper_curve.py, or pass --convention pooled and say "
            f"so wherever the number is quoted")
    floors = [frontier(rows, q, key)[1].min() for q in qps]
    if a.db_lo is None:
        # Round up so the printed interval is a clean number that every
        # frontier genuinely spans.
        a.db_lo = float(np.ceil(max(floors) * 1000) / 1000)
    print(f"  per-rate floors: "
          + ", ".join(f"qp{q} {f:.4f}" for q, f in zip(qps, floors)))

    print(f"  frontier: {a.curve}")
    print(f"  test set: {d.get('n_sequences', '?')} CTC sequences, "
          f"checkpoint {d.get('ckpt', '?')}")
    print(f"  dB convention: {a.convention}\n")
    # Three decimals, not two: the auto-derived lower limit is the largest
    # per-rate floor rounded up to the nearest 0.001 (0.057 here), and printing
    # it as 0.06 reported an interval that was not the one integrated over.
    print(f"  BD-saving  = mean compute saved over dB in "
          f"[{a.db_lo:.3f}, {a.db_hi:.3f}]")
    print(f"  BD-quality = mean dB paid over saving in "
          f"[{a.sv_lo:.0f}%, {a.sv_hi:.0f}%]\n")
    print(f"  {'qp':>4}{'BD-saving':>12}{'BD-quality':>13}{'floor dB':>11}"
          f"{'no-drift gain':>15}")

    out = {"curve": a.curve, "convention": a.convention,
           "n_sequences": d.get("n_sequences"), "ckpt": d.get("ckpt"),
           # Carried through so any table built from bd_*.json can show WHICH
           # snapshot each row is. ckpt_step.pth.tar is overwritten as training
           # proceeds, so two rows naming the same path were compared at 8,000
           # and 12,000 steps once, and nothing in the numbers said so.
           "ckpt_epoch": d.get("ckpt_epoch"), "ckpt_step": d.get("ckpt_step"),
           "db_interval": [a.db_lo, a.db_hi],
           "saving_interval": [a.sv_lo, a.sv_hi], "rows": []}
    for qp, floor in zip(qps, floors):
        S, dB = frontier(rows, qp, key)
        bs = bd_saving(S, dB, a.db_lo, a.db_hi)
        bq = bd_quality(S, dB, a.sv_lo, a.sv_hi)

        # What the anchor is worth.
        #
        # The frontier's floor is the deepest exit's residual against the
        # released decoder -- drift, not routing. Every operating point pays it
        # before buying any saving, so it eats the dB budget from below exactly
        # as the ceiling limits it from above. Shifting the curve down by the
        # floor estimates the frontier of a decoder whose deepest exit were
        # exactly the released one, which is what --anchor_weight is for.
        #
        # An estimate, not a measurement: it assumes removing the drift moves
        # the whole curve rigidly rather than changing its shape. It bounds the
        # prize, and says whether tightening the anchor is worth spending on.
        bs0 = bd_saving(S, dB - floor, a.db_lo, a.db_hi)
        out["rows"].append({"qp": qp, "bd_saving_pct": bs, "bd_quality_db": bq,
                            "floor_db": float(floor),
                            "bd_saving_pct_zero_drift": bs0,
                            "n_points": int(len(S))})
        # None when shifting the curve down by the floor drops its top below
        # the interval's upper limit -- refusing to extrapolate rather than
        # inventing the missing stretch.
        gain = f"{'n/a':>15}" if (bs is None or bs0 is None) \
            else f"{bs0 - bs:>+14.2f}"
        print(f"  {qp:>4}{'  n/a' if bs is None else f'{bs:>11.2f}%'}"
              f"{'  n/a' if bq is None else f'{bq:>12.4f}'}"
              f"{floor:>11.4f}{gain}")

    ok = [r for r in out["rows"] if r["bd_saving_pct"] is not None]
    if ok:
        m = float(np.mean([r["bd_saving_pct"] for r in ok]))
        mq = float(np.mean([r["bd_quality_db"] for r in ok
                            if r["bd_quality_db"] is not None]))
        out["mean_bd_saving_pct"] = m
        out["mean_bd_quality_db"] = mq
        print(f"\n  mean over {len(ok)} rates: BD-saving {m:.2f}%, "
              f"BD-quality {mq:.4f} dB")
        # Averaged over the SAME rates on both sides. Comparing a 5-rate mean
        # against a 4-rate one would attribute the missing rate to the anchor.
        both = [r for r in ok if r["bd_saving_pct_zero_drift"] is not None]
        if both:
            g = float(np.mean([r["bd_saving_pct_zero_drift"] - r["bd_saving_pct"]
                               for r in both]))
            out["no_drift_gain_pts"] = g
            out["no_drift_gain_rates"] = [r["qp"] for r in both]
            print(f"  removing the deepest exit's drift would be worth {g:+.2f} "
                  f"points of BD-saving, averaged over qp"
                  + "/".join(str(r["qp"]) for r in both)
                  + f" ({len(ok) - len(both)} rate(s) not measurable: shifting "
                    f"the curve down puts its top below {a.db_hi:.3f} dB)")

    (ROOT / a.out).write_text(json.dumps(out, indent=2))
    print(f"\n  wrote {a.out}")
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
