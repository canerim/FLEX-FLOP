"""How much does the achievable saving depend on the content?

The headline is one number per rate, averaged over 40 CTC sequences. An average
answers "what does this buy on the test set" and hides the question a deployment
asks: is the saving uniform, or does the mean come from a handful of easy clips
while the hard ones get almost nothing?

The allocation is global -- one lambda prices compute for every tile in the set
-- so at a fixed operating point each sequence's own saving and dB can be read
out directly, and their spread is the answer. That is what `paper_curve.py`
records under `op_points[*]["per_sequence"]`.

Two things this deliberately does NOT do:

  * It does not re-optimise lambda per sequence. That would measure a different
    system -- one that tunes itself per clip -- and would flatter the spread by
    construction, since each sequence would sit at its own best point.
  * It does not drop outliers. A sequence where early exit buys nothing is the
    most informative point on the plot.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parent.parent


def main(argv):
    ap = argparse.ArgumentParser()
    ap.add_argument("--curve", default="results/paper_curve_grid128.json")
    ap.add_argument("--target", type=float, default=0.10,
                    help="which operating point to break down, in dB")
    ap.add_argument("--out", default="results/per_sequence.json")
    a = ap.parse_args(argv)

    d = json.loads((ROOT / a.curve).read_text())
    ops = d.get("op_points")
    if not ops:
        raise SystemExit(f"{a.curve} has no op_points -- regenerate it with the "
                         f"current paper_curve.py")

    # Both denominators, because the two differ by more than the rounding a
    # reader would assume. `saving_pct` counts against OUR deepest exit, which
    # costs 1.0095 released decodes -- it pays for seam repair and the release
    # does not -- and `saving_pct_vs_release` counts against the released
    # decoder itself, which is what every headline in this work quotes. A
    # spread table in one denominator sitting beside a headline in the other is
    # how a reader concludes the two disagree.
    out = {"curve": a.curve, "target_db": a.target,
           "n_sequences": d.get("n_sequences"), "ckpt": d.get("ckpt"),
           "denominator_rows": "our own deepest exit (cost[-1])",
           "denominator_rows_vs_release": "the released decoder (1.0)",
           "rows": [], "rows_vs_release": []}
    print(f"  saving per sequence at a {a.target:.2f} dB budget, "
          f"{d.get('n_sequences')} CTC sequences\n")
    print(f"  {'qp':>4}{'mean':>9}{'sd':>8}{'min':>8}{'p25':>8}{'median':>9}"
          f"{'p75':>8}{'max':>8}   worst sequence")

    for op in ops:
        if abs(op["target_db"] - a.target) > 1e-9 or not op.get("per_sequence"):
            continue
        ps = op["per_sequence"]
        for key, dest in (("saving_pct", out["rows"]),
                          ("saving_pct_vs_release", out["rows_vs_release"])):
            if key not in ps[0]:
                continue
            v = np.array([r[key] for r in ps])
            qq = np.percentile(v, [25, 50, 75])
            dest.append({"qp": op["qp"], "mean": float(v.mean()),
                         "sd": float(v.std()), "min": float(v.min()),
                         "p25": float(qq[0]), "median": float(qq[1]),
                         "p75": float(qq[2]), "max": float(v.max()),
                         "worst_seq": ps[int(v.argmin())]["seq"],
                         "best_seq": ps[int(v.argmax())]["seq"],
                         "n": len(ps)})
        sv = np.array([r["saving_pct"] for r in ps])
        worst = min(ps, key=lambda r: r["saving_pct"])
        best = max(ps, key=lambda r: r["saving_pct"])
        q = np.percentile(sv, [25, 50, 75])
        # The per-sequence list travels with the first denominator's row so
        # the file carries every sequence's own numbers, not just the summary.
        out["rows"][-1]["per_sequence"] = ps
        print(f"  {op['qp']:>4}{sv.mean():>8.1f}%{sv.std():>8.1f}{sv.min():>8.1f}"
              f"{q[0]:>8.1f}{q[1]:>9.1f}{q[2]:>8.1f}{sv.max():>8.1f}   "
              f"{Path(worst['seq']).stem[:24]}")

    if not out["rows"]:
        raise SystemExit(f"no operating point at {a.target} dB in {a.curve}")

    # The claim this measurement exists to test.
    r = out["rows"]
    ratio = [x["max"] / x["min"] if x["min"] > 0 else float("inf") for x in r]
    print(f"\n  best-to-worst sequence ratio: "
          + ", ".join(f"qp{x['qp']} {v:.1f}x" for x, v in zip(r, ratio)))
    print("  The mean is not what any single clip gets. Quoting only the mean")
    print("  would promise a deployment something the hardest content does not")
    print("  deliver.")

    (ROOT / a.out).write_text(json.dumps(out, indent=2))
    print(f"\n  wrote {a.out}")
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
