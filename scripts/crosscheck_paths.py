"""Do the two measurement paths agree? They are supposed to, and did not.

`paper_curve.py` and `signalled_curve.py` reach the same quantity by different
routes. paper_curve pools every tile of every frame into one table and allocates
globally; signalled_curve keeps frames separate, allocates within each, and adds
the exit map's entropy-coded cost to the bitrate. Written separately, they share
no code beyond the model and the cost table.

They disagreed by 5.4 points until three causes were separated: signalled_curve
read the best sample of a lambda grid instead of bisecting, the two used
different frame counts, and -- the real one -- they meant different things by
"dB below the released decoder", one pooling all tiles into a single MSE and the
other averaging a per-frame decibel. That last is worth a quarter to a third of
a 0.1 dB budget.

With all three settled they agree to 0.16 points, and the residual is
interpolation: paper_curve's sweep is read between samples while signalled_curve
bisects to the budget exactly.

This exists so the agreement is re-checked rather than remembered. Two
implementations landing on the same number is the only evidence available that
neither has a bug the other lacks.

    python scripts/crosscheck_paths.py        # non-zero if they drift apart
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
    ap.add_argument("--signalled", default="results/signalled_grid128.json")
    ap.add_argument("--budget", type=float, default=0.10)
    ap.add_argument("--tol", type=float, default=0.5,
                    help="points of saving. Loose enough to absorb reading a "
                         "sweep between samples, tight enough that any of the "
                         "three causes already found would breach it -- the "
                         "smallest of them was 0.6 points.")
    a = ap.parse_args(argv)

    pc = json.loads((ROOT / a.curve).read_text())
    sg = json.loads((ROOT / a.signalled).read_text())

    # Comparing two files says nothing unless they describe the same experiment.
    problems = []
    for field in ("ckpt", "n_sequences", "frames_per_seq"):
        if pc.get(field) != sg.get(field):
            problems.append(f"{field}: curve {pc.get(field)!r} vs signalled "
                            f"{sg.get(field)!r}")
    if problems:
        print("  the two files are not describing the same measurement:")
        for p in problems:
            print("    " + p)
        return 1

    if "db_vs_uf_per_frame" not in pc["rows"][0]:
        print(f"  {a.curve} predates the per-frame convention; regenerate it")
        return 1

    sgm = {r["qp"]: r for r in sg["rows"]}
    print(f"  {pc['n_sequences']} sequences, {pc['frames_per_seq']} frame(s) each, "
          f"{Path(pc['ckpt']).parent.name}")
    print(f"  saving at a {a.budget:.2f} dB per-frame budget\n")
    print(f"  {'qp':>4}{'paper_curve':>14}{'signalled':>12}{'gap':>9}")

    worst, rows = 0.0, []
    for qp in sorted(sgm):
        pts = sorted((r["db_vs_uf_per_frame"], r["saving_pct"])
                     for r in pc["rows"] if r["qp"] == qp)
        D = [p[0] for p in pts]
        S = [p[1] for p in pts]
        if not (min(D) <= a.budget <= max(D)):
            print(f"  {qp:>4}   frontier does not reach the budget")
            continue
        x = float(np.interp(a.budget, D, S))
        y = sgm[qp]["saving_pct"]
        worst = max(worst, abs(x - y))
        rows.append({"qp": qp, "paper_curve": x, "signalled": y, "gap": x - y})
        print(f"  {qp:>4}{x:>13.1f}%{y:>11.1f}%{x - y:>+9.2f}")

    print(f"\n  worst disagreement {worst:.2f} points (tolerance {a.tol:.2f})")
    (ROOT / "results/crosscheck_paths.json").write_text(json.dumps(
        {"budget_db": a.budget, "worst_gap_pts": worst, "rows": rows}, indent=2))
    if worst > a.tol:
        print("  THE TWO PATHS DISAGREE. Something differs between them that is "
              "not accounted for -- check the dB convention, the lambda "
              "readout and the frame count before trusting either number.")
        return 1
    print("  agreed.")
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
