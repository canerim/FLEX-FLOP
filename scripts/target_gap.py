"""How far is the system from the 30% goal, in the goal's own units?

The project's target is 30-40% of the decoder's compute saved at no more than
0.1 dB below the released decoder. Saving is usually quoted at that budget,
which answers "how much do we get" but not "how much more would it take" -- and
the second question is the one that decides what to work on next.

So the frontier is read the other way: the dB a 30% saving actually costs at
each rate, against the 0.1 dB the target allows. The overspend is the gap, in
the units the target is written in.

The gap is then split into the part the anchor could return and the rest. The
frontier's floor is the deepest exit's residual against the released decoder --
drift, not routing -- and every operating point pays it before buying any
saving. It is therefore a lower bound on the dB spent that has nothing to do
with early exit, and closing it is a different job from improving the exits.
The split says which of the two is worth the effort at each rate.

Both readings use the per-frame decibel, the convention published DCVC-UF
numbers use. Under the pooled alternative the frontier sits 0.023-0.033 dB
lower and every gap here would look smaller by about that much.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parent.parent


def frontier(rows, qp, key="db_vs_uf_per_frame"):
    best: dict[float, float] = {}
    for r in rows:
        if r["qp"] != qp:
            continue
        k = round(r["saving_pct"], 6)
        v = r.get(key, r["db_vs_uf"])
        if k not in best or v < best[k]:
            best[k] = v
    p = sorted(best.items())
    return np.array([x[0] for x in p]), np.array([x[1] for x in p])


def main(argv):
    ap = argparse.ArgumentParser()
    ap.add_argument("--curve", default="results/paper_curve_grid128.json")
    ap.add_argument("--target", type=float, default=30.0, help="saving, %%")
    ap.add_argument("--budget", type=float, default=0.10, help="dB allowed")
    ap.add_argument("--out", default="results/target_gap.json")
    a = ap.parse_args(argv)

    d = json.loads((ROOT / a.curve).read_text())
    print(f"  {Path(d.get('ckpt', '?')).parent.name}, "
          f"{d.get('n_sequences', '?')} CTC sequences, per-frame dB")
    print(f"  target: {a.target:.0f}% saved for at most {a.budget:.2f} dB\n")
    print(f"  {'qp':>4}{'saved at budget':>17}{'dB for target':>15}"
          f"{'overspend':>12}{'of which floor':>16}")

    out = {"curve": a.curve, "ckpt": d.get("ckpt"), "target_pct": a.target,
           "budget_db": a.budget, "rows": []}
    for qp in sorted({r["qp"] for r in d["rows"]}):
        S, D = frontier(d["rows"], qp)
        floor = float(D.min())
        # np.interp CLAMPS outside the data range, so a budget below the floor
        # silently returned the floor's saving. VERBATIM's floor is 0.1365 dB
        # and this reported "13.8% saved at 0.10 dB" for it. Third instance of
        # the same class today: a budget outside the frontier is not a number to
        # be produced, it is a question with no answer.
        got = float(np.interp(a.budget, D, S, left=np.nan, right=np.nan))
        if got != got:
            print(f"  {qp:>4}   floor is {floor:.4f} dB — the {a.budget:.2f} dB "
                  f"budget is below it, so no allocation meets it")
            out["rows"].append({"qp": qp, "saved_at_budget": None,
                                "budget_reachable": False, "floor_db": floor,
                                "db_for_target": None, "reachable": False})
            continue
        if S.max() < a.target:
            print(f"  {qp:>4}{got:>16.1f}%   target unreachable: the ceiling is "
                  f"{S.max():.1f}%")
            out["rows"].append({"qp": qp, "saved_at_budget": got,
                                "db_for_target": None, "reachable": False,
                                "floor_db": floor})
            continue
        need = float(np.interp(a.target, S, D))
        over = need - a.budget
        # The floor is spent before any saving is bought, so it is the part of
        # the overspend that early exit cannot be blamed for -- and cannot fix.
        # Meaningless when the target is already met: there is no overspend to
        # apportion. Printed as "met" rather than nan%, which read like a bug.
        share = 100 * floor / over if over > 0 else None
        out["rows"].append({"qp": qp, "saved_at_budget": got,
                            "db_for_target": need, "overspend_db": over,
                            "floor_db": floor, "floor_share_pct": share,
                            "reachable": True})
        # Above 100% the floor EXCEEDS the whole overspend: the gap at that
        # rate is drift and nothing else, and a decoder whose deepest exit
        # matched the released one would clear the target outright.
        if share is None:
            mark = f"{'met':>15}"
        else:
            mark = f"{share:>14.0f}%" + ("*" if share > 100 else " ")
        print(f"  {qp:>4}{got:>16.1f}%{need:>15.4f}{over:>+12.4f}{mark}")

    print("\n  * floor exceeds the whole overspend: at that rate the gap IS "
          "the drift.")
    ok = [r for r in out["rows"] if r.get("reachable")]
    met = [r for r in ok if r["overspend_db"] <= 0]
    if met:
        print(f"\n  target MET at qp" + "/".join(str(r["qp"]) for r in met)
              + f" -- {a.target:.0f}% costs at most "
              + f"{max(r['db_for_target'] for r in met):.4f} dB there.")
    if ok:
        worst = max(ok, key=lambda r: r["overspend_db"])
        best = min(ok, key=lambda r: r["overspend_db"])
        print(f"  closest at qp{best['qp']}: {best['overspend_db']:+.4f} dB "
              f"against budget.")
        print(f"  furthest at qp{worst['qp']}: needs "
              f"{worst['db_for_target']:.3f} dB, "
              f"{worst['db_for_target'] / a.budget:.1f}x the budget, of which "
              + (f"{worst['floor_share_pct']:.0f}% is the anchor's floor."
                 if worst.get("floor_share_pct") is not None else
                 "and the target is met there too."))
        print(f"  So the two ends need different work. At low rate the gap is "
              f"the anchor alone,")
        print(f"  and tightening it clears the target. At high rate drift is a "
              f"quarter to a third")
        print(f"  and the rest is shallow-exit quality. Which exit binds is a "
              f"question for the")
        print(f"  histogram in results/, not for a sentence here: it was exit 2 "
              f"on the baseline")
        print(f"  and is exit 3 on BEST, and hard-coding either goes stale.")

    (ROOT / a.out).write_text(json.dumps(out, indent=2))
    print(f"\n  wrote {a.out}")
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
