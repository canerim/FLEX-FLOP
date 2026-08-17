"""Does the trade-off conclusion survive a different integration interval?

A BD number is defined by the interval it integrates over, so the obvious
objection to quoting one is that the interval was chosen to flatter it. The
answer has to be measured, not asserted: recompute over a range of defensible
intervals and show what moves and what does not.

What moves: the absolute mean BD-saving, by about six points across the
intervals tried. That is expected and is why the interval is always printed
next to the number.

What does not: the ordering and the size of the rate effect. qp0 exceeds qp63
by 13.8 to 16.3 points on every interval, so "the saving falls with rate, and
falls by roughly fifteen points across the QP range" is a statement about the
decoder rather than about the interval.

Intervals wide enough that a rate's frontier does not span them are reported as
such and excluded, rather than silently averaged over fewer rates -- which
would make an interval look better simply by dropping the rate that struggles
in it.
"""

from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent


def combos(curve: str, convention: str):
    """Intervals to test, derived from the frontier rather than hard-coded.

    They were hard-coded first, chosen around the pooled floors. Switching to
    the per-frame convention raised every floor and four of the seven became
    unspannable, so the sweep silently shrank to three intervals -- the script
    said so, but a robustness check that quietly tests half of what it used to
    is not much of a check. The lower limits are anchored to the largest floor
    now, so the sweep is the same shape whichever convention is asked for.
    """
    d = json.loads((ROOT / curve).read_text())
    key = ("db_vs_uf_per_frame" if convention == "per_frame" else "db_vs_uf")
    floors = []
    for qp in sorted({r["qp"] for r in d["rows"]}):
        floors.append(min(r.get(key, r["db_vs_uf"]) for r in d["rows"]
                          if r["qp"] == qp))
    base = round(max(floors) + 5e-4, 3)
    return [(base, hi) for hi in (0.20, 0.25, 0.30, 0.40)] + \
           [(round(base + d_, 3), 0.30) for d_ in (0.02, 0.04)] + \
           [(round(base + 0.04, 3), 0.25)]


def main():
    conv = sys.argv[1] if len(sys.argv) > 1 else "per_frame"
    COMBOS = combos("results/paper_curve_grid128.json", conv)
    print(f"  BD-saving under different integration intervals "
          f"({conv} convention)\n")
    print(f"  {'dB interval':>18}{'mean':>10}{'qp0':>9}{'qp63':>9}"
          f"{'qp0 - qp63':>13}")
    out = []
    for lo, hi in COMBOS:
        r = subprocess.run(
            [str(ROOT / ".venv/bin/python"), str(ROOT / "scripts/bd_saving.py"),
             "--db_lo", str(lo), "--db_hi", str(hi), "--convention", conv,
             "--out", "/tmp/bd_tmp.json"],
            capture_output=True, text=True, cwd=ROOT)
        if r.returncode != 0:
            print(f"  [{lo:.3f},{hi:.3f}]  failed")
            continue
        d = json.loads(Path("/tmp/bd_tmp.json").read_text())
        ok = [x for x in d["rows"] if x["bd_saving_pct"] is not None]
        if len(ok) < len(d["rows"]):
            print(f"  [{lo:.3f},{hi:.3f}]  only {len(ok)}/{len(d['rows'])} rates "
                  f"span it -- excluded, since averaging over fewer rates would "
                  f"flatter the interval")
            continue
        m, a_, b_ = d["mean_bd_saving_pct"], ok[0], ok[-1]
        out.append({"db_lo": lo, "db_hi": hi, "mean_bd_saving_pct": m,
                    "qp_low": a_["bd_saving_pct"], "qp_high": b_["bd_saving_pct"],
                    "rate_effect_pts": a_["bd_saving_pct"] - b_["bd_saving_pct"]})
        print(f"  [{lo:.3f},{hi:.3f}]{m:>9.2f}%{a_['bd_saving_pct']:>8.1f}%"
              f"{b_['bd_saving_pct']:>8.1f}%{out[-1]['rate_effect_pts']:>12.1f} pt")

    if out:
        ms = [o["mean_bd_saving_pct"] for o in out]
        es = [o["rate_effect_pts"] for o in out]
        print(f"\n  mean BD-saving spans {min(ms):.1f}-{max(ms):.1f}% "
              f"({max(ms) - min(ms):.1f} points) across intervals,")
        print(f"  but the rate effect stays {min(es):.1f}-{max(es):.1f} points. "
              f"The level depends on the")
        print(f"  interval; the conclusion does not.")
        (ROOT / "results/bd_sensitivity.json").write_text(
            json.dumps({"convention": conv, "rows": out}, indent=2))
        print("\n  wrote results/bd_sensitivity.json")
    return 0


if __name__ == "__main__":
    sys.exit(main())
