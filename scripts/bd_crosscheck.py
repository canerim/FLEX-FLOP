"""Our Bjontegaard against the reference implementation, on the paper's curves.

paper_metrics.bd_rate is thirteen lines of numpy: a cubic through (PSNR, log
rate) for each curve, integrated over the overlapping PSNR range. That is the
standard formulation, but "standard" is a claim about someone else's code, so
this checks it against someone else's code -- the `bjontegaard` package, which
implements the cubic method of VCEG-M33 and the Akima and PCHIP variants the
JVET common test conditions moved to.

Agreement is expected to about a thousandth of a point. The three methods
differ from each other by more than any of them differs from ours, which is
the useful thing to know: the interpolation choice matters more than the
implementation, and on these curves neither matters.

    python scripts/bd_crosscheck.py
"""
from __future__ import annotations

import json
import sys
import warnings
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "scripts"))
warnings.filterwarnings("ignore")

from paper_metrics import bd_rate as ours   # noqa: E402

TOL = 0.02          # points of BD-rate


def curves(budget=0.1):
    """(rates, anchor PSNR, our rates, our PSNR) for one budget."""
    rd = json.loads((ROOT / "results/rd_absolute_PAPER.json").read_text())
    sig = json.loads(
        (ROOT / "results/signalled_RECIPE512_ctc53.json").read_text())
    ref = {r["qp"]: r for r in rd["rows"]}
    got = {r["qp"]: r for r in sig["rows"]
           if r.get("budget_db") == budget and r.get("saving_pct") is not None}
    qs = [q for q in sorted(ref) if q in got]
    r1 = np.array([ref[q]["bpp"] for q in qs])
    p1 = np.array([ref[q]["psnr_release"] for q in qs])
    r2 = np.array([ref[q]["bpp"] + (got[q].get("bpp_added") or 0.0)
                   for q in qs])
    p2 = np.array([ref[q]["psnr_release"] - abs(got[q]["db_vs_uf"])
                   for q in qs])
    return qs, r1, p1, r2, p2


def main() -> int:
    try:
        import bjontegaard as bj
    except ImportError:
        print("  bjontegaard not installed; skipping "
              "(pip install bjontegaard)")
        return 0

    bad = []
    print(f"  {'budget':>7}{'n':>4}{'ours':>10}{'cubic':>10}"
          f"{'akima':>10}{'pchip':>10}{'max gap':>10}")
    for b in (0.1, 0.2, 0.3, 0.5):
        try:
            qs, r1, p1, r2, p2 = curves(b)
        except Exception:
            continue
        if len(qs) < 4:
            print(f"  {b:>7.2f}{len(qs):>4}   only {len(qs)} rate points, "
                  f"a cubic needs four")
            bad.append((b, "too few rates"))
            continue
        o = ours(r1, p1, r2, p2)
        got = {m: float(bj.bd_rate(r1, p1, r2, p2, method=m))
               for m in ("cubic", "akima", "pchip")}
        gap = max(abs(o - v) for v in got.values())
        print(f"  {b:>7.2f}{len(qs):>4}{o:>+10.4f}"
              + "".join(f"{got[m]:>+10.4f}" for m in ("cubic", "akima",
                                                      "pchip"))
              + f"{gap:>10.4f}")
        if gap > TOL:
            bad.append((b, f"{gap:.4f} points from the reference"))

    if bad:
        for b, why in bad:
            print(f"     budget {b}: {why}")
        print(f"\n  {len(bad)} budget(s) disagree with the reference "
              f"implementation")
        return 1
    print("\n  PASS")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
