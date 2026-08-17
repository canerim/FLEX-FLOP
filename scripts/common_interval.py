"""Pick a dB interval that every curve being compared actually spans.

Two failures led here, and they are opposites.

Deriving the interval per curve made checkpoints incomparable: BEST128's
tighter anchor put its floor at 0.033 where the baseline sat at 0.064, so the
two were integrated over different spans and BEST128 came out 1.1 points worse
when it was 0.6 better.

Pinning it to the baseline's [0.064, 0.30] then broke the other way. BEST's
exits are good enough that its whole frontier at qp0 spans only 0.197 dB --
its shallowest exit costs that much, against the baseline's 0.309 -- so the
fixed upper limit ran off the end of the curve, bd_saving correctly returned
None for qp0 and qp16, and the reported mean silently became a mean over three
rates compared against the baseline's five. Which is the first mistake again,
one level up.

The interval has to come from the curves being compared, all of them:

    lower = the largest floor over every (curve, rate)
    upper = the smallest maximum over every (curve, rate)

Every curve then spans it, no rate is dropped, and the numbers are means over
the same set. A better decoder narrowing the interval is not a defect; a
frontier that reaches 42.5% saving for 0.197 dB is simply a shorter curve, and
comparing over the part they share is the only comparison available.

    python scripts/common_interval.py results/curve_BEST.json results/paper_curve_grid128.json
"""

from __future__ import annotations

import json
import math
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent


def span(path: Path, key="db_vs_uf_per_frame"):
    """(largest floor, smallest max) over the rates in one curve file."""
    d = json.loads(path.read_text())
    floors, tops = [], []
    for qp in sorted({r["qp"] for r in d["rows"]}):
        v = [r.get(key, r["db_vs_uf"]) for r in d["rows"] if r["qp"] == qp]
        floors.append(min(v))
        tops.append(max(v))
    return max(floors), min(tops)


def common(paths) -> tuple[float, float]:
    lo, hi = -math.inf, math.inf
    for p in paths:
        a, b = span(Path(p))
        lo, hi = max(lo, a), min(hi, b)
    # Round inward so the printed interval is clean AND still spanned.
    return math.ceil(lo * 1000) / 1000, math.floor(hi * 1000) / 1000


def main(argv):
    if not argv:
        argv = [str(p) for p in sorted((ROOT / "results").glob("curve_*.json"))]
        argv.append(str(ROOT / "results/paper_curve_grid128.json"))
    argv = [a for a in argv if Path(a).exists()]
    if len(argv) < 1:
        raise SystemExit("no curve files")
    for p in argv:
        a, b = span(Path(p))
        print(f"  {Path(p).name:<32} spans [{a:.4f}, {b:.4f}]")
    lo, hi = common(argv)
    if hi <= lo:
        raise SystemExit(f"\n  these curves share no interval: [{lo}, {hi}]. "
                         f"Comparing their integrals is not possible; quote "
                         f"per-rate points instead.")
    print(f"\n  common interval: [{lo:.3f}, {hi:.3f}]")
    print(f"  use it for every curve in the comparison:")
    print(f"    --db_lo {lo:.3f} --db_hi {hi:.3f}")
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
