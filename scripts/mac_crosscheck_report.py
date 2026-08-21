"""Report the MAC cross-check, and fail if the two counters have drifted.

scripts/mac_crosscheck.py does the work and needs a forward pass; this reads
what it wrote, so check_paper can run without a card. If the file is missing
the check says so rather than passing quietly -- an absent cross-check is not
a passed one.

    python scripts/mac_crosscheck_report.py
"""
from __future__ import annotations

import json
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
SRC = ROOT / "results/mac_crosscheck.json"
TOL = 0.01          # points of saving


def main() -> int:
    if not SRC.exists():
        print("  no MAC cross-check yet: run scripts/mac_crosscheck.py")
        return 1
    d = json.loads(SRC.read_text())
    rows = d["rows"]
    worst = max(abs(r["saving_pct_fvcore"] - r["saving_pct_macmeter"])
                for r in rows)
    for r in rows:
        gap = abs(r["saving_pct_fvcore"] - r["saving_pct_macmeter"])
        if gap > TOL:
            print(f"     exit {r['exit']}: MacMeter "
                  f"{r['saving_pct_macmeter']:.3f}% against fvcore "
                  f"{r['saving_pct_fvcore']:.3f}%")
    # The ceiling is the number the paper prints, so it is named separately.
    print(f"  ceiling {d['ceiling_pct_macmeter']:.2f}% (MacMeter) against "
          f"{d['ceiling_pct_fvcore']:.2f}% (fvcore), "
          f"worst exit differs by {worst:.4f} points")
    return 1 if worst > TOL else 0


if __name__ == "__main__":
    raise SystemExit(main())
