"""Do the numbers written in docs/ still match the files in results/?

Every table in docs/ was typed by hand from a measurement. Measurements are
recomputed whenever a checkpoint lands, and nothing in markdown notices. This
project has already shipped a figure captioned from one checkpoint while
plotting another (DECISIONS 54), and a saving quoted against the wrong
denominator across every document (58). A prose document is exactly where that
rot is invisible.

So: recompute the headline quantities from results/, and check each one appears
verbatim somewhere in docs/. A miss is not automatically an error -- a number
may have been superseded on purpose, or written to a different precision -- but
it is always something to look at.

    python scripts/check_docs.py          # exit 1 if anything is missing
"""

from __future__ import annotations

import json
import re
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from flexuf.config import FlexUFConfig  # noqa: E402
from flexuf.cost import exit_costs  # noqa: E402

DOCS = ["README.md", "01-experiment-plan.md", "02-architecture.md",
        "03-results.md", "04-open-questions.md", "05-decision-ab.md"]


def main() -> int:
    cfg = FlexUFConfig(**json.loads((ROOT / "runs/BEST/meta.json")
                                    .read_text())["config"])
    C = exit_costs(cfg, "head")
    D = float(C[-1])

    def saving(fname):
        p = ROOT / "results" / fname
        if not p.exists():
            return None
        d = json.loads(p.read_text())
        return {r["qp"]: (r.get("saving_pct_vs_release")
                          or 100 - (100 - r["saving_pct"]) * D)
                for r in d["rows"] if r.get("saving_pct") is not None}

    truth = {}
    A = saving("signalled_BEST_0817_1542.json")
    B1 = saving("router_BEST_v2.json")
    B2 = saving("router_BEST_v2_lowlam.json")
    for q in (0, 16, 32, 48, 63):
        if A:
            truth[f"A @0.1 dB, qp{q}"] = A[q]
        if B1 and B2:
            truth[f"B @0.1 dB, qp{q}"] = max(B1[q], B2[q])
    truth["ladder ceiling"] = 100 * (1 - float(C[cfg.split_depth]))
    # In units of a stock decode, which is how the docs write it -- not as a
    # percentage. Checking it as 100.95 reported a false miss.
    truth["deepest exit cost"] = D

    text = " ".join((ROOT / "docs" / d).read_text() for d in DOCS
                    if (ROOT / "docs" / d).exists())
    nums = set(re.findall(r"\d+\.\d+", text))

    print(f"  {'quantity':<22}{'computed':>10}   in docs/?")
    missing = 0
    for k, v in truth.items():
        found = any(f"{v:.{p}f}" in nums for p in (1, 2, 3, 4))
        print(f"  {k:<22}{v:>10.4f}   {'yes' if found else 'NO'}")
        missing += not found
    print(f"\n  {len(truth) - missing}/{len(truth)} present. "
          f"{'All headline numbers are current.' if not missing else ''}")
    return 1 if missing else 0


if __name__ == "__main__":
    sys.exit(main())
