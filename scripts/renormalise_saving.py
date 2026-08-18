"""Re-express saving against the RELEASED decoder instead of our own deepest exit.

Both curve producers compute

    saving = 1 - cost[k].mean() / cost[-1]

where `cost` is `exit_costs(...)` in units of one stock decode. `cost[-1]` is
our ladder at full depth, which is **1.0095**, not 1.0: the deepest exit pays
`seam_repair="grid"` (+0.95%) that the released decoder does not.

So every headline number answers "how much does early exiting save compared to
running OUR ladder at full depth", while every sentence around it claims "saved
versus the released DCVC-UF decoder". Those differ, always in our favour, and
the seam-repair tax -- which the accounting section claims is included -- is
exactly what falls out of the denominator.

The fix is exact algebra on the stored curves, no GPU and no re-measurement.
With `D = cost[-1]` and `s` the reported saving,

    cost_fraction   X = (1 - s/100) * D          (in units of a stock decode)
    saving vs stock   = 100 * (1 - X)

The relation is affine, so it can be applied to any stored row, and it lowers
every number by roughly 0.55-0.75 points. Ranking between checkpoints is
unaffected -- a monotone map cannot reorder them -- but the distance to the 30%
target is not, and neither is any absolute claim.

    python scripts/renormalise_saving.py                    # the headline table
    python scripts/renormalise_saving.py --write            # patch results/*.json
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))


def deepest_cost(cfg_dict: dict) -> float:
    from flexuf.config import FlexUFConfig
    from flexuf.cost import exit_costs
    return float(exit_costs(FlexUFConfig(**cfg_dict), "head")[-1])


def vs_release(saving_pct: float, D: float) -> float:
    """Reported saving (÷ our deepest) -> saving ÷ the released decoder."""
    return 100.0 * (1.0 - (1.0 - saving_pct / 100.0) * D)


def main(argv):
    ap = argparse.ArgumentParser()
    ap.add_argument("--curve", default="results/signalled_BEST_0817_1542.json")
    ap.add_argument("--ckpt_cfg", default="runs/BEST/meta.json")
    ap.add_argument("--write", action="store_true",
                    help="add saving_pct_vs_release to the file in place")
    a = ap.parse_args(argv)

    cfg = json.loads((ROOT / a.ckpt_cfg).read_text())["config"]
    D = deepest_cost(cfg)
    print(f"  ladder at full depth costs {D:.4f} stock decodes "
          f"(+{100*(D-1):.2f}% seam repair)\n")

    f = ROOT / a.curve
    d = json.loads(f.read_text())
    print(f"  {'qp':>4}{'reported':>11}{'vs release':>13}{'difference':>13}")
    for r in d["rows"]:
        s = r.get("saving_pct")
        if s is None:
            continue
        v = vs_release(s, D)
        r["saving_pct_vs_release"] = v
        print(f"  {r['qp']:>4}{s:>10.2f}%{v:>12.2f}%{v - s:>12.2f}")

    if a.write:
        d["saving_denominator"] = {
            "reported_saving_pct_divides_by": "our deepest exit",
            "deepest_exit_cost_in_stock_decodes": D,
            "saving_pct_vs_release_divides_by": "the released decoder (1.0)"}
        f.write_text(json.dumps(d, indent=2))
        print(f"\n  wrote {a.curve}")
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
