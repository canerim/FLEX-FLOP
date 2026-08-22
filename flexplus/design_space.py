"""Which ladder shapes can reach 55% of the decoder at a 0.2 dB budget.

The paper's configuration is K=6 exits with the split at depth j=2, and its
architectural ceiling -- every tile on the shallowest rung it is allowed -- is
38.3% counted, 39.1% modelled. At 0.2 dB the trained system already delivers
36.8% of that 38.3%, which is 96% of the ceiling. The routing is not what is
short. The ceiling is.

So the question is not "route better" but "which (K, j) has a ceiling high
enough, and what does moving there cost". This enumerates the space from the
cost model alone -- no card, no checkpoint, seconds -- and marks the ones that
could reach the target if they routed as well as the current system does.

    python flexplus/design_space.py
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from flexuf.config import FlexUFConfig       # noqa: E402
from flexuf.cost import exit_costs           # noqa: E402

# What the deployed system gets out of its own ceiling, measured: the epoch-4
# run delivers 36.81% against a 38.33% counted ceiling at 0.2 dB. A new shape
# is credited with the same efficiency, not with perfect routing -- the point
# is to compare shapes, and assuming perfection would rank them by ceiling
# alone.
REALISED_AT_TWO_TENTHS = 36.81 / 38.33
TARGET = 55.0


def main() -> int:
    rows = []
    # K must divide the 12 trunk blocks: config.py refuses anything else,
    # because an exit that lands mid-block has no boundary to leave through.
    for K in (1, 2, 3, 4, 6, 12):
        for j in range(0, min(K, 6)):
            cfg = FlexUFConfig(num_exits=K, split_depth=j,
                               adapter_kind="scaled")
            c = exit_costs(cfg, "head")
            ceiling = 100 * (1 - float(c[j]))
            # Blocks that stop being shared when the split moves up. The seam
            # is a function of exactly this: b = (K - j) * blocks_per_exit,
            # and the contamination law makes the corrupted fraction
            # 1 - ((F - 2b)/F)^2.
            per_exit = 12 // K if K <= 12 else 1
            tiled_blocks = 12 - j * (12 / K)
            rows.append({
                "K": K, "j": j,
                "ceiling_pct": ceiling,
                "reachable_at_0.2db_pct": ceiling * REALISED_AT_TWO_TENTHS,
                "tiled_blocks": tiled_blocks,
                "deepest_cost": float(c[-1]),
                "n_adapters": K - 1,
                "meets_target": ceiling * REALISED_AT_TWO_TENTHS >= TARGET,
            })

    rows.sort(key=lambda r: (-r["reachable_at_0.2db_pct"], r["K"]))
    print(f"  ceiling and what it would deliver at 0.2 dB if the new shape "
          f"routed as well\n  as the current one does "
          f"({100 * REALISED_AT_TWO_TENTHS:.0f}% of its own ceiling)\n")
    print(f"  {'K':>3}{'j':>3}{'ceiling':>10}{'at 0.2 dB':>11}"
          f"{'tiled blocks':>14}{'adapters':>10}   target")
    for r in rows:
        if r["K"] not in (6, 12) and not r["meets_target"]:
            continue
        mark = "  <-- reaches 55%" if r["meets_target"] else ""
        print(f"  {r['K']:>3}{r['j']:>3}{r['ceiling_pct']:>9.2f}%"
              f"{r['reachable_at_0.2db_pct']:>10.1f}%"
              f"{r['tiled_blocks']:>14.0f}{r['n_adapters']:>10}{mark}")

    # The current configuration, named so the table has a reference point.
    cur = next(r for r in rows if r["K"] == 6 and r["j"] == 2)
    print(f"\n  the paper's shape is K=6 j=2: ceiling {cur['ceiling_pct']:.2f}%, "
          f"{cur['tiled_blocks']:.0f} tiled blocks")
    best = [r for r in rows if r["meets_target"]]
    if best:
        # Among the shapes that reach the target, the one that disturbs the
        # seam least: the seam grows with the number of per-tile blocks, and
        # a shape with the same tiled_blocks as today has today's seam.
        safest = min(best, key=lambda r: (r["tiled_blocks"], r["K"]))
        print(f"  fewest tiled blocks among those that reach it: "
              f"K={safest['K']} j={safest['j']}, {safest['tiled_blocks']:.0f} "
              f"blocks against the current {cur['tiled_blocks']:.0f}")
    out = Path(__file__).resolve().parent / "results/design_space.json"
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(
        {"realised_fraction_at_0.2db": REALISED_AT_TWO_TENTHS,
         "target_pct": TARGET, "rows": rows}, indent=2))
    print(f"\n  wrote {out.relative_to(ROOT)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
