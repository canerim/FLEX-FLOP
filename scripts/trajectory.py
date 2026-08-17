"""How a run's frontier moves as it trains.

Every checkpoint is measured and the results accumulate in results/ as
signalled_<TAG>_<time>.json. Read as a series rather than one at a time they
answer the question each new checkpoint exists to answer: is this still
improving, and where.

Only files carrying ckpt_epoch/ckpt_step are plotted. Earlier ones name a
checkpoint path that was later overwritten -- ckpt_step.pth.tar is rewritten
every --ckpt_every steps -- so their position on a training axis is unknown, and
placing them by file mtime would put a measurement wherever the evaluation queue
happened to reach it. They are listed as unplaceable instead of guessed at.

    python scripts/trajectory.py            # all runs
    python scripts/trajectory.py BEST       # one
"""

from __future__ import annotations

import glob
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
STEPS_PER_EPOCH = 47451


def series(tag: str):
    """(cumulative step, {qp: saving}) per measurement, oldest first."""
    out, unplaceable = [], 0
    for f in glob.glob(str(ROOT / f"results/signalled_{tag}_*.json")):
        d = json.loads(Path(f).read_text())
        e, st = d.get("ckpt_epoch"), d.get("ckpt_step")
        if e is None:
            unplaceable += 1
            continue
        # An epoch checkpoint has no step: it is written when the epoch ends.
        cum = (e + 1) * STEPS_PER_EPOCH if st is None else e * STEPS_PER_EPOCH + st
        out.append((cum, {r["qp"]: r.get("saving_pct") for r in d["rows"]}))
    return sorted(out), unplaceable


def main(argv):
    tags = argv or sorted({Path(f).name.split("_")[1]
                           for f in glob.glob(str(ROOT / "results/signalled_*_*.json"))})
    any_series = False
    for tag in tags:
        pts, skipped = series(tag)
        if not pts:
            print(f"  {tag}: no measurement carries a checkpoint position"
                  + (f" ({skipped} unplaceable)" if skipped else ""))
            continue
        any_series = True
        qps = sorted(pts[0][1])
        print(f"\n  {tag}" + (f"   ({skipped} earlier measurement(s) cannot be "
                              f"placed on the training axis)" if skipped else ""))
        print(f"  {'cum. step':>11}" + "".join(f"{'qp' + str(q):>9}" for q in qps))
        # Two entries at the same step are the same checkpoint measured twice,
        # which happens when a marker is cleared and the chain re-runs. Kept and
        # not deduplicated: identical numbers at the same step are a
        # reproducibility check that passed, and hiding them would hide that.
        for cum, row in pts:
            cells = "".join(
                f"{row[q]:>8.1f}%" if row.get(q) is not None else f"{'n/a':>9}"
                for q in qps)
            print(f"  {cum:>11,}{cells}")
        if len(pts) > 1:
            first, last = pts[0], pts[-1]
            deltas = [(last[1][q] - first[1][q]) for q in qps
                      if last[1].get(q) is not None and first[1].get(q) is not None]
            if deltas:
                print(f"  {'change':>11}" + "".join(
                    f"{d:>+8.1f} " for d in deltas))
    if not any_series:
        print("\n  Nothing to plot yet: a trajectory needs two measurements of the")
        print("  same run that both record where they sit in training. The next")
        print("  checkpoint of each run will provide the first placeable one.")
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
