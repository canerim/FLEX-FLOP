"""Which of the numbers in these documents can be traced to a set of weights.

Every results file records the checkpoint it was measured on. Three kinds of
answer are possible and only two of them are useful:

  pinned          ckpt_PAPER, ckpt_PIN_*, a warm start -- a file that does not
                  move, so the measurement can be repeated exactly;
  identifiable    a name the watcher overwrites, but the file records the epoch
                  or step it was at, so the weights are named even though the
                  path is not;
  lost            a name the watcher overwrites and nothing else. The path
                  points at whatever that run has trained to since.

The third kind is not a bug -- the supplement says so about several files by
name -- but it is a number the paper should know about itself, and it is a
number that should not grow. This is a ratchet: it fails when the count of
lost-provenance files read by either document goes above the baseline below.
Lowering the baseline after fixing one is the point.

    python scripts/check_provenance.py
"""
from __future__ import annotations

import glob
import json
import os
import re
import sys
from pathlib import Path

R = Path(__file__).resolve().parent.parent
# Lowered from 20 on 2026-08-21: the adapter and coupling ablations were
# traced back to the pinned checkpoint and re-measured on it. Lowering this
# after converting one is the point of the number.
BASELINE = 18
MOVING = {"ckpt_eval.pth.tar", "ckpt_step.pth.tar"}
IDENT = ("ckpt_epoch", "ckpt_step", "epoch", "step", "cumulative_step",
         "ckpt_mtime", "ckpt_sha")


def read_by_documents() -> set[str]:
    used: set[str] = set()
    for f in [R / "scripts/build_pdf.py", R / "scripts/make_paper_tables.py"] + \
             [Path(x) for x in sorted(glob.glob(str(R / "scripts/supp/*.py")))]:
        used |= set(re.findall(r"([A-Za-z0-9_.\-]+\.json)", f.read_text()))
    return used


# The commit that corrected the FFN adapter's cost from 2C^2 to 5C^2
# (8792f0b, 2026-08-19 08:26). Anything modelled before it used the wrong
# adapter cost, which is how the ceiling read 41.9 instead of 39.1 and how the
# coupling ablation's savings came out two and a half points high.
COST_FIX = 1787120787          # commit 8792f0b, 2026-08-19 08:26
STALE_BASELINE = 28
SAVKEY = re.compile(r"saving|saved|cost|ceiling|gmac|kmac", re.I)


def pre_cost_fix(used: set[str]) -> list[tuple[str, str]]:
    """Files the documents read that were measured before the cost model was
    corrected, and that carry a saving or a cost.

    Not all of them are wrong: a hook-counted saving does not go through the
    model at all, and several of these are named in the supplement as modelled
    and optimistic by a stated amount. What the list is for is knowing which
    numbers are in that category without having to remember.
    """
    import datetime
    out = []
    for f in sorted(glob.glob(str(R / "results/*.json"))):
        b = os.path.basename(f)
        if b not in used or os.path.getsize(f) > 20_000_000:
            continue
        mt = os.path.getmtime(f)
        if mt >= COST_FIX:
            continue
        try:
            if not SAVKEY.search(open(f).read(200_000)):
                continue
        except Exception:
            continue
        out.append((datetime.datetime.fromtimestamp(mt).strftime("%m-%d %H:%M"), b))
    return out


def main() -> int:
    used = read_by_documents()
    pinned = ident = 0
    lost = []
    for f in sorted(glob.glob(str(R / "results/*.json"))):
        if os.path.getsize(f) > 20_000_000:
            continue
        try:
            d = json.loads(open(f).read())
        except Exception:
            continue
        if not isinstance(d, dict):
            continue
        ck = str(d.get("ckpt") or d.get("weights") or "")
        if not ck:
            continue
        base = os.path.basename(f)
        if base not in used:
            continue
        if os.path.basename(ck) not in MOVING:
            pinned += 1
        elif any(d.get(k) not in (None, "None", "") for k in IDENT):
            ident += 1
        else:
            lost.append(base)
    print(f"  of the results files these documents read: {pinned} on a "
          f"checkpoint that does not move, {ident} on one that does but "
          f"recording which, {len(lost)} recording neither")
    for b in sorted(lost):
        print(f"     {b}")
    stale = pre_cost_fix(used)
    print(f"\n  {len(stale)} of them were measured before the cost model was "
          f"corrected on 2026-08-19 and carry a saving or a cost")
    for d, b in stale[:6]:
        print(f"     {d}  {b}")
    if len(stale) > 6:
        print(f"     ... and {len(stale) - 6} more")

    over_lost = len(lost) > BASELINE
    over_stale = len(stale) > STALE_BASELINE
    over = over_lost or over_stale
    if over_lost:
        print(f"\n  {len(lost)} lost against a baseline of {BASELINE} -- a "
              f"measurement has been added without recording its checkpoint")
    if over_stale:
        print(f"\n  {len(stale)} pre-cost-fix against a baseline of "
              f"{STALE_BASELINE} -- a file older than the correction has been "
              f"brought into the documents")
    if over:
        pass
    else:
        print(f"\n  PASS ({len(lost)} lost against {BASELINE}, "
              f"{len(stale)} pre-cost-fix against {STALE_BASELINE})")
    return 1 if over else 0


if __name__ == "__main__":
    sys.exit(main())
