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
BASELINE = 20
MOVING = {"ckpt_eval.pth.tar", "ckpt_step.pth.tar"}
IDENT = ("ckpt_epoch", "ckpt_step", "epoch", "step", "cumulative_step",
         "ckpt_mtime", "ckpt_sha")


def read_by_documents() -> set[str]:
    used: set[str] = set()
    for f in [R / "scripts/build_pdf.py", R / "scripts/make_paper_tables.py"] + \
             [Path(x) for x in sorted(glob.glob(str(R / "scripts/supp/*.py")))]:
        used |= set(re.findall(r"([A-Za-z0-9_.\-]+\.json)", f.read_text()))
    return used


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
    over = len(lost) > BASELINE
    if over:
        print(f"\n  {len(lost)} lost, baseline {BASELINE} -- a new measurement "
              f"has been added without recording its checkpoint")
    else:
        print(f"\n  PASS ({len(lost)} lost against a baseline of {BASELINE})")
    return 1 if over else 0


if __name__ == "__main__":
    sys.exit(main())
