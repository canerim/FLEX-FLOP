"""Which checkpoint a script reads when nobody says.

Fourteen scripts defaulted to runs/<run>/ckpt_eval.pth.tar. That file is not a
checkpoint, it is a name the watcher rewrites every time a new one lands, so
re-running any of those scripts a week later reproduces the command and not
the result. Several of them draw figures the paper prints.

pinned() prefers a pin in the same run directory -- ckpt_PAPER.pth.tar, the
file every table in the paper was computed from -- and falls back to the
moving name with a line on stderr saying so, since a run that was never
pinned is worth knowing about rather than worth crashing on.

    from ckpt import pinned
    ap.add_argument("--ckpt", default=pinned("runs/BEST/ckpt_eval.pth.tar"))
"""
from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
PINS = ("ckpt_PAPER.pth.tar",)


def pinned(path: str) -> str:
    p = Path(path)
    if p.name not in ("ckpt_eval.pth.tar", "ckpt_step.pth.tar"):
        return path
    for name in PINS:
        cand = p.with_name(name)
        if (ROOT / cand).exists():
            return str(cand)
    print(f"  {p.parent.name} has no pinned checkpoint; using {p.name}, "
          f"which the watcher rewrites", file=sys.stderr)
    return path
