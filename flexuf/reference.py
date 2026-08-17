"""Pick the warm-start file that matches a checkpoint's ladder shape.

Every measurement here compares against the RELEASED DCVC-UF decoder, and the
way that decoder is made comparable is `scripts/warmstart_from_release.py`,
which re-expresses the released weights in a K-exit ladder. The re-expression is
a bijection over the released tensors, so the deepest exit is bit-exact -- but
the file's key layout depends on K, because K decides how the 12 residual blocks
are grouped.

Three scripts had `runs/warmstart/ckpt_warmstart.pth.tar` written into them,
which is the K=6 file. FINE12 trains a K=12 ladder, and its first evaluation
failed on two of three checks with

    missing ['dec.groups.6.0.dc.0.weight', ...]
    unexpected ['dec.groups.0.1.dc.0.weight', ...]

That is the guard in `load_flexuf_state` working exactly as intended -- a
mismatched reference raises instead of loading quietly, and a reference loaded
quietly would have made every number below it meaningless while looking fine.
The failure was the right outcome; needing a human to notice it was not.

Resolution is by K, since that is what the file's layout depends on.
"""

from __future__ import annotations

from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
WARMSTART_DIR = ROOT / "runs" / "warmstart"


def reference_for(cfg, explicit: str | None = None) -> str:
    """Path to the released-DCVC-UF reference expressed in `cfg`'s ladder.

    `explicit` wins when given, so a caller can still point at any file; it is
    returned unchanged and not checked against cfg, because overriding the
    default is exactly the case where the caller knows better.
    """
    if explicit:
        return explicit
    k = getattr(cfg, "num_exits", 6)
    p = (WARMSTART_DIR / "ckpt_warmstart.pth.tar" if k == 6
         else WARMSTART_DIR / f"ckpt_warmstart_K{k}.pth.tar")
    if not p.exists():
        raise SystemExit(
            f"no reference for a K={k} ladder at {p}.\n"
            f"Build it with:  FLEXUF_K={k} FLEXUF_J=<j> "
            f"./.venv/bin/python scripts/warmstart_from_release.py\n"
            f"Refusing to substitute the K=6 file: it would either raise inside "
            f"load_flexuf_state or, worse, load and silently measure against the "
            f"wrong decoder.")
    return str(p)
