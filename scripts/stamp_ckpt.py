"""Record which checkpoint a result file was measured on, if it does not say.

Twenty-nine of the files the documents read carried no checkpoint field, so
after a repin nobody could tell whether they had moved. Rather than edit
twenty producers, the driver stamps each output with the checkpoint it just
passed in. A file that already names a checkpoint is left alone: its producer
knows better than the driver does.

    python scripts/stamp_ckpt.py results/foo.json runs/RECIPE512/ckpt_PAPER.pth.tar
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

R = Path(__file__).resolve().parent.parent


def stamp(path: str, ckpt: str) -> int:
    p = Path(path) if Path(path).is_absolute() else R / path
    if not p.exists() or p.stat().st_size == 0:
        print(f"  {p.name}: absent, not stamped")
        return 1
    try:
        d = json.loads(p.read_text())
    except Exception as e:
        print(f"  {p.name}: not JSON ({e})")
        return 1
    if not isinstance(d, dict):
        print(f"  {p.name}: not an object, cannot stamp")
        return 1
    if "ckpt" in d and "ckpt_epoch" in d:
        return 0
    import torch
    cp = Path(ckpt) if Path(ckpt).is_absolute() else R / ckpt
    epoch = None
    try:
        epoch = torch.load(cp, map_location="cpu",
                           weights_only=False).get("epoch")
    except Exception as e:
        print(f"  cannot read {cp}: {e}")
        return 1
    d.setdefault("ckpt", str(ckpt))
    if epoch is not None:
        d.setdefault("ckpt_epoch", int(epoch))
    d.setdefault("stamped_by", "scripts/stamp_ckpt.py")
    p.write_text(json.dumps(d, indent=2))
    print(f"  {p.name}: stamped epoch {epoch}")
    return 0


if __name__ == "__main__":
    if len(sys.argv) < 3:
        raise SystemExit(__doc__)
    raise SystemExit(stamp(sys.argv[1], sys.argv[2]))
