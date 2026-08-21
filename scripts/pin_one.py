"""Copy one boundary checkpoint into a pin, weights only.

status_latest.pth.tar carries the optimiser state as well -- 140 MB of a
322 MB file -- and a pin only has to reproduce a decode. Written to the name
given and renamed by the caller, because the evaluation scripts poll for these.
"""
import json
import sys
from pathlib import Path

import torch


def main() -> int:
    src, dst = sys.argv[1], sys.argv[2]
    d = torch.load(src, map_location="cpu", weights_only=False)
    sd = d.get("state_dict") or d.get("net")
    if sd is None:
        print(f"  {src}: no weights in it", file=sys.stderr)
        return 1
    out = {"state_dict": sd, "epoch": d.get("epoch")}
    cfg = d.get("config")
    if cfg is None:
        # status_latest.pth.tar carries weights, optimiser and epoch and no
        # config; the run's meta.json has it. A pin that cannot say what
        # architecture it is is a pin somebody has to guess at later, which is
        # most of the reason the twenty lost files are lost.
        meta = Path(src).parent / "meta.json"
        if meta.exists():
            try:
                cfg = json.loads(meta.read_text())["config"]
            except Exception:
                cfg = None
    if cfg is not None:
        out["config"] = cfg
    torch.save(out, dst)
    return 0


if __name__ == "__main__":
    sys.exit(main())
