"""Turn a trainer resume-file into something the evaluation scripts may read.

The problem this exists for
---------------------------
`train_flexuf_image.py` writes three kinds of file:

    status_latest.pth.tar   every epoch  {"net", "opt", "epoch"}
    ckpt_epo{N}.pth.tar     every 5th    {"state_dict", "epoch", "config"}
    ckpt_step.pth.tar       --ckpt_every {"state_dict", "config", "epoch", "step"}

Measured epoch durations on this dataset are 13-21 h, and the runs are budgeted
for 2-3 days. So no run reaches epoch 5, and `ckpt_epo0` is the ONLY numbered
checkpoint any of them will ever produce -- one measurement per run for the
whole experiment, with every later epoch invisible.

`status_latest` holds the same weights and is written every epoch. But it
carries no `config`, and every evaluation script does

    cfg = FlexUFConfig(**ck["config"]) if "config" in ck else FlexUFConfig()

so reading it directly builds a DEFAULT ladder. For BEST128 (128px tiles) and
FINE12 (K=12) that is not the model that was trained. FINE12 would at least
fail loudly, because a K=6 ladder cannot absorb K=12 weights; BEST128 would
not -- the shapes match and only the tile size differs, so the run would be
measured under the wrong tiling and report a number nobody could trace.

So the config is taken from `meta.json`, which the trainer writes at startup
from the same `cfg` object, and stamped into a proper evaluation checkpoint.

Reading a file the trainer is writing
-------------------------------------
`torch.save` to `status_latest` is not atomic, so a read racing the write gets a
truncated file. The write happens once per 13-21 h and takes a few seconds, so
requiring the file to be at least `--min_age` seconds old makes the race
vanishingly unlikely, and the load is guarded anyway: a failure here must not
look like a result.
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import torch


def promote(run_dir: Path, min_age: float = 120.0) -> Path | None:
    """Write `<run>/ckpt_eval.pth.tar` from the newest usable state. None if nothing new."""
    src = run_dir / "status_latest.pth.tar"
    meta_p = run_dir / "meta.json"
    if not src.exists() or not meta_p.exists():
        return None

    age = time.time() - src.stat().st_mtime
    if age < min_age:
        print(f"{run_dir.name}: status_latest is {age:.0f}s old, "
              f"waiting for {min_age:.0f}s to be sure the write finished")
        return None

    dst = run_dir / "ckpt_eval.pth.tar"
    if dst.exists() and dst.stat().st_mtime >= src.stat().st_mtime:
        return None  # already promoted this one

    try:
        ck = torch.load(src, map_location="cpu", weights_only=False)
    except Exception as e:  # truncated, or mid-write despite the age check
        print(f"{run_dir.name}: could not read {src.name} ({e}); not promoting")
        return None

    sd = ck.get("net") or ck.get("state_dict")
    if sd is None:
        print(f"{run_dir.name}: {src.name} holds no weights; not promoting")
        return None

    cfg = json.loads(meta_p.read_text())["config"]
    tmp = dst.with_suffix(".tmp")
    torch.save({"state_dict": sd, "config": cfg, "epoch": ck.get("epoch")}, tmp)
    tmp.replace(dst)
    print(f"{run_dir.name}: epoch {ck.get('epoch')} -> {dst.name} "
          f"(K={cfg.get('num_exits')}, j={cfg.get('split_depth')}, "
          f"p={cfg.get('latent_patch')})")
    return dst


def main(argv):
    ap = argparse.ArgumentParser()
    ap.add_argument("runs", nargs="+", help="run directories")
    ap.add_argument("--min_age", type=float, default=120.0)
    a = ap.parse_args(argv)
    made = [promote(Path(r), a.min_age) for r in a.runs]
    return 0 if any(m is not None for m in made) else 1


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
