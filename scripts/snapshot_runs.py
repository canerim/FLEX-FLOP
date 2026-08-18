"""Freeze the live training configurations into a file the docs can read.

The flags a run was launched with live in exactly one place: the argv of the
process. `runs/<tag>/meta.json` records the model config but not the training
recipe, so `--anchor_weight`, `--distill_weight`, `--new_lr_scale` and the rest
exist only for as long as the process does. Every one of those is load-bearing
-- CONTROL and BEST differ in four of them and behave differently across their
second epoch -- so a figure or a document that describes the runs from memory
is describing something unverifiable.

This writes `docs/runs.json`: one entry per live run, the parsed argv, and the
progress read from its train_log.jsonl. Re-run it whenever a run is launched or
restarted; the file is what the topology figure and the experiment plan quote.

Runs that have exited are kept from the previous snapshot with `live: false`,
so killing a job does not silently delete it from the record.

    python scripts/snapshot_runs.py
"""

from __future__ import annotations

import json
import re
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
OUT = ROOT / "docs" / "runs.json"


def parse_argv(line: str) -> dict:
    """`--flag a b --other` -> {"flag": "a b", "other": True}."""
    # A value run ends at the next flag, and the trainer takes single-dash
    # flags too (-n, -e). Breaking only on "--" swallowed them, so VERBATIM's
    # epoch_offset read "90 -e 15" and batch_size read "8 -n 8 -e 16": the
    # figure then printed a schedule position that does not exist.
    def is_flag(t):
        return len(t) > 1 and t[0] == "-" and not t[1].isdigit()

    toks = line.split()
    d, i = {}, 0
    while i < len(toks):
        if is_flag(toks[i]):
            v, j = [], i + 1
            while j < len(toks) and not is_flag(toks[j]):
                v.append(toks[j])
                j += 1
            d[toks[i].lstrip("-")] = " ".join(v) if v else True
            i = j
        else:
            i += 1
    return d


def progress(tag: str) -> dict:
    f = ROOT / "runs" / tag / "train_log.jsonl"
    if not f.exists():
        return {}
    rs = [json.loads(l) for l in f.read_text().splitlines() if l.strip()]
    if not rs:
        return {}
    return {"epoch": max(r["epoch"] for r in rs), "step": rs[-1]["step"],
            "records": len(rs)}


def main() -> int:
    ps = subprocess.run(["ps", "-eo", "args"], capture_output=True,
                        text=True).stdout
    live = {}
    for l in ps.splitlines():
        if "train_flexuf_image.py" not in l or "grep" in l:
            continue
        m = re.search(r"--tag (\S+)", l)
        if not m or m.group(1) in live:
            continue  # dataloader workers repeat the parent's argv
        live[m.group(1)] = parse_argv(l[l.index("train_flexuf"):])

    old = json.loads(OUT.read_text())["runs"] if OUT.exists() else {}
    runs = {t: {**old.get(t, {}), "argv": a, "live": True, **progress(t)}
            for t, a in live.items()}
    for t, r in old.items():
        if t not in runs:
            runs[t] = {**r, "live": False, **progress(t)}

    # The flags that actually separate the runs. Everything else is shared and
    # would only pad the figure.
    keys = sorted({k for r in runs.values() for k in r["argv"]})
    ignore = {"save_dir", "tag", "device", "train_dataset", "pretrain"}
    differing = [k for k in keys
                 if k not in ignore
                 and len({str(r["argv"].get(k)) for r in runs.values()}) > 1]
    shared = {k: runs[next(iter(runs))]["argv"][k] for k in keys
              if k not in ignore and k not in differing}

    OUT.parent.mkdir(parents=True, exist_ok=True)
    OUT.write_text(json.dumps(
        {"runs": runs, "differing_flags": differing, "shared_flags": shared},
        indent=1))
    print(f"  {len(runs)} runs ({sum(r['live'] for r in runs.values())} live), "
          f"{len(differing)} differing flags -> docs/runs.json")
    for t in sorted(runs):
        r = runs[t]
        print(f"    {t:<11} {'live ' if r['live'] else 'ended'} "
              f"ep{r.get('epoch','?')} s{r.get('step','?')}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
