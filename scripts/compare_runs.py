"""Tabulate every signalled measurement in results/, side by side.

Numbers accumulate one checkpoint at a time and get compared in a shell one-off,
which is where several of today's mistakes came from: two files measured on
different frame counts, two BD numbers integrated over different intervals, two
decibels that meant different things. Each looked like a result until the
metadata was checked.

So this refuses to put two files in the same table unless they agree on the
things that make a comparison mean anything -- sequence count, frames per
sequence, and the dB budget each was bisected to. Mismatched files are listed
below the table with what differs, rather than dropped silently or, worse,
included.

Configuration is printed alongside, because a difference in the numbers is only
attributable if you can see how many differences there are in the setup. FINE12
against BEST128 differs in four things at once, and no column of savings can
say which one moved.

    python scripts/compare_runs.py
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
RESULTS = ROOT / "results"


def config_of(tag: str) -> str:
    m = ROOT / "runs" / tag / "meta.json"
    if not m.exists():
        return ""
    c = json.loads(m.read_text()).get("config", {})
    return (f"K={c.get('num_exits')} j={c.get('split_depth')} "
            f"p={c.get('latent_patch')} {c.get('adapter_kind')}")


def steps_of(d: dict) -> str:
    """Cumulative step of the MEASURED checkpoint.

    This read the run's train_log and reported where training had got to, which
    labelled a step-4000 measurement as 6,800 -- the run had simply moved on.
    Older files predate ckpt_epoch/ckpt_step and get "?" rather than a number
    that is wrong."""
    e, st = d.get("ckpt_epoch"), d.get("ckpt_step")
    if e is None:
        return "?"
    return f"{e * 47451 + (st or 0):,}"


def main(argv):
    ap = argparse.ArgumentParser()
    ap.add_argument("--budget", type=float, default=0.10)
    a = ap.parse_args(argv)

    files = sorted(RESULTS.glob("signalled_*.json"))
    if not files:
        print("  no signalled measurements in results/")
        return 1

    # Newest per tag: a run is re-measured at each checkpoint and only the
    # latest describes the model as it stands.
    latest: dict[str, tuple[Path, dict]] = {}
    for f in files:
        d = json.loads(f.read_text())
        tag = Path(d.get("ckpt", "")).parent.name or f.stem
        if tag not in latest or f.stat().st_mtime > latest[tag][0].stat().st_mtime:
            latest[tag] = (f, d)

    key = lambda d: (d.get("n_sequences"), d.get("frames_per_seq"))  # noqa: E731
    counts: dict[tuple, int] = {}
    for _, d in latest.values():
        counts[key(d)] = counts.get(key(d), 0) + 1
    main_key = max(counts, key=counts.get)

    ok = {t: (f, d) for t, (f, d) in latest.items() if key(d) == main_key}
    odd = {t: (f, d) for t, (f, d) in latest.items() if key(d) != main_key}

    qps = sorted({r["qp"] for _, d in ok.values() for r in d["rows"]})
    print(f"  signalled system at a {a.budget:.2f} dB budget, "
          f"{main_key[0]} sequences, {main_key[1]} frame(s) each\n")
    hdr = f"  {'run':<18}{'ckpt step':>10}  {'config':<24}" + \
          "".join(f"{'qp' + str(q):>8}" for q in qps)
    print(hdr)
    for tag in sorted(ok):
        d = ok[tag][1]
        row = {r["qp"]: r for r in d["rows"]}
        cells = "".join(
            f"{row[q]['saving_pct']:>7.1f}%" if q in row else f"{'-':>8}"
            for q in qps)
        print(f"  {tag:<18}{steps_of(d):>10}  {config_of(tag):<24}{cells}")

    if odd:
        print("\n  NOT COMPARABLE, and left out of the table rather than mixed "
              "into it:")
        for tag, (_, d) in sorted(odd.items()):
            print(f"    {tag}: {d.get('n_sequences')} sequences, "
                  f"{d.get('frames_per_seq')} frame(s) -- the table is "
                  f"{main_key[0]}/{main_key[1]}")
    print("\n  Differences in the numbers are attributable only as far as the "
          "configs differ")
    print("  by one thing. Read the config column before reading the savings.")
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
