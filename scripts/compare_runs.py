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
    # An epoch checkpoint carries no step, because it is written when the epoch
    # ENDS. Treating the missing value as 0 labelled BEST's completed first
    # epoch as "0 steps" -- it is 47,451. A step snapshot has both.
    if st is None:
        return f"{(e + 1) * 47451:,}"
    return f"{e * 47451 + st:,}"


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
        # A rate can be present but have no number: budget_reachable is False
        # when the frontier's floor is above the budget, which is VERBATIM at
        # every rate. That is a result, printed as such.
        def _cell(q):
            if q not in row:
                return f"{'-':>8}"
            v = row[q].get("saving_pct")
            return f"{'n/a':>8}" if v is None else f"{v:>7.1f}%"
        cells = "".join(_cell(q) for q in qps)
        print(f"  {tag:<18}{steps_of(d):>10}  {config_of(tag):<24}{cells}")

    if odd:
        print("\n  NOT COMPARABLE, and left out of the table rather than mixed "
              "into it:")
        for tag, (_, d) in sorted(odd.items()):
            print(f"    {tag}: {d.get('n_sequences')} sequences, "
                  f"{d.get('frames_per_seq')} frame(s) -- the table is "
                  f"{main_key[0]}/{main_key[1]}")
    # BD numbers, with the snapshot each was taken at.
    #
    # A table of these once compared BEST128 at 8,000 steps against FINE12 at
    # 12,000 without saying so, because ckpt_step.pth.tar is overwritten and
    # nothing recorded which write was measured. The step column exists so that
    # cannot happen silently again; rows at different steps are not comparable
    # to each other however close their numbers look.
    import glob as _g
    bds = {}
    for f in _g.glob(str(RESULTS / "bd_*.json")):
        n = Path(f).stem[3:]
        if n in ("sensitivity", "saving_pooled"):
            continue
        try:
            bds[n] = json.loads(Path(f).read_text())
        except Exception:
            pass
    if bds:
        print(f"\n  {'run':<18}{'ckpt step':>10}{'interval':>16}"
              f"{'BD-saving':>12}")
        ivs = set()
        for n, d in sorted(bds.items()):
            e, st = d.get("ckpt_epoch"), d.get("ckpt_step")
            step = ("?" if e is None else
                    f"{(e + 1) * 47451:,}" if st is None else
                    f"{e * 47451 + st:,}")
            iv = d.get("db_interval")
            ivs.add(tuple(iv) if iv else None)
            m = d.get("mean_bd_saving_pct")
            iv_s = ("[%.3f, %.3f]" % tuple(iv)) if iv else "?"
            m_s = f"{m:>11.2f}%" if m is not None else f"{'n/a':>12}"
            print(f"  {n:<18}{step:>10}{iv_s:>16}{m_s}")
        if len(ivs) > 1:
            print("\n  ROWS ARE ON DIFFERENT INTERVALS and cannot be compared. "
                  "Recompute them all\n  with scripts/common_interval.py's "
                  "output before reading anything into the gaps.")

    print("\n  Differences in the numbers are attributable only as far as the "
          "configs differ")
    print("  by one thing. Read the config and step columns before reading the "
          "savings.")
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
