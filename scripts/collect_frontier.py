"""Rebuild a frontier table from whatever per-beta evaluations exist on disk.

Split out of `sweep_frontier.sh` so the table is a *derived* artifact rather than
something accumulated as the sweep runs. The distinction matters: an accumulated
file is only correct if nothing ever interrupts the process writing it, and
something always does. Deriving it means the table can be regenerated at any
time, in any order, from partial results, and is always consistent with the
eval.json files actually present.

    python scripts/collect_frontier.py <sweep-dir> <out.tsv>
"""

from __future__ import annotations

import glob
import json
import os
import re
import sys


def main() -> int:
    if len(sys.argv) < 3:
        print(__doc__)
        return 2
    out_dir, sum_path = sys.argv[1], sys.argv[2]

    rows = []
    for ev in glob.glob(os.path.join(out_dir, "beta_*", "eval.json")):
        m = re.search(r"beta_([0-9.]+)", ev)
        if not m:
            continue
        try:
            d = json.load(open(ev))
        except Exception:
            continue  # a partially written file from an interrupted run
        for qp, res in d.get("results", {}).items():
            r = res.get("routed")
            if r is None:
                continue
            rows.append((
                float(m.group(1)), int(qp),
                r["saving_pct"], r["psnr_loss_dB"], r["exit_share"],
                d.get("control_max_diff"),
            ))

    rows.sort()
    with open(sum_path, "w") as f:
        f.write("beta\tqp\tsaving_pct\tpsnr_loss_dB\texit_share\tcontrol_max_diff\n")
        for beta, qp, sv, db, share, ctrl in rows:
            f.write(f"{beta:g}\t{qp}\t{sv}\t{db}\t{share}\t{ctrl}\n")

    if not rows:
        print("  (no completed evaluations yet)")
        return 0

    for beta, qp, sv, db, share, ctrl in rows:
        # A frontier point measured against a non-bit-exact anchor is not a
        # frontier point, so the control travels with every row.
        warn = "" if ctrl == 0.0 else f"  !! control={ctrl}"
        print(f"  beta={beta:>7g} qp={qp}  saving {sv:6.2f}%  "
              f"PSNR loss {db:+.4f} dB  share {share}{warn}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
