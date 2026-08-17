"""Check every number in `paper/theory.tex`'s tables against the JSON it came from.

`paper/README.md` claims the tables are checked programmatically before commit.
That claim was true only in the sense that the numbers had been compared by
hand at the time they were written -- which is not the same thing, and stops
being true the moment a script that produces one of the JSONs changes. A claim
about verification has to be re-runnable, so here it is.

    python scripts/verify_theory_tables.py        # exits non-zero on mismatch

Each table is parsed out of the .tex by its \\label, so the check breaks loudly
if a table is renamed or removed rather than silently passing on nothing.
"""

from __future__ import annotations

import json
import re
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
TEX = ROOT / "paper" / "theory.tex"


def rows_of(label: str) -> list[list[str]]:
    """Body rows of the tabular carrying `label`, as lists of stripped cells."""
    src = TEX.read_text()
    i = src.find(f"\\label{{{label}}}")
    if i < 0:
        raise SystemExit(f"no table labelled {label} in {TEX.name}")
    body = src[i:src.find("\\end{tabular}", i)]
    body = body[body.find("\\midrule") + len("\\midrule"):]
    out = []
    for line in body.splitlines():
        line = line.strip()
        if not line or line.startswith("\\"):
            continue
        line = line.rstrip("\\")
        cells = []
        for c in line.split("&"):
            # LaTeX decoration is not data: \textbf{...} wraps the highlighted
            # entry and \% is an escaped percent sign, both of which would
            # otherwise reach float().
            c = c.replace("\\textbf{", "").replace("}", "").replace("\\%", "")
            cells.append(c.strip())
        out.append(cells)
    return out


def load(name: str):
    p = ROOT / "results" / name
    if not p.exists():
        raise SystemExit(f"missing {p} -- run the script that produces it first")
    return json.loads(p.read_text())


def close(a: float, b: float, tol: float = 5e-4) -> bool:
    return abs(a - b) <= tol


def main() -> int:
    bad: list[str] = []

    def check(where, printed, actual, tol=5e-4):
        if not close(float(printed), float(actual), tol):
            bad.append(f"{where}: table says {printed}, results say {actual:.4f}")

    # Table 1 -- per-exit dB below the released decoder, and bpp.
    # db_per_exit is indexed by EXIT NUMBER 0..K-1; exits 0 and 1 sit above the
    # j=2 split and are never routed to, so the table prints 2..5.
    wq = load("why_qp.json")
    by_qp = {r["qp"]: r for r in wq["rows"]}
    for row in rows_of("tab:exits"):
        qp = int(row[0])
        r = by_qp[qp]
        for k, printed in enumerate(row[1:5], start=2):
            check(f"tab:exits qp{qp} exit{k}", printed, r["db_per_exit"][k])
        check(f"tab:exits qp{qp} bpp", row[5], r["bpp"])

    # Table 2 -- adaptivity gain, as a percentage of J_ad.
    # The table prints Delta as a percentage of J_ad, the ADAPTIVE (oracle)
    # cost -- the scale-free form of Theorem 5. The JSON stores the two costs
    # and their difference, so the ratio is formed here rather than read off,
    # which is also what makes this a check and not a tautology.
    tc = load("theory_check.json")
    lam_cols = ["1e-05", "3e-05", "1e-04", "3e-04"]
    for row in rows_of("tab:delta"):
        qp = row[0]
        d = tc[qp]["delta"]
        for lam, printed in zip(lam_cols, row[1:5]):
            e = d.get(lam)
            if e is None:
                bad.append(f"tab:delta qp{qp} lambda={lam}: absent from results "
                           f"(have {sorted(d)})")
                continue
            check(f"tab:delta qp{qp} lam{lam}", printed,
                  100 * e["delta"] / e["J_oracle"], tol=5e-3)

    # Table 4 -- log-convexity test.
    lc = load("logconvexity.json")
    for row in rows_of("tab:logconvex"):
        qp = row[0]
        r = lc[qp]
        check(f"tab:logconvex qp{qp} n", row[1], r["n"], tol=0.5)
        check(f"tab:logconvex qp{qp} holds", row[2], r["frac_ok"],
              tol=5e-3)
        check(f"tab:logconvex qp{qp} min", row[3], r["worst"])

    # Table 3 is a linear interpolation ALONG the frontier rather than a direct
    # readout, so it is recomputed here from the same curve the table was drawn
    # from -- otherwise this script would only be checking that a number equals
    # itself.
    pc = load("paper_curve_grid128.json")["rows"]
    for row in rows_of("tab:marginal"):
        qp = int(row[0])
        pts = sorted((r["saving_pct"], r["db_vs_uf"]) for r in pc if r["qp"] == qp)

        def db_at(s):
            for (s0, d0), (s1, d1) in zip(pts, pts[1:]):
                if s0 <= s <= s1:
                    t = (s - s0) / (s1 - s0) if s1 > s0 else 0.0
                    return d0 + t * (d1 - d0)
            return None

        for lo, printed in zip((10, 15, 20, 25), row[1:5]):
            a, b = db_at(lo), db_at(lo + 5)
            if a is None or b is None:
                bad.append(f"tab:marginal qp{qp} {lo}->{lo+5}: frontier does not "
                           f"reach that saving")
                continue
            check(f"tab:marginal qp{qp} {lo}->{lo+5}", printed, b - a, tol=1e-3)

    if bad:
        print(f"{len(bad)} MISMATCH(ES):")
        for b in bad:
            print("  " + b)
        return 1
    print("theory.tex: every table entry matches the results it was drawn from")
    return 0


if __name__ == "__main__":
    sys.exit(main())
