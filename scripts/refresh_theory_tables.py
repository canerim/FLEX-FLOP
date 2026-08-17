"""Rewrite `paper/theory.tex`'s table bodies from the results, and test the prose.

Why this is not just a convenience
----------------------------------
The CTC test set grew from 10 sequences to 40 (MCL-JCV), and `why_qp.py` and
`theory_check.py` both measure on it, so Tables 1 and 2 moved. Retyping 62 numbers by hand is how a transcription error gets into a
paper.

But a writer that only refreshed the numbers would be *worse* than retyping,
because the captions and the surrounding prose make claims ABOUT those numbers:

    Table 1  "grows 4.3x from the lowest to the highest rate"
    Table 2  "rises monotonically with rate"
    Table 4  "holds everywhere, with a margin that grows tenfold"
    body     "would understate its own contribution by a factor of seven"

The 10-to-40 sequence move already exercised this: it moved 3.4x to 4.3x and
six to seven, and both were caught here rather than shipped.

Silently updating the numbers under a claim that no longer holds is the exact
failure this is meant to prevent. So every such claim is re-derived from the
new data and reported; the numbers are written, and any claim that has stopped
being true is printed as a WARNING for a human to rewrite. Nothing in the prose
is edited automatically -- a sentence is an argument, not a cell.

    python scripts/refresh_theory_tables.py --check   # report only, write nothing
    python scripts/refresh_theory_tables.py           # rewrite the table bodies
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
TEX = ROOT / "paper" / "theory.tex"


def load(name):
    p = ROOT / "results" / name
    if not p.exists():
        raise SystemExit(f"missing {p}")
    return json.loads(p.read_text())


def splice(src: str, label: str, body: str) -> str:
    """Replace the rows between \\midrule and \\bottomrule of the table at `label`."""
    i = src.find(f"\\label{{{label}}}")
    if i < 0:
        raise SystemExit(f"no table labelled {label}")
    a = src.find("\\midrule", i) + len("\\midrule")
    b = src.find("\\bottomrule", a)
    return src[:a] + "\n" + body + "\n" + src[b:]


def main(argv):
    ap = argparse.ArgumentParser()
    ap.add_argument("--curve", default="paper_curve_grid128.json",
                    help="frontier feeding Table 3; must be the same checkpoint "
                         "as why_qp.json and theory_check.json")
    ap.add_argument("--check", action="store_true",
                    help="report claim status without touching the file")
    a = ap.parse_args(argv)

    src = TEX.read_text()
    warn: list[str] = []
    note: list[str] = []

    # ---- Table 1: per-exit dB, and the "3.4x" claim ------------------------
    wq = {r["qp"]: r for r in load("why_qp.json")["rows"]}
    rows = []
    for qp in (0, 32, 63):
        r = wq[qp]
        cells = " & ".join(f"{r['db_per_exit'][k]:.4f}" for k in (2, 3, 4, 5))
        rows.append(f"{qp:<2} & {cells} & {r['bpp']:.4f} \\\\")
    src = splice(src, "tab:exits", "\n".join(rows))

    ratio = wq[63]["db_per_exit"][2] / wq[0]["db_per_exit"][2]
    note.append(f"Table 1: exit-2 penalty grows {ratio:.1f}x from qp0 to qp63")
    if abs(ratio - 4.3) > 0.15:
        warn.append(f"caption says 'grows 4.3x'; measured {ratio:.2f}x")

    # ---- Table 2: adaptivity gain, monotonicity, and "a factor of six" -----
    tc = load("theory_check.json")
    lams = ["1e-05", "3e-05", "1e-04", "3e-04"]
    qps = [0, 16, 32, 48, 63]
    tbl, peak = [], []
    for qp in qps:
        d = tc[str(qp)]["delta"]
        vals = [100 * d[l]["delta"] / d[l]["J_oracle"] for l in lams]
        peak.append(vals[1])
        cells = [f"{v:.2f}\\%" for v in vals]
        if qp == qps[-1]:
            cells[1] = f"\\textbf{{{vals[1]:.2f}\\%}}"
        tbl.append(f"{qp:<2} & " + " & ".join(cells) + " \\\\")
    src = splice(src, "tab:delta", "\n".join(tbl))

    mono = all(x < y for x, y in zip(peak, peak[1:]))
    note.append(f"Table 2: Delta/J at lambda=3e-5 runs "
                f"{' -> '.join(f'{v:.2f}%' for v in peak)}")
    if not mono:
        warn.append("caption says Delta/J 'rises monotonically with rate'; it "
                    "no longer does")
    factor = peak[-1] / peak[0] if peak[0] else float("inf")
    note.append(f"body: low-rate understatement factor {factor:.1f}x")
    if abs(factor - 7) > 1.0:
        warn.append(f"body says 'a factor of seven'; measured {factor:.1f}x")

    # ---- Table 4: log-convexity, "everywhere", "margin grows with rate" ----
    lc = load("logconvexity.json")
    rows, worst = [], []
    for qp in qps:
        r = lc[str(qp)]
        worst.append(r["worst"])
        rows.append(f"{qp:<2} & {r['n']} & {r['frac_ok']:.1f}\\% & {r['worst']:.4f} \\\\")
    src = splice(src, "tab:logconvex", "\n".join(rows))

    if any(lc[str(q)]["frac_ok"] < 100 for q in qps):
        warn.append("caption says the condition 'holds everywhere'; it now fails "
                    "on part of the frontier -- Proposition on log-convexity is "
                    "an EQUIVALENCE, so this does not break the theory, but the "
                    "claim that dB(S) is convex HERE would no longer follow")
    if not all(x < y for x, y in zip(worst, worst[1:])):
        warn.append("caption says the margin 'grows with rate'; it no longer "
                    "increases monotonically")
    if worst[-1] / worst[0] < 8 or worst[-1] / worst[0] > 12:
        warn.append(f"caption says the margin grows 'tenfold'; measured "
                    f"{worst[-1] / worst[0]:.1f}x")

    # ---- Table 3: marginal dB per +5 points, interpolated ------------------
    pc = load(a.curve)["rows"]
    rows = []
    for qp in (0, 32, 63):
        pts = sorted((r["saving_pct"], r["db_vs_uf"]) for r in pc if r["qp"] == qp)

        def db_at(s):
            for (s0, d0), (s1, d1) in zip(pts, pts[1:]):
                if s0 <= s <= s1:
                    t = (s - s0) / (s1 - s0) if s1 > s0 else 0.0
                    return d0 + t * (d1 - d0)
            return None

        marg = []
        for lo in (10, 15, 20, 25):
            x, y = db_at(lo), db_at(lo + 5)
            marg.append(None if x is None or y is None else y - x)
        if any(m is None for m in marg):
            warn.append(f"Table 3 qp{qp}: the frontier no longer reaches 30% "
                        f"saving; those columns cannot be filled")
            continue
        if not all(u < v for u, v in zip(marg, marg[1:])):
            warn.append(f"caption says each further point 'costs more than the "
                        f"last'; at qp{qp} the marginal cost is not increasing")
        rows.append(f"{qp:<2} & " + " & ".join(f"{m:.4f}" for m in marg) + " \\\\")
    if len(rows) == 3:
        src = splice(src, "tab:marginal", "\n".join(rows))

    for n in note:
        print("  " + n)
    if warn:
        print(f"\n{len(warn)} CLAIM(S) IN THE PROSE NO LONGER HOLD -- rewrite by hand:")
        for w in warn:
            print("  ! " + w)

    if a.check:
        print("\n--check: nothing written")
    else:
        TEX.write_text(src)
        print(f"\nwrote {TEX.relative_to(ROOT)}")
    return 1 if warn else 0


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
