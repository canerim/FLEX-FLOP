"""What only the built PDF can show.

Every other check in this repo reads the source. These read the artefact,
because the defects they catch are invisible before rendering:

  - a macro that did not expand, or expanded with its backslash still attached
    ("that network costs \\454 GMAC" shipped in one build)
  - a cross-reference token that was never resolved
  - a caption still saying "Figure N."
  - a brace or a math delimiter that leaked out of a macro value
  - a word repeated across a line break

The doubled-word test has a known false positive: pdftotext reads a two-column
page in one stream, so the last word of one column can abut the first word of
the next, and a table's columns do the same. Those are reported separately.

    python scripts/check_render.py [pdf ...]
"""

from __future__ import annotations

import re
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
DEFAULT = ["paper/FLEX-UF.pdf", "paper/FLEX-UF-supp.pdf"]


def text(pdf):
    return subprocess.run(["pdftotext", str(pdf), "-"],
                          capture_output=True, text=True).stdout


def main(argv):
    bad = 0
    for name in (argv or DEFAULT):
        p = ROOT / name
        if not p.exists():
            print(f"  {name}: missing")
            bad += 1
            continue
        t = text(p)
        flat = re.sub(r"\s+", " ", t)
        checks = [
            ("unexpanded macro", re.findall(r"\\[A-Za-z]{3,}", t)),
            ("stray backslash", re.findall(r"\\\d", t)),
            ("unresolved reference token", re.findall(r"\[\[[a-z]+:", t)),
            # "Table N." reached a built PDF: the protocol table is written
            # with rows_tbl and a hand-made caption, so it never went through
            # the numbering that fig() and tbl() apply.
            ("caption still says Figure/Table/Section N",
             re.findall(r"(?:Figure|Table|Section|Eq\.) N\b", t)),
            ("brace from a macro value", re.findall(r"[{}]", t)),
            ("math delimiter", re.findall(r"\$", t)),
        ]
        print(f"  {name}")
        for label, hits in checks:
            if hits:
                bad += 1
                print(f"     {label}: {len(hits)}  e.g. {hits[0]!r}")
        # doubled words, minus the column-abutment false positives
        dbl = [m.group(0) for m in re.finditer(r"\b([A-Za-z]{3,})\s+\1\b",
                                               flat, re.I)]
        real = [d for d in dbl if d.split()[0] == d.split()[1]]
        if real:
            print(f"     repeated word (check for column abutment): "
                  f"{', '.join(sorted(set(real)))}")
        if not any(h for _, h in checks):
            print("     clean")
    print(f"\n  {'PASS' if not bad else str(bad) + ' item(s)'}")
    return 0 if not bad else 1


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))
