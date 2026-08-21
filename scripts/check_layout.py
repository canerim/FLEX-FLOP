"""Every column should reach the bottom of the page.

reportlab has no float mechanism. A figure is one KeepTogether, and when it does
not fit in the space left in a column it moves to the next one and leaves the
rest of that column empty. Nothing else in this repo notices: the claims are
still right, the twins still agree, the figure is still present and correctly
numbered. A reader notices immediately -- the author found page 15's right column
ending at 31% of the page by looking at it.

Measured on the artefact, column by column, with the footer page number excluded.
That exclusion is the whole trick: the page number sits at the bottom centre, so
a naive "lowest text on the page" reads 0.96 for a page that is otherwise blank.

The last two columns are exempt: they are the end of the references, and a
document has to stop somewhere.

    python scripts/check_layout.py [pdf]
"""

from __future__ import annotations

import re
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
FULL = 0.88          # a column that reaches this is not worth complaining about
WORD = re.compile(r'<word xMin="([\d.]+)" yMin="([\d.]+)" '
                  r'xMax="([\d.]+)" yMax="([\d.]+)">')


def columns(pdf):
    out = subprocess.run(["pdftotext", "-bbox", str(pdf), "-"],
                         capture_output=True, text=True).stdout
    pages = re.findall(r'<page width="([\d.]+)" height="([\d.]+)">(.*?)</page>',
                       out, re.S)
    cols = []
    for n, (w, h, body) in enumerate(pages, 1):
        W, H = float(w), float(h)
        ws = [tuple(map(float, t)) for t in WORD.findall(body)]
        ws = [t for t in ws
              if not (abs((t[0] + t[2]) / 2 - W / 2) < 0.06 * W
                      and t[3] > 0.94 * H)]
        for tag in ("L", "R"):
            v = [t[3] for t in ws
                 if ((t[0] + t[2]) / 2 < W / 2) == (tag == "L")]
            cols.append((n, tag, (max(v) / H if v else 0.0)))
    return cols


def main(argv):
    pdf = ROOT / (argv[0] if argv else "paper/FLEX-UF.pdf")
    cols = columns(pdf)
    body = cols[:-2]
    short = [c for c in body if c[2] < FULL]
    waste = sum(max(0.0, 0.93 - c[2]) for c in body)
    print(f"  {len(cols)} columns, {len(cols) // 2} pages")
    print(f"  total empty column space {waste:.2f} of a column")
    for n, tag, f in short:
        print(f"     p{n}{tag} ends at {f:.2f}")
    print(f"  {len(body) - len(short)}/{len(body)} columns reach {FULL:.2f}")
    print(f"\n  {'PASS' if not short else str(len(short)) + ' short column(s)'}")
    return 0 if not short else 1


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))
