"""What the page looks like, measured rather than read.

Every other check on the built PDF reads its text. A figure drawn over a
column, a page that came out blank, a table that overflowed into the gutter
-- none of those change the text layer, and all of them are obvious to
anyone who looks at the page. Nobody looks at twenty pages before every
push.

This renders the PDF and measures ink: coverage inside the printable box
per page, and coverage in the strip between the two columns. A page with
almost no ink is a page something went wrong on; a page with far too much
is a figure sitting on top of text; ink in the gutter is a full-width
figure, which is legitimate and rare, or an overflow, which is not.

The wide figure is named rather than tolerated by threshold, so a second
one appearing is a finding.

    python scripts/check_ink.py [paper/FLEX-UF.pdf]
"""
from __future__ import annotations

import re
import subprocess
import sys
import tempfile
from pathlib import Path

R = Path(__file__).resolve().parent.parent
DPI = 60
MARGIN_IN = 0.62
GUTTER_IN = 0.10
# Pages allowed ink between the columns, and why.
WIDE = {"paper/FLEX-UF.pdf": {5: "the patchify figure, which spans both columns"}}


def main(argv=None) -> int:
    rel = (argv or ["paper/FLEX-UF.pdf"])[0]
    pdf = R / rel
    if not pdf.exists():
        print(f"  no {rel}")
        return 1
    try:
        from PIL import Image
        import numpy as np
    except Exception as e:                       # pragma: no cover
        print(f"  cannot measure ink: {e}")
        return 0
    with tempfile.TemporaryDirectory() as td:
        subprocess.run(["pdftoppm", "-r", str(DPI), "-png", str(pdf),
                        f"{td}/pg"], check=True)
        files = sorted(Path(td).glob("pg-*.png"),
                       key=lambda f: int(re.search(r"pg-(\d+)", f.name).group(1)))
        m = int(MARGIN_IN * DPI)
        bad, cov_all = [], []
        allow = WIDE.get(rel, {})
        for f in files:
            n = int(re.search(r"pg-(\d+)", f.name).group(1))
            a = np.asarray(Image.open(f).convert("L"), dtype=np.float32) / 255.0
            ink = a < 0.55
            h, w = ink.shape
            cov = float(ink[m:h - m, m:w - m].mean())
            cov_all.append(cov)
            g0 = int(w / 2 - GUTTER_IN * DPI)
            g1 = int(w / 2 + GUTTER_IN * DPI)
            gut = float(ink[m:h - m, g0:g1].mean())
            if cov < 0.02:
                bad.append(f"page {n}: almost no ink ({cov:.3f})")
            elif cov > 0.42:
                bad.append(f"page {n}: ink over most of the box ({cov:.3f}), "
                           f"which is what a figure on top of text looks like")
            if gut > 0.06 and n not in allow:
                bad.append(f"page {n}: ink between the columns ({gut:.3f}) and "
                           f"no full-width figure is expected there")
        for b in bad:
            print(f"     {b}")
        print(f"\n  {len(files)} pages measured, coverage "
              f"{min(cov_all):.3f} to {max(cov_all):.3f}, "
              f"{len(bad)} page(s) worth looking at")
    return 1 if bad else 0


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
