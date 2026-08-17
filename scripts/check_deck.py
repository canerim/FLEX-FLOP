"""Render the deck and check every slide still carries a figure.

The requirement is explicit -- a figure on every page -- and it is the kind that
breaks silently. A figure goes missing when its source JSON is renamed, when a
filename in make_deck.py stops matching a file, or when bullets grow until the
image is squeezed to nothing. None of those raise; the build reports 12 slides
either way.

Rendering is the only check that sees what a reader sees. Three faults today
were invisible until the deck was rendered and looked at: a slide whose text and
its own figure disagreed by 1.5 points, a figure sized to 60% of the width
available to it, and a panel plotting a stale measurement beside a fresh bullet.

The test is crude on purpose. It measures ink in the lower half of each slide,
where slide_fig puts the figure, and fails a slide with almost none. A wrong
figure still passes -- that needs eyes -- but a missing or collapsed one cannot.

    python scripts/check_deck.py            # exits non-zero if a slide is bare
"""

from __future__ import annotations

import glob
import shutil
import subprocess
import sys
import tempfile
from pathlib import Path

import numpy as np
from PIL import Image

ROOT = Path(__file__).resolve().parent.parent
DECK = ROOT / "paper" / "FLEX-UF_LMT.pptx"
MIN_INK = 0.5     # percent of dark pixels in the lower half


def main() -> int:
    if not DECK.exists():
        print(f"  {DECK} not built")
        return 1
    if not shutil.which("libreoffice"):
        print("  libreoffice not available; cannot render, so cannot check")
        return 1

    with tempfile.TemporaryDirectory() as td:
        r = subprocess.run(["libreoffice", "--headless", "--convert-to", "pdf",
                            "--outdir", td, str(DECK)],
                           capture_output=True, timeout=600)
        pdf = Path(td) / (DECK.stem + ".pdf")
        if not pdf.exists():
            print(f"  render failed: {r.stderr.decode()[:200]}")
            return 1
        subprocess.run(["pdftoppm", "-png", "-r", "50", str(pdf),
                        str(Path(td) / "s")], check=True)
        pages = sorted(glob.glob(str(Path(td) / "s-*.png")))
        if not pages:
            print("  no pages rendered")
            return 1

        bare = []
        print(f"  {len(pages)} slides")
        for i, f in enumerate(pages, start=1):
            im = np.asarray(Image.open(f).convert("L"))
            low = im[int(im.shape[0] * 0.55):, :]
            ink = float((low < 200).mean() * 100)
            # The title slide has no figure by design and is exempt.
            flag = ""
            if i > 1 and ink < MIN_INK:
                bare.append(i)
                flag = "   NO FIGURE"
            print(f"    slide {i:>2}: lower-half ink {ink:5.2f}%{flag}")

        if bare:
            print(f"\n  slides {bare} have no figure. The requirement is one on "
                  f"every page; check make_deck.py's filename for those slides "
                  f"and whether results/ still holds what they read.")
            return 1
    print("\n  every slide carries a figure. Note this cannot tell a WRONG "
          "figure from a right one -- that still needs looking.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
