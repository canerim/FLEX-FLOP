"""A figure should be drawn at the size it is printed at.

matplotlib sets type in points on the figure canvas. Placing a figure drawn
7.2 inches wide into a 3.49 inch column divides every label by 2.06, so an
8 pt axis label prints at 3.9 and a 6 pt annotation at 2.9. Nothing in the
build complains: the picture is there, the caption renders, the numbers are
right, and the text is unreadable.

This compares each figure's drawn width -- its pixel width over the 500 dpi
every producer here saves at -- against the width the document gives it, and
reports the ratio. A figure drawn at naturestyle's W1 lands at 1.00; one
drawn at W2 and placed in a column lands at 2.06.

    python scripts/check_fig_scale.py
"""
from __future__ import annotations

import re
import sys
from pathlib import Path

from PIL import Image

R = Path(__file__).resolve().parent.parent
DPI = 500
# build_pdf.build(): M, GAP = 0.62 * inch, 0.28 * inch on letter.
PAGE_W, MARGIN, GAP = 8.5, 0.62, 0.28
COL_W = (PAGE_W - 2 * MARGIN - GAP) / 2
FULL_W = PAGE_W - 2 * MARGIN
# 1.35 is where a 8 pt label drops below 6 pt, naturestyle's stated floor.
TOL = 1.35


def main() -> int:
    bp = (R / "scripts/build_pdf.py").read_text()
    col = set(re.findall(r'(?<!_wide)\bfigure\(\s*"([^"]+\.png)"', bp))
    wide = set(re.findall(r'figure_wide\(\s*"([^"]+\.png)"', bp))

    # A producer that sets its type for the column has already multiplied
    # every size by naturestyle.MAX_COLUMN_SCALE, so the geometric shrink is
    # not the shrink the reader sees. Without this the check stayed red after
    # the thing it was written to catch had been fixed.
    import sys as _sys
    _sys.path.insert(0, str(R / "scripts"))
    import naturestyle as _ns
    scaled = set()
    _fstem = []
    for prod in sorted(list((R / "scripts").glob("*.py"))
                       + list((R / "scripts/supp").glob("*.py"))):
        t = prod.read_text()
        if "for_column()" not in t:
            continue
        # The name often appears as a default path, "docs/figures/x.png",
        # so match the basename wherever it occurs rather than a bare literal.
        scaled |= {m.rsplit("/", 1)[-1]
                   for m in re.findall(r'["\']([A-Za-z0-9_./\-]+\.png)["\']', t)}
        # Some producers name their output with an f-string --
        # f"saturation_{TAG}.png" -- which the literal match above cannot see,
        # so a figure that does set its type for the column was reported as if
        # it did not. Turn the braces into a wildcard and match on that.
        for pat in re.findall(r'f["\']([A-Za-z0-9_./\-]*\{[^"\']*\.png)["\']', t):
            rx = re.compile("^" + re.sub(r"\{[^}]*\}", r"[A-Za-z0-9_.\\-]+",
                                         re.escape(pat).replace(r"\{", "{")
                                         .replace(r"\}", "}")
                                         .rsplit("/", 1)[-1]) + "$")
            _fstem.append(rx)

    import json as _json
    _apath = R / "results/figure_audit.json"
    try:
        _audit = _json.loads(_apath.read_text()) if _apath.exists() else {}
    except Exception:
        _audit = {}

    rows = []
    for names, avail, where in ((col, COL_W, "column"),
                                (wide, FULL_W, "full width")):
        for n in sorted(names):
            p = R / "paper/figures" / n
            if not p.exists():
                continue
            w, h = Image.open(p).size
            drawn = w / DPI
            # The figure itself records what its type was multiplied by, as
            # of the last time it was drawn. Fall back to reading the
            # producer's source for figures drawn before that was recorded.
            rec = _audit.get(n) or {}
            if "type_scale" in rec:
                type_scale = rec["type_scale"] or 1.0
            else:
                _is = n in scaled or any(rx.match(n) for rx in _fstem)
                type_scale = _ns.MAX_COLUMN_SCALE if _is else 1.0
            rows.append(((drawn / avail) / type_scale, n, drawn, avail,
                         where, type_scale))

    bad = sorted([r for r in rows if r[0] > TOL], reverse=True)
    for ratio, n, drawn, avail, where, ts in bad:
        note = f", type set x{ts:g}" if ts > 1 else ""
        print(f"     {n:<32} drawn {drawn:.2f}in, {where} gives "
              f"{avail:.2f}in{note} -> text at 1/{ratio:.2f}")
    ok = len(rows) - len(bad)
    print(f"  {ok}/{len(rows)} figures print their type at full size or "
          f"close, {len(bad)} still shrink by more than {TOL:g}x")
    return 1 if bad else 0


if __name__ == "__main__":
    sys.exit(main())
