"""Nothing in a figure may sit on top of anything else, and nothing may be
drawn where it cannot be seen.

The measurement is in naturestyle: every savefig audits the figure it just
wrote and records the result in results/figure_audit.json. This reads that
file, insists that the figures the paper and the supplement actually use are
in it, and fails on any that are not clean.

A figure that has never been regenerated has no entry, which is not the same
as passing, so the coverage is reported separately from the failures.

    python scripts/check_fig_overlap.py
"""
from __future__ import annotations

import json
import re
import sys
from pathlib import Path

R = Path(__file__).resolve().parent.parent
AUDIT = R / "results/figure_audit.json"


def used() -> set[str]:
    """Every .png named by the paper or the supplement builders."""
    out = set()
    for f in [R / "scripts/build_pdf.py", *sorted((R / "scripts/supp").glob("[a-z]_*.py"))]:
        out |= set(re.findall(r'"([A-Za-z0-9_.-]+\.png)"', f.read_text()))
    return out


def main() -> int:
    if not AUDIT.exists():
        print("  no figure audit yet: regenerate the figures once")
        return 1
    db = json.loads(AUDIT.read_text())
    names = used()
    bad = [(n, d) for n, d in sorted(db.items())
           if n in names and (d.get("overlaps") or d.get("outside"))]
    seen = names & set(db)
    for n, d in bad:
        for o in d.get("overlaps", []):
            print(f"     {n}: \"{o['a']}\" over \"{o['b']}\" ({o['frac']:.0%})")
        for o in d.get("outside", []):
            print(f"     {n}: \"{o['label']}\" entirely outside the axes")
    # Naming the gap rather than reporting a clean sweep over part of the set:
    # the figures drawn from the model need a GPU and a checkpoint, and
    # regenerating one to audit it would replace a published picture.
    missing = sorted(names - seen)
    print(f"  {len(seen)}/{len(names)} figures audited, {len(bad)} with "
          f"collisions or invisible series")
    if missing:
        print(f"     not yet audited ({len(missing)}): "
              f"{', '.join(m[:-4] for m in missing[:6])}"
              f"{' ...' if len(missing) > 6 else ''}")
    return 1 if bad else 0


if __name__ == "__main__":
    sys.exit(main())
