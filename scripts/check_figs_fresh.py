"""Is every figure in the paper newer than the data it was drawn from?

A figure is a cached artefact. When a measurement is re-run, the PNG does not
follow, and nothing in the build says so: the caption still renders, the
numbering is still continuous, the picture is still a picture. Four figures in
this paper were found showing a 41.9% ceiling that the FFN accounting had
corrected to 38.3 days earlier, and one drew a router that was not the paper's.

For each figure this finds the script that writes it, reads the result files
that script opens, and compares mtimes. It is a coarse test -- a script may not
use every file it mentions -- so it reports rather than fails by default.

    python scripts/check_figs_fresh.py [--fail]
"""

from __future__ import annotations

import os
import re
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
SKIP = {"build_pdf.py", "build_supp_pdf.py", "make_deck.py", "make_report.py",
        "check_figs_fresh.py",
        "paper_figures.py", "make_slide_figs.py", "make_slide_figs2.py",
        "make_figs_nature.py"}


def producers():
    """figure name -> the script that saves it, preferring one that writes here."""
    out = {}
    for f in sorted(ROOT.glob("scripts/*.py")):
        if f.name in SKIP:
            continue
        t = f.read_text(errors="ignore")
        if "savefig" not in t and "save(fig" not in t:
            continue
        # Names appear both bare and inside a path string, and the path form
        # is the common one: "docs/figures/power.png". Matching only the bare
        # form found no producer for seventeen of the thirty-five figures.
        # A producer both names the figure and writes it. Matching any mention
        # of "name.png" made this file its own producer, via its docstring.
        names = set(re.findall(r'figures/([A-Za-z_0-9]+)\.png', t))
        names |= set(re.findall(r'save\(\s*fig,\s*"([A-Za-z_0-9]+)\.png"', t))
        names |= set(re.findall(r'savefig\([^)]*?"([A-Za-z_0-9]+)\.png"', t))
        # f-string filenames: f"saturation_{TAG}.png" and the --out default of
        # a script that takes one. Two figures had no producer for this reason.
        # f-string filenames, with or without a directory in the same literal:
        # f"saturation_{TAG}.png" is written as _d / f"saturation_{TAG}.png".
        # The braces come out and what is left is the stem the file will have.
        for m in re.findall(r'f"([A-Za-z_0-9{}./]*?)\.png"', t):
            stem = re.sub(r"\{[^{}]*\}", "", m).strip("/").split("/")[-1]
            if stem:
                names.add(stem)
        names |= set(re.findall(r'out="[^"]*?figures/([A-Za-z_0-9]+)\.png"', t))
        writes_here = "paper/figures" in t or "paper\" / \"figures" in t
        for n in names:
            if n not in out or (writes_here and not out[n][1]):
                out[n] = (f, writes_here)
    return {k: v[0] for k, v in out.items()}


def supplement_figures():
    """The names the supplement asks for, which build_supp_pdf resolves from
    docs/figures first and paper/figures second."""
    import glob
    out = set()
    for f in sorted(glob.glob(str(ROOT / "scripts/supp/[a-z]_*.py"))):
        out |= set(re.findall(r'k\.fig\(\s*"([A-Za-z_0-9]+)\.png"',
                              Path(f).read_text(errors="ignore")))
    return sorted(out)


def twins() -> list[str]:
    """The same figure name in docs/figures and paper/figures must be the
    same figure.

    The paper reads paper/figures; the supplement prefers docs/figures. Twelve
    producers wrote only one of the two, so the paper printed a tiles_unequal
    that differed from the one its own script produces, and nothing complained:
    the copy it printed was newer than the script, so this file passed it.
    naturestyle mirrors every write now, and this is what proves it.
    """
    bad = []
    a, b = ROOT / "docs/figures", ROOT / "paper/figures"
    for f in sorted(b.glob("*.png")):
        g = a / f.name
        if g.exists() and g.read_bytes() != f.read_bytes():
            bad.append(f.name)
    return bad


def main(argv):
    fail = "--fail" in argv
    paper = sorted(set(re.findall(
        r'figure(?:_wide)?\(\s*"([A-Za-z_0-9]+)\.png"',
        (ROOT / "scripts/build_pdf.py").read_text())))
    prod = producers()
    stale, unknown = [], []
    for n in paper:
        png = ROOT / "paper/figures" / f"{n}.png"
        if not png.exists():
            continue
        src = prod.get(n)
        if src is None:
            # A stem from an f-string ("saturation_{TAG}") or a --out default
            # ("qualitative", which also writes qualitative_q63): match the
            # longest producer stem that this figure's name starts with.
            cands = [k for k in prod if n.startswith(k) and k]
            src = prod[max(cands, key=len)] if cands else None
        if src is None:
            unknown.append(n)
            continue
        deps = [ROOT / "results" / d
                for d in set(re.findall(r'"([A-Za-z_0-9./]+\.json)"',
                                        src.read_text(errors="ignore")))]
        deps = [d for d in deps if d.exists()]
        # A script that dumps its own statistics writes them after the PNG,
        # so its own output is always newer and always a false positive.
        written = set(re.findall(
            r'json\.dump\([^)]*open\([^"\']*["\'][^"\']*?([A-Za-z_0-9]+\.json)',
            src.read_text(errors="ignore")))
        written |= set(re.findall(
            r'open\([^)]*?([A-Za-z_0-9]+\.json)[^)]*?["\']w["\']',
            src.read_text(errors="ignore")))
        deps = [d for d in deps if d.name not in written]
        newer = [d for d in deps if d.stat().st_mtime > png.stat().st_mtime]
        if newer:
            stale.append((n, src.name, newer))
    # The supplement's figures too: they are built by the same scripts and go
    # stale the same way, and two of them had not been regenerated since their
    # producer changed.
    for n in supplement_figures():
        if n in paper:
            continue
        png = None
        for d in ("docs/figures", "paper/figures"):
            if (ROOT / d / f"{n}.png").exists():
                png = ROOT / d / f"{n}.png"
                break
        src = prod.get(n)
        if png is None or src is None:
            continue
        deps = [ROOT / "results" / d
                for d in set(re.findall(r'"([A-Za-z_0-9./]+\.json)"',
                                        src.read_text(errors="ignore")))]
        deps = [d for d in deps if d.exists()]
        if src.stat().st_mtime > png.stat().st_mtime:
            deps.append(src)
        if [d for d in deps if d.stat().st_mtime > png.stat().st_mtime]:
            stale.append((n + " (supp)", src.name,
                          [d for d in deps
                           if d.stat().st_mtime > png.stat().st_mtime]))

    print(f"  {len(paper)} paper figures and "
          f"{len(supplement_figures())} supplement figures, "
          f"{len(prod)} producers found")
    for n, s, newer in stale:
        when = time.strftime("%m-%d %H:%M",
                             time.localtime(max(d.stat().st_mtime for d in newer)))
        print(f"     STALE {n} ({s}): {len(newer)} input(s) newer, "
              f"latest {when} -- {newer[0].name}")
    if unknown:
        print(f"     no producer identified: {', '.join(unknown)}")
    tw = twins()
    for n in tw:
        print(f"     {n} differs between docs/figures and paper/figures")
    bad = (stale and fail) or tw
    if not bad:
        note = "PASS"
    else:
        note = ", ".join(
            x for x in (f"{len(stale)} stale figure(s)" if stale else "",
                        f"{len(tw)} divergent copy/copies" if tw else "") if x)
    print(f"\n  {note}")
    return 1 if bad else 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))
