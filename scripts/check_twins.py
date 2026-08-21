"""paper/main.tex and scripts/build_pdf.py carry the same paper. Prove it.

They have drifted twice in one night: nine figures existed only in the reportlab
build and two only in the LaTeX source, and a whole section was added to one and
not the other. Neither build complains, because each is internally consistent.
This is the check that is not.

It compares structure rather than wording, because the two files legitimately
differ in markup: \\emph against <i>, \\cite{key} against [n], macros expanded at
build time in one and by bibtex in the other.

    python scripts/check_twins.py
"""

from __future__ import annotations

import re
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
TEX = (ROOT / "paper/main.tex").read_text()
PY = (ROOT / "scripts/build_pdf.py").read_text()


def norm(t):
    return re.sub(r"[^a-z0-9 ]", "", t.lower()).strip()


def main():
    bad = 0

    tex_sec = [norm(m) for m in re.findall(r"\\section\{([^}]+)\}", TEX)]
    py_sec = [norm(re.sub(r"^\d+\.\s*", "", m))
              for m in re.findall(r'h1\("([^"]+)"\)', PY)]
    py_sec = [s for s in py_sec if s != "references"]
    print(f"  sections: tex {len(tex_sec)}, py {len(py_sec)}")
    for a, b in ((tex_sec, py_sec), (py_sec, tex_sec)):
        for s in a:
            if s not in b:
                print(f"     only in one: {s}")
                bad += 1

    tex_sub = [norm(m) for m in re.findall(r"\\subsection\{([^}]+)\}", TEX)]
    py_sub = [norm(re.sub(r"^\d+\.\d+\s*", "", m))
              for m in re.findall(r'h2\("([^"]+)"\)', PY)]
    only_tex = [s for s in tex_sub if s not in py_sub]
    only_py = [s for s in py_sub if s not in tex_sub]
    print(f"  subsections: tex {len(tex_sub)}, py {len(py_sub)}")
    for s in only_tex:
        print(f"     only in main.tex: {s}")
    # build_pdf uses h2 for paragraph headings too, which main.tex writes as
    # \paragraph; those are not a divergence.
    par_head = [norm(m) for m in re.findall(r"\\paragraph\{([^}]+)\}", TEX)]
    unmatched = [s for s in only_py if s not in par_head]
    for s in unmatched:
        print(f"     only in build_pdf: {s}")
    bad += len(only_tex) + len(unmatched)

    tex_fig = sorted(re.findall(
        r"includegraphics\[[^\]]*\]\{figures/([a-z_0-9]+)\.png\}", TEX))
    py_fig = sorted(re.findall(r'figure(?:_wide)?\("([a-z_0-9]+)\.png"', PY))
    # the banner figure is emitted by fig() directly, not through figure()
    banner = re.findall(r'fig\("([a-z_0-9]+)\.png", PW', PY)
    py_fig += banner
    print(f"  figures: tex {len(tex_fig)}, py {len(py_fig)}")
    for f in set(tex_fig) - set(py_fig):
        print(f"     only in main.tex: {f}")
        bad += 1
    for f in set(py_fig) - set(tex_fig):
        print(f"     only in build_pdf: {f}")
        bad += 1

    tex_tab = sorted(re.findall(r"\\input\{tables/([a-z_0-9]+)\}", TEX))
    py_tab = sorted(re.findall(r'tbl\("([a-z_0-9]+)"', PY))
    tex_tab = [t for t in tex_tab if t not in ("macros",)]
    print(f"  tables: tex {len(tex_tab)}, py {len(py_tab)}")
    for t in set(tex_tab) ^ set(py_tab):
        print(f"     only in one: {t}")
        bad += 1

    # A duplicated subsection number in the reportlab build points every
    # "Section 5.9" in the prose at two different places. LaTeX numbers its own
    # subsections so it cannot happen there, which is exactly why the reportlab
    # side needs checking.
    import collections as _c
    _n = [h for h in re.findall(r'h2\(\s*r?"([^"]+)"', PY)
          if re.match(r"\d+\.\d+ ", h)]
    _dup = [k for k, v in _c.Counter(x.split()[0] for x in _n).items() if v > 1]
    if _dup:
        bad += len(_dup)
        for d in _dup:
            print(f"     duplicate subsection number {d}")
    print(f"  subsection numbers: {len(_n)}, {len(_dup)} duplicated")

    print(f"\n  {'PASS' if not bad else str(bad) + ' divergence(s)'}")
    return 0 if not bad else 1


if __name__ == "__main__":
    raise SystemExit(main())
