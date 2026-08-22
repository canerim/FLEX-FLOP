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
    # The teaser is passed to fig() through a variable, so it is not in the
    # literal scan; it is whichever of the two banner files is present.
    for _b in ("flexuf_overview", "dcvcuf_framework"):
        if (ROOT / "paper/figures" / f"{_b}.png").exists():
            py_fig.append(_b)
            break
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

    # A macro used by one build and not the other is how the two carried
    # different numbers while passing every check here: build_pdf said
    # \\BlendGainMid and main.tex still had "+2.1" typed into the sentence.
    # Comparing the sets is enough -- if both name the macro, both print
    # whatever it currently holds.
    defined = set(re.findall(r"\\newcommand\{\\([A-Za-z]+)\}",
                             (ROOT / "paper/tables/macros.tex").read_text()))
    used_tex = set(re.findall(r"\\([A-Za-z]+)", TEX)) & defined
    used_py = set(re.findall(r"\\\\?([A-Za-z]+)", PY)) & defined
    # One quantity, two renderings: main.tex sets three of these in a table,
    # in maths, and build_pdf sets the same three in a sentence, in text.
    # \XMath in one document and \X in the other is one number in two
    # typesettings, not a divergence.
    def _paired(name, other):
        return (name.endswith("Math") and name[:-4] in other) or \
               (name + "Math") in other
    only_t = sorted(m for m in used_tex - used_py if not _paired(m, used_py))
    only_p = sorted(m for m in used_py - used_tex if not _paired(m, used_tex))
    print(f"  macros: tex {len(used_tex)}, py {len(used_py)}")
    for m_ in only_t:
        print(f"     only main.tex uses \\{m_}")
    for m_ in only_p:
        print(f"     only build_pdf uses \\{m_}")
    bad += len(only_t) + len(only_p)

    # And the prose itself, at paragraph granularity. Three paragraphs lived
    # only in the reportlab build for days -- the opening of Section 4, the
    # reason the reporting conventions are stated at all, and the confound in
    # the tile-size comparison, which is the one that matters. Matching is by
    # three forty-character probes taken from across each paragraph, because a
    # single prefix match reports every paragraph whose first line wraps
    # differently or opens with a macro.
    def _n(t):
        t = re.sub(r"\[\[[a-z]+:[a-z0-9_]+\]\]", " ", t)
        t = re.sub(r"\\\\?[A-Za-z]+\*?", " ", t)
        t = re.sub(r"<[^>]+>", " ", t)
        return re.sub(r"[^a-z]", "", t.lower())

    pars = []
    for m_ in re.finditer(r"\n    par\(", PY):
        seg, d, e = PY[m_.end():m_.end() + 3500], 1, None
        for i_, c_ in enumerate(seg):
            if c_ == "(":
                d += 1
            elif c_ == ")":
                d -= 1
                if d == 0:
                    e = i_
                    break
        t = "".join(x.group(1) for x in
                    re.finditer(r'r?f?"((?:[^"\\]|\\.)*)"', seg[:e]))
        if len(_n(t)) > 120:
            pars.append(t)
    T = _n(re.sub(r"%.*", "", TEX))
    missing = []
    for t in pars:
        n_ = _n(t)
        # Six probes of thirty characters, spread across the paragraph. Three
        # of forty reported the released-decoder paragraph as missing because
        # main.tex carries one extra \ref in the middle of it, which moves
        # every later probe.
        step = max(1, (len(n_) - 30) // 5)
        probes = [n_[i_:i_ + 30] for i_ in range(0, len(n_) - 30, step)][:6]
        if not any(pr in T for pr in probes if len(pr) == 30):
            missing.append(re.sub(r"\s+", " ", re.sub(r"<[^>]+>", "", t))[:70])
    print(f"  paragraphs: py {len(pars)}, {len(missing)} not in main.tex")
    for t in missing:
        print(f"     only in build_pdf: {t}")
    bad += len(missing)

    print(f"\n  {'PASS' if not bad else str(bad) + ' divergence(s)'}")
    return 0 if not bad else 1


if __name__ == "__main__":
    raise SystemExit(main())
