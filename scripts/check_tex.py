"""Would the LaTeX build compile? There is no LaTeX here, so check what can be.

The supplement's .tex files are generated from the same modules that build the
PDF, and nothing on this machine can compile them. These are the failures that
would be silent until someone tried: a percent sign that comments out the rest
of a line, an unbalanced brace, an environment that never closes, and a command
that is neither a macro this project defines nor one LaTeX provides.

    python scripts/check_tex.py
"""

from __future__ import annotations

import glob
import re
from collections import Counter
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent

# Commands LaTeX, amsmath, graphicx or the cvpr class provide. Not exhaustive:
# it is the list this project's own text actually reaches for, and a name
# arriving here that is genuinely standard belongs in it.
KNOWN = {
    "section", "subsection", "subsubsection", "paragraph", "label", "ref",
    "cite", "textbf", "emph", "textit", "small", "footnotesize", "begin",
    "end", "item", "itemize", "enumerate", "itemsep", "table", "figure",
    "tabular", "toprule", "midrule", "bottomrule", "multicolumn", "centering",
    "resizebox", "columnwidth", "textwidth", "linewidth", "includegraphics",
    "caption", "input", "vspace", "hspace", "clearpage", "newpage",
    "times", "approx", "leq", "geq", "le", "ge", "neq", "rightarrow", "to",
    "mapsto", "infty", "propto", "cdot", "pm", "ast", "star", "dagger",
    "alpha", "beta", "gamma", "delta", "epsilon", "varepsilon", "zeta", "eta",
    "theta", "kappa", "lambda", "mu", "nu", "xi", "pi", "rho", "sigma", "tau",
    "phi", "varphi", "chi", "psi", "omega", "Delta", "Sigma", "Omega", "Phi",
    "Lambda", "prime", "bar", "hat", "tilde", "vec", "dot", "ddot",
    "sum", "prod", "int", "frac", "sqrt", "log", "ln", "exp", "max", "min",
    "arg", "lim", "sup", "inf", "mathrm", "mathbb", "mathcal", "mathbf",
    "operatorname", "text", "left", "right", "big", "Big", "bigoplus",
    "bigcup", "bigcap", "oplus", "otimes", "cup", "cap", "in", "notin",
    "subset", "subseteq", "forall", "exists", "partial", "nabla", "ell",
    "lceil", "rceil", "lfloor", "rfloor", "langle", "rangle", "quad", "qquad",
    "ldots", "dots", "cdots", "circ", "Longleftrightarrow", "Leftrightarrow",
    "documentclass", "usepackage", "newcommand", "renewcommand", "setcounter",
    "maketitle", "appendix", "title", "author", "IfFileExists", "bibliography",
    "bibliographystyle", "texttt", "url", "href", "footnote", "dB",
    "bigl", "bigr", "Bigl", "Bigr", "bmod", "pmod", "lVert", "rVert",
    "lvert", "rvert", "textcolor", "etal", "note", "sim", "simeq",
    "approxeq", "equiv", "colon",
}


def main():
    macros = set(re.findall(
        r"\\newcommand\{\\(\w+)\}",
        (ROOT / "paper/tables/macros.tex").read_text()))
    files = sorted(glob.glob(str(ROOT / "paper/supp/[a-z]_*.tex")))
    files += [str(ROOT / "paper/main.tex")]
    bad = 0
    for f in files:
        s = Path(f).read_text()
        name = Path(f).name
        issues = []

        for n, line in enumerate(s.splitlines(), 1):
            if line.lstrip().startswith("%"):
                continue
            # Two legitimate uses of a bare %: {% opening a group and a
            # trailing % that joins this line to the next without a space.
            # Neither is a stray comment.
            probe = re.sub(r"%\s*$", "", line.replace("{%", "{"))
            if re.search(r"(?<!\\)%", probe):
                issues.append(f"bare % on line {n}")
                break

        depth = s.count("{") - s.count("}")
        if depth:
            issues.append(f"braces {depth:+d}")

        envs = Counter(re.findall(r"\\begin\{(\w+\*?)\}", s))
        ends = Counter(re.findall(r"\\end\{(\w+\*?)\}", s))
        if envs != ends:
            issues.append(f"environments {dict(envs - ends)}{dict(ends - envs)}")

        body = re.sub(r"(?m)^\s*%.*$", "", s)
        unknown = sorted({m for m in re.findall(r"\\([A-Za-z]+)", body)
                          if m not in macros and m not in KNOWN})
        if unknown:
            issues.append(f"undefined: {', '.join(unknown[:6])}")

        if issues:
            bad += 1
            print(f"  {name:<24} {' | '.join(issues)}")

    print(f"  {len(files)} files checked")
    bad += check_refs_and_bib(ROOT)
    print(f"\n  {'PASS' if not bad else str(bad) + ' issue(s)'}")
    return 0 if not bad else 1


def check_refs_and_bib(root):
    """Dangling \\ref and uncited bib entries, across main.tex and its inputs.

    \\ref{sec:tiles} pointed at a label that never existed, so two sentences in
    the built paper would have read "Section ??". bibtex is equally quiet the
    other way: an entry nothing cites is silently dropped, which makes the
    LaTeX reference list shorter than the reportlab one and renumbers everything
    after it. Both are invisible in the source and obvious in the artefact.
    """
    import os
    main = (root / "paper/main.tex").read_text()
    files = [root / "paper/main.tex"]
    for i in re.findall(r"\\input\{([^}]+)\}", main):
        f = root / "paper" / (i if i.endswith(".tex") else i + ".tex")
        if f.exists():
            files.append(f)
    s = "\n".join(f.read_text() for f in files)
    labels = set(re.findall(r"\\label\{([^}]+)\}", s))
    refs = set(re.findall(r"\\(?:ref|autoref|cref)\{([^}]+)\}", s))
    dangling = sorted(refs - labels)
    bibf = root / "paper/refs.bib"
    uncited = []
    if bibf.exists():
        bib = set(re.findall(r"@\w+\{([^,]+),", bibf.read_text()))
        cited = set()
        for m in re.finditer(r"\\cite[a-z]*\{([^}]*)\}", s):
            cited |= {k.strip() for k in m.group(1).split(",")}
        uncited = sorted(bib - cited)
    # An abstract is read on its own, off the paper's own page. A figure
    # reference there points at something the reader cannot see; one arrived
    # when the cross-references were placed automatically.
    _m = re.search(r"\\begin\{abstract\}(.*?)\\end\{abstract\}", main, re.S)
    _abs = re.findall(r"\\(?:ref|autoref|cref)\{[^}]*\}", _m.group(1)) if _m else []
    for r in _abs:
        print(f"     REFERENCE IN THE ABSTRACT: {r}")

    # A numbered display equation nothing points at is a number on the page for
    # no reason; CVPR sets those unnumbered. And an equation that ends without
    # punctuation reads as if the sentence stopped.
    _eq = re.findall(r"\\label\{(eq:[^}]+)\}", s)
    _uneq = [e for e in _eq
             if not re.search(r"\\(?:ref|eqref)\{" + re.escape(e) + r"\}", s)]
    for e in _uneq:
        print(f"     numbered equation never referenced: {e}")
    _nopunct = []
    for m in re.finditer(r"\\begin\{(equation|align)\}(.*?)\\end\{\1\}", s, re.S):
        body = re.sub(r"\\label\{[^}]*\}", "", m.group(2)).strip()
        if body and body[-1] not in ".,;":
            _nopunct.append(re.sub(r"\s+", " ", body)[:40])
    for b in _nopunct:
        print(f"     display equation ends without punctuation: {b}")

    for r in dangling:
        print(f"     DANGLING \\ref{{{r}}} -- prints as ??")
    for k in uncited:
        print(f"     UNCITED bib entry {k} -- bibtex will drop it")
    print(f"  refs: {len(refs)} used, {len(dangling)} dangling; "
          f"bib: {len(uncited)} uncited")
    return (len(dangling) + len(uncited) + len(_abs)
            + len(_uneq) + len(_nopunct))


if __name__ == "__main__":
    raise SystemExit(main())
