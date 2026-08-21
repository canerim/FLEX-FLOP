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


DIRECTION = {
    # macro-name fragment -> words that mean the opposite of what it holds
    "Beats": ("gives up", "loses", "behind", "short of", "falls short"),
    "Ahead": ("gives up", "loses", "behind", "short of", "falls short"),
    "Recovers": ("gives up", "loses"),
    "Gap": ("beats", "ahead of"),
    "Loses": ("beats", "ahead of"),
}


def directions() -> list[str]:
    """A macro that means "ahead by" must not sit in a sentence that says
    "gives up".

    Section C of the supplement wrote "The rule gives up \\RateRankBeatsBy
    saving points to the trained head" where that macro holds the rule's
    *margin over* the head. The number was right, the sentence said the
    opposite of the abstract, and every numeric check passed because the value
    matched its own definition.
    """
    import glob
    out = []
    for f in [str(ROOT / "scripts/build_pdf.py"), str(ROOT / "paper/main.tex")] \
            + sorted(glob.glob(str(ROOT / "scripts/supp/*.py"))):
        src = open(f).read()
        for m in re.finditer(r"\\\\?([A-Za-z]+)", src):
            name = m.group(1)
            for frag, words in DIRECTION.items():
                if frag not in name:
                    continue
                # the sentence around the use, in the source's own text
                lo = max(0, m.start() - 260)
                ctx = re.sub(r"\s+", " ", src[lo:m.end() + 120]).lower()
                ctx = ctx.rsplit(".", 1)[-1] if ". " in ctx[-200:] else ctx
                for w in words:
                    if w in ctx:
                        out.append(f"{Path(f).name}: \\{name} used in a "
                                   f"sentence that says \"{w}\"")
                        break
    return sorted(set(out))


def panel_commas() -> list[str]:
    """Panel labels are set "a, the decoded frame": letter, comma, lower case.

    Sixteen captions had "<b>a</b> The decoded frame", which reads as a
    sentence that begins with a stray letter. The built PDF cannot be checked
    for this -- the bold run is gone by then -- so it is checked at the source,
    where <b>a</b> is unambiguous. A label followed by a colon, an en-dash or a
    second label is a deliberate grouping and passes; a label preceded by a
    word is a reference to a panel in running text, not a label.
    """
    import glob
    out = []
    for f in [str(ROOT / "scripts/build_pdf.py")] + sorted(
            glob.glob(str(ROOT / "scripts/supp/*.py"))):
        s = open(f).read()
        for m in re.finditer(r"(.{0,14})<b>([a-f])</b>(.)", s):
            before, letter, after = m.groups()
            if after in ",:\u2013-" or after == "<":
                continue
            if re.search(r"(?:panel|panels|axis of|and|in)\s+$", before):
                continue
            out.append(f"{Path(f).name}: <b>{letter}</b> not followed by a "
                       f"comma")
    return out


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
    # A doubled backslash before a macro survives LaTeX as a line break and
    # prints as a literal in the reportlab build: the abstract read "\\454 GMAC"
    # for one build that way.
    _dbl = re.findall(r"\\\\\\\\[A-Za-z]+", s)
    for d in set(_dbl):
        print(f"     DOUBLED BACKSLASH before a macro: {d}")

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

    _panels = panel_commas() + directions()
    for b in _panels:
        print(f"     {b}")

    for r in dangling:
        print(f"     DANGLING \\ref{{{r}}} -- prints as ??")
    for k in uncited:
        print(f"     UNCITED bib entry {k} -- bibtex will drop it")
    print(f"  refs: {len(refs)} used, {len(dangling)} dangling; "
          f"bib: {len(uncited)} uncited")
    return (len(dangling) + len(uncited) + len(_abs)
            + len(_uneq) + len(_nopunct) + len(set(_dbl)) + len(_panels))


if __name__ == "__main__":
    raise SystemExit(main())
