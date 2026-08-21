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
    print(f"\n  {'PASS' if not bad else str(bad) + ' file(s) with issues'}")
    return 0 if not bad else 1


if __name__ == "__main__":
    raise SystemExit(main())
