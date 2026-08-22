"""A LaTeX backend for the supplement, with the same interface the sections use.

paper/supplementary.tex inputs eight section files that have never existed: the
supplement was written as reportlab modules and the LaTeX path was left behind.
Anyone compiling the submission with LaTeX gets a supplement with nothing in it.

Rather than write those files by hand and let them drift, this renders them from
the same modules. `TexKit` exposes the toolkit `Kit` exposes, so a section is
written once and comes out twice.

    from supp_tex import TexKit
    TexKit(slug).render(module.content)

Two things it deliberately does not do. It does not expand \\Macro, because
LaTeX will; the reportlab side expands them because nothing else would. And it
does not renumber floats, because LaTeX counts them itself from
supp/_counters.tex.
"""

from __future__ import annotations

import json
import re
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
RESULTS = ROOT / "results"
SUPP_TEX = ROOT / "paper" / "supp"

# Characters the sections write as unicode because reportlab wants them that
# way, and their LaTeX spellings. Greek is left to the macros where the section
# used one.
UNI = {
    "×": r"$\times$", "−": r"$-$", "≈": r"$\approx$", "≤": r"$\leq$",
    "≥": r"$\geq$", "→": r"$\rightarrow$", "∞": r"$\infty$", "∝": r"$\propto$",
    "·": r"$\cdot$", "±": r"$\pm$", "λ": r"$\lambda$", "β": r"$\beta$",
    "α": r"$\alpha$", "ρ": r"$\rho$", "σ": r"$\sigma$", "τ": r"$\tau$",
    "μ": r"$\mu$", "Δ": r"$\Delta$", "θ": r"$\theta$", "φ": r"$\varphi$",
    "²": r"$^2$", "³": r"$^3$", "°": r"$^\circ$", "ŷ": r"$\hat{y}$",
    "“": "``", "”": "''", "’": "'", "–": "--", "—": "---",
}


def esc(t: str) -> str:
    """Escape what LaTeX would misread, and leave what it should read."""
    t = str(t)
    # protect existing commands and math before touching anything
    keep = []

    def stash(m):
        keep.append(m.group(0))
        return f"\x00{len(keep) - 1}\x00"

    t = re.sub(r"\$[^$]*\$|\\[A-Za-z]+", stash, t)
    for a, b in UNI.items():
        t = t.replace(a, b)
    for a in ("&", "#"):
        t = t.replace(a, "\\" + a)
    t = re.sub(r"(?<!\\)%", r"\\%", t)
    t = re.sub(r"(?<!\\)_", r"\\_", t)
    t = re.sub(r"\x00(\d+)\x00", lambda m: keep[int(m.group(1))], t)
    # the tags the sections use for emphasis
    t = re.sub(r"<b>(.*?)</b>", r"\\textbf{\1}", t, flags=re.S)
    t = re.sub(r"<i>(.*?)</i>", r"\\emph{\1}", t, flags=re.S)
    t = re.sub(r"<sub>(.*?)</sub>", r"$_{\1}$", t, flags=re.S)
    t = re.sub(r"<super>(.*?)</super>", r"$^{\1}$", t, flags=re.S)
    t = re.sub(r"<[^>]+>", "", t)
    return t


class TexKit:
    """The same surface as Kit, emitting LaTeX."""

    def __init__(self, slug: str):
        self.slug = slug
        self.out: list[str] = []
        self.R, self.RESULTS = ROOT, RESULTS
        self.colw, self.fullw = 251.0, 530.0
        self._json: dict = {}
        self._nfig = self._ntab = 0
        self.used_results: list[str] = []
        self.bad_glyphs: set = set()
        self.REFS: list = []

    # -- structure ---------------------------------------------------------
    def h1(self, title):
        self.out.append(f"\\section{{{esc(title)}}}\n"
                        f"\\label{{supp:{self.slug}}}\n")
        return "A"

    def h2(self, title):
        self.out.append(f"\\subsection{{{esc(title)}}}\n")
        return "A.1"

    def h3(self, title):
        self.out.append(f"\\paragraph{{{esc(title)}}}\n")

    # -- text --------------------------------------------------------------
    def par(self, text):
        self.out.append(esc(text) + "\n")

    def note(self, text):
        self.out.append("{\\small " + esc(text) + "}\n")

    def bullets(self, items):
        self.out.append("\\begin{itemize}\\itemsep1pt")
        for it in items:
            self.out.append(f"\\item {esc(it)}")
        self.out.append("\\end{itemize}\n")

    def spacer(self, h=4):
        self.out.append(f"\\vspace{{{h}pt}}\n")

    # -- floats ------------------------------------------------------------
    def rows(self, rows, cap="", header=True):
        self._ntab += 1
        if not rows:
            return self._ntab
        ncol = max(len(r) for r in rows)
        spec = "l" + "r" * (ncol - 1)
        body = []
        for n, r in enumerate(rows):
            cells = [esc(c) for c in r] + [""] * (ncol - len(r))
            body.append(" & ".join(cells) + r" \\")
            if header and n == 0:
                body.append(r"\midrule")
        self.out.append(
            "\\begin{table}[t]\n\\centering\n\\resizebox{\\columnwidth}{!}{%\n"
            f"\\begin{{tabular}}{{{spec}}}\n\\toprule\n"
            + "\n".join(body)
            + "\n\\bottomrule\n\\end{tabular}}\n"
            + (f"\\caption{{{esc(cap)}}}\n" if cap else "")
            + "\\end{table}\n")
        return self._ntab

    def tbl(self, name, cap=""):
        self._ntab += 1
        self.out.append(
            "\\begin{table}[t]\n\\centering\n"
            f"\\resizebox{{\\columnwidth}}{{!}}{{\\input{{tables/{name}}}}}\n"
            + (f"\\caption{{{esc(cap)}}}\n" if cap else "")
            + "\\end{table}\n")
        return self._ntab

    def fig(self, name, cap="", maxh=None):
        self._nfig += 1
        stem = name[:-4] if name.endswith(".png") else name
        self.out.append(
            "\\begin{figure}[t]\n\\centering\n"
            f"\\includegraphics[width=\\columnwidth]{{figures/{stem}.png}}\n"
            + (f"\\caption{{{esc(cap)}}}\n" if cap else "")
            + "\\end{figure}\n")
        return self._nfig

    def figwide(self, name, cap="", height=None):
        self._nfig += 1
        stem = name[:-4] if name.endswith(".png") else name
        self.out.append(
            "\\begin{figure*}[t]\n\\centering\n"
            f"\\includegraphics[width=\\textwidth]{{figures/{stem}.png}}\n"
            + (f"\\caption{{{esc(cap)}}}\n" if cap else "")
            + "\\end{figure*}\n")
        return self._nfig

    def eq(self, latex, tag=True):
        env = "equation" if tag else "equation*"
        self.out.append(f"\\begin{{{env}}}\n{latex}\n\\end{{{env}}}\n")
        return 0

    # -- data --------------------------------------------------------------
    def J(self, *filenames):
        """The first of these that exists; see build_supp_pdf.J."""
        filename = next((n for n in filenames if (RESULTS / n).exists()),
                        filenames[0])
        if filename not in self._json:
            p = RESULTS / filename
            if not p.exists():
                raise FileNotFoundError(str(p))
            self._json[filename] = json.loads(p.read_text())
            self.used_results.append(f"{self.slug}: {filename}")
        return json.loads(json.dumps(self._json[filename]))

    def macro(self, name):
        src = (ROOT / "paper/tables/macros.tex").read_text()
        m = re.search(r"\\newcommand\{\\" + name + r"\}\{([^}]*)\}", src)
        if not m:
            raise KeyError(name)
        return m.group(1)

    def peek_fig(self):
        return self._nfig + 1

    def peek_tbl(self):
        return self._ntab + 1

    # -- driver ------------------------------------------------------------
    def render(self, content):
        content(self)
        SUPP_TEX.mkdir(parents=True, exist_ok=True)
        path = SUPP_TEX / f"{self.slug}.tex"
        path.write_text(
            f"% Generated by scripts/supp_tex.py from scripts/supp/{self.slug}.py.\n"
            "% Do not edit: edit the module, which also builds the PDF version.\n\n"
            + "\n".join(self.out) + "\n")
        return path
