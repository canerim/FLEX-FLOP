"""Render the supplementary material to PDF, in the main paper's style.

`paper/supplementary.tex` remains the submission source. This produces a
reviewable PDF from the same generated tables, the same macros and the same
`results/` files, so what is on screen is what the measurements say. It is the
sibling of `scripts/build_pdf.py` and it imports that module rather than
restating it: the page geometry, the Times body style, the LaTeX-table reader,
the macro expansion and the mathtext equation renderer are all taken from
there, so the two documents cannot drift apart typographically.

Run it with no arguments:

    ./.venv/bin/python scripts/build_supp_pdf.py

which writes `paper/FLEX-UF-supp.pdf`.


HOW A SECTION IS WRITTEN
========================

One section is one module in `scripts/supp/`, exposing a single function:

    def content(k):
        k.h1("Router feature ablations")
        ...

`ORDER` below is the list of module names, without the `.py`, in the order
they appear in the document. A module on disk that is not in `ORDER` is not
built, and the build says so rather than silently dropping it. Alphabetical by
filename is the default habit, but `ORDER` is what runs.

Each module name is also its LaTeX slug: `scripts/supp/router.py` pairs with
`paper/supp/router.tex`, and `paper/supplementary.tex` inputs the same slugs in
the same order. Keep the two in step.


THE TOOLKIT `k`
===============

Everything a section may emit goes through `k`. The signatures below are
stable; treat them as an interface, not an implementation detail.

Structure
---------
    k.h1(title)                 -> str
        Section heading, auto-numbered A, B, C, ... across the whole
        supplement. Returns the letter it was given. Resets the h2 counter.

    k.h2(title)                 -> str
        Subsection heading, auto-numbered A.1, A.2, ... within the current
        section. Returns the label.

    k.h3(title)                 -> None
        Unnumbered bold sub-subheading on its own line. For a LaTeX
        \\paragraph-style run-in heading instead, put it in the text:
        k.par("<b>Costs.</b> With K exits over ...").

Text
----
    k.par(text)                 -> None
        A body paragraph. `<b>` and `<i>` are allowed, and so are the LaTeX
        forms `\\textbf{...}`, `\\emph{...}`, `\\%`, `\\times`, `\\lambda` and
        the other TeX-isms `build_pdf.sub` understands. Every `\\Macro` from
        `paper/tables/macros.tex` is expanded here; an unexpanded one is
        reported at the end of the build.

    k.bullets(items)            -> None
        A bulleted list. `items` is a list of strings, each formatted as
        k.par formats a paragraph.

    k.note(text)                -> None
        A small-print line in caption type. This is where a provenance
        sentence goes: the results file a table was read from, and the
        checkpoint if it is not runs/RECIPE512/ckpt_PAPER.pth.tar.

Floats
------
    k.tbl(name, cap)            -> int
        The generated table `paper/tables/<name>.tex`, at column width, with
        `cap` beneath it. Returns the table number.

    k.rows(rows, cap, header=True)  -> int
        An inline table built from data rather than from a .tex file. `rows`
        is a list of lists; cells may be str, int or float. With header=True
        the first row is treated as a header: ruled beneath, and reprinted if
        the table breaks across a column. First column is left-aligned, the
        rest right-aligned. Returns the table number.

    k.fig(name, cap, maxh=None) -> int
        A column-width figure. `name` is a filename, looked up in
        `docs/figures/` first and then `paper/figures/`. Capped at 1.25 inches
        tall unless `maxh` (in points) says otherwise, so that a figure and its
        caption finish the column they start in. Returns the figure number.

    k.figwide(name, cap)        -> int
        The same figure spanning both columns. reportlab has no float
        mechanism, so this forces a page break and puts the figure in a
        full-width band at the top of the next page, which is what LaTeX would
        do with a figure* anyway. Returns the figure number.

    k.eq(latex, tag=True)       -> int
        A display equation, rendered by `build_pdf._render_math` (matplotlib
        mathtext, stix, at body size). `latex` is math-mode source without the
        surrounding dollars. Numbered at the right margin unless tag=False.
        Returns the equation number. Mathtext is not LaTeX: `\\frac`, `\\mathrm`,
        `\\propto`, `\\approx`, `\\times`, sub- and superscripts all work, but
        `{+}` spacing tricks, `\\!` and `\\text{}` do not.

    Figure, table and equation numbering continues from the main paper. The
    starting values are not typed anywhere. This PDF continues from what the
    main paper's own PDF build ends at, obtained by running that build with
    the write stubbed out (`_main_counters`). `paper/supp/_counters.tex`, which
    `supplementary.tex` inputs, continues instead from what `main.tex` itself
    ends at, counted from its source (`_tex_counters`), because that is the
    document LaTeX numbers. Inserting a figure in the main paper therefore
    renumbers both without anyone editing this file, and if the two counts
    disagree the build says so.

Data
----
    k.J(filename)               -> parsed JSON
        Load `results/<filename>`. Raises FileNotFoundError, naming the path,
        if it is not there. Every call returns a fresh object, so a section
        cannot corrupt another section's copy. The build prints the list of
        files read, which is the provenance record for the document.

    k.macro(name)               -> str
        The value of `\\<name>` from `paper/tables/macros.tex`, as a string.
        Raises KeyError on a name that does not exist, which is the point:
        it is how an invented macro name gets caught. In prose, prefer writing
        `\\Ceiling` straight into k.par and letting it expand; use k.macro when
        the value has to be computed with or compared against something.

Odds and ends
-------------
    k.spacer(h=4)               -> None    vertical air, in points
    k.peek_fig()                -> int     the number the next figure will get
    k.peek_tbl()                -> int     the number the next table will get
    k.colw, k.fullw             -> float   column and text width, in points
    k.R, k.RESULTS              -> Path    repository root, results directory
    k.REFS                      -> list    the main paper's reference list, so
                                           a bracketed [14] in supplementary
                                           prose means what it means there

What renders and what does not
------------------------------
The body face is Times-Roman. reportlab falls back to Symbol for Greek and for
most mathematical operators, so λ β α Δ ρ θ φ × − ≈ ∞ ∝ → ≤ ∈ ∑ √ ∂ and ⋅ are
all safe, as is everything in Latin-1, which covers ² ³ ° ± · × and the en dash.
It has no glyph for superscript or subscript digits (10⁻⁵, xₖ), for ℝ, or for
anything further out, and draws a black box. Write

    "4.7×10<super>-5</super>"     not     "4.7×10⁻⁵"
    "c<sub>k</sub>" or "c_k"      not     "cₖ"

Every string this module sets is checked, and the build names any character it
cannot draw, with the section it came from. A box in a submission PDF is the
kind of defect nobody notices until a reviewer does.

WHAT THE BUILD TELLS YOU
========================

The build is quiet when everything is in order and prints a line per problem
otherwise. Read them; each one is a defect that survives into the PDF.

    UNEXPANDED MACROS            a \\Name in prose that macros.tex does not
                                 define. It prints as literal backslash-Name.
                                 Either the name is wrong or
                                 scripts/make_paper_tables.py has not been run.
    NO GLYPH IN Times-Roman      a character that draws as a black box, named
                                 with its code point and the section it is in.
    MISSING SECTION              a name in ORDER with no module on disk.
    NOT BUILT                    a module on disk with no name in ORDER.
    ORDER and supplementary.tex  the \\input lines and ORDER list different
      disagree                   sections, or the same sections in a different
                                 order, so the two documents disagree about
                                 what section B is.
    main.tex itself ends at      build_pdf.py and main.tex no longer agree on
                                 how many figures, tables or equations the main
                                 paper has. The PDF continues from the first,
                                 supplementary.tex from the second.

It also prints every results/ file each section read, which is the provenance
record for the document, and the list of sections it built.


Two things the toolkit will not do for you. It will not check that a number in
your prose is in the file you say it is in; `scripts/check_paper.py` is where
that belongs. And `build_pdf.sub` closes a `<b>` or `<i>` at the next `}`, so
a caption carrying a raw brace and a bold run in the same sentence will close
the bold early. Write `\\textbf{...}` or `<b>...</b>`, not both, and keep raw
braces out of captions.
"""
import contextlib
import hashlib
import importlib
import io
import json
import re
import shutil
import sys
from pathlib import Path

from reportlab.lib import colors
from reportlab.lib.enums import TA_CENTER
from reportlab.lib.pagesizes import letter
from reportlab.lib.units import inch
from reportlab.pdfbase.pdfmetrics import stringWidth
from reportlab.platypus import (BaseDocTemplate, Frame, PageTemplate, Image,
                                Paragraph, Spacer, Table, TableStyle,
                                KeepTogether, FrameBreak, NextPageTemplate,
                                PageBreak)

sys.path.insert(0, str(Path(__file__).resolve().parent))
import build_pdf as B   # noqa: E402  -- the style and the toolkit both live there

R = B.R
RESULTS = R / "results"
SUPP_PY = R / "scripts" / "supp"
SUPP_TEX = R / "paper" / "supp"
DOC_FIGS = R / "docs" / "figures"
EQ_DIR = R / "docs" / "figures" / "_eq" / "supp"

# ---------------------------------------------------------------- the order
# The sections, in the order they appear, named by module. This list is what
# runs; a file in scripts/supp/ that is missing from it is reported and skipped.
# The same names, in the same order, are the \input lines in
# paper/supplementary.tex.
ORDER = [
    "a_implementation",
    "b_complexity",
    "c_derivations",
    "d_full_results",
    "e_budget",
    "f_router",
    "g_algorithms",
    "h_limits",
]


def tex_order():
    """The slugs paper/supplementary.tex inputs, in the order it inputs them.

    The LaTeX submission and this PDF are two renderings of one document, and
    they only agree about what section B is if ORDER and the \\input lines
    agree. Eight people writing sections in parallel will not notice a
    disagreement by reading; the build says so instead. supplementary.tex is
    read here and never written.
    """
    f = R / "paper" / "supplementary.tex"
    if not f.exists():
        return []
    src = re.sub(r"(?<!\\)%.*", "", f.read_text())
    return [m.group(1) for m in re.finditer(r"\\input\{supp/([A-Za-z0-9_]+)\}", src)
            if not m.group(1).startswith("_")]


def discovered():
    """Every section module on disk, alphabetical. The default for ORDER."""
    if not SUPP_PY.exists():
        return []
    return sorted(p.stem for p in SUPP_PY.glob("*.py")
                  if not p.stem.startswith("__"))


# ------------------------------------------------------- continuing the count
def _tex_counters(path=None):
    """(figures, tables, equations) that main.tex ends at, read off its source.

    supplementary.tex is a LaTeX document, so its counters have to continue
    from the LaTeX document beside it, not from the reportlab rebuild of that
    document. The two are separate files and they are edited separately, so
    they can disagree: at the time of writing main.tex sets 5 numbered
    equations where build_pdf sets 7, because the two do not agree on which
    displays carry a tag. Counting the source is the only reading that makes
    \input{supp/_counters} correct.

    Comments are stripped first. figure* and table* share the counters of
    figure and table. For the multi-row display environments a row is numbered
    unless it carries \nonumber or \notag; the starred forms number nothing
    and are not matched at all.
    """
    src = (path or (R / "paper" / "main.tex")).read_text()
    src = re.sub(r"(?<!\\)%.*", "", src)
    nfig = len(re.findall(r"\\begin\{figure\*?\}", src))
    ntab = len(re.findall(r"\\begin\{table\*?\}", src))
    neq = 0
    for m in re.finditer(r"\\begin\{(equation|align|alignat|gather|multline|"
                         r"eqnarray)\}(.*?)\\end\{\1\}", src, re.S):
        env, body = m.group(1), m.group(2)
        if env in ("equation", "multline"):
            neq += 1
        else:
            neq += sum(1 for row in re.split(r"\\\\", body)
                       if row.strip() and "\\nonumber" not in row
                       and "\\notag" not in row)
    return nfig, ntab, neq


def _main_counters():
    """(figures, tables, equations) the main paper's own PDF build ends at.

    This is the count the supplement's PDF continues from, and it is a
    different question from _tex_counters above: that one numbers the LaTeX
    submission, this one numbers the PDF sitting next to it.

    Typed constants would be wrong the first time anyone inserts a figure into
    main.tex, and this document exists to be numbered after that one. So run
    the main paper's own build with the PDF write stubbed out: every fig() and
    tbl() call still fires, the counters still advance, and nothing is written
    to disk. It costs about six tenths of a second.

    build_pdf.UNEXPANDED collects macro names the main paper failed to expand.
    Clear it afterwards so the warning at the end of this build is about the
    supplement and not about a document somebody else is editing.

    That build appends this supplement to the paper, which would count the
    supplement's own floats into the answer. Two things stop it. Registering
    this module under its own name means the import inside build_pdf.build()
    finds it rather than loading a second copy with its own counters, and
    stubbing _sections leaves the supplement's flow empty, so the count that
    comes back is the main paper's alone.
    """
    B._FIGN[0] = B._TABN[0] = B._EQN[0] = 0
    real_build = BaseDocTemplate.build
    real_sections = globals()["_sections"]
    prev = sys.modules.get("build_supp_pdf")
    try:
        BaseDocTemplate.build = lambda self, story, *a, **kw: None
        sys.modules["build_supp_pdf"] = sys.modules[__name__]
        globals()["_sections"] = lambda kit: []
        with contextlib.redirect_stdout(io.StringIO()):
            B.build("/dev/null")
    finally:
        BaseDocTemplate.build = real_build
        globals()["_sections"] = real_sections
        if prev is None:
            sys.modules.pop("build_supp_pdf", None)
        else:
            sys.modules["build_supp_pdf"] = prev
    B.UNEXPANDED.clear()
    return B._FIGN[0], B._TABN[0], B._EQN[0]


def _letter(n):
    """1 -> A, 26 -> Z, 27 -> AA. Section labels, appendix style."""
    s = ""
    while n > 0:
        n, r = divmod(n - 1, 26)
        s = chr(ord("A") + r) + s
    return s


# ---------------------------------------------------------------- extra styles
H3 = B.S("h3", fontName="Times-Bold", fontSize=8.8, leading=10.6,
         spaceBefore=5, spaceAfter=2, alignment=0)
BULLET = B.S("bullet", leftIndent=9, bulletIndent=0, spaceAfter=3)
NOTE = B.S("note", fontSize=7.0, leading=8.4, spaceBefore=1, spaceAfter=5,
           alignment=0)
SUBTITLE = B.S("subtitle", fontName="Times-Bold", fontSize=11.5, leading=13,
               alignment=TA_CENTER, spaceAfter=4)


@contextlib.contextmanager
def _figdir(d):
    """Point build_pdf.fig at another directory for the length of the call.

    fig() resolves `FIGS / name` from module globals at call time, so this is
    enough to reuse its scaling and caption handling verbatim on a figure that
    lives in docs/figures rather than paper/figures.
    """
    old = B.FIGS
    B.FIGS = d
    try:
        yield
    finally:
        B.FIGS = old


def _own_copy(path, latex):
    """Give this equation a PNG that nothing else will overwrite.

    build_pdf._render_math names its output after the first 60 characters of
    the source with everything but letters and digits collapsed to
    underscores, and it writes rather than caches. Two equations that agree
    that far apart land on the same path, and because reportlab opens an image
    when it builds the document rather than when the flowable is made, the
    first equation then prints the second one's picture. This is not
    hypothetical: \frac{(k-1)b}{N} and \frac{(k+1)b}{N} collapse to the same
    key, and the second of them is already in the legacy section. Copying each
    render to a name carrying a hash of the whole source removes the collision
    inside this document and against the main paper, which shares the cache
    directory. The fix belongs in build_pdf; it is here because that file is
    being edited elsewhere.
    """
    EQ_DIR.mkdir(parents=True, exist_ok=True)
    dst = EQ_DIR / f"{hashlib.sha1(latex.encode()).hexdigest()[:12]}.png"
    shutil.copyfile(path, dst)
    return str(dst)


def _label(cap, word):
    """Prepend a bold '<word> 0.' for _autonum to overwrite, if absent."""
    if re.search(rf"{word}\s+\d+\.", cap):
        return cap
    return f"<b>{word} 0.</b> {cap}"


# ------------------------------------------------------------- glyph guard
# The body face is Times-Roman, one of the base-14 fonts, in WinAnsiEncoding.
# reportlab silently falls back to Symbol for Greek and for most mathematical
# operators, so λ, β, Δ, ×, −, ≈, ∞, ∝, → and ⋅ all set correctly. It has
# nothing to fall back to for superscript and subscript digits (U+2070..U+209F),
# for blackboard bold, or for anything outside Latin-1 with an accent, and draws
# a black box instead. A box in a submission PDF is the kind of defect nobody
# notices until a reviewer does, so every string this module sets is checked.
#
# The test: characters Latin-1 can encode are fine by construction; for the rest,
# a width equal to the notdef width means reportlab has no glyph. Write 10<super>-5</super>
# rather than 10⁻⁵, and C² and 10³ are fine as they are.
_NOTDEF_W = stringWidth("\uf8ff", "Times-Roman", 10)   # private use: never a glyph


def _glyphs(s, where, sink):
    for ch in set(s):
        if ch in " \n\t":
            continue
        try:
            ch.encode("cp1252")
            continue
        except UnicodeEncodeError:
            pass
        if stringWidth(ch, "Times-Roman", 10) == _NOTDEF_W:
            sink.add((ch, f"U+{ord(ch):04X}", where))
    return s


# ---------------------------------------------------------------- the toolkit
class Kit:
    """What a section module is handed. See the module docstring for the API."""

    def __init__(self, colw, fullw):
        self.colw = colw
        self.fullw = fullw
        self.R = R
        self.RESULTS = RESULTS
        self.REFS = B.REFS
        self.flow = []
        self.used_results = []
        self.bad_glyphs = set()
        self._sec = 0
        self._sub = 0
        self._sec_letter = ""
        self._json_cache = {}
        self._section = "<preamble>"

    # -- internal ---------------------------------------------------------
    def _add(self, f):
        self.flow.append(f)

    def _t(self, text):
        """Expand macros, then check every character has a glyph."""
        return _glyphs(B.sub(str(text)), self._section, self.bad_glyphs)

    # -- structure --------------------------------------------------------
    def h1(self, title):
        self._sec += 1
        self._sub = 0
        self._sec_letter = _letter(self._sec)
        self._add(Paragraph(self._t(f"{self._sec_letter}. {title}"), B.H1))
        return self._sec_letter

    def h2(self, title):
        self._sub += 1
        lab = f"{self._sec_letter or 'A'}.{self._sub}"
        self._add(Paragraph(self._t(f"{lab}. {title}"), B.H2))
        return lab

    def h3(self, title):
        self._add(Paragraph(self._t(title), H3))

    # -- text -------------------------------------------------------------
    def par(self, text):
        self._add(Paragraph(self._t(text), B.BODY))

    def bullets(self, items):
        for it in items:
            self._add(Paragraph(self._t(it), BULLET, bulletText="•"))

    def note(self, text):
        self._add(Paragraph(self._t(text), NOTE))

    def spacer(self, h=4):
        self._add(Spacer(1, h))

    # -- floats -----------------------------------------------------------
    def tbl(self, name, cap):
        t = B.tex_table(name, self.colw)
        t.spaceBefore = 6
        cap = B._autonum(_label(self._t(cap), "Table"), B._TABN, "Table")
        n = B._TABN[0]
        block = [t, Spacer(1, 3), Paragraph(cap, B.CAP)]
        self._emit_table_block(t, block)
        return n

    def rows(self, rows, cap, header=True):
        t = self._data_table(rows, header)
        t.spaceBefore = 6
        cap = B._autonum(_label(self._t(cap), "Table"), B._TABN, "Table")
        n = B._TABN[0]
        block = [t, Spacer(1, 3), Paragraph(cap, B.CAP)]
        self._emit_table_block(t, block)
        return n

    def _emit_table_block(self, t, block):
        """Keep a table with its caption unless it is too tall to place whole.

        Past about half a frame, KeepTogether cannot fit the block into a
        column somebody has already started writing in, so reportlab pushes the
        whole thing to the next frame and strands the text above it. Over that
        size, let it break the way a longtable does; repeatRows=1 reprints the
        header on the far side.
        """
        if t.wrap(self.colw, B.FRAME_H)[1] > 0.5 * B.FRAME_H:
            for f in block:
                self._add(f)
        else:
            self._add(KeepTogether(block))

    def _data_table(self, rows, header):
        text_rows = [[self._t(c) for c in r] for r in rows]
        ncol = max(len(r) for r in text_rows)
        text_rows = [r + [""] * (ncol - len(r)) for r in text_rows]
        fs = 7.2 if ncol <= 7 else 6.0
        cw = B._col_widths(text_rows, ncol, self.colw, fs)
        cells = [[Paragraph(c, B.S("cell", fontSize=fs, leading=fs * 1.16,
                                  alignment=0 if i == 0 else 2))
                  for i, c in enumerate(r)] for r in text_rows]
        t = Table(cells, colWidths=cw, hAlign="CENTER",
                  repeatRows=1 if header else 0)
        style = [
            ("LINEABOVE", (0, 0), (-1, 0), 0.8, colors.black),
            ("LINEBELOW", (0, -1), (-1, -1), 0.8, colors.black),
            ("TOPPADDING", (0, 0), (-1, -1), 1.6),
            ("BOTTOMPADDING", (0, 0), (-1, -1), 1.6),
            ("LEFTPADDING", (0, 0), (-1, -1), B.CELL_PAD),
            ("RIGHTPADDING", (0, 0), (-1, -1), B.CELL_PAD),
            ("VALIGN", (0, 0), (-1, -1), "MIDDLE"),
        ]
        if header:
            style.append(("LINEBELOW", (0, 0), (-1, 0), 0.4, colors.black))
        t.setStyle(TableStyle(style))
        return t

    def fig(self, name, cap, maxh=None):
        d = DOC_FIGS if (DOC_FIGS / name).exists() else B.FIGS
        with _figdir(d):
            flow = B.fig(name, self.colw, _label(self._t(cap), "Figure"),
                         maxh=maxh if maxh else 1.25 * inch)
        n = B._FIGN[0]
        self._add(KeepTogether(flow))
        return n

    def figwide(self, name, cap, height=B.WIDE_BAND - 0.62 * inch):
        d = DOC_FIGS if (DOC_FIGS / name).exists() else B.FIGS
        self._add(NextPageTemplate("wide"))
        self._add(PageBreak())
        with _figdir(d):
            flow = B.fig(name, self.fullw, _label(self._t(cap), "Figure"),
                         maxh=height)
        n = B._FIGN[0]
        for f in flow:
            self._add(f)
        self._add(FrameBreak())
        self._add(NextPageTemplate("rest"))
        return n

    def eq(self, latex, tag=True):
        B._EQN[0] += 1
        n = B._EQN[0]
        tagw = 0.34 * inch
        avail = self.colw - 2 * tagw
        path, w, h = B._render_math(latex)
        path = _own_copy(path, latex)
        if w > avail:
            h *= avail / w
            w = avail
        im = Image(path, width=w, height=h)
        num = (Paragraph(f"({n})", B.S("eqnum", fontSize=B.BODY.fontSize,
                                       leading=B.BODY.fontSize, alignment=2))
               if tag else "")
        t = Table([["", im, num]], colWidths=[tagw, avail, tagw])
        t.setStyle(TableStyle([
            ("ALIGN", (1, 0), (1, 0), "CENTER"),
            ("ALIGN", (2, 0), (2, 0), "RIGHT"),
            ("VALIGN", (0, 0), (-1, -1), "MIDDLE"),
            ("LEFTPADDING", (0, 0), (-1, -1), 0),
            ("RIGHTPADDING", (0, 0), (-1, -1), 0),
            ("TOPPADDING", (0, 0), (-1, -1), 0),
            ("BOTTOMPADDING", (0, 0), (-1, -1), 0)]))
        t.spaceBefore = t.spaceAfter = B.DISPLAY_SKIP
        self._add(t)
        return n

    # -- data -------------------------------------------------------------
    def J(self, filename):
        p = RESULTS / filename
        if filename not in self._json_cache:
            if not p.exists():
                raise FileNotFoundError(
                    f"{self._section}: no such results file: {p}")
            self._json_cache[filename] = p.read_text()
            self.used_results.append(f"{self._section}: {filename}")
        return json.loads(self._json_cache[filename])

    def macro(self, name):
        if name not in B.MACROS:
            raise KeyError(
                f"{self._section}: no macro \\{name} in paper/tables/"
                f"macros.tex. Run scripts/make_paper_tables.py, or use a name "
                f"that exists.")
        return B.MACROS[name]

    # -- peeking ----------------------------------------------------------
    def peek_fig(self):
        return B._FIGN[0] + 1

    def peek_tbl(self):
        return B._TABN[0] + 1


# ---------------------------------------------------------------- the build
def _sections(kit):
    """Import every module named in ORDER and let it fill the flow."""
    on_disk = set(discovered())
    built = []
    for name in ORDER:
        if name not in on_disk:
            print(f"  MISSING SECTION: scripts/supp/{name}.py is in ORDER but "
                  f"not on disk")
            continue
        kit._section = name
        mod = importlib.import_module(f"supp.{name}")
        importlib.reload(mod)
        if not hasattr(mod, "content"):
            print(f"  scripts/supp/{name}.py has no content(k); skipped")
            continue
        mod.content(kit)
        built.append(name)
    for name in sorted(on_disk - set(ORDER)):
        print(f"  NOT BUILT: scripts/supp/{name}.py is on disk but not in "
              f"ORDER")
    return built


def build(out="paper/FLEX-UF-supp.pdf"):
    nfig, ntab, neq = _main_counters()
    tfig, ttab, teq = _tex_counters()
    print(f"  main paper ends at Figure {nfig}, Table {ntab}, "
          f"equation ({neq}); the supplement continues from there")
    if (tfig, ttab, teq) != (nfig, ntab, neq):
        print(f"  main.tex itself ends at Figure {tfig}, Table {ttab}, "
              f"equation ({teq}); supplementary.tex continues from those, "
              f"and the two documents number differently until main.tex and "
              f"build_pdf.py are brought back into step")

    PW, PH = letter
    M, GAP = 0.62 * inch, 0.28 * inch
    colw = (PW - 2 * M - GAP) / 2
    fullw = PW - 2 * M

    # The LaTeX source needs the same three starting counts, and typing them
    # into supplementary.tex would put a number in a file that nothing
    # regenerates. Emit them instead; supplementary.tex \inputs this.
    SUPP_TEX.mkdir(parents=True, exist_ok=True)
    (SUPP_TEX / "_counters.tex").write_text(
        "% Generated by scripts/build_supp_pdf.py. Do not edit.\n"
        "% The final figure, table and equation counts of paper/main.tex,\n"
        "% counted from its source, so that the supplement's numbering\n"
        "% continues rather than restarting.\n"
        f"\\setcounter{{figure}}{{{tfig}}}\n"
        f"\\setcounter{{table}}{{{ttab}}}\n"
        f"\\setcounter{{equation}}{{{teq}}}\n")

    tex = tex_order()
    if tex != ORDER:
        print(f"  ORDER and paper/supplementary.tex disagree. This file "
              f"builds {ORDER}; supplementary.tex inputs {tex}. The two "
              f"documents will not number their sections alike until they "
              f"match.")

    kit = Kit(colw, fullw)
    built = _sections(kit)

    # The same sections, rendered to LaTeX. paper/supplementary.tex inputs eight
    # files that had never been written, so the LaTeX submission carried an
    # empty supplement. Writing them by hand would have left two sources to keep
    # in step; this renders them from the one that already exists.
    import importlib
    from supp_tex import TexKit
    for _name in ORDER:
        try:
            _mod = importlib.import_module(f"supp.{_name}")
            importlib.reload(_mod)
            TexKit(_name).render(_mod.content)
        except Exception as _e:
            print(f"  LaTeX for {_name}: {type(_e).__name__}: {_e}")


    # Title block, measured rather than guessed. A band taller than what it
    # holds leaves both columns of page 1 short, which is the defect the main
    # paper's own template comments complain about.
    title = [
        Paragraph("Where to Stop:<br/>Tile-Adaptive Early Exit in a Learned "
                  "Image Decoder", B.TITLE),
        Paragraph("Supplementary Material", SUBTITLE),
        Paragraph("Anonymous CVPR submission &nbsp;&nbsp;·&nbsp;&nbsp; "
                  "Paper ID ****", B.AUTH),
    ]
    band = sum(f.wrap(fullw, PH)[1] + f.getSpaceAfter() for f in title) + 6

    doc = BaseDocTemplate(str(R / out), pagesize=letter,
                          leftMargin=M, rightMargin=M,
                          topMargin=0.7 * inch, bottomMargin=0.7 * inch,
                          title="Where to Stop: Tile-Adaptive Early Exit in a "
                                "Learned Image Decoder (Supplementary "
                                "Material)",
                          author="FLEX-UF")
    H = PH - 1.4 * inch

    def frames(top, ids):
        """A full-width band `top` points tall, then two columns beneath it."""
        h = H - top
        return [Frame(M, PH - 0.7 * inch - top, fullw, top, id=ids[0],
                      leftPadding=0, rightPadding=0, topPadding=0,
                      bottomPadding=0),
                Frame(M, 0.7 * inch, colw, h, id=ids[1], leftPadding=0,
                      rightPadding=0, topPadding=0, bottomPadding=0),
                Frame(M + colw + GAP, 0.7 * inch, colw, h, id=ids[2],
                      leftPadding=0, rightPadding=0, topPadding=0,
                      bottomPadding=0)]

    f_l = Frame(M, 0.7 * inch, colw, H, id="l", leftPadding=0, rightPadding=0,
                topPadding=0, bottomPadding=0)
    f_r = Frame(M + colw + GAP, 0.7 * inch, colw, H, id="r", leftPadding=0,
                rightPadding=0, topPadding=0, bottomPadding=0)

    def num(canv, d):
        canv.saveState()
        canv.setFont("Times-Roman", 8)
        canv.setFillColor(colors.HexColor("#666666"))
        canv.drawCentredString(PW / 2, 0.42 * inch, str(canv.getPageNumber()))
        canv.restoreState()

    doc.addPageTemplates([
        PageTemplate(id="first", frames=frames(band, ["ban", "l1", "r1"]),
                     onPage=num),
        PageTemplate(id="rest", frames=[f_l, f_r], onPage=num),
        PageTemplate(id="wide", frames=frames(B.WIDE_BAND,
                                              ["wide", "lw", "rw"]),
                     onPage=num)])

    story = list(title) + [NextPageTemplate("rest"), FrameBreak()] + kit.flow
    doc.build(story)

    if kit.used_results:
        print("  results read: "
              + ", ".join(dict.fromkeys(kit.used_results)))
    if kit.bad_glyphs:
        print("  NO GLYPH IN Times-Roman, these will print as black boxes: "
              + ", ".join(f"{c} ({u}, in {w})"
                          for c, u, w in sorted(kit.bad_glyphs,
                                                key=lambda x: x[1])))
    if B.UNEXPANDED:
        print("  UNEXPANDED MACROS (run make_paper_tables.py): "
              + ", ".join(sorted(B.UNEXPANDED)))
    print(f"  sections built: {', '.join(built) or '(none)'}")
    print(f"  -> {out}")


if __name__ == "__main__":
    build(sys.argv[1] if len(sys.argv) > 1 else "paper/FLEX-UF-supp.pdf")
