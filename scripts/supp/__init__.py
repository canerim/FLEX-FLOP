"""Sections of the supplementary material, one module per section.

Each module here exposes a single function, `content(k)`, which fills the
document by calling methods on the toolkit `k`. Nothing else in the module is
read. The full toolkit API is documented in `scripts/build_supp_pdf.py`; the
short version is:

    k.h1 / k.h2 / k.h3     headings, auto-numbered A, B, A.1, A.2
    k.par / k.bullets      body text
    k.note                 a small-print provenance line
    k.tbl / k.rows         a generated table, or one built from data
    k.fig / k.figwide      a figure at column or full width
    k.eq                   a display equation
    k.J / k.macro          a results/ file, or a value from macros.tex

A module is built only if its name appears in `ORDER` in
`scripts/build_supp_pdf.py`, and its name is also its LaTeX slug: the module
`foo.py` pairs with `paper/supp/foo.tex`, which `paper/supplementary.tex`
inputs at the same position.

Two rules hold everywhere in this package. Every number comes from a file in
`results/`, named in the text or in a `k.note` beside the table it appears in.
Where `paper/tables/macros.tex` defines a macro for a number, write the macro
(`\\Ceiling`, not `39.1`) and let `k.par` expand it.
"""
