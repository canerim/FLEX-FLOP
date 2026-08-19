# Paper source

`main.tex` is written for the official CVPR author kit. It looks for `cvpr.sty`
and falls back to `cvpr_fallback.sty`, a minimal stand-in, so the source
compiles on a machine without the kit installed.

**Nothing numeric in `main.tex` is typed.** Tables live in `tables/*.tex` and
inline scalars are macros in `tables/macros.tex`, both generated from the
measurement files:

```bash
python scripts/make_paper_tables.py     # regenerates tables/ from results/
python scripts/paper_figures.py         # copies and crops figures into figures/
cd paper && latexmk -pdf main.tex
```

Regenerate after any re-measurement. A paper whose numbers are retyped from a
terminal disagrees with its own repository within a week; this one cannot.

## What is where

| file | contents |
|---|---|
| `main.tex` | the paper |
| `refs.bib` | bibliography |
| `tables/` | generated LaTeX tables and `\newcommand` macros |
| `figures/` | figures, copied from `docs/figures/` by `paper_figures.py` |
| `theory.tex` | earlier derivations, kept for reference, not part of the paper |

## Length

The reportlab build (`scripts/build_pdf.py`, which is what produces
`FLEX-UF.pdf` on a machine with no LaTeX) currently runs to **ten body pages
plus one of references**. CVPR allows eight body pages. This is deliberate for
now: the document doubles as the project's technical report, and every section
in it is a measurement someone asked for. A submission still has to lose two
pages. The cheapest, in the order we would cut them:

1. **§5.6 the free baseline** and **§5.7 partial signalling** → supplementary,
   leaving one paragraph each in §5.5. Together about two pages with their two
   figures and two tables. Note that this is where the strongest single result
   lives (a parameter-free rule that beats a trained head at every rate, and a
   half map that
   beats a complete one), so cutting it is a positioning decision, not a
   tidying one.
2. **§5.8 does the map have to be recomputed** → supplementary. Half a page.
3. **§5.10 the right ladder depends on the budget** → merge into §5.4, which
   already makes the saturation argument the table illustrates.
4. One of Figure 9 (the trade-off curve) and Figure 12 (floor/saturation). They
   are two views of the same structure, and the caption of either can carry
   both.

What we would not cut: the operating-structure section, the seam sections, or
the wall-clock section. Those three are the parts a reader cannot reconstruct
from the headline number.

`main.tex` compiles shorter than the reportlab build because the real CVPR
template is denser; measure it there before deciding what to remove.
