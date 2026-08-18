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
