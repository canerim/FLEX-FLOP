# Current manuscript

Overleaf main document: **`main.tex`**, compiler **pdfLaTeX**. This repository's
root is the `cvpr2027/` subtree of the FLEX-PLUS research workspace.

The 27 September 2026 evidence revision contains a complete initial narrative,
six editable vector figure sets, generated numerical macros, paired source
tables and a source-hash manifest. Build with:

```bash
latexmk -pdf -interaction=nonstopmode -halt-on-error main.tex
```

The active sections are `sec/revision_*.tex`. Earlier `sec/0_abstract.tex` through
`sec/5_conclusion.tex` remain in the repository for reference and are not included
by `main.tex`. The independent D2/D4/D6 training is ongoing; its intermediate
validation values are not presented as final manuscript results.

- Figure atlas: `figs/refresh20260927/figure_atlas.pdf`.
- Individual editable figures: PDF and SVG in the same directory.
- Data and generated macros: `data/refresh20260927/`.
- Full captions and measurement scope: `figs/refresh20260927/captions.json`.
- Figure generators in the parent research repository:
  `scripts/paper_refresh_data.py` and `scripts/paper_refresh_figures.py`.

This is a research draft. The plotted router sweep uses per-frame source
calibration; MAC savings exclude router/signalling overhead; historical timing
does not establish end-to-end acceleration. These boundaries are stated in the
abstract, methods and captions rather than hidden in plotting notes.
