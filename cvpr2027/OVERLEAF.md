# Current manuscript

Overleaf main document: **`main.tex`**, compiler **pdfLaTeX**. This repository's
root is the `cvpr2027/` subtree of the FLEX-PLUS research workspace.

The 27 September 2026 evidence revision contains a DCVC-UF-only early-exit manuscript,
twelve editable vector figure sets across the main paper and protocol
supplement, generated numerical macros, paired source tables and hash
manifests. Build with:

```bash
latexmk -pdf -interaction=nonstopmode -halt-on-error main.tex
latexmk -pdf -interaction=nonstopmode -halt-on-error supplement.tex
```

The active sections are `sec/revision_*.tex`. Earlier `sec/0_abstract.tex` through
`sec/5_conclusion.tex` remain in the repository for reference and are not included
by `main.tex`. The independent D2/D4/D6 training is ongoing; its intermediate
validation values are not presented as final manuscript results.

- Figure atlas: `figs/refresh20260927/figure_atlas.pdf`.
- Protocol and planned codec-bank atlas: `figs/extended20260927/extended_atlas.pdf`.
- Stand-alone protocol supplement: `supplement.tex` / `supplement.pdf`.
- Prioritised Turkish ablation plan: `ABLATION_PLAN_TR.md`.
- Claim assessment and reviewer questions: `REVIEWER_NOTES_TR.md`.
- Individual editable figures: PDF and SVG in the same directory.
- Data and generated macros: `data/refresh20260927/`.
- Full captions and measurement scope: `figs/refresh20260927/captions.json`.
- Portable figure generators: `scripts/build_figures.py` and
  `scripts/build_extended_figures.py`. Both consume the bundled data; no GPU,
  checkpoints or source dataset are required to regenerate the plots.
- Palette, typography and evidence styles: `FIGURE_STYLE.md`.
- Runtime scope and required full-bitstream benchmark: `RUNTIME_AUDIT_TR.md`.
- Evidence and editorial decisions: `WRITING_AND_EVIDENCE.md`.

To regenerate figures on a CPU Python environment with Liberation Sans fonts:

```bash
python -m pip install -r scripts/figure-requirements.txt
python scripts/build_evidence_tables.py
python scripts/build_figures.py
python scripts/build_extended_figures.py
python scripts/audit_cap_influence.py > data/refresh20260927/cap_influence_audit.json
python scripts/verify_bundle.py --check-pdfs
```

The original table extraction and checkpoint audits remain in the parent
FLEX research repository under `scripts/paper_refresh_data.py` and
`scripts/audit_*`. Regenerating those audits requires the archived raw
records/checkpoints. Source hashes captured during this revision are
inspection provenance, not retrospective proof of historical execution.
`figure_reproduction_report.json` records a separate reproduction using only
the bundled data and scripts: all 38 PDF/SVG/PNG artifacts were byte-identical.
`verification_report.json` records data consistency, PDF references, font
embedding and page checks. The checks do not rerun the historical codec.

This is a research draft. The plotted router sweep uses per-frame source
calibration; MAC savings exclude router/signalling overhead; historical timing
does not establish end-to-end acceleration. These boundaries are stated in the
abstract, methods and captions rather than hidden in plotting notes.
