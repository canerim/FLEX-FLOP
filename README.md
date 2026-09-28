# FLEX-UF: Spatially Adaptive Computation for DCVC-UF

The active manuscript studies **spatially adaptive DCVC-UF computation**.
Its evaluated mechanism is shared-latent early exit; the complementary
ClassSR-inspired 2/4/6/8/10/12 independent-codec bank is a controlled study
with ongoing training and prospective routing. Unfinished experiments are not assigned
invented PSNR, bitrate or runtime results.

- Main document: [main.tex](main.tex) / [main.pdf](main.pdf).
- Supplement: [supplement.tex](supplement.tex) / [supplement.pdf](supplement.pdf).
- Overleaf and reproduction instructions: [OVERLEAF.md](OVERLEAF.md).
- Prioritised ablations: [ABLATION_PLAN_TR.md](ABLATION_PLAN_TR.md).
- Research assessment and reviewer questions: [REVIEWER_NOTES_TR.md](REVIEWER_NOTES_TR.md).
- Generated conceptual overview and full prompts: [IMAGEGEN.md](IMAGEGEN.md).
- Figure palette and visual rules: [FIGURE_STYLE.md](FIGURE_STYLE.md).
- Runtime scope: [RUNTIME_AUDIT_TR.md](RUNTIME_AUDIT_TR.md).
- Metric correction: [METRICS.md](METRICS.md). The archived `db_rgb` field
  measures unclipped YCbCr 4:4:4 MSE-ratio loss; current plots use that label.
- Sequence-disjoint scalar-calibration diagnostic: supplement S5 and
  [plot data](data/crossfit20260927/analysis.json). This uses archived padded
  error tables; it is not a new final-reconstruction or latency benchmark.
- Architectural MAC accounting: [data](data/depthmacs20260927/analysis.json)
  and supplement Figure S8 distinguish synthesis, neural entropy recovery
  and encoder-plus-reconstruction costs. These are not runtime measurements.

Select **main.tex** and **pdfLaTeX** in Overleaf. The active text is in
`sec/revision_*.tex`; old sections remain inactive historical material.
The original template documentation is preserved in
[AUTHOR_KIT_README.md](AUTHOR_KIT_README.md).

## Frozen epoch-20 actual-byte study

The supplement now includes all100 DIV2K centre512 crops × D2/D4/D6/releasedD12 × five QPs (2,000 independent encode/decode checks). D2/D4/D6 are unfinished epoch20/105 checkpoints, not final models; releasedD12 has different training history. At0.2 actual payload bpp on99 common supported images, D6−D2 is0.1834dB RGB PSNR. The expanded coverage, actual-versus-estimated rate, feature associations and whole-crop allocation controls are in `data/research20260927`. The latter is an optimistic allocation diagnostic, not a trained patch router.

`python scripts/build_research_figures.py` renders the portable new vector figures. `python scripts/reproduce_figures.py` verifies bit-identical artifact reproduction from the bundled data, without training checkpoints, source images or GPU. The full suite currently contains23 vector figure sets; the two conceptual AI illustrations remain separately identified. Figure 1 is preserved exactly in the September 28 redesign.
