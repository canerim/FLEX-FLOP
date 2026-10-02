# RegLIC: Region-Adaptive Learned Image Compression

The active manuscript studies **spatially adaptive DCVC-UF computation**.
Its evaluated mechanism is shared-latent early exit; the complementary
ClassSR-inspired 2/4/6/8/10/12 independent-codec bank is a controlled study
with ongoing training and prospective routing. Unfinished experiments are not assigned
invented PSNR, bitrate or runtime results.

- Main document: [main.tex](main.tex) / [main.pdf](main.pdf).
- Supplement: [supplement.tex](supplement.tex) / [supplement.pdf](supplement.pdf).
- Overleaf and reproduction instructions: [OVERLEAF.md](OVERLEAF.md).
- Context-preserving versus learned-repair control: [CONTEXT_CONTROL_TR.md](CONTEXT_CONTROL_TR.md).
- Prioritised ablations: [ABLATION_PLAN_TR.md](ABLATION_PLAN_TR.md).
- Research assessment and reviewer questions: [REVIEWER_NOTES_TR.md](REVIEWER_NOTES_TR.md).
- Generated conceptual overview and full prompts: [IMAGEGEN.md](IMAGEGEN.md).
- Primary-source related-work audit: [RELATED_WORK_AUDIT.md](RELATED_WORK_AUDIT.md).
- Figure palette and visual rules: [FIGURE_STYLE.md](FIGURE_STYLE.md).
- Runtime scope: [RUNTIME_AUDIT_TR.md](RUNTIME_AUDIT_TR.md).
- Metric correction: [METRICS.md](METRICS.md). The archived `db_rgb` field
  measures unclipped YCbCr 4:4:4 MSE-ratio loss; current plots use that label.
- Sequence-disjoint scalar-calibration diagnostic: supplement S5 and
  [plot data](data/crossfit20260927/analysis.json). This uses archived padded
  error tables; it is not a new final-reconstruction or latency benchmark.
- Architectural MAC accounting: [data](data/depthmacs20260927/analysis.json)
  and the supplementary arithmetic-denominator figure distinguish synthesis, neural entropy recovery
  and encoder-plus-reconstruction costs. These are not runtime measurements.

Select **main.tex** and **pdfLaTeX** in Overleaf. The active text is in
`sec/revision_*.tex`; old sections remain inactive historical material.
The original template documentation is preserved in
[AUTHOR_KIT_README.md](AUTHOR_KIT_README.md).

## Fixed epoch-20 and epoch-30 actual-byte studies

The supplement includes 100 DIV2K centre crops of 512 × 512 pixels, four
models (D2/D4/D6 and released D12), and five QPs: 2,000 independent
encode/decode checks. The shallow models are unfinished epoch-20/105
checkpoints; released D12 has a different training history. At 0.2 actual
payload bpp on 99 commonly supported images, D6 exceeds D2 by 0.1834 dB
RGB PSNR. True-byte curves, feature associations and whole-crop allocation
controls are in `data/research20260927`. The allocation control is an
optimistic diagnostic, not a trained patch router.

The predeclared epoch-30 milestone repeats all 2,000 cases. The same 99-image,
0.2-payload-bpp comparison gives D6−D2 = 0.1071 dB and D6−D4 = −0.0045 dB
(95% paired interval −0.0172 to +0.0087). The ordering is not stable during
unfinished training. Both milestones remain in the supplement; geometry
controls and plotted epoch-20 curves retain their original checkpoints.
`data/refresh20260927/depth_milestone_audit.json` records the paired check.

`python scripts/build_research_figures.py` renders the portable vector
figures. `python scripts/reproduce_figures.py` checks byte-identical
reproduction from bundled data without checkpoints, source images or GPU.
The suite contains portable vector figures and separately identified
conceptual AI illustrations. The current Figure 1 is a compact, text-free
RegLIC illustration of spatially variable synthesis depth. Figure 2
shows a full-size independent-depth bank on page 2; a separate early-exit
system diagram appears in the method section. The remaining main figures
include an active-tile trace, measured budget trajectories and eight
source groups with recorded maps.
No tables are typeset in the main paper or supplement: evidence is plotted.
`build_editorial_figures.py` renders the four added evidence plots. Numerical
exports, 22 macros and the paired milestone audit remain checked against
bundled JSON; unused table-form exports are retained only as audit artifacts.

## Fixed-policy and spatial-region controls

The fixed-Q90 placement intervention completes 424 map cases on all 53 QP32
frames, preserving depth histograms within equal valid-tile extents. Original
router placement gains 0.01393 dB against the mean MSE of three shuffled maps;
dithering gains 0.00036 dB. The intervals condition on those sampled
permutations. This supports spatial placement value without establishing a
net-runtime or external-test advantage. Complete records are bundled as
`data/research20260927/shared_crossfit_qp32/placement_replay.json`.

The fixed-control replay, now summarised in main Table 2, retains all 53 QP32 frames and all 318 fixed-policy outputs. Both paired router-minus-dither confidence intervals include zero. Fresh CPU router inference reproduces all 1,765 tile choices under each of the two fixed controls; log-probabilities themselves are not bit-identical across the historical GPU and CPU paths.

The independent-region control retains 240 model–image–QP cases, with actual research-stream bytes and independent decoding. Moving padding from the source image to the latent recovers 0.2652 dB for D6 at 0.2 payload bpp, but independently coded halo32 regions still lose 1.0262 dB against full-frame D6. These are interim CPU geometry results, not native CUDA timings. A source-derived stream-dependency diagram explains why encoder stage times must not simply be added.

The full QP32 released-anchor replay compares stock D12 with the same 53 sources and support: mean RGB PSNR 34.71763 dB versus 34.70855 dB for e15 full-frame. Fixed-map region coalescing completes 800 independently decoded outputs; frozen-weight adapter/repair interventions complete 212 reconstructions. These are scoped controls, not final model-bank routing or native runtime results. The two-page Turkish research brief is maintained in the parent repository under `docs/research/2026-09-27-six-hour/RESEARCH_BRIEF_TR.pdf`.
