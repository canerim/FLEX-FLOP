# Figure editorial review — 28 September 2026

The first work block began at 05:48:59 UTC. Actual Figures 1 and 2 are protected by SHA-256; their PNGs, generation provenance and manuscript figure captions have not been edited.

## Reader-facing changes

- Replace the main paper's repeated two-panel delivered frontier with a single-column paired-increment forest plot. The complete four-policy frontier stays in the supplement. Same values, intervals, source-aware protocol and 265 cases.
- Add an 89 × 64 mm execution diagram derived from the recorded videoSRC05/QP32 map. All 40 padded tile coordinates are retained at every stage. Active counts 40/27/7/3 and exit counts 13/20/4/3 are data-derived. Colour identifies the current executed stage; pale cells already exited. These are logical tile-stage counts, not GPU kernel calls or timing.
- Redraw the spatial comparison as four wider panels: source luma, router, Bayer and source-informed search. The constant depth-eight map is removed from the picture, with its measured values retained in the caption. No example or outcome was reselected.
- Increase the main trade-off's label, tick and marker sizes. Highlight the 0.1 dB operating point in both panels. Keep paired intervals and the shallow-exit ceiling.
- Apply a 6.5 pt text floor and at least 7 pt axis / 7.5 pt panel-title sizes to all generated vector figures. Top headings and long bottom notes move to exported captions; blank canvas is trimmed without shrinking physical text or axes. Analytical assumptions stay visible.
- Use one ordered six-depth palette in diagrams, maps and the independent-depth RD plots. Router/search/dither/uniform remain teal/violet/amber/slate in policy comparisons, with marker and line-style cues.
- Compact the source-reference diagram from 103 to 78 mm, codec-bank diagram from 111 to 86 mm, and encoder decision-cost schematic from 116 to 92 mm. The fixed coordinate layouts preserve connectivity and counts; text remains physical-size controlled.
- Shorten long rotated labels and separate count labels from markers. Keep original numerical precision in bundled data.

## Visual review

Reviewed individual renders of all figure families, including paired confidence intervals, low-effect repair results, region geometry, source features, coverage, released anchor, mean/Q90 control, accounting scopes and analytical scenarios. Reviewed manuscript pages containing the protected mechanism, new single-column diagram and main results at page scale. The two new main-paper elements preserve the eight-page body in the first layout build.

A bounding-box overlap screen found one candidate in the qualitative panel (0801 versus an inactive tick). Visual inspection confirms the image axis has ticks disabled; there is no printed collision. Hash verification correctly rejected stale plot evidence after a label-script edit; affected research plots are being regenerated before the final integrity and isolated-reproduction checks.

## Scope

No experimental rows, selection rules, image appearance, confidence intervals, metric anchors or inference results were modified. Raster dataset panels continue to use the bundled source thumbnail / predeclared reconstruction windows. The new execution diagram is native vector artwork, not generated bitmap imagery. Final PDF pagination and reproduction are checked again after prose revision.

## Figure-phase verification

Checked 2026-09-28T06:17:25.182102+00:00: all 34 vector figure sets pass; minimum exported text size 6.5 pt, no text outside the canvas, 29 manuscript/supplement panel-caption mappings verified. All 117 PDF/SVG/PNG artifacts reproduce byte-identically from bundled data and scripts in isolation. Main and supplement compile with embedded fonts, no Type 3 fonts, unresolved references or overfull boxes. Main body remains eight pages. Both protected raster hashes remain unchanged. Full reports are figure_phase_verification.json and figure_phase_reproduction.json.
