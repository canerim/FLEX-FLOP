# Narrative and evidence decisions

The manuscript now centres on **spatial early exit within DCVC-UF**. The
opening moves from uniform synthesis depth to three concrete obstacles:
intermediate feature compatibility, tile boundaries and decision cost. The
method addresses those with nested execution, pointwise adaptation, shared
repair and a stopping predictor. Router-versus-dither analysis evaluates one
component of that system rather than defining the entire paper's story.

Only DCVC-UF numerical experiments appear in the active manuscript and plot
bundle. Other reconstruction networks have been removed from the results.
Adaptive-codec and patch-exit papers remain relevant related work, with no
unsubstantiated claim to have invented content-dependent computation.

The independent 2/4/6/8/10/12 study tests a second question: how much is lost
by sharing one representation? Its training status, exact parameter counts
and evaluation controls are explicit. A separate analytical sensitivity
plot exposes assumptions for pending runtime measurements; it does not
fabricate quality, bitrate, speedup observations or uncertainty intervals.

## Narrative decisions

The results proceed from the value of uniform depth to the total early-exit
saving, the extra benefit of spatial allocation, its survival under a common
measured loss cap, and the encoder work needed to obtain a decision. The
paper finishes with the controlled depth study and the evidence needed for
full codec execution. Detailed provenance audits sit in the supplement.

Primary examples informing this structure include
[Shallow Decoders](https://openaccess.thecvf.com/content/ICCV2023/html/Yang_Computationally-Efficient_Neural_Image_Compression_with_Shallow_Decoders_ICCV_2023_paper.html)
for connecting a decoder design to compute asymmetry, and
[DCVC-RT](https://arxiv.org/abs/2502.20762) for treating execution costs as
part of the research question. Their prose and figures are not copied.

## Evidence-to-claim map

| Claim | Direct evidence | Boundary |
|---|---|---|
| Uniform shared-exit depth has a measurable quality cost | Archived uniform-map error columns and a verified released-weight anchor | Padded RGB; not independent depth-model quality |
| Conditional tile depths reduce modelled synthesis work | Recorded maps and cost vector; actual conditional implementation | Router, signalling and system latency are separate |
| Learned allocation adds a budget-dependent margin | Paired common-cohort router–dither differences; sequence bootstrap | Per-frame source calibration; one codec/router checkpoint |
| Spatial placement contains useful table information | Exact expected permutation at unchanged exit histogram | Table MSE, not new mixed reconstructions |
| Quality targets need a common evaluation reference | Code trace, checkpoint comparison, exported losses | Historical execution hashes were not retained; both anchors need replay |
| Source-error information has a measurable acquisition cost | Archived table-construction stage summaries | Pooled upper medians; no trial uncertainty or current end-to-end claim |
| D2/D4/D6 are designed to measure a capacity frontier | Training in progress | Final RD/time evidence is pending; same-recipe D12 needed |
| A six-expert MLP can save net runtime | Proposed ablations and implementation path | Hypothesis; not an established result |

## Figure decisions

The vector figures use a common 183 mm width, restrained colours, distinct
markers, embedded fonts, units and panel letters. Confidence bands identify
their resampling unit. Raw observations, cost models and proposed mechanisms
are labelled separately. Diagram geometry shows tensor resolution, active
tile populations, shared computation, independent codec ownership or stream
components. It does not encode invented throughput.

These export choices follow the [Nature Research Figure Guide](https://research-figure-guide.nature.com/figures/building-and-exporting-figure-panels/):
editable vector assets, final-size legibility and consistent typography.
They are production choices, not a claim of journal acceptance or scientific
equivalence. No external paper figure or prose was copied.

## Remaining evidence needed for a stronger submission

The decisive next measurements are common-reference cropped reconstructions,
held-out control calibration, full coded bytes and paired end-to-end latency.
For the independent bank, add the same-recipe D12 control before attributing
released-model gaps to depth, and quantify patching/grouping overhead before
expanding the MLP. The detailed priority order is in `ABLATION_PLAN_TR.md`.
