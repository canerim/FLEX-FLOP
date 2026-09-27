# Narrative and evidence decisions

The manuscript asks what learned spatial allocation contributes beyond using
less decoder capacity. That question gives the introduction, the controls
and the primary figure the same organising principle. The current results
belong to the shared-latent e15 system. The independent depth-reduced codec
bank is an ongoing experiment with a separate representation and protocol.

## How the narrative was revised

The opening moves from the reconstruction problem to the ambiguity in a
headline compute saving, then introduces the experiment that resolves part
of that ambiguity. Contributions name a mechanism, an experimentally
isolated quantity and a measurement finding. They do not convert planned
experiments into results. Each results paragraph states its finding before
the protocol details needed to interpret it.

The related-work section is organised by comparison axis: spatial capacity,
adaptive compression, complete codec competition, practical execution and
perception. Each group explains which design dimension is shared and what
our evaluation measures. It does not rely on an unsupported “first” claim.

Three primary-source examples informed this organisation, without borrowing
their prose. [Shallow Decoders](https://openaccess.thecvf.com/content/ICCV2023/html/Yang_Computationally-Efficient_Neural_Image_Compression_with_Shallow_Decoders_ICCV_2023_paper.html)
connects a decoder-design choice to the compute asymmetry it addresses.
[DCVC-RT](https://arxiv.org/abs/2502.20762) makes execution constraints part of
the problem formulation. [What Matters in Practical Learned Image Compression](https://arxiv.org/abs/2605.05148)
organises design choices around quality and measured device runtime. The
corresponding editorial choice here is to tie every claimed improvement to
its comparison and measurement scope.

## Evidence-to-claim map

| Claim | Direct evidence | Boundary |
|---|---|---|
| Conditional tile depths reduce modelled synthesis work | Recorded maps and cost vector; actual conditional implementation | Router, signalling and system latency are separate |
| Learned allocation adds a budget-dependent margin | Paired common-cohort router–dither differences; sequence bootstrap | Per-frame source calibration; one codec/router checkpoint |
| Spatial placement contains useful table information | Exact expected permutation at unchanged exit histogram | Table MSE, not new mixed reconstructions |
| Quality targets need a common evaluation reference | Code trace, checkpoint comparison, exported losses | Historical execution hashes were not retained; both anchors need replay |
| Source-error information has a measurable acquisition cost | Archived table-construction stage summaries | Pooled upper medians; no trial uncertainty or current end-to-end claim |
| D2/D4/D6 quantify a useful capacity frontier | Training in progress | Final RD/time evidence is pending; same-recipe D12 needed |
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
