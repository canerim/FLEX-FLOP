# Narrative and evidence decisions

The paper's question is **where additional DCVC-UF synthesis depth earns its
cost**. The evaluated mechanism is spatial early exit from one shared coded
representation. Independent 2/4/6/8/10/12-block codecs are a complementary
study of specialisation versus reuse, with ongoing training and prospective
routing. They are not a bank-to-exit cascade.

## Reading order

1. The introduction starts with two image regions that benefit differently
   from another block pair, then distinguishes selecting a codec before
   encoding from stopping after recovering a common latent.
2. The method explains the full-frame prefix, active-tile execution, exit
   alignment and assembled reconstruction. The new single-column diagram
   turns a recorded map into actual active counts, 40 → 27 → 7 → 3.
3. The experiments separate global truncation, blind depth mixtures and
   content placement. They move from source-calibrated allocation to a
   common delivered-loss cap, then to fixed-control reconstruction replay.
4. Fixed-control uncertainty, adapter/repair interventions and the small
   stream-only check appear in the main paper, rather than being buried
   behind the favourable allocation curve.
5. The independent bank and discussion state the next falsifiable tests.
   Detailed launch provenance, input branches and target formulas belong
   in the supplement, not in the opening argument.

The title makes spatially adaptive early exit explicit. The main document
retains the CVPR template and an eight-page body; prose is edited to fit
rather than reducing type or margins. Only DCVC-UF numerical comparisons
appear in the active manuscript.

## Evidence-to-claim map

| Claim | Direct evidence | Boundary |
|---|---|---|
| Synthesis depth has a measurable reconstruction value | Actual uniform-map reconstructions at 6/8/10/12 blocks | Padded equal-channel YCbCr 4:4:4 against released full-frame; not independent-model quality |
| Conditional maps omit synthesis work | Active-tile execution and per-exit arithmetic | Synthesis MACs exclude entropy, routing and system latency |
| Content placement adds a budget-dependent allocation margin | Paired router–dither comparisons and common delivered-cap selection | Per-source calibration; finite recorded candidates and fixed weights |
| Fixing the control changes the conclusion | 318 actual QP32 outputs, with paired saving intervals crossing zero | Development-used sequence folds; two cross-fold similar-content candidates |
| The full-depth anchor has measurable drift | Aligned 53-frame QP32 released/e15 replay | CPU FP32, image-pad256, explicit RGB conversion; not the native CUDA path |
| Adapters carry a large learned correction | 212 fixed-weight component interventions | Removing a module is not retraining without it; repair's small effect needs cost testing |
| One entropy payload supports different maps | Eight stream-only CPU cases; same inner bytes and exact reconstruction | Two sources, four maps each; research headers and map bytes are charged |
| Specialisation has a depth frontier and spatial overhead | Frozen epoch-20 true-byte curves and fixed-map geometry controls | Interim models; released D12 has a different training history |
| A six-expert selector saves net runtime | Proposed study only | Requires matched training, achieved RD and complete execution measurements |

Main-table numbers are generated from bundled JSON. Frame-bootstrap
intervals are conditional on fixed weights and fitted controls; they do not
cover training-seed, calibration-fit or external-domain uncertainty.
Retrospective quality selection is described as source-aware, with its
candidate-acquisition cost excluded from the synthesis MAC quantity.

## Visual decisions

Figures 1 and 2 and their captions are protected. Other vector figures use
consistent policy colours, an ordered depth palette, explicit units and
embedded fonts. The main active-tile diagram and paired-cap plot use an
89 mm column; wider multi-panel figures use 183 mm. The exported text floor
is 6.5 pt, with larger axes and panel titles. Redundant on-figure prose is
moved into captions. Measured points, architectural counts and analytical
assumptions remain visibly distinct.

The scientific plots and native diagrams follow the production principles
of the [Nature figure guide](https://research-figure-guide.nature.com/figures/building-and-exporting-figure-panels/):
editable vector artwork, consistent typography and final-size legibility.
This is a production standard, not a claim of journal acceptance. Conceptual
AI artwork is identified in its captions and in `IMAGEGEN.md`; experimental
images and plotted observations are not generated illustrations.

## Next research priorities

The main remaining tests are content-grouped untouched calibration/evaluation,
native stream correctness followed by paired total latency, and equal-budget
adapter/repair retraining. A same-recipe D12 is needed before isolating depth
from released-model training history. An independent bank must first account
for patching, entropy boundaries and grouping before expanding its predictor.
See `ABLATION_PLAN_TR.md` for controls and `RELATED_WORK_AUDIT.md` for the
primary-source comparison.
