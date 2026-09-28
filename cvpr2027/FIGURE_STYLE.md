# FLEX-UF figure system

The figures share a 183 mm canvas width, Liberation Sans type, embedded
TrueType PDF fonts and editable SVG text. Quantitative plots and exact
execution diagrams are generated from bundled source tables with Matplotlib.
The protected opening overview and the shared-exit mechanism are separate AI-generated conceptual rasters,
labelled as such in their captions; neither contains experimental reconstructions.
Their prompts and provenance are recorded in `IMAGEGEN.md`. The data figures use archived source-luma thumbnails and explicitly
identified, predeclared RGB reconstruction windows as raster content. Diagrams use tensor planes, nested block
strips, active-tile grids, feature joins, expert queues and coded-stream
segments to explain the computation.

| Meaning | Colour | Additional cue |
|---|---|---|
| Learned router / adaptive path | Teal `#008A96` | Square marker |
| Source-informed search | Muted violet `#79679A` | Circle marker |
| Bayer dithering | Amber `#BF783D` | Triangle marker |
| Uniform depth | Slate `#81909C` | Diamond marker |
| Text and axes | Ink `#20313E` | Direct labels and explicit units |
| Grid | Pale grey `#E5EAED` | Fine horizontal guides |

Depth diagrams and maps use an ordered six-colour scale: `#D8ECE7`, `#A7D2C9`, `#70B6AB`,
`#34968F`, `#267382`, `#244B68` for 2/4/6/8/10/12 blocks. The current
shared-exit maps use only the last four colours. Every map has a numerical
key, so colour is not the sole carrier of depth information.

Measured reconstructions use filled markers. Bootstrap bands are labelled
with their resampling unit. Analytical runtime scenarios use dashed lines,
open markers and an explicit **not measured** heading. Training/planned
status is printed in the depth diagram; no inferred performance is encoded
by network width, queue occupancy or diagram geometry.

Each plot states its reference and support: nominal source-table targets,
cropped e15 delivered loss, or padded released-reference uniform quality.
These are deliberately distinct axes. Connecting discrete operating points
guides the eye; it creates no additional observation.

Export dimensions, font sizes and text bounds are recorded in each figure
folder's `layout_audit.json`. Artifact hashes and an isolated CPU rebuild
support reproducibility. PDF page layout is also reviewed after compilation;
passing a text-bound check alone does not establish publication quality.

The September 28 revision preserves Figure 1 byte for byte. Figure 2 follows
its layered scientific-illustration style while depicting the implemented
shared-exit connectivity: stem and QP feed the MLP; beta adjusts its output
scores; all four exits reach the common feature canvas. Its synthetic scene
is not a qualitative codec result.

Plots use 6.5 pt body/axis text, 6 pt ticks, bold 8 pt lower-case panel
letters, restrained strokes and open white space at 183 mm width. Duplicate
poster headings are moved to captions and unused top canvas is trimmed
without changing physical axis, text or marker sizes. Explicit analytical
assumptions remain visible. Smaller exceptional annotations are recorded
in the layout audits; figures are inspected at compiled page scale.

Design references: Nature's [panel preparation guide](https://research-figure-guide.nature.com/figures/building-and-exporting-figure-panels/)
and [figure specifications](https://research-figure-guide.nature.com/figures/preparing-figures-our-specifications/).
These inform visual decisions; they are not a claim of journal endorsement
or a replacement for CVPR submission requirements.
