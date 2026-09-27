# FLEX-UF figure system

The figures share a 183 mm canvas width, Liberation Sans type, embedded
TrueType PDF fonts and editable SVG text. Quantitative plots and exact
execution diagrams are generated from bundled source tables with Matplotlib.
The new opening overview is a separate AI-generated conceptual raster,
labelled as such in the caption; it contains no experimental reconstructions.
Its prompts and provenance are recorded in `IMAGEGEN.md`. The data figures
use only archived source-luma thumbnails as raster content. Diagrams use tensor planes, nested block
strips, active-tile grids, feature joins, expert queues and coded-stream
segments to explain the computation.

| Meaning | Colour | Additional cue |
|---|---|---|
| Learned router / adaptive path | Teal `#007F86` | Square marker |
| Source-informed search | Muted violet `#63527C` | Circle marker |
| Bayer dithering | Amber `#BB7534` | Triangle marker |
| Uniform depth | Slate `#8999A3` | Diamond marker |
| Text and axes | Ink `#183342` | Direct labels and explicit units |
| Grid | Pale grey `#E2E9EB` | Fine horizontal guides |

Depth uses an ordered six-colour scale: `#DCECEB`, `#A8D4CE`, `#60B1A8`,
`#2B8D8C`, `#286777`, `#244457` for 2/4/6/8/10/12 blocks. The current
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

Main-figure heights are 60 mm (exact execution), 65 mm (nominal budgets),
63 mm (delivered cap) and 49 mm (spatial maps), plus the 61 mm conceptual
overview. The old four-panel sensitivity view lives in the supplement.
Figures were redesigned at these sizes rather than shrunk with their old
labels. Main plots use 7 pt axis labels and legends, 6.3 pt ticks, consistent
markers, and visible paired confidence intervals.
