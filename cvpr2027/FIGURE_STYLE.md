# RegLIC figure system

The main figure system uses 183 mm full-width and 89 mm single-column canvases, Liberation Sans type, embedded
TrueType PDF fonts and editable SVG text. Quantitative plots and exact
execution diagrams are generated from bundled source tables with Matplotlib.
The new introduction figure is a text-free conceptual raster at column width.
The original opening overview supplies the cropped independent-bank system
diagram on page 2. The active shared-exit mechanism is an AI-generated conceptual raster,
labelled as such in its caption. Neither contains experimental reconstructions.
Their prompts and provenance are recorded in `IMAGEGEN.md`. The data figures use archived source-luma thumbnails and explicitly
identified, predeclared RGB reconstruction windows as raster content. Diagrams use tensor planes, nested block
strips, active-tile grids, feature joins, expert queues and coded-stream
segments to explain the computation.

| Meaning | Colour | Cue in the budget comparison |
|---|---|---|
| Learned router / adaptive path | Teal `#008A96` | Square marker |
| Source-informed search | Muted violet `#79679A` | Circle marker |
| Bayer dithering | Amber `#BF783D` | Triangle marker |
| Uniform depth | Slate `#81909C` | Diamond marker |
| Text and axes | Ink `#20313E` | Direct labels and explicit units |
| Grid | Pale grey `#E5EAED` | Fine horizontal guides |

Depth diagrams and maps use an ordered six-colour scale: `#6CB7A4`, `#46A49D`, `#278B92`,
`#22738A`, `#245B7A`, `#243F5C` for 2/4/6/8/10/12 blocks. The current
shared-exit maps use only the last four colours. Every map has a numerical
key, so colour is not the sole carrier of depth information.

Markers distinguish policies or calibration variants according to each
figure's explicit key. In the fixed-control replay, filled circles denote
mean calibration and hollow squares denote Q90; both are measured outputs.
Bootstrap bands are labelled with their resampling unit. Analytical runtime
scenarios use dashed lines, open markers and an explicit **not measured**
heading, so fill alone never identifies a measurement. Training/planned
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

The September 28 editorial revision preserved its then-current Figures 1 and 2.
The October 2 RegLIC revision replaces the opening figure with a compact,
single-column conceptual illustration; the system figure depicts the implemented
shared-exit connectivity: stem and QP feed the MLP; beta adjusts its output
scores; all four exits reach the common feature canvas. Its synthetic scene
is not a qualitative codec result.

Plots export with at least 7 pt text, 7.5 pt axis labels,
8 pt panel titles and bold 8 pt panel letters. The 7 pt export floor stays
above 6.5 pt after scaling to the actual CVPR column widths; the verifier
checks this scaling explicitly. Repeated poster headings and long figure footnotes
move into the exported captions; unused top and bottom canvas is removed
without changing physical text, axis or marker sizes. Explicit analytical
assumptions remain visible. Line styles and marker shapes supplement colour.
All depths use one ordered palette; policy comparisons keep their separate
teal/violet/amber/slate key.

The new 89 mm execution diagram uses the recorded 5-by-8 router map,
not a simulated population. Its rows show 40/27/7/3 active tiles, with
13/20/4/3 departures. Colour identifies the executed stage, while pale cells
have already exited. The compact delivered-cap plot replaces a duplicate
full-width frontier in the main paper; the full comparison remains in the
supplement. No numerical values, confidence intervals or reference anchors
were changed for visual presentation.

Design references: Nature's [panel preparation guide](https://research-figure-guide.nature.com/figures/building-and-exporting-figure-panels/)
and [figure specifications](https://research-figure-guide.nature.com/figures/preparing-figures-our-specifications/).
These inform visual decisions; they are not a claim of journal endorsement
or a replacement for CVPR submission requirements.

The second September 28 revision replaces all manuscript tables with plots.
Figure 3 uses oblique planes preserving real tile positions; Figure 4a uses
orthographic 3D, a categorical policy axis and only six measured points per
policy. Connecting curtains are guides, not fitted surfaces. Figure 6 contains
eight source groups selected by class quotas and name hashes, with source
luma plus five recorded maps per group; infeasible maps remain as N/A.
Gallery images are spatial context rather than reconstructed quality results.
