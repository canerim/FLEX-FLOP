# Predeclared seam-locality mechanism audit

The matched 2×2 context/repair experiment shows that active-neighbour
context and trained seam repair interact. The next question is spatial:
does active context repair error concentrated near *internal tile seams*,
or is the small frame-average PSNR gain distributed across each tile?
The local Mosaic study's boundary-band diagnostic motivates testing this
mechanism directly. This is an exploratory diagnostic on already-used
DIV2K validation images, not a new held-out performance estimate.

`audit_context_seam_locality.py` replays all 24 DIV2K validation crops at
QP 0/16/32/48/63 on the original calibration-locked beta maps. It decodes
each frozen FUFREF2 stream once and reconstructs with the same e15 model
using (a) isolated tiles, no repair and (b) active-replicate neighbours,
no repair. Archived scalar Δ444 values must reproduce the paired full-image
gain within 1e-5 dB. The run has no training and uses one low-priority CPU
core after the β-calibration/validation chain finishes.

Each source pixel is assigned to exactly one distance band from the nearest
*internal* 256-pixel tile boundary: [0,8), [8,16), [16,32), [32,64) or
[64,∞) pixels. For each band and case, the script records YCbCr 4:4:4
channel-mean squared errors and active-over-isolated PSNR gain. It also
records the fraction of image pixels in the band and its signed share of
net squared-error reduction. The source's luma Sobel magnitude is stratified
into [0,10), [10,25), [25,50), [50,100), [100,200) and [200,∞) on an
8-bit scale, to distinguish distance effects from image-edge density.
`analyze_context_seam_locality.py` pools those strata and uses 10,000
image-cluster bootstrap draws for the per-band paired means.

Evidence that would support the proposed boundary mechanism is a stronger
gain in near-seam than far-interior bands, including within comparable
Sobel strata. A flat profile would weaken that interpretation even if
whole-image PSNR improves. The result will not by itself establish native
decoder latency, rate change or deployment benefit.
