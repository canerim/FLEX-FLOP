# Frozen active-context spatial-placement control

This control was specified while the new active-replicate beta calibration
was still running (96/120 cases) and before its DIV2K validation. It asks
whether a fixed, content-blind tile ordering loses quality to the learned
router **at precisely the same selected exit histogram** under the new
active-neighbour/no-repair decoder. Use the frozen DIV2K validation24 crops
at QPs 0/16/32/48/63. Do not change the calibration-only beta policy from
any outcome of this control.

For every 768x512 crop, the tile grid is 2x3. Assign the selected six
depths in descending order to the ascending ranks of the existing fixed
8x8 Bayer pattern, cropped to that grid. The original and Bayer maps
therefore execute exactly the same number of each exit, retain the same
stream/latent/checkpoint/decoder, and have identical analytical synthesis
convolution MAC. Reconstruct both maps with stage-synchronous
active-replicate depthwise context and with the old seam repair disabled.
Verify the original-map result against the independently queued beta
validation replay before accepting the control.

The primary estimand is the image-clustered mean of
`10 log10(D444_Bayer / D444_router)` over 24 images x five QPs. Report a
95% image bootstrap interval, per-QP effects, ties, directions and the
full case-level values. Positive values favour router placement. This is
**not** the independently calibrated scalar dithering policy, which can
choose a different depth histogram. It isolates position at fixed
histogram and cost; it cannot prove an equal-quality net speedup. Reuse
only the already archived FUFREF2 streams; run untimed CPU FP32 with one
low-priority thread after the other queued analyses, without GPU access.
