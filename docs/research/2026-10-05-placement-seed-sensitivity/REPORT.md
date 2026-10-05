# Placement intervention: permutation-seed sensitivity

The frozen QP32/Q90, 53-sequence reconstruction intervention uses three recorded, stratified assignment permutations per image and policy. Its equal-sequence mean RGB placement gain is **0.01393 dB** for the router and **0.00036 dB** for dithering; the paired advantage is **0.01357 dB**. This compares each original depth map with permutations of *its own* exit histogram, preserving synthesis MAC within each policy.

| Resampling unit | Router 95% interval (dB) | Dither 95% interval (dB) | Router minus dither (dB) |
| --- | ---: | ---: | ---: |
| Images, recorded permutation mean fixed | [0.00946, 0.01918] | [-0.00029, 0.00102] | [0.00914, 0.01876] |
| Permutation indices, images fixed | [0.01274, 0.01500] | [0.00002, 0.00071] | [0.01224, 0.01475] |
| Nested images and permutation indices | [0.00940, 0.01937] | [-0.00037, 0.00113] | [0.00907, 0.01904] |

The nested interval remains positive under this sensitivity check. Each of 20,000 paired draws samples 53 images with replacement, then samples three of the *observed* permutation indices for each selected image; router and dither use the same sampled indices. We recompute `10 log10(mean shuffled RGB MSE / original RGB MSE)` per sampled image before averaging. All 106 per-sequence point estimates reproduce the published JSON to 1e-12. The exact input-file hashes, random seed and results are in [analysis.json](analysis.json); [the script](../../../proof/cpu_early_exit/audit_placement_mc_uncertainty.py) regenerates it.

This is a sensitivity analysis of three existing permutation outcomes, not an interval over all unseen permutations. It does not address selection or training-seed uncertainty, and this 53-sequence cohort was used in method development. The numerical advantage should therefore be interpreted as a frozen-cohort placement control, not an independent generalization estimate.
