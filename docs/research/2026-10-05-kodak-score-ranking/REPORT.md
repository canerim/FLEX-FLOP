# Do frozen router scores rank real bitstream placement quality?

**Yes, within the sampled same-histogram alternatives.** Across all 24 Kodak images and QPs 0/16/32/48/63, the frozen decoder-visible router score prefers the lower-RGB-MSE map in **323 of 402** distinct blind-map pairs (80.35%; 24-image cluster-bootstrap 95% interval **[74.06%, 86.01%]**). For the 167 pairs separated by at least 0.01 dB, it is correct in **156** (93.41%). These are within-image/QP comparisons on the *same entropy-decoded latent, six exit counts, weights, and analytical synthesis MAC*.

| QP | Correct / compared | Fraction |
|---:|---:|---:|
| 0 | 73 / 91 | 80.22% |
| 16 | 51 / 58 | 87.93% |
| 32 | 76 / 105 | 72.38% |
| 48 | 65 / 82 | 79.27% |
| 63 | 58 / 66 | 87.88% |

For each image/QP, we deduplicated the three preselected SHA256-seeded random permutations and the predetermined Bayer-rank assignment, then **excluded the router's selected map**. Excluding that map is essential: the router's map maximizes its own separable score by construction, so including it would inflate agreement. We compared every remaining distinct pair. The score is the sum of frozen per-tile log probabilities for the assigned exit. The calibrated cost term cancels exactly because the two maps have the same depth histogram. Ties are defined by <1e-10 score difference or <1e-12 RGB MSE difference; there were no pairwise ties. A 10,000-draw bootstrap resamples 24 images with all five QPs and all within-image pairs together, rather than treating 402 pairs as independent.

This is evidence that the frozen score contains useful information about **spatial allocation**, over and above the router's depth proportions. It does **not** establish an equal-quality compute advantage over autonomous dithering, which chooses its own histogram; it also does not imply a calibrated distortion predictor or optimality over all permutations. Kodak informed prior development, and the three random assignments plus Bayer do not cover every possible map. The >0.01 dB subset is a sensitivity check chosen for interpretability after the basic ranking audit, not a new held-out endpoint. No rate or latency is measured.

The [audit script](../../../proof/cpu_early_exit/audit_kodak_score_ranking.py) recomputes the scores from the frozen scan records and the actual reconstruction MSEs; its [JSON output](../../../proof/cpu_early_exit/results/kodak_score_ranking_20261005.json) stores source hashes and the score/MSE of every blind candidate. The result suggests a concrete next method target: improve the *calibration and budget selection* of a placement-sensitive score, while keeping its decoder-only inputs and checking real bitstream quality on separate images.
