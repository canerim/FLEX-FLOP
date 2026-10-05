# Kodak: router placement versus matched-histogram Bayer

The frozen decoder-side router gains **0.01425 dB mean RGB PSNR** over a fixed Bayer-rank placement of *exactly the same six exit depths* on all 24 Kodak images at QPs 0/16/32/48/63. A 10,000-draw bootstrap clustered by image gives a 95% interval of **[0.00871, 0.02066] dB**. Router placement wins 84 of 120 image-QP cases, Bayer wins three, and 33 maps are identical. Mean YCbCr 4:4:4 gain is 0.01516 dB.

| QP | Router gain over Bayer, RGB | 95% image-cluster interval | Router / Bayer / same-map cases |
| ---: | ---: | ---: | ---: |
| 0 | 0.00770 dB | [0.00425, 0.01176] | 18 / 2 / 4 |
| 16 | 0.00615 dB | [0.00292, 0.01052] | 15 / 0 / 9 |
| 32 | 0.01240 dB | [0.00786, 0.01761] | 20 / 0 / 4 |
| 48 | 0.01772 dB | [0.01000, 0.02641] | 16 / 1 / 7 |
| 63 | 0.02728 dB | [0.01561, 0.04030] | 15 / 0 / 9 |

The control assigns the router-selected depth multiset to the six equal-area Kodak tiles by sorting deeper exits onto lower positions of the pre-existing 8×8 Bayer rank pattern cropped to the 2×3 or 3×2 grid. It uses neither source quality nor tile content after taking the histogram from the frozen router. Each comparison reuses the same pinned FUFREF2 entropy payload, e15 weights, full-frame repair/head and analytical synthesis MAC; the original routed RGB and YCbCr 4:4:4 MSEs reproduce the preceding bitstream placement replay to 1e-12. All 120 cases were retained, and source/model code hashes captured during the run were unchanged at completion.

This is a **matched-histogram placement control**, not the archived scalar ordered-dither policy, which chooses its own adjacent-depth histogram. It isolates the spatial value of the router given its global exit proportions; it does not establish net equal-quality MAC advantage over the deployed dithering policy. The corresponding random-permutation gain is 0.01608 dB [0.01122, 0.02130]. Their paired difference is 0.00183 dB [−0.00239, 0.00592], so the two blind placement controls are not distinguishable here. Neither control measures side-information bytes, native CUDA bitstreams or latency. Kodak has informed earlier project analysis and is not an untouched test set.

The complete [case records and summary](../../../proof/cpu_early_exit/results/kodak24_qp5_bayer_histogram_20261005), [replay](../../../proof/cpu_early_exit/kodak_bayer_histogram_replay.py), [summary script](../../../proof/cpu_early_exit/summarize_kodak_bayer_histogram.py) and [two-panel vector figure](../../figures/kodak-placement-controls-20261005/kodak_placement_controls.pdf) are included for audit.
