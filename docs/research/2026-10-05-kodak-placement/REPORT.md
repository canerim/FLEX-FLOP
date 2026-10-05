# Kodak bitstream placement intervention

**Result.** On all 24 Kodak images at five quality indices, the frozen decoder-side router's original exit map is better than three fixed spatial permutations of **that same map's six depth assignments** by 0.01608 dB mean RGB PSNR. The 24-image cluster-bootstrap 95% interval is [0.01122, 0.02130] dB; a nested resampling of images and the three recorded permutations gives [0.01109, 0.02142] dB. The RGB gain is positive in 96/120 image-QP cases, negative in 2, and exactly zero in 22 unchanged-map cases. Mean YCbCr 4:4:4 gain is 0.01783 dB.

| QP | Mean RGB placement gain | 95% image-cluster interval | Positive / negative / unchanged |
| ---: | ---: | ---: | ---: |
| 0 | 0.00608 dB | [0.00322, 0.00961] | 18 / 2 / 4 |
| 16 | 0.00608 dB | [0.00372, 0.00882] | 19 / 0 / 5 |
| 32 | 0.01361 dB | [0.00836, 0.01990] | 20 / 0 / 4 |
| 48 | 0.02155 dB | [0.01456, 0.02894] | 20 / 0 / 4 |
| 63 | 0.03309 dB | [0.02209, 0.04480] | 19 / 0 / 5 |

The paired QP63-minus-QP0 change is 0.02701 dB [0.01694, 0.03778] across the same 24 images. It describes this frozen control and its QP-dependent depth histograms; it does not isolate an intrinsic bitrate effect on decoder capacity.

The protocol reuses 120 pinned FUFREF2 research bitstreams and archived, CTC-calibrated QP-specific router scores. Each stream is independently entropy-decoded to the same latent and passed through the same e15 decoder. Three predetermined SHA256-seeded permutations per image-QP rearrange the six 256×256 tiles, preserving the multiset of exits and therefore exact modelled convolution MACs. Each original reconstruction reproduces the archived routed YUV 6:1:1 PSNR within 1e-5 dB before the RGB MSE comparison. The original and shuffled maps use the same source, latent, weights, padding and full-frame repair/head. All 120 cases are retained; no outcome-based map selection or quality fallback is used.

This is a transfer-cohort placement control, not a new bitrate or latency result. Kodak has informed earlier project analysis, the model and router were developed before this intervention, and three permutations do not exhaust unseen assignment tails. FUFREF2 is the pinned research format rather than Microsoft's native CUDA stream. The map is external side information, so its byte cost is not measured here. The image-cluster intervals condition on frozen weights and controller; nested resampling checks only the three observed permutation outcomes. The separate Bayer-rank matched-histogram control is still running and should be reported alongside this result when complete.

The complete per-case rows, input hashes, seeds, quality metrics and intervals are in [`results/kodak24_qp5_placement_20261005`](../../../proof/cpu_early_exit/results/kodak24_qp5_placement_20261005). [`kodak_placement_replay.py`](../../../proof/cpu_early_exit/kodak_placement_replay.py) performs the decode and replay; [`summarize_kodak_placement.py`](../../../proof/cpu_early_exit/summarize_kodak_placement.py) computes the paired image-cluster and nested intervals; [`plot_kodak_placement.py`](../../../proof/cpu_early_exit/plot_kodak_placement.py) renders the vector figure. Critical local source hashes recorded during the run matched again at completion.
