# Where active-neighbour context helps

We compared the frozen original-beta exit maps with the seam repair switched off in both arms: isolated tiles versus stage-synchronous active-replicate context. The same 24 disjoint DIV2K validation images, five QPs, FUFREF2 research streams and e15 weights were used throughout. This is a pixel-error mechanism test, not a new beta selection, latency measurement or native DCVC-UF BD-rate result.

Across 120 cases, active context improves mean full-crop YCbCr 4:4:4 PSNR by **0.03101 dB** (24-image cluster-bootstrap 95% CI 0.02149–0.04243 dB). Every one of the 24 image-level five-QP aggregates has positive net error reduction. The first 8 pixels around internal tile seams occupy **6.72%** of pixels but account for **62.40%** of the total signed MSE reduction (image-cluster bootstrap CI **55.99–68.75%**). This supports the specific boundary-context mechanism rather than only a global score improvement.

| Distance from internal seam | Pixel share | Share of total MSE reduction | Mean local PSNR gain, 95% image-cluster CI |
|---|---:|---:|---:|
| 0–8 px | 6.72% | 62.40% | 0.2354 dB [0.1756, 0.3039] |
| 8–16 px | 6.92% | 7.70% | 0.0309 dB [0.0212, 0.0416] |
| 16–32 px | 13.05% | 4.06% | 0.0114 dB [0.0065, 0.0177] |
| 32–64 px | 22.98% | 5.63% | 0.0103 dB [0.0055, 0.0164] |
| 64+ px | 50.33% | 20.21% | 0.0179 dB [0.0098, 0.0280] |

The signed MSE shares are additive over pixels; the local PSNR gains are not. The 64+ px bin still contributes because it contains half the image, despite its much smaller per-pixel improvement. A Sobel-stratified audit in the archived analysis shows positive pooled gains in every measured texture stratum within the first 8 px, though the absolute gains vary. This is a spatial association under a controlled decoder intervention, not a proof that distance alone causes the improvement.

Sources: `proof/cpu_early_exit/results/div2k_beta/quality_floor/context_seam_locality_validation24.json` (SHA256 `d45a5b4cfdf9214b2db1f807c440dfb7d84ae320c4e10398506e488acad0b8e5`), `context_seam_locality_analysis.json`, and `fig_context_seam_locality_evidence.json` in the same directory. The evidence JSON records plot/source hashes and the bootstrap interval; `plot_context_seam_locality.py` regenerates the figure.
