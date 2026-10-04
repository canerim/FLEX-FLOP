# Kodak 24 × 5 QP: router control transfer audit

This audit asks whether the QP-dependent routing seen in the nine-stream pilot
persists on all 24 Kodak images, and whether that trend comes from the router's
QP-dependent price. It uses 120 real FUFREF2 research bitstreams, decoded
independently on CPU. Each image supplies six 256 × 256 tiles. The same decoded
latent is reconstructed with the CTC-calibrated QP-specific beta and a
counterfactual that holds beta at its QP16 value. The router head still receives
QP in both arms. Raw bytes, hashes and per-case results are in
[`proof/cpu_early_exit/results/kodak24_qp5`](../../../proof/cpu_early_exit/results/kodak24_qp5)
and the matched quality replay is in
[`proof/cpu_early_exit/results/kodak24_fixed_beta_quality`](../../../proof/cpu_early_exit/results/kodak24_fixed_beta_quality).
The [summary JSON](../../../cvpr2027/figs/kodak24_qp/fig_kodak24_qp_policy.json)
and [vector figure](../../../cvpr2027/figs/kodak24_qp/fig_kodak24_qp_policy.pdf)
verify all 120 result hashes before aggregating.

| QP | Calibrated β | Deepest tiles | Calibrated MAC saved | Calibrated Δ₄₄₄ | Fixed-β MAC saved | Fixed-β Δ₄₄₄ | Cases over 0.1 dB, calibrated / fixed |
|---:|---:|---:|---:|---:|---:|---:|---:|
| 0 | 30.68 | 0.0% | 31.88% | 0.074 dB | 26.29% | 0.056 dB | 7 / 3 |
| 16 | 14.02 | 0.0% | 24.81% | 0.059 dB | 24.81% | 0.059 dB | 1 / 1 |
| 32 | −3.76 | 47.9% | 11.56% | 0.040 dB | 24.82% | 0.080 dB | 1 / 5 |
| 48 | −18.46 | 59.7% | 9.29% | 0.047 dB | 27.67% | 0.147 dB | 2 / 22 |
| 63 | −23.52 | 65.3% | 8.28% | 0.063 dB | 30.52% | 0.269 dB | 3 / 24 |

Every Kodak image has the same 512 × 768 pixel area, including portrait
orientations, and six tiles. Equal weighting of the 24 images and five QPs is
therefore also equal pixel/tile weighting *on this grid*. It is not a measured
deployment bitrate mix. The calibrated mean synthesis MAC saving is **17.16%**
[14.81, 19.58]% by a descriptive 24-image clustered bootstrap that keeps each
image's five QPs together. Its mean Δ₄₄₄ is **0.0565 dB** [0.0473, 0.0666].
The fixed-beta arm saves **26.82%** at **0.1223 dB** [0.1091, 0.1368]. The
paired fixed-minus-calibrated differences are +9.66 MAC percentage points
[8.69, 10.61] and +0.0658 dB [0.0586, 0.0742]. A total of 14/120 calibrated
and 55/120 fixed-beta cases exceed 0.1 dB. The CTC beta was calibrated for a
*mean* budget; these counts are not violations of a promised per-image bound.

The paper's earlier **27.97%** is a different experiment: the nominal
0.1 dB source-informed sweep on 53 CTC sequences × five QPs, with a
source-dependent control selected from archived candidate maps. The Kodak
result freezes one CTC-derived beta per QP before seeing each Kodak image and
runs the actual decoder-side router and assembled reconstruction from coded
bytes. The CTC beta-calibration means themselves are
35.27/31.85/25.44/23.25/20.38% at QP 0/16/32/48/63; Kodak achieves
31.88/24.81/11.56/9.29/8.28%. The large high-QP gap is a control-transfer
finding, not a direct contradiction of the source-informed nominal sweep.

The fixed-beta intervention identifies how much of the route change is caused
by the changing price **for the same router scores**. It does **not** establish
that compressed latents intrinsically need less or more synthesis capacity:
the head still sees QP, latent statistics change with QP, and the fixed-price
reconstructions lose substantial quality at QP48/63. The most defensible
interpretation is that CTC-calibrated beta is conservative on Kodak at high
QP, while holding the QP16 price everywhere is too aggressive. There may be
useful operating points between them; tuning that interval on Kodak and
reporting it on the same 24 images would leak evaluation data.

The immediate policy ablation should fit beta on an independent image set,
freeze it, then evaluate all 24 Kodak images once. A sequence-disjoint CTC
calibration split or a separate public image validation split can provide the
fit; compare mean-budget and high-quantile controls under the same achieved
Δ₄₄₄ cap, and report the router's own MAC and actual latency. A separate
QP-blind-head ablation would test whether latent content alone carries the
rate signal. Neither ablation requires retraining the DCVC-UF decoder.

Δ₄₄₄ here is `10 log10(MSE_assembled / MSE_e15_full_frame)` on unclipped,
equal-channel YCbCr 4:4:4; it is the paper's budget metric. The independent
released-D12-relative 6:1:1 weighted YUV PSNR loss averages 0.0919 dB for
the calibrated arm and 0.1573 dB for fixed beta, but its reference and
aggregation differ, so it must not be compared with the 0.1 dB Δ₄₄₄ target.
Analytical MAC includes suffixes, adapters and seam repair and excludes
router work. FUFREF2 is not Microsoft's native CUDA stream. The loaded CPU
host prevented claim-grade latency timing; this report makes no runtime or
full-codec speedup claim. The idle-host matched CPU watcher remains separate.
