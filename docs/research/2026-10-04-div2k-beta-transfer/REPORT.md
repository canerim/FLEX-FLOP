# Independent β calibration and Kodak transfer — 4 October 2026

## Decision

The frozen **per-QP mean-budget policy** transfers to Kodak24×QP5 at **25.88% analytical synthesis MAC saving with 0.0976 dB mean Δ444**. This is the useful average operating point between the earlier CTC-β policy (17.16%, 0.0565 dB) and fixed-QP16 β (26.82%, 0.1223 dB). It is **not** a per-image or per-QP quality guarantee: 47/120 Kodak cases exceed 0.10 dB, and QP48/63 means are 0.1275/0.1261 dB. The 24-image cluster-bootstrap 95% interval for mean Kodak Δ444 is [0.0832, 0.1128] dB. The exploratory global-budget rule reaches 23.09% at 0.09994 dB on calibration but fails transfer (23.07% at 0.1131 dB on held-out DIV2K; 28.64% at 0.1228 dB on Kodak).

All MAC percentages are analytical **synthesis-stage** savings relative to e15 full-depth synthesis, including adapters and seam repair but excluding router, entropy recovery, transfer, and complete-codec latency. Δ444 is `10 log10(MSE_policy / MSE_e15_full)` in centred YCbCr444 from an actual assembled reconstruction and the same FUFREF2 bitstream. It is not the released-decoder YUV611 PSNR gap. No latency was measured on this loaded host.

## Locked protocol

The source is the original 100-image DIV2K validation PNG collection. Files are sorted by SHA256 of filename; the first 24 form calibration and the next disjoint 24 form internal validation. Each image is converted to RGB, centre-cropped to 768×512, then encoded at QP 0/16/32/48/63 using the released DCVC-UF analysis/entropy weights. The released model and e15 model have identical nondecoder tensors. Decoder-side stem/QP router scores, true research bitstreams, full e15 reference reconstructions, exact exit maps, patch assembly, and seam repair are used. A cache-versus-direct decoder check was pixel-exact on the first calibration image at every QP.

The candidate grid was fixed before evaluation: β from −50 to 60 in steps of 5, plus the exact prior CTC β at each QP and the exact prior QP16 β. The primary rule chooses, independently for each QP, the candidate with maximum mean analytical MAC saving among those with mean calibration Δ444 ≤0.10 dB. The selected β for QP 0/16/32/48/63 is **40/15/15/10/−10**. Manifest SHA256: `0468dc26738fcc16024abd224ba2c805ee25911553f68896f9ddafbaedadf922`; primary policy SHA256: `80f0a2f1d9cdda4931a956878a07df05c646e9c6a621c48d0593d3a10a1c5e47`. That policy was committed and pushed before either Kodak transfer. Kodak records were never used to calculate the selected β values. Kodak had, however, already been inspected in earlier exploratory work; it should not be described as globally untouched.

After seeing only the calibration aggregate, we added an **exploratory** alternative: maximize equal-QP mean saving subject to equal-QP mean calibration Δ444 ≤0.10 dB, with an exact Pareto search over the same candidates. Its β is **50/15/15/14.0229/5**, policy SHA256 `607e5a2da71b5eee106fd33388fa689bc3fc78283b848b1d58887fd093a91f7d`. This rule was also committed and pushed before the Kodak replay, but its rule itself was chosen after inspecting calibration, so it is an ablation rather than the predeclared primary outcome. No held-out DIV2K quality values were used to choose its β.

## Measured results

| Policy / cohort | MAC saved | Mean Δ444 | Cases above 0.10 dB |
|:--|--:|--:|--:|
| Prior CTC β → Kodak | 17.16% | 0.0565 dB | 14/120 |
| Fixed QP16 β → Kodak | 26.82% | 0.1223 dB | 55/120 |
| Primary β → DIV2K calibration | 20.73% | 0.0851 dB | 34/120 |
| Primary β → disjoint DIV2K | 20.05% | 0.0889 dB | 35/120 |
| **Primary β → Kodak** | **25.88%** | **0.0976 dB** | **47/120** |
| Exploratory global β → DIV2K calibration | 23.09% | 0.09994 dB | 45/120 |
| Exploratory global β → disjoint DIV2K | 23.07% | 0.1131 dB | 50/120 |
| Exploratory global β → Kodak | 28.64% | 0.1228 dB | 60/120 |

Kodak means give each of 24 images and five QPs equal weight. Both transferred policies replay the original 120 Kodak research streams and the archived router log probabilities; the output auditor checks stream, scan, policy, and checkpoint hashes. The 48 QP16/32 cases, where both β tables coincide, have identical exit counts, reconstructed quality, and MAC values across the two replays.

| Kodak QP | Primary MAC saved | Primary mean Δ444 | Cases above 0.10 dB |
|--:|--:|--:|--:|
| 0 | 34.98% | 0.0910 dB | 9/24 |
| 16 | 25.46% | 0.0610 dB | 1/24 |
| 32 | 25.39% | 0.0825 dB | 5/24 |
| 48 | 25.35% | 0.1275 dB | 17/24 |
| 63 | 18.23% | 0.1261 dB | 15/24 |

The worst primary-policy Kodak case is `kodim07` at QP63, 0.2989 dB; the worst held-out DIV2K case is `0844` at QP48, 0.4961 dB. The calibration outlier `0828` at QP0 is 0.7918 dB. Its source MSE and exit map were reconstructed with the independent direct decoder path; the direct and cached Δ444 values were exactly equal. These are real tail failures, not a cache artefact. The primary policy also loses 0.1382 dB YUV611 PSNR on average relative to the *released* decoder on Kodak; this is a different reference and metric from the 0.0976 dB Δ444 result.

### A tiling floor beneath the router loss

Further CPU-only decomposition of all 120 calibration cases finds that **five cases cannot meet 0.10 dB using any β on the locked candidate grid**. They are all five QPs of DIV2K `0828`. Even with all six tiles sent to the deepest exit, its Δ444 is **0.1148/0.1636/0.1517/0.1438/0.1452 dB** at QP 0/16/32/48/63. This is a floor of the tiled reconstruction relative to full-frame e15; changing the scalar router price cannot remove it. Other exit assignments outside the locked β grid are not excluded by this audit.

The direct decoder reproduced the QP0 primary-map loss (0.7918 dB) and all-deep floor (0.1148 dB) exactly. A 16-pixel band around the internal tile boundaries covers **13.64%** of pixels but contains **72.43%** of the all-deep-versus-full squared difference energy; mean difference in the band is **16.6×** the interior. At QP63 the corresponding figures are 53.78% of energy and 7.37×. Thus QP0 has both a large router-dependent excess and a structural seam floor, whereas QP63 is dominated by the floor. The audited replay JSON and vector figure are in `proof/cpu_early_exit/results/div2k_beta/quality_floor/` and `worst_case_direct_replay.json`.

A five-QP CPU counterfactual on this same source confirms the mechanism: a sufficient clipped feature halo, or depthwise-only canvas coupling, makes uniform all-deep output bit-for-bit identical to full-frame e15 when seam repair is disabled. At the **frozen mixed router map**, QP0 remains poor even with exact context (0.7202 dB versus 0.7918 dB deployed), whereas QP16–63 fall to 0.0107–0.0159 dB. The full halo costs 1.3149× full-frame synthesis MAC at the QP16–63 map, so it is not a practical saving mechanism; depthwise coupling yields 0.0556–0.0991 dB on those QPs but has **unmeasured latency** and no general guarantee. See [EXACT_CONTEXT_0828.md](EXACT_CONTEXT_0828.md) for the control design, cost accounting, and limitations.

The missing guarantee is material: two post-hoc held-out stress cases show that mixed-depth canvas coupling can be neutral or harmful. On `0898/QP48`, coupling increases loss by 0.2001 dB against a matched zero-pad/no-repair control, despite four of six tiles being deepest. [MIXED_COUPLING_STRESS.md](MIXED_COUPLING_STRESS.md) records the matched padding/repair controls. This rules out coupling as an untested global switch; a dependency-band design or a different decoder policy needs separate validation.

A source-informed oracle constrained to the **same finite β candidate grid** can reduce the calibration count above 0.10 dB from 34 to only those five structural cases, with **21.60%** mean MAC saving versus the primary fixed policy's 20.73%. It consults reconstruction loss and is not a deployable decoder rule. A fixed per-QP Q90 constraint on the calibration set chooses β = **10/10/−3.7647/−15/−23.5172** and saves only **11.20%** mean MAC on those same 120 cases. This is an exploratory calibration analysis, not a held-out transfer claim. It indicates that simply tightening the scalar budget trades away much of the compute benefit, while a seam-aware decoder or an additional confidence/fallback mechanism addresses a different failure mode.

The failure split is actionable: **29 of the primary policy's 34 calibration violations** would fall below 0.10 dB if an ideal source-informed detector switched just those frames to tiled all-deep; the remaining five are image 0828's structural floor. This binary fallback would reduce mean synthesis saving from 20.73% to **13.48%**. The finite-grid source-informed oracle reaches **21.60%** while leaving the same five failures, because it chooses less costly acceptable maps per case rather than always paying for all-deep. Neither uses a decoder-available risk signal. The gap quantifies the value of a learned quality-risk predictor, but it does **not** establish that such a predictor can achieve the oracle frontier.

An exploratory first decoder-side risk test using **only QP and the six selected exit indices** did not close that gap. Its violation-detection AUC fell from 0.785 on calibration to **0.646** on disjoint DIV2K validation, slightly below a mean-depth heuristic's 0.651. The calibration-locked threshold flagged 74/120 validation cases yet missed 7/35 quality violations. [ROUTE_ONLY_RISK.md](ROUTE_ONLY_RISK.md) gives features, thresholding, image-cluster uncertainty, and provenance. This is a negative ablation, not evidence of a deployable quality guard.

On Kodak24×QP5, the unchanged risk model ranks cases better (AUC **0.779**) but the same threshold flags **97/120** to catch 44/47 violations. Kodak was not used in this logistic fit but had been inspected earlier in the project; this diagnostic does not rescue the fallback-compute tradeoff or establish external generalization.

## Research implication and next ablation

The primary policy improves the mean Kodak quality–compute tradeoff without fitting β on Kodak, but a mean-only calibration rule permits visible high-loss outliers and underprotects QP48/63. The global budget rule illustrates why fitting exactly to 0.10 dB on one cohort has no transfer margin. The next scientifically useful policy ablation is a **separately fitted, rate-aware Q90 or worst-case loss constraint**, with a fresh calibration/validation split and a new untouched image cohort for final selection. This should be compared with the present frozen primary rule on both mean MAC and the fraction of cases above 0.10 dB; it should not be retuned on these Kodak results and then presented as held out. A measured end-to-end decoder benchmark remains separate from these analytical MAC results.

Reproduction entry points are `proof/cpu_early_exit/div2k_beta_transfer.py`, `run_locked_beta_after_calibration.sh`, `fit_global_beta_from_calibration.py`, `run_global_beta_after_primary.sh`, `kodak_locked_beta_transfer.py`, and `summarize_locked_beta_transfer.py`. Audited summaries and the vector comparison figure are in `proof/cpu_early_exit/results/div2k_beta/`.
