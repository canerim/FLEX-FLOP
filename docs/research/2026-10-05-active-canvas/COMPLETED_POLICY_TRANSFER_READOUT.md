# Frozen early-exit policies: completed transfer readout

All results below keep the FUFREF2 research bitstream, e15 weights, six 256-pixel tiles, active-replicate neighbour fallback and repair-off synthesis fixed. Prices were selected on 24 DIV2K calibration images before evaluating the separate 24-image × five-QP DIV2K validation split. The reported compute reduction is **exact synthesis-convolution MAC**, not end-to-end codec latency. `Δ444` is cropped YCbCr 4:4:4 PSNR loss relative to full-frame e15 on the same stream. The DIV2K validation split has informed earlier project work, so these are exploratory transfer results rather than a globally untouched final benchmark.

| Calibration target | Validation mean Δ444 | Exact synthesis-conv MAC saved | Cases with Δ444 >0.1 dB | QPs whose mean misses their own calibration target |
|---:|---:|---:|---:|---|
| 0.075 dB | 0.06927 dB | 20.081% | 23/120 | 32, 48 |
| 0.100 dB | 0.08892 dB | 22.564% | 34/120 | 0, 63 |
| 0.125 dB | 0.11554 dB | 25.817% | 47/120 | 63 |
| 0.150 dB | 0.13492 dB | 28.271% | 59/120 | 63 |
| Original beta on same decoder | 0.06919 dB | 20.564% | 24/120 | — |
| Calibration-only tail rule | 0.03268 dB | 13.558% | 8/120 | Tail objective misses at QP16/32: 3/24 each |

The four budget policies were selected by mean calibration loss *per QP*, then transferred unchanged. Each candidate selected by the rounded historical MAC proxy is also the exact-convolution optimum on calibration: 20/20 QP × budget comparisons agree. Nevertheless, no budget policy meets its own mean target at every QP on validation. The 0.100-dB policy pools below 0.1 dB, but QP0/63 mean losses are 0.10301/0.11520 dB and 9/11 of their 24 cases exceed 0.1 dB. The 0.075-dB policy stays below 0.1 dB mean at every QP but fails its tighter target at QP32/48. The originally deployed price is slightly better than that 0.075-dB candidate on this split in pooled quality, MAC and tail count; this is an observation, not a post-hoc policy-selection rule.

The tail policy was preselected on calibration under mean loss ≤0.1 dB and at most two >0.1-dB cases per QP there. On validation it rescues 26 of the 34 primary-policy threshold failures and introduces none, reducing failures to 8/120. It sacrifices **9.007 percentage points** of exact synthesis-convolution MAC saving (paired 24-image CI −10.267 to −7.752 points) while improving mean quality by 0.05624 dB. It still misses its per-QP tail-count target at QP16 and QP32 (3/24 each), so it is a risk-reducing operating point rather than a validated hard guarantee. A truly untouched cohort and a decoder-side, source-free risk predictor remain necessary for a deployable guarantee.

## What the router placement buys

The fixed Bayer-order control keeps the six-tile exit histogram **identical in every case** to the 0.100-dB policy, so analytical synthesis MAC is identical. On the same 120 DIV2K cases, router placement improves YCbCr 4:4:4 PSNR by **0.03082 dB** on average (24-image clustered 95% CI **0.01095–0.06100 dB**). In 50 cases the maps are identical; among the 70 distinct-map cases, router wins 63, Bayer wins 7, and the conditional mean gain is 0.05284 dB. The QP-wise gains rise from 0.0018 dB at QP0 to 0.0711 dB at QP63. This isolates spatial placement from depth histogram and nominal MAC, but it does not compare against an independently calibrated dithering policy or measured latency.

## CLIC transfer and released reference

The frozen 0.100-dB price was also replayed on 39 geometry-eligible CLIC images × five QPs (195/195 cases; two images excluded *only* because they are smaller than the fixed 768×512 crop). It saves **27.153%** exact synthesis-convolution MACs (image-bootstrap CI 24.278–29.957%) with **0.09463 dB** mean Δ444 loss (CI 0.07218–0.12213 dB). The pooled mean again hides a tail: 64/195 cases exceed 0.1 dB; QP48/63 means are 0.10149/0.13561 dB, with 16/21 of 39 cases above 0.1 dB.

Per-image five-point PCHIP BD-rate on actual FUFREF2 research-stream rates and YUV 6:1:1 PSNR is available for all 39 images. The active new-price decoder is **+2.774% BD-rate versus released D12** (image-bootstrap CI +2.399 to +3.173%). On that same stream, e15 full-frame is +0.325% versus released D12 and the deployed old-beta tiled decoder is +2.960%; the new policy improves the latter by −0.184% BD-rate (CI −0.249 to −0.118%). This makes the remaining released gap visible: the new context and price improve the tiled path modestly, but they do not yet close the full-frame-to-tile quality gap. These BD-rate numbers are **not native DCVC-UF codec measurements**. CLIC files also appeared in earlier other-decoder work, so the cohort is a frozen-rule transfer check, not an untouched external test set.

The case-level cross-check matched all 120 DIV2K case and stream hashes across the primary, budget, tail and Bayer files; every 0.100-dB budget route/reconstruction equals the primary route/reconstruction; every Bayer histogram equals the router histogram. The CLIC summary hash matches the completed 195-case raw file. The next publication gate is a cohort that was never consulted during development, plus end-to-end rate/latency under the released codec path.

Source files: `proof/cpu_early_exit/results/div2k_beta/quality_floor/active_replicate_budget_sweep_policy.json`, `active_replicate_budget_transfer_validation24.json`, `active_replicate_budget_transfer_analysis.json`, `active_replicate_tail_locked_policy.json`, `active_replicate_tail_validation24.json`, `active_replicate_tail_validation24_analysis.json`, `active_beta_bayer_validation24.json`, `active_beta_bayer_validation24_analysis.json`, and `proof/cpu_early_exit/results/clic39_active_beta/{manifest,raw,summary}.json`. The vector summary is `fig_active_beta_policy_frontier.pdf` in the quality-floor folder.
