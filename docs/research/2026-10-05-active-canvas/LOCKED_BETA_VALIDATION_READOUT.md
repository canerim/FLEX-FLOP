# Locked active-context beta: disjoint DIV2K transfer

The 24-image calibration selected beta 45/15/15/10/5 at QP 0/16/32/48/63 before any of these 24 validation images were decoded. The validation replay is now complete: 120/120 real FUFREF2 research-stream cases, with the frozen e15/router, six 256-pixel tiles, active-neighbour replicate fallback, and seam repair disabled. The comparison below holds the decoder and bitstream fixed and changes only the beta-driven route. It measures cropped YCbCr 4:4:4 PSNR loss against full-frame e15 and **exact synthesis-convolution MACs**, not full-codec latency or released-D12 BD-rate.

| QP | Original beta | Locked beta | Original loss (dB) | Locked loss (dB) | Original MAC saved | Locked MAC saved | Cases >0.1 dB, original → locked |
|---:|---:|---:|---:|---:|---:|---:|---:|
| 0 | 40 | 45 | 0.07692 | 0.10301 | 33.00% | 36.10% | 7 → 9 / 24 |
| 16 | 15 | 15 | 0.05618 | 0.05618 | 22.65% | 22.65% | 3 → 3 / 24 |
| 32 | 15 | 15 | 0.07924 | 0.07924 | 21.17% | 21.17% | 4 → 4 / 24 |
| 48 | 10 | 10 | 0.09098 | 0.09098 | 16.55% | 16.55% | 7 → 7 / 24 |
| 63 | −10 | 5 | 0.04261 | 0.11520 | 9.46% | 16.35% | 3 → 11 / 24 |

Across all 120 cases, the locked policy saves **22.564%** exact synthesis-convolution MACs (24-image cluster-bootstrap 95% CI 19.383–25.795%) at **0.08892 dB** mean loss (CI 0.06192–0.12128 dB). The original beta on the *same active-replicate/no-repair decoder* saves 20.564% at 0.06919 dB. The paired change is +2.000 percentage points saving (CI +1.509 to +2.499) for +0.01974 dB loss (CI +0.01255 to +0.02781). The locked policy produces 34/120 cases over 0.1 dB, versus 24/120 for the original route; none are rescued and ten newly cross the threshold. QP0 and QP63 also miss the 0.1 dB **mean** target on this validation split. Thus a pooled mean below 0.1 dB is not a per-QP or per-image guarantee.

The route change is concentrated at the two endpoints: QP0 changes 15/24 images and 30 tile decisions; QP63 changes 19/24 images and 48 tile decisions. Every changed tile exits earlier, and every changed image incurs additional quality loss. The middle three QPs have identical routes. In particular, the QP63 jump from 9.46% to 16.35% saving costs 0.07259 dB and moves eight additional images above 0.1 dB. This localises the transfer problem to the aggressive endpoint prices rather than a global decoder regression.

The calibration-only selector audit independently recomputed exact convolution cost for every candidate map. It selected the same beta at all five QPs as the earlier rounded-cost proxy, so the observed transfer failure is **not** an arithmetic ranking bug. The most informative next control is the already queued calibration-only tail-constrained policy, evaluated once on this untouched split. Its result must be reported regardless of sign, and the primary locked policy must not be changed using these validation labels. A third CLIC39 cohort and matched-depth-histogram Bayer placement control are also queued. CLIC images appeared in earlier project work, so they are an external transfer cohort for this frozen rule, not a globally untouched benchmark.

The four calibration-only mean-loss budgets (0.075/0.10/0.125/0.15 dB) were also audited against exact convolution MAC ranking: **all 20 QP × budget prices are identical** to those selected by the older rounded proxy. This confirms the forthcoming budget sweep is not an artefact of proxy misranking; its *validation* quality and saving remain to be measured.

Sources: `proof/cpu_early_exit/results/div2k_beta/quality_floor/active_replicate_beta_validation24.json` (SHA256 `35d767f4edc13585946265c287f93c64d7cc6de9e31dbae4e8c93472d9da14a9`), `active_replicate_beta_validation24_analysis.json`, `active_beta_exact_selector_audit.json`, and `fig_active_beta_locked_transfer.pdf` in the same directory. Frozen policy SHA256: `5704033849f9e9f63ca3814ab92b55ba274643758a313ad11d7515e9e4ee321d`.
