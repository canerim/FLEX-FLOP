# Exact conv-MAC check of the CTC early-exit budget archive

The frozen source-calibrated 53-sequence × five-QP CTC archive is recounted against the released D12 full-synthesis convolution count on each **padded** frame. The routed count includes exit adapters and seam repair. No model is run and neither quality, mode-map bytes nor latency is remeasured. This is an arithmetic correction to the prior normalized cost model. The 25-sequence fixed cohort is complete at every budget; the budget-complete cohort has 25/51/53/53/53/53 sequences.

| Nominal Δ444 target | n fixed | Router exact saving | Dither exact saving | Router − dither exact | Router − dither old model |
|---:|---:|---:|---:|---:|---:|
| 0.05 dB | 25 | 18.72% | 16.62% | +2.10 pp [+1.23, +3.04] | +2.12 pp |
| 0.1 dB | 25 | 32.69% | 30.29% | +2.40 pp [+1.75, +3.06] | +2.39 pp |
| 0.15 dB | 25 | 36.77% | 35.49% | +1.28 pp [+0.87, +1.71] | +1.28 pp |
| 0.2 dB | 25 | 38.02% | 37.70% | +0.32 pp [+0.11, +0.57] | +0.31 pp |
| 0.3 dB | 25 | 38.33% | 38.33% | +0.00 pp [+0.00, +0.00] | +0.00 pp |
| 0.5 dB | 25 | 38.33% | 38.33% | +0.00 pp [+0.00, +0.00] | +0.00 pp |

The corresponding complete-cohort result at 0.10 dB is **27.54%** exact conv saving for router and **24.97%** for dither (+2.57 pp paired). The archived output PSNR and BD-rate proxy remain unchanged; the table only replaces the arithmetic denominator and block accounting. The source archive was calibrated using source quality, so this is not a deployable policy result. The 95% intervals resample complete sequences and retain their five QPs.

Reproduce with `python3 proof/cpu_early_exit/audit_ctc_exact_conv_mac.py`. The [analysis JSON](analysis.json) records every case and source SHA256.

## Manuscript refresh

The main nominal sweep uses every frame–QP pair feasible for all four policies, rather than requiring five-QP-complete sequences. At the 0.10 dB target this is 263 pairs: router **27.29%**, dither **24.78%**, paired difference **+2.51 percentage points** with a sequence-cluster 95% interval of **[1.82, 3.22]**. Thus this number and the 51-sequence complete-cohort result above have different inclusion rules.

The retrospective common-delivered-loss-cap analysis keeps all 265 pairs. At a 0.10 dB cap, the cheapest recorded eligible map gives router **27.07%**, dither **24.18%**, and **+2.89 points** paired [**2.08, 3.74**]. One router case and three dither cases use the zero-saving full-frame fallback. The common-cap selector observes the final quality of up to six already-decoded candidate maps, so its saving is a finite-pool diagnostic, not a deployable inference-time guarantee. The same exact convolution accounting is now used for these two paper protocols and uniform-depth controls. It remains distinct from full-codec latency and from the independent D2–D12 model bank.

Reproduce the nominal and delivered-cap summaries with `python3 scripts/paper_refresh_data.py` and `python3 scripts/audit_delivered_frontier.py`, using the frozen source archive and [per-case exact MAC file](../../../cvpr2027/data/bd_rate_budget_20261005/exact_mac_by_case.json). The corresponding paper source, figures and compiled PDFs are in the `cvpr2027` repository at commit `739c413`.
