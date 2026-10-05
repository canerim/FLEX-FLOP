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
