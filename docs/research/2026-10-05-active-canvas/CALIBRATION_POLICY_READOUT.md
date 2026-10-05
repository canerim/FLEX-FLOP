# Active-replicate beta: completed calibration-only readout

All 120 DIV2K calibration cases and 1,004 distinct archived candidate
maps finished before the new beta was locked. The frozen policy has SHA256
`5704033849f9e9f63ca3814ab92b55ba274643758a313ad11d7515e9e4ee321d`.
The old beta is shown on the **same active-replicate/no-repair decoder**,
not on the original isolated decoder. Savings below are independently
recounted exact synthesis-convolution MAC versus released D12; loss is
cropped `Delta444` against full-frame e15. Each row has 24 images.

| QP | Old beta | Old exact saving | Old mean loss | Old >0.1 dB | New beta | New exact saving | New mean loss | New >0.1 dB |
|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| 0  | 40 | 31.65% | 0.08641 dB | 6 | 45 | 33.51% | 0.09168 dB | 6 |
| 16 | 15 | 21.65% | 0.04152 dB | 3 | 15 | 21.65% | 0.04152 dB | 3 |
| 32 | 15 | 21.79% | 0.06651 dB | 3 | 15 | 21.79% | 0.06651 dB | 3 |
| 48 | 10 | 17.87% | 0.07130 dB | 7 | 10 | 17.87% | 0.07130 dB | 7 |
| 63 | -10 | 13.33% | 0.06078 dB | 7 | 5 | 17.44% | 0.09826 dB | 9 |

The gain from retuning comes only at QP0 and QP63. QP63 is near the
mean-loss target and its calibration tail worsens from 7 to 9/24; this
is why the separately predeclared tail-constrained policy is worth
testing. Calibration alone cannot establish transfer to DIV2K validation,
Kodak or CLIC, and its losses must not be reused as held-out evidence.
The locked beta remains unchanged even when all calibration candidates
are ranked by exact, rather than legacy-proxy, MAC saving. The queued
validation and post-lock selector audit will establish the corresponding
held-out quality and exact compute trade-off.
