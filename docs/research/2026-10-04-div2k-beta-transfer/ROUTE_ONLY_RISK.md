# Exit map alone is an unreliable quality-risk signal

This is a **post-hoc exploratory negative ablation**, not a trained router or a
quality guarantee. The 24 DIV2K calibration images (120 image–QP cases) fit a
fixed L2-regularized logistic model for whether the locked primary β output
loses more than 0.10 dB Δ444 versus full-frame e15. The disjoint next 24
images (120 cases) evaluate it once. Labels require source reconstruction
quality; inference inputs are decoder-known **QP and six tile exit indices
only**. The seven predeclared features are normalized QP, mean and minimum
depth, fraction at deepest exit, fraction shallow, mean adjacent depth jump,
and fraction of unequal adjacent tile boundaries. The image pixels, latent
activations, reference MSE, and router logits are absent from the predictor.

| Split | Violations | AUC | Average precision | Recall at locked threshold | Flagged cases | Precision |
|:--|--:|--:|--:|--:|--:|--:|
| Calibration | 34/120 | 0.785 | 0.657 | 30/34 = 88.2% | 73/120 | 41.1% |
| Disjoint validation | 35/120 | **0.646** | 0.397 | **28/35 = 80.0%** | **74/120** | **37.8%** |

The threshold was selected on calibration to catch at least 80% of its
violations and was not adjusted on validation. A simpler mean-exit-depth score
gets **0.651 AUC** on validation, slightly better than the fitted model. A
2,000-resample image-cluster bootstrap over the 24 validation images gives a
95% interval of **[0.475, 0.803]** for fitted AUC and **[−0.130, 0.117]** for
its AUC difference from mean depth. The held-out sample is too small to
resolve a subtle difference, but there is plainly no evidence of a useful
gain from this map-only model. It flags 61.7% of cases and still misses seven
threshold violations. Sending every flagged case to all-deep would therefore
be a large compute cost with no per-case quality guarantee; we have not
reconstructed that policy and make no fallback-quality claim.

The result is consistent with the `0828`/`0844` mechanism audit: the same
coarse exit histogram does not tell the decoder whether errors come from
content, a poor shallow choice, or a tile boundary. A future predictor must
use additional **decoder-available** information, such as calibrated router
logit margins, latent/stem statistics, or a cheap reconstruction-consistency
probe. It must be fit on one cohort, thresholded there, and evaluated on a new
independent cohort with actual bitstream quality, MAC, and latency. We should
not tune further feature combinations against this validation set and call it
held out.

The script `proof/cpu_early_exit/audit_route_only_risk.py` checks manifest and
policy hashes, fixes ridge strength at 1.0, and archives coefficients,
per-case predictions, and bootstrap protocol in
`proof/cpu_early_exit/results/div2k_beta/route_only_risk_audit.json`.
