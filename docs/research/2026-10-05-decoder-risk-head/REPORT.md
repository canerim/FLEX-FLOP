# Decoder-visible early-exit risk head: exploratory cross-fit

The current router's source-calibrated MAC margin is encouraging, but fixed-control quality remains uncertain. I tested whether its frozen tile logits can predict the **incremental reconstruction error** from exiting at 6, 8 or 10 blocks instead of 12. A ridge head uses four router log-probabilities, their entropy and top-two margin. It sees no source pixels or error labels for a held-out sequence. All five QPs of each CTC sequence stay in the same fold; the price or frame-risk cap is fitted on the other four folds to target a 0.1 dB 90th-percentile table loss.

| Frozen tile-table policy, 265 frame–QP cases | Mean nominal MAC saving | Q90 table loss | Cases above 0.1 dB |
|---|---:|---:|---:|
| Original logit router, Q90 control | 17.59% | 0.1035 dB | 32 |
| Predicted excess error, fixed price | 19.41% | 0.1135 dB | 33 |
| Predicted excess error, decoder-visible per-frame cap | 19.62% | 0.1116 dB | 34 |

The extra computation saving does **not** satisfy the same achieved quality risk. In particular, the adaptive cap is worse than the original router on QP32: 17.70% versus 16.74% nominal saving, but 0.1172 versus 0.1111 dB Q90 table loss and eight versus six cases above target. This is not a quality-matched gain and does not merit a main-paper improvement claim.

The diagnostic explains the miss. Across held-out tiles, predicting excess MSE from the frozen logits gives R² between **0.027 and 0.370** across the five QPs and three shallow exits. The six-block prediction is biased low by 0.0022–0.0028 in released-reference-normalized MSE. Thus the score contains some ranking information but is a weak absolute risk estimate, especially in its underpredicted tail. The per-frame cap needs an absolute risk signal; low average regression error is insufficient for a Q90 guarantee.

This experiment uses CTC labels that have informed prior development, and its outcome is a separable tile-error table, not a mixed-map neural reconstruction. The adaptive selector also enumerates 251 price candidates offline; its controller latency has not been measured. Future work should fit spatially aligned, decoder-visible latent/stem features on a genuinely separate training cohort and validate a tail-risk objective on untouched sources before replacing the deployed router. The earlier DIV2K route-plus-latent image-level classifier also failed to improve violation detection reliably, so simply adding global latent moments is not enough.

Reproduce the fixed-price head with `python3 proof/cpu_early_exit/audit_decoder_risk_head.py` and the adaptive cap with `python3 proof/cpu_early_exit/audit_adaptive_risk_cap.py`. [Fixed-price evidence](analysis.json) and [adaptive evidence](adaptive_analysis.json) include fold assignments, input hashes, all 265 held-out decisions and per-QP outcomes. Both run CPU-only with one BLAS thread.
