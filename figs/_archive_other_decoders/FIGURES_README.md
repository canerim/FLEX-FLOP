# DCVC-UF, NoGAN-MS and ELIC re-evaluated: routing rules, content-blind dither, and the Bjøntegaard price of a compute budget

Every DCVC-UF number below is a real tiled decode of the rebuilt map (53 CTC intra frames × 5 qp), scored in DCVC-UF's own metric (PSNR 6:1:1 in 4:2:0) against the released decoder's full-frame decode of the same latent; rate is the real rANS stream. Image-decoder numbers are real decodes on Kodak-24 / CLIC-41 (RGB PSNR). Means are over frames feasible under all rules at that budget.

## Key results — DCVC-UF

- 0.1 dB budget (n = 263/265): saving oracle 29.5 % · router 28.0 % · dither 25.4 % · uniform 21.1 %; delivered loss (6:1:1) 0.087 / 0.083 / 0.088 / 0.069 dB (max 0.291); rate-equivalent of the loss vs released +2.48 / +2.37 / +2.52 / +2.01 % (classic BD-rate, two cubic fits: +0.56 / +0.44 / +0.56 / +0.07 %).
- 0.2 dB budget (n = 265/265): saving oracle 36.6 % · router 36.3 % · dither 35.0 % · uniform 32.0 %; delivered loss (6:1:1) 0.148 / 0.146 / 0.148 / 0.123 dB (max 0.381); rate-equivalent of the loss vs released +4.02 / +3.99 / +4.03 / +3.42 % (classic BD-rate, two cubic fits: +2.16 / +2.14 / +2.17 / +1.47 %).
- 0.3 dB budget (n = 265/265): saving oracle 38.4 % · router 38.3 % · dither 38.0 % · uniform 36.6 %; delivered loss (6:1:1) 0.178 / 0.178 / 0.178 / 0.161 dB (max 0.486); rate-equivalent of the loss vs released +4.74 / +4.72 / +4.73 / +4.34 % (classic BD-rate, two cubic fits: +2.89 / +2.88 / +2.88 / +2.53 %).
- Content-following premium over dither at 0.1 dB: +4.1 points (oracle), +2.5 (router); router captures 62 % of it with 8 bits/frame and 0.5 % of decode time.
- The dB budget is set in the RGB-ratio convention; delivered in 6:1:1 it reads 0.087 dB (oracle, 0.1 dB budget) — the two conventions differ by roughly a third of the budget, so every figure states which one it plots.
- Map agreement with the oracle at 0.1 dB: router 75 %, dither 53 %, uniform 40 % of tiles.
- Exit usage at 0.1 dB (exits 2/3/4/5, % of tiles): oracle 62/22/11/5; router 54/28/10/7; dither 36/41/19/4; uniform 18/48/27/7.
- Per class at 0.1 dB (oracle / router / dither / uniform): HEVC_B 31.0/29.5/26.5/22.9; MCL-JCV 31.8/30.2/27.2/22.5; UVG 29.3/27.7/26.1/22.4; HEVC_E 25.5/23.6/17.2/12.3; HEVC_C 23.4/21.1/21.6/15.9; HEVC_D 19.1/19.1/19.5/16.7.

## Key results — image decoders

- NoGAN-MS, Kodak-24, 0.1 dB (n = 24/24): oracle 18.2 %, dither 10.8 %, dither 4×4 12.0 %, uniform 16.1 %; premium over best content-blind (uniform) +2.1 points.
- NoGAN-MS, Kodak-24, 0.3 dB (n = 24/24): oracle 36.9 %, dither 28.8 %, dither 4×4 30.4 %, uniform 34.9 %; premium over best content-blind (uniform) +2.0 points.
- NoGAN-MS, CLIC-41, 0.1 dB (n = 41/41): oracle 33.3 %, dither 18.9 %, dither 4×4 21.2 %, uniform 26.9 %; premium over best content-blind (uniform) +6.4 points.
- NoGAN-MS, CLIC-41, 0.3 dB (n = 41/41): oracle 53.8 %, dither 42.6 %, dither 4×4 45.1 %, uniform 48.5 %; premium over best content-blind (uniform) +5.2 points.
- ELIC (MAC stand-ins), Kodak-24, 0.1 dB (n = 24/24): oracle 14.6 %, dither 8.4 %, dither 4×4 4.3 %, uniform 9.8 %; premium over best content-blind (uniform) +4.8 points.
- ELIC (MAC stand-ins), Kodak-24, 0.3 dB (n = 24/24): oracle 26.2 %, dither 19.9 %, dither 4×4 16.2 %, uniform 20.4 %; premium over best content-blind (uniform) +5.8 points.
- ELIC (MAC stand-ins), CLIC-41, 0.1 dB (n = 41/41): oracle 20.8 %, dither 8.4 %, dither 4×4 6.8 %, uniform 9.7 %; premium over best content-blind (uniform) +11.1 points.
- ELIC (MAC stand-ins), CLIC-41, 0.3 dB (n = 41/41): oracle 36.8 %, dither 21.6 %, dither 4×4 20.4 %, uniform 22.0 %; premium over best content-blind (uniform) +14.7 points.

## Figures

- `fig1_rd_bd` — (a) RD curves, released vs oracle-routed at 0.1 and 0.3 dB; (b) delivered PSNR loss per rate point, all rules, 0.1 dB; (c) BD-rate vs released as a function of budget, all rules.
- `fig2_bd_frontier` — DCVC-UF: MAC saving against BD-rate cost, four rules, budgets as points.
- `fig3_saving_vs_budget` — saving vs budget: DCVC-UF (CTC), NoGAN-MS and ELIC on Kodak-24 and CLIC-41.
- `fig4_premium` — (a) oracle − best content-blind rule vs budget, five datasets; (b) DCVC-UF: oracle − dither, router − dither, dither − uniform.
- `fig5_exit_usage` — share of tiles per exit: rules at 0.1 and 0.3 dB; oracle per qp.
- `fig6_maps` — example frames (qp 32): luma thumbnail and the oracle / router / dither / uniform maps at 0.1 dB with saving, delivered loss and agreement.
- `fig7_agreement_class` — (a) map agreement with the oracle vs budget; (b) per-class saving at 0.1 dB.
- `fig8_rung_ladders` — dB cost vs saving of every uniform rung, three decoders.
- `fig9_per_image` — per-image oracle vs dither / uniform saving at 0.1 dB, Kodak and CLIC.
- `fig10_distortion_compute` — image decoders: delivered loss vs saving, all rules.
- `fig11_dcvcuf_per_qp` — DCVC-UF per qp, real decodes.
- `uf_analysis.json`, `eval_rules_ctc_e15.json` (per frame: maps, PSNR, saving), `nullmodels/*.json` — the numbers behind every panel.

## Methods (one line each)

- Rules: oracle `argmin_k M[t,k] + λ c_k` (encoder table); router `argmax_k log p_k − β c_k` (decoder-side, ce_soft head); dither: two adjacent rungs mixed by an 8×8 Bayer threshold pattern (content-blind); uniform: one rung. One parameter per frame, bisected to the cheapest value with delivered loss ≤ budget (DCVC-UF: on the tile table, then the map is decoded for real; image decoders: 14-step bisection on real decodes).
- Rate-equivalent of the loss: one monotone (PCHIP) interpolant of the released RD curve per sequence (real rANS bpp, PSNR 6:1:1, five qp); for each qp the extra rate the released decoder would need to fall to the routed PSNR, averaged in log domain — the Bjøntegaard question asked without fitting two nearly identical curves. Classic BD-rate (two cubic fits) is reported beside it; its per-sequence values are noisy because the curves differ by ~0.1 dB.
- Cost: DCVC-UF exits 2…5 at 0.609/0.758/0.870/1.010 of the released decoder (256-px tiles, trunk halo 0, grid seam repair); image decoders via `true_cost` (a block runs wherever any cell within its receptive field asks for it).
- Checkpoints: DCVC-UF RECIPE512 e15; NoGAN-MS q3 60k (cell 64 px); ELIC N=192 λ=0.016, MAC-priced stand-ins, 20k (cell 256 px). TF32 off, A6000.
- Code: `flexplus/dump_router_lp.py`, `flexplus/null_models_uf.py`, `flexplus/eval_rules_ctc.py`, `evc/flexevc/null_models.py`; analysis `scratchpad/rep/analyze_uf.py`, figures `scratchpad/rep/plot_eval2.py`.
