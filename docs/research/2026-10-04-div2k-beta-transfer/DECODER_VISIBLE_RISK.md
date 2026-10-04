# Decoder-visible risk after exact context

The full exact-context validation left 24/120 image-QP cases above the
0.1 dB Delta444 threshold. A source-informed uniform-depth fallback removes
all 24 with one extra exit group in 22 cases and two in the remaining two,
but its decisions cannot be used by a decoder. This experiment asks whether
a low-cost risk signal can identify such cases *before* synthesis from the
actual bitstream, QP, and fixed router exit map.

The model sees no source pixels, reconstruction error, or full-depth decoder
activations. Candidate features were fixed before fitting: stream length;
latent absolute moments, zeros, and spatial variation; per-tile latent
energy and its interaction with shallow exit selection; plus QP and route
geometry. A regularized logistic regression (L2 penalty 10) trains on the
24-image DIV2K calibration partition (120 QP cases). Its threshold is chosen
to catch at least 80% of calibration violations. We compare it with the
same regression using route geometry and QP alone, then test both on the
disjoint 24-image validation partition.

The validation images have already been inspected in preceding mechanism
analyses. Thus this is an **exploratory transfer check**, not an untouched
external test or an online learned router result. Neither the model nor
its threshold changes the deployed decoder in this work. AUC, average
precision, false-alarm count, and missed violations should be reported
together; a high AUC with almost every case flagged would not be useful.

## Results

Exact-context labels have **24/120** violations in each split. The
route/QP-only model reaches calibration AUC **0.849** and validation AUC
**0.755**. Adding latent and stream features raises calibration AUC to
**0.918**, but validation AUC is only **0.760**. A 24-image cluster bootstrap
puts the validation AUC difference (expanded minus route-only) in
**[-0.088, 0.080]** at 95%; the apparent calibration gain does not
transfer reliably.

| Model | Validation AP | Flagged at calibration-locked threshold | True violations caught | Missed violations |
|---|---:|---:|---:|---:|
| Route + QP | 0.410 | 43/120 | 15/24 | 9/24 |
| Route + QP + latent/stream | 0.479 | 37/120 | 13/24 | 11/24 |

The expanded score flags fewer cases, but at the locked threshold it misses
more genuine failures. At an equal budget of the top 43 ranked validation
cases, it catches 16 rather than 15 violations, a marginal gain. Neither
signal approaches a reliable quality guarantee. This is a **negative
predictive ablation**, not a successful router upgrade. The result is
consistent with limited calibration size, image-cluster shift, or the
available latent summaries missing the local reconstruction difficulty;
those are hypotheses, not established causes. We should not build the
paper's main compute claim around this classifier.

The next defensible step is a fresh held-out set plus a risk objective that
uses spatially aligned decoder-visible features during training, with
calibration and evaluation fixed before looking at outcomes. Meanwhile,
the frozen depth router and exact-context quality study remain the measured
results; actual sparse execution latency is still absent.

## Reproduce

1. Run `audit_exact_context_pilot.py --split calibration --cases
   /tmp/flexplus-div2k-beta-calibration --images 24 --output
   proof/cpu_early_exit/results/div2k_beta/quality_floor/exact_context_calibration24.json`
   on the pinned released FUFREF2 streams and e15 checkpoint.
2. Run `extract_decoder_visible_features.py` against both complete exact
   calibration and validation JSON files.
3. Run `audit_latent_risk.py` on those features. Its fixed feature groups,
   L2 penalty, threshold rule, and image-cluster bootstrap are in the code.

All steps run on CPU with `CUDA_VISIBLE_DEVICES=''` and one low-priority
thread; the official D8/D10/D12 trainings remain untouched.
