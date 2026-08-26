# Appendix B (DRAFT — not inserted into the paper)

**Fitting the adapters where they are used, in closed form and without training**

Status: measured and reproducible; awaiting a decision on whether it enters the
paper. Nothing in `~/FLEX-UF/paper` has been changed for it.

---

The ladder is trained under one distribution over exits and deployed under
another. At the 0.1 dB operating point the deployed allocation sends 60.9% of
tiles to $e_2$ and 21.7% to $e_3$, while `exit_weights` gives all six rungs the
same auxiliary weight and `forward_random_depth` samples uniformly over the
reachable ones. Measured as a divergence, KL(deployed ‖ training) is 0.966 nat
on the auxiliary path and 0.559 on the rate-distortion path: $e_2$ carries six
tiles in ten and receives one sixth of the gradient, while $e_0$ and $e_1$ take
a third of it and are unreachable under the clamp.

This is the covariate shift CalexNet (arXiv:2509.08318) identifies for
early-exit branches on a frozen backbone — *branches train on samples they will
never see at inference* — and its remedy is importance weighting onto the
distribution that actually reaches each branch. Here the remedy needs no
training at all, because every adapter in this decoder ends in a pointwise
linear map,

$$\mathrm{Adapter}_k(f) = f + W_k\,h_k(f),$$

with $h_k = f$ for the $1{\times}1$ adapter and $h_k = \mathrm{act}(\mathrm{pw\_in}\,f)$
for the FFN one, and the target is known: the deepest exit's raw feature, which
is what the shared head was fitted to consume and what makes $e_5$ bit-exact
with the released decoder. Solving for $W_k$ is then ridge regression, in the
manner of AdaRound and BRECQ (arXiv:2102.05426), which reconstruct a frozen
network layer by layer on a small calibration set rather than retraining it:

$$W_k^\* = \arg\min_W \sum_{x \in S_k} \big\|W h_k(x) - \big(f_{K-1}(x) - f_k(x)\big)\big\|^2 + \mu\|W\|^2 .$$

$S_k$ is the whole of it: only the tiles whose Lagrangian allocation actually
sends them to exit $k$, at a multiplier chosen on the calibration set so that
its exit usage matches the deployed one. Both conditions matter and were
measured separately.

**What it costs.** One forward pass over 900 OpenImages crops at five rates —
4500 tile decisions — and three linear solves of size $385\times385$. Two
minutes on one card. 443,520 parameters change, 0.98% of the decoder; the other
44,999,878 are untouched, and the bitstream is not involved at any point.

**What it buys**, on 53 CTC intra frames at 0.1 dB with a per-frame guarantee:

| configuration | mean MAC saving |
|---|---|
| baseline, epoch 9 | 28.32% |
| **closed-form refit** | **28.74%** |
| refit, prior from a held-out half of the video set | 28.71% |
| 6000-step fine-tune, usage-matched weights | 28.32% |
| 6000-step fine-tune, uniform weights | 28.32% |

**Three controls, and each removes a different explanation.** Refitting on all
tiles rather than the survivors gives +0.19 instead of +0.64 under a global
multiplier, so two thirds of the gain is the conditioning and not the refit.
Refitting on survivors selected at the *wrong* multiplier gives −0.01: the same
method, the same data, the wrong conditional, and the gain vanishes entirely.
And deriving the usage prior from a disjoint half of the video set moves the
result by 0.03 points, so nothing is taken from the test half.

**The negative result is the informative one.** A corrected fine-tune — anchor
on the released decoder, adapters at the main run's own learning rate, 6000
steps — returns exactly the baseline, with the usage prior and without it. The
adapters are not undertrained. They have converged, for an average distribution,
and stochastic gradient descent on OpenImages cannot be aimed away from it:
every step sees every tile, and the multiplier matching that makes the closed
form work has no analogue in the training loop. The closed form can state the
conditional directly. That is the whole of the difference, and it is why two
minutes beats an hour.

---

*Reproduce:* `flexplus/calex_refit.py` (refit), `flexplus/signalled_perframe.py`
(measurement), `flexplus/results/day_summary.json` (all seven configurations).
