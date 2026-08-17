# `paper/theory.tex`

An exact characterisation of what content-adaptive early exit can achieve, in
NeurIPS format: numbered assumptions, propositions and theorems, each theorem
stating the assumptions it needs, all proofs given.

Two pages: derivation (§1–3), experiments (§4), proofs (§5).

## Reproducing every number

| table | script | output |
|---|---|---|
| 1 — per-exit dB vs QP | `scripts/why_qp.py` | `results/why_qp.json` |
| 2 — adaptivity gain Δ/J | `scripts/theory_check.py` | `results/theory_check.json` |
| 3 — marginal dB per +5% | `scripts/paper_curve.py` | `results/paper_curve_grid128.json` |

Every figure in the tables was checked against those files programmatically
before the paper was committed; the tables carry four decimals because that is
the precision at which they were measured, and rounding them to three made the
check ambiguous.

## What the paper does and does not claim

**Claimed and proved.** The achievable (cost, distortion) set under fixed
proportions is the convex hull of the per-exit points; under per-tile assignment
it is the Minkowski average of the per-tile hulls; the gap between the two is
exactly `min-of-average − average-of-min ≥ 0`, vanishing iff every tile prefers
the same exit; the frontier is recovered from the Lagrangian value by Fenchel
conjugacy and sweeping λ is exhaustive.

**Measured, not proved.** That the frontier is convex in (saving, dB). It is
convex in (cost, MSE) by the theorems, but dB is a logarithm of MSE and log of a
convex function need be neither convex nor concave. The paper says so in
Remark 6 rather than quietly implying the stronger statement.

**Out of scope.** Theorem 3 bounds what adaptivity is worth, not what a decoder
that must *infer* the assignment can realise. In this system that gap is closed
by signalling the assignment at 1.3e-4 bpp, four orders of magnitude below the
frame rate — but that is engineering, not part of the characterisation.
