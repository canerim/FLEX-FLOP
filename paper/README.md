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
| 4 — log-convexity test | `scripts/logconvex_check.py` | `results/logconvexity.json` |

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

**Proved as an equivalence, with the condition then measured.** Convexity in
(saving, dB) does *not* follow from the theorems — dB is a logarithm of MSE and
log of a convex function need be neither. Proposition 6 gives the exact
necessary and sufficient condition, `D·D'' ≥ (D')²`, equivalently
`d(1/λ)/dC ≥ 1/D`, equivalently "distortion falls at least exponentially in
compute". Table 4 tests it on the measured frontier: it holds at 100% of points
at every QP, with the margin growing from 0.28 at qp0 to 0.79 at qp63. So the
convexity is established — by an equivalence plus a verified hypothesis, not by
assertion.

An earlier draft stated only "we measure it, we do not prove it". That was
weaker than necessary: the condition is elementary once written down, and
writing it down also exposed a distinction the draft had missed — under
fixed-proportion mixing the hull has |K| vertices and dB is piecewise *concave*,
while under per-tile assignment it has up to N(|K|−1) and is smooth. Conflating
the two is how one becomes confident about a shape that is an artefact of vertex
spacing.

**Out of scope.** Theorem 3 bounds what adaptivity is worth, not what a decoder
that must *infer* the assignment can realise. In this system that gap is closed
by signalling the assignment at 1.3e-4 bpp, four orders of magnitude below the
frame rate — but that is engineering, not part of the characterisation.
