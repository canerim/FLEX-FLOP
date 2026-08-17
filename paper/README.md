# `theory.tex` — the compute–distortion frontier of a multi-exit decoder

Two pages, NeurIPS form: numbered assumptions, propositions and theorems, each
theorem naming the assumptions it uses, every proof given.

§1–3 derivation · §4 experiments · §5 proofs

## Results

| | statement |
|---|---|
| **Prop. 2** | Fixed-proportion mixing achieves exactly the convex hull of the `K` per-exit points |
| **Thm. 3** | Per-tile assignment achieves the Minkowski average of the `N` per-tile hulls |
| **Thm. 4** | Their gap is `min-of-average − average-of-min ≥ 0`, zero iff every tile prefers the same exit |
| **Prop. 5, Cor. 6** | The frontier is the Fenchel conjugate of the Lagrangian value; a λ-sweep is exhaustive |
| **Prop. 7** | `dB(S)` is convex **iff** `D·D'' ≥ (D')²`, i.e. iff distortion falls at least exponentially in compute |

Theorem 4 is the point. `Δ(λ)` depends only on the per-tile distortions, so the
value of content adaptivity can be measured before any router exists — which
separates *is adaptivity worth anything here* from *does this router realise it*.

Proposition 7 is an equivalence, not an assertion: convexity in `(saving, dB)`
does **not** follow from Theorems 3–4, because `dB` is a logarithm of `D` and the
log of a convex function need be neither convex nor concave. The condition is
then tested (Table 4) rather than assumed.

## Reproducing the tables

| table | script | output |
|---|---|---|
| 1 — per-exit dB vs QP | `scripts/why_qp.py` | `results/why_qp.json` |
| 2 — adaptivity gain `Δ/J` | `scripts/theory_check.py` | `results/theory_check.json` |
| 3 — marginal dB per +5% | `scripts/paper_curve.py` | `results/paper_curve_grid128.json` |
| 4 — log-convexity test | `scripts/logconvex_check.py` | `results/logconvexity.json` |

Every figure in the tables is checked against these files programmatically
before commit. Four decimals throughout, because that is the precision at which
they were measured.

## Scope

Theorem 4 bounds what adaptivity is *worth*, not what a decoder that must
**infer** the assignment can realise. In this system that gap is closed by
signalling the assignment at `1.3e-4` bpp — four orders of magnitude below the
frame rate — so the oracle is attained rather than approached. That is
engineering, and is stated as a remark rather than folded into the
characterisation.

Distortion is always measured against the **released** decoder decoding the
**identical** bitstream: encoder, hyperprior and entropy model are byte-identical
(`max|diff| = 0`, asserted), so the only difference is synthesis.
