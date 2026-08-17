# `theory.tex` — the compute–distortion frontier of a multi-exit decoder

Two pages, NeurIPS form: numbered assumptions, propositions and theorems, each
theorem naming the assumptions it uses, every proof given.

§1–3 derivation · §4 experiments · §5 proofs

## Results

Referred to by name, not number: `amsthm` shares one counter across
environments, so the printed numbers move whenever an assumption is added.

| | statement |
|---|---|
| *Fixed-proportion mixing* | achieves exactly the convex hull of the `K` per-exit points |
| *Per-tile assignment* | achieves the Minkowski average of the `N` per-tile hulls |
| *Adaptivity gain* | their gap is `min-of-average − average-of-min ≥ 0`, zero iff every tile prefers the same exit |
| *Frontier recovery* | the frontier is the Fenchel conjugate of the Lagrangian value; a λ-sweep is exhaustive |
| *Convexity in (S, dB)* | convex **iff** `D·D'' ≥ (D')²`, i.e. iff distortion falls at least exponentially in compute |

*Adaptivity gain* is the point. `Δ(λ)` depends only on the per-tile distortions, so the
value of content adaptivity can be measured before any router exists — which
separates *is adaptivity worth anything here* from *does this router realise it*.

*Convexity in (S, dB)* is an equivalence, not an assertion: it does **not**
follow from the two hull results, because `dB` is a logarithm of `D` and the
log of a convex function need be neither convex nor concave. The condition is
then tested (Table 4) rather than assumed.

## Reproducing the tables

| table | script | output |
|---|---|---|
| 1 — per-exit dB vs QP | `scripts/why_qp.py` | `results/why_qp.json` |
| 2 — adaptivity gain `Δ/J` | `scripts/theory_check.py` | `results/theory_check.json` |
| 3 — marginal dB per +5% | `scripts/paper_curve.py` | `results/paper_curve_grid128.json` |
| 4 — log-convexity test | `scripts/logconvex_check.py` | `results/logconvexity.json` |

`python scripts/verify_theory_tables.py` re-checks all 62 table entries against
those files and exits non-zero on any mismatch. It parses each table out of the
`.tex` by its `\label`, so renaming or deleting a table fails loudly instead of
passing on nothing; and Table 3, which is an interpolation along the frontier
rather than a direct readout, is recomputed from the curve rather than compared
against itself.

Four decimals throughout, because that is the precision at which they were
measured.

## Scope

*Adaptivity gain* bounds what adaptivity is *worth*, not what a decoder that must
**infer** the assignment can realise. In this system that gap is closed by
signalling the assignment at `1.3e-4` bpp — four orders of magnitude below the
frame rate — so the oracle is attained rather than approached. That is
engineering, and is stated as a remark rather than folded into the
characterisation.

Distortion is always measured against the **released** decoder decoding the
**identical** bitstream: encoder, hyperprior and entropy model are byte-identical
(`max|diff| = 0`, asserted), so the only difference is synthesis.

Two decibels exist and the tables use one of them. Pooling every tile of every
frame into a single MSE matches the Lagrangian `J = D + λC` the theory is
written about; averaging a per-frame decibel is what `~/DCVC/test_video.py`
computes, and therefore what a published DCVC-UF number means. On an identical
assignment they differ by 0.023–0.033 dB — a quarter to a third of a 0.1 dB
budget, with pooling always the flattering one (`scripts/db_convention.py`).
The tables pool; anything quoted against published results does not. Both are
stored in every curve JSON as `db_vs_uf` and `db_vs_uf_per_frame`.
