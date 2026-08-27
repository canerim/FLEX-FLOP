# Mixture-of-experts on the FFN — a brainstorm, with three measurements that decide it

**Nothing was changed.** Three read-only probes were run against
`runs/RECIPE512/ckpt_PIN_e9.pth.tar`; no weight was written and no training
started. Scripts: `flexplus/ffn_probe.py`, `flexplus/ffn_sparsity_probe.py`.

---

## 1. Why the FFN is the right target

The trunk is **89.4%** of the decode. Each of its twelve blocks is

```
dc  : Conv1×1(C→C) → WSiLU → DWConv3×3 → Conv1×1(C→C)    [residual]
ffn : Conv1×1(C→4C) → WSiLUChunkAdd → Conv1×1(C→C)       [residual]
```

At C = 384 the FFN is **5C² = 737k MAC/px** against the block's 7C² + 9C =
1.036M, so the FFN is **71% of a block** and therefore about **64% of the
whole decoder**. Two thirds of everything we are trying to save is in there.
Early exit skips whole blocks; a mixture of experts would route *within* one.
The two axes are orthogonal — depth against width — so they compose in
principle.

## 2. What the literature offers

Four families, and only two of them are training-free.

| work | what it does | needs training? |
|---|---|---|
| [MoEfication](https://aclanthology.org/2022.findings-acl.71/) (ACL 2022) | carves a trained FFN into experts by clustering co-activating neurons, then learns a router | router only |
| [CMoE](https://arxiv.org/abs/2502.04416) (2025) | same, but the router is built **analytically from activation statistics**; a 7B model in five minutes | no |
| [Deja Vu](https://arxiv.org/pdf/2310.17157) (ICML 2023) | predicts, per input, which FFN neurons matter, with a small learned predictor | predictor |
| [TEAL](https://arxiv.org/abs/2408.14690) (ICLR 2025) | magnitude-thresholds hidden states, 40–50% sparsity, **no training**, works on SiLU models | no |

CMoE is the closest in spirit to what we already did with the closed-form
adapter refit: build the routing from statistics rather than from gradients.
TEAL matters because our activation is WSiLU — `sigmoid(4x)·x` — and the
zero-counting that MoEfication relies on
[does not exist outside ReLU](https://openreview.net/pdf?id=osoWxY8q2E).

**Prior work to position against:**
[MoECodec](https://arxiv.org/abs/2606.21033) (June 2026) already replaces the
FFN layers of a *transformer* compression model with token-wise MoE, with
expert-choice routing and a spatial total-variation regulariser. It is
trained end to end and aimed at machine perception. Our angles that survive
beside it: a **convolutional** decoder, a **frozen** one, and a
**bitstream-preserving** constraint it does not carry.

## 3. Three measurements, and what they decide

### 3.1  The chunk-add already looks like four experts. It is not.

`WSiLUChunkAdd` is not a reshape. It is a **sum over four strided groups**:

$$\text{out}[j] \;=\; \sum_{r=0}^{3} \mathrm{WSiLU}\big(h[4j+r]\big)$$

so every output slot is an additive mixture of four sub-units, and the four
groups partition the hidden layer exactly. Dropping group *r* means not
computing one quarter of the expand, taking the FFN from 5C² to 4C². The
architecture appears to have carved the experts for us.

Measured over 4 CTC frames × 3 rates, all 12 blocks:

| | result |
|---|---|
| energy share per group | **22–28%**, i.e. 25% ± 3 everywhere |
| pairwise correlation between groups | **−0.30**, in every block |
| relative block error if one group is dropped | 6–12% (blocks 0–8), 24–31% (block 9), **127–161% (block 11)** |

The groups are not specialists. They are **balanced and anti-correlated** —
and −0.30 is within a whisker of the −1/3 that four zero-mean terms would show
if their sum were constant. The FFN's output is a small residue of four large,
mutually cancelling terms. Drop one and the residue is destroyed, catastrophically
so in the last block.

**Verdict: routing among the four groups is dead.** Not "expensive" — wrong.

### 3.2  Low rank is more expensive, not less

If the groups cannot be split, perhaps the maps are redundant. Singular values
of every block's FFN weights, rank needed for 99% of the energy:

| | conv1 [1536×384] | conv2 [384×384] | cost of the factorised form |
|---|---|---|---|
| range over 12 blocks | **323–351** / 384 | 213–290 / 384 | **1.16× the dense cost** |

A factorisation wins only below rank ≈ 307, and the spectrum does not decay
that far. **Verdict: low-rank factorisation is dead too**, and it loses by 16%.

### 3.3  Contextual sparsity is real — and this architecture has already spent it

Per feature position, how much of the hidden state's magnitude sits in how few
of the 1536 channels, and what a magnitude threshold actually costs the block:

| | share of channels | | keep 75% | keep 50% | keep 25% |
|---|---|---|---|---|---|
| 50% of magnitude | ~26% | blocks 0–8 | 0.3–0.8% | 2.4–4.3% | 6.3–8.9% |
| 90% of magnitude | ~70% | block 10 | 2.2% | 7.1% | 11.2% |
| 99% of magnitude | ~91% | block 11 | **10.3%** | **19.4%** | **24.8%** |

So contextual sparsity exists — keeping three quarters of the hidden channels
costs under 1% of the block output in nine of twelve blocks. TEAL would find
this.

**And it buys nothing here, for a reason that is worth writing down.**
In a standard transformer FFN the down-projection is 4C → C and costs 4C², so
zeroing hidden units skips columns and halves the layer. In this decoder the
chunk-add collapses 4C → C **for free** before the down-projection, so conv2
is only C² — one fifth of the FFN. Thresholding the hidden state therefore
saves nothing: the surface TEAL exploits has already been removed by design.

To save anything one must skip work in conv1, the 4C² expand — and that
requires *predicting* which of the 4C outputs will be large **before**
computing them. That is Deja Vu, and it needs a trained predictor.

## 4. What this leaves

The three probes say the same thing from three directions: **there is no
input-independent redundancy in this FFN.** The compression training has left
nothing structural on the table — no dispensable group, no low-rank shortcut,
and the one contextual shortcut has been designed away.

That is not a null result. It says the only exploitable axis is
input-dependence at a **coarser** granularity than the neuron, which is
exactly the axis early exit already uses, and it explains why the depth axis
has been the productive one all along.

### Directions that survive, ranked

1. **Nothing on the FFN, training-free.** State it and move on. The negative
   result is worth a paragraph in the paper: it tells a reader why we route
   depth rather than width, with numbers rather than taste.

2. **Deja Vu-style predictor on conv1, if training returns to the table.**
   Predict the top-*k* of 4C before computing them. The ceiling is real —
   keeping 50% costs 2–4% of the block in nine blocks — but it needs a trained
   predictor, a gather kernel over output channels, and it stacks a second
   prediction error on top of the router's. Expected value: moderate. Risk:
   high.

3. **New experts rather than carved ones.** Train a genuine MoE FFN — this is
   MoECodec's move, and it would mean retraining the decoder, changing the
   architecture, and giving up the frozen-backbone claim that is our whole
   position. Against the constraint we have set ourselves, this is not a
   variant of our work; it is a different paper.

4. **The last block is the exception worth a second look.** It is 10× more
   sensitive than the others on every probe — 127–161% error from a dropped
   group against 6–12% elsewhere, 10.3% from a 75% threshold against 0.3–0.8%.
   Everything the decoder does last matters disproportionately. That is a
   claim about where capacity should go, and it is measurable without any MoE
   at all.

### And the one place width might still pay

Not in the FFN — in the **depthwise 3×3**, which is 9C MAC/px, negligible in
arithmetic but the only spatial operator in the block and therefore the only
thing that makes tiling hurt. The floor we keep running into is made there.
Nothing in the MoE literature addresses it, which is itself informative:
our binding constraint is not the one that literature was written for.
