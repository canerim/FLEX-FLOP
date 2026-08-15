# FLEX-UF

Content-adaptive early-exit decoding for **DCVC-UF** (Microsoft, CVPR 2026).

The goal is one sentence: **spend far fewer decoder FLOPs, lose almost no PSNR**,
by letting each spatial tile leave the decoder at the depth its content actually
needs instead of paying the full 12-block trunk everywhere.

This is the successor to [FLEX-FLOP](https://github.com/circuitmaster/Flex-Flop),
which established the idea on DCVC-RT. Here it is rebuilt on UF and, unlike the
predecessor, **trained from scratch on Microsoft's own recipe** rather than
warm-started from a released checkpoint.

Every design decision, with its reasoning and the measurement behind it, is in
[DECISIONS.md](DECISIONS.md).

---

## The opportunity, measured

`scripts/mac_audit.py` puts forward hooks on every convolution of UF's
`IntraDecoder` and counts MACs from real tensor shapes. On a 1920×1088 frame:

| part | GMAC | share |
|---|---:|---:|
| opening upsample | 37.01 | 8.16% |
| **12-block trunk** | **405.64** | **89.44%** |
| head | 10.89 | 2.40% |
| **total** | **453.54** | 100% |
| *of which pointwise 1×1* | *452.02* | *99.66%* |

Nine tenths of the decode is twelve identical blocks that every tile pays for
regardless of difficulty, and virtually all of it is pointwise channel mixing.
Since the upsample and head always run, they set a floor of 10.56%, so exiting
after *g* of 12 blocks costs `0.1056 + (g/12)·0.8944`:

| exit after | 2 blocks | 4 | 6 | 8 | 10 | 12 |
|---|---:|---:|---:|---:|---:|---:|
| **saved** | **74.5%** | 59.6% | 44.7% | 29.8% | 14.9% | 0% |

## Architecture

```
y_hat ──> upsample ──> groups 0..j-1  ────────────────> shared, full-frame
                                        │                (no seams, no saving)
                                   patchify (128×128 RGB tiles, pure reshape)
                                        │
                            groups j..K-1, per tile ──> tiles drop out at their exit
                                        │                (this is the saving)
                                   1×1 adapter
                                        │
                                   stitch ──> head ──> PixelShuffle(8) ──> RGB
```

- **K = 6 exits** over the 12 trunk blocks (b = 2 blocks per exit).
- **Adapter = a single trainable 1×1 convolution**, residual, zero-initialised.
  A DepthConvBlock is `8C²+9C` MAC/px of which the FFN is 74.8% and the only
  spatial operator is 0.3% — so an early exit loses almost purely *pointwise*
  capacity, and a 1×1 is the matched replacement. It also has **no receptive
  field**, so it adds exactly zero patch-boundary penalty.
- **Zero-init** means an untrained adapter is the identity, so the deepest exit
  is **bit-exact stock UF** (verified, `max|Δ| = 0.0`).

## Following ClassSR

[ClassSR](https://arxiv.org/abs/2103.04039) decomposes an image into sub-images,
scores each one's difficulty, and routes it to a branch of matching capacity. The
structure is the same here; the difference is that our branches are *prefixes of
one decoder*, so routing costs no extra parameters and the deepest branch is the
unmodified codec.

Its routing losses are used directly: **Eq. (3) Class-Loss** (pushes the
distribution toward one-hot so the trained soft blend matches the deployed
argmax) and **Eq. (4) Average-Loss** (stops the degenerate solution of sending
everything to the deepest exit). We add a complexity term `β·Σᵢ Pᵢ·Cᵢ` because a
ladder has a continuum of costs where ClassSR had three fixed branches.

## Training objective

From Scardapane et al., *"Why should we add early exits to neural networks?"*
(Cognitive Computation 2020), joint training:

> **Eq. (6)**  `f* = arg min { L + Σᵢ αᵢ·Lᵢ }`
> **Eq. (7)**  `Lᵢ = Σₙ ℓ(yₙ, cᵢ(xₙ))`

with `ℓ` instantiated as Microsoft's own RD cost `λ·mse + bpp`.

One trap, handled: **bpp is identical at every exit**, because entropy decoding
runs once, full-frame, before the decoder is called. Writing Eq. (6) naively sums
the same rate term K times against K different distortion terms, multiplying the
effective weight on rate by K and moving the model off the operating point the
λ's encode. Normalising by the total weight makes it exactly

```
L = λ · (weighted mean over exits of mse_k) + bpp
```

so Microsoft's RD point is preserved and the objective only redistributes
distortion supervision across the ladder.

## The halo, and why it is where it is

A tile decoded in isolation has its borders convolved against padding. The fix is
a halo of real context — but a haloed tile computes `((F+2h)/F)²` times as many
pixels as it keeps, and placement decides whether that is affordable. Measured
net saving (`flexuf/cost.py`, j=2, 128px tiles, halo 2 latent px):

| exit | halo at head | halo tapered in trunk | halo in trunk |
|---:|---:|---:|---:|
| 3 | 28.9% | 6.1% | **−9.5%** |
| 5 | 0.0% | −22.1% | **−74.5%** |

Carrying the halo through the per-patch trunk makes the decode *more expensive
than decoding everything in full*. So the halo goes where it is affordable:

- **router input — 4 latent px, generous.** The router reads the latent and costs
  ~0.009% of the decode, so context is free, and clean per-tile statistics are
  what stop it routing on border artefacts.
- **head — 2 latent px.** 2.40% of MACs; FLEX measured a 1-px feature halo
  removes 99.1% of the seam.
- **per-patch trunk — none.** The arithmetic does not allow it.

## Experiments

Three runs, one per idle GPU, each differing from E1 in exactly one variable.

| | GPU | j | tile | halo | adapter | tests |
|---|---:|---:|---|---|---|---|
| **E1** | 4 | 2 | 128px | 2 lat | 1×1 | headline: max routable range |
| **E2** | 6 | 4 | 128px | 2 lat | 1×1 | the split-depth axis |
| **E3** | 7 | 2 | 64px | 2 lat | 1×1 | the tile-size axis (Boundary Law) |

GPUs 0/1/2/5 are other people's jobs and GPU 3 holds 44 GB belonging to another
user despite reading 0% utilisation. Runs are pinned with `CUDA_VISIBLE_DEVICES`
so they cannot touch another card.

## Recipe fidelity

`train_flexuf_image.py` reproduces `~/DCVC/train_image.py` exactly — the 105-epoch
schedule verbatim, AdamW, lr 2e-4→1e-6, 256×256 crops until epoch 90 then
512×512, batch 16, `clip_grad_norm_(0.1)` with non-finite batches skipped,
λ log-spaced 10→2048 across the 64 QPs, uniform QP sampling per sample. The loss
is the single deliberate change and is normalised to preserve the RD point.

One forced deviation: the README's `cu130` PyTorch cannot run on this machine's
535 driver (CUDA 13 needs r580+), so `cu124` is used. Training is pure PyTorch,
so this affects nothing but the bitstream extensions.

## Controls

Asserted with zero tolerance, because each one silently invalidates everything
downstream if it drifts:

| control | result |
|---|---|
| warm-start round-trip: deepest exit vs stock UF | `max|Δ| = 0.0` |
| untrained adapters are the identity at all 6 exits | `max|Δ| = 0.0` |
| j=K hybrid path vs full decode (patchify/stitch/head wiring) | `max|Δ| = 0.0` |
| patchify → unpatchify round-trip | `max|Δ| = 0.0` |

## Layout

```
flexuf/
  config.py            every knob, with the reason for its default
  model.py             FlexUFIntra — DMCI with a multi-exit decoder
  losses.py            Eq.(6)-(7) joint objective, RD-normalised
  cost.py              measured MAC accounting, halo charged honestly
  backbone/
    decoder.py         the K-exit ladder, hybrid j-split decode
    warmstart.py       exact key remap from stock UF + bit-exactness check
  router/
    router.py          ClassSR Class-Module: tile signals + tiny MLP
    losses.py          ClassSR Eq.(3) Class-Loss, Eq.(4) Average-Loss
scripts/
  mac_audit.py         measures where the decoder's compute actually goes
  prepare_openimages.py  tarball -> description.json the recipe expects
  pipeline.sh          download -> extract -> prepare -> launch, autonomous
  launch_experiments.sh  the three runs, pinned to GPUs 4/6/7
  status.sh            one-screen health, incl. the exit-collapse diagnostic
tests/
  test_equivalence.py  the zero-tolerance controls
train_flexuf_image.py  Microsoft's recipe, multi-exit objective
DECISIONS.md           every step and why it was taken
```

## References

- **DCVC-UF** — [arXiv:2606.04410](https://arxiv.org/abs/2606.04410), [microsoft/DCVC](https://github.com/microsoft/DCVC)
- **ClassSR** — Kong, Zhao, Qiao, Dong, CVPR 2021, [arXiv:2103.04039](https://arxiv.org/abs/2103.04039)
- **Early exits** — Scardapane, Scarpiniti, Baccarelli, Uncini, Cognitive Computation 2020, [arXiv:2004.12814](https://arxiv.org/abs/2004.12814)
- **FLEX-FLOP** — the predecessor on DCVC-RT
