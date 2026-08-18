# The seam artefact: what went wrong, and what fixed it

## The bargain, and the bill

Early exit needs tiles. A frame decoded as one piece pays whatever its hardest
region demands; cut it into tiles and an easy tile can leave the network early.
That is the whole idea, and it is not optional — without tiles there is nothing
to route.

The bill arrived immediately, and it was larger than the entire budget.

![the problem](figures/seam_problem.png)

Above is a 1080p frame decoded twice from the **same bitstream**, with the same
weights, at full depth everywhere. No tile exits early; nothing about the ladder
is switched on. The only difference is that the right-hand decode was done tile
by tile. Everything bright in the error map is the tiling and nothing else.

The grid is not subtle. It is the tile lattice, drawn onto the picture by the
decoder itself, and at qp 63 it costs **0.6768 dB** — nearly seven times the
0.1 dB budget this project works to. Before a single tile has saved a single MAC, the
method has already spent five times what it is allowed to.

That is the problem. The rest of this document is how it went from 0.68 dB to
0.06, why the two decisive steps were free, why the clever fix was rejected, and
why the module that costs something is on the wrong side of its own test.

Every number below is measured with **early exit switched off** — every tile at
full depth — so the tiling is the only difference from a full-frame decode and
nothing here is contaminated by the ladder.

## 1. Where it comes from

A 3×3 depthwise convolution at feature position (i, j) computes

```
y[i,j]  =  Σ_{u,v ∈ {−1,0,1}}  w[u,v] · f[i+u, j+v]
```

Decoded full-frame, `f[i+u, j+v]` are the neighbour's real features. Decoded per
tile, positions outside the tile **do not exist**, and the kernel is handed
whatever the padding rule invents.

Only the 3×3 depthwise has spatial extent. Everything else in a `DepthConvBlock`
is 1×1, and a 1×1 has no neighbours to miss:

| | MAC/pixel | share of the block |
|---|---|---|
| block total, `8C² + 9C` at C = 384 | 1,183,104 | 100% |
| the 3×3 depthwise, `9C` | 3,456 | **0.29%** |

That single fact shapes the rest of the design. It is why the exit adapters are
1×1 on purpose — they add capacity without adding a receptive field, so they
cost no seam — and it is why canvas coupling (§7) is affordable at all.

### The damage propagates inward

Each convolution extends the contaminated region by one ring. With `b` depthwise
layers running per tile on a tile of side `F`, the corrupted fraction is

```
1 − ((F − 2b) / F)²
```

At the shipped setting — `F = 32` feature pixels (256 RGB), `b = 8` per-tile
blocks — that is **0.750**: three quarters of the tile is within reach of at
least one invented value.

> The small-`b` approximation `≈ 4b/F` appears in some earlier notes. It is not
> valid here: at b = 8, F = 32 it gives 1.000 against the exact 0.750. Use the
> exact form.

So this is not a thin line of bad pixels at the border. It is a structured error
across most of the tile, which is what the amplified error maps show.

Two consequences drive everything below: the damage scales with the tile
**perimeter**, and with **how many layers run per tile** — that is, with the
split depth `j`.

## 2. How bad it was

The picture above, as a number. Pure seam penalty, 256 px tiles, dB above the
released decoder:

| padding | qp 0 | qp 32 | qp 63 |
|---|---|---|---|
| zeros (stock behaviour) | 0.1264 | 0.2866 | **0.6768** |

Note that it worsens with rate. At low rate the reconstruction is smooth and a
wrong neighbour costs little; at high rate the latent carries detail, the border
pixels have somewhere to fall, and the same mistake costs five times as much.
Which means the artefact is worst exactly where the method already struggles.

## 3. Padding is an estimator, and that is the useful way to see it

Padding is not a formatting detail. It is an estimator of the neighbour the tile
cannot see, and the seam penalty is that estimator's error. For a border column
`x[0]` with unseen neighbour `x[−1]`:

| mode | estimate of `x[−1]` | what it assumes |
|---|---|---|
| zeros | `0` | the signal is centred at zero — after the −0.5 shift, mid grey |
| replicate | `x[0]` | local constancy: a zeroth-order hold |
| linear | `2·x[0] − x[1]` | a locally linear signal: first-order extrapolation |
| arls | `a·x[0]`, `a` fitted per channel by least squares | an AR(1) process (arXiv:2502.12300) |

Measured, 256 px tiles, 40 CTC sequences, 80 frames:

| mode | qp 0 | qp 32 | qp 63 |
|---|---|---|---|
| zeros | 0.1264 | 0.2866 | 0.6768 |
| **replicate** | **0.0707** | **0.1234** | **0.2179** |
| linear | 0.1980 | 0.2857 | 0.3836 |
| arls | 0.0611 | 0.1074 | 0.1972 |

> **These numbers changed on 18 August, and the earlier ones were wrong.** This
> script installed the padding wrapper on the decoder's trunk groups and then
> computed its reference with `forward_full`, which runs those same modules — so
> the full-frame baseline was padded with the mode under test as well. On one
> 1080p frame at qp 63 that moved the reference by −0.186 dB and deflated the
> measured seam from 0.265 to 0.079. Every non-`zeros` row was affected;
> `zeros` was not, because wrapping with zeros is a no-op. That is exactly why
> the bug survived review: the one row that could be cross-checked against a
> second implementation was the one row that was already right. The fix is a
> separate, never-wrapped model for the reference. The table above is the
> re-measurement, now on all 40 sequences rather than the 10 that were on disk
> when the original was taken, and `scripts/seam_vs_qp.py` reproduces it
> independently (0.0687 / 0.1208 / 0.2161 for replicate at one frame per
> sequence).
>
> **Every conclusion below survived the correction.** `linear` is still worse
> than doing nothing clever — by more than before. `arls` still wins on quality
> and still loses on cost: its margin over replicate at qp 63 is 0.0207 dB
> against 0.0191 before, i.e. 0.0019 dB per point of decode either way.

**The `linear` row is the one to read twice.** A higher-order estimator is
*worse* — 2.5× worse than replicate at qp 63, and worse than doing nothing
clever at all. Extrapolating a gradient past a boundary amplifies whatever noise
sits on that boundary; assuming constancy does not. Guessing harder is not the
same as guessing better.

**`arls` wins on quality and was rejected on cost.** It is the best estimator
here — 0.1972 against replicate's 0.2179 at qp 63 — and it costs **+10.7% of
decode wall-clock**. That is 0.021 dB for 10.7% of the decode, when the entire
budget is 0.1 dB and the entire saving is ~30%. The trade does not close. It was
implemented, measured, and dropped.

## 4. Tile size: the largest lever, and it is free

From §1 the damage scales with the perimeter, so doubling the tile side should
roughly halve it. It does:

| mode, qp 63 | 128 px | 256 px | ratio |
|---|---|---|---|
| zeros | 1.1670 | 0.6768 | ×0.58 |
| replicate | 0.2125 | 0.2179 | — |

The 128 px column has **not** been re-measured against a clean reference yet — it
carries the same bug — so only the `zeros` row is comparable, and even that
crosses two different sequence sets (9 then, 40 now). The re-measurement is
queued. What the perimeter argument predicts is ×0.50; what the one clean pair
gives is ×0.58 across a changed test set.

This costs **nothing in compute** — a tiled decode's MAC count does not depend
on the tile size at all. What it costs is *routing granularity*: 40 tiles per
1080p frame at 256 px against 160 at 128 px. Fewer, larger tiles means locally
easy content gets averaged in with content that is not.

`BEST128` exists to measure that trade rather than argue it.

### The halo obeys the same law, which is why it is on the head only

A tile of side `F` carried with a halo of `h` computes `((F+2h)/F)²` as many
pixels:

| | F = 16 | F = 32 |
|---|---|---|
| h = 2 | 1.56× | 1.27× |
| h = 4 | 2.25× | 1.56× |

Applying 2.25× to the per-tile **trunk** would consume more than routing saves —
the arithmetic does not close. So the halo is carried on the **head** only,
which is 2.40% of MACs, and the adapters are trained to absorb the residual.

An earlier iteration of this project rejected the trunk halo at 1.745× cost. That
rejection was right about the number and wrong about the target: it haloed the
whole block rather than the 0.29% of it that has any spatial extent.

## 5. Grid seam repair: told where to look

A translation-invariant 3×3 applied to the stitched canvas has to *infer* which
pixels sit on a boundary, from content alone — and it applies the same correction
to the ~25% that are clean interior, where any correction is damage. It is being
asked a harder question than the one we have.

But the lattice is not unknown. `unpatchify` lays tiles on a fixed P×P grid from
the origin, identically at training and inference. So gate the correction by
position *within* a tile:

```
Repair(f)  =  f  +  G[i mod P, j mod P] · PW( WSiLU( DW3×3(f) ) )
```

- `G` is P×P = 256 scalars, shared across all 384 channels — **0.0007%** of the
  decoder's parameters, and no MAC beyond one broadcast multiply.
- `G` is initialised at `exp(−d/τ)`, with `d` the distance in feature pixels to
  the nearest tile edge. At step 0 the gate already sits on the seam; training
  refines it rather than discovering it.
- `PW` is zero-initialised, so the whole module starts as the identity and the
  warm start stays bit-exact against released DCVC-UF.

Cost: **0.95% of the decode**, charged inside every saving this project reports —
including the deepest exit, which is why our full-depth path costs 1.0095 stock
decodes rather than 1.0.

## 6. What the repair is actually worth — an uncomfortable measurement

Repair on against off, 8 sequences at qp 63, same latents and same exit maps:

| regime | repair OFF | repair ON | Δ |
|---|---|---|---|
| every tile at full depth | 0.0672 | 0.0750 | **+0.0078 — worse** |
| routed at 0.1 dB | 0.1259 | 0.1239 | −0.0019 |

So it earns 0.002 dB in the regime that matters, for 0.95% of the decode, and it
makes the **floor** worse — the same floor that already consumes 30% of the
budget at qp 63.

Per point of decode: **0.0020 dB**. `arls` was rejected at **0.0018 dB** per
point. By the rule this project used to reject `arls`, grid seam repair is on the
wrong side of its own test.

**The caveat that keeps this from being a verdict:** the module was trained
jointly with everything else, so switching it off at inference is not the same as
training without it. The clean test is a run with `seam_repair=none` at 256 px,
and none of the six live runs is that — `VERBATIM` uses `none` but differs in six
other flags. It is on the next-experiment list.

## 7. Canvas coupling: implemented, measured, and not used here

If the seam comes from a 3×3 meeting invented values, the complete fix is to give
it the real neighbour. Canvas coupling does exactly that, and only for the 3×3 —
the 0.29% of the block that needs it.

- cost: **+0.03%** of the decode
- at uniform depth, full-frame against tiled: `max |difference| = 0.0`

The seam does not shrink under it; it stops existing.

**But `tile_coupling = False` in all six runs.** Every result in this repository
uses replicate padding plus grid repair, not coupling. It is an available option
and not what these numbers describe — a distinction an earlier version of the
deck blurred.

## 8. Where it ended up

| step | qp 63 seam penalty | cost |
|---|---|---|
| stock behaviour: 128 px tiles, zeros | 1.1670 dB | — |
| → 256 px tiles | 0.6768 | free |
| → replicate padding | **0.2179** | free |
| → training the ladder with the seam present | **0.0583** | free |
| → grid seam repair | ≈0.0008 dB at best (§6) | 0.95% |

From eleven times the budget to well inside it — and every step that mattered
cost nothing. The largest single factor is the one that is easiest to overlook:
**training**. Replicate leaves 0.218 dB and the trained ladder leaves 0.058, so
more than two thirds of what replicate could not fix was absorbed by the weights
learning to live with it. No module in this document removes as much.

## 9. How we know it works

- **Uniform-depth equivalence.** With canvas coupling and every tile at the same
  depth, tiled decode and full-frame decode agree to `max|diff| = 0.0`. Without
  coupling the residual is the 0.107 dB above.
- **The warm start is unaffected.** Seam repair and the adapters are
  zero-initialised, so at step 0 the ladder reproduces released DCVC-UF exactly
  (`tests/test_reference_is_the_release.py`, `max|diff| = 0.0` at qp 0/32/63).
- **The pixels.** `scripts/seam_patches.py` crops a 192 px window across a tile
  boundary and decodes it under each padding mode, with error maps against the
  full-frame decode of the same latent.

![seam at pixel scale](figures/seam_patches.png)

## Reproducing

```bash
python scripts/seam_patches.py --device cuda:0     # the pixel-scale figure
# the tables come from results/ctc_seam_p256.json and ctc_seam_p128.json
```
