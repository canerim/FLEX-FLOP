"""Router v2: a learned view of everything the decoder already has.

What v1 got wrong, measured
---------------------------
v1 read a 1x1 of rank 16 off the stem, pooled it, and passed 33 numbers through a
64-unit MLP. Three defects, each found by measurement rather than taste:

1. **It ignored the entropy model.** Of the six hand-made signals, the two with
   the strongest correlation to the oracle's choice were `scales_mean` (-0.117)
   and `scales_max` (-0.097) -- the entropy model's predicted Gaussian scales.
   They are computed during the decode and were thrown away. So was the latent
   itself. The router was denied the one quantity that is literally a per-tile
   difficulty estimate.

2. **Capacity was in the wrong place.** The MLP runs on ONE vector per tile --
   40 of them for a 1080p frame at 256px tiles -- so its width is free, while the
   1x1 runs per pixel and is the only thing that costs. v1 kept the MLP narrow
   and the 1x1 narrow, paying for caution twice.

3. **Its objective was indirect.** Expected regret is the right thing to
   MINIMISE, but agreement with the oracle is what was asked for, and regret only
   reaches it via consequences. Cross-entropy to the oracle's own choice supplies
   the direct signal; weighting each tile by how much the choice COSTS keeps it
   honest, so the router spends its capacity where a mistake matters.

Cost, still the binding constraint
----------------------------------
    stem 1x1 384->48, at 1/64 of the pixel count      288 MAC/RGB-px   0.133%
    latent+scales 1x1 512->32, at 1/256                64              0.030%
    MLP, one vector per tile                            ~0
    ------------------------------------------------------------------------
                                                                       0.162%

GridSeamRepair is 0.951% and a trunk block 7.45%, so this is still small change
against the thing it is deciding about.

Which of those inputs carries the signal
----------------------------------------
The parameter-free rule in `scripts/raterank_curve.py` reads one number per tile
off the entropy coder -- how many bits that tile cost -- and beats this head.
That makes "why does the learned router exist" a fair question, and the only
honest answer is a measurement: give the head one group of inputs at a time and
see which group the agreement survives.

`inputs` selects the live groups. A group that is not live is zeroed AFTER its
projection, so every variant has the same architecture, the same parameter count
and the same optimiser state shape, and the ONLY difference between them is how
much the head is allowed to know. `bits` is a new group and a new pathway, so it
is built only when `inputs` is given at all: without it this class is, down to
the last bit, the 144,030-parameter head the paper reports, and every router
checkpoint already on disk still loads.

Note what `qp` is before reading anything into it: one number per FRAME, the
same for every tile in that frame. It can tell the head which operating point it
is at and it cannot, even in principle, tell two tiles of one frame apart. It is
therefore left live in the single-group variants, so that they differ in
PER-TILE information and in nothing else, and `inputs="qp"` is the control that
measures what agreement is reachable with no per-tile information at all.
"""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F

#: The head's input groups, in the order the docstring introduces them.
INPUT_GROUPS = ("stem", "latent", "scales", "bits", "qp")


def parse_inputs(spec) -> set:
    """Turn an ablation spec into the set of input groups that stay live.

    Accepts None (everything live), "all", a comma-separated list of group
    names, or any iterable of them, so the same spelling works on a command
    line and in code.
    """
    if spec is None:
        return set(INPUT_GROUPS)
    names = spec.split(",") if isinstance(spec, str) else list(spec)
    live: set = set()
    for raw in names:
        n = str(raw).strip()
        if not n:
            continue
        if n == "all":
            live |= set(INPUT_GROUPS)
        elif n in INPUT_GROUPS:
            live.add(n)
        else:
            raise ValueError(f"unknown router input group {n!r}; expected one "
                             f"of {INPUT_GROUPS} or 'all'")
    if not live:
        raise ValueError("at least one input group has to stay live")
    return live


def bits_features(bits: torch.Tensor) -> torch.Tensor:
    """Estimated bits per latent position -> the two channels the head reads.

    The parameter-free rival routes on b(t) = tile bits / mean tile bits, and
    its own fit says the variable that behaves linearly is log b rather than b
    (`scripts/raterank_curve.py`, where the fitted exponent comes out negative).
    So the head is handed both: the frame-normalised rate, whose mean over a
    tile IS b(t) exactly, and its log.

    Normalising per frame rather than globally is deliberate. The absolute rate
    level is a property of the frame, and the head already has qp for that; what
    decides a tile is how it compares with the rest of its own frame.
    """
    b = bits.sum(1, keepdim=True) if bits.shape[1] > 1 else bits
    n = b / b.mean(dim=(1, 2, 3), keepdim=True).clamp_min(1e-12)
    return torch.cat([n, n.clamp_min(1e-6).log()], dim=1)


def _pool(z: torch.Tensor, patch: int) -> torch.Tensor:
    """[B,C,h,w] -> [B*nh*nw, 2C] as (mean, std) per tile, unpatchify order."""
    B, C, h, w = z.shape
    v = (z.view(B, C, h // patch, patch, w // patch, patch)
          .permute(0, 2, 4, 1, 3, 5)
          .reshape(-1, C, patch * patch))
    return torch.cat([v.mean(-1), v.std(-1)], dim=1)


class StemRouterHeadV2(nn.Module):
    def __init__(self, stem_ch: int, latent_ch: int, num_exits: int,
                 min_exit: int = 0, r_stem: int = 48, r_lat: int = 32,
                 hidden: int = 256, inputs=None, r_bits: int = 8,
                 with_bits: bool | None = None):
        super().__init__()
        live = parse_inputs(inputs)
        # The bits pathway is a genuinely new input, so building it changes the
        # parameter count and the shape of every stored state dict. It is
        # therefore off whenever `inputs` is not given, which keeps the default
        # head identical to the one the paper reports and keeps every router
        # checkpoint on disk loadable. Once `inputs` IS given the head belongs
        # to the ablation family, and there the pathway is built even for the
        # variants that zero it, because the whole point of the ablation is that
        # the variants differ in information and in nothing else.
        if with_bits is None:
            with_bits = inputs is not None
        if inputs is None and not with_bits:
            # With no pathway to read the entropy coder's output there is
            # nothing for the group to be live on, and dropping it here is what
            # makes the no-argument head the pre-ablation one. Asking for bits
            # by name and refusing the pathway is a different thing entirely,
            # and is an error rather than a silent demotion.
            live.discard("bits")
        if "bits" in live and not with_bits:
            raise ValueError("inputs asks for 'bits' but with_bits is False")
        self.min_exit = min_exit
        self.with_bits = bool(with_bits)
        self.r_bits = r_bits
        #: Which groups are live, as a canonical string, so a run can be
        #: identified from its own checkpoint. It is a plain attribute and not a
        #: buffer on purpose: a buffer would add a key to state_dict and stop
        #: the pre-ablation checkpoints from loading strictly.
        self.inputs = ",".join(g for g in INPUT_GROUPS if g in live)
        #: 1.0 or 0.0 per group, applied after that group's projection. A live
        #: group is multiplied by exactly 1.0, which is exact in floating point,
        #: so a fully live head computes what it computed before the ablation
        #: existed; a dead one is multiplied by 0.0, which removes the
        #: information while leaving the parameters in place.
        self.gate = {g: float(g in live) for g in INPUT_GROUPS}
        self.proj_stem = nn.Conv2d(stem_ch, r_stem, 1)
        self.proj_lat = nn.Conv2d(2 * latent_ch, r_lat, 1)
        d = 2 * r_stem + 2 * r_lat + 1
        if self.with_bits:
            self.proj_bits = nn.Conv2d(2, r_bits, 1)
            d += 2 * r_bits
        self.mlp = nn.Sequential(
            nn.LayerNorm(d),
            nn.Linear(d, hidden), nn.SiLU(),
            nn.Linear(hidden, hidden), nn.SiLU(),
            nn.Linear(hidden, num_exits),
        )
        nn.init.normal_(self.mlp[-1].weight, std=1e-2)
        nn.init.zeros_(self.mlp[-1].bias)
        # Loss-free load balancing (arXiv:2408.15664 / OpenReview y1iU5czYpE):
        # a per-exit bias added to the scores BEFORE the decision, nudged by usage
        # rather than by a gradient. An auxiliary balance loss injects
        # interference gradients into the shared objective; this does not touch
        # the gradient at all. Registered as a buffer because it is updated by
        # rule, not learned.
        self.register_buffer("bias", torch.zeros(num_exits))

    def forward(self, stem, y_hat, scales, qp, feature_patch, latent_patch,
                bits=None):
        B = stem.shape[0]
        n_tiles = B * (stem.shape[2] // feature_patch) * (stem.shape[3] // feature_patch)
        if self.gate["stem"]:
            a = _pool(self.proj_stem(stem), feature_patch)
        else:
            # Skipping the convolution rather than multiplying its output by
            # zero, which is the same thing for both the value and the gradient
            # and does not spend the compute.
            a = stem.new_zeros(n_tiles, 2 * self.proj_stem.out_channels)
        if scales.shape[1] != y_hat.shape[1]:
            scales = scales[:, : y_hat.shape[1]]
        g_lat, g_sc = self.gate["latent"], self.gate["scales"]
        if g_lat and g_sc:
            # Both live: the fused convolution over the concatenation, exactly
            # as it was before the ablation, so the default path is unchanged
            # down to the last bit.
            p = self.proj_lat(torch.cat([y_hat, scales], 1))
        else:
            # A 1x1 over a concatenation is the sum of two 1x1s over the parts,
            # so splitting the weight lets one part be zeroed on its own without
            # adding or removing a parameter. The bias belongs to neither part
            # and is added once either way: it is a constant across tiles, so it
            # carries no per-tile information and gating it would be changing
            # capacity rather than information.
            C = y_hat.shape[1]
            w = self.proj_lat.weight
            terms = []
            if g_lat:
                terms.append(F.conv2d(y_hat, w[:, :C]))
            if g_sc:
                terms.append(F.conv2d(scales, w[:, C:]))
            p = self.proj_lat.bias[None, :, None, None].expand(
                B, w.shape[0], y_hat.shape[2], y_hat.shape[3])
            # `.contiguous()` when nothing was added to it, because _pool
            # reshapes and a broadcast view has no memory to reshape.
            p = sum(terms) + p if terms else p.contiguous()
        b = _pool(p, latent_patch)
        q = (qp.reshape(-1, 1).float() / 63.0).repeat_interleave(n_tiles // B, 0)
        q = q * self.gate["qp"]
        parts = [a, b]
        if self.with_bits:
            if self.gate["bits"]:
                if bits is None:
                    raise ValueError(
                        "inputs asks for 'bits' but the caller passed none")
                parts.append(_pool(self.proj_bits(bits_features(bits)),
                                   latent_patch))
            else:
                parts.append(a.new_zeros(n_tiles, 2 * self.r_bits))
        parts.append(q)
        logits = self.mlp(torch.cat(parts, dim=1)) + self.bias[None, :]
        if self.min_exit > 0:
            # -inf, not -1e4. A constant sentinel is a mask only while the
            # head's own logits stay well above it, and this head's did not:
            # nothing in a cross-entropy or a regret objective penalises a
            # common offset, one drifted in during training, and the raw
            # outputs settled near -10000. A typical row then read
            #
            #   [-10000.0, -10000.0, -10044.5, -10003.8, -10011.1, -10018.7]
            #
            # where the two "masked" entries are the LARGEST. After softmax
            # they held nearly all the mass, argmax picked one, and the caller's
            # clamp turned it into the cheapest real exit -- so a large share of
            # every configuration-B allocation went to the cheapest rung for a
            # reason unrelated to the tile (DECISIONS 89).
            #
            # -inf cannot drift. The row always has min_exit < num_exits finite
            # entries, so softmax and cross-entropy stay well defined.
            logits = logits.clone()
            logits[:, : self.min_exit] = float("-inf")
        return logits

    @torch.no_grad()
    def rebalance(self, chosen: torch.Tensor, target: torch.Tensor, rate=1e-2):
        """Nudge the bias so usage drifts toward `target` (the oracle's own mix).

        Balanced toward the ORACLE's distribution, not toward uniform. Uniform is
        what MoE wants because its experts are interchangeable; ours are not --
        at a high lambda the oracle genuinely sends everything to one exit, and
        forcing spread there would be forcing mistakes.
        """
        used = torch.bincount(chosen, minlength=self.bias.numel()).float()
        used = used / used.sum().clamp_min(1)
        self.bias += rate * (target - used)
        self.bias -= self.bias.mean()


def oracle_ce_loss(logits, mses, costs, lam: float, label_smooth: float = 0.0,
                   min_exit: int = 0):
    """Cost-sensitive cross-entropy to the oracle's choice.

    Plain CE treats every tile equally, but they are not: for many tiles two
    exits are within a hair of each other and picking either is fine, while for a
    few the wrong pick is most of the frame's error. Weighting each tile by the
    REGRET of its worst alternative concentrates capacity where the decision
    actually matters, and leaves the ties alone.

    Returns (loss, oracle_label, agreement).
    """
    lag = mses + lam * costs[None, :]
    best, k = lag.min(dim=1)
    # Exits below the split depth do not exist: `tiled_exit_mses` fills those
    # columns with the same decode as exit `min_exit`, and their costs are equal
    # too, so argmin can land on one of them arbitrarily. Since the mask is now
    # -inf, a label there would make the cross-entropy infinite. They are
    # duplicates of `min_exit`, so clamping is exact rather than a repair.
    if min_exit > 0:
        k = k.clamp(min=min_exit)
    # How much is at stake on this tile: spread between best and worst option.
    stake = (lag.max(dim=1).values - best)
    w = stake / stake.mean().clamp_min(1e-12)
    ce = F.cross_entropy(logits, k, reduction="none", label_smoothing=label_smooth)
    agree = (logits.argmax(1) == k).float().mean()
    return (w * ce).mean(), k, agree
