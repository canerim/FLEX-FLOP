"""The multi-exit training objective.

Two papers meet here.

1. Scardapane, Scarpiniti, Baccarelli, Uncini — *"Why should we add early exits
   to neural networks?"*, Cognitive Computation 12 (2020), arXiv:2004.12814.
   Section 5 is a taxonomy of ways to train multi-exit networks; its **joint
   training** formulation is what we use.

       Eq. (6)   f* = arg min  { L  +  sum_{i=1}^{L-1} alpha_i * L_i }
       Eq. (7)   L_i = sum_{n=1}^{N} l( y_n , c_i(x_n) )

   L is the final (deepest) exit's loss; each L_i is an *auxiliary* — the paper
   calls it a companion — loss on an intermediate exit, weighted by alpha_i.

2. Kong, Zhao, Qiao, Dong — *"ClassSR"*, CVPR 2021. Supplies the routing losses
   used when the Class-Module (our router) is trained: Eq. (3) Class-Loss and
   Eq. (4) Average-Loss. See `flexuf/router/losses.py`.

Instantiating Eq. (7) for a compression model
---------------------------------------------
The per-sample loss `l` is Microsoft's own rate-distortion cost
(`~/DCVC/src/utils/common.py:166-171`):

    l = lambda * mse + bpp

so the auxiliary loss at exit k is  L_k = mean_n( lambda_n * mse_k,n + bpp_n ).

One subtlety that must be handled or the recipe is silently changed: **bpp is
identical at every exit**, because entropy decoding runs once, full-frame, before
the decoder is called. Writing Eq. (6) naively would therefore sum the same rate
term K times while summing K *different* distortion terms, multiplying the
effective weight on rate by K and moving the model to a different point on the RD
curve than Microsoft's lambdas specify.

We avoid that by normalising Eq. (6) by its total weight:

    L_total = [ L_{K-1} + sum_k alpha_k L_k ] / [ 1 + sum_k alpha_k ]

which is algebraically

    L_total = lambda * (weighted mean of mse_k)  +  bpp

i.e. exactly Microsoft's loss with the distortion term replaced by a weighted
mean over exits. The RD operating point the lambdas encode is preserved exactly;
all the objective does is spread distortion supervision across the ladder.
"""

from __future__ import annotations

from typing import Optional, Sequence

import torch


def exit_weights(
    num_exits: int,
    aux_weight: float = 1.0,
    schedule: str = "constant",
    epoch: int = 0,
    warmup_epochs: int = 10,
    device=None,
) -> torch.Tensor:
    """alpha_i of Eq. (6), plus the implicit weight 1.0 on the deepest exit.

    Returns a length-K vector; index K-1 is always 1.0 (the "final loss" L).

    Why a warmup option exists: FLEX-FLOP's from-scratch attempt collapsed with
    every exit stuck near 26 dB and no differentiation between them. Ramping
    alpha from 0 lets the network first become a competent codec at full depth —
    establishing the quality anchor the frontier is measured against — before it
    is also asked to be good when truncated. `schedule="constant"` reproduces
    MSDNet's finding that equal weights "work well in practice"; use it as the
    default and treat warmup as the mitigation if differentiation fails.
    """
    # Observation for a future run, NOT acted on here.
    #
    # Every exit gets the same weight, and under a j-split the first j+1 exits
    # all cost the same: exit_costs is 0.5716 three times over for K=6, j=2,
    # because groups 0..j-1 run full-frame for every tile whatever the
    # assignment. Exit 2 dominates 0 and 1 on average -- 0.167 dB below the
    # release at qp0 against 0.409 and 0.606 -- so a third of the auxiliary
    # loss's exit weight trains exits that can never save more compute than
    # exit 2 and are usually worse.
    #
    # The fraction is a property of the j-split, not of K: K=6/j=2 has 2 of 6
    # dominated and K=12/j=4 has 4 of 12, both a third. So a finer ladder does
    # not waste proportionally more -- FINE12 buys 8 distinct operating points
    # against BEST's 4 at the same overhead, which is what it was built to do.
    # (I expected the opposite and checked before writing it down.)
    #
    # Not obviously waste: on roughly an eighth of the tiles that take the
    # cheapest cost, exit 0 or 1 genuinely reconstructs better than exit 2, and
    # that choice is free because the three share a cost. So the question is
    # whether their weight should be reduced rather than removed, and it is a
    # question for an experiment, not for an edit to a running one.
    w = torch.ones(num_exits, dtype=torch.float32, device=device)
    a = aux_weight
    if schedule == "warmup" and warmup_epochs > 0:
        a = aux_weight * min(1.0, epoch / warmup_epochs)
    w[: num_exits - 1] = a
    return w


def multi_exit_rd_loss(
    mses: Sequence[torch.Tensor],
    bpp: torch.Tensor,
    lambdas: torch.Tensor,
    weights: Optional[torch.Tensor] = None,
) -> dict:
    """Eq. (6)-(7) with l = lambda*mse + bpp, weight-normalised.

    Args:
        mses:    K tensors of shape [B] — per-sample MSE at each exit.
        bpp:     [B] — shared across exits.
        lambdas: [B] — per-sample lambda, drawn with the QP (image_dataset.py:58).
        weights: [K] — alpha_i from :func:`exit_weights`; None means all ones.

    Returns a dict with the scalar `loss` plus per-exit diagnostics.
    """
    K = len(mses)
    if weights is None:
        weights = torch.ones(K, dtype=torch.float32, device=bpp.device)
    weights = weights.to(bpp.device)
    # An exit the caller did not decode arrives as None. It can only arrive that
    # way when its weight is zero, so dropping it and its weight together leaves
    # the weighted mean and the normaliser exactly as they were.
    if any(m is None for m in mses):
        keep = [i for i, m in enumerate(mses) if m is not None]
        assert float(weights[[i for i in range(K) if i not in keep]].abs().sum()) == 0.0, \
            "an exit with a non-zero weight was not decoded"
        mses = [mses[i] for i in keep]
        weights = weights[keep]
        K = len(mses)
    w_sum = weights.sum()

    stacked = torch.stack(list(mses), dim=0)              # [K, B]
    weighted_mse = (weights[:, None] * stacked).sum(0) / w_sum   # [B]

    costs = lambdas * weighted_mse + bpp
    loss = costs.mean()

    return {
        "loss": loss,
        "costs": costs,
        "weighted_mse": weighted_mse.mean().detach(),
        "per_exit_mse": stacked.mean(dim=1).detach(),
        "bpp": bpp.mean().detach(),
        "weights": weights.detach(),
    }


def per_exit_rd(
    mses: Sequence[torch.Tensor], bpp: torch.Tensor, lambdas: torch.Tensor
) -> torch.Tensor:
    """The raw L_k of Eq. (7), unweighted, for logging the ladder's shape.

    Monitoring these is how the "all exits collapse to the same quality" failure
    is caught early: a healthy ladder has L_0 > L_1 > ... > L_{K-1} with a clear
    spread, and the FLEX collapse showed up as all K values sitting on top of
    each other.
    """
    return torch.stack([(lambdas * m + bpp).mean() for m in mses])


def psnr_from_mse(mse: torch.Tensor) -> torch.Tensor:
    """MSE (on the [-0.5, 0.5] YCbCr scale UF trains in) -> dB.

    UF's `get_mse` operates on the shifted YCbCr tensor whose dynamic range is 1,
    so PSNR = 10*log10(1/mse) with no extra peak term.
    """
    return 10.0 * torch.log10(1.0 / mse.clamp_min(1e-10))


def ladder_distill_loss(feats, teacher: str = "adjacent"):
    """Each exit's adapted feature should look like a deeper exit's.

    Why this is worth a term of its own
    -----------------------------------
    The adapters are currently supervised only through pixels: a 3-channel target,
    at the far end of a head that mixes 384 channels down to 192 and then shuffles
    them by 8. That is a long, lossy path for a gradient to travel, and the entire
    job of an adapter is stated much more directly in feature space: *produce what
    the skipped blocks would have produced*, because the head downstream is fixed
    and was fitted to exactly that.

    The multi-exit self-distillation literature reaches the same place from the
    classification side -- distil the deep exit into the shallow ones -- and adds
    a caveat this implementation takes seriously: too large a student-teacher gap
    HURTS the shallowest exits (FITEE 2024, "Multi-exit self-distillation with
    appropriate teachers"). Hence `adjacent`, where exit k imitates exit k+1 and
    the chain carries the rest. That also happens to match our structure exactly:
    a shallow exit is literally a prefix of a deep one, so consecutive exits are
    one group apart by construction rather than by choice.

    Normalised by the teacher's own variance so the weight means the same thing at
    every exit and every QP; an unnormalised version would silently weight
    high-energy features more, which is the opposite of what is wanted -- the hard
    tiles are not the bright ones.
    """
    total, n = 0.0, 0
    K = len(feats)
    for k in range(K - 1):
        t = feats[K - 1] if teacher == "deepest" else feats[k + 1]
        t = t.detach()
        total = total + ((feats[k] - t) ** 2).mean() / t.var().clamp_min(1e-8)
        n += 1
    return total / max(n, 1)
