"""ClassSR's routing losses (Eq. 1-4), adapted to an exit ladder.

Kong, Zhao, Qiao, Dong — *"ClassSR: A General Framework to Accelerate
Super-Resolution Networks by Data Characteristic"*, CVPR 2021.

    Eq. (1)  y = sum_i  f_i(x) * P_i(x)          soft weighted output
    Eq. (2)  L = w1*L1 + w2*L_c + w3*L_a          w1=2000, w2=1, w3=6
    Eq. (3)  L_c = - sum_{i<j} | P_i(x) - P_j(x) |            Class-Loss
    Eq. (4)  L_a = sum_i | sum_j P_i(x_j) - B/M |             Average-Loss

Why all three terms are needed
------------------------------
Eq. (1) makes routing differentiable by decoding a tile at *every* exit and
blending by probability, so a gradient reaches the router. But a soft blend is
not what runs at inference — inference takes an argmax — so training would
optimise a different function than it deploys.

Eq. (3) closes that gap. It rewards *spread* between the class probabilities,
pushing the distribution toward one-hot so the soft blend converges to the hard
choice. ClassSR: it "can greatly enlarge the probability gap between different
classification results so that the maximum probability value will be close to 1"
— it prefers [0.90, 0.05, 0.05] over [0.34, 0.33, 0.33].

Eq. (4) stops the degenerate solution. Nothing in Eq. (1) or (3) prevents the
router sending every tile to the deepest exit, which is optimal for distortion
and saves nothing. L_a penalises deviation from a balanced assignment across the
batch. ClassSR raises the batch size to 96 for exactly this reason — the term is
a batch statistic and needs enough samples to be meaningful. Note the paper's own
remark that the probability *sum* is used rather than a count "because statistic
number do not propagate gradients".

Our one addition
----------------
ClassSR's three branches have fixed, known costs, and it balances usage uniformly
(B/M per class). An exit ladder has a *continuum* of costs and we want a specific
point on the compute/quality frontier, not a uniform split. So `target_share`
lets the balance term aim at a chosen distribution, and `complexity_penalty` adds
the explicit Lagrangian term that makes the frontier sweepable:

    L_comp = beta * sum_i P_i(x) * C_i

with C_i the measured relative cost of exit i (`flexuf/cost.py`). Sweeping beta
traces the frontier; ClassSR did not need this because it had only three fixed
branches.
"""

from __future__ import annotations

from typing import Optional

import torch


def class_loss(probs: torch.Tensor) -> torch.Tensor:
    """Eq. (3) — Class-Loss. Rewards a peaked distribution.

    L_c = - sum_{i<j} |P_i - P_j|, averaged over tiles.

    Returned with the paper's sign, so it is *minimised* and its minimum is the
    one-hot corner. Normalised by the number of pairs so its scale does not move
    when K changes.
    """
    P = probs.shape[1]
    diff = (probs.unsqueeze(2) - probs.unsqueeze(1)).abs()  # [B, K, K]
    iu = torch.triu_indices(P, P, offset=1, device=probs.device)
    pair_sum = diff[:, iu[0], iu[1]].sum(dim=1)
    return -(pair_sum.mean()) / max(iu.shape[1], 1)


def average_loss(
    probs: torch.Tensor, target_share: Optional[torch.Tensor] = None
) -> torch.Tensor:
    """Eq. (4) — Average-Loss. Keeps every exit in use.

    L_a = sum_i | sum_j P_i(x_j) - B * target_i |

    `target_share=None` reproduces ClassSR's uniform B/M. A non-uniform target
    aims the ladder at a chosen operating point instead.
    """
    B, K = probs.shape
    if target_share is None:
        target = torch.full((K,), 1.0 / K, device=probs.device, dtype=probs.dtype)
    else:
        target = target_share.to(probs.device, probs.dtype)
        target = target / target.sum()
    return (probs.sum(dim=0) - B * target).abs().sum() / B


def complexity_loss(probs: torch.Tensor, costs: torch.Tensor) -> torch.Tensor:
    """Expected relative decode cost of the routing decision.

    The term ClassSR does not have. Minimising it pushes tiles shallower;
    balanced against distortion it is what draws the frontier.
    """
    return (probs * costs.to(probs.device).unsqueeze(0)).sum(dim=1).mean()


def soft_blend(x_hats: list[torch.Tensor], probs: torch.Tensor) -> torch.Tensor:
    """Eq. (1) — the differentiable output: sum_i f_i(x) * P_i(x).

    Args:
        x_hats: K tensors, each [P, 3, H, W] — the tile decoded at every exit.
        probs:  [P, K].
    """
    stacked = torch.stack(x_hats, dim=1)              # [P, K, 3, H, W]
    w = probs.view(probs.shape[0], probs.shape[1], 1, 1, 1)
    return (stacked * w).sum(dim=1)


def router_objective(
    mses_per_exit: torch.Tensor,
    probs: torch.Tensor,
    costs: torch.Tensor,
    *,
    w_image: float = 2000.0,
    w_class: float = 1.0,
    w_avg: float = 6.0,
    beta: float = 0.0,
    target_share: Optional[torch.Tensor] = None,
    normalize_image: bool = True,
) -> dict:
    """Eq. (2) with the complexity term added.

        L = w1*L_image + w2*L_c + w3*L_a + beta*L_comp

    Args:
        mses_per_exit: [P, K] — tile MSE at each exit, precomputed.
        probs:         [P, K] — router probabilities.
        costs:         [K]    — measured relative cost of each exit.

    Defaults w1=2000, w2=1, w3=6 are ClassSR's own values (Sec. 3.6).

    Why `normalize_image` defaults to True
    --------------------------------------
    ClassSR tuned those weights against *its* image loss, an L1 norm on SR
    outputs. Ours is MSE on the shifted-YCbCr tensor, a different scale, and the
    imbalance that creates is not subtle. Measured on an untrained checkpoint:

        w1*L_image  = 2000 * 4.9  ~ 9800
        w3*L_a      =    6 * 0.15 ~    0.9
        beta*L_comp =    1 * 0.62 ~    0.62

    Four orders of magnitude, so Class-Loss and Average-Loss do nothing at all
    and the router collapses onto whichever exit currently has the lowest MSE.
    At a *converged* model the imbalance disappears by itself (MSE ~1e-3 makes
    w1*L_image ~2, comparable to the rest) -- but routers here are trained
    against intermediate checkpoints of varying quality, so a weighting that is
    only correct at convergence is a trap.

    Dividing by the deepest exit's MSE makes L_image a dimensionless ratio: 1.0
    means "as good as decoding in full", 1.2 means "20% more error than full
    decode". That is comparable across checkpoints, across QPs and across the
    three experiments, and ClassSR's *relative* weights transfer intact.
    """
    if normalize_image:
        ref = mses_per_exit[:, -1:].detach().clamp_min(1e-10)
        l_image = (probs * (mses_per_exit / ref)).sum(dim=1).mean()
    else:
        l_image = (probs * mses_per_exit).sum(dim=1).mean()
    l_c = class_loss(probs)
    l_a = average_loss(probs, target_share)
    l_comp = complexity_loss(probs, costs)

    loss = w_image * l_image + w_class * l_c + w_avg * l_a + beta * l_comp
    return {
        "loss": loss,
        "l_image": l_image.detach(),
        "l_class": l_c.detach(),
        "l_avg": l_a.detach(),
        "l_comp": l_comp.detach(),
        "exit_share": probs.mean(dim=0).detach(),
    }
