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
import torch.nn.functional as F


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


def regret_objective(
    mses_per_exit: torch.Tensor,
    logits: torch.Tensor,
    costs: torch.Tensor,
    *,
    lam: float,
    tau: float = 1.0,
    hard: bool = True,
    w_avg: float = 0.0,
) -> dict:
    """Minimise the gap to the oracle directly, instead of a surrogate for it.

    The oracle this project measures against is not a soft blend. For each tile it
    takes

        k* = argmin_k ( mse_k + lam * C_k )

    and the router's whole job is to reproduce that choice from signals. ClassSR's
    Eq. (2) does not state that job: it minimises an expected distortion, then adds
    a Class-Loss whose purpose is to *repair* the mismatch between the soft blend
    used in training and the argmax used at inference, and an Average-Loss to stop
    the result collapsing. Three terms and three weights, none of which is the
    quantity we actually care about.

    The quantity we care about has a name -- expected regret:

        L = sum_i P_i * [ (mse_i + lam*C_i) - min_k (mse_k + lam*C_k) ]

    Every term is non-negative, it is zero exactly when P puts all its mass on k*,
    and its VALUE is the excess Lagrangian cost the router is paying against the
    oracle. So it is not only the right objective, it is also the right progress
    metric: "0.004" means "0.004 above the oracle", in the units of the frontier.
    Sweeping `lam` traces the frontier the same way beta did, but each point is now
    a well-posed problem rather than a balance of three surrogates.

    Class-Loss is dropped because the mismatch it patches is removed at the source:
    with `hard=True` the forward pass uses a Gumbel-Softmax straight-through sample,
    so training decides exactly as inference does -- one exit, chosen -- while
    gradients still flow through the soft probabilities. This is the standard fix
    for discrete gating in spatially adaptive inference (Verelst & Tuytelaars,
    "Dynamic Convolutions: Exploiting Spatial Sparsity for Faster Inference",
    CVPR 2020, arXiv:1912.03203), and the structure there -- a small gate choosing
    per spatial unit whether to spend compute -- is ours exactly.

    Average-Loss is kept but OFF by default. ClassSR needs it because its branches
    are separate networks that receive no gradient when unused. Our exits share
    weights by construction and the decoder is frozen here, so an unused exit
    degrades nothing; forcing usage would mean deliberately routing tiles to exits
    the objective says are wrong. It stays available (`w_avg > 0`) because that
    argument should be checked rather than believed.

    Args:
        mses_per_exit: [P, K] tile MSE at each exit, from the frozen decoder.
        logits:        [P, K] raw router outputs (NOT probabilities).
        costs:         [K]    measured relative cost of each exit.
        lam:           the Lagrange multiplier; sweep it to trace the frontier.
    """
    lagrangian = mses_per_exit + lam * costs[None, :]        # [P, K]
    best, k_star = lagrangian.min(dim=1)                     # [P]

    if hard:
        probs = F.gumbel_softmax(logits, tau=tau, hard=True, dim=1)
    else:
        probs = F.softmax(logits / tau, dim=1)

    regret = (probs * (lagrangian - best[:, None])).sum(1)
    # Scale-free: divided by the oracle's own cost, so the number means "fraction
    # above the oracle" and is comparable across QPs, checkpoints and lam.
    loss = (regret / best.clamp_min(1e-12)).mean()

    share = probs.mean(0)
    out = {"loss": loss, "l_regret": loss, "exit_share": share.detach(),
           "oracle_choice": k_star.detach(), "probs": probs.detach()}
    if w_avg > 0:
        uniform = torch.full_like(share, 1.0 / share.numel())
        out["l_avg"] = ((share - uniform) ** 2).sum()
        out["loss"] = out["loss"] + w_avg * out["l_avg"]

    # Reported, never optimised: what fraction of tiles the router places exactly
    # where the oracle would. The loss can fall while this stalls -- a router that
    # hedges between two near-equal exits pays little regret and gets the choice
    # wrong -- so both are logged and neither is trusted alone.
    with torch.no_grad():
        out["oracle_agree"] = (probs.argmax(1) == k_star).float().mean()
    return out
