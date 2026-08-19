"""Configuration C's selection rule, on synthetic tables.

The endpoints are the whole argument: rho=0 must be exactly configuration B and
rho=1 exactly configuration A, so that the columns between them measure an
interpolation rather than three unrelated systems. The full-scale run checks
this against independently measured files; this checks the code path itself, in
a second, on tables where the answer is known by construction.
"""
import sys
from pathlib import Path

import pytest
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))

from hybrid_curve import h2, hybrid_map  # noqa: E402

K, T = 6, 40
COST = torch.tensor([0.5809, 0.5809, 0.5809, 0.7300, 0.8697, 1.0095])


def tables(seed=0):
    g = torch.Generator().manual_seed(seed)
    # distortion falls with depth, by a per-tile amount, so every tile has a
    # genuine trade-off rather than a constant preference
    base = torch.rand(T, 1, generator=g) * 1e-4 + 1e-5
    drop = torch.rand(T, K, generator=g).sort(dim=1, descending=True).values
    M = base * drop
    lp = torch.log_softmax(torch.randn(T, K, generator=g) * 3, dim=1)
    return M, lp


@pytest.mark.parametrize("seed", range(5))
@pytest.mark.parametrize("lam", [0.0, 1e-6, 1e-5, 1e-3])
def test_endpoints_are_exact(seed, lam):
    M, lp = tables(seed)
    beta = 12.5
    k0, s0 = hybrid_map(M, lp, COST, beta, lam, 0.0)
    k1, s1 = hybrid_map(M, lp, COST, beta, lam, 1.0)
    assert s0 == 0 and s1 == T
    assert torch.equal(k0, (lp - beta * COST[None, :]).argmax(1))
    assert torch.equal(k1, (M + lam * COST[None, :]).argmin(1))


@pytest.mark.parametrize("rho", [0.05, 0.1, 0.25, 0.5, 0.9])
def test_overridden_set_is_the_largest_regrets(rho):
    M, lp = tables(3)
    lam, beta = 1e-5, 12.5
    k, s = hybrid_map(M, lp, COST, beta, lam, rho)
    assert s == round(rho * T)
    kr = (lp - beta * COST[None, :]).argmax(1)
    ko = (M + lam * COST[None, :]).argmin(1)
    L = M + lam * COST[None, :]
    regret = (L.gather(1, kr[:, None]) - L.gather(1, ko[:, None])).squeeze(1)
    changed = (k != kr) | (kr == ko)     # a tile whose two choices agree is
    over = torch.zeros(T, dtype=torch.bool)             # indistinguishable
    over[regret.topk(s).indices] = True
    # every overridden tile takes the oracle's exit
    assert torch.equal(k[over], ko[over])
    # and no tile outside the set was moved
    assert torch.equal(k[~over], kr[~over])
    # the set is a suffix of the regret ordering: nothing outside beats
    # anything inside
    assert regret[over].min() >= regret[~over].max() - 1e-12
    assert changed.any() or s == 0


def test_recovery_is_monotone_in_rho():
    """More overrides can only lower the objective, at a fixed multiplier."""
    M, lp = tables(1)
    lam, beta = 1e-5, 12.5
    L = M + lam * COST[None, :]
    prev = None
    for rho in (0.0, 0.05, 0.1, 0.25, 0.5, 0.75, 1.0):
        k, _ = hybrid_map(M, lp, COST, beta, lam, rho)
        obj = L.gather(1, k[:, None]).mean().item()
        if prev is not None:
            assert obj <= prev + 1e-12
        prev = obj


def test_mask_cost_is_the_binary_entropy():
    assert h2(0.0) == 0.0 and h2(1.0) == 0.0
    assert h2(0.5) == pytest.approx(1.0)
    assert h2(0.1) == pytest.approx(0.4689955935892812, rel=1e-9)
    # and it is symmetric, which is what makes "signal 90%" as cheap to flag
    # as "signal 10%"
    for p in (0.05, 0.2, 0.35):
        assert h2(p) == pytest.approx(h2(1 - p))
