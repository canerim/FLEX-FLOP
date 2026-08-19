"""The exit mask has to survive the head's own scale.

StemRouterHeadV2 suppresses exits below the split depth. It used to do that by
assigning -1e4, which is a mask only while the head's own logits stay well above
it. Training let a common offset drift in -- nothing in a cross-entropy or a
regret objective penalises one -- and the raw outputs settled near -10000, at
which point the "mask" became the largest entry in every row (DECISIONS 89).

These check the property rather than the constant: whatever the head's scale,
a masked exit must never be selected and must carry no probability.
"""
import sys
from pathlib import Path

import pytest
import torch
import torch.nn.functional as F

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from flexuf.router.head2 import StemRouterHeadV2, oracle_ce_loss  # noqa: E402

K, J = 6, 2
COST = torch.tensor([0.5809, 0.5809, 0.5809, 0.7300, 0.8697, 1.0095])


def _head(offset):
    """A head whose final layer emits an arbitrary common offset."""
    h = StemRouterHeadV2(8, 4, K, min_exit=J, r_stem=4, r_lat=4, hidden=8).eval()
    with torch.no_grad():
        h.mlp[-1].bias.add_(offset)
    return h


@pytest.mark.parametrize("offset", [0.0, -10.0, -1e3, -1e4, -1e5, 1e4])
def test_masked_exits_are_never_selected(offset):
    torch.manual_seed(0)
    h = _head(offset)
    with torch.no_grad():
        lg = h(torch.randn(1, 8, 8, 8), torch.randn(1, 4, 4, 4),
               torch.randn(1, 4, 4, 4), torch.zeros(1, dtype=torch.int32), 4, 2)
    assert torch.isneginf(lg[:, :J]).all(), (
        f"at offset {offset} the mask is not below every real logit: "
        f"{lg[0].tolist()}")
    assert torch.isfinite(lg[:, J:]).all()
    lp = F.log_softmax(lg, 1)
    assert (lp[:, :J] == float("-inf")).all()
    assert int(lg.argmax(1).min()) >= J
    # and the tilt cannot rescue them either, for any multiplier
    for beta in (-1e4, -1.0, 0.0, 1.0, 1e4):
        assert int((lp - beta * COST[None, :]).argmax(1).min()) >= J


def test_cross_entropy_stays_finite_when_the_oracle_prefers_a_dead_column():
    """tiled_exit_mses fills columns below the split depth with exit-j's decode.

    Their costs are equal too, so argmin can land on one of them arbitrarily.
    With an -inf mask that would make the loss infinite; the label is clamped
    instead, which is exact rather than a repair because they are duplicates.
    """
    torch.manual_seed(1)
    mses = torch.rand(8, K) * 1e-4
    mses[:, :J + 1] = mses[:, J:J + 1]
    lg = torch.randn(8, K)
    lg[:, :J] = float("-inf")
    loss, k, _ = oracle_ce_loss(lg, mses, COST, 1e-5, min_exit=J)
    assert torch.isfinite(loss)
    assert int(k.min()) >= J
