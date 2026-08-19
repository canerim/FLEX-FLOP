"""A zeroed input group must be unable to reach the output at all.

The router input ablation only means something if the five variants differ in
information and in NOTHING else. Two things have to hold for that, and both are
properties rather than opinions, so they are measured here:

  1. a group that is not live cannot change a single logit, however violently
     its tensor is changed;
  2. every variant has the same architecture and the same parameter count, so a
     difference in agreement cannot be a difference in capacity.

The third property is compatibility: with no `inputs` argument the head is the
one the paper trained, down to the keys of its state dict, because
scripts/router_curve.py and friends load those checkpoints into a default head.
"""
import sys
from pathlib import Path

import pytest
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from flexuf.router.head2 import (INPUT_GROUPS, StemRouterHeadV2,  # noqa: E402
                                 parse_inputs)

K, J, SC, LC, FP, LP = 6, 2, 8, 4, 4, 2
VARIANTS = ["stem", "latent", "scales", "bits", "qp", "all"]


def _inputs(seed):
    """One batch of everything the head reads, at toy width."""
    g = torch.Generator().manual_seed(seed)
    return dict(
        stem=torch.randn(2, SC, 2 * FP, 2 * FP, generator=g),
        y_hat=torch.randn(2, LC, 2 * LP, 2 * LP, generator=g),
        scales=torch.rand(2, LC, 2 * LP, 2 * LP, generator=g) + 0.1,
        qp=torch.randint(0, 64, (2,), generator=g, dtype=torch.int32),
        bits=torch.rand(2, LC, 2 * LP, 2 * LP, generator=g) * 3 + 0.01,
    )


def _head(spec):
    torch.manual_seed(0)
    return StemRouterHeadV2(SC, LC, K, min_exit=J, r_stem=4, r_lat=4, hidden=8,
                            inputs=spec).eval()


def _logits(h, d):
    return h(d["stem"], d["y_hat"], d["scales"], d["qp"], FP, LP, bits=d["bits"])


@pytest.mark.parametrize("spec", VARIANTS)
@pytest.mark.parametrize("group", INPUT_GROUPS)
def test_a_dead_group_cannot_reach_the_output(spec, group):
    h, a, b = _head(spec), _inputs(1), _inputs(2)
    # Replace exactly one group with a completely different tensor.
    key = {"latent": "y_hat", "scales": "scales"}.get(group, group)
    c = dict(a); c[key] = b[key] if group != "qp" else (63 - a["qp"]).to(torch.int32)
    with torch.no_grad():
        la, lc = _logits(h, a), _logits(h, c)
    fin = torch.isfinite(la)
    moved = float((lc[fin] - la[fin]).abs().max())
    if group in parse_inputs(spec):
        # The live group has to matter, or the variant is not measuring it.
        assert moved > 0, f"{spec}: live group {group} changed nothing"
    else:
        assert moved == 0.0, f"{spec}: dead group {group} moved a logit by {moved}"


def test_every_variant_has_the_same_parameters():
    n = {v: sum(p.numel() for p in _head(v).parameters()) for v in VARIANTS}
    assert len(set(n.values())) == 1, n
    keys = {v: tuple(_head(v).state_dict()) for v in VARIANTS}
    assert len(set(keys.values())) == 1, "the variants are not one architecture"


def test_the_default_head_is_still_the_paper_head():
    """No `inputs` means no bits pathway, so old checkpoints still load."""
    h = StemRouterHeadV2(384, 256, K, min_exit=J)
    assert h.inputs == "stem,latent,scales,qp"
    assert not h.with_bits
    assert not hasattr(h, "proj_bits")
    assert sum(p.numel() for p in h.parameters()) == 144_024
    assert tuple(h.state_dict()) == (
        "bias", "proj_stem.weight", "proj_stem.bias", "proj_lat.weight",
        "proj_lat.bias", "mlp.0.weight", "mlp.0.bias", "mlp.1.weight",
        "mlp.1.bias", "mlp.3.weight", "mlp.3.bias", "mlp.5.weight",
        "mlp.5.bias")
    # And it still runs without being handed bits it has no pathway for.
    d = _inputs(3)
    with torch.no_grad():
        lg = StemRouterHeadV2(SC, LC, K, min_exit=J, r_stem=4, r_lat=4,
                              hidden=8).eval()(
            d["stem"], d["y_hat"], d["scales"], d["qp"], FP, LP)
    assert lg.shape == (2 * 4, K)


def test_asking_for_bits_without_the_pathway_is_an_error():
    with pytest.raises(ValueError):
        StemRouterHeadV2(SC, LC, K, inputs="bits", with_bits=False)
    with pytest.raises(ValueError):
        parse_inputs("stem,entropy")
    # And a head that IS reading bits must not be handed None instead.
    h = _head("bits")
    d = _inputs(4)
    with pytest.raises(ValueError):
        h(d["stem"], d["y_hat"], d["scales"], d["qp"], FP, LP)
