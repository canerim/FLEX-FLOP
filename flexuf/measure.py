"""Count the multiply-accumulates a decode actually performed.

Why this file exists
--------------------
Every saving this project reported came from `flexuf.cost`, an arithmetic model
of the decoder written by hand. The decoder is separate code. A model and the
code it models drift, and this one did -- three times:

  * the exit clamp: `forward()` clamps below the split depth, the model did not
  * the FFN adapter: 5C^2 in the module, 2C^2 in the model
  * the adapter at a clamped exit: billed at the exit the map NAMES rather than
    the one that RUNS

Each was caught by a test that compares the model against a hook count. That is
backwards. The hook count is the ground truth, it costs one forward pass, and
the routed decode is already being run to measure distortion -- so the saving
can simply BE measured, and the model demoted to what it is genuinely needed
for: a per-tile price the Lagrangian can evaluate inside an argmin over exits,
for every tile, at every multiplier of a bisection, where running a decode per
candidate is not possible.

Model where a model is unavoidable; measure what is reported.
"""

from __future__ import annotations

import torch
import torch.nn as nn


class MacMeter:
    """MACs of every conv and linear executed inside the context.

    Counts from the OUTPUT shape, so a module run on 40 tiles of 32x32 costs
    what it actually cost, and a module skipped by an early exit costs nothing
    because its hook never fires. That is the whole point: there is no
    bookkeeping to get wrong.
    """

    def __init__(self, module: nn.Module):
        self.module, self.total, self._h = module, 0.0, []

    def __enter__(self):
        def hook(m, _i, o):
            if isinstance(o, (tuple, list)):
                o = o[0]
            if isinstance(m, nn.Conv2d):
                # positions = BATCH * H * W. The batch is not 1 here: the
                # per-tile trunk runs 40 tiles as 40 batch elements, so
                # counting H*W alone undercounts it by the tile count. The
                # first version of this file did exactly that and reported the
                # deepest exit at 0.43 of a released decode instead of 1.01.
                pos = o.numel() // o.shape[1]
                self.total += (m.in_channels * m.out_channels
                               * m.kernel_size[0] * m.kernel_size[1]
                               * pos / m.groups)
            elif isinstance(m, nn.Linear):
                n = 1
                for d in o.shape[:-1]:
                    n *= d
                self.total += m.in_features * m.out_features * n
        self._h = [m.register_forward_hook(hook) for m in self.module.modules()
                   if isinstance(m, (nn.Conv2d, nn.Linear))]
        return self

    def __exit__(self, *exc):
        for h in self._h:
            h.remove()
        self._h = []
        return False


@torch.no_grad()
def measured_cost(dec, ref_dec, y, quant_step, exit_map) -> float:
    """Relative cost of one routed decode: routed MACs / released MACs.

    Both on the same latent, both counted the same way, so the ratio needs no
    calibration constant and no share table. This is the number a saving should
    be quoted from.
    """
    with MacMeter(dec) as routed:
        dec(y, quant_step, exit_map=exit_map)
    with MacMeter(ref_dec) as released:
        ref_dec.forward_full(y, quant_step)
    return routed.total / released.total


@torch.no_grad()
def measured_saving_pct(dec, ref_dec, y, quant_step, exit_map) -> float:
    """100 * (1 - measured_cost), i.e. what the paper calls the saving."""
    return 100.0 * (1.0 - measured_cost(dec, ref_dec, y, quant_step, exit_map))
