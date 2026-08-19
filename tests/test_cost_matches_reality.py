"""The control that was missing: does the cost model bill what the code runs?

Why this exists
---------------
Every saving figure the project reports comes from `flexuf.cost`, which is an
arithmetic model of the decoder written by hand. The decoder itself is separate
code. Nothing tied the two together, and they drifted:

`MultiExitIntraDecoder.forward` does `exit_map.clamp(min=j)`, so a tile assigned
an exit shallower than the split still runs group j before leaving. The cost
model used `k_eff = max(k, j-1)` and billed such a tile as if it had left AT the
split, omitting group j entirely. With a router that assigned exit 0 to every
tile, that reported 58.70% of compute saved where the truth was 43.79%.

The existing controls could not have caught it. They assert that the ladder
*computes* the same thing as stock UF (max|diff| = 0), which stayed true the
whole time — the reconstruction was right, only the price tag was wrong.

The general lesson, which is why this file is separate and its docstring is
long: when a cost model and the code it models live apart, they drift, and the
drift is always OPTIMISTIC — an over-estimate gets investigated, an
under-estimate gets published.

So: run the real decode, count real MACs with hooks, and require the model to
agree.

Run:  python tests/test_cost_matches_reality.py
"""

from __future__ import annotations

import os
import sys
from pathlib import Path

import torch
import torch.nn as nn

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
sys.path.insert(0, str(Path.home() / "DCVC"))

from flexuf.backbone.decoder import MultiExitIntraDecoder  # noqa: E402
from flexuf.config import LATENT_CH, TRUNK_CH, FlexUFConfig  # noqa: E402
from flexuf.cost import frame_relative_cost  # noqa: E402

DEVICE = os.environ.get("FLEXUF_TEST_DEVICE",
                        "cuda:0" if torch.cuda.is_available() else "cpu")
# Big enough that the amortised stem is not a rounding error, small enough to be
# quick: 64x64 latent -> 128x128 feature -> 1024x1024 RGB.
LAT = 64

# The model is an idealisation in two places we accept:
#   - PixelShuffle and reshapes are free in both (they really are, 0 MAC)
#   - the measured share constants come from a 1920x1088 audit, and the
#     upsample/head/trunk ratio moves very slightly with resolution
# 3% absolute on a fraction-of-full-decode figure is tight enough to have caught
# the 15-point error above by a wide margin.
TOL = 0.03


class MacCounter:
    """Count MACs of every conv actually executed during a forward pass."""

    def __init__(self, model: nn.Module):
        self.total = 0
        self.handles = [
            m.register_forward_hook(self._hook)
            for m in model.modules()
            if isinstance(m, nn.Conv2d)
        ]

    def _hook(self, mod: nn.Conv2d, _inp, out):
        if not isinstance(out, torch.Tensor):
            return
        positions = out.shape[0] * out.shape[2] * out.shape[3]
        taps = mod.kernel_size[0] * mod.kernel_size[1]
        self.total += positions * out.shape[1] * taps * (mod.in_channels // mod.groups)

    def close(self):
        for h in self.handles:
            h.remove()


@torch.no_grad()
def measure(dec, y, q, exit_map) -> int:
    c = MacCounter(dec)
    dec(y, q, exit_map=exit_map)
    c.close()
    return c.total


@torch.no_grad()
def measure_full(dec, y, q) -> int:
    c = MacCounter(dec)
    dec.forward_full(y, q)
    c.close()
    return c.total


def check(cfg: FlexUFConfig, describe: str, make_map) -> tuple[float, float]:
    torch.manual_seed(0)
    dec = MultiExitIntraDecoder(cfg).to(DEVICE).eval()
    y = torch.randn(1, LATENT_CH, LAT, LAT, device=DEVICE)
    q = torch.rand(1, TRUNK_CH, 1, 1, device=DEVICE) + 0.5

    n_tiles = (LAT * 2 // cfg.feature_patch) ** 2
    em = make_map(n_tiles, cfg).to(DEVICE)

    measured = measure(dec, y, q, em) / measure_full(dec, y, q)
    predicted = frame_relative_cost(em, cfg, "head")
    err = abs(measured - predicted)
    status = "ok " if err <= TOL else "OFF"
    print(f"  [{status}] {describe:<34} measured {measured:6.3f}  "
          f"model {predicted:6.3f}  diff {err:+.4f}")
    return measured, predicted


def main() -> int:
    print(f"\ndevice: {DEVICE}   tolerance: {TOL}\n")
    print("does the cost model bill what the decoder runs?")
    bad = []

    # Every adapter kind, not just the default. The shipped runs (RECIPE512,
    # BEST) use "scaled", and this file passed for a year without ever
    # constructing one: the FFN adapter was billed at 2C^2 where it costs 5C^2,
    # so the cheapest exit was priced 0.581 against a measured 0.617. A control
    # that does not cover the shipped configuration is not a control.
    for j, kind in [(2, "conv1x1"), (2, "ffn"), (2, "scaled"),
                    (4, "conv1x1"), (4, "scaled")]:
        cfg = FlexUFConfig(split_depth=j, latent_patch=8, latent_halo=2,
                           adapter_kind=kind)
        K = cfg.num_exits

        cases = [
            (f"j={j} {kind} all deepest",
             lambda n, c: torch.full((n,), c.num_exits - 1, dtype=torch.long)),
            # The case that exposed the bug: exits BELOW the split, which the
            # decoder clamps up to j but the model used to bill as if they had
            # left at the split.
            (f"j={j} {kind} all exit 0 (clamped to j)",
             lambda n, c: torch.zeros(n, dtype=torch.long)),
            (f"j={j} {kind} all at the split",
             lambda n, c: torch.full((n,), c.split_depth, dtype=torch.long)),
            (f"j={j} {kind} mixed uniform over exits",
             lambda n, c: torch.arange(n, dtype=torch.long) % c.num_exits),
            (f"j={j} {kind} half shallow half deep",
             lambda n, c: torch.where(torch.arange(n) < n // 2,
                                      torch.tensor(c.split_depth),
                                      torch.tensor(c.num_exits - 1)).long()),
        ]
        for desc, mk in cases:
            m, p = check(cfg, desc, mk)
            if abs(m - p) > TOL:
                bad.append((desc, m, p))
        print()
        del K

    if bad:
        print("COST MODEL DISAGREES WITH REALITY:")
        for d, m, p in bad:
            print(f"  {d}: measured {m:.3f} vs model {p:.3f}")
        return 1
    print("cost model matches the executed decode in every case\n")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
