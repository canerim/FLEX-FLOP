"""What an exit adapter actually costs, and how far it actually reaches.

Figure 2 used to carry hand-typed constants, and one of them was the superseded
2C^2 for the FFN adapter. flexuf/cost.py records why that number is wrong: the
module is PW C->4C then PW C->C, which is 5C^2, and the shipped cost table was
corrected once it was counted with hooks. A drawing is exactly the place where a
stale constant survives longest, so every number the figure prints is counted
here instead, off the real modules, and written out with its provenance.

Two things are measured:

  cost      MAC per feature pixel, from flexuf.measure.MacMeter, which counts
            off the executed output shapes rather than off an arithmetic model.
  footprint how many output pixels one input pixel can reach. This is the claim
            that makes the adapters safe inside a tiled decode: a pointwise
            operator adds no receptive field, so it adds no seam. Measured by
            perturbing a single input pixel and counting where the output moves.

The per-convolution profile is emitted as well, because the figure draws each
module to scale and a drawing that recomputes its own geometry is a second
implementation of the thing it is meant to report.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import torch

R = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(R))
sys.path.insert(0, str(Path.home() / "DCVC"))

from flexuf.config import TRUNK_CH, FlexUFConfig  # noqa: E402
from flexuf.backbone.decoder import Conv1x1Adapter, FFNAdapter  # noqa: E402
from flexuf.measure import MacMeter  # noqa: E402
from src.layers.layers import DepthConvBlock  # noqa: E402

# Pinned at epoch 0 on purpose. ckpt_eval.pth.tar is overwritten by a watcher
# every epoch, so a figure built from it cannot be reproduced tomorrow.
CKPT = R / "runs/RECIPE512/ckpt_PAPER.pth.tar"
OUT = R / "results/adapter_cost.json"

# 32x32 is the feature tile the per-tile trunk actually runs on at the shipped
# 256 px tile, so the count is taken at the shape the decoder really sees.
H = W = 32


def macs_per_pixel(module: torch.nn.Module, c: int = TRUNK_CH) -> float:
    """MAC per output feature pixel, counted with hooks on one forward pass."""
    x = torch.randn(1, c, H, W)
    with MacMeter(module) as m:
        module(x)
    return m.total / (H * W)


def profile(module: torch.nn.Module) -> list:
    """Every convolution in the module, in execution order, with its cost.

    Reported so the figure can lay the block and the two adapters on one axis at
    true relative size, and with the channel width each convolution writes out,
    so the expand-then-contract shape of the FFN is visible rather than asserted.
    """
    rows = []
    for name, m in module.named_modules():
        if not isinstance(m, torch.nn.Conv2d):
            continue
        rows.append({
            "name": name,
            "mac_per_px": (m.in_channels * m.out_channels * m.kernel_size[0]
                           * m.kernel_size[1] / m.groups),
            "out_channels": m.out_channels,
            "kernel": m.kernel_size[0],
            "depthwise": m.groups == m.in_channels and m.groups > 1,
        })
    return rows


def footprint(module: torch.nn.Module, c: int = TRUNK_CH) -> int:
    """Output pixels one input pixel can move, counted rather than asserted.

    The adapters are zero-initialised so that they are the identity at step 0,
    which would make every difference below exactly zero and the measurement
    vacuous. The footprint is a property of the operator graph and not of the
    weights, so the weights are randomised first and the graph is what is being
    probed.
    """
    torch.manual_seed(0)
    for p in module.parameters():
        torch.nn.init.normal_(p, std=0.05)
    module.eval()
    with torch.no_grad():
        x = torch.zeros(1, c, H, W)
        a = module(x)
        x[0, :, H // 2, W // 2] = 1.0
        b = module(x)
    moved = (b - a).abs().amax(dim=1)[0] > 1e-8
    return int(moved.sum().item())


def main() -> None:
    ck = torch.load(CKPT, map_location="cpu", weights_only=False)
    cfg = FlexUFConfig(**ck["config"])
    K, j, b = cfg.num_exits, cfg.split_depth, cfg.blocks_per_exit
    C = TRUNK_CH

    block = DepthConvBlock(C, C)
    conv1x1 = Conv1x1Adapter(C)
    ffn = FFNAdapter(C, cfg.adapter_expand)

    block_mac = macs_per_pixel(block)

    def with_share(rows):
        for r in rows:
            r["share_of_block"] = r["mac_per_px"] / block_mac
        return rows

    kinds = {
        "conv1x1": {"mac_per_px": macs_per_pixel(conv1x1),
                    "footprint_px": footprint(Conv1x1Adapter(C)),
                    "profile": with_share(profile(conv1x1))},
        "ffn": {"mac_per_px": macs_per_pixel(ffn),
                "footprint_px": footprint(FFNAdapter(C, cfg.adapter_expand)),
                "profile": with_share(profile(ffn))},
    }
    for v in kinds.values():
        v["blocks"] = v["mac_per_px"] / block_mac
        v["in_C_squared"] = v["mac_per_px"] / C**2

    # Which adapter each exit wears is not a choice made in the drawing: it is
    # read back off the shipped state dict, so the figure cannot disagree with
    # the weights that were trained.
    sd = ck.get("state_dict", ck)
    worn = {}
    for k in range(K):
        if any(key.startswith(f"dec.adapters.{k}.pw_in") for key in sd):
            worn[k] = "ffn"
        elif any(key.startswith(f"dec.adapters.{k}.conv") for key in sd):
            worn[k] = "conv1x1"
        else:
            worn[k] = None

    exits = []
    for k in range(K):
        skipped = (K - 1 - k) * b
        kind = worn[k]
        exits.append({
            "exit": k,
            "blocks_skipped": skipped,
            "adapter_kind": kind,
            "adapter_blocks": kinds[kind]["blocks"] if kind else 0.0,
            # forward() clamps the exit map at j, so an exit below the split
            # depth is never the one a tile actually leaves through.
            "reachable": k >= j,
        })

    out = {
        "ckpt": str(CKPT.relative_to(R)),
        "epoch": ck.get("epoch"),
        "config": ck["config"],
        "trunk_channels": C,
        "feature_tile": [H, W],
        "block": {"mac_per_px": block_mac,
                  "in_C_squared": block_mac / C**2,
                  "footprint_px": footprint(DepthConvBlock(C, C)),
                  "profile": with_share(profile(block))},
        "adapters": kinds,
        "exits": exits,
    }
    OUT.write_text(json.dumps(out, indent=2))
    print(f"  -> {OUT.relative_to(R)}")
    print(f"  block {block_mac:,.0f} MAC/px = {block_mac / C**2:.3f} C^2, "
          f"footprint {out['block']['footprint_px']} px")
    for name, v in kinds.items():
        print(f"  {name:<8} {v['mac_per_px']:,.0f} MAC/px = "
              f"{v['in_C_squared']:.3f} C^2 = {v['blocks']:.4f} block, "
              f"footprint {v['footprint_px']} px")
    for e in exits:
        print(f"  exit {e['exit']}  skips {e['blocks_skipped']:>2} blocks  "
              f"adapter {str(e['adapter_kind']):<8} "
              f"{e['adapter_blocks']:.4f} block  "
              f"reachable {e['reachable']}")


if __name__ == "__main__":
    main()
