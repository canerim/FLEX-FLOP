"""Measure — not estimate — where the DCVC-UF intra decoder spends its compute.

Why this exists
---------------
The whole FLEX-UF premise is "skip trailing decoder blocks per patch and save a
lot of FLOPs for almost no PSNR". Whether that premise holds depends entirely on
*how the decoder's MACs are distributed*:

  - if the 12-block trunk dominates, an early-exit ladder over the trunk is worth
    building, because dropping blocks drops real compute;
  - if the head or the opening upsample dominates, early exit buys nothing,
    because those always run.

FLEX measured 89.4% trunk / 2.5% head on DCVC-RT. UF is the same block family at
a different width (C=384 vs 368), so the shares should be close — but "should be"
is not a measurement. This script attaches forward hooks to every leaf conv and
counts MACs from the real tensor shapes seen during a real forward pass.

Run:
    python scripts/mac_audit.py --device 4 --height 1080 --width 1920
"""

import argparse
import sys
from collections import OrderedDict
from pathlib import Path

import torch
import torch.nn as nn

DCVC_ROOT = Path.home() / "DCVC"
sys.path.insert(0, str(DCVC_ROOT))

from src.models.image_model import (  # noqa: E402
    IntraDecoder,
    g_ch_enc_dec,
    g_ch_src,
    g_ch_y,
)


def conv_macs(module: nn.Conv2d, out: torch.Tensor) -> int:
    """MACs for one conv = output positions x kernel taps x in-channels-per-group.

    Multiply-accumulate, not FLOPs: one MAC = one multiply + one add. This is the
    convention FLEX used, so the two projects' numbers stay comparable.
    """
    out_positions = out.shape[0] * out.shape[2] * out.shape[3]
    taps = module.kernel_size[0] * module.kernel_size[1]
    in_per_group = module.in_channels // module.groups
    return out_positions * out.shape[1] * taps * in_per_group


def audit(height: int, width: int, device: str):
    dec = IntraDecoder().to(device).eval()

    # The decoder consumes the latent, which is the frame downsampled by 16
    # (pixel_unshuffle 8 in the encoder, then one stride-2 conv). One latent
    # pixel therefore expands to 16x16 RGB pixels.
    lat_h, lat_w = height // 16, width // 16

    records: "OrderedDict[str, int]" = OrderedDict()
    handles = []

    def make_hook(name):
        def hook(mod, _inp, out):
            if isinstance(out, torch.Tensor):
                records[name] = records.get(name, 0) + conv_macs(mod, out)

        return hook

    for name, mod in dec.named_modules():
        if isinstance(mod, nn.Conv2d):
            handles.append(mod.register_forward_hook(make_hook(name)))

    x = torch.randn(1, g_ch_y, lat_h, lat_w, device=device)
    q = torch.ones(1, g_ch_enc_dec, 1, 1, device=device)
    with torch.no_grad():
        recon = dec(x, q)

    for h in handles:
        h.remove()

    total = sum(records.values())

    # Group the leaf convs into the three structural parts that matter for early
    # exit: the opening upsample (always runs), the 12-block trunk (the part a
    # ladder can skip), and the head (always runs, and in FLEX carried 75% of the
    # patch-boundary penalty).
    groups = {"upsample dec_1[0]": 0, "trunk dec_1[1..12]": 0, "head dec_2": 0}
    per_block = OrderedDict()
    for name, macs in records.items():
        if name.startswith("dec_2"):
            groups["head dec_2"] += macs
        elif name.startswith("dec_1.0"):
            groups["upsample dec_1[0]"] += macs
        else:
            groups["trunk dec_1[1..12]"] += macs
            blk = ".".join(name.split(".")[:2])
            per_block[blk] = per_block.get(blk, 0) + macs

    # Pointwise vs spatial: FLEX's width axis rests on the claim that ~99.7% of
    # decoder MACs are 1x1 convs, which carry no receptive field and so can be
    # narrowed without any patch-boundary cost. Re-check it here for UF.
    pw = sum(m for n, m in records.items()
             if dict(dec.named_modules())[n].kernel_size == (1, 1))
    spatial = total - pw

    px = height * width
    print(f"\n{'=' * 74}")
    print(f"DCVC-UF IntraDecoder MAC audit  —  {width}x{height} frame")
    print(f"{'=' * 74}")
    print(f"latent            {tuple(x.shape)}")
    print(f"reconstruction    {tuple(recon.shape)}")
    print(f"channel widths    latent={g_ch_y}  trunk C={g_ch_enc_dec}  pre-shuffle={g_ch_src}")
    print(f"decoder params    {sum(p.numel() for p in dec.parameters()):,}")
    print()
    print(f"TOTAL             {total / 1e9:10.2f} GMAC   ({total / px / 1e3:.1f} kMAC/px)")
    print()
    print(f"{'structural part':<26} {'GMAC':>10} {'share':>9}")
    print(f"{'-' * 48}")
    for k, v in groups.items():
        print(f"{k:<26} {v / 1e9:10.2f} {100 * v / total:8.2f}%")
    print()
    print(f"{'kernel type':<26} {'GMAC':>10} {'share':>9}")
    print(f"{'-' * 48}")
    print(f"{'pointwise 1x1':<26} {pw / 1e9:10.2f} {100 * pw / total:8.2f}%")
    print(f"{'spatial (3x3 dw etc)':<26} {spatial / 1e9:10.2f} {100 * spatial / total:8.2f}%")

    print()
    print("per trunk block (the units an exit ladder groups over):")
    for blk, macs in per_block.items():
        print(f"  {blk:<16} {macs / 1e9:8.2f} GMAC  {100 * macs / total:6.2f}%")

    # What an exit ladder actually buys. Exiting after g of the 12 trunk blocks
    # skips the remaining (12-g), but upsample and head still run.
    trunk = groups["trunk dec_1[1..12]"]
    fixed = total - trunk
    n_blocks = len(per_block)
    per_blk = trunk / n_blocks
    print()
    print(f"early-exit ceiling  (upsample+head always run = {100 * fixed / total:.2f}% of MACs)")
    print(f"{'exit after g blocks':<22} {'rel. cost':>10} {'saved':>9}")
    print(f"{'-' * 44}")
    for g in range(2, n_blocks + 1, 2):
        cost = fixed + g * per_blk
        print(f"  g={g:<19} {cost / total:10.3f} {100 * (1 - cost / total):8.1f}%")
    print(f"{'=' * 74}\n")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--height", type=int, default=1080)
    ap.add_argument("--width", type=int, default=1920)
    ap.add_argument("--device", default="4", help="CUDA index, or 'cpu'")
    a = ap.parse_args()
    dev = a.device if a.device == "cpu" else f"cuda:{a.device}"
    audit(a.height, a.width, dev)
