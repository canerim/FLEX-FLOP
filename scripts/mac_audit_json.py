"""Structured re-run of scripts/mac_audit.py.

Same hooks, same MAC convention, same IntraDecoder; writes JSON instead of
printing, so the supplement can read per-module shapes and per-leaf-conv counts
rather than quoting a terminal. CPU only: the audit needs shapes, not weights.
"""
import json
import sys
from collections import OrderedDict
from pathlib import Path

import torch
import torch.nn as nn

R = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(R / "scripts"))
DCVC_ROOT = Path.home() / "DCVC"
sys.path.insert(0, str(DCVC_ROOT))

from src.models.image_model import (IntraDecoder, g_ch_enc_dec, g_ch_src,
                                    g_ch_y)


def conv_macs(m, out):
    pos = out.shape[0] * out.shape[2] * out.shape[3]
    taps = m.kernel_size[0] * m.kernel_size[1]
    return pos * out.shape[1] * taps * (m.in_channels // m.groups)


def audit(h, w):
    dec = IntraDecoder().eval()
    lat_h, lat_w = h // 16, w // 16
    rec = OrderedDict()
    hs = []

    def mk(name):
        def hook(mod, inp, out):
            if isinstance(out, torch.Tensor):
                e = rec.setdefault(name, dict(
                    mac=0, kernel=mod.kernel_size[0], stride=mod.stride[0],
                    in_ch=mod.in_channels, out_ch=mod.out_channels,
                    groups=mod.groups, calls=0, out_hw=None, in_hw=None))
                e["mac"] += conv_macs(mod, out)
                e["calls"] += 1
                e["out_hw"] = [int(out.shape[2]), int(out.shape[3])]
                e["in_hw"] = [int(inp[0].shape[2]), int(inp[0].shape[3])]
        return hook

    for n, m in dec.named_modules():
        if isinstance(m, nn.Conv2d):
            hs.append(m.register_forward_hook(mk(n)))
    x = torch.randn(1, g_ch_y, lat_h, lat_w)
    q = torch.ones(1, g_ch_enc_dec, 1, 1)
    with torch.no_grad():
        recon = dec(x, q)
    for handle in hs:
        handle.remove()

    total = sum(v["mac"] for v in rec.values())
    parts = {"upsample": 0, "trunk": 0, "head": 0}
    per_block = OrderedDict()
    for n, v in rec.items():
        if n.startswith("dec_2"):
            parts["head"] += v["mac"]
            v["part"] = "head"
        elif n.startswith("dec_1.0"):
            parts["upsample"] += v["mac"]
            v["part"] = "upsample"
        else:
            parts["trunk"] += v["mac"]
            v["part"] = "trunk"
            blk = ".".join(n.split(".")[:2])
            per_block[blk] = per_block.get(blk, 0) + v["mac"]
    pw = sum(v["mac"] for v in rec.values() if v["kernel"] == 1)
    px = h * w
    return dict(
        what="per-leaf-convolution MAC audit of the released DCVC-UF intra "
             "decoder, by forward hook on the real tensor shapes",
        source="scripts/mac_audit.py, re-run to write JSON rather than stdout",
        weights="none loaded: torch-initialised IntraDecoder(); a MAC count "
                "depends on shapes, not on values",
        device="cpu",
        mac_convention="one MAC = one multiply and one add; "
                       "out_positions x out_channels x taps x in_ch/groups",
        frame=[w, h], latent=[int(g_ch_y), lat_h, lat_w],
        recon=[int(recon.shape[1]), int(recon.shape[2]), int(recon.shape[3])],
        ch_latent=int(g_ch_y), ch_trunk=int(g_ch_enc_dec),
        ch_preshuffle=int(g_ch_src),
        decoder_params=int(sum(p.numel() for p in dec.parameters())),
        total_mac=int(total), total_gmac=total / 1e9, kmac_per_px=total / px / 1e3,
        parts={a: dict(gmac=b / 1e9, share=b / total) for a, b in parts.items()},
        pointwise_share=pw / total, spatial_share=1 - pw / total,
        n_trunk_blocks=len(per_block),
        per_block_gmac={a: b / 1e9 for a, b in per_block.items()},
        layers=[dict(name=n, **v) for n, v in rec.items()],
    )


out = {"1920x1088": audit(1088, 1920), "1280x768": audit(768, 1280),
       "832x512": audit(512, 832)}
p = R / "results" / "mac_audit.json"
p.write_text(json.dumps(out, indent=1))
print("wrote", p)
