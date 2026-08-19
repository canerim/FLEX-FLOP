"""Measure the tile rule the deployed decode actually follows, at four resolutions.

The Method section says "256 px tiles" and quotes "2 tiles" for 416x240, and a
reader cannot rebuild the rule from that. This script does not restate the rule
from the source, it exercises it: for each CTC resolution it pads a frame exactly
as every evaluation script here does, encodes it with the real analysis transform,
and instruments `patchify` inside the real decode so the tile grid is read off the
tensor that the per-tile trunk is handed rather than off arithmetic done on paper.

Four questions, answered by observation:
  1. is the tile count a ceiling or a floor of the frame divided by the tile side
  2. is the edge tile cropped, or is the frame padded, and with which mode
  3. how the feature-space tile side follows from the RGB tile side
  4. where the padding is applied: before the decoder, inside patchify, or at the 3x3
"""
import json, sys, time
from pathlib import Path

import torch
import torch.nn.functional as F

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
sys.path.insert(0, str(Path.home() / "DCVC"))

from flexuf.config import FlexUFConfig, PIXELS_PER_LATENT, UPSAMPLE_FACTOR
from flexuf.model import FlexUFIntra, load_flexuf_state
from flexuf.backbone import decoder as D

CKPT = Path(__file__).resolve().parents[1] / "runs/RECIPE512/ckpt_PAPER.pth.tar"
RES = [(1920, 1080), (1280, 720), (832, 480), (416, 240)]
DEV = "cpu"  # every GPU on this box is carrying a training run; geometry needs none

ck = torch.load(CKPT, map_location="cpu")
cfg = FlexUFConfig(**ck["config"])
net = FlexUFIntra(cfg).to(DEV).eval()
load_flexuf_state(net, ck, where=f"{CKPT.name}: ")

P = cfg.rgb_patch
Fp = cfg.feature_patch

# Instrument the real patchify rather than reimplementing it, so what is reported
# is what the per-tile trunk was actually handed on this forward pass.
seen = []
_orig = D.patchify
def spy(x, patch):
    out, nh, nw = _orig(x, patch)
    seen.append(dict(feat_in=list(x.shape), patch=patch, nh=nh, nw=nw,
                     tiles=list(out.shape)))
    return out, nh, nw
D.patchify = spy

# Does patchify itself pad? Ask it for a grid it cannot form and record the answer.
try:
    _orig(torch.zeros(1, 384, 40, 40), Fp)
    patchify_pads = "yes: it accepted a non-divisible map"
except AssertionError as e:
    patchify_pads = f"no: it asserts, {e.args[0].strip()}"

rows = []
with torch.no_grad():
    for W, H in RES:
        t0 = time.time()
        x = torch.rand(1, 3, H, W, device=DEV)
        # The rule, verbatim from scripts/signalled_curve.py: pad the RGB frame up
        # to whole tiles on the bottom and right only, replicate mode, BEFORE the
        # encoder is ever called. Nothing downstream ever crops a partial tile.
        ph, pw = (-H) % P, (-W) % P
        xp = F.pad(x, (0, pw, 0, ph), mode="replicate") if (ph or pw) else x
        qp = torch.full((1,), cfg.qp, dtype=torch.int32, device=DEV)
        y, q, _ = net._encode_to_latent(xp, qp)
        seen.clear()
        rec = net.dec(y, q)
        s = seen[-1]
        rows.append(dict(
            name=f"{W}x{H}", W=W, H=H, rgb_patch=P, feature_patch=Fp,
            pad_right=pw, pad_bottom=ph,
            padded=[xp.shape[-1], xp.shape[-2]],
            latent=[y.shape[-1], y.shape[-2]],
            feature_in_patchify=[s["feat_in"][3], s["feat_in"][2]],
            patch_arg=s["patch"], nh=s["nh"], nw=s["nw"],
            tiles_measured=s["nh"] * s["nw"], tiles_tensor=s["tiles"][0],
            ceil_formula=(-(-H // P)) * (-(-W // P)),
            floor_formula=(H // P) * (W // P),
            recon=[rec.shape[-1], rec.shape[-2]],
            seconds=round(time.time() - t0, 1)))
        print(rows[-1], flush=True)

out = dict(
    checkpoint=str(CKPT), epoch=ck["epoch"], config=ck["config"],
    device=DEV,
    rgb_patch=P, feature_patch=Fp,
    latent_patch=cfg.latent_patch,
    pixels_per_latent=PIXELS_PER_LATENT, upsample_factor=UPSAMPLE_FACTOR,
    pad_site="RGB frame, before the encoder (scripts/signalled_curve.py:169-170)",
    pad_mode="replicate, bottom and right only: F.pad(x, (0, pw, 0, ph))",
    patchify_pads=patchify_pads,
    rows=rows)
res = Path(__file__).resolve().parents[1] / "results/tile_definition.json"
res.write_text(json.dumps(out, indent=2))
print("wrote", res)
