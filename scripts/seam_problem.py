"""The seam artefact at frame scale -- the problem, before any fix.

`seam_patches.py` shows a 192px window and answers "what does the fix do".
This one answers the question that comes first: what is actually wrong.

A whole 1080p frame, decoded tile by tile with the stock padding, and its error
against the full-frame decode of the SAME latent. Same weights, same bitstream,
same arithmetic away from the borders -- so every bright pixel in the error map
is the tiling and nothing else.

The grid should be legible without being pointed at. If it is not, the figure
has failed and the problem is smaller than claimed.
"""
from __future__ import annotations
import argparse, sys
from pathlib import Path
import numpy as np, torch, torch.nn.functional as F

R = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(R)); sys.path.insert(0, str(R / "scripts"))
sys.path.insert(0, str(Path.home() / "DCVC"))
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
from matplotlib.patches import Rectangle  # noqa: E402
import naturestyle as ns  # noqa: E402
ns.apply()

ap = argparse.ArgumentParser()
ap.add_argument("--seq", default="Bosphorus")
ap.add_argument("--qp", type=int, default=63)
ap.add_argument("--amp", type=float, default=30.0)
ap.add_argument("--device", default="cuda:0")
ap.add_argument("--out", default="docs/figures/seam_problem.png")
a = ap.parse_args()

import ctc_intra as C
from src.utils.transforms import ycbcr2rgb
from flexuf.config import FlexUFConfig
from flexuf.model import FlexUFIntra, load_flexuf_state
from flexuf.reference import reference_for

dev = a.device
ck = torch.load(R / "runs/BEST/ckpt_eval.pth.tar", map_location="cpu", weights_only=False)
cfg = FlexUFConfig(**ck["config"])
net = FlexUFIntra(cfg).to(dev).eval(); load_flexuf_state(net, ck)
ref = FlexUFIntra(cfg).to(dev).eval()
load_flexuf_state(ref, torch.load(reference_for(cfg, None), map_location="cpu",
                                  weights_only=False))
K, j, P = cfg.num_exits, cfg.split_depth, cfg.rgb_patch

seqs, _ = C.discover([])
s = ([q for q in seqs if a.seq in q["name"]] or seqs)[0]
x, _ = C.read_frames(s["path"], s["w"], s["h"], 1, 1)
x = x[0:1].to(dev); _, _, H, W = x.shape
ph, pw = (-H) % P, (-W) % P
xp = F.pad(x, (0, pw, 0, ph), mode="replicate") if (ph or pw) else x
nt = ((H + ph) // P) * ((W + pw) // P)
qp = torch.full((1,), a.qp, dtype=torch.int32, device=dev)
deep = torch.full((nt,), K - 1, dtype=torch.long, device=dev)

keep_mode, keep_rep = cfg.tile_pad_mode, net.dec.seam_repair
with torch.no_grad():
    y, q, _ = net._encode_to_latent(xp, qp)
    full = ref.dec.forward_full(y, q)
    cfg.tile_pad_mode = "zeros"; net.dec.seam_repair = None
    bad = net.dec(y, q, exit_map=deep)
    cfg.tile_pad_mode = keep_mode; net.dec.seam_repair = keep_rep
    db = 10 * torch.log10(((bad - xp) ** 2).mean() / ((full - xp) ** 2).mean())

def rgb(t):
    v = ycbcr2rgb((t + 0.5).clamp(0, 1))[0].clamp(0, 1)
    return v.permute(1, 2, 0).cpu().numpy()[:H, :W]

ref_img = rgb(full); bad_img = rgb(bad)
err = np.abs(bad_img - ref_img).mean(-1)[:H, :W]

fig = plt.figure(figsize=(ns.W2, 3.4))
gs = fig.add_gridspec(2, 3, width_ratios=[1.35, 1.35, 1.0], hspace=0.12, wspace=0.08)
ZR, ZC, ZS = 384, 640, 320          # a window on a grid crossing

a0 = fig.add_subplot(gs[:, 0]); a0.imshow(ref_img)
a0.set_title("full frame", fontsize=6.5, color=ns.INK, loc="left")
a0.add_patch(Rectangle((ZC, ZR), ZS, ZS, ec="#00e5ff", fc="none", lw=0.9))

a1 = fig.add_subplot(gs[:, 1])
a1.imshow(np.clip(err * a.amp, 0, 1), cmap="inferno", vmin=0, vmax=1)
a1.set_title(f"|error| ×{a.amp:.0f}   {db.item():+.3f} dB", fontsize=6.5,
             color=ns.VERM, loc="left")
a1.add_patch(Rectangle((ZC, ZR), ZS, ZS, ec="#00e5ff", fc="none", lw=0.9))

a2 = fig.add_subplot(gs[0, 2])
a2.imshow(ref_img[ZR:ZR+ZS, ZC:ZC+ZS])
a2.set_title("zoom", fontsize=6, color=ns.INK, loc="left")
a3 = fig.add_subplot(gs[1, 2])
a3.imshow(np.clip(err[ZR:ZR+ZS, ZC:ZC+ZS] * a.amp, 0, 1), cmap="inferno",
          vmin=0, vmax=1)
a3.set_title("zoom", fontsize=6, color=ns.VERM, loc="left")
for A in (a0, a1, a2, a3):
    A.set_xticks([]); A.set_yticks([])
fig.tight_layout()
for o in (R / a.out, R / "results/seam_problem.png"):
    fig.savefig(o, dpi=300, bbox_inches="tight", facecolor="white")
print(f"  {P}px tiles, {(H+ph)//P}x{(W+pw)//P} grid   penalty {db.item():+.4f} dB")
print(f"  wrote {a.out}")
