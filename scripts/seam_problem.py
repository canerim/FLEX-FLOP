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

# Layout. The three panels are placed by hand in figure coordinates rather than
# by a gridspec, because their aspects differ and a grid cannot make differing
# aspects sit flush. Two 16:9 frames and one column of two squares: solving
# 2*AR*h + s = width with s = (h - gap)/2 gives the row height h directly, and
# every panel then abuts its neighbour with no white gutter to trim.
#
# The earlier version spanned the frame across two rows of a uniform grid and
# left about half the figure as margin, with a title line wide enough to be
# clipped by the figure box.
AR = W / H
GAP, TOP = 0.03, 0.17
h = (ns.W2 - 1.5 * GAP) / (2 * AR + 0.5)
sq = 0.5 * (h - GAP)
fw = AR * h
fig = plt.figure(figsize=(ns.W2, h + TOP))
FH = h + TOP


def place(x, y, w, hh):
    A = fig.add_axes([x / ns.W2, y / FH, w / ns.W2, hh / FH])
    A.set_xticks([]); A.set_yticks([])
    for sp in A.spines.values():
        sp.set_visible(False)
    return A


def label(A, text, colour, letter, panel_w):
    # The panel letter sits on the title line rather than hanging outside the
    # axes. Hanging it outside needs a left margin, and with the panels placed
    # edge to edge there is none: the first letter was clipped by the figure
    # box. The offset is converted from points through this panel's own width,
    # so the gap after the letter is the same on a wide panel and a narrow one.
    A.text(0, 1.015, letter, transform=A.transAxes, fontsize=7,
           color=ns.INK, ha="left", va="bottom", fontweight="bold")
    A.text(7.5 / 72 / panel_w, 1.015, text, transform=A.transAxes, fontsize=6,
           color=colour, ha="left", va="bottom")


ZR, ZC, ZS = 384, 640, 320          # a window on a grid crossing
a0 = place(0, 0, fw, h)
a0.imshow(ref_img, aspect="auto")
label(a0, "decoded full frame", ns.INK, "a", fw)
a0.add_patch(Rectangle((ZC, ZR), ZS, ZS, ec="#00e5ff", fc="none", lw=0.7))

a1 = place(fw + GAP, 0, fw, h)
a1.imshow(np.clip(err * a.amp, 0, 1), cmap="inferno", vmin=0, vmax=1,
          aspect="auto")
label(a1, f"|error| from tiling alone, ×{a.amp:.0f}   {db.item():+.3f} dB",
      ns.VERM, "b", fw)
a1.add_patch(Rectangle((ZC, ZR), ZS, ZS, ec="#00e5ff", fc="none", lw=0.7))

x2 = 2 * (fw + GAP)
a2 = place(x2, h - sq, sq, sq)
a2.imshow(ref_img[ZR:ZR+ZS, ZC:ZC+ZS], aspect="auto")
label(a2, "zoom", ns.INK, "c", sq)
a3 = place(x2, 0, sq, sq)
a3.imshow(np.clip(err[ZR:ZR+ZS, ZC:ZC+ZS] * a.amp, 0, 1), cmap="inferno",
          vmin=0, vmax=1, aspect="auto")

# No suptitle. The sequence, the quality index and the padding rule belong in
# the caption, where a reader looks for provenance.
for o in (R / a.out, R / "results/seam_problem.png"):
    fig.savefig(o, dpi=500, facecolor="white")
print(f"  {P}px tiles, {(H+ph)//P}x{(W+pw)//P} grid   penalty {db.item():+.4f} dB")
print(f"  wrote {a.out}")
