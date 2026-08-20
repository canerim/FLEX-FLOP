"""What patchify actually does, on a real frame.

The operation is two lines of tensor algebra and it is the hinge of the whole
method, so it is worth drawing exactly rather than describing. It is also the
one step readers assume happens in the pixel domain and it does not: the decoder
patchifies the FEATURE map after the shared stem, at one eighth of frame
resolution, so a 256 px tile of the picture is a 32x32 tile of features.

Everything here is captured from a real decode of a real frame rather than
mocked: the frame, the stem feature, the tile grid and the reconstruction all
come from the same forward pass.

    python scripts/patchify_figure.py --device cuda:0
"""

from __future__ import annotations

import argparse, sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn.functional as F
from matplotlib.patches import FancyArrowPatch, FancyBboxPatch, Rectangle

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT)); sys.path.insert(0, str(ROOT / "scripts"))
sys.path.insert(0, str(Path.home() / "DCVC"))

import ctc_intra as C                                     # noqa: E402
import naturestyle as ns                                  # noqa: E402
from flexuf.backbone.decoder import patchify, unpatchify  # noqa: E402
from flexuf.config import FlexUFConfig                    # noqa: E402
from flexuf.model import FlexUFIntra, load_flexuf_state   # noqa: E402

ns.apply()


def main(argv):
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", default="runs/RECIPE512/ckpt_PAPER.pth.tar")
    ap.add_argument("--seq", default="Bosphorus")
    ap.add_argument("--qp", type=int, default=32)
    ap.add_argument("--device", default="cuda:0")
    a = ap.parse_args(argv)

    torch.cuda.set_device(a.device)
    ck = torch.load(ROOT / a.ckpt, map_location="cpu", weights_only=False)
    cfg = FlexUFConfig(**ck["config"])
    net = FlexUFIntra(cfg).to(a.device).eval()
    load_flexuf_state(net, ck)
    j, P, Fp = cfg.split_depth, cfg.rgb_patch, cfg.feature_patch

    seqs, _ = C.discover([])
    s_ = ([q for q in seqs if a.seq in q["name"]] or seqs)[0]
    name = s_["name"]
    x, _ = C.read_frames(s_["path"], s_["w"], s_["h"], 1, 1)
    x = x[0:1].to(a.device)
    B, _, H0, W0 = x.shape
    ph, pw = (-H0) % P, (-W0) % P
    xp = F.pad(x, (0, pw, 0, ph), mode="replicate") if (ph or pw) else x
    H, W = xp.shape[-2:]
    qp = torch.full((B,), a.qp, dtype=torch.int32, device=a.device)

    with torch.no_grad():
        y, q, _ = net._encode_to_latent(xp, qp)
        f = net.dec.upsample(y)
        for g in range(j):
            f = net.dec.groups[g](f)
        tiles, nh, nw = patchify(f, Fp)
        back = unpatchify(tiles, nh, nw, batch=B)
        exact = bool(torch.equal(back, f))
        rec = net.dec.forward_full(y, q)

    def rgb(t):
        from src.utils.transforms import ycbcr2rgb
        v = ycbcr2rgb((t + 0.5).clamp(0, 1))[0].clamp(0, 1)
        return v.permute(1, 2, 0).cpu().numpy()

    img = rgb(xp)[:H0, :W0]
    # One number per feature position, so the tiling is visible: the root mean
    # square over the 384 channels. Any single channel would be an arbitrary
    # choice and most of them are nearly flat.
    fm = f[0].pow(2).mean(0).sqrt().cpu().numpy()
    fh, fw = fm.shape

    print(f"  {name}")
    print(f"  frame {H0}x{W0}, padded {H}x{W}, feature {fh}x{fw} "
          f"(1/{H // fh} of the frame)")
    print(f"  tiles {nh}x{nw} = {nh * nw}, each {Fp}x{Fp} features "
          f"= {P}x{P} pixels")
    print(f"  patchify: {tuple(f.shape)} -> {tuple(tiles.shape)}")
    print(f"  unpatchify(patchify(f)) == f exactly: {exact}")

    # ---------------------------------------------------------------- layout
    fig = plt.figure(figsize=(ns.W2, 2.78))
    # The three top panels have different aspects, so the row is sized by the
    # tallest and the band beneath it has to be tight or the figure is mostly
    # white. hspace is negative for that reason.
    gs = fig.add_gridspec(2, 3, height_ratios=[1.0, 0.50],
                          width_ratios=[1.0, 1.0, 0.78],
                          hspace=0.02, wspace=0.16)

    def bare(ax):
        ax.set_xticks([]); ax.set_yticks([])
        for sp in ax.spines.values():
            sp.set_linewidth(0.4); sp.set_color(ns.GRID)
        return ax

    # a. the frame, with the grid the decoder will impose
    a0 = bare(fig.add_subplot(gs[0, 0]))
    a0.imshow(img)
    for r in range(1, nh):
        a0.axhline(r * P, color="#00e5ff", lw=0.55)
    for c in range(1, nw):
        a0.axvline(c * P, color="#00e5ff", lw=0.55)
    a0.set_title(f"frame, {W0}×{H0}, padded to {W}×{H}", fontsize=5.4,
                 color=ns.INK2, loc="left", pad=3)
    ns.panel(a0, "a")

    # b. the stem feature, same grid, one eighth the size
    a1 = bare(fig.add_subplot(gs[0, 1]))
    a1.imshow(fm, cmap="magma")
    for r in range(1, nh):
        a1.axhline(r * Fp, color="#00e5ff", lw=0.55)
    for c in range(1, nw):
        a1.axvline(c * Fp, color="#00e5ff", lw=0.55)
    a1.set_title(f"stem feature after {j} shared blocks, {fw}×{fh}×{f.shape[1]}",
                 fontsize=5.4, color=ns.INK2, loc="left", pad=3)
    ns.panel(a1, "b")

    # c. four tiles, as the trunk now sees them: separate batch elements
    gc = gs[0, 2].subgridspec(2, 2, hspace=0.12, wspace=0.12)
    pick = [0, nw // 2, nh * nw // 2 + 1, nh * nw - 1]
    vmin, vmax = fm.min(), fm.max()
    for n, t in enumerate(pick):
        ax = bare(fig.add_subplot(gc[n // 2, n % 2]))
        ax.imshow(tiles[t, 0:].pow(2).mean(0).sqrt().cpu().numpy(),
                  cmap="magma", vmin=vmin, vmax=vmax)
        ax.set_xlabel(f"tile {t}", fontsize=4.4, color=ns.INK2, labelpad=1)
        if n == 0:
            ax.set_title(f"{nh * nw} independent tiles, {Fp}×{Fp} each",
                         fontsize=5.4, color=ns.INK2, loc="left", pad=3)
            ns.panel(ax, "c")

    # d. the tensor algebra, written out
    ax = fig.add_subplot(gs[1, :]); ax.axis("off")
    ax.set_xlim(0, 100); ax.set_ylim(0, 30)

    def box(cx, w, txt, sub, colour, y=14.5, h=8.2):
        ax.add_patch(FancyBboxPatch((cx - w / 2, y - h / 2), w, h,
                                    boxstyle="round,pad=0.3,rounding_size=0.8",
                                    fc="white", ec=colour, lw=0.8, zorder=2))
        ax.text(cx, y + 1.5, txt, ha="center", va="center", fontsize=5.4,
                color=ns.INK, zorder=3)
        ax.text(cx, y - 2.0, sub, ha="center", va="center", fontsize=4.6,
                color=ns.INK2, zorder=3)

    def arr(x0, x1, label):
        ax.add_patch(FancyArrowPatch((x0, 14.5), (x1, 14.5), arrowstyle="-|>",
                                     mutation_scale=6, lw=0.7, color="#7a8899",
                                     shrinkA=2, shrinkB=2, zorder=1))
        ax.text((x0 + x1) / 2, 19.6, label, ha="center", va="bottom",
                fontsize=4.6, color=ns.INK2)

    C_ = f.shape[1]
    box(11, 20, f"f  [1, {C_}, {fh}, {fw}]", "the stem's output", ns.BLUE)
    box(37, 24, f"view  [1, {C_}, {nh}, {Fp}, {nw}, {Fp}]",
        "split each axis into tile and offset", "#7a8899")
    box(64, 24, f"permute  [1, {nh}, {nw}, {C_}, {Fp}, {Fp}]",
        "move the tile axes to the front", "#7a8899")
    box(90, 18, f"reshape  [{nh * nw}, {C_}, {Fp}, {Fp}]",
        "tiles become the batch", ns.VERM)
    arr(21.4, 24.8, "no copy")
    arr(49.4, 51.8, "no copy")
    arr(76.4, 80.8, "one copy")
    # The way back. Every exit writes its tile into the same buffer, and the
    # buffer is unpatchified once, so the inverse runs on the mixed-depth
    # result rather than per tile.
    ax.add_patch(FancyArrowPatch((90, 8.4), (11, 8.4), arrowstyle="-|>",
                                 mutation_scale=6, lw=0.7, color=ns.GREEN,
                                 connectionstyle="arc3,rad=0.055",
                                 shrinkA=2, shrinkB=2, zorder=1))
    ax.text(50, 4.4, f"unpatchify: view [1, {nh}, {nw}, {C_}, {Fp}, {Fp}] "
                     f"→ permute → reshape [1, {C_}, {fh}, {fw}]",
            ha="center", va="center", fontsize=4.9, color=ns.GREEN)
    ax.text(50, 1.0,
            "Zero multiply-accumulates in either direction, and the round trip "
            "is exact: unpatchify(patchify(f)) equals f bit for bit. What "
            "changes is that a 3×3 in the trunk now sees zero padding where a "
            "neighbour used to be, and that a tile can stop early.",
            ha="center", va="center", fontsize=4.8, color=ns.INK2)
    ns.panel(ax, "d", dx=-0.005, dy=0.92)

    out = ROOT / "docs/figures/patchify.png"
    fig.savefig(out, dpi=500, bbox_inches="tight", pad_inches=0.02,
                facecolor="white")
    print(f"  wrote {out.relative_to(ROOT)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))
