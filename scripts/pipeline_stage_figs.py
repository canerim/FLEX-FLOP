"""Diagrams for the tail of the decoder: seam repair, and the head.

Both are drawn from the modules themselves. The gate is the trained tensor, not
a sketch of one, and the head's shapes are read off a real forward pass, so
neither figure can drift from the code.

    python scripts/pipeline_stage_figs.py --device cuda:0
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
sys.path.insert(0, str(Path(__file__).resolve().parent))
from gpu import pick as _gpu  # noqa: E402

import ctc_intra as C                                     # noqa: E402
import naturestyle as ns                                  # noqa: E402
from flexuf.backbone.decoder import patchify, unpatchify  # noqa: E402
from flexuf.config import FlexUFConfig                    # noqa: E402
from flexuf.model import FlexUFIntra, load_flexuf_state   # noqa: E402

ns.apply(ns.for_column())
def bare(ax):
    ax.set_xticks([]); ax.set_yticks([])
    for sp in ax.spines.values():
        sp.set_linewidth(0.4); sp.set_color(ns.GRID)
    return ax


def seam_fig(net, cfg, f, dev):
    """The gate, where it sits, and what it multiplies."""
    rep = net.dec.seam_repair
    if rep is None:
        print("  no seam repair in this checkpoint; skipping")
        return
    P = rep.patch
    g = rep.gate.detach()[0, 0].cpu().numpy()
    with torch.no_grad():
        h = rep.pw(rep.act(rep.dw(f)))
        H, W = f.shape[-2:]
        gg = rep.gate.repeat(1, 1, -(-H // P), -(-W // P))[:, :, :H, :W]
        corr = (gg * h)[0].pow(2).mean(0).sqrt().cpu().numpy()

    fig = plt.figure(figsize=(ns.W2, 1.62))
    gs = fig.add_gridspec(1, 4, width_ratios=[0.62, 0.62, 1.5, 1.0],
                          wspace=0.3)

    a0 = bare(fig.add_subplot(gs[0, 0]))
    im = a0.imshow(g, cmap="viridis", vmin=0, vmax=max(1.0, g.max()))
    a0.set_title(f"gate G\n{P}×{P}", fontsize=ns.fs(6), color=ns.INK2, loc="left",
                 pad=3)
    ns.panel(a0, "a")
    cb = fig.colorbar(im, ax=a0, fraction=0.046, pad=0.03)
    cb.ax.tick_params(labelsize=ns.fs(6), length=1.5, width=0.4)

    a1 = bare(fig.add_subplot(gs[0, 1]))
    a1.imshow(np.tile(g, (3, 3)), cmap="viridis", vmin=0,
              vmax=max(1.0, g.max()))
    for t in (1, 2):
        a1.axhline(t * P - 0.5, color="w", lw=0.5)
        a1.axvline(t * P - 0.5, color="w", lw=0.5)
    a1.set_title("tiled over the canvas", fontsize=ns.fs(6), color=ns.INK2,
                 loc="left", pad=3)
    ns.panel(a1, "b")

    a2 = bare(fig.add_subplot(gs[0, 2]))
    a2.imshow(corr, cmap="inferno")
    a2.set_title("what it actually adds,\non a real frame", fontsize=ns.fs(6),
                 color=ns.INK2, loc="left", pad=3)
    ns.panel(a2, "c")

    a3 = fig.add_subplot(gs[0, 3]); a3.axis("off")
    a3.set_xlim(0, 10); a3.set_ylim(0, 10)
    d = np.minimum(np.arange(P), P - 1 - np.arange(P)).astype(float)
    # A cut through the middle of a tile, not along its top edge: the edge row
    # is high everywhere and hides the shape the gate actually has.
    prof = g[P // 2, :]
    a3b = a3.inset_axes([0.06, 0.16, 0.92, 0.66])
    a3b.plot(np.arange(P), prof, color=ns.VERM, lw=1.0)
    a3b.set_xlabel("position across a tile", fontsize=ns.fs(6))
    a3b.set_ylabel("gate", fontsize=ns.fs(6))
    a3b.tick_params(labelsize=ns.fs(6), length=1.5, width=0.4)
    for sp in ("top", "right"):
        a3b.spines[sp].set_visible(False)
    a3b.axhline(prof.min(), color=ns.GRID, lw=0.5, ls=":")
    a3b.text(P / 2, prof.min(), f"{prof.min():.2f} in the interior",
             fontsize=ns.fs(6), color=ns.INK2, ha="center", va="bottom")
    ns.panel(a3, "d", dx=-0.02, dy=0.92)

    for _d in (ROOT / "docs/figures", ROOT / "paper/figures"):
        _d.mkdir(parents=True, exist_ok=True)
        fig.savefig(_d / "seam_gate.png", dpi=500, bbox_inches="tight",
                    pad_inches=0.02, facecolor="white")
    out = ROOT / "paper/figures/seam_gate.png"
    # The caption quotes the gate's corner value and its plateau. They were
    # typed into it; a figure that prints its numbers to the terminal and not
    # to a file is a figure whose caption cannot be checked.
    import json as _json
    (ROOT / "results/seam_gate.json").write_text(_json.dumps({
        "ckpt": str(getattr(net, "_ckpt_path", "")),
        "epoch": int(getattr(net, "_ckpt_epoch", -1)),
        "gate_min": float(g.min()), "gate_max": float(g.max()),
        "gate_mean": float(g.mean()),
    }, indent=2))
    print(f"  wrote {out.relative_to(ROOT)}   gate min {g.min():.3f} "
          f"max {g.max():.3f} mean {g.mean():.3f}")


def head_fig(net, cfg, f, q, dev):
    """The head: one block, then PixelShuffle back to the frame."""
    with torch.no_grad():
        scaled = f * q
        blk = net.dec.head(scaled)
        img = F.pixel_shuffle(blk, 8)

    fig = plt.figure(figsize=(ns.W2, 1.28))
    ax = fig.add_axes([0, 0, 1, 1]); ax.set_xlim(0, 100); ax.set_ylim(0, 22)
    ax.axis("off")

    def box(cx, w, t, sub, col, y=13.0, h=8.0):
        ax.add_patch(FancyBboxPatch((cx - w / 2, y - h / 2), w, h,
                                    boxstyle="round,pad=0.3,rounding_size=0.8",
                                    fc="white", ec=col, lw=0.8, zorder=2))
        ax.text(cx, y + 1.4, t, ha="center", va="center", fontsize=ns.fs(6),
                color=ns.INK, zorder=3)
        ax.text(cx, y - 2.1, sub, ha="center", va="center", fontsize=ns.fs(6),
                color=ns.INK2, zorder=3)

    def arr(x0, x1, lab=""):
        ax.add_patch(FancyArrowPatch((x0, 13.0), (x1, 13.0), arrowstyle="-|>",
                                     mutation_scale=6, lw=0.7, color="#7a8899",
                                     shrinkA=2, shrinkB=2, zorder=1))
        if lab:
            ax.text((x0 + x1) / 2, 17.6, lab, ha="center", va="bottom",
                    fontsize=ns.fs(6), color=ns.INK2)

    C_, Hh, Wf = f.shape[1], f.shape[2], f.shape[3]
    box(11, 20, f"stitched feature\\n[1, {C_}, {Hh}, {Wf}]",
        "every tile written back", ns.BLUE)
    box(35, 17, "× quantisation step",
        "one scalar per quality index", "#7a8899")
    box(59, 19, f"head: one DepthConvBlock\\n{C_} → {blk.shape[1]}",
        "the only block after the trunk", ns.ORANGE)
    box(86, 21, f"PixelShuffle 8\\n[1, 3, {img.shape[2]}, {img.shape[3]}]",
        "channels become pixels", ns.GREEN)
    arr(21.4, 26.2); arr(43.8, 49.2); arr(68.8, 75.2)
    ax.text(50, 3.0, "The head runs once, on the whole frame, after every tile "
            "has been written back. It is 2.4% of the decode and it is the same "
            "block whatever depth the tiles took.",
            ha="center", va="center", fontsize=ns.fs(6), color=ns.INK2)

    out = ROOT / "docs/figures/head.png"
    for _d in (ROOT / "docs/figures", ROOT / "paper/figures"):
        _d.mkdir(parents=True, exist_ok=True)
        fig.savefig(_d / "head.png", dpi=500, bbox_inches="tight",
                    pad_inches=0.02, facecolor="white")
    print(f"  wrote {out.relative_to(ROOT)}   "
          f"{tuple(f.shape)} -> {tuple(blk.shape)} -> {tuple(img.shape)}")


def main(argv):
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", default="runs/RECIPE512/ckpt_PAPER.pth.tar")
    ap.add_argument("--seq", default="Bosphorus")
    ap.add_argument("--qp", type=int, default=32)
    ap.add_argument("--device", default=_gpu("cuda:0"))
    a = ap.parse_args(argv)

    torch.cuda.set_device(a.device)
    ck = torch.load(ROOT / a.ckpt, map_location="cpu", weights_only=False)
    cfg = FlexUFConfig(**ck["config"])
    net = FlexUFIntra(cfg).to(a.device).eval()
    load_flexuf_state(net, ck)
    net._ckpt_path = a.ckpt
    net._ckpt_epoch = int(ck.get("epoch", -1))

    seqs, _ = C.discover([])
    s_ = ([q for q in seqs if a.seq in q["name"]] or seqs)[0]
    x, _ = C.read_frames(s_["path"], s_["w"], s_["h"], 1, 1)
    x = x[0:1].to(a.device)
    P = cfg.rgb_patch
    H0, W0 = x.shape[-2:]
    xp = F.pad(x, (0, (-W0) % P, 0, (-H0) % P), mode="replicate")
    qp = torch.full((1,), a.qp, dtype=torch.int32, device=a.device)
    with torch.no_grad():
        y, q, _ = net._encode_to_latent(xp, qp)
        f = net.dec.upsample(y)
        for g in range(cfg.num_exits):
            f = net.dec.groups[g](f)
    seam_fig(net, cfg, f, a.device)
    head_fig(net, cfg, f, q, a.device)
    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))
