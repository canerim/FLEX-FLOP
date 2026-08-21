"""Grid seam repair, shown on the grid, on one real frame.

The gate figure says what the module is. This one says what it does to a
picture: the same latent decoded in tiles with the repair off and on, the
correction it adds, and a zoom on one grid crossing.

Everything is one forward pass of the pinned checkpoint on Bosphorus.

    python scripts/seam_repair_grid_figure.py --device cuda:0
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn.functional as F
from matplotlib.patches import Rectangle

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "scripts"))
sys.path.insert(0, str(Path.home() / "DCVC"))

import ctc_intra as C                                     # noqa: E402
import naturestyle as ns                                  # noqa: E402
from flexuf.config import FlexUFConfig                    # noqa: E402
from flexuf.model import FlexUFIntra, load_flexuf_state   # noqa: E402
from src.utils.transforms import ycbcr2rgb                # noqa: E402

ns.apply()


def main(argv):
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", default="runs/RECIPE512/ckpt_PAPER.pth.tar")
    ap.add_argument("--seq", default="Bosphorus")
    ap.add_argument("--qp", type=int, default=63)
    ap.add_argument("--device", default="cuda:0")
    ap.add_argument("--amp", type=float, default=25.0)
    a = ap.parse_args(argv)

    torch.cuda.set_device(a.device)
    ck = torch.load(ROOT / a.ckpt, map_location="cpu", weights_only=False)
    cfg = FlexUFConfig(**ck["config"])
    net = FlexUFIntra(cfg).to(a.device).eval()
    load_flexuf_state(net, ck)
    K, j, P = cfg.num_exits, cfg.split_depth, cfg.rgb_patch

    seqs, _ = C.discover([])
    s_ = ([q for q in seqs if a.seq in q["name"]] or seqs)[0]
    x, _ = C.read_frames(s_["path"], s_["w"], s_["h"], 1, 1)
    x = x[0:1].to(a.device)
    H0, W0 = x.shape[-2:]
    xp = F.pad(x, (0, (-W0) % P, 0, (-H0) % P), mode="replicate")
    H, W = xp.shape[-2:]
    nh, nw = H // P, W // P
    qp = torch.full((1,), a.qp, dtype=torch.int32, device=a.device)
    deep = torch.full((nh * nw,), K - 1, dtype=torch.long, device=a.device)

    keep = net.dec.seam_repair
    with torch.no_grad():
        y, q, _ = net._encode_to_latent(xp, qp)
        net.dec.seam_repair = None
        off = net.dec(y, q, exit_map=deep)
        net.dec.seam_repair = keep
        on = net.dec(y, q, exit_map=deep)
        ref = net.dec.forward_full(y, q)

    def rgb(t):
        v = ycbcr2rgb((t + 0.5).clamp(0, 1))[0].clamp(0, 1)
        return v.permute(1, 2, 0).cpu().numpy()[:H0, :W0]

    img_off, img_on, img_ref = rgb(off), rgb(on), rgb(ref)
    e_off = np.abs(img_off - img_ref).mean(-1)
    e_on = np.abs(img_on - img_ref).mean(-1)
    delta = np.abs(img_on - img_off).mean(-1)

    def db(t):
        return float(10 * torch.log10(((t - xp) ** 2).mean()
                                      / ((ref - xp) ** 2).mean()))
    d_off, d_on = db(off), db(on)

    # A crossing of the tile grid, away from the frame edge.
    r0, c0, S = 2 * P - 96, 3 * P - 96, 192

    # Three rows of two at single-column width rather than two rows of three
    # at double. A full-width figure in this two-column flow abandons the rest
    # of the page it starts on, and this figure is not worth a page.
    fig = plt.figure(figsize=(ns.W1, 3.25))
    gs = fig.add_gridspec(3, 2, height_ratios=[1.0, 1.0, 1.0],
                          hspace=0.24, wspace=0.06)

    def bare(ax):
        ax.set_xticks([])
        ax.set_yticks([])
        for sp in ax.spines.values():
            sp.set_linewidth(0.4)
            sp.set_color(ns.GRID)
        return ax

    def grid(ax, off_r=0, off_c=0):
        for r in range(1, nh):
            ax.axhline(r * P - off_r, color="#00e5ff", lw=0.45, alpha=0.75)
        for c in range(1, nw):
            ax.axvline(c * P - off_c, color="#00e5ff", lw=0.45, alpha=0.75)

    a0 = bare(fig.add_subplot(gs[0, 0]))
    a0.imshow(img_ref)
    grid(a0)
    a0.add_patch(Rectangle((c0, r0), S, S, ec=ns.VERM, fc="none", lw=0.9))
    a0.set_title(f"{a.seq} at q{a.qp}, the {nh}x{nw} tile grid", fontsize=5.2,
                 color=ns.INK2, loc="left", pad=3)
    ns.panel(a0, "a")

    a1 = bare(fig.add_subplot(gs[0, 1]))
    a1.imshow(np.clip(e_off * a.amp, 0, 1), cmap="inferno", vmin=0, vmax=1)
    a1.set_title(f"repair off, {d_off:+.3f} dB", fontsize=5.2, color=ns.VERM,
                 loc="left", pad=3)
    ns.panel(a1, "b")

    a2 = bare(fig.add_subplot(gs[1, 0]))
    a2.imshow(np.clip(e_on * a.amp, 0, 1), cmap="inferno", vmin=0, vmax=1)
    a2.set_title(f"repair on, {d_on:+.3f} dB", fontsize=5.2, color=ns.GREEN,
                 loc="left", pad=3)
    ns.panel(a2, "c")

    a3 = bare(fig.add_subplot(gs[1, 1]))
    a3.imshow(img_off[r0:r0 + S, c0:c0 + S])
    grid(a3, off_r=r0, off_c=c0)
    a3.set_title("zoom, repair off", fontsize=5.2, color=ns.VERM, loc="left",
                 pad=3)
    ns.panel(a3, "d")

    a4 = bare(fig.add_subplot(gs[2, 0]))
    a4.imshow(img_on[r0:r0 + S, c0:c0 + S])
    grid(a4, off_r=r0, off_c=c0)
    a4.set_title("zoom, repair on", fontsize=5.2, color=ns.GREEN, loc="left",
                 pad=3)
    ns.panel(a4, "e")

    a5 = bare(fig.add_subplot(gs[2, 1]))
    a5.imshow(np.clip(delta[r0:r0 + S, c0:c0 + S] * a.amp * 2, 0, 1),
              cmap="inferno", vmin=0, vmax=1)
    grid(a5, off_r=r0, off_c=c0)
    a5.set_title(f"what the repair changed, x{a.amp * 2:.0f}", fontsize=5.2,
                 color=ns.INK2, loc="left", pad=3)
    ns.panel(a5, "f")

    # paper_figures/ is the directory the author asked for; paper/figures/ is
    # the one build_pdf actually reads. Writing only the first left the
    # paper's copy at whatever had been placed there by hand.
    for out in (ROOT / "docs/figures/seam_repair_grid.png",
                ROOT / "paper/figures/seam_repair_grid.png",
                ROOT / "paper_figures/seam_repair_grid.png"):
        out.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(out, dpi=500, bbox_inches="tight", pad_inches=0.02,
                    facecolor="white")
    print(f"  {s_['name']}  q{a.qp}  {nh}x{nw} tiles")
    print(f"  seam penalty: repair off {d_off:+.4f} dB, on {d_on:+.4f} dB, "
          f"recovered {d_off - d_on:+.4f}")
    print(f"  mean |change| from the repair: {delta.mean():.5f} "
          f"(peak {delta.max():.4f})")
    print("  wrote docs/figures, paper/figures and paper_figures "
          "copies of seam_repair_grid.png")
    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))
