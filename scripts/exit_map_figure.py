"""Which exit each tile takes, drawn on the frame itself.

The saving is a number; this is the picture behind it. ClassSR shows which
patches its classifier sends to which branch; the same view here answers the
question a reader actually has -- does the router follow the content, or is it
doing something arbitrary that happens to average out?

Three panels:

  1. the decoded frame with every tile tinted by its assigned exit
  2. the same assignment as a bare map, so the pattern is visible without the
     picture competing with it
  3. the per-tile error the assignment was chosen from, which is what makes the
     pattern explicable rather than decorative

The assignment is the real one: the Lagrangian allocation at the 0.1 dB budget,
recomputed here from the per-tile errors rather than reused from a histogram, so
the tiles in the picture are the tiles that were actually measured.

Colour is deliberately NOT a rainbow. Exits are ordered -- shallow to deep is a
scale, not a set of categories -- so the map uses a sequential ramp and a
reader can see depth without consulting the legend.

    python scripts/exit_map_figure.py --seq Bosphorus --qp 32
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F

R = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(R))
sys.path.insert(0, str(R / "scripts"))
sys.path.insert(0, str(Path.home() / "DCVC"))

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
from matplotlib.colors import ListedColormap, BoundaryNorm  # noqa: E402
from matplotlib.patches import Patch  # noqa: E402
import naturestyle as ns  # noqa: E402

ns.apply()


def main(argv):
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", default="runs/BEST/ckpt_eval.pth.tar")
    ap.add_argument("--seq", default="Bosphorus")
    ap.add_argument("--qp", type=int, default=32)
    ap.add_argument("--budget", type=float, default=0.1)
    ap.add_argument("--device", default="cuda:0")
    ap.add_argument("--out", default="docs/figures/exit_map.png")
    # Provenance beside the picture: without it the figure cannot say which
    # checkpoint routed the tiles it draws, which is the defect that made every
    # earlier qualitative figure in this project unciteable.
    ap.add_argument("--sidecar", default=None,
                    help="JSON file recording checkpoint, sequence, rate, "
                         "budget, the exit histogram and the per-tile penalty")
    a = ap.parse_args(argv)

    import ctc_intra as C
    from src.utils.transforms import ycbcr2rgb
    from flexuf.config import FlexUFConfig
    from flexuf.cost import exit_costs
    from flexuf.eval import per_tile_mse, tiled_exit_mses
    from flexuf.model import FlexUFIntra, load_flexuf_state
    from flexuf.reference import reference_for

    dev = a.device
    ck = torch.load(R / a.ckpt, map_location="cpu", weights_only=False)
    cfg = FlexUFConfig(**ck["config"])
    net = FlexUFIntra(cfg).to(dev).eval()
    load_flexuf_state(net, ck)
    ref = FlexUFIntra(cfg).to(dev).eval()
    load_flexuf_state(ref, torch.load(reference_for(cfg, None),
                                      map_location="cpu", weights_only=False))
    cost = exit_costs(cfg, "head").to(dev)
    K, j, P = cfg.num_exits, cfg.split_depth, cfg.rgb_patch

    seqs, _ = C.discover([])
    s = [q for q in seqs if a.seq in q["name"]] or seqs
    s = s[0]
    x, _ = C.read_frames(s["path"], s["w"], s["h"], 1, 1)
    x = x[0:1].to(dev)
    _, _, H, W = x.shape
    ph, pw = (-H) % P, (-W) % P
    xp = F.pad(x, (0, pw, 0, ph), mode="replicate")
    nh, nw = (H + ph) // P, (W + pw) // P
    qp = torch.full((1,), a.qp, dtype=torch.int32, device=dev)

    with torch.no_grad():
        y, q, _ = net._encode_to_latent(xp, qp)
        full = ref.dec.forward_full(y, q)

        def per_tile(img):
            return per_tile_mse(img, xp, nh, nw, P)

        # DEPLOYED path -- one tiled decode per exit. forward_all_exits runs
        # full frame and would cancel the tiling penalty out of every number on
        # this figure (flexuf/eval.py).
        M = tiled_exit_mses(net.dec, y, q, xp, cfg)         # [tiles, K]
        Rt = per_tile(full)          # each tile's OWN reference error
        Rm = Rt.mean()               # frame-level reference, for the dB budget

        # Bisect lambda for exactly the budget, the same rule the frontier uses.
        lo, hi = 0.0, 1.0
        def at(lam):
            k = (M + lam * cost[None, :]).argmin(1).clamp(min=j)
            db = 10 * torch.log10(M.gather(1, k[:, None]).squeeze(1).mean() / Rm)
            return k, db.item()
        if at(hi)[1] < a.budget:
            k_sel = at(hi)[0]
        else:
            for _ in range(60):
                mid = 0.5 * (lo + hi)
                if at(mid)[1] <= a.budget:
                    lo = mid
                else:
                    hi = mid
            k_sel = at(lo)[0]
        k_sel_db = at(lo)[1]
        saved = 100 * (1 - cost[k_sel].mean()).item()
        km = k_sel.view(nh, nw).cpu().numpy()

        # Per-tile penalty must divide by THAT TILE's reference error, not by
        # the frame mean. Dividing by the mean made easy tiles read -6 dB --
        # "better than the released decoder", which is not a thing an exit can
        # be; it was just measuring that the tile is easier than average.
        pen = (10 * torch.log10(M.gather(1, k_sel[:, None]).squeeze(1)
                                / Rt.clamp_min(1e-12))
               ).view(nh, nw).cpu().numpy()

        rgb = ycbcr2rgb((full + 0.5).clamp(0, 1))[0].clamp(0, 1)
        rgb = rgb.permute(1, 2, 0).cpu().numpy()[:H, :W]

    # exits 0..j-1 cost the same as j and the allocator never picks them, so the
    # scale starts at j -- labelling six rungs when four have distinct prices
    # would misdescribe the ladder.
    used = list(range(j, K))
    ramp = plt.get_cmap("YlOrRd")(np.linspace(0.15, 0.88, len(used)))
    cmap = ListedColormap(ramp)
    norm = BoundaryNorm(np.arange(j - 0.5, K + 0.5), cmap.N)

    fig, ax = plt.subplots(1, 3, figsize=(ns.W2, 2.5))
    ax[0].imshow(rgb)
    ax[0].imshow(np.kron(km, np.ones((P, P)))[:H, :W], cmap=cmap, norm=norm,
                 alpha=0.55, interpolation="nearest")
    for r in range(1, nh):
        ax[0].axhline(r * P, color="white", lw=0.35, alpha=0.6)
    for c in range(1, nw):
        ax[0].axvline(c * P, color="white", lw=0.35, alpha=0.6)
    ax[0].set_title(f"qp {a.qp}", fontsize=6, color=ns.INK2, loc="left")

    im = ax[1].imshow(km, cmap=cmap, norm=norm, interpolation="nearest")
    for (r, c), v in np.ndenumerate(km):
        ax[1].text(c, r, str(v), ha="center", va="center", fontsize=4.5,
                   color="white" if v >= K - 2 else ns.INK)
    ax[1].set_title(f"{saved:.1f}% saved, {k_sel_db:.3f} dB", fontsize=6,
                    color=ns.INK2, loc="left")

    im2 = ax[2].imshow(pen, cmap="viridis", interpolation="nearest")
    ax[2].set_title("dB per tile", fontsize=6, color=ns.INK2, loc="left")
    cb = fig.colorbar(im2, ax=ax[2], fraction=0.046, pad=0.02)
    cb.ax.tick_params(labelsize=4.5)

    for A in ax:
        A.set_xticks([]); A.set_yticks([])
    # Legend under the panel rather than over the picture: at 4 pt on a bright
    # frame it was unreadable and covering the water it was meant to explain.
    ax[0].legend(handles=[Patch(facecolor=ramp[i],
                                label=f"exit {k} · {100*(1-float(cost[k])):.0f}% saved")
                          for i, k in enumerate(used)],
                 loc="upper center", bbox_to_anchor=(0.5, -0.02), fontsize=4.6,
                 ncol=2, frameon=False)
    for i, l in enumerate("abc"):
        ns.panel(ax[i], l, dx=-0.06, dy=1.04)
    fig.tight_layout(w_pad=1.4)
    out = R / a.out
    out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out, dpi=300, bbox_inches="tight", facecolor="white")
    plt.close(fig)
    hist = np.bincount(km.ravel(), minlength=K)
    print(f"  {nh}x{nw} = {nh*nw} tiles, histogram {hist.tolist()}")
    print(f"  {saved:.2f}% saved at {k_sel_db:.4f} dB")
    print(f"  wrote {a.out}")

    if a.sidecar:
        import json
        (R / a.sidecar).write_text(json.dumps(
            {"figure": str(a.out),
             "what": "which exit each tile takes, and what each tile pays",
             "ckpt": a.ckpt, "ckpt_epoch": ck.get("epoch"),
             "ckpt_step": ck.get("step"),
             "seq": s["name"], "cls": s["cls"],
             "resolution": [s["w"], s["h"]], "qp": a.qp,
             "budget_db": a.budget, "delivered_db": k_sel_db,
             "saving_pct_vs_release": saved,
             "grid": [nh, nw], "n_tiles": int(nh * nw), "tile_px": P,
             "exit_hist": hist.tolist(),
             "exit_share_pct": [100 * float(h) / max(1, int(hist.sum()))
                                for h in hist],
             "exit_map_row_major": km.ravel().tolist(),
             "tile_penalty_db": pen.ravel().tolist(),
             "tile_penalty_db_max": float(pen.max()),
             "tile_penalty_db_mean": float(pen.mean()),
             "penalty_definition": "10 log10(tile MSE at its chosen exit / the "
                                   "released decoder's MSE on the same tile)",
             "exit_cost_vector": [float(c) for c in cost.tolist()]},
            indent=2))
        print(f"  wrote {a.sidecar}")
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
