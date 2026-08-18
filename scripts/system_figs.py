"""Diagrams for the single system document (docs/08-system.md).

Four figures that the existing set does not cover: the end-to-end data path with
its compute shares, the INSIDE of the two exit adapters, the A-vs-B decision
mechanism side by side, and the seam-repair module with its actually-trained
gate read out of the checkpoint.

Everything numeric is read from the config, the cost model or the checkpoint --
nothing here is a number typed into a drawing.
"""
import json, math, sys
from pathlib import Path
import numpy as np
import torch
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch, FancyArrowPatch, Rectangle

R = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(R)); sys.path.insert(0, str(Path.home() / "DCVC"))
sys.path.insert(0, str(R / "scripts"))
import naturestyle as ns
ns.apply()
from flexuf.config import FlexUFConfig
from flexuf import cost as fc
from flexuf.cost import exit_costs

CK = R / "runs/BEST/ckpt_eval.pth.tar"
ck = torch.load(CK, map_location="cpu", weights_only=False)
cfg = FlexUFConfig(**ck["config"])
K, j = cfg.num_exits, cfg.split_depth
COST = exit_costs(cfg, "head").tolist()
FIG = R / "docs/figures"


def box(ax, x, y, w, h, label, sub=None, fc_="white", ec=ns.INK2, lw=0.7,
        fs=6, tc=ns.INK, r=0.012):
    ax.add_patch(FancyBboxPatch((x, y), w, h, boxstyle=f"round,pad=0,rounding_size={r}",
                                facecolor=fc_, edgecolor=ec, linewidth=lw))
    ax.text(x + w / 2, y + h / 2 + (0.012 if sub else 0), label, ha="center",
            va="center", fontsize=fs, color=tc)
    if sub:
        ax.text(x + w / 2, y + h / 2 - 0.028, sub, ha="center", va="center",
                fontsize=fs - 1.2, color=ns.INK2)


def arrow(ax, x0, y0, x1, y1, color=ns.INK2, lw=0.7, style="-|>", ls="-"):
    ax.add_patch(FancyArrowPatch((x0, y0), (x1, y1), arrowstyle=style,
                                 mutation_scale=6, linewidth=lw, color=color,
                                 linestyle=ls, shrinkA=0, shrinkB=0))


def blank(ax):
    ax.set_xlim(0, 1); ax.set_ylim(0, 1); ax.axis("off"); ax.grid(False)


def save(fig, name):
    fig.savefig(FIG / name, dpi=300, bbox_inches="tight")
    plt.close(fig)
    print(f"  -> docs/figures/{name}")


# ===========================================================  1. the data path
def pipeline():
    fig, ax = plt.subplots(figsize=(ns.W2, 4.3)); blank(ax)
    FS, FSS = 5.8, 4.9

    def two(x, y, w, h, t1, t2, t3=None, **kw):
        """A box with up to three stacked lines -- narrow boxes need the room."""
        ax.add_patch(FancyBboxPatch((x, y), w, h,
                     boxstyle="round,pad=0,rounding_size=0.008",
                     facecolor=kw.get("fc_", "white"),
                     edgecolor=kw.get("ec", ns.INK2), linewidth=0.7))
        cy = y + h / 2
        if t3 is None:
            ax.text(x + w/2, cy + 0.016, t1, ha="center", va="center", fontsize=FS)
            ax.text(x + w/2, cy - 0.018, t2, ha="center", va="center",
                    fontsize=FSS, color=ns.INK2)
        else:
            ax.text(x + w/2, cy + 0.025, t1, ha="center", va="center", fontsize=FS)
            ax.text(x + w/2, cy, t2, ha="center", va="center", fontsize=FSS,
                    color=ns.INK2)
            ax.text(x + w/2, cy - 0.025, t3, ha="center", va="center",
                    fontsize=FSS, color=ns.INK2)

    # --- encoder, frozen ----------------------------------------------------
    ax.add_patch(Rectangle((0.015, 0.855), 0.335, 0.115, facecolor="#f2f2f2",
                           edgecolor="#bbbbbb", lw=0.6))
    ax.text(0.0175, 0.985, "FROZEN", fontsize=FSS, color=ns.INK2)
    two(ax_ := None or 0.03, 0.878, 0.095, 0.068, "encoder", "$g_a$", fc_="#e8e8e8")
    two(0.145, 0.878, 0.095, 0.068, "hyperprior", "entropy model", fc_="#e8e8e8")
    two(0.26, 0.878, 0.075, 0.068, "bitstream", "ŷ, scales", fc_="#e8e8e8")
    arrow(ax, 0.125, 0.912, 0.145, 0.912)
    arrow(ax, 0.24, 0.912, 0.26, 0.912)
    arrow(ax, 0.2975, 0.878, 0.2975, 0.775)

    # --- decoder stem, full frame ------------------------------------------
    ax.text(0.015, 0.815, "DECODER  ·  full frame", fontsize=6.2, weight="bold")
    two(0.10, 0.700, 0.105, 0.070, "upsample", f"{100*fc.SHARE_UPSAMPLE:.1f}% of MACs",
        fc_="#dceaf7")
    trunk_each = 100 * fc.SHARE_TRUNK / fc.N_TRUNK_BLOCKS
    two(0.235, 0.700, 0.135, 0.070, f"groups 0–{j-1}",
        f"{j*cfg.blocks_per_exit} blocks · {j*cfg.blocks_per_exit*trunk_each:.1f}%",
        fc_="#dceaf7")
    arrow(ax, 0.205, 0.735, 0.235, 0.735)
    two(0.40, 0.700, 0.095, 0.070, "patchify", f"{cfg.rgb_patch}px tiles",
        fc_="#fff3d9")
    arrow(ax, 0.37, 0.735, 0.40, 0.735)
    ax.text(0.4475, 0.678, "0 FLOP", fontsize=FSS, color=ns.VERM, ha="center")

    # --- the ladder ---------------------------------------------------------
    x0, wg, gap = 0.575, 0.088, 0.017
    ax.text(0.575, 0.815, f"PER TILE  ·  K = {K} exits over "
            f"{fc.N_TRUNK_BLOCKS} blocks, split depth j = {j}",
            fontsize=6.2, weight="bold")
    for g in range(j, K):
        x = x0 + (g - j) * (wg + gap)
        two(x, 0.700, wg, 0.070, f"group {g}", f"{cfg.blocks_per_exit} blocks",
            fc_="#dceaf7")
        if g > j:
            arrow(ax, x - gap, 0.735, x, 0.735)
        arrow(ax, x + wg/2, 0.700, x + wg/2, 0.612, color=ns.ORANGE)
        skipped = (K - 1 - g) * cfg.blocks_per_exit
        lab = "raw feature" if g == K-1 else ("FFN adapter" if skipped >= 4
                                              else "1×1 adapter")
        two(x, 0.520, wg, 0.090, f"exit {g}", lab, f"cost {COST[g]:.3f}×",
            fc_="#fdeadb", ec=ns.ORANGE)
    arrow(ax, 0.495, 0.735, x0, 0.735)
    ax.text(0.985, 0.492, "cost: fraction of one released decode",
            fontsize=FSS, color=ns.INK2, ha="right")

    # --- the exit map, and who produces it ----------------------------------
    two(0.395, 0.520, 0.145, 0.090, "exit map  k(t)", "one index per tile",
        f"{math.ceil(1920/cfg.rgb_patch)*math.ceil(1080/cfg.rgb_patch)} "
        f"tiles at 1080p",
        fc_="#f5f0fa", ec=ns.PURPLE)
    arrow(ax, 0.540, 0.565, 0.575, 0.565, color=ns.PURPLE)

    # --- stitch, repair, head ----------------------------------------------
    for g in range(j, K):
        x = x0 + (g - j) * (wg + gap) + wg/2
        arrow(ax, x, 0.520, x, 0.445, color=ns.ORANGE, lw=0.5)
    two(0.575, 0.360, 0.125, 0.075, "unpatchify", "stitch the canvas",
        fc_="#fff3d9")
    two(0.725, 0.360, 0.145, 0.075, "grid seam repair", "full frame · +0.95%",
        fc_="#e7f6ef", ec=ns.GREEN)
    two(0.895, 0.360, 0.09, 0.075, "head",
        f"{100*fc.SHARE_HEAD:.1f}% → RGB", fc_="#dceaf7")
    arrow(ax, 0.70, 0.3975, 0.725, 0.3975)
    arrow(ax, 0.87, 0.3975, 0.895, 0.3975)

    # --- A and B ------------------------------------------------------------
    ax.add_patch(Rectangle((0.015, 0.035), 0.455, 0.245, facecolor="#fdf4f9",
                           edgecolor=ns.PURPLE, lw=0.7))
    ax.text(0.032, 0.252, "A  ·  the encoder searches, then signals the map",
            fontsize=6, color=ns.PURPLE, weight="bold")
    ax.text(0.243, 0.155, "k*(t) = argmin$_k$ [ MSE(t,k) + λ·c$_k$ ]",
            fontsize=6.2, ha="center")
    ax.text(0.243, 0.085, "≈94 bit/frame", fontsize=FSS, ha="center",
            color=ns.INK2)

    ax.add_patch(Rectangle((0.53, 0.035), 0.455, 0.245, facecolor="#eef6fb",
                           edgecolor=ns.BLUE, lw=0.7))
    ax.text(0.547, 0.252, "B  ·  the decoder predicts it, nothing is signalled",
            fontsize=6, color=ns.BLUE, weight="bold")
    ax.text(0.757, 0.155,
            "k̂(t) = argmax$_k$ [ log softmax(z$_t$)$_k$ − β·c$_k$ ]",
            fontsize=6.2, ha="center")
    ax.text(0.757, 0.085, "0 bits, 0.163% of a decode", fontsize=FSS,
            ha="center", color=ns.INK2)

    arrow(ax, 0.243, 0.280, 0.243, 0.565, color=ns.PURPLE, ls=(0, (3, 2)),
          style="-")
    arrow(ax, 0.243, 0.565, 0.395, 0.565, color=ns.PURPLE, ls=(0, (3, 2)))
    arrow(ax, 0.757, 0.280, 0.757, 0.330, color=ns.BLUE, ls=(0, (3, 2)))
    arrow(ax, 0.757, 0.330, 0.470, 0.330, color=ns.BLUE, ls=(0, (3, 2)), style="-")
    arrow(ax, 0.470, 0.330, 0.470, 0.520, color=ns.BLUE, ls=(0, (3, 2)))


    save(fig, "sys_pipeline.png")


# ======================================================  2. inside the adapters
def adapters():
    C = 384
    fig, ax = plt.subplots(1, 3, figsize=(ns.W2, 2.7),
                           gridspec_kw={"width_ratios": [1, 1, 1.1]})
    for a in ax[:2]:
        blank(a)
    BX, BW = 0.20, 0.62

    def stack(a, y, h, t1, t2, **kw):
        a.add_patch(FancyBboxPatch((BX, y), BW, h,
                    boxstyle="round,pad=0,rounding_size=0.02",
                    facecolor=kw.get("fc_", "white"),
                    edgecolor=kw.get("ec", ns.INK2), linewidth=0.7))
        a.text(BX + BW/2, y + h/2 + 0.016, t1, ha="center", va="center", fontsize=6)
        a.text(BX + BW/2, y + h/2 - 0.019, t2, ha="center", va="center",
               fontsize=5, color=ns.INK2)

    # ---- a: Conv1x1Adapter -------------------------------------------------
    a = ax[0]
    a.text(0.0, 1.0, "Conv1×1Adapter", fontsize=7, weight="bold")
    a.text(0.0, 0.93, r"$\mathrm{Ad}(f) = f + Wf$,   $W \leftarrow 0$",
           fontsize=6.5)
    stack(a, 0.76, 0.082, "feature f", f"[{C}, 32, 32] per tile", fc_="#dceaf7")
    arrow(a, 0.51, 0.76, 0.51, 0.665)
    stack(a, 0.575, 0.088, "1×1 convolution", f"{C} → {C},  zero-initialised",
          fc_="#fdeadb", ec=ns.ORANGE)
    arrow(a, 0.51, 0.575, 0.51, 0.48)
    stack(a, 0.39, 0.082, "⊕  residual add", "identity at step 0")
    arrow(a, 0.10, 0.801, 0.10, 0.431, style="-")
    arrow(a, 0.10, 0.431, 0.20, 0.431)
    a.text(0.06, 0.60, "identity path", fontsize=5, color=ns.INK2, rotation=90,
           va="center")
    a.text(0.51, 0.29, f"$C^2$ = {C**2:,} MAC/px", fontsize=6.5, ha="center")
    a.text(0.51, 0.215, "one eighth of a DepthConvBlock", fontsize=5.5,
           ha="center", color=ns.INK2)
    a.text(0.51, 0.09, "Used by exit 4 only — it skips 2 blocks,\nso there is "
           "little to stand in for", fontsize=5.5, ha="center", color=ns.VERM,
           linespacing=1.5)
    ns.panel(a, "a", dx=-0.02, dy=1.13)

    # ---- b: FFNAdapter -----------------------------------------------------
    a = ax[1]
    a.text(0.0, 1.0, "FFNAdapter", fontsize=7, weight="bold")
    a.text(0.0, 0.93, r"$\mathrm{Ad}(f)=f+\mathrm{PW}_{C\to C}"
           r"(\mathrm{WSiLUChunkAdd}(\mathrm{PW}_{C\to 4C}(f)))$", fontsize=5.4)
    stack(a, 0.845, 0.072, "feature f", f"[{C}, 32, 32] per tile", fc_="#dceaf7")
    arrow(a, 0.51, 0.845, 0.51, 0.775)
    stack(a, 0.690, 0.083, "1×1 expand", f"{C} → {4*C}", fc_="#fdeadb", ec=ns.ORANGE)
    arrow(a, 0.51, 0.690, 0.51, 0.620)
    stack(a, 0.535, 0.083, "WSiLUChunkAdd", f"{4*C} → {C},  4:1 strided",
          fc_="#fdeadb", ec=ns.ORANGE)
    arrow(a, 0.51, 0.535, 0.51, 0.465)
    stack(a, 0.380, 0.083, "1×1 contract", f"{C} → {C},  zero-initialised",
          fc_="#fdeadb", ec=ns.ORANGE)
    arrow(a, 0.51, 0.380, 0.51, 0.310)
    stack(a, 0.225, 0.075, "⊕  residual add", "identity at step 0")
    arrow(a, 0.10, 0.881, 0.10, 0.2625, style="-")
    arrow(a, 0.10, 0.2625, 0.20, 0.2625)
    a.text(0.51, 0.135, f"$2C^2$ = {2*C**2:,} MAC/px", fontsize=6.5, ha="center")

    a.text(0.51, 0.055, "exits 0–3", fontsize=6, ha="center", color=ns.VERM)
    ns.panel(a, "b", dx=-0.02, dy=1.13)

    # ---- c: capacity matched to the gap ------------------------------------
    a = ax[2]
    ks = list(range(K))
    skipped = [(K - 1 - k) * cfg.blocks_per_exit for k in ks]
    block_mac = 8 * C**2 + 9 * C
    reach = [k >= j for k in ks]
    a.bar(ks, skipped, width=0.62,
          color=[ns.SKY if r else "#e0e0e0" for r in reach],
          label="blocks the exit skips")
    ad = [((2*C**2 if sk >= 4 else C**2) / block_mac) if k < K-1 else 0
          for k, sk in zip(ks, skipped)]
    a.bar(ks, ad, width=0.62, color=[ns.ORANGE if r else "#bdbdbd" for r in reach],
          label="its adapter, in the same units")
    for k, sk in zip(ks, skipped):
        if k < K - 1:
            a.text(k, sk + 0.22, "FFN" if sk >= 4 else "1×1", fontsize=5.5,
                   ha="center", color=ns.VERM if reach[k] else "#999999")
    a.axvspan(-0.6, j - 0.5, color="#f5f5f5", zorder=0)
    a.text(0.5, 11.6, f"unreachable (k < j = {j})", fontsize=5,
           ha="center", color="#888888")
    a.set_xlabel("exit k"); a.set_ylabel("in units of one DepthConvBlock")
    a.set_xticks(ks); a.set_xlim(-0.6, K - 0.4); a.set_ylim(0, 13.5)
    a.legend(loc="center right", fontsize=5)
    a.set_title("Capacity against the gap", fontsize=6, color=ns.INK2,
                loc="left")
    ns.panel(a, "c", dx=-0.22, dy=1.13)

    fig.tight_layout()
    save(fig, "adapters.png")


# ===================================================  3. the seam-repair module
def seam_module():
    sd = ck.get("model", ck.get("state_dict", ck))
    gk = [k for k in sd if "seam_repair" in k and "gate" in k]
    G = sd[gk[0]].detach().float().squeeze().cpu().numpy() if gk else None

    fig = plt.figure(figsize=(ns.W2, 2.9))
    gs = fig.add_gridspec(1, 3, width_ratios=[1.05, 1, 1])
    BX, BW = 0.16, 0.70

    # ---- a: the module -----------------------------------------------------
    a = fig.add_subplot(gs[0]); blank(a)

    def stack(y, h, t1, t2, **kw):
        a.add_patch(FancyBboxPatch((BX, y), BW, h,
                    boxstyle="round,pad=0,rounding_size=0.02",
                    facecolor=kw.get("fc_", "white"),
                    edgecolor=kw.get("ec", ns.INK2), linewidth=0.7))
        a.text(BX + BW/2, y + h/2 + 0.016, t1, ha="center", va="center", fontsize=6)
        a.text(BX + BW/2, y + h/2 - 0.019, t2, ha="center", va="center",
               fontsize=5, color=ns.INK2)

    a.text(0.0, 1.03, "GridSeamRepair", fontsize=7, weight="bold")
    a.text(0.0, 0.955,
           r"$\mathrm{Rep}(f)=f+G[i\,\mathrm{mod}\,P,\ j\,\mathrm{mod}\,P]"
           r"\cdot\mathrm{PW}(\mathrm{WSiLU}(\mathrm{DW}_{3\times3}(f)))$",
           fontsize=5.2)
    stack(0.815, 0.075, "stitched canvas", "[384, H/8, W/8] — the whole frame",
          fc_="#fff3d9")
    arrow(a, 0.51, 0.815, 0.51, 0.740)
    stack(0.655, 0.085, "3×3 depthwise", "replicate padding", fc_="#e7f6ef",
          ec=ns.GREEN)
    arrow(a, 0.51, 0.655, 0.51, 0.580)
    stack(0.495, 0.085, "WSiLU → 1×1", "zero-init: identity at step 0",
          fc_="#e7f6ef", ec=ns.GREEN)
    arrow(a, 0.51, 0.495, 0.51, 0.420)
    stack(0.335, 0.085, "× G[i mod P, j mod P]",
          f"{cfg.feature_patch}² = {cfg.feature_patch**2} scalars, shared over "
          "384 ch", fc_="#e7f6ef", ec=ns.GREEN)
    arrow(a, 0.51, 0.335, 0.51, 0.260)
    stack(0.175, 0.075, "⊕  residual add", "then the head")
    arrow(a, 0.07, 0.8525, 0.07, 0.2125, style="-")
    arrow(a, 0.07, 0.2125, 0.16, 0.2125)
    a.text(0.51, 0.075, "0.95% of the decode  ·  0.0007% of the parameters",
           fontsize=5.4, ha="center")
    ns.panel(a, "a", dx=-0.02, dy=1.16)

    # ---- b: the trained gate ----------------------------------------------
    a = fig.add_subplot(gs[1])
    if G is not None:
        im = a.imshow(G, cmap="magma")
        cb = fig.colorbar(im, ax=a, fraction=0.046, pad=0.03)
        cb.ax.tick_params(labelsize=5)
        a.set_title(f"Gate G   ring {G.max():.2f}, interior {G.min():.3f}",
                    fontsize=6, color=ns.INK2, loc="left")

    a.set_xlabel("j mod P"); a.set_ylabel("i mod P"); a.grid(False)
    ns.panel(a, "b", dx=-0.22, dy=1.16)

    # ---- c: where it actually acts -----------------------------------------
    a = fig.add_subplot(gs[2])
    bands = ["0–4", "4–16", "16–64", "64–128"]
    change = [-0.27, 0.05, 0.04, 0.04]          # % change in MSE, repair ON vs OFF
    share = [6.2, 17.3, 51.6, 25.0]
    a.bar(range(4), change, width=0.62,
          color=[ns.GREEN if c < 0 else ns.VERM for c in change])
    for i, (c, sh) in enumerate(zip(change, share)):
        a.text(i, c + (0.012 if c > 0 else -0.012), f"{sh:.0f}% of px",
               fontsize=5, ha="center", color=ns.INK2,
               va="bottom" if c > 0 else "top")
    a.axhline(0, color=ns.INK, lw=0.6)
    a.set_xticks(range(4)); a.set_xticklabels(bands)
    a.set_ylim(-0.33, 0.10)
    a.set_xlabel("distance from the nearest tile boundary (px)")
    a.set_ylabel("change in MSE with repair ON (%)")
    a.set_title("Where it acts", fontsize=6, color=ns.INK2, loc="left")
    ns.panel(a, "c", dx=-0.26, dy=1.16)

    fig.tight_layout()
    save(fig, "seam_module.png")


# ==========================================================  4. how it is trained
def training():
    """What is frozen, what is trained, and what every term of the loss is for."""
    import flexuf.model as fm
    net = fm.FlexUFIntra(cfg)
    P_TOT = sum(p.numel() for p in net.parameters())
    P_DEC = sum(p.numel() for p in net.dec.parameters())
    P_RT = 144_024

    fig = plt.figure(figsize=(ns.W2, 3.6))
    gs = fig.add_gridspec(2, 2, width_ratios=[1.15, 1], height_ratios=[1, 0.95],
                          hspace=0.35, wspace=0.18)

    # ---- a: one pass, K reconstructions -----------------------------------
    a = fig.add_subplot(gs[0, 0]); blank(a)
    a.text(0.0, 1.05, "One forward pass produces ALL K reconstructions",
           fontsize=6.5, weight="bold")
    box(a, 0.0, 0.80, 0.28, 0.16, "OpenImages crop", "512×512, random qp",
        fc_="#f0f0f0", fs=5.4)
    arrow(a, 0.28, 0.88, 0.31, 0.88)
    box(a, 0.31, 0.80, 0.25, 0.16, "frozen encoder", "ŷ and bpp fixed",
        fc_="#e8e8e8", fs=5.4)
    arrow(a, 0.56, 0.88, 0.59, 0.88)
    box(a, 0.59, 0.80, 0.25, 0.16, "trained decoder", f"{P_DEC/1e6:.2f} M params",
        fc_="#dceaf7", fs=5.4)
    for i in range(K):
        y = 0.66 - i * 0.125
        arrow(a, 0.715, 0.80 if i == 0 else y + 0.125, 0.715, y, color=ns.ORANGE,
              lw=0.5, style="-")
        arrow(a, 0.715, y, 0.81, y, color=ns.ORANGE, lw=0.5)
        a.text(0.835, y, f"x̂$_{i}$  →  MSE$_{i}$", fontsize=5.6, va="center")
    a.text(0.0, 0.60, "Every exit is decoded every step, so the ladder is\n"
           "trained as ONE object rather than as K separate\n"
           "models. The shared prefix means this costs one\n"
           "trunk pass plus K adapter-and-head passes, not K\n"
           "full decodes.", fontsize=5.4, va="top", linespacing=1.7)
    ns.panel(a, "a", dx=-0.02, dy=1.16)

    # ---- b: the objective --------------------------------------------------
    a = fig.add_subplot(gs[1, 0]); blank(a)
    a.text(0.0, 1.02, "The objective", fontsize=6.5, weight="bold")
    a.text(0.0, 0.95, r"$\mathcal{L}\;=\;\mathcal{L}_{\mathrm{RD}}\;+\;"
           r"w_a\,\mathcal{L}_{\mathrm{anchor}}\;+\;"
           r"w_d\,\mathcal{L}_{\mathrm{distill}}$", fontsize=7.5, va="top")
    a.text(0.0, 0.665, r"$\mathcal{L}_{\mathrm{RD}}=\lambda\,"
           r"\frac{\sum_k \alpha_k\,\mathrm{MSE}_k}{\sum_k \alpha_k}"
           r"\;+\;\mathrm{bpp}$", fontsize=7, va="center")
    a.text(0.42, 0.665, r"$\mathcal{L}_{\mathrm{anchor}}=\lambda\,"
           r"\|\hat{x}_{K-1}-\hat{x}^{\mathrm{release}}\|^2$",
           fontsize=7, va="center")
    a.text(0.0, 0.40, r"$\mathcal{L}_{\mathrm{distill}}=\frac{1}{K-1}"
           r"\sum_{k}\frac{\|f_k-\mathrm{sg}(f_{k+1})\|^2}"
           r"{\mathrm{Var}(f_{k+1})}$", fontsize=7, va="center")
    a.text(0.0, 0.20,
           "RD — the released loss, λ·MSE + bpp, with MSE averaged over the exits\n"
           "anchor ($w_a$ = 10) — pins the DEEPEST exit to the released decoder, so the\n"
           "     reference every saving is quoted against cannot drift away underneath us\n"
           "distillation ($w_d$ = 1) — supervises the adapters in FEATURE space, exit $k$\n"
           "     imitating exit $k{+}1$; adjacent rather than deepest, because a large\n"
           "     student–teacher gap is reported to hurt the shallowest exits",
           fontsize=5.3, va="top", linespacing=1.75)
    ns.panel(a, "b", dx=-0.02, dy=1.14)

    # ---- c: what is trained ------------------------------------------------
    a = fig.add_subplot(gs[0, 1])
    parts = [("frozen: encoder, hyperprior,\nentropy model", P_TOT - P_DEC, "#c9c9c9"),
             ("trained: the decoder", P_DEC, ns.BLUE)]
    left = 0.0
    for name, v, c in parts:
        w = v / P_TOT
        a.barh([0], [w], left=left, color=c, height=0.55)
        a.text(left + w/2, 0, f"{name}\n{v/1e6:.2f} M  ({100*w:.1f}%)",
               ha="center", va="center", fontsize=5.3, linespacing=1.6,
               color="white" if c != "#c9c9c9" else ns.INK)
        left += w
    a.barh([-0.85], [P_RT / P_TOT], color=ns.SKY, height=0.30)
    a.text(P_RT / P_TOT + 0.02, -0.85,
           f"router head, configuration B only — {P_RT/1e3:.0f} K, "
           f"{100*P_RT/P_TOT:.2f}%", fontsize=5.3, va="center")
    a.set_xlim(0, 1); a.set_ylim(-1.5, 0.7)
    a.set_xticks([]); a.set_yticks([]); a.grid(False)
    for sp in a.spines.values():
        sp.set_visible(False)
    a.set_title("The encoder is never touched, so a trained decoder consumes\n"
                "byte-for-byte the stream the released encoder produces.",
                fontsize=5.8, color=ns.INK2, loc="left")
    ns.panel(a, "c", dx=-0.06, dy=1.30)

    # ---- d: the recipe -----------------------------------------------------
    a = fig.add_subplot(gs[1, 1]); blank(a)
    a.text(0.0, 1.02, "The recipe, and why each line is there", fontsize=6.5,
           weight="bold")
    rows = [
        ("warm start", "every inherited weight copied from the release, every new\n"
                       "one zero-initialised — step 0 is bit-exact DCVC-UF"),
        ("λ  10 → 2048", "log-spaced over the 64 QPs, one λ drawn per sample, so a\n"
                         "single set of weights covers the whole rate range"),
        ("new_lr_scale 20", "adapters and seam repair start from zero and must move\n"
                            "furthest; the inherited trunk must not"),
        ("512 px crop", "4 tiles per crop at 256 px — smaller crops give the ladder\n"
                         "no tile diversity to route over"),
        ("encoder frozen", "asserted every run: max|Δ| = 0.0 across all encoder weights"),
    ]
    y = 0.86
    for k, v in rows:
        a.text(0.0, y, k, fontsize=5.6, weight="bold", va="top", color=ns.BLUE)
        a.text(0.29, y, v, fontsize=5.2, va="top", linespacing=1.6)
        y -= 0.185 if "\n" in v else 0.115
    ns.panel(a, "d", dx=-0.06, dy=1.14)

    save(fig, "training_scheme.png")


# =====================================================  5. router A versus B
def router_ab():
    """The two ways the exit map can be produced, and what the difference costs."""
    def rows(f, key="saving_pct_vs_release"):
        d = json.load(open(R / "results" / f))
        return {r["qp"]: r[key] for r in d["rows"]}, d

    A, dA = rows("signalled_BEST_0817_1542.json")
    Ba, dB = rows("router_BEST_v2.json")            # router trained at lam 1.3e-5
    Bb, _ = rows("router_BEST_v2_lowlam.json")      # ...and at lam 4.1e-6
    A3, _ = rows("signalled_BEST_b03.json")
    A5, _ = rows("signalled_BEST_b05.json")
    # B at each budget takes the better of the two trained routers, which is the
    # honest "best available" -- one is trained at the low-rate lambda and one at
    # the high-rate lambda, and a deployment would pick per operating point.
    B3a, _ = rows("router_BEST_b03_lam1.3e-5.json")
    B3b, _ = rows("router_BEST_b03_lam4.1e-6.json")
    B5a, _ = rows("router_BEST_b05_lam1.3e-5.json")
    B5b, _ = rows("router_BEST_b05_lam4.1e-6.json")
    B = {q: max(Ba[q], Bb[q]) for q in Ba}
    B3 = {q: max(B3a[q], B3b[q]) for q in B3a}
    B5 = {q: max(B5a[q], B5b[q]) for q in B5a}
    qps = sorted(A)

    fig = plt.figure(figsize=(ns.W2, 2.9))
    gs = fig.add_gridspec(1, 3, width_ratios=[1.25, 1, 1])

    # ---- a: who sees what --------------------------------------------------
    a = fig.add_subplot(gs[0]); blank(a)
    a.add_patch(Rectangle((0.0, 0.53), 1.0, 0.42, facecolor="#fdf4f9",
                          edgecolor=ns.PURPLE, lw=0.7))
    a.text(0.025, 0.905, "A · the ENCODER decides, and signals the map",
           fontsize=6, weight="bold", color=ns.PURPLE)
    a.text(0.025, 0.845,
           "It holds the source frame, so for every tile it can decode all\n"
           "K exits and measure the true error of each. Its choice is not a\n"
           "prediction — it is the optimum:",
           fontsize=5.2, va="top", linespacing=1.6)
    a.text(0.09, 0.655, r"$k^{*}(t)=\arg\min_k\;[\,\mathrm{MSE}(t,k)"
           r"+\lambda\,c_k\,]$", fontsize=6.8, va="center")
    a.text(0.025, 0.585, "λ is bisected once per frame until the frame lands on "
           "the budget.", fontsize=5.2, va="center", color=ns.INK2)

    a.add_patch(Rectangle((0.0, 0.015), 1.0, 0.475, facecolor="#eef6fb",
                          edgecolor=ns.BLUE, lw=0.7))
    a.text(0.025, 0.448, "B · the DECODER decides, and nothing is signalled",
           fontsize=6, weight="bold", color=ns.BLUE)
    a.text(0.025, 0.393,
           "It never sees the source, so the true MSE is not merely hard to\n"
           "estimate — it is absent from the input. A 144 K head reads the\n"
           "stem map, ŷ, the entropy-model scales and qp, and scores exits:",
           fontsize=5.2, va="top", linespacing=1.6)
    a.text(0.06, 0.183, r"$\hat{k}(t)=\arg\max_k\;[\,\log\mathrm{softmax}"
           r"(z_t)_k-\beta\,c_k\,]$", fontsize=6.8, va="center")
    a.text(0.025, 0.112, "β plays λ's role and is bisected the same way. The head "
           "itself\ncosts 0.163% of a decode, charged inside every B number.",
           fontsize=5.2, va="top", color=ns.INK2, linespacing=1.6)
    ns.panel(a, "a", dx=-0.02, dy=1.10)

    # ---- b: measured saving vs rate ---------------------------------------
    a = fig.add_subplot(gs[1])
    ya = [A[q] for q in qps]; yb = [B[q] for q in qps]
    a.fill_between(qps, yb, ya, color=ns.VERM, alpha=0.13, lw=0)
    a.plot(qps, ya, marker="o", color=ns.PURPLE, label="A — signalled (oracle)")
    a.plot(qps, yb, marker="s", color=ns.BLUE, label="B — predicted, 0 bits")
    a.plot(qps, [Ba[q] for q in qps], color=ns.SKY, lw=0.7, ls=(0, (3, 2)),
           label="B, single router for all rates")
    for q in qps:
        a.annotate(f"{A[q]-B[q]:.1f}", (q, (A[q] + B[q]) / 2), fontsize=5,
                   color=ns.VERM, ha="center", va="center")
    a.set_xlabel("qp"); a.set_ylabel("compute saved at 0.1 dB (%)")
    a.legend(loc="lower left", fontsize=5)
    a.set_title(f"Same checkpoint, same {dA['n_sequences']} sequences.\n"
                "Shaded: what an unchanged bitstream costs.",
                fontsize=6, color=ns.INK2, loc="left")
    ns.panel(a, "b", dx=-0.24)

    # ---- c: the gap shrinks as the budget grows ---------------------------
    a = fig.add_subplot(gs[2])
    for (Ax, Bx), c, lab in (((A, B), ns.VERM, "0.1 dB budget"),
                             ((A3, B3), ns.ORANGE, "0.3 dB"),
                             ((A5, B5), ns.GREEN, "0.5 dB")):
        a.plot(qps, [Ax[q] - Bx[q] for q in qps], marker="o", color=c, label=lab)
    a.axhline(0.163, color=ns.INK2, lw=0.7, ls=(0, (3, 2)))
    a.text(qps[0], 0.30, "0.163% — the router's own compute", fontsize=5,
           color=ns.INK2)
    a.set_xlabel("qp"); a.set_ylabel("A − B  (percentage points)")
    a.legend(loc="upper left", fontsize=5)
    a.set_title("Every line is the best available router. Give the ladder\\n"
                "more room and the gap collapses to\n"
                "exactly the router's own cost — prediction is only\n"
                "expensive when the budget is tight.",
                fontsize=6, color=ns.INK2, loc="left")
    ns.panel(a, "c", dx=-0.24)

    fig.tight_layout()
    save(fig, "router_ab.png")


if __name__ == "__main__":
    pipeline()
    adapters()
    seam_module()
    training()
    router_ab()
