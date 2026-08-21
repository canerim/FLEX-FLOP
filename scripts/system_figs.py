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
from savings import sv, pick
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
    # 0.012/-0.028 left the two lines 17% overlapped in training_scheme, where
    # the axes is short enough that four hundredths of it is under a line of
    # 6 pt type. The pair is set from the font size instead, so it stays clear
    # whatever shape the diagram is.
    gap = 0.021 * (fs / 6.0)
    ax.text(x + w / 2, y + h / 2 + (gap if sub else 0), label, ha="center",
            va="center", fontsize=fs, color=tc)
    if sub:
        ax.text(x + w / 2, y + h / 2 - 1.9 * gap, sub, ha="center",
                va="center", fontsize=fs - 1.2, color=ns.INK2)


def arrow(ax, x0, y0, x1, y1, color=ns.INK2, lw=0.7, style="-|>", ls="-"):
    ax.add_patch(FancyArrowPatch((x0, y0), (x1, y1), arrowstyle=style,
                                 mutation_scale=6, linewidth=lw, color=color,
                                 linestyle=ls, shrinkA=0, shrinkB=0))


def blank(ax):
    ax.set_xlim(0, 1); ax.set_ylim(0, 1); ax.axis("off"); ax.grid(False)


def save(fig, name):
    # Both directories: build_pdf reads paper/figures, and a figure written
    # only to docs/figures goes stale there in silence.
    for d in (FIG, R / "paper" / "figures"):
        d.mkdir(parents=True, exist_ok=True)
        fig.savefig(d / name, dpi=500, bbox_inches="tight", pad_inches=0.02,
                    facecolor="white")
    plt.close(fig)
    print(f"  -> {name}")


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
    """Figure 2: what an exit adapter is made of, and which exit wears which.

    Two facts have to land and both of them are geometric, so both are drawn
    rather than written. An exit adapter is not a new operator: it is the tail
    of a DepthConvBlock at the size the block's own tail is, which is why panel
    a right-aligns the three profiles and why they coincide exactly. And the
    only convolution anywhere in the block that has spatial reach is the 3x3
    depthwise, which neither adapter contains, which is why an adapter cannot
    add a tile-boundary penalty of its own.

    Every length here is read from results/adapter_cost.json, counted with hooks
    off the real modules by scripts/adapter_cost.py, because a drawing is where
    a stale constant survives longest. The version this replaces printed the
    superseded 2C^2 for the FFN adapter, which flexuf/cost.py had already
    recorded as wrong once it was counted at 5C^2.

    Drawn at Nature's single-column width because the paper places the figure at
    column width. Authored double-column it printed at 0.485 scale, which put
    all 678 characters of its labelling below the 5 pt floor.
    """
    d = json.load(open(R / "results/adapter_cost.json"))
    K_, j_ = d["config"]["num_exits"], d["config"]["split_depth"]

    fig, ax = plt.subplots(1, 2, figsize=(ns.W1, 1.72),
                           gridspec_kw={"width_ratios": [1.32, 1]})

    # ---- a: an adapter is the tail of the block, at true relative size -----
    a = ax[0]
    lanes = [("block", d["block"]["profile"], "#c4c4c4"),
             ("FFN", d["adapters"]["ffn"]["profile"], ns.ORANGE),
             ("1×1", d["adapters"]["conv1x1"]["profile"], ns.BLUE)]
    widest = max(r["out_channels"] for r in d["block"]["profile"])
    for i, (_, prof, colour) in enumerate(lanes):
        # Right-aligned deliberately. An adapter reproduces the LAST convolutions
        # of the block, so ending the lanes together lands each adapter exactly
        # under the segments it stands in for, and the reader sees the match
        # instead of being told about it.
        x = 1.0 - sum(r["share_of_block"] for r in prof)
        for r in prof:
            w = r["share_of_block"]
            h = 0.74 * r["out_channels"] / widest
            if r["depthwise"]:
                # This one is 0.33% of the block, so at true width it is a
                # hairline and is drawn as one. Its absence from both adapter
                # lanes is the entire seam argument.
                a.plot([x + w / 2] * 2, [-i, -i + h], color=ns.VERM, lw=1.1,
                       solid_capstyle="butt", zorder=3)
            else:
                a.add_patch(Rectangle((x, -i), w, h, facecolor=colour,
                                      edgecolor="white", lw=0.35))
            x += w
    a.plot([], [], color=ns.VERM, lw=1.1, label="3×3")
    a.legend(loc="lower left", fontsize=5.5, handlelength=0.9,
             borderpad=0.0, handletextpad=0.4)
    a.set_xlim(-0.02, 1.02); a.set_ylim(-2.42, 0.86)
    a.set_xticks([0, 0.5, 1.0]); a.set_xticklabels(["0", "0.5", "1"])
    a.set_yticks([-i + 0.09 for i in range(len(lanes))])
    a.set_yticklabels([n for n, _, _ in lanes])
    a.tick_params(axis="y", length=0)
    a.spines["left"].set_visible(False)
    a.set_xlabel("share of one DepthConvBlock")
    a.grid(False)
    ns.panel(a, "a", dx=-0.24, dy=1.14)

    # ---- b: capacity against the gap it stands in for ----------------------
    b = ax[1]
    ks = [e["exit"] for e in d["exits"]]
    face = {"ffn": ns.ORANGE, "conv1x1": ns.BLUE, None: "white"}
    # Exits shallower than the split depth are unreachable: forward() clamps the
    # map at j, so no tile ever leaves through them.
    b.axvspan(-0.55, j_ - 0.5, color="#f2f2f2", zorder=0)
    # The rule the ladder applies: four skipped blocks or more gets the FFN.
    b.axhline(4, color=ns.INK2, lw=0.6, ls=(0, (1, 2)), zorder=1)
    b.plot(ks, [e["blocks_skipped"] for e in d["exits"]], color=ns.INK2,
           lw=0.7, zorder=2)
    for e in d["exits"]:
        # An opaque white disc first, so that a faded marker fades against the
        # page rather than against the line running underneath it, which turned
        # the two unreachable exits brown instead of pale.
        b.plot(e["exit"], e["blocks_skipped"], marker="o", ms=4.2, lw=0,
               mfc="white", mec="none", zorder=2.5)
        b.plot(e["exit"], e["blocks_skipped"], marker="o", ms=4.2,
               mfc=face[e["adapter_kind"]], mec=ns.INK2, mew=0.5, zorder=3,
               alpha=1.0 if e["reachable"] else 0.3)
    for kind, lab in (("ffn", "FFN"), ("conv1x1", "1×1"), (None, "none")):
        b.plot([], [], marker="o", ms=4.2, lw=0, mfc=face[kind], mec=ns.INK2,
               mew=0.5, label=lab)
    b.legend(loc="upper right", fontsize=5.5, handlelength=0.8,
             borderpad=0.0, handletextpad=0.2, labelspacing=0.25)
    b.set_xticks(ks); b.set_xlim(-0.55, K_ - 0.45); b.set_ylim(-0.9, 11.6)
    # Blocks come whole, so the ticks are the numbers of blocks that exist.
    b.set_yticks(range(0, 11, 2))
    b.set_xlabel("exit $k$"); b.set_ylabel("blocks skipped")
    ns.panel(b, "b", dx=-0.30, dy=1.14)

    fig.tight_layout(w_pad=1.8)
    save(fig, "adapters.png")


# ===================================================  3. the seam-repair module
def seam_module():
    """What the gate learned, and where the module changes the error.

    Both panels are a quantity against distance from the tile boundary, because
    that is the only axis the module's claim is about: it is gated by position
    within a tile, so it is supposed to act on the boundary ring and switch
    itself off inside. The previous version of this figure spent one panel
    redrawing the module as a stack of labelled boxes, which is the equation two
    paragraphs below it, and one panel on the gate as a 32x32 image, in which a
    one-cell ring reads as a hairline on a black square.

    The gate is read from a PINNED checkpoint. It used to be read from
    `runs/BEST/ckpt_eval.pth.tar`, which the watcher rewrites every epoch, so the
    numbers the caption quoted could not be reproduced a day later.
    """
    import seam_spatial as ss

    GCK = R / "runs/BEST/ckpt_epo0.pth.tar"
    gck = torch.load(GCK, map_location="cpu", weights_only=False)
    g = ss.gate_profile(gck)
    g["checkpoint"], g["epoch"] = str(GCK), int(gck.get("epoch", -1))
    e = json.loads((R / "results/seam_spatial_published.json").read_text())

    fig, ax = plt.subplots(1, 2, figsize=(ns.W2, 2.4))

    # ---- a: what the gate learned ------------------------------------------
    a = ax[0]
    x = g["distance_px"]
    # The shaded band is the spread across cells at the same distance, which is
    # where the corner peak lives; the ring is not one number.
    a.fill_between(x, g["trained_min"], g["trained_max"], color=ns.BLUE,
                   alpha=0.22, lw=0)
    a.plot(x, g["trained_mean"], marker="o", ms=2.5, color=ns.BLUE,
           label="trained")
    a.plot(x, g["init"], color=ns.INK2, lw=0.8, ls=(0, (3, 2)),
           label="initialised")
    a.set_xlim(0, g["tile_px"] / 2)
    a.set_ylim(0, 1.05)
    a.set_xlabel("distance from the tile boundary (px)")
    a.set_ylabel("gate $G$")
    a.legend(loc="upper right")

    # ---- b: where it changes the error -------------------------------------
    a = ax[1]
    share, chg = e["share_pct"], e["change_pct"]
    left = np.concatenate([[0.0], np.cumsum(share)[:-1]])
    # Bar width is the share of pixels, so a bar's AREA is what its band
    # contributes to the frame average. That is the whole argument in one
    # picture: the win is a sliver and the loss is most of the frame.
    a.bar(left, chg, width=share, align="edge", edgecolor="white", lw=0.5,
          color=[ns.BLUE if c < 0 else ns.VERM for c in chg])
    a.axhline(0, color=ns.INK, lw=0.6)
    a.set_xlim(0, 100)
    a.set_ylim(-0.30, 0.09)
    a.set_xticks(left + np.asarray(share) / 2)
    a.set_xticklabels([f"{b[0]}\u2013{b[1]}" for b in e["bands_px"]])
    a.set_xlabel("distance from the tile boundary (px)")
    a.set_ylabel("change in error (%)")
    a.grid(False)
    a.grid(True, axis="y")

    for i, l in enumerate("ab"):
        ns.panel(ax[i], l, dx=-0.22)
    fig.tight_layout(w_pad=1.6)
    save(fig, "seam_module.png")

    # Both panels' provenance in one place, so the next person does not have to
    # work out which checkpoint the picture came from.
    (R / "results/seam_module.json").write_text(json.dumps(
        {"figure": "docs/figures/seam_module.png", "gate": g, "effect": e},
        indent=1))


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
    def rows(*names, key="saving_pct", budget=None):
        """Canonical file, canonical definition.

        First-that-exists preferred router_..._b01_fixed.json, an earlier
        experiment, over the pinned checkpoint's file, and the saving came from
        the arithmetic model rather than the hook count. Both are what
        make_paper_tables fixed for the tables; the figures beside them kept
        the old behaviour.
        """
        d, _ = pick(*names)
        if d is None:
            raise FileNotFoundError(names[0])
        rs = [r for r in d["rows"] if r.get("budget_reachable", True)
              and (budget is None or r.get("budget_db") is None
                   or abs(r["budget_db"] - budget) < 1e-9)]
        return {r["qp"]: sv(r, key) for r in rs if sv(r, key) is not None}, d

    # RECIPE512 on the deployed path, one router trained at a single lambda.
    # The older BEST files are on the full-frame table (flexuf/eval.py) and must
    # not be mixed in.
    sig, dA = rows("signalled_RECIPE512_ctc53.json", key="saving_pct",
                   budget=0.1)
    A, _ = rows("signalled_RECIPE512_ctc53.json", budget=0.1)
    A3, _ = rows("signalled_RECIPE512_ctc53.json", budget=0.3)
    A5, _ = rows("signalled_RECIPE512_ctc53.json", budget=0.5)
    B, dB = rows("router_RECIPE512_b01_PAPER.json",
                 "router_RECIPE512_b01_fixed.json", "router_RECIPE512_b01.json")
    B3, _ = rows("router_RECIPE512_b03_PAPER.json",
                 "router_RECIPE512_b03_fixed.json", "router_RECIPE512_b03.json")
    B5 = {}
    Ba = B
    qps = sorted(q for q in A if q in B)

    fig = plt.figure(figsize=(ns.W2, 2.7))
    gs = fig.add_gridspec(1, 3, width_ratios=[1.35, 1, 1])

    # ---- a: who sees what --------------------------------------------------
    a = fig.add_subplot(gs[0]); blank(a)

    def chip(x, y, w, h, txt, fc, ec, fs=5.4, bold=False):
        a.add_patch(FancyBboxPatch((x, y), w, h, boxstyle="round,pad=0.006",
                                   facecolor=fc, edgecolor=ec, lw=0.7))
        a.text(x + w / 2, y + h / 2, txt, ha="center", va="center", fontsize=fs,
               color=ec if bold else ns.INK, weight="bold" if bold else "normal")

    def arrow(x0, y, x1, c=ns.INK2):
        a.annotate("", (x1, y), (x0, y), arrowprops=dict(
            arrowstyle="-|>", lw=0.7, color=c, mutation_scale=6))

    for y0, tag, col, fc in ((0.56, "A", ns.PURPLE, "#fdf4f9"),
                             (0.06, "B", ns.BLUE, "#eef6fb")):
        a.add_patch(Rectangle((0.0, y0), 1.0, 0.38, facecolor=fc,
                              edgecolor=col, lw=0.7))
        a.text(0.02, y0 + 0.335, tag, fontsize=7, weight="bold", color=col)

    # A row
    y = 0.80
    chip(0.07, y - 0.045, 0.15, 0.09, "source", "#ffffff", ns.PURPLE)
    chip(0.07, y - 0.155, 0.15, 0.09, "$\\hat{y}$", "#ffffff", ns.PURPLE)
    arrow(0.23, y - 0.055, 0.31, ns.PURPLE)
    chip(0.32, y - 0.155, 0.26, 0.20, "decode\nall $K$", "#ffffff", ns.PURPLE)
    arrow(0.59, y - 0.055, 0.66, ns.PURPLE)
    # The paper prices the map at 89 bits a frame over 40 tiles, entropy
    # coded. "3 b/tile" was the raw index width and read as a different
    # number from the one every table gives.
    chip(0.67, y - 0.155, 0.28, 0.20, "map\n+89 b/frame", "#ffffff",
         ns.PURPLE)
    a.text(0.5, 0.585, r"$k^{*}=\arg\min_k\,[\,\mathrm{MSE}_k+\lambda c_k]$",
           fontsize=6.2, ha="center", va="center", color=ns.PURPLE)

    # B row
    y = 0.30
    chip(0.07, y - 0.045, 0.15, 0.09, "$\\hat{y}$", "#ffffff", ns.BLUE)
    chip(0.07, y - 0.155, 0.15, 0.09, "qp, $\\sigma$", "#ffffff", ns.BLUE)
    arrow(0.23, y - 0.055, 0.31, ns.BLUE)
    chip(0.32, y - 0.155, 0.26, 0.20, "head\n144 K", "#ffffff", ns.BLUE)
    arrow(0.59, y - 0.055, 0.66, ns.BLUE)
    chip(0.67, y - 0.155, 0.28, 0.20, "map\n+0 bits", "#ffffff", ns.BLUE)
    a.text(0.5, 0.085, r"$\hat{k}=\arg\max_k\,[\,\log p_k-\beta c_k]$",
           fontsize=6.2, ha="center", va="center", color=ns.BLUE)
    ns.panel(a, "a", dx=-0.02, dy=1.06)

    # ---- b: measured saving vs rate ---------------------------------------
    a = fig.add_subplot(gs[1])
    ya = [A[q] for q in qps]; yb = [B[q] for q in qps]
    a.fill_between(qps, yb, ya, color=ns.VERM, alpha=0.13, lw=0)
    a.plot(qps, ya, marker="o", color=ns.PURPLE, label="A  signalled")
    a.plot(qps, yb, marker="s", color=ns.BLUE, label="B  predicted")
    for q in qps:
        a.annotate(f"{A[q]-B[q]:.1f}", (q, (A[q] + B[q]) / 2), fontsize=5,
                   color=ns.VERM, ha="center", va="center")
    a.set_xlabel("qp"); a.set_ylabel("saved at 0.1 dB (%)")
    a.legend(loc="lower left", fontsize=5.4)
    ns.panel(a, "b", dx=-0.26)

    # ---- c: the gap shrinks as the budget grows ---------------------------
    a = fig.add_subplot(gs[2])
    for (Ax, Bx), c, lab in (((A, B), ns.VERM, "0.1 dB"),
                             ((A3, B3), ns.ORANGE, "0.3 dB")):
        qq = [q for q in qps if q in Ax and q in Bx]
        a.plot(qq, [Ax[q] - Bx[q] for q in qq], marker="o", color=c, label=lab)
    a.axhline(0.163, color=ns.INK2, lw=0.7, ls=(0, (3, 2)))
    a.text(qps[0], 0.55, "router cost", fontsize=5, color=ns.INK2)
    a.set_xlabel("qp"); a.set_ylabel("A $-$ B  (points)")
    a.legend(loc="upper left", fontsize=5.4)
    ns.panel(a, "c", dx=-0.26)

    fig.tight_layout()
    save(fig, "router_ab.png")


if __name__ == "__main__":
    # Named on the command line, a single figure is redrawn on its own. Four of
    # the five read runs/BEST/ckpt_eval.pth.tar, which a watcher overwrites
    # every epoch, so redrawing all five to fix one of them silently rewrites
    # the other four from a different checkpoint than the one they were
    # published from.
    ALL = {"pipeline": pipeline, "adapters": adapters,
           "seam_module": seam_module, "training": training,
           "router_ab": router_ab}
    for name in (sys.argv[1:] or list(ALL)):
        ALL[name]()
