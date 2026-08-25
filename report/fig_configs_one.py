"""A, B and C on one sheet, on one grammar, with A explained properly.

The three configurations share a decoder -- the same pinned checkpoint, the
same ladder, the same Lagrangian -- and differ in exactly one thing: who
decides the exit map, and what that decision costs to transmit. Drawn on
three separate sheets that difference is asserted; drawn on one, with the
same boxes in the same columns, it is visible.

A gets the room it needs. It is the configuration whose allocation is not a
prediction at all: the encoder holds the source, so it can decode the tile at
every exit and tabulate the error instead of guessing it, and the map it
sends is then the exact minimiser of a separable objective rather than an
approximation to one. Every step of that is drawn, including the two places
the number is deliberately reported against ourselves -- the map's entropy is
counted over six symbols when a codec would ship four, and the saving is the
wall-measured one rather than the additive model, which runs 0.58-0.77 points
optimistic.

B removes the map from the bitstream and puts the decision in the decoder,
where the only inputs are things the decode already holds. C keeps both and
spends a few bits where the predictor is worst, which works because regret is
extremely concentrated: at qp0 the worst tenth of tiles carries 99.8% of it.

Colour carries one meaning. Encoder-side work is warm, decoder-side work is
cool, anything that reaches the bitstream is the accent, and nothing is
coloured for decoration.
"""
import json, sys
from pathlib import Path
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch, FancyArrowPatch, Rectangle
sys.path.insert(0, str(Path(__file__).resolve().parent))
import style as S

S.setup()
HERE = Path(__file__).resolve().parent
OUT = HERE / "fig"
UF = Path.home() / "FLEX-UF"
QPS = [0, 16, 32, 48, 63]
PIX = 1920 * 1080

ENC, ENC_F = S.VERM, "#fdece2"
DEC, DEC_F = S.BLUE, "#e7f1f9"
SIG, SIG_F = S.GREEN, "#e6f5ee"
NEU, NEU_F = S.MUTED, "#f5f5f5"

W, H = 14.4, 26.0                      # drawing units


def box(ax, x, y, w, h, label, fc, ec, fs=5.8, lw=0.75, tc=None, z=3):
    ax.add_patch(FancyBboxPatch(
        (x, y), w, h, boxstyle="round,pad=0.006,rounding_size=0.10",
        facecolor=fc, edgecolor=ec, linewidth=lw, zorder=z))
    ax.text(x + w / 2, y + h / 2, label, ha="center", va="center", fontsize=fs,
            color=tc or S.INK, zorder=z + 1, linespacing=1.32)


def arrow(ax, p, q, color=NEU, lw=0.75, ls="-", z=2, ms=5.5):
    ax.add_patch(FancyArrowPatch(p, q, arrowstyle="-|>", mutation_scale=ms,
                                 color=color, lw=lw, linestyle=ls,
                                 shrinkA=0, shrinkB=0, zorder=z))


def elbow(ax, pts, color=NEU, lw=0.9, z=4):
    """A right-angled route, arrowhead on the last leg.

    Needed because what the encoder produces has to enter the BITSTREAM, and
    the bitstream is upstream of the boxes the search sits under. A straight
    vertical arrow from the coding step lands on a decoder box instead, which
    is the opposite of what the configuration does.
    """
    for a, b in zip(pts[:-1], pts[1:-1]):
        ax.plot([a[0], b[0]], [a[1], b[1]], color=color, lw=lw,
                solid_capstyle="round", zorder=z)
    arrow(ax, pts[-2], pts[-1], color, lw=lw, z=z, ms=6.0)


def pipeline(ax, y, channel_text, channel_accent=None):
    """The row every configuration shares, in the same columns every time."""
    h = 0.80
    box(ax, 0.05, y, 1.35, h, "encoder\n(frozen)", ENC_F, ENC)
    arrow(ax, (1.40, y + h / 2), (1.72, y + h / 2))
    ax.add_patch(Rectangle((1.72, y - 0.02), 4.10, h + 0.04,
                           facecolor="#f0f0f0", edgecolor="none", zorder=1))
    ax.text(1.86, y + h - 0.22, "bitstream", fontsize=5.6, color=S.INK2,
            va="center", ha="left")
    ax.text(1.86, y + 0.24, channel_text, fontsize=5.7, va="center",
            ha="left", color=channel_accent or S.INK2,
            fontweight="bold" if channel_accent else "normal")
    arrow(ax, (5.82, y + h / 2), (6.14, y + h / 2))
    box(ax, 6.14, y, 1.42, h, "parse,\ndilate $D(x)$", DEC_F, DEC)
    arrow(ax, (7.56, y + h / 2), (7.88, y + h / 2))
    box(ax, 7.88, y, 1.46, h, "trunk, only\nwhere $D\\geq b$", DEC_F, DEC)
    arrow(ax, (9.34, y + h / 2), (9.66, y + h / 2))
    box(ax, 9.66, y, 1.34, h, "adapter,\nstitch, head", DEC_F, DEC)
    arrow(ax, (11.00, y + h / 2), (11.34, y + h / 2))
    ax.text(11.42, y + h / 2, "image", fontsize=5.8, va="center", color=S.INK)
    return h


def band_title(ax, y, tag, title, sub, colour):
    ax.text(0.05, y, tag, fontsize=9.4, fontweight="bold", color=colour,
            va="top", ha="left")
    ax.text(0.98, y, title, fontsize=9.4, fontweight="bold", color=S.INK,
            va="top", ha="left")
    ax.text(0.05, y - 0.62, sub, fontsize=6.2, color=S.INK2, va="top",
            ha="left")


def rule(ax, y):
    ax.plot([0.05, W - 0.15], [y, y], color="#dddddd", lw=0.5, zorder=0)


def inset(fig, x0, y0, w, h):
    return fig.add_axes([x0 / W, y0 / H, w / W, h / H])


# --------------------------------------------------------------------------
# the measurements the diagram is annotated with
# --------------------------------------------------------------------------
def data():
    rel = {r["qp"]: r for r in json.loads(
        (UF / "results/rd_absolute_PAPER.json").read_text())["rows"]}
    A = json.loads((UF / "results/signalled_RECIPE512_ctc53.json").read_text())
    B = json.loads((UF / "results/router_RECIPE512_b01_e4head.json").read_text())
    C = json.loads(
        (UF / "results/hybrid_RECIPE512_b01_e4head.json").read_text())
    a = {r["qp"]: r for r in A["rows"]
         if r.get("budget_reachable") and abs(r.get("budget_db", 0) - 0.1) < 1e-9}
    b = {r["qp"]: r for r in B["rows"]}
    c = {}
    for r in C["rows"]:
        c.setdefault(r["qp"], []).append(r)
    for q in c:
        c[q].sort(key=lambda r: r["rho"])
    ev = B["router2_meta"]["eval"]
    return dict(rel=rel, a=a, b=b, c=c, meta=B, ev=ev,
                heldout=B["router2_meta"]["heldout_agree"])


def sv(r):
    return r.get("saving_pct_measured", r.get("saving_pct_vs_release"))


# --------------------------------------------------------------------------
# A
# --------------------------------------------------------------------------
def band_A(fig, ax, D, top):
    a = D["a"]
    band_title(ax, top, "A", "signalled",
               "the encoder holds the source, so it does not predict the "
               "allocation -- it measures it, minimises exactly, and sends "
               "the answer", ENC)
    y = top - 2.00
    pipeline(ax, y, "latent  +  exit map,  64-91 bits per frame", SIG)

    ys, hs = top - 4.20, 1.30
    steps = [
        ("1", "decode the tile at\nEVERY exit $k$", ENC_F, ENC),
        ("2", "tabulate the exact\nerror $m(t,k)$", ENC_F, ENC),
        ("3", "$k^{*}(t)=\\arg\\min_{k\\geq j}$\n$[\\,m(t,k)+\\lambda c_k\\,]$",
         ENC_F, ENC),
        ("4", "bisect $\\lambda$ on the\nTRUE frame dB", ENC_F, ENC),
        ("5", "entropy-code:\n$H\\!\\cdot\\!N+48$ bits", SIG_F, SIG),
    ]
    wS, gap = 1.92, 0.315
    for i, (n, lab, fc, ec) in enumerate(steps):
        x = 0.05 + i * (wS + gap)
        box(ax, x, ys, wS, hs, lab, fc, ec, fs=5.6)
        ax.add_patch(FancyBboxPatch(
            (x + 0.04, ys + hs - 0.40), 0.36, 0.34,
            boxstyle="round,pad=0.004,rounding_size=0.08",
            facecolor=ec, edgecolor="none", zorder=5))
        ax.text(x + 0.22, ys + hs - 0.23, n, ha="center", va="center",
                fontsize=5.4, color="white", fontweight="bold", zorder=6)
        if i < len(steps) - 1:
            arrow(ax, (x + wS, ys + hs / 2), (x + wS + gap, ys + hs / 2),
                  ENC if i < 3 else SIG)
    # the map is the only thing that reaches the channel
    x5 = 0.05 + 4 * (wS + gap) + wS / 2
    elbow(ax, [(x5, ys + hs), (x5, y - 0.48), (4.55, y - 0.48),
               (4.55, y - 0.03)], SIG, lw=1.0)
    ax.text(4.72, y - 0.30, "the map, into the file", fontsize=5.5, color=SIG,
            fontweight="bold", va="center")

    n1 = ("Steps 1-3 are what makes A exact and not merely good: the cost "
          "$c_k=(k{+}1)\\cdot 2$ blocks is closed form and the objective is "
          "separable over tiles,\nso the per-tile argmin IS the constrained "
          "optimum -- there is no search left to improve. Step 4 bisects on "
          "the decoder's own output, not on the table.")
    n2 = ("Two places the number is reported against us. The entropy in step "
          "5 counts six symbols where a codec would ship four (the clamp "
          "makes $e_0,e_1,e_2$\nidentical), inflating the map by $\\leq$7 "
          "bits; and the saving below is wall-measured, 0.58-0.77 points "
          "under what the additive MAC model claims.")
    ax.text(0.05, ys - 0.42, n1, fontsize=5.3, color=ENC, va="top",
            linespacing=1.45)
    ax.text(0.05, ys - 1.30, n2, fontsize=5.3, color=S.INK2, va="top",
            linespacing=1.45)
    res = "   ".join(f"qp{q} {sv(a[q]):.1f}%" for q in QPS)
    ax.text(0.05, ys - 2.18, "measured saving at 0.1 dB, 53 CTC frames:  "
            + res, fontsize=5.9, color=S.INK, va="top", fontweight="bold")

    # inset: what the map costs against what the frame costs
    ia = inset(fig, 11.52, ys + 0.30, 2.62, 2.05)
    fb = [D["rel"][q]["bpp"] * PIX for q in QPS]
    mb = [a[q]["map_bits"] for q in QPS]
    xs = np.arange(len(QPS))
    ia.bar(xs - 0.19, fb, 0.36, color="#d9d9d9", edgecolor="none")
    ia.bar(xs + 0.19, mb, 0.36, color=SIG, edgecolor="none")
    ia.set_yscale("log"); ia.set_ylim(20, 4e6)
    ia.set_xticks(xs); ia.set_xticklabels([f"{q}" for q in QPS], fontsize=5.2)
    ia.set_xlabel("qp", fontsize=5.4, labelpad=1.0)
    ia.tick_params(labelsize=5.2, length=1.6, pad=1.0)
    ia.set_yticks([1e2, 1e4, 1e6])
    ia.set_yticklabels(["$10^2$", "$10^4$", "$10^6$"], fontsize=5.2)
    ia.set_title("bits per frame", fontsize=5.6, pad=2.0, color=S.INK)
    ia.text(0.02, 0.965, "frame", transform=ia.transAxes, fontsize=5.2,
            color="#9a9a9a", va="top")
    ia.text(0.02, 0.80, "exit map", transform=ia.transAxes, fontsize=5.2,
            color=SIG, va="top", fontweight="bold")
    r = [m / f for m, f in zip(mb, fb)]
    ia.text(0.5, -0.40, f"the map is {min(r)*1e4:.2f}-{max(r)*1e4:.2f}"
            r"$\times 10^{-4}$ of the frame",
            transform=ia.transAxes, fontsize=5.0, ha="center", color=S.INK2)
    return ys - 2.75


# --------------------------------------------------------------------------
# B
# --------------------------------------------------------------------------
def band_B(fig, ax, D, top):
    b, ev = D["b"], D["ev"]
    band_title(ax, top, "B", "bitstream-identical",
               "the map leaves the file entirely; the decoder recovers it "
               "from what the decode already holds, at 0.163% of a decode "
               "and zero added bits", DEC)
    y = top - 2.00
    pipeline(ax, y, "latent only  --  byte for byte the released file")

    ys, hs = top - 4.10, 1.30
    box(ax, 0.05, ys, 2.30, hs,
        "stem feature\n$1{\\times}1$ 384$\\to$48, pooled", DEC_F, DEC, fs=5.5)
    box(ax, 2.62, ys, 2.30, hs,
        "latent + scales\n$1{\\times}1$ 512$\\to$32, pooled", DEC_F, DEC, fs=5.5)
    box(ax, 5.19, ys, 1.00, hs, "qp", DEC_F, DEC, fs=5.8)
    for x in (1.20, 3.77, 5.69):
        arrow(ax, (x, ys), (x, ys - 0.44), DEC)
    ax.plot([1.20, 5.69], [ys - 0.44, ys - 0.44], color=DEC, lw=0.7, zorder=2)
    arrow(ax, (3.45, ys - 0.44), (3.45, ys - 0.86), DEC)
    ax.text(5.86, ys - 0.44, "$d=161$", fontsize=5.5, color=DEC, va="center")
    box(ax, 1.55, ys - 2.20, 3.80, 1.20,
        "MLP  161-256-256-6,  144,024 parameters\n"
        "trained by regret-weighted cross-entropy", DEC_F, DEC, fs=5.5)
    arrow(ax, (5.35, ys - 1.60), (6.55, ys - 1.60), DEC)
    box(ax, 6.55, ys - 2.20, 2.90, 1.20,
        "same Lagrangian as A, on\n$\\hat m$: $\\arg\\min_k[\\hat m+\\lambda c_k]$",
        DEC_F, DEC, fs=5.5)
    elbow(ax, [(8.00, ys - 1.00), (8.00, y - 0.48), (6.85, y - 0.48),
               (6.85, y - 0.03)], DEC, lw=1.0)
    ax.text(8.14, y - 0.34, "$\\hat k(t)$, never transmitted", fontsize=5.5,
            color=DEC, fontweight="bold", va="center")

    n = ("Every input is a tensor the decoder has already produced, so B adds "
         "no parse, no side channel and no dependency on the encoder. It is "
         "the only\nconfiguration that can be dropped in front of a file that "
         "was encoded before this work existed.")
    ax.text(0.05, ys - 2.60, n, fontsize=5.3, color=DEC, va="top",
            linespacing=1.45)
    res = "   ".join(f"qp{q} {sv(b[q]):.1f}%" for q in QPS)
    gap = "   ".join(f"{sv(b[q]) - sv(D['a'][q]):.1f}" for q in QPS)
    ax.text(0.05, ys - 3.35, "measured saving at 0.1 dB:  " + res,
            fontsize=5.9, color=S.INK, va="top", fontweight="bold")
    ax.text(0.05, ys - 3.85, "B $-$ A:  " + gap
            + "  points -- what predicting instead of measuring costs, and it "
              "grows with the rate", fontsize=5.5, color=S.INK2, va="top")

    ib = inset(fig, 11.52, ys - 1.55, 2.62, 1.95)
    vals = [ev["agree"] * 100, ev["constant_best_agree"] * 100]
    ib.bar([0, 1], vals, 0.52, color=[DEC, "#d9d9d9"], edgecolor="none")
    ib.errorbar([0], [vals[0]], yerr=[ev["stderr_across_frames"] * 100],
                fmt="none", ecolor=S.INK, elinewidth=0.6, capsize=1.6,
                capthick=0.6)
    for i, v in enumerate(vals):
        ib.text(i, v + 3.0, f"{v:.1f}", ha="center", fontsize=5.4,
                color=S.INK, fontweight="bold" if i == 0 else "normal")
    ib.axhline(100, color=ENC, lw=0.6, ls=(0, (2.2, 1.6)))
    ib.text(1.42, 100, "A", fontsize=5.4, color=ENC, va="center",
            fontweight="bold")
    ib.set_ylim(0, 118); ib.set_xlim(-0.62, 1.62)
    ib.set_xticks([0, 1])
    ib.set_xticklabels(["router", "best\nconstant"], fontsize=5.2)
    ib.set_yticks([0, 50, 100]); ib.tick_params(labelsize=5.2, length=1.6,
                                                pad=1.0)
    ib.set_title("% of tiles routed to the\noracle's exit", fontsize=5.6,
                 pad=2.0, color=S.INK, linespacing=1.25)
    ib.text(0.5, -0.34, f"600 frames, {ev['n_tiles']} tiles; "
            f"held-out {D['heldout']*100:.1f}%", transform=ib.transAxes,
            fontsize=5.0, ha="center", color=S.MUTED)
    return ys - 4.35


# --------------------------------------------------------------------------
# C
# --------------------------------------------------------------------------
def band_C(fig, ax, D, top):
    c = D["c"]
    band_title(ax, top, "C", "partial signalling",
               "the encoder runs B's predictor too, sees exactly where it "
               "will be wrong, and spends a handful of bits only there", SIG)
    y = top - 2.00
    pipeline(ax, y, "latent  +  a mask over the worst $\\rho$ of tiles", SIG)

    ys, hs = top - 4.05, 1.15
    box(ax, 0.05, ys, 2.05, hs, "oracle map $k^{*}$\n(A, steps 1-3)",
        ENC_F, ENC, fs=5.5)
    box(ax, 0.05, ys - 1.55, 2.05, hs, "a REPLICA of B's\npredictor $\\hat k$",
        DEC_F, DEC, fs=5.5)
    box(ax, 2.60, ys - 0.78, 2.55, 1.55,
        "regret of each tile\n"
        "$\\Delta(t)=[\\hat m+\\lambda c_{\\hat k}]-[m^{*}+\\lambda c_{k^{*}}]$",
        ENC_F, ENC, fs=5.4)
    arrow(ax, (2.10, ys + hs / 2), (2.60, ys + 0.35), ENC)
    arrow(ax, (2.10, ys - 1.55 + hs / 2), (2.60, ys - 0.35), DEC)
    box(ax, 5.65, ys - 0.78, 2.05, 1.55,
        "rank, keep the\ntop $\\rho$", ENC_F, ENC, fs=5.5)
    arrow(ax, (5.15, ys), (5.65, ys), ENC)
    box(ax, 8.20, ys - 0.78, 2.20, 1.55,
        "entropy-code\nthose tiles only", SIG_F, SIG, fs=5.5)
    arrow(ax, (7.70, ys), (8.20, ys), ENC)
    elbow(ax, [(9.30, ys + 0.77), (9.30, y - 0.52), (4.90, y - 0.52),
               (4.90, y - 0.03)], SIG, lw=1.0)
    ax.text(5.07, y - 0.32, "the mask, into the file", fontsize=5.5,
            color=SIG, fontweight="bold", va="center")
    box(ax, 6.14, y - 1.62, 1.42, 0.88, "in the mask?\nuse it : predict",
        DEC_F, DEC, fs=5.4)
    arrow(ax, (6.85, y - 0.74), (6.85, y - 0.03), DEC)

    q0 = c[0]
    lor = next(r for r in q0 if abs(r["rho"] - 0.10) < 1e-9)
    n = ("C is worth drawing only because regret is not spread out. At qp0 "
         f"the Gini of $\\Delta$ over tiles is {q0[0]['gini_regret']:.2f}, and "
         f"the worst tenth -- {lor['overridden_per_frame']:.1f} of "
         f"{lor['tiles_per_frame']:.0f} tiles --\ncarries "
         f"{lor['lorenz_at_rho']*100:.1f}% of it. $\\rho=0$ is exactly B and "
         "$\\rho=1$ is exactly A; what a tenth of the map actually buys "
         "back is below, and it is not all of the gap.")
    ax.text(0.05, ys - 2.15, n, fontsize=5.3, color=SIG, va="top",
            linespacing=1.45)
    # The maximum over rho is rho=1, which is A -- reporting it would say
    # nothing about C. The operating point that means something is the knee:
    # what a tenth of the map buys back of the gap B left open.
    knee = {q: next(r for r in c[q] if abs(r["rho"] - 0.10) < 1e-9)
            for q in QPS}
    frac = {q: (knee[q]["saving_pct_vs_release"] - c[q][0]["saving_pct_vs_release"])
            / (c[q][-1]["saving_pct_vs_release"]
               - c[q][0]["saving_pct_vs_release"]) for q in QPS}
    res = "   ".join(f"qp{q} {knee[q]['saving_pct_vs_release']:.1f}%"
                     for q in QPS)
    ax.text(0.05, ys - 3.05,
            "at $\\rho=0.1$, {:.0f} bits per frame:  ".format(
                knee[0]["map_bits"]) + res,
            fontsize=5.9, color=S.INK, va="top", fontweight="bold")
    ax.text(0.05, ys - 3.55,
            "which is {:.0f}-{:.0f}% of everything A had over B, for a "
            "quarter of A's map. The curve is monotone in $\\rho$: the "
            "maximum is $\\rho=1$, and $\\rho=1$ is A.".format(
                100 * min(frac.values()), 100 * max(frac.values())),
            fontsize=5.5, color=S.INK2, va="top")

    ic = inset(fig, 11.52, ys - 1.95, 2.62, 2.55)
    for i, q in enumerate(QPS):
        xs = [r["map_bits"] for r in c[q]]
        vs = [r["saving_pct_vs_release"] for r in c[q]]
        col = plt.cm.viridis(0.08 + 0.78 * i / (len(QPS) - 1))
        ic.plot(xs, vs, color=col, lw=0.9, zorder=3)
        ic.plot(xs[0], vs[0], "o", ms=2.4, mfc="white", mec=col, mew=0.7,
                zorder=4)
        ic.plot(xs[-1], vs[-1], "o", ms=2.4, color=col, zorder=4)
        ic.text(xs[-1] + 4, vs[-1], f"{q}", fontsize=5.0, color=col,
                va="center")
    ic.set_xlim(-6, 128); ic.set_ylim(16.5, 36.5)
    ic.set_xticks([0, 50, 100]); ic.set_yticks([20, 25, 30, 35])
    ic.tick_params(labelsize=5.2, length=1.6, pad=1.0)
    ic.set_xlabel("map bits per frame", fontsize=5.4, labelpad=1.0)
    ic.set_ylabel("saving (%)", fontsize=5.4, labelpad=1.0)
    ic.set_title("$\\rho$ sweeps B$\\,\\to\\,$A", fontsize=5.6, pad=2.0,
                 color=S.INK)
    ic.text(0.03, 0.05, "○ $\\rho=0$ = B    ● $\\rho=1$ = A",
            transform=ic.transAxes, fontsize=5.0, color=S.INK2)
    return ys - 4.05


def main():
    D = data()
    fig = plt.figure(figsize=(7.0, 7.0 * H / W))
    ax = fig.add_axes([0, 0, 1, 1]); ax.set_xlim(0, W); ax.set_ylim(0, H)
    ax.axis("off")
    yA = band_A(fig, ax, D, H - 0.45)
    rule(ax, yA - 0.25)
    yB = band_B(fig, ax, D, yA - 0.85)
    rule(ax, yB - 0.25)
    band_C(fig, ax, D, yB - 0.85)
    fig.savefig(OUT / "fig16_configs.pdf")
    fig.savefig(OUT / "fig16_configs.png")
    plt.close(fig)
    print("  fig16_configs yazildi")


if __name__ == "__main__":
    main()
