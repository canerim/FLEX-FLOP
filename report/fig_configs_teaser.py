"""The three configurations, one file each, in the teaser's language.

They share a decoder -- same checkpoint, same ladder, same Lagrangian -- and
differ in one thing: who decides the exit map, and what that decision costs to
transmit. Drawn on one grammar so a talk can put them in sequence and only the
decision path moves.

Warm is encoder-side, cool is decoder-side, and the accent is the only thing
that touches the bitstream. Every number is measured on the deployed tiled
path at epoch 9, 53 CTC intra frames, with the multiplier bisected per frame
so that no frame is served worse than the budget.
"""
import json, sys
from pathlib import Path
import numpy as np
import matplotlib.pyplot as plt
sys.path.insert(0, str(Path(__file__).resolve().parent))
from teaser_style import *   # noqa

setup()
HERE = Path(__file__).resolve().parent
OUT = HERE / "fig"; OUT.mkdir(exist_ok=True)
RES = HERE.parent / "flexplus" / "results"
UF = Path.home() / "FLEX-UF"
QPS = [0, 16, 32, 48, 63]

ENC = ("#fbdedb", "#d98b83")
DEC = ("#dfe8f5", "#6f9fd4")
SIG = ("#d9efe2", "#4fa87c")
SIGT = "#2f7d55"
E = EXITS


def numbers():
    pf = json.loads((RES / "signalled_perframe_e9.json").read_text())
    ca = json.loads((RES / "signalled_perframe_calexA.json").read_text())
    A = json.loads((UF / "results/signalled_RECIPE512_ctc53.json").read_text())
    B = json.loads((UF / "results/router_RECIPE512_b01_e4head.json").read_text())
    C = json.loads((UF / "results/hybrid_RECIPE512_b01_e4head.json").read_text())
    sv = lambda r: r.get("saving_pct_measured", r.get("saving_pct_vs_release"))
    a = {r["qp"]: r for r in A["rows"]
         if r.get("budget_reachable") and abs(r.get("budget_db", 0) - 0.1) < 1e-9}
    b = {r["qp"]: r for r in B["rows"]}
    c = {}
    for r in C["rows"]:
        c.setdefault(r["qp"], []).append(r)
    return dict(
        pf_mean=float(np.mean([r["saving_pct_measured"] for r in pf["rows"]])),
        ca_mean=float(np.mean([r["saving_pct_measured"] for r in ca["rows"]])),
        pf_bits=(min(r["map_bits"] for r in pf["rows"]),
                 max(r["map_bits"] for r in pf["rows"])),
        A_mean=float(np.mean([sv(a[q]) for q in QPS])),
        A_bits=(min(a[q]["map_bits"] for q in QPS),
                max(a[q]["map_bits"] for q in QPS)),
        B_mean=float(np.mean([sv(b[q]) for q in QPS])),
        C=c, router_pct=B["router_compute_share_pct"],
        agree=B["router2_meta"]["eval"]["agree"] * 100,
    )


N = numbers()


def frame(title, tag, sub, channel, accent, draw, foot, fname):
    fig, ax = canvas(11.0, 4.9, (0, 28), (0, 12))
    ax.text(0.30, 11.65, tag, fontsize=14, fontweight="bold", va="top",
            color=accent)
    ax.text(2.05, 11.65, title, fontsize=14, fontweight="bold", va="top")
    ax.text(0.30, 10.85, sub, fontsize=8.2, color=GREY, va="top")

    # the row every configuration shares
    y = 8.15
    box(ax, 0.30, y, 1.90, 1.45, "source\nframe", GREY_B, fs=7.4)
    arrow(ax, (2.20, y + 0.72), (2.75, y + 0.72))
    box(ax, 2.75, y, 2.05, 1.45, "encoder\n(frozen)", ENC, fs=7.4)
    arrow(ax, (4.80, y + 0.72), (5.35, y + 0.72))
    ax.add_patch(plt.Rectangle((5.35, y - 0.05), 7.55, 1.55,
                               facecolor="#f0f0f0", edgecolor="none", zorder=1))
    ax.text(5.60, y + 1.18, "bitstream", fontsize=7.2, color=GREY, va="center")
    ax.text(5.60, y + 0.45, channel, fontsize=7.8, va="center",
            color=accent if accent != GREY else INK,
            fontweight="bold" if accent != GREY else "normal")
    arrow(ax, (12.90, y + 0.72), (13.45, y + 0.72))
    box(ax, 13.45, y, 2.05, 1.45, "parse,\ndilate", DEC, fs=7.4)
    arrow(ax, (15.50, y + 0.72), (16.05, y + 0.72))
    box(ax, 16.05, y, 2.35, 1.45, "trunk, only\nwhere $D \\geq b$", DEC, fs=7.4)
    arrow(ax, (18.40, y + 0.72), (18.95, y + 0.72))
    box(ax, 18.95, y, 2.30, 1.45, "adapter,\nstitch, head", DEC, fs=7.4)
    arrow(ax, (21.25, y + 0.72), (21.80, y + 0.72))
    box(ax, 21.80, y, 1.75, 1.45, "image", GREY_B, fs=7.6)

    draw(ax, y)

    for i, (lab, val, col) in enumerate(foot):
        x = 0.30 + i * 6.7
        ax.text(x, 1.85, lab, fontsize=7.2, color=GREY, va="top")
        ax.text(x, 1.20, val, fontsize=9.6, color=col, va="top",
                fontweight="bold")
    fig.savefig(OUT / f"{fname}.pdf"); fig.savefig(OUT / f"{fname}.png")
    plt.close(fig)
    print(f"  {fname} yazildi")


# ------------------------------------------------------------------ A
def draw_A(ax, y):
    group(ax, 0.30, 3.55, 12.60, 3.30,
          "Encoder side — it holds the source, so it measures instead of "
          "predicting", ("#fdf5f4", "#d98b83"), fs=7.4, label_dy=-0.30)
    # Explicit positions. A cumulative expression put box four on top of the
    # entropy coder the first time this was drawn.
    steps = [("decode the tile\nat EVERY exit", 0.70, 2.35, ENC),
             ("tabulate the\nexact $m(t,k)$", 3.33, 2.25, ENC),
             ("$\\arg\\min_k[\\,m+\\lambda c_k\\,]$", 5.86, 2.55, ENC),
             ("bisect $\\lambda$ per\nFRAME", 8.69, 1.95, ENC),
             ("entropy-\ncode", 10.92, 1.50, SIG)]
    for i, (lab, x, w, fill) in enumerate(steps):
        box(ax, x, 4.35, w, 1.90, lab, fill, fs=7.0)
        if i:
            px = steps[i - 1][1] + steps[i - 1][2]
            arrow(ax, (px, 5.30), (x, 5.30), ENC[1], lw=1.0, ms=8)
    elbow(ax, [(11.67, 6.25), (11.67, 7.35), (8.60, 7.35), (8.60, 8.05)],
          SIGT, lw=1.4)
    ax.text(8.85, 7.55, "the map, into the file", fontsize=7.4, color=SIGT,
            fontweight="bold")
    gridglyph(ax, 14.10, 4.40, 2.40, 2.40, 5, 5,
              colors=[E[2], E[2], E[3], E[2], E[4], E[2], E[3], E[2], E[2],
                      E[5], E[3], E[2], E[2], E[4], E[2], E[2], E[2], E[3],
                      E[2], E[2], E[4], E[2], E[2], E[2], E[3]])
    ax.text(15.30, 7.05, "the exact map", ha="center", fontsize=7.6,
            fontweight="bold")
    ax.text(17.10, 6.30,
            "The objective is separable over tiles and the cost is closed\n"
            "form, so the per-tile argmin IS the constrained optimum.\n"
            "There is no search left to improve.",
            fontsize=7.2, color=ENC[1], va="top")
    ax.text(17.10, 4.35,
            "Bisecting per frame rather than per rate turns the budget\n"
            "from a set average into a guarantee, and costs 16 bits.",
            fontsize=7.2, color=GREY, va="top")


frame("signalled", "A",
      "the encoder measures the allocation, minimises it exactly, and sends "
      "the answer",
      f"latent  +  exit map,  {N['pf_bits'][0]:.0f}–{N['pf_bits'][1]:.0f} bits per frame",
      SIGT, draw_A,
      [("mean MAC saving, 0.1 dB", f"{N['pf_mean']:.2f}%", INK),
       ("frames inside the budget", "264 / 265", SIGT),
       ("added to the bitstream", f"{N['pf_bits'][1]:.0f} bits, "
        "about $2\\times10^{-4}$ of the frame", GREY)],
      "fig_cfgA_teaser")


# ------------------------------------------------------------------ B
def draw_B(ax, y):
    group(ax, 0.30, 3.55, 12.60, 3.30,
          "Decoder side — every input is a tensor the decode already produced",
          ("#f2f6fc", "#6f9fd4"), fs=7.4, label_dy=-0.30)
    box(ax, 0.70, 4.85, 2.35, 1.40, "stem feature\n1×1 384→48", DEC, fs=7.0)
    box(ax, 3.30, 4.85, 2.55, 1.40, "latent + scales\n1×1 512→32", DEC, fs=7.0)
    box(ax, 6.10, 4.85, 0.95, 1.40, "qp", GREY_B, fs=7.6)
    # The three inputs join a bus and the bus enters the MLP. The first
    # version dropped an arrow below the bus into empty space.
    for x in (1.88, 4.58, 6.58):
        ax.plot([x, x], [4.85, 4.30], color=DEC[1], lw=1.0, zorder=4)
    ax.plot([1.88, 6.58], [4.30, 4.30], color=DEC[1], lw=1.0, zorder=4)
    box(ax, 7.35, 4.85, 5.15, 1.40,
        "MLP 161-256-256-6,  144,024 parameters\n"
        f"agrees with the oracle on {N['agree']:.1f}% of tiles", DEC, fs=7.0)
    elbow(ax, [(6.58, 4.30), (6.95, 4.30), (6.95, 5.55), (7.35, 5.55)],
          DEC[1], lw=1.0, ms=8)
    ax.text(4.05, 4.05, "$d = 161$", fontsize=7.2, color=DEC[1], ha="center")
    elbow(ax, [(9.90, 6.25), (9.90, 7.35), (14.45, 7.35), (14.45, 8.05)],
          DEC[1], lw=1.4)
    ax.text(10.15, 7.55, "$\\hat k(t)$ — never transmitted", fontsize=7.4,
            color=DEC[1], fontweight="bold")
    gridglyph(ax, 14.10, 4.40, 2.40, 2.40, 5, 5,
              colors=[E[2], E[2], E[3], E[2], E[3], E[2], E[3], E[2], E[2],
                      E[4], E[3], E[2], E[2], E[3], E[2], E[2], E[2], E[4],
                      E[2], E[2], E[3], E[2], E[2], E[2], E[3]])
    ax.text(15.30, 7.05, "the predicted map", ha="center", fontsize=7.6,
            fontweight="bold")
    ax.text(17.10, 6.30,
            "The file is byte for byte the released one, so this is the only\n"
            "configuration that can be dropped in front of a bitstream\n"
            "encoded before the method existed.",
            fontsize=7.2, color=DEC[1], va="top")
    ax.text(17.10, 4.35,
            "And the only one that cannot guarantee the budget: it never\n"
            "sees the source, so it cannot measure what it is delivering.",
            fontsize=7.2, color=GREY, va="top")


frame("bitstream-identical", "B",
      "nothing is added to the file; the decoder recovers the map from what "
      "it already holds",
      "latent only  —  byte for byte the released file", GREY, draw_B,
      [("mean MAC saving, 0.1 dB", f"{N['B_mean']:.2f}%", INK),
       ("added to the bitstream", "0 bits", DEC[1]),
       ("router cost", f"{N['router_pct']:.3f}% of one decode", GREY)],
      "fig_cfgB_teaser")


# ------------------------------------------------------------------ C
def draw_C(ax, y):
    group(ax, 0.30, 3.55, 12.60, 3.30,
          "Both sides — the encoder runs a REPLICA of the decoder's predictor",
          ("#fbf6f4", "#c08a6a"), fs=7.4, label_dy=-0.30)
    box(ax, 0.70, 5.55, 2.30, 1.15, "oracle map $k^{*}$", ENC, fs=7.0)
    box(ax, 0.70, 4.05, 2.30, 1.15, "replica of B's $\\hat k$", DEC, fs=7.0)
    box(ax, 3.30, 4.55, 3.30, 1.65,
        "regret of each tile\n$\\Delta(t)=[\\hat m+\\lambda c_{\\hat k}]-"
        "[m^{*}+\\lambda c_{k^{*}}]$", ENC, fs=6.2)
    arrow(ax, (3.00, 6.10), (3.30, 5.75), ENC[1], lw=1.0, ms=8)
    arrow(ax, (3.00, 4.60), (3.30, 5.00), DEC[1], lw=1.0, ms=8)
    box(ax, 6.90, 4.55, 2.25, 1.65, "rank, keep\nthe top $\\rho$", ENC, fs=7.0)
    arrow(ax, (6.60, 5.38), (6.90, 5.38), ENC[1], lw=1.0, ms=8)
    box(ax, 9.45, 4.55, 3.05, 1.65, "entropy-code\nthose tiles only", SIG,
        fs=7.0)
    arrow(ax, (9.15, 5.38), (9.45, 5.38), ENC[1], lw=1.0, ms=8)
    elbow(ax, [(11.00, 6.20), (11.00, 7.35), (8.60, 7.35), (8.60, 8.05)],
          SIGT, lw=1.4)
    ax.text(8.85, 7.55, "the worst tiles, and the price", fontsize=7.4,
            color=SIGT, fontweight="bold")
    cols = [E[2]] * 25
    for i in (4, 9, 13, 20):
        cols[i] = E[5]
    gridglyph(ax, 14.10, 4.40, 2.40, 2.40, 5, 5, colors=cols)
    ax.text(15.30, 7.05, "only these are sent", ha="center", fontsize=7.6,
            fontweight="bold")
    ax.text(17.10, 6.30,
            "Regret is not spread out. At qp0 its Gini over tiles is 0.95 and\n"
            "the worst tenth — 3.4 tiles of 33 — carries 99.8% of it, so a\n"
            "handful of overrides reaches most of what B gave up.",
            fontsize=7.2, color=SIGT, va="top")
    ax.text(17.10, 4.35,
            "$\\rho = 0$ is exactly B and $\\rho = 1$ is exactly A. The same\n"
            "channel carries the per-frame multiplier that makes B guarantee.",
            fontsize=7.2, color=GREY, va="top")


knee = next(r for r in N["C"][0] if abs(r["rho"] - 0.10) < 1e-9)
frame("partial signalling", "C",
      "the map is sent only where the predictor is worst, and the price is "
      "sent always",
      "latent  +  a mask over the worst $\\rho$ of tiles", SIGT, draw_C,
      [("at $\\rho=0.1$, qp0", f"{knee['saving_pct_vs_release']:.2f}%", INK),
       ("map bits at that point", f"{knee['map_bits']:.0f} bits/frame", SIGT),
       ("tiles overridden", f"{knee['overridden_per_frame']:.1f} of "
        f"{knee['tiles_per_frame']:.0f}", GREY)],
      "fig_cfgC_teaser")
