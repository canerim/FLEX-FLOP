"""Three diagrams, one per configuration, on one grammar.

A, B and C share a decoder -- the same pinned checkpoint, the same ladder --
and differ only in who decides the exit map and what that decision costs to
transmit. Drawing them on one grammar makes that the visible thing: the same
boxes in the same places, with the decision path moving between the encoder
side, the decoder side, and both.

Colour carries one meaning only. Encoder-side work is warm, decoder-side work
is cool, and the signalled path -- the only thing that touches the bitstream --
is the accent. Nothing is coloured for decoration.
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
RES = HERE.parent / "flexplus" / "results"
QPS = [0, 16, 32, 48, 63]

ENC = S.VERM          # encoder-side decisions
ENC_F = "#fdece2"
DEC = S.BLUE          # decoder-side decisions
DEC_F = "#e7f1f9"
SIG = S.GREEN         # anything that reaches the bitstream
SIG_F = "#e6f5ee"
NEU = S.MUTED
NEU_F = "#f4f4f4"


def box(ax, x, y, w, h, label, fc, ec, fs=6.0, lw=0.8, tc=None, z=2):
    ax.add_patch(FancyBboxPatch((x, y), w, h,
                                boxstyle="round,pad=0.004,rounding_size=0.045",
                                facecolor=fc, edgecolor=ec, linewidth=lw, zorder=z))
    ax.text(x + w / 2, y + h / 2, label, ha="center", va="center", fontsize=fs,
            color=tc or S.INK, zorder=z + 1, linespacing=1.3)


def arrow(ax, p, q, color=NEU, lw=0.8, ls="-", z=1):
    ax.add_patch(FancyArrowPatch(p, q, arrowstyle="-|>", mutation_scale=6,
                                 color=color, lw=lw, linestyle=ls,
                                 shrinkA=0, shrinkB=0, zorder=z))


def channel(ax, x0, x1, y, label, extra=None):
    """The bitstream, drawn as a band both sides write into and read from."""
    ax.add_patch(Rectangle((x0, y - 0.20), x1 - x0, 0.40, facecolor="#f0f0f0",
                           edgecolor="none", zorder=0))
    # The label rides ABOVE the band: the signalled path enters from below and
    # a caption inside the band collided with it in the first draft.
    ax.text(x0 + 0.05, y + 0.34, label, ha="left", fontsize=6.0, color=S.INK2)
    if extra:
        ax.text(x0 + 0.05, y + 0.68, extra, ha="left", fontsize=6.0,
                color=SIG, fontweight="bold")


def savings_strip(ax, y, vals, color, note):
    """The measured per-rate saving, under the diagram it belongs to."""
    x0, w = 0.9, 1.55
    for i, (q, v) in enumerate(zip(QPS, vals)):
        x = x0 + i * w
        ax.add_patch(Rectangle((x, y), w * 0.82, 0.30, facecolor=color,
                               alpha=0.13, edgecolor="none"))
        ax.text(x + w * 0.41, y + 0.15, f"{v:.1f}%", ha="center", va="center",
                fontsize=6.6, color=color, fontweight="bold")
        ax.text(x + w * 0.41, y - 0.20, f"qp {q}", ha="center", fontsize=5.7,
                color=S.INK2)
    ax.text(x0 + 5 * w + 0.10, y + 0.15, note, ha="left", va="center",
            fontsize=6.0, color=S.INK2)


def frame(name, title, subtitle, draw, vals, color, note, h=2.55):
    fig, ax = plt.subplots(figsize=(6.5, h))
    ax.set_xlim(0, 13.0); ax.set_ylim(0.0, 7.05); ax.axis("off")
    ax.text(0.0, 6.98, title, fontsize=8.6, fontweight="bold",
            color=S.INK, va="top")
    ax.text(0.0, 6.52, subtitle, fontsize=6.6, color=S.INK2, va="top")
    draw(ax)
    savings_strip(ax, 0.42, vals, color, note)
    fig.savefig(OUT / f"{name}.pdf"); fig.savefig(OUT / f"{name}.png")
    plt.close(fig)
    print(f"  {name} yazildi")


def _common(ax, y):
    """The parts every configuration shares: encode, channel, decode, ladder."""
    box(ax, 0.10, y, 1.15, 0.62, "source\nframe", NEU_F, NEU)
    arrow(ax, (1.25, y + 0.31), (1.60, y + 0.31))
    box(ax, 1.60, y, 1.30, 0.62, "encoder\n(frozen)", ENC_F, ENC)
    arrow(ax, (2.90, y + 0.31), (3.25, y + 0.31))
    box(ax, 8.10, y, 1.60, 0.62, "trunk, run\nwhere $D\\geq b$", DEC_F, DEC)
    arrow(ax, (9.70, y + 0.31), (10.05, y + 0.31))
    box(ax, 10.05, y, 1.15, 0.62, "stitch\n+ head", DEC_F, DEC)
    arrow(ax, (11.20, y + 0.31), (11.55, y + 0.31))
    box(ax, 11.55, y, 1.15, 0.62, "image", NEU_F, NEU)


def draw_A(ax):
    y = 4.55
    channel(ax, 3.25, 8.10, y + 0.31, "bitstream",
            "latent  +  exit map   (64-91 bits / frame)")
    _common(ax, y)
    # encoder-side search
    box(ax, 1.40, y - 2.35, 2.00, 0.70,
        "decode every exit,\ntabulate $m(x,k)$", ENC_F, ENC, fs=5.9)
    arrow(ax, (2.25, y), (2.40, y - 1.65), ENC, ls=(0, (2.4, 1.6)))
    box(ax, 3.70, y - 2.35, 2.10, 0.70,
        "argmin$_k\\,[\\,m+\\lambda c\\,]$\nbisect $\\lambda$ to 0.1 dB",
        ENC_F, ENC, fs=5.9)
    arrow(ax, (3.40, y - 2.00), (3.70, y - 2.00), ENC)
    box(ax, 6.10, y - 2.35, 1.60, 0.70, "entropy-code\nthe map", SIG_F, SIG, fs=5.9)
    arrow(ax, (5.80, y - 2.00), (6.10, y - 2.00), ENC)
    arrow(ax, (6.90, y - 1.65), (6.90, y + 0.11), SIG, lw=1.1)
    ax.text(7.00, y - 0.80, "map", fontsize=5.9, color=SIG, fontweight="bold")
    ax.text(1.40, y - 2.72, "encoder side: has the source, so the table is exact",
            fontsize=5.9, color=ENC)


def draw_B(ax):
    y = 4.55
    channel(ax, 3.25, 8.10, y + 0.31, "bitstream",
            "latent only  --  byte for byte the released file")
    _common(ax, y)
    box(ax, 1.95, y - 2.45, 2.95, 0.78,
        "stem  1x1  384$\\to$48\nlatent+scales  1x1  512$\\to$32",
        DEC_F, DEC, fs=5.7)
    arrow(ax, (3.42, y), (3.42, y - 1.67), DEC, ls=(0, (2.4, 1.6)))
    arrow(ax, (4.90, y - 2.06), (5.20, y - 2.06), DEC)
    box(ax, 5.20, y - 2.45, 2.70, 0.78,
        "MLP  161-256-256-6\n144,024 parameters", DEC_F, DEC, fs=5.7)
    arrow(ax, (7.90, y - 2.06), (8.55, y - 0.02), DEC, lw=1.1)
    ax.text(8.58, y - 1.05, "$\\hat{k}(x)$", fontsize=6.4, color=DEC,
            fontweight="bold")
    ax.text(1.95, y - 2.80,
            "decoder side: every input is already in the decode; "
            "0.163% of a decode", fontsize=5.9, color=DEC)


def draw_C(ax):
    y = 4.55
    channel(ax, 3.25, 8.10, y + 0.31, "bitstream",
            "latent  +  a mask over the worst $\\rho$ of cells")
    _common(ax, y)
    box(ax, 1.15, y - 2.35, 1.85, 0.70, "oracle map\n$k^{*}(x)$", ENC_F, ENC, fs=5.9)
    box(ax, 1.15, y - 3.35, 1.85, 0.70, "run the SAME\npredictor as B",
        DEC_F, DEC, fs=5.9)
    arrow(ax, (2.05, y), (2.05, y - 1.65), ENC, ls=(0, (2.4, 1.6)))
    box(ax, 3.35, y - 2.90, 2.05, 0.80,
        "rank cells by\n$m(\\hat k)-m(k^{*})$", ENC_F, ENC, fs=5.9)
    arrow(ax, (3.00, y - 2.00), (3.35, y - 2.35), ENC)
    arrow(ax, (3.00, y - 3.00), (3.35, y - 2.65), DEC)
    box(ax, 5.75, y - 2.90, 1.95, 0.80,
        "send top $\\rho$\nonly", SIG_F, SIG, fs=6.0)
    arrow(ax, (5.40, y - 2.50), (5.75, y - 2.50), ENC)
    arrow(ax, (6.72, y - 2.10), (6.72, y + 0.11), SIG, lw=1.1)
    ax.text(6.82, y - 0.90, "mask", fontsize=5.9, color=SIG, fontweight="bold")
    box(ax, 8.10, y - 1.55, 1.60, 0.62, "merge: mask\nelse predict", DEC_F, DEC,
        fs=5.9)
    arrow(ax, (8.90, y - 0.93), (8.90, y), DEC)
    ax.text(1.15, y - 3.72,
            "both sides run; the map is spent where the predictor is worst",
            fontsize=5.9, color=S.INK2)


def main():
    import sys as _s
    _s.path.insert(0, str(HERE))
    from build_report import J
    sv = lambda r: r.get("saving_pct_measured", r.get("saving_pct_vs_release"))
    A = J("../../FLEX-UF/results/signalled_RECIPE512_ctc53.json") \
        if False else json.loads((Path.home() / "FLEX-UF/results/signalled_RECIPE512_ctc53.json").read_text())
    B = json.loads((Path.home() / "FLEX-UF/results/router_RECIPE512_b01_e4head.json").read_text())
    Ch = json.loads((Path.home() / "FLEX-UF/results/hybrid_raterank_b01.json").read_text())

    def per(d):
        ok = [r for r in d["rows"] if r.get("budget_reachable")
              and abs(r.get("budget_db", 0.1) - 0.1) < 1e-9]
        m = {r["qp"]: sv(r) for r in ok}
        return [m[q] for q in QPS]

    best = {}
    for r in Ch["rows"]:
        if not r.get("budget_reachable"):
            continue
        q, s = r["qp"], sv(r)
        if q not in best or s > best[q][1]:
            best[q] = (r.get("rho"), s)
    cvals = [best[q][1] for q in QPS]
    rhos = "/".join(f"{best[q][0]:g}" for q in QPS)

    OUT.mkdir(exist_ok=True)
    frame("fig_configA", "Configuration A  --  signalled",
          "the encoder searches the exact per-cell table and transmits the map; "
          "the decoder obeys it", draw_A, per(A), ENC,
          "oracle allocation, 64-91 bits/frame", h=2.55)
    frame("fig_configB", "Configuration B  --  bitstream-identical",
          "nothing is added to the file; a 144k-parameter head predicts the map "
          "from what the decode already holds", draw_B, per(B), DEC,
          "zero added bits, 0.163% of a decode", h=2.55)
    frame("fig_configC", "Configuration C  --  partial signalling",
          "the encoder sends the map only for the cells where the predictor is "
          "worst; the decoder predicts the rest", draw_C, cvals, SIG,
          f"best $\\rho$ per rate: {rhos}", h=2.85)


if __name__ == "__main__":
    main()
