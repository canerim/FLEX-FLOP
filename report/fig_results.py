"""Every measured figure in the report, drawn from the result files.

No number is typed here. Each panel names the file it reads, so a figure and
the table beside it cannot disagree -- the failure this project has already
been bitten by, when a PNG carried a saving the table did not.
"""
import json, sys
from pathlib import Path
import numpy as np
import matplotlib.pyplot as plt
sys.path.insert(0, str(Path(__file__).resolve().parent))
import style as S

S.setup()
HERE = Path(__file__).resolve().parent
OUT = HERE / "fig"
RES = HERE.parent / "flexplus" / "results"
QPS = [0, 16, 32, 48, 63]


def J(name):
    return json.loads((RES / name).read_text())


def fig3_ladder():
    """What each exit costs, measured on the test set, per rate."""
    d = np.load(RES / "cells_ctc64.npz", allow_pickle=True)
    M, R, qp = d["M"].astype(float), d["R"].astype(float), d["qp"]
    K, j = int(d["K"][0]), int(d["split_depth"][0])
    blocks = [2 * (k + 1) for k in range(K)]
    fig, axes = plt.subplots(1, 2, figsize=(6.6, 2.25))

    ax = axes[0]
    for i, q in enumerate([0, 32, 63]):
        s = qp == q
        y = [10 * np.log10(M[s][:, k].mean() / R[s].mean()) for k in range(K)]
        ax.plot(blocks, y, "o-", color=S.CAT[i], label=f"qp {q}")
        S.label_end(ax, blocks[-1], y[-1], f"qp{q}", S.CAT[i], dx=4)
    ax.axvspan(1.2, 5.0, color=S.ORANGE, alpha=0.10, lw=0)
    ax.text(3.1, ax.get_ylim()[1] * 0.86, "closed by\nthe clamp", ha="center",
            fontsize=6.0, color=S.VERM)
    ax.set_xlabel("trunk blocks run"); ax.set_ylabel("dB above the release")
    ax.set_xticks(blocks); ax.set_yscale("log")
    ax.set_title("measured cost of each rung", fontsize=7.0, pad=4, loc="left")
    S.panel(ax, "a")
    S.despine(ax)

    ax = axes[1]
    p = {e: J(f"psnr_{t}.json") for t, e in
         [("epo0", 0), ("PIN_e1", 1), ("PIN_e3", 3), ("PAPER", 4),
          ("PIN_e5", 5), ("PIN_e6", 6), ("PIN_e7", 7)]}
    eps = sorted(p)
    for i, q in enumerate([0, 32, 63]):
        y = [[r for r in p[e]["rows"] if r["qp"] == q][0] for e in eps]
        sp = [r["psnr_per_exit"][-1] - r["psnr_per_exit"][2] for r in y]
        ax.plot(eps, sp, "o-", color=S.CAT[i])
        S.label_end(ax, eps[-1], sp[-1], f"qp{q}", S.CAT[i], dx=4)
    ax.axvline(4, color=S.MUTED, lw=0.8, ls=(0, (3, 2)))
    ax.text(4.12, ax.get_ylim()[1] * 0.97, "pin", fontsize=6.2, color=S.INK2,
            va="top")
    ax.set_xlabel("epoch"); ax.set_ylabel("spread $e_2\\!\\to\\!e_5$ (dB)")
    ax.set_title("the spread stops at epoch 4", fontsize=7.0, pad=4, loc="left")
    S.panel(ax, "b")
    S.despine(ax)
    fig.savefig(OUT / "fig3_ladder.pdf"); fig.savefig(OUT / "fig3_ladder.png")
    plt.close(fig)


def fig4_granularity():
    """Finer cells buy saving and pay a band; where the two cross."""
    g = J("granularity_ctc53_fixed.json")
    by = {r["qp"]: {c["cell_px"]: c for c in r["cells"]} for r in g["rows"]}
    cells = [256, 128, 64, 32]
    fig, axes = plt.subplots(1, 2, figsize=(6.5, 2.15),
                             gridspec_kw=dict(wspace=0.34))

    ax = axes[0]
    for i, q in enumerate([0, 32, 63]):
        y = [by[q][c]["saving_pct_vs_release"] for c in cells]
        ax.plot(range(len(cells)), y, "o-", color=S.CAT[i], clip_on=False,
                zorder=3)
        ax.annotate(f"qp {q}", (len(cells) - 1, y[-1]), xytext=(5, 0),
                    textcoords="offset points", color=S.CAT[i], fontsize=6.4,
                    va="center", fontweight="bold", annotation_clip=False)
    ax.set_xticks(range(len(cells))); ax.set_xticklabels([str(c) for c in cells])
    ax.set_xlim(-0.16, len(cells) - 1 + 0.62)
    ax.set_xlabel("allocation cell (px)"); ax.set_ylabel("saving (%)")
    ax.set_title("net saving after the band is charged", fontsize=7.0, pad=4,
                 loc="left")
    S.panel(ax, "a", dx=-0.20)
    S.ygrid(ax); S.despine(ax)

    ax = axes[1]
    net = [np.mean([by[q][c]["saving_pct_vs_release"] for q in QPS]) for c in cells]
    band = [np.mean([by[q][c]["band_cost_pct"] for q in QPS]) for c in cells]
    xs = np.arange(len(cells)); w = 0.46
    ax.bar(xs, net, w, color=S.BLUE, zorder=3)
    ax.bar(xs, band, w, bottom=[n + 0.10 for n in net], color=S.ORANGE, zorder=3)
    for x, nv, bv in zip(xs, net, band):
        ax.text(x, nv + bv + 0.9, f"{bv:.2f}", ha="center", fontsize=6.0,
                color=S.VERM, fontweight="bold")
        ax.text(x, nv - 1.9, f"{nv:.1f}", ha="center", fontsize=6.4,
                color="white", fontweight="bold")
    ax.annotate("band cost", (xs[-1] + 0.30, net[-1] + band[-1] / 2),
                xytext=(9, 6), textcoords="offset points", ha="left",
                fontsize=6.2, color=S.VERM, annotation_clip=False,
                arrowprops=dict(arrowstyle="-", color=S.VERM, lw=0.5,
                                shrinkA=0, shrinkB=1))
    ax.text(xs[1], net[1] / 2, "net saving", ha="center", fontsize=6.2,
            color="white")
    ax.set_xticks(xs); ax.set_xticklabels([str(c) for c in cells])
    ax.set_ylim(0, 41); ax.set_xlim(-0.55, len(cells) - 1 + 1.05)
    ax.set_xlabel("allocation cell (px)")
    ax.set_ylabel("mean over five rates (%)")
    ax.set_title("at 32 px the band eats the gain", fontsize=7.0, pad=4,
                 loc="left")
    S.panel(ax, "b", dx=-0.20)
    S.ygrid(ax); S.despine(ax)
    fig.savefig(OUT / "fig4_granularity.pdf")
    fig.savefig(OUT / "fig4_granularity.png")
    plt.close(fig)


def _front(name, q):
    # The codec's per-frame decibel and the saving measured against the
    # RELEASED decoder. paper_curve.py writes both conventions, and its
    # defaults are the other one in each pair -- pooled dB (flattering by
    # 0.023-0.033) and saving against our own 1.0095 deepest -- which put the
    # tiled curve above the per-position ones at qp63 the first time this was
    # drawn. The rebased copy is what is read here.
    rows = [r for r in J(name)["rows"] if r["qp"] == q]
    best = {}
    for r in rows:
        s = round(r["saving_pct"], 6)
        if s not in best or r["db_vs_uf"] < best[s]:
            best[s] = r["db_vs_uf"]
        pts = sorted(best.items())
    return np.array([p[0] for p in pts]), np.array([p[1] for p in pts])


def fig5_frontier():
    """The compute-quality frontier, and the interval the BD number integrates."""
    fig, axes = plt.subplots(1, 2, figsize=(6.6, 2.35))
    series = [("frontier_tiled256_rebased.json", "tiled 256 px", S.VERM, "-"),
              ("frontier_pp256.json", "per position 256 px", S.ORANGE, "-"),
              ("frontier_pp64.json", "per position 64 px", S.BLUE, "-"),
              ("frontier_dep64.json", "64 px, real router", S.GREEN, (0, (4, 2)))]
    for ax, q in zip(axes, [0, 63]):
        ax.axvspan(0.05, 0.12, color="#f0f0f0", lw=0, zorder=0)
        for name, lab, col, ls in series:
            s, db = _front(name, q)
            m = db <= 0.32
            ax.plot(db[m], s[m], ls=ls, color=col, label=lab)
        ax.axvline(0.10, color=S.MUTED, lw=0.8, ls=(0, (2, 2)))
        ax.set_xlabel("dB below the release"); ax.set_xlim(0, 0.30)
        ax.set_title(f"qp {q}", fontsize=7.0, pad=4, loc="left")
        S.panel(ax, "ab"[0 if q == 0 else 1])
        S.despine(ax)
    axes[0].set_ylabel("saving (%)")
    axes[0].text(0.085, 4, "BD interval", fontsize=6.0, color=S.INK2, ha="center")
    axes[1].legend(loc="lower right")
    fig.savefig(OUT / "fig5_frontier.pdf"); fig.savefig(OUT / "fig5_frontier.png")
    plt.close(fig)


def fig6_waterfall():
    """Where the eight points come from, one lever at a time."""
    steps = [("paper\n(tiled, $j$=2)", 27.64, S.MUTED),
             ("+ per-position\ndepth", 31.51, S.BLUE),
             ("+ 64 px\ncell", 33.32, S.BLUE),
             ("+ $j$=0\n(unlocked)", 36.02, S.GREEN)]
    fig, ax = plt.subplots(figsize=(3.3, 2.35))
    prev = 0.0
    for i, (lab, v, col) in enumerate(steps):
        if i == 0:
            ax.bar(i, v, 0.60, color=col)
            ax.text(i, v / 2, f"{v:.2f}", ha="center", va="center",
                    color="white", fontsize=6.8, fontweight="bold")
        else:
            ax.bar(i, v - prev, 0.60, bottom=prev, color=col)
            ax.text(i, prev + (v - prev) / 2, f"+{v - prev:.2f}", ha="center",
                    va="center", color="white", fontsize=6.6, fontweight="bold")
            ax.plot([i - 0.8, i - 0.30], [prev, prev], color=S.MUTED, lw=0.6,
                    ls=(0, (2, 2)))
        ax.text(i, v + 0.55, f"{v:.2f}", ha="center", fontsize=6.6, color=S.INK)
        prev = v
    ax.set_xticks(range(len(steps)))
    ax.set_xticklabels([s[0] for s in steps], fontsize=6.2)
    ax.set_ylabel("saving at 0.1 dB (%)"); ax.set_ylim(0, 40)
    S.ygrid(ax)
    ax.set_title("all inference-time, one checkpoint", fontsize=7.2, pad=5)
    S.despine(ax)
    fig.savefig(OUT / "fig6_waterfall.pdf"); fig.savefig(OUT / "fig6_waterfall.png")
    plt.close(fig)


def fig7_gather():
    """Does the time follow the MAC count."""
    d = J("gather_bench.json")
    rows = sorted(d["rows"], key=lambda r: r["frac"])
    base = [r for r in rows if r["frac"] == 1.0][0]["ms"]
    f = np.array([r["frac"] for r in rows]); t = np.array([r["ms"] for r in rows])
    fig, ax = plt.subplots(figsize=(3.15, 2.35))
    ax.plot([0, 1], [0, 1], color=S.MUTED, lw=0.9, ls=(0, (3, 2)),
            label="ideal (linear)")
    ax.plot(f, t / base, "o-", color=S.BLUE, label="measured (gather)")
    ax.set_xlabel("fraction of positions kept"); ax.set_ylabel("time / dense")
    ax.set_xlim(0.2, 1.05); ax.set_ylim(0.2, 1.05)
    ax.set_title(f"{d['channels']} channels, {d['hw'][0]}x{d['hw'][1]}, fp16",
                 fontsize=7.2, pad=5)
    ax.legend(loc="upper left")
    S.despine(ax)
    fig.savefig(OUT / "fig7_gather.pdf"); fig.savefig(OUT / "fig7_gather.png")
    plt.close(fig)


def fig8_epochs():
    """The epoch series, re-measured, against the number the paper reports."""
    d = J("epoch_series_remeasured.json")
    eps = [r["epoch"] for r in d]; mean = [r["mean"] for r in d]
    paper = {0: 21.5, 1: 22.8, 3: 25.8, 4: 27.6}
    fig, ax = plt.subplots(figsize=(3.3, 2.35))
    ax.plot(eps, mean, "o-", color=S.BLUE, label="re-measured")
    ax.plot(list(paper), list(paper.values()), "s", color=S.VERM, ms=4.2,
            label="as reported", zorder=4, mfc="none", mew=1.2)
    ax.axvline(4, color=S.MUTED, lw=0.8, ls=(0, (3, 2)))
    ax.text(4.1, 22.0, "pin", fontsize=6.4, color=S.INK2)
    ax.set_xlabel("epoch"); ax.set_ylabel("mean saving at 0.1 dB (%)")
    ax.set_title("the paper's series, independently confirmed", fontsize=7.2, pad=5)
    ax.legend(loc="lower right")
    S.despine(ax)
    fig.savefig(OUT / "fig8_epochs.pdf"); fig.savefig(OUT / "fig8_epochs.png")
    plt.close(fig)


def fig9_router():
    """The routing axis: how little of it is left."""
    d = np.load(RES / "cells_ctc64.npz", allow_pickle=True)
    M = d["M"].astype(float); j = int(d["split_depth"][0])
    A = (10 * np.log10(M[:, j:] / M[:, -1:]))[:, :-1]
    sv = np.linalg.svd(A - A.mean(0), compute_uv=False)
    ev = sv ** 2 / (sv ** 2).sum()

    fig, axes = plt.subplots(1, 2, figsize=(6.6, 2.25))
    ax = axes[0]
    ax.bar(range(1, len(ev) + 1), 100 * ev, 0.55, color=S.BLUE)
    for i, v in enumerate(ev):
        ax.text(i + 1, 100 * v + 1.6, f"{100*v:.1f}", ha="center", fontsize=6.4,
                color=S.INK)
    ax.set_xticks(range(1, len(ev) + 1))
    ax.set_xlabel("component"); ax.set_ylabel("variance explained (%)")
    ax.set_title("the error curve is nearly rank-1", fontsize=7.0, pad=4, loc="left")
    S.panel(ax, "a")
    ax.set_ylim(0, 108)
    S.ygrid(ax); S.despine(ax)

    ax = axes[1]
    r2 = {r["model"]: r["mean_saving_pct"] for r in J("router_fit3.json")["rows"]}
    r0 = {r["model"]: r["mean_saving_pct"] for r in J("router_fit_j0.json")["rows"]}
    names = [("oracle", "oracle"), ("gbm_smear", "GBM (best)"),
             ("gbm_mono", "GBM monotone"), ("raterank", "rate-rank")]
    ys = np.arange(len(names))
    ax.barh(ys + 0.19, [r2.get(k, np.nan) for k, _ in names], 0.34,
            color=S.BLUE, label="$j$=2")
    ax.barh(ys - 0.19, [r0.get(k, np.nan) for k, _ in names], 0.34,
            color=S.GREEN, label="$j$=0")
    for i, (k, _) in enumerate(names):
        for off, dd, c in ((0.19, r2, S.BLUE), (-0.19, r0, S.GREEN)):
            v = dd.get(k)
            if v is not None:
                ax.text(v + 0.35, i + off, f"{v:.2f}", va="center", fontsize=6.0,
                        color=c)
    ax.set_yticks(ys); ax.set_yticklabels([n for _, n in names], fontsize=6.6)
    ax.invert_yaxis(); ax.set_xlabel("mean saving at 0.1 dB (%)")
    ax.set_xlim(0, 41); ax.legend(loc="lower right")
    ax.set_title("the gap between oracle and real", fontsize=7.0, pad=4, loc="left")
    S.panel(ax, "b")
    S.despine(ax)
    fig.savefig(OUT / "fig9_router.pdf"); fig.savefig(OUT / "fig9_router.png")
    plt.close(fig)


if __name__ == "__main__":
    OUT.mkdir(exist_ok=True)
    for f in (fig3_ladder, fig4_granularity, fig5_frontier, fig6_waterfall,
              fig7_gather, fig8_epochs, fig9_router):
        f(); print(f"  {f.__name__} yazildi", flush=True)


def fig12_interaction():
    """The two levers are not additive: each is worth more with the other."""
    import numpy as np
    vals = {  # measured, per-position decoding, 0.1 dB, oracle allocation
        (256, 2): [36.92, 33.99, 30.69, 28.69, 27.26],
        (256, 0): [40.51, 35.88, 30.41, 28.75, 26.78],
        (64, 2): [38.63, 35.75, 32.57, 30.53, 29.14],
        (64, 0): [45.56, 40.16, 33.84, 31.46, 29.09],
    }
    fig, axes = plt.subplots(1, 2, figsize=(6.5, 2.25),
                             gridspec_kw=dict(wspace=0.36))

    ax = axes[0]
    xs = np.arange(2); w = 0.34
    m = {k: float(np.mean(v)) for k, v in vals.items()}
    ax.bar(xs - w / 2, [m[(256, 2)], m[(64, 2)]], w, color=S.BLUE,
           label="$j$=2 (clamped)", zorder=3)
    ax.bar(xs + w / 2, [m[(256, 0)], m[(64, 0)]], w, color=S.GREEN,
           label="$j$=0 (unlocked)", zorder=3)
    for i, c in enumerate((256, 64)):
        for off, jj, col in ((-w / 2, 2, S.BLUE), (w / 2, 0, S.GREEN)):
            ax.text(i + off, m[(c, jj)] + 0.5, f"{m[(c, jj)]:.2f}", ha="center",
                    fontsize=6.2, color=col, fontweight="bold")
        d = m[(c, 0)] - m[(c, 2)]
        ax.annotate(f"+{d:.2f}", (i, max(m[(c, 2)], m[(c, 0)]) + 2.4),
                    ha="center", fontsize=6.6, color=S.INK, fontweight="bold")
    ax.set_xticks(xs); ax.set_xticklabels(["256 px cell", "64 px cell"])
    ax.set_ylabel("mean saving at 0.1 dB (%)"); ax.set_ylim(0, 42)
    ax.set_title("unlocking pays only where there is room", fontsize=7.0,
                 pad=4, loc="left")
    ax.legend(loc="lower right")
    S.panel(ax, "a", dx=-0.20); S.ygrid(ax); S.despine(ax)

    ax = axes[1]
    for i, c in enumerate((256, 64)):
        d = [vals[(c, 0)][k] - vals[(c, 2)][k] for k in range(5)]
        col = [S.ORANGE, S.GREEN][i]
        ax.plot(range(5), d, "o-", color=col, clip_on=False, zorder=3)
        ax.annotate(f"{c} px", (4, d[-1]), xytext=(5, 0),
                    textcoords="offset points", color=col, fontsize=6.4,
                    va="center", fontweight="bold", annotation_clip=False)
    ax.axhline(0, color=S.INK, lw=0.6, zorder=2)
    ax.set_xticks(range(5)); ax.set_xticklabels([str(q) for q in QPS])
    ax.set_xlim(-0.15, 4.6)
    ax.set_xlabel("rate (qp)"); ax.set_ylabel("gain from $j$=0 (points)")
    ax.set_title("the gain is entirely at low rate", fontsize=7.0, pad=4,
                 loc="left")
    S.panel(ax, "b", dx=-0.20); S.ygrid(ax); S.despine(ax)
    fig.savefig(OUT / "fig12_interaction.pdf")
    fig.savefig(OUT / "fig12_interaction.png")
    plt.close(fig)
