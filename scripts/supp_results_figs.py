"""Figures for the supplement's Complete results section.

Nothing here measures anything. Every panel is a drawing of a file already in
`results/`, and the file is named in the docstring of the function that draws
it, so that a figure and the table beside it cannot disagree. The reason these
exist at all is that the section they belong to was printing grids of numbers
where the surveyed supplements print a picture: a six-class by five-rate grid
at three budgets is 90 cells, which is a heatmap, and a frontier is a curve.

    ./.venv/bin/python scripts/supp_results_figs.py

writes six PNGs into docs/figures/, all prefixed `res_`. CPU only.

The two decibel conventions are kept apart everywhere below, because they
differ by about a third of a 0.1 dB budget on this decoder:

  per frame   mean over frames of the per-frame PSNR difference. This is what
              results/supp_per_class_budgets.json and
              results/signalled_RECIPE512_ctc53.json bisect against, and what
              the main paper's budget means.
  pooled      one PSNR from the mean squared error over the whole set, then
              differenced. This is what results/supp_paper_curve_PAPER.json
              targets in its `target_db`, and it reports the per-frame value
              beside it as `db_vs_uf_per_frame`.

An axis labelled with a budget says which one it is.
"""
import json
import sys
from pathlib import Path

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

R = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(R / "scripts"))
import naturestyle as ns
ns.apply()

OUT = R / "docs" / "figures"
RES = R / "results"

CLASSES = ["MCL-JCV", "UVG", "HEVC_B", "HEVC_E", "HEVC_C", "HEVC_D"]
QPS = [0, 16, 32, 48, 63]
CCOL = {"MCL-JCV": ns.BLUE, "UVG": ns.SKY, "HEVC_B": ns.GREEN,
        "HEVC_E": ns.PURPLE, "HEVC_C": ns.ORANGE, "HEVC_D": ns.VERM}
QCOL = dict(zip(QPS, [ns.BLUE, ns.SKY, ns.GREEN, ns.ORANGE, ns.VERM]))
# e2 is the shallowest selectable exit; deeper is darker.
ECOL = ["#f0f0f0", "#bdd7e7", "#6baed6", "#2171b5", "#08306b"]


def J(name):
    return json.loads((RES / name).read_text())


def nice(c):
    return c.replace("HEVC_", "HEVC ")


def save(fig, name):
    fig.savefig(OUT / name, bbox_inches="tight", pad_inches=0.02, dpi=300)
    plt.close(fig)
    print(f"  -> docs/figures/{name}")


# --------------------------------------------------------------- the grid
def grid():
    """results/supp_per_class_budgets.json, 53 sequences, per-frame budget."""
    d = J("supp_per_class_budgets.json")
    idx = {(r["qp"], round(r["budget_db"], 3)): r for r in d["rows"]}

    def mat(budget, field):
        return np.array([[idx[(q, budget)]["per_class"][c][field] for q in QPS]
                         for c in CLASSES])

    # One column wide and stacked, not a full-width band. A band forces a
    # page break in the reportlab build, which abandons whatever is left of
    # the page it breaks from; three heatmaps of six rows fit a column
    # comfortably and cost no white page.
    fig, ax = plt.subplots(3, 1, figsize=(ns.W1, 3.55))
    panels = [(mat(0.1, "saving"), "compute saved (%), 0.1 dB budget",
               0, 40, "%.1f"),
              (mat(0.3, "saving"), "compute saved (%), 0.3 dB budget",
               0, 40, "%.1f"),
              (mat(0.1, "db"), "quality given up (dB), 0.1 dB budget",
               0, 0.13, "%.3f")]
    for j3, (a, (m, title, lo, hi, fmt)) in enumerate(zip(ax, panels)):
        a.imshow(m, cmap="cividis", vmin=lo, vmax=hi, aspect="auto")
        a.set_xticks(range(len(QPS)),
                     [f"q{q}" for q in QPS] if j3 == 2 else [""] * len(QPS))
        a.set_yticks(range(len(CLASSES)), [nice(c) for c in CLASSES])
        a.set_title(title, fontsize=6, color=ns.INK2, loc="left", pad=2.5)
        a.grid(False)
        rng = hi - lo
        for i in range(m.shape[0]):
            for j in range(m.shape[1]):
                v = m[i, j]
                a.text(j, i, fmt % v, ha="center", va="center", fontsize=5.4,
                       color="white" if (v - lo) / rng < 0.55 else "black")
        for sp in a.spines.values():
            sp.set_visible(False)
        a.tick_params(length=0)
    for a, L in zip(ax, "abc"):
        ns.panel(a, L, dx=-0.30, dy=1.30)
    fig.subplots_adjust(hspace=0.32)
    save(fig, "res_grid.png")


# ------------------------------------------------------------ the frontier
def frontier():
    """results/supp_paper_curve_PAPER.json op_points, split by class.

    Class membership comes from results/supp_opquality_PAPER.json, which
    records the class of each of the 53 sequences beside its measurement.
    A point is the mean over that class's sequences, so the x axis is the
    per-frame convention within the class even though the operating point
    itself was bisected on the pooled one.
    """
    d = J("supp_paper_curve_PAPER.json")
    cls = {r["seq"]: r["cls"] for r in J("supp_opquality_PAPER.json")["per_sequence"]}
    ceiling = J("saturation_RECIPE512_ctc53.json")["ceiling_pct"]

    fig, ax = plt.subplots(1, 6, figsize=(ns.W2, 2.05), sharex=True, sharey=True)
    for a, c in zip(ax, CLASSES):
        for q in QPS:
            xs, ys = [], []
            for o in d["op_points"]:
                if o["qp"] != q or not o.get("per_sequence"):
                    continue
                v = [r for r in o["per_sequence"] if cls.get(r["seq"]) == c]
                if not v:
                    continue
                xs.append(float(np.mean([r["db_vs_uf"] for r in v])))
                ys.append(float(np.mean([r["saving_pct_vs_release"] for r in v])))
            o = np.argsort(xs)
            a.plot(np.array(xs)[o], np.array(ys)[o], marker="o", ms=2.4, lw=1.0,
                   color=QCOL[q], label=f"q{q}")
        a.axhline(ceiling, ls=":", lw=0.6, color=ns.INK2)
        a.set_title(nice(c), fontsize=6.5, color=CCOL[c], loc="left")
        a.set_xlim(0, 0.62)
        a.set_xticks([0.0, 0.2, 0.4, 0.6])
        a.set_ylim(-3, 44)
    ax[0].set_ylabel("compute saved (%)")
    ax[0].legend(loc="lower right", fontsize=4.6, handlelength=1.0,
                 labelspacing=0.25, borderpad=0.1)
    ax[0].text(0.02, ceiling + 1.2, "ceiling", fontsize=4.8, color=ns.INK2)
    fig.subplots_adjust(bottom=0.26, wspace=0.16)
    fig.supxlabel("quality given up (dB), mean over the sequences of the class",
                  fontsize=6.5, y=0.075)
    for a, L in zip(ax, "abcdef"):
        ns.panel(a, L, dx=-0.20, dy=1.24)
    save(fig, "res_frontier.png")


# ---------------------------------------------------------- per sequence
def spread():
    """results/supp_per_sequence_PAPER_b0*.json, released-decoder denominator."""
    b010 = J("supp_per_sequence_PAPER_b010.json")
    budgets = [(0.10, "b010"), (0.15, "b015"), (0.20, "b020"),
               (0.25, "b025"), (0.30, "b030")]

    fig, ax = plt.subplots(1, 2, figsize=(ns.W1, 2.35))

    a = ax[0]
    rng = np.random.default_rng(0)
    data = []
    for i, q in enumerate(QPS):
        row = next(r for r in b010["rows"] if r["qp"] == q)
        v = np.array([s["saving_pct_vs_release"] for s in row["per_sequence"]])
        data.append(v)
        a.scatter(i + rng.uniform(-0.16, 0.16, v.size), v, s=2.2,
                  color=QCOL[q], alpha=0.55, linewidths=0)
    bp = a.boxplot(data, positions=range(len(QPS)), widths=0.55,
                   showfliers=False, patch_artist=False)
    for part in ("boxes", "whiskers", "caps", "medians"):
        for ln in bp[part]:
            ln.set_color(ns.INK2)
            ln.set_linewidth(0.6)
    a.axhline(0, ls="--", lw=0.6, color=ns.VERM)
    a.set_xticks(range(len(QPS)), [f"q{q}" for q in QPS])
    a.set_ylabel("compute saved (%)")
    a.set_title("53 sequences, 0.1 dB pooled", fontsize=6, color=ns.INK2,
                loc="left")
    ns.panel(a, "a", dx=-0.24)

    b = ax[1]
    for q in QPS:
        xs, ys, lo, hi = [], [], [], []
        for t, tag in budgets:
            d = J(f"supp_per_sequence_PAPER_{tag}.json")
            row = next((r for r in d["rows_vs_release"] if r["qp"] == q), None)
            if row is None:
                continue
            xs.append(t)
            ys.append(row["mean"])
            # The distribution is left-skewed once most sequences sit at the
            # ceiling, so the mean can fall below p25; clip rather than let
            # matplotlib refuse a negative whisker.
            lo.append(max(0.0, row["mean"] - row["p25"]))
            hi.append(max(0.0, row["p75"] - row["mean"]))
        b.errorbar(xs, ys, yerr=[lo, hi], marker="o", ms=2.6, lw=1.0,
                   elinewidth=0.6, capsize=1.5, color=QCOL[q], label=f"q{q}")
    b.set_xlabel("budget (dB, pooled)")
    b.set_ylabel("compute saved (%)")
    b.set_title("mean, with the quartiles", fontsize=6, color=ns.INK2,
                loc="left")
    b.legend(loc="lower right", fontsize=5, handlelength=1.0,
             labelspacing=0.25, ncol=2, columnspacing=0.8)
    ns.panel(b, "b", dx=-0.24)
    fig.subplots_adjust(wspace=0.42)
    save(fig, "res_spread.png")


# ------------------------------------------------------------- exit shares
def exits():
    """results/supp_paper_curve_PAPER.json exit histograms."""
    d = J("supp_paper_curve_PAPER.json")
    by = {(o["qp"], round(o["target_db"], 3)): o for o in d["op_points"]}

    def shares(o):
        """Percentage of tiles at each exit, with the tie folded in.

        paper_curve.py takes an argmin over the full [tiles, K] table, and
        flexuf/eval.py fills the columns below the split depth with the decode
        at the split depth, so those columns are exactly tied and argmin
        returns the first of them. Bins 0, 1 and 2 are therefore one exit and
        the file's own bin 2 is empty; folding them is what makes this
        histogram mean the same thing as the one in
        results/supp_per_class_budgets.json, which clamps before counting.
        """
        h = np.array(o["hist"], float)
        h[2] += h[0] + h[1]
        h[0] = h[1] = 0.0
        return 100 * h / h.sum()

    fig, ax = plt.subplots(1, 2, figsize=(ns.W1, 2.2))

    a = ax[0]
    bottom = np.zeros(len(QPS))
    for e in (2, 3, 4, 5):
        v = np.array([shares(by[(q, 0.1)])[e] for q in QPS])
        a.bar(range(len(QPS)), v, bottom=bottom, width=0.66,
              color=ECOL[e - 1], edgecolor="white", linewidth=0.4,
              label=f"e{e}")
        bottom += v
    a.set_xticks(range(len(QPS)), [f"q{q}" for q in QPS])
    a.set_ylabel("share of tiles (%)")
    a.set_ylim(0, 100)
    a.set_title("0.1 dB, by rate", fontsize=6, color=ns.INK2, loc="left")
    a.legend(loc="upper center", bbox_to_anchor=(0.5, -0.14), ncol=4,
             fontsize=5, handlelength=0.9, columnspacing=0.9)
    ns.panel(a, "a", dx=-0.24)

    b = ax[1]
    # 0.05 dB is below the floor at q63 and 0.5 dB is past saturation; the
    # curve file writes no histogram for either, which is the honest record of
    # an operating point that does not exist.
    tg = [t for t in (0.05, 0.1, 0.15, 0.2, 0.25, 0.3)
          if by[(63, t)].get("hist")]
    bottom = np.zeros(len(tg))
    for e in (2, 3, 4, 5):
        v = np.array([shares(by[(63, t)])[e] for t in tg])
        b.bar(range(len(tg)), v, bottom=bottom, width=0.66,
              color=ECOL[e - 1], edgecolor="white", linewidth=0.4)
        bottom += v
    b.set_xticks(range(len(tg)), ["%g" % t for t in tg])
    b.set_xlabel("budget (dB, pooled)")
    b.set_ylim(0, 100)
    b.set_title("q63, by budget", fontsize=6, color=ns.INK2, loc="left")
    ns.panel(b, "b", dx=-0.20)
    fig.subplots_adjust(wspace=0.34)
    save(fig, "res_exits.png")


# ------------------------------------------------------- rate and quality
def rd():
    """results/supp_opquality_PAPER.json, the 0.1 dB per-frame operating point."""
    d = J("supp_opquality_PAPER.json")
    rows = sorted(d["rows"], key=lambda r: r["bpp"])
    bpp = [r["bpp"] for r in rows]

    fig, ax = plt.subplots(1, 2, figsize=(ns.W1, 2.2))
    a = ax[0]
    a.plot(bpp, [r["psnr_released"] for r in rows], marker="o", ms=3,
           lw=1.1, color=ns.INK2, label="released decoder")
    a.plot(bpp, [r["psnr_routed"] for r in rows], marker="s", ms=3,
           lw=1.1, ls="--", color=ns.VERM, label="ours, 0.1 dB budget")
    a.set_xlabel("bpp")
    a.set_ylabel("PSNR (dB)")
    a.legend(loc="lower right", fontsize=5, handlelength=1.4)
    a.set_title("the two curves overlap", fontsize=6, color=ns.INK2,
                loc="left")
    ns.panel(a, "a", dx=-0.26)

    b = ax[1]
    x = np.arange(len(rows))
    order = sorted(d["rows"], key=lambda r: r["qp"])
    b.bar(x - 0.19, [-r["psnr_delta"] for r in order], width=0.36,
          color=ns.BLUE, label="PSNR")
    b.bar(x + 0.19,
          [r["ms_ssim_db_released"] - r["ms_ssim_db_routed"] for r in order],
          width=0.36, color=ns.ORANGE, label="MS-SSIM, in dB")
    b.set_xticks(x, [f"q{r['qp']}" for r in order])
    b.set_ylabel("quality given up (dB)")
    b.legend(loc="upper left", fontsize=5, handlelength=0.9)
    b.set_title("the same loss, two metrics", fontsize=6, color=ns.INK2,
                loc="left")
    ns.panel(b, "b", dx=-0.26)
    fig.subplots_adjust(wspace=0.42)
    save(fig, "res_rd.png")


# ---------------------------------------------------------------- the tail
def tail():
    """results/supp_opquality_PAPER.json, per-tile penalties at 0.1 dB."""
    d = J("supp_opquality_PAPER.json")

    fig, ax = plt.subplots(1, 2, figsize=(ns.W1, 2.2))
    a = ax[0]
    for q in QPS:
        t = d["tail_by_exit"][str(q)]
        es = [e for e in (2, 3, 4, 5) if t.get(str(e))]
        a.plot(es, [t[str(e)]["mean_db"] for e in es], marker="o", ms=2.6,
               lw=1.0, color=QCOL[q], label=f"q{q}")
        a.plot(es, [t[str(e)]["p95_db"] for e in es], marker="^", ms=2.4,
               lw=0.8, ls=":", color=QCOL[q])
    a.set_xticks([2, 3, 4, 5], ["e2", "e3", "e4", "e5"])
    a.set_xlabel("exit the tile took")
    a.set_ylabel("tile penalty (dB)")
    a.set_title("mean (solid), p95 (dotted)", fontsize=6, color=ns.INK2,
                loc="left")
    a.legend(loc="upper right", fontsize=5, ncol=2, handlelength=1.0,
             columnspacing=0.8, labelspacing=0.25)
    ns.panel(a, "a", dx=-0.26)

    b = ax[1]
    x = np.arange(len(QPS))
    pic = [d["tail_by_padding"][str(q)]["no_padding"]["mean_db"] for q in QPS]
    pad = [d["tail_by_padding"][str(q)]["some_padding"]["mean_db"] for q in QPS]
    pic95 = [d["tail_by_padding"][str(q)]["no_padding"]["p95_db"] for q in QPS]
    pad95 = [d["tail_by_padding"][str(q)]["some_padding"]["p95_db"] for q in QPS]
    b.bar(x - 0.19, pic, width=0.36, color=ns.GREEN, label="picture only")
    b.bar(x + 0.19, pad, width=0.36, color=ns.VERM, label="contains padding")
    b.plot(x - 0.19, pic95, "_", ms=6, mew=1.0, color=ns.INK)
    b.plot(x + 0.19, pad95, "_", ms=6, mew=1.0, color=ns.INK)
    b.set_xticks(x, [f"q{q}" for q in QPS])
    b.set_ylabel("tile penalty (dB)")
    b.legend(loc="upper right", fontsize=5, handlelength=0.9)
    b.set_title("bars mean, ticks p95", fontsize=6, color=ns.INK2, loc="left")
    ns.panel(b, "b", dx=-0.26)
    fig.subplots_adjust(wspace=0.42)
    save(fig, "res_tail.png")


if __name__ == "__main__":
    OUT.mkdir(parents=True, exist_ok=True)
    grid()
    frontier()
    spread()
    exits()
    rd()
    tail()
