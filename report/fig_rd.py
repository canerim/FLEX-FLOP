"""Rate-distortion in the form the field reads, plus the axis we actually move.

The first two panels are the plot every learned-compression paper opens with:
bits per pixel against PSNR, one curve per method, a zoom inset on the region
where the curves separate. Ours sit slightly BELOW the released decoder by
construction -- we spend a fixed quality budget to buy compute -- so an RD plot
alone would read as a loss, and putting only that on a slide would be
misleading.

The third panel carries the axis the work is about. Same abscissa, same
sequence of operating points, and on the ordinate what a decode costs. The
three panels together say the whole thing: the rate is identical, the quality
falls by the budget, and the arithmetic falls by a third.

Same checkpoint everywhere: RECIPE512 epoch 9, 256 px tiles.
"""
import json, sys
from pathlib import Path
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.axes_grid1.inset_locator import inset_axes, mark_inset
sys.path.insert(0, str(Path(__file__).resolve().parent))
import style as S

S.setup()
OUT = Path(__file__).resolve().parent / "fig"
RES = Path(__file__).resolve().parent.parent / "flexplus" / "results"
QPS = [0, 16, 32, 48, 63]
DECGMAC = 454.0

# Budgets to draw, and how. The released decoder is the anchor; ours are the
# same latents decoded less far.
BUD = [(0.05, "#d62728", "--", "^"), (0.10, "#1f77b4", "-", "s"),
       (0.15, "#ff7f0e", "-", "o"), (0.20, "#2ca02c", "--", "x"),
       (0.30, "#8c564b", "--", "v")]


def load(f):
    p = RES / f
    return json.loads(p.read_text()) if p.exists() else None


def series(d):
    """{budget: (bpp[], psnr[], saving[])} plus the released anchor.

    The deepest exit is kept beside it. It is the ladder run to full depth,
    which carries no adapter and is bit-exact with the released decoder, so
    the two curves should sit on top of each other -- and a reader who cannot
    see that has to take it on trust.
    """
    rel_b, rel_p, deep_p = [], [], []
    by = {}
    for r in d["rows"]:
        if r.get("mode") == "per_image" or not r.get("budget_reachable"):
            continue
        by.setdefault(r["budget_db"], {})[r["qp"]] = r
    anchor_done = set()
    for b in sorted(by):
        for q in QPS:
            if q in by[b] and q not in anchor_done:
                rel_b.append(by[b][q]["bpp"])
                rel_p.append(by[b][q]["psnr_release_rgb"])
                deep_p.append(by[b][q]["psnr_deepest_rgb"])
                anchor_done.add(q)
    o = np.argsort(rel_b)
    rel = (np.array(rel_b)[o], np.array(rel_p)[o], np.array(deep_p)[o])
    out = {}
    for b in sorted(by):
        qs = [q for q in QPS if q in by[b]]
        if len(qs) < 3:
            continue
        bp = np.array([by[b][q]["bpp"] for q in qs])
        ps = np.array([by[b][q]["psnr_release_rgb"] - by[b][q]["db_vs_uf"]
                       for q in qs])
        sv = np.array([by[b][q].get("saving_pct_measured",
                                    by[b][q]["saving_pct_vs_release"])
                       for q in qs])
        o = np.argsort(bp)
        out[b] = (bp[o], ps[o], sv[o])
    return rel, out


def frame(ax):
    for sp in ax.spines.values():
        sp.set_visible(True); sp.set_linewidth(0.7); sp.set_color("#444444")
    ax.grid(True, color="#e2e2e2", lw=0.45, zorder=0)
    ax.set_axisbelow(True)
    ax.tick_params(labelsize=6.4, length=2.4, width=0.6)


def rd_panel(ax, d, title, zoom=None):
    rel, ser = series(d)
    ax.plot(rel[0], rel[1], "-", color="#111111", lw=1.5, marker="D", ms=3.4,
            markerfacecolor="#111111", zorder=8, label="DCVC-UF (released)")
    ax.plot(rel[0], rel[2], "-", color="#9a9a9a", lw=2.6, alpha=0.55,
            zorder=7, label="FLEX, deepest exit (bit-exact)")
    for b, c, ls, mk in BUD:
        if b not in ser:
            continue
        bp, ps, _ = ser[b]
        ax.plot(bp, ps, ls, color=c, lw=1.15, marker=mk, ms=3.2,
                markerfacecolor="none", markeredgewidth=0.9, zorder=5,
                label=f"FLEX,  {b:.2f} dB budget")
    frame(ax)
    ax.set_xlabel("Bit-rate (bpp)", fontsize=7.6)
    ax.set_ylabel("PSNR (dB)  ↑", fontsize=7.6)
    ax.set_title(title, fontsize=8.6, fontweight="bold", pad=4)
    ax.legend(fontsize=5.2, loc="lower right", frameon=True, framealpha=0.95,
              edgecolor="#999999", handlelength=1.6, borderpad=0.45,
              labelspacing=0.30)
    if zoom:
        x0, x1, y0, y1 = zoom
        axi = inset_axes(ax, width="36%", height="30%", loc="upper left",
                         borderpad=1.4)
        axi.plot(rel[0], rel[2], "-", color="#9a9a9a", lw=2.4, alpha=0.55,
                 zorder=7)
        axi.plot(rel[0], rel[1], "-", color="#111111", lw=1.3, marker="D",
                 ms=2.8, markerfacecolor="#111111", zorder=8)
        for b, c, ls, mk in BUD:
            if b not in ser:
                continue
            bp, ps, _ = ser[b]
            axi.plot(bp, ps, ls, color=c, lw=1.0, marker=mk, ms=2.6,
                     markerfacecolor="none", markeredgewidth=0.8, zorder=5)
        axi.set_xlim(x0, x1); axi.set_ylim(y0, y1)
        frame(axi)
        axi.tick_params(labelsize=4.8, length=1.6)
        mark_inset(ax, axi, loc1=2, loc2=4, fc="none", ec="#777777",
                   lw=0.6, ls=(0, (2.4, 1.8)))
    return rel, ser


def cost_panel(ax, d, title):
    rel, ser = series(d)
    ax.plot(rel[0], np.full(rel[0].size, DECGMAC), "-", color="#111111",
            lw=1.5, marker="D", ms=3.4, markerfacecolor="#111111", zorder=8,
            label="DCVC-UF (released)")
    for b, c, ls, mk in BUD:
        if b not in ser:
            continue
        bp, _, sv = ser[b]
        ax.plot(bp, DECGMAC * (1 - sv / 100), ls, color=c, lw=1.15, marker=mk,
                ms=3.2, markerfacecolor="none", markeredgewidth=0.9, zorder=5,
                label=f"FLEX,  {b:.2f} dB budget")
    frame(ax)
    ax.set_xlabel("Bit-rate (bpp)", fontsize=7.6)
    ax.set_ylabel("decoder GMAC per 1080p frame  ↓", fontsize=7.6)
    ax.set_title(title, fontsize=8.6, fontweight="bold", pad=4)
    ax.legend(fontsize=5.2, loc="lower right", frameon=True, framealpha=0.95,
              edgecolor="#999999", handlelength=1.6, borderpad=0.45,
              labelspacing=0.30)


def main():
    K = load("kodak_rd.json") or load("kodak_sweep.json")
    C = load("clic_rd.json") or load("clic_sweep.json")
    if K is None and C is None:
        print("  veri yok"); return
    # Only the panels there is data for. CLIC can still be running.
    have = [("rd", K, "Kodak (PSNR)")] if K else []
    if C:
        have.append(("rd", C, "CLIC 2020 professional (PSNR)"))
    if K:
        have.append(("cost", K, "Kodak (decoder complexity)"))
    n = len(have)
    fig = plt.figure(figsize=(3.85 * n, 3.35))
    w = 0.255 * 3 / n if n else 0.255
    left, gap = 0.055 / (n / 3), 0.327 * 3 / n
    axes = [fig.add_axes([left + i * gap, 0.155, w * n / 3 * (3 / n) * 0.98,
                          0.735]) for i in range(n)]
    for ax, (kind, d, title) in zip(axes, have):
        if kind == "rd":
            # Put the inset where the curves actually separate. They differ by
            # the budget -- a tenth of a decibel on a twelve-decibel axis -- so
            # a zoom chosen by eye shows five lines on top of each other, which
            # is what the first draft did. Pick the rate with the widest spread
            # and window tightly around it.
            _rel, ser = series(d)
            common = set.intersection(*[set(np.round(ser[b][0], 6))
                                        for b in ser]) if ser else set()
            best, spread = None, -1.0
            for x in sorted(common):
                vs = [float(ser[b][1][np.argmin(np.abs(ser[b][0] - x))])
                      for b in ser]
                vs.append(float(_rel[1][np.argmin(np.abs(_rel[0] - x))]))
                if max(vs) - min(vs) > spread:
                    spread, best = max(vs) - min(vs), x
            if best is None:
                rd_panel(ax, d, title)
            else:
                vs = [float(ser[b][1][np.argmin(np.abs(ser[b][0] - best))])
                      for b in ser]
                vs.append(float(_rel[1][np.argmin(np.abs(_rel[0] - best))]))
                pad = max(0.05, 0.28 * spread)
                rd_panel(ax, d, title,
                         zoom=(best * 0.955, best * 1.045,
                               min(vs) - pad, max(vs) + pad))
        else:
            cost_panel(ax, d, title)
    fig.savefig(OUT / "fig19_rd.pdf"); fig.savefig(OUT / "fig19_rd.png")
    print("  fig19_rd yazildi")


if __name__ == "__main__":
    main()
