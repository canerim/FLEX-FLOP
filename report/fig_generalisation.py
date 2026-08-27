"""Three test sets, and the law that predicts the fourth.

Kodak and CLIC are where an image-compression reader looks first, and the CTC
intra frames the paper reports are neither. Measuring all three at once turns
a question about generalisation into a measurement, and it turns out the three
are not three answers but one: what a budget buys is set almost entirely by
how much of it survives the tiling floor.

a and b are the operating curves — mean saving against the quality budget, one
line per rate. c is what the guarantee can promise on each set: the fraction
of images whose floor already exceeds the budget, and which therefore cannot
be served at any allocation. d pools every rate of every set against the room
the budget leaves, and fits one line through them.
"""
import json, sys
from pathlib import Path
import numpy as np
import matplotlib.pyplot as plt
sys.path.insert(0, str(Path(__file__).resolve().parent))
import style as S

S.setup()
OUT = Path(__file__).resolve().parent / "fig"
RES = Path(__file__).resolve().parent.parent / "flexplus" / "results"
UF = Path.home() / "FLEX-UF"
QPS = [0, 16, 32, 48, 63]
COL = {q: plt.cm.viridis(0.08 + 0.80 * i / 4) for i, q in enumerate(QPS)}


def load(f):
    p = RES / f
    return json.loads(p.read_text()) if p.exists() else None


def curves(d, mode=None):
    """{qp: (budgets, savings)} for the global-lambda rows."""
    out = {}
    for r in d["rows"]:
        if mode == "per_image":
            if r.get("mode") != "per_image":
                continue
            v = r["saving_pct_vs_release"]
        else:
            if r.get("mode") == "per_image" or not r.get("budget_reachable"):
                continue
            v = r.get("saving_pct_measured", r.get("saving_pct_vs_release"))
        out.setdefault(r["qp"], []).append((r["budget_db"], v, r))
    for q in out:
        out[q].sort(key=lambda t: t[0])
    return out


def panel_curves(ax, d, title, letter):
    c = curves(d)
    for q in QPS:
        if q not in c:
            continue
        b = [x[0] for x in c[q]]; v = [x[1] for x in c[q]]
        ax.plot(b, v, "-o", color=COL[q], ms=2.6, lw=1.15, zorder=3)
        ax.text(b[-1] + 0.006, v[-1], f"q{q}", fontsize=5.8, color=COL[q],
                va="center")
    ax.set_xlabel("quality budget (dB)")
    ax.set_ylabel("mean MAC saving (%)")
    ax.set_title(title, fontsize=7.4, pad=3.5, color=S.INK)
    ax.set_xlim(0.03, 0.335)
    S.ygrid(ax)
    S.panel(ax, letter, dx=-0.175)


def main():
    K = load("kodak_sweep.json")
    C = load("clic_sweep.json")
    if K is None or C is None:
        print("  veri henuz yok:",
              "kodak" if K is None else "", "clic" if C is None else "")
        return

    fig = plt.figure(figsize=(7.0, 5.0))
    ax1 = fig.add_axes([0.085, 0.585, 0.375, 0.345])
    ax2 = fig.add_axes([0.590, 0.585, 0.375, 0.345])
    ax3 = fig.add_axes([0.085, 0.085, 0.375, 0.345])
    ax4 = fig.add_axes([0.590, 0.085, 0.375, 0.345])

    panel_curves(ax1, K, f"Kodak, {K['n_images']} images", "a")
    panel_curves(ax2, C, f"CLIC 2020 professional, {C['n_images']} images", "b")

    # ---- c: what fraction cannot be served at all ------------------------
    for d, lab, ls in ((K, "Kodak", "-"), (C, "CLIC", "--")):
        pi = curves(d, "per_image")
        for q in (0, 32, 63):
            if q not in pi:
                continue
            b = [x[0] for x in pi[q]]
            f = [100 * x[2]["infeasible"] / x[2]["n"] for x in pi[q]]
            ax3.plot(b, f, ls, color=COL[q], lw=1.15, zorder=3,
                     marker="o" if ls == "-" else "s", ms=2.4)
    ax3.set_xlabel("quality budget (dB)")
    ax3.set_ylabel("images the budget cannot reach (%)")
    ax3.set_xlim(0.03, 0.335)
    ax3.set_title("the floor decides what a guarantee can promise",
                  fontsize=7.4, pad=3.5, color=S.INK)
    ax3.text(0.30, 0.90, "solid Kodak\ndashed CLIC", transform=ax3.transAxes,
             fontsize=5.8, color=S.MUTED, va="top", ha="left", linespacing=1.3)
    S.ygrid(ax3); S.panel(ax3, "c", dx=-0.175)

    # ---- d: room against saving, every rate of every set -----------------
    xs, ys, cs, labs = [], [], [], []
    for d, lab, mk in ((K, "Kodak", "o"), (C, "CLIC", "s")):
        pi = curves(d, "per_image")
        for q in QPS:
            for b, v, r in pi.get(q, []):
                room = b - r["mean_floor_db"]
                if room <= 0:
                    continue
                xs.append(room); ys.append(v); cs.append(COL[q])
                labs.append(mk)
    ctc = load("guarantee_tiled_e9.json")
    pf = load("signalled_perframe_e9.json")
    if pf:
        for r in pf["rows"]:
            fl = float(np.mean([z["floor_db"] for z in r["per_frame"]]))
            xs.append(0.1 - fl); ys.append(r["saving_pct_measured"])
            cs.append(COL[r["qp"]]); labs.append("^")
    xs, ys = np.array(xs), np.array(ys)
    for mk in ("o", "s", "^"):
        m = np.array([l == mk for l in labs])
        if m.sum():
            ax4.scatter(xs[m], ys[m], s=11, marker=mk,
                        c=[cs[i] for i in np.where(m)[0]], zorder=3,
                        linewidths=0.4, edgecolor="white")
    A = np.polyfit(xs, ys, 1)
    xx = np.linspace(xs.min(), xs.max(), 50)
    ax4.plot(xx, np.polyval(A, xx), color=S.INK, lw=0.9,
             ls=(0, (3.0, 2.0)), zorder=2)
    r = float(np.corrcoef(xs, ys)[0, 1])
    ax4.text(0.04, 0.94, f"$r = {r:.3f}$   ({xs.size} points)",
             transform=ax4.transAxes, fontsize=6.6, va="top",
             fontweight="bold")
    ax4.text(0.04, 0.845, "▲ CTC intra   ● Kodak   ■ CLIC",
             transform=ax4.transAxes, fontsize=5.8, color=S.MUTED, va="top")
    ax4.set_xlabel("room the budget leaves:  budget − tiling floor (dB)")
    ax4.set_ylabel("mean MAC saving (%)")
    ax4.set_title("one law across three sets and five rates",
                  fontsize=7.4, pad=3.5, color=S.INK)
    S.ygrid(ax4); S.panel(ax4, "d", dx=-0.175)

    fig.savefig(OUT / "fig18_generalisation.pdf")
    fig.savefig(OUT / "fig18_generalisation.png")
    print(f"  fig18_generalisation yazildi   (d: r={r:.3f}, n={xs.size})")


if __name__ == "__main__":
    main()
