"""Three budgets, three surfaces: what each one buys, frame by frame.

A saving quoted at a budget is one number standing for 265 measurements --
53 CTC frames at five rates -- and the number hides the two things worth
seeing. First, the spread: content decides how much depth a frame can give
up, and the spread across frames is wider than the spread across rates.
Second, saturation. Each frame has a ceiling, the saving when every cell
already sits at the shallowest exit it may take, and a rate is saturated
when its whole row rests on that ceiling. That is a surface touching a
surface, which a table cannot show and a 3D plot can.

Same axes, same z range and the same frame ordering in all three panels, so
the panels differ only in the budget. The ordering is by mean saving at the
perfect budget, fixed once and reused, so a ridge in one panel is the same
frame in the others.

CPU only, from the dumped tables. Nothing under ~/FLEX-UF is touched.
"""
import json, sys
from pathlib import Path
import numpy as np
import matplotlib.pyplot as plt
from matplotlib import cm
from matplotlib.colors import Normalize, LightSource
sys.path.insert(0, str(Path(__file__).resolve().parent))
sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "flexplus"))
import style as S
from budget_search import Sweep, QPS
from router_fit import dilated_blocks

S.setup()
OUT = Path(__file__).resolve().parent / "fig"
RES = Path(__file__).resolve().parent.parent / "flexplus" / "results"


def frame_savings(s, q, lam):
    """Saving of every frame separately, not their pooled cost."""
    frl, sel, shapes = s.per[q]
    from router_fit import allocate
    k = allocate(s.M[sel], s.ck, lam, s.j)
    out = []
    for f, sh in zip(frl, shapes):
        mp = k[f].reshape(*sh)
        frac = dilated_blocks(mp, s.cf, s.C.blocks_per_exit, s.K)
        c = (s.C.SHARE_UPSAMPLE
             + s.C.SHARE_TRUNK * frac / s.C.N_TRUNK_BLOCKS
             + s.C.adapter[mp.reshape(-1)].mean() + s.C.SHARE_HEAD)
        out.append(100 * (1 - c))
    return np.array(out)


def lam_for(s, q, b):
    if s.ceiling_db(q) <= b:
        return 1e9, True
    lo, hi = 0.0, 1e-8
    while s.at(q, hi)[0] <= b and hi < 1e9:
        hi *= 4
    for _ in range(45):
        mid = 0.5 * (lo + hi)
        if s.at(q, mid)[0] <= b:
            lo = mid
        else:
            hi = mid
    return lo, False


def smooth_x(Z, n):
    """Interpolate along the rate axis for rendering only.

    The five rate points are the measurement; a surface drawn on five columns
    reads as a staircase and invites the eye to see steps that are an artefact
    of the sampling. Nothing between the columns is claimed.
    """
    xs = np.linspace(0, Z.shape[1] - 1, n)
    out = np.empty((Z.shape[0], n))
    for i in range(Z.shape[0]):
        out[i] = np.interp(xs, np.arange(Z.shape[1]), Z[i])
    return out, xs


def draw(label, kicker, note, Z, C, sat, agg, zlim, nfr):
    NX = 61
    Zs, xs = smooth_x(Z, NX)
    Cs, _ = smooth_x(C, NX)
    ys = np.arange(nfr)
    X, Y = np.meshgrid(xs, ys)

    fig = plt.figure(figsize=(3.42, 3.16))
    ax = fig.add_axes([-0.055, 0.135, 1.05, 0.755], projection="3d")

    ls = LightSource(azdeg=300, altdeg=58)
    norm = Normalize(zlim[0], zlim[1])
    fc = ls.shade(Zs, cmap=cm.viridis, norm=norm, vert_exag=0.6,
                  blend_mode="soft")
    ax.plot_surface(X, Y, Zs, facecolors=fc, linewidth=0, antialiased=True,
                    shade=False, rstride=1, cstride=1, zorder=3)

    # The ceiling is one number, not a surface: when every cell takes the
    # shallowest exit it may, the map is uniform, so there is no band and the
    # adapter is the same everywhere -- content drops out. Drawn last and
    # translucent, as glass over the surface, with its rim picked out so it
    # is visible where the surface sits far below it.
    zc = float(C.mean())
    ax.plot_surface(X, Y, np.full_like(X, zc), color=(0.55, 0.57, 0.60),
                    alpha=0.13, linewidth=0, antialiased=True, shade=False,
                    zorder=8)
    rim = ([(-0.25, 0), (len(QPS) - 0.75, 0), (len(QPS) - 0.75, nfr - 1),
            (-0.25, nfr - 1), (-0.25, 0)])
    ax.plot([p[0] for p in rim], [p[1] for p in rim], [zc] * len(rim),
            color=(0.52, 0.54, 0.57), lw=0.55, ls=(0, (2.4, 1.8)), zorder=9)

    # A saturated rate is a row lying exactly on its ceiling. Draw it as a
    # single line along that row so the touching is legible, not inferred.
    for i, t in enumerate(sat):
        if t:
            ax.plot(np.full(nfr, i), ys, Z[:, i] + 0.05, color=S.VERM,
                    lw=1.15, zorder=6, solid_capstyle="round")

    ax.set_xticks(range(len(QPS)))
    ax.set_xticklabels([f"q{q}" for q in QPS])
    ax.set_yticks([0, nfr // 2, nfr - 1])
    ax.set_yticklabels(["1", str(nfr // 2 + 1), str(nfr)])
    ax.set_zlim(*zlim)
    from matplotlib.ticker import MaxNLocator
    ax.zaxis.set_major_locator(MaxNLocator(5, integer=True))
    ax.set_xlim(-0.25, len(QPS) - 0.75); ax.set_ylim(0, nfr - 1)
    ax.set_xlabel("rate point", labelpad=-4)
    ax.set_ylabel("CTC frame, ranked", labelpad=-4)
    ax.tick_params(labelsize=6.0, pad=-2.0, length=0)
    ax.tick_params(axis="z", pad=-0.5, labelsize=5.8)
    ax.view_init(elev=21, azim=-58)
    ax.set_box_aspect((1.20, 1.0, 0.64), zoom=1.12)
    for axis in (ax.xaxis, ax.yaxis, ax.zaxis):
        axis.pane.fill = False
        axis.pane.set_edgecolor("none")
        axis.line.set_color(S.MUTED); axis.line.set_linewidth(0.5)
    ax.grid(False)
    for axis in (ax.xaxis, ax.yaxis):
        axis._axinfo["grid"]["linewidth"] = 0
    ax.zaxis._axinfo["grid"].update(color=(0.87, 0.87, 0.88), linewidth=0.4)

    fig.text(0.035, 0.982, label, fontsize=9.6, fontweight="bold", va="top",
             color=S.INK)
    fig.text(0.035, 0.913, kicker, fontsize=6.6, va="top", color=S.VERM,
             fontweight="bold")
    fig.text(0.035, 0.870, note, fontsize=6.2, va="top", color=S.INK2)
    fig.text(0.968, 0.982, "MAC saving, per frame", fontsize=6.0, va="top",
             ha="right", color=S.MUTED)
    fig.text(0.968, 0.940, f"glass: the ceiling, {float(C.mean()):.2f}%",
             fontsize=5.6, va="top", ha="right", color=S.MUTED)

    y0 = 0.082
    fig.text(0.035, y0, "mean saving", fontsize=6.0, color=S.MUTED,
             va="center")
    x = 0.335
    for q, v, t in zip(QPS, agg, sat):
        fig.text(x, y0, f"q{q}", fontsize=5.6, color=S.MUTED, ha="center",
                 va="center")
        fig.text(x, y0 - 0.046, f"{v:.1f}" + ("●" if t else ""),
                 fontsize=6.8, ha="center", va="center",
                 color=S.VERM if t else S.INK,
                 fontweight="bold" if t else "normal")
        x += 0.130
    fig.text(0.035, y0 - 0.046, "● on the ceiling", fontsize=6.0,
             color=S.MUTED, va="center")
    return fig


def main():
    s = Sweep(str(RES / "cells_ctc64_e8.npz"), 2)
    perfect = s.ceiling_db(0)
    nfr = len(s.per[0][0])
    print(f"  tam butce {perfect:.9f} dB, {nfr} kare, {len(QPS)} oran",
          flush=True)

    ceil = np.column_stack([frame_savings(s, q, 1e9) for q in QPS])
    budgets = [(0.10, "0.10 dB", "the reported budget",
                "no rate has reached its ceiling; every frame still has "
                "depth to give up"),
               (perfect, "0.121317 dB", "the perfect budget",
                "exactly one rate rests on the ceiling, and it is the "
                "lowest one"),
               (0.20, "0.20 dB", "past the point of return",
                "three rates are pinned to the ceiling; their extra "
                "tolerance buys nothing")]
    Zs, sats, aggs = [], [], []
    for b, *_ in budgets:
        cols, sat, agg = [], [], []
        for q in QPS:
            lam, t = lam_for(s, q, b)
            cols.append(frame_savings(s, q, lam))
            sat.append(t)
            agg.append(s.at(q, lam)[1])
        Zs.append(np.column_stack(cols)); sats.append(sat); aggs.append(agg)

    lo = min(Z.min() for Z in Zs); hi = max(ceil.max(), max(Z.max() for Z in Zs))
    pad = 0.06 * (hi - lo)
    zlim = (lo - pad, hi + pad)

    # Rank frames once, at the perfect budget, and reuse the order everywhere:
    # a ridge must be the same frame in all three panels or the panels cannot
    # be compared at all.
    order = np.argsort(Zs[1].mean(1))
    names = ["fig15a_frames_010", "fig15b_frames_perfect",
             "fig15c_frames_020"]
    dump = {}
    for (b, label, kicker, note), Z, sat, agg, name in zip(
            budgets, Zs, sats, aggs, names):
        fig = draw(label, kicker, note, Z[order], ceil[order], sat, agg,
                   zlim, nfr)
        fig.savefig(OUT / f"{name}.pdf"); fig.savefig(OUT / f"{name}.png")
        plt.close(fig)
        dump[f"{b:.6f}"] = {
            "mean_saving_pct": agg, "saturated": [bool(t) for t in sat],
            "frame_min_pct": Z.min(0).tolist(),
            "frame_max_pct": Z.max(0).tolist()}
        print(f"  {name}: " + "  ".join(
            f"q{q}:{v:.2f}{'*' if t else ''}"
            for q, v, t in zip(QPS, agg, sat))
            + f"   kare araligi {Z.min():.1f}-{Z.max():.1f}%", flush=True)
    (RES / "budget3d.json").write_text(json.dumps(
        {"ckpt_epoch": 8, "cell_px": 64, "split_depth": 2, "n_frames": nfr,
         "perfect_db": perfect, "frame_order": order.tolist(),
         "ceiling_mean_pct": ceil.mean(0).tolist(), "budgets": dump},
        indent=2))
    print("  yazildi results/budget3d.json")


if __name__ == "__main__":
    main()
