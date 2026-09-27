"""Editable publication figures: measured evidence, tensor geometry, execution.

CPU only. Run paper_refresh_data.py first. Raster content is restricted to the
archived luma thumbnail; diagrams, plots, labels and keys remain vector objects.
"""
from __future__ import annotations

import argparse
import csv
import datetime
import hashlib
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib import colors
from matplotlib.backends.backend_pdf import PdfPages
from matplotlib.lines import Line2D
from matplotlib.patches import Rectangle, Polygon, FancyArrowPatch, PathPatch
from matplotlib.path import Path as MPath
from matplotlib.text import Text
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
# The identical script also ships inside the stand-alone Overleaf subtree.
BUNDLED = (ROOT / "data/refresh20260927/analysis.json").exists()
DATA = ROOT / ("data/refresh20260927" if BUNDLED else "paper/data/refresh20260927")
OUT = ROOT / ("figs/refresh20260927" if BUNDLED else "paper/figures/refresh20260927")
THUMBNAILS = DATA / "source_thumbnails.npz" if BUNDLED else ROOT / "flexplus/results/eval_rules_thumbs.npz"
MM = 1 / 25.4
INK, MUTED, GRID = "#183342", "#5A6C75", "#E2E9EB"
BLUE, ORANGE, TEAL, GREY = "#007F86", "#BB7534", "#63527C", "#8999A3"
DEPTH_ALL = ["#DCECEB", "#A8D4CE", "#60B1A8", "#2B8D8C", "#286777", "#244457"]
DEPTH = DEPTH_ALL[2:]
COL = dict(oracle=TEAL, router=BLUE, dither=ORANGE, uniform=GREY)
MARK = dict(oracle="o", router="s", dither="^", uniform="D")
LABEL = dict(oracle="Source-informed search", router="Router", dither="Bayer dither", uniform="Uniform depth")
BUD = [.05, .1, .15, .2, .3, .5]
plt.rcParams.update({
    "font.family": "Liberation Sans", "font.size": 7,
    "axes.labelsize": 7, "axes.titlesize": 7.2, "xtick.labelsize": 6.3,
    "ytick.labelsize": 6.3, "legend.fontsize": 6.3, "axes.linewidth": .55,
    "axes.spines.top": False, "axes.spines.right": False,
    "xtick.major.width": .5, "ytick.major.width": .5,
    "xtick.major.size": 2.5, "ytick.major.size": 2.5,
    "lines.linewidth": 1.1, "lines.markersize": 3,
    "figure.facecolor": "white", "axes.facecolor": "white",
    "savefig.facecolor": "white", "text.color": INK,
    "axes.labelcolor": INK, "xtick.color": MUTED, "ytick.color": MUTED,
    "legend.frameon": False, "pdf.fonttype": 42, "ps.fonttype": 42,
    "svg.fonttype": "none", "svg.hashsalt": "flexuf-refresh-20260927",
    "axes.unicode_minus": True,
})
AUDIT, CAPTIONS = [], []
PDF_META = {"CreationDate": datetime.datetime(2026, 9, 27, tzinfo=datetime.timezone.utc),
            "ModDate": datetime.datetime(2026, 9, 27, tzinfo=datetime.timezone.utc),
            "Creator": "FLEX publication figure generator"}


def load():
    return json.loads((DATA / "analysis.json").read_text())


def text(ax, x, y, s, **kw):
    return ax.text(x, y, s, **(dict(va="center", ha="left") | kw))


def panel(ax, letter, title):
    ax.text(-.14, 1.065, letter, transform=ax.transAxes, fontsize=7, weight="bold", va="bottom")
    ax.set_title(title, loc="left", pad=9, fontsize=7)
    ax.grid(axis="y", color=GRID, lw=.45, zorder=0)
    ax.set_axisbelow(True)


def schematic(h):
    fig = plt.figure(figsize=(183 * MM, h * MM))
    ax = fig.add_axes([0, 0, 1, 1], xlim=(0, 183), ylim=(0, h))
    ax.axis("off")
    return fig, ax


def heading(ax, x, y, letter, title):
    text(ax, x, y, letter, weight="bold", fontsize=7)
    text(ax, x + 4.5, y, title, weight="bold", fontsize=7)


def arrow(ax, start, end, color=MUTED, lw=.8, curve=0, style="-", zorder=2):
    p = FancyArrowPatch(start, end, arrowstyle="-|>", mutation_scale=6,
                        connectionstyle=f"arc3,rad={curve}", color=color,
                        lw=lw, linestyle=style, zorder=zorder)
    ax.add_patch(p)
    return p


def tensor(ax, x, y, w, h, color="#e5edf2", layers=3, grid=4):
    for i in reversed(range(layers)):
        dx, dy = i * .8, i * .65
        ax.add_patch(Rectangle((x + dx, y + dy), w, h, facecolor=color,
                               edgecolor=MUTED, lw=.45, zorder=3))
    for i in range(1, grid):
        ax.plot([x + w * i / grid] * 2, [y, y + h], color="white", lw=.45, zorder=4)
        ax.plot([x, x + w], [y + h * i / grid] * 2, color="white", lw=.45, zorder=4)


def strip(ax, x, y, n, w=1.9, h=10, color=BLUE):
    for i in range(n):
        ax.add_patch(Rectangle((x + i * (w + .7), y), w, h,
                               facecolor=color, edgecolor="white", lw=.4))


def tilemap(ax, x, y, w, h, exits, threshold=None, outline=True):
    nh, nw = exits.shape
    for iy in range(nh):
        for ix in range(nw):
            k = int(exits[iy, ix])
            fill = DEPTH[k - 2] if threshold is None or k >= threshold else "white"
            ax.add_patch(Rectangle((x + ix * w / nw, y + (nh - iy - 1) * h / nh),
                                   w / nw, h / nh, facecolor=fill,
                                   edgecolor="white" if fill != "white" else GRID, lw=.4))
    if outline:
        ax.add_patch(Rectangle((x, y), w, h, fill=False, ec=MUTED, lw=.45))


def depth_key(ax, x, y):
    for i, (c, k) in enumerate(zip(DEPTH, [6, 8, 10, 12])):
        ax.add_patch(Rectangle((x + i * 16, y - 1), 2.5, 2.5, color=c, ec=MUTED, lw=.3))
        text(ax, x + 3.5 + i * 16, y, str(k), fontsize=6)
    text(ax, x + 60, y, "executed trunk blocks", fontsize=6)


def audit_and_save(fig, name, caption, book):
    fig.canvas.draw()
    renderer = fig.canvas.get_renderer()
    outside, text_boxes = [], []
    ignored = set()
    for ax in fig.axes:
        for axis, limits in [(ax.xaxis, ax.get_xlim()), (ax.yaxis, ax.get_ylim())]:
            lo, hi = sorted(limits)
            for tick in axis.get_major_ticks() + axis.get_minor_ticks():
                if not lo - 1e-9 <= tick.get_loc() <= hi + 1e-9:
                    ignored.update([id(tick.label1), id(tick.label2)])
    # Inspect labels actually in view; matplotlib retains off-axis tick artists.
    for obj in fig.findobj(Text):
        if id(obj) in ignored or not obj.get_visible() or not obj.get_text().strip():
            continue
        box = obj.get_window_extent(renderer)
        if box.width == 0 or box.height == 0:
            continue
        if obj.axes is not None and obj not in obj.axes.texts and obj not in [obj.axes.title, obj.axes._left_title,
                obj.axes.xaxis.label, obj.axes.yaxis.label]:
            # Tick labels can be outside the nominal limits and clipped away.
            if not obj.axes.bbox.expanded(1.8, 1.8).overlaps(box):
                continue
        if box.x0 < -1 or box.y0 < -1 or box.x1 > fig.bbox.width + 1 or box.y1 > fig.bbox.height + 1:
            outside.append(obj.get_text())
        text_boxes.append(dict(text=obj.get_text(), size_pt=obj.get_fontsize(),
                               bbox_px=list(box.bounds)))
    if outside:
        raise RuntimeError(f"Text outside {name}: {outside}")
    AUDIT.append(dict(figure=name, outside_canvas=outside, text=text_boxes,
                      width_mm=fig.get_figwidth() / MM, height_mm=fig.get_figheight() / MM))
    for ext in ("pdf", "svg", "png"):
        metadata = PDF_META if ext == "pdf" else {"Date": "2026-09-27T00:00:00Z"} if ext == "svg" else None
        fig.savefig(OUT / f"{name}.{ext}", dpi=400, metadata=metadata)
        if ext == "svg":
            p = OUT / f"{name}.{ext}"
            p.write_text("\n".join(line.rstrip() for line in p.read_text().splitlines()) + "\n")
    book.savefig(fig)
    CAPTIONS.append(dict(name=name, caption=caption))
    plt.close(fig)


def architecture(d, book):
    fig, ax = schematic(60)
    ex = d["example"]
    m = np.array(ex["rules"]["router"]["0.1"]["map"]).reshape(ex["grid"])
    heading(ax, 3, 56, "a", "Recover once; assign a stopping depth to each tile")
    tensor(ax, 5, 40, 13, 8, "#E7EEF2", grid=4)
    text(ax, 12, 36, "Latent", ha="center", fontsize=6.5)
    arrow(ax, (21, 44), (29, 44))
    strip(ax, 31, 40, 4, w=2.2, h=8)
    text(ax, 37, 36, "Upsample + stem", ha="center", fontsize=6.5)
    arrow(ax, (44, 44), (57, 44))
    tensor(ax, 60, 40, 14, 8, grid=4)
    text(ax, 67, 36, "384-channel features", ha="center", fontsize=6.5)
    arrow(ax, (78, 44), (88, 44), color=BLUE)
    text(ax, 106, 47, "Pool + QP → MLP", ha="center", fontsize=7, color=BLUE)
    text(ax, 106, 41, "Scores + control β", ha="center", fontsize=6.5)
    arrow(ax, (128, 44), (142, 44), color=BLUE)
    tilemap(ax, 147, 39, 19, 11.9, m)
    text(ax, 157, 35, "Depth map", ha="center", fontsize=6.5)
    ax.plot([3, 180], [32, 32], color=GRID, lw=.6)
    heading(ax, 3, 28, "b", "Continue only the surviving tiles")
    xs = [7, 39, 71, 103]
    for j, x in enumerate(xs):
        n = int((m >= j + 2).sum())
        tilemap(ax, x, 12, 20, 12.5, m, threshold=j + 2)
        text(ax, x + 10, 9, f"{n}/40 · depth {6+2*j}", ha="center", fontsize=6.5)
        if j < 3:
            arrow(ax, (x + 21, 18), (x + 30, 18))
        # Each exit writes its features to a common canvas; lanes stay separate.
        yy = 5.8 - j * .95
        ax.plot([x, x, 139], [12, yy, yy], color=DEPTH[j], lw=.9)
    ax.plot([139, 139], [2.95, 5.8], color=MUTED, lw=.7)
    arrow(ax, (139, 4.5), (146, 13), color=MUTED)
    tilemap(ax, 148, 13, 19, 11.9, m)
    text(ax, 157.5, 28, "Stitch → repair → head", ha="center", fontsize=6.5)
    text(ax, 157.5, 9, "Canvas → image", ha="center", fontsize=6.5)
    text(ax, 4, 1.5, "Pointwise exit adapters; deepest exit uses identity", fontsize=6.2)
    audit_and_save(fig, "fig1_shared_latent_system",
        "Exact execution structure of the evaluated shared-latent early-exit path. Four full-frame stem blocks precede tile execution. The router consumes stem statistics and QP; its control is source-calibrated in the reported sweep. The actual videoSRC05 QP32 map at the 0.1 dB target leaves 40, 27, 7 and 3 tiles active at total depths 6, 8, 10 and 12. Blank cells have departed. Exit adapters write into a common canvas before full-frame repair and reconstruction. Tensor values are schematic; tile assignments and counts are recorded. Colours run from light teal at depth 6 to navy at depth 12.", book)


def budget(d, book):
    fig, axs = plt.subplots(1, 2, figsize=(183 * MM, 65 * MM), gridspec_kw={"width_ratios":[1.08,1]})
    fig.subplots_adjust(left=.074, right=.975, bottom=.23, top=.76, wspace=.35)
    ax=axs[0]; panel(ax,"a","Depth supplies most of the saving")
    for rule in LABEL:
        rr=[r for r in d["summary"] if r["rule"]==rule]
        ax.plot(BUD,[r["mean"] for r in rr],color=COL[rule],marker=MARK[rule],label=LABEL[rule])
    ax.axhline(d["ceiling_pct"],color=MUTED,lw=.6,ls=(0,(3,2)))
    ax.text(.29,40.4,"39.12% ceiling",fontsize=6.5,color=MUTED)
    ax.set(xlabel="Nominal 444-MSE target (dB)",ylabel="Synthesis MAC saving (%)",xlim=(.035,.52),ylim=(8,44))
    ax.set_xticks([.05,.1,.2,.3,.5],[".05",".10",".20",".30",".50"])
    ax=axs[1]; panel(ax,"b","Content adds value at tighter targets")
    ax.grid(False); ax.grid(axis="x",color=GRID,lw=.45)
    ax.axhspan(.55,1.45,color="#F0F5F4",zorder=0)
    for name,c,marker,dy in [("oracle_minus_dither",TEAL,"o",-.13),("router_minus_dither",BLUE,"s",.13)]:
        rr=[r for r in d["contrasts"] if r["contrast"]==name]
        v=np.array([r["mean"] for r in rr]);lo=np.array([r["lo"] for r in rr]);hi=np.array([r["hi"] for r in rr])
        ax.errorbar(v,np.arange(6)+dy,xerr=[v-lo,hi-v],fmt=marker,color=c,ms=3.3,lw=.9,capsize=2)
    ax.axvline(0,color=MUTED,lw=.6)
    ax.set_yticks(range(6),[".05",".10",".15",".20",".30",".50"])
    ax.set(xlim=(-.4,7.6),ylim=(5.6,-.6),xlabel="Extra MAC saving over dither (pp)",ylabel="Nominal target (dB)")
    handles=[Line2D([],[],color=COL[k],marker=MARK[k],label=LABEL[k]) for k in LABEL]
    fig.legend(handles=handles,loc="upper center",bbox_to_anchor=(.52,1.01),ncol=4,fontsize=7,handlelength=1.8,columnspacing=2.1)
    fig.text(.5,.035,"Paired 95% sequence intervals · source-calibrated controls · arithmetic savings, excluding routing and signalling",ha="center",fontsize=6.5,color=MUTED)
    audit_and_save(fig,"fig2_budget_value",
        "Compact nominal-budget comparison. Panel a separates total modelled decoder arithmetic saving from the additional value of spatial placement. Panel b reports the paired search-minus-dither and router-minus-dither means with 95% sequence-cluster bootstrap intervals; the shaded row is the 0.1 dB target. Common cohort sizes are 204, 263, 265, 265, 265 and 265. Colours and markers consistently identify four policies. Controls use per-frame source calibration; nominal targets do not guarantee final cropped quality. No measured runtime reduction is implied.",book)


def budget_detail(d, book):
    fig, axs = plt.subplots(2, 2, figsize=(183 * MM, 119 * MM))
    fig.subplots_adjust(left=.085, right=.977, bottom=.115, top=.90, wspace=.29, hspace=.58)
    ax = axs[0, 0]; panel(ax, "a", "Total saving and simple controls")
    for rule in LABEL:
        rs = [r for r in d["summary"] if r["rule"] == rule]
        ax.plot(BUD, [r["mean"] for r in rs], marker=MARK[rule], color=COL[rule], label=LABEL[rule])
    ax.axhline(d["ceiling_pct"], color=MUTED, lw=.7, ls=(0, (3, 2)))
    ax.text(.31, 41, f"Ladder ceiling: {d['ceiling_pct']:.2f}%", fontsize=6)
    ax.set(ylim=(0, 45), ylabel="Synthesis MAC saving (%)")
    ax.legend(loc="lower right", fontsize=5.8, handlelength=1.3, labelspacing=.35)

    ax = axs[0, 1]; panel(ax, "b", "The additional value of routing shrinks")
    for contrast, c, marker, label in [("oracle_minus_dither", TEAL, "o", "Search − dither"),
                                        ("router_minus_dither", BLUE, "s", "Router − dither")]:
        rs = [r for r in d["contrasts"] if r["contrast"] == contrast]
        y = np.array([r["mean"] for r in rs]); lo = np.array([r["lo"] for r in rs]); hi = np.array([r["hi"] for r in rs])
        ax.fill_between(BUD, lo, hi, color=c, alpha=.10, lw=0)
        ax.plot(BUD, y, color=c, marker=marker, label=label)
    ax.axhline(0, color=MUTED, lw=.6)
    ax.set(ylim=(-.6, 7.6), ylabel="Additional saving (percentage points)")
    ax.legend(loc="upper right", handlelength=1.5)

    ax = axs[1, 0]; panel(ax, "c", "The gain depends on rate and budget")
    ax.grid(False)
    mat = np.array([[next(r["mean"] for r in d["by_qp"] if r["qp"] == q and r["budget"] == b)
                     for b in BUD] for q in [0, 16, 32, 48, 63]])
    lim = max(abs(mat.min()), abs(mat.max()))
    cmap = colors.LinearSegmentedColormap.from_list("premium", [ORANGE, "#fafafa", BLUE])
    ax.imshow(mat, cmap=cmap, norm=colors.TwoSlopeNorm(0, -lim, lim), aspect="auto")
    for y in range(5):
        for x in range(6):
            ax.text(x, y, f"{mat[y,x]:.2f}", ha="center", va="center", fontsize=6,
                    color="white" if abs(mat[y,x]) > .65 * lim else INK)
    ax.set_xticks(range(6), [".05", ".10", ".15", ".20", ".30", ".50"])
    ax.set_yticks(range(5), [0, 16, 32, 48, 63])
    ax.set(xlabel="Target 444-MSE loss budget (dB)", ylabel="Quality index")
    ax.text(.5, -.30, "Cell: router − dither, percentage points", transform=ax.transAxes, ha="center", fontsize=6)

    ax = axs[1, 1]; panel(ax, "d", "The trend survives a fixed sample cohort")
    rs = [r for r in d["contrasts"] if r["contrast"] == "router_minus_dither"]
    ax.plot(BUD, [r["mean"] for r in rs], "s-", color=BLUE, label="Common cohort at each budget")
    ax.plot(BUD, [r["stable_mean"] for r in rs], "o--", color=INK, mfc="white", label="Same 204 frame–QP pairs throughout")
    ax.axhline(0, color=MUTED, lw=.6)
    ax.set(ylim=(-.3, 3.2), ylabel="Router − dither (percentage points)")
    ax.legend(loc="upper right", fontsize=5.5, handlelength=1.3)
    for a in [axs[0, 0], axs[0, 1], axs[1, 1]]:
        a.set(xlabel="Target 444-MSE loss budget (dB)", xlim=(.03, .52))
        a.set_xticks([.05, .1, .2, .3, .5], [".05", ".10", ".20", ".30", ".50"])
    audit_and_save(fig, "fig7_budget_sensitivity",
        "The value of spatial routing depends on the quality budget. Means use the common feasible frame–QP pairs of all four rules at each budget: 204, 263, 265, 265, 265 and 265, respectively. These are 53 first intra frames at five QPs before exclusions. Shading in b shows paired 95% sequence-cluster bootstrap intervals (5,000 draws, fixed checkpoint). Panel c stratifies the paired premium by QP, and d keeps the same 204 pairs across all budgets. The cost is the synthesis MAC model and excludes routing/bitstream overhead. Per-frame source-informed calibration is used for all curves; target losses are not guarantees or exactly matched achieved losses.", book)


def quality(d, book):
    with (DATA / "samples.csv").open() as f:
        samples = list(csv.DictReader(f))
    fig, axs = plt.subplots(1, 3, figsize=(183 * MM, 69 * MM))
    fig.subplots_adjust(left=.075, right=.982, bottom=.24, top=.81, wspace=.38)
    ax = axs[0]; panel(ax, "a", "Achieved quality differs across rules")
    for rule in LABEL:
        rs = [r for r in d["summary"] if r["rule"] == rule]
        ax.plot([r["rgb_mean"] for r in rs], [r["mean"] for r in rs],
                marker=MARK[rule], color=COL[rule], label=LABEL[rule])
    ax.set(xlabel="Mean delivered 444-MSE loss (dB)", ylabel="Synthesis MAC saving (%)", ylim=(5, 42))
    ax.legend(loc="lower right", fontsize=5.5, handlelength=1.1)

    ax = axs[1]; panel(ax, "b", "Reporting differs from selection")
    for metric, c, marker, label in [("rgb_loss", BLUE, "o", "444-MSE"), ("yuv_loss", ORANGE, "s", "YUV 6:1:1")]:
        vals = np.sort([float(r[metric]) for r in samples if float(r["budget"]) == .1 and r["rule"] == "router"])
        ax.step(vals, np.arange(1, len(vals) + 1) / len(vals), where="post", color=c, label=label)
    ax.axvline(.1, color=INK, ls="--", lw=.7)
    ax.text(.112, .28, "0.1 dB target", rotation=90, fontsize=6)
    ax.set(xlabel="Delivered loss (dB)", ylabel="Cumulative fraction of frame–QP pairs", xlim=(-.015, .305), ylim=(0, 1.02))
    ax.legend(loc="lower right", fontsize=6)

    ax = axs[2]; panel(ax, "c", "Losses above the nominal target")
    rs = [r for r in d["summary"] if r["budget"] == .1]
    for i, metric in enumerate(["rgb_over", "yuv_over"]):
        yy = np.arange(4) + (i - .5) * .22
        ax.scatter([r[metric] for r in rs], yy, color=[BLUE, ORANGE][i], marker=["o", "s"][i], s=14,
                   label=["444-MSE", "YUV 6:1:1"][i], zorder=3)
        for y, r in zip(yy, rs):
            ax.text(r[metric] + 1.5, y, str(r[metric]), va="center", fontsize=6)
    ax.set_yticks(range(4), ["Search", "Router", "Dither", "Uniform"])
    ax.invert_yaxis(); ax.set(xlabel="Above target among 263 paired cases", xlim=(-2, max(r["yuv_over"] for r in rs) + 12))
    ax.legend(loc="lower right", fontsize=6)
    audit_and_save(fig, "fig3_delivered_quality",
        "Selection and reporting use different domains and anchors. Panel a plots the recorded mean 444-MSE loss against synthesis MAC savings; joining segments guide the eye. Panels b and c use the 263 common cases at a 0.1 dB target. Counts are reported losses above target + 0.0001 dB. Crucially, selection uses a separate released-weight reference on padded YCbCr 4:4:4, whereas exported losses use the cropped full-frame fine-tuned e15 output (the raw field psnr_release is misleading). These counts do not isolate budget-enforcement error under a common anchor. A new evaluation must measure both references on the same cropped support. No verified fallback was applied in the archive.", book)


def decision_cost(d, book):
    fig, ax = schematic(116)
    heading(ax, 3, 111, "a", "The source is available at the encoder, but the error table must be computed")
    # Alternative execution traces: shared latent, not six independent encodes.
    tensor(ax, 7, 83, 12, 12, grid=3)
    text(ax, 13, 78, "Latent", ha="center")
    for i, yy in enumerate([96, 92, 88, 84]):
        arrow(ax, (23, 89), (34, yy), color=DEPTH[i])
        strip(ax, 36, yy - 1, 4 + i * 2, w=1.3, h=2, color=DEPTH[i])
    text(ax, 48, 102, "Repeated tiled decodes", ha="center", fontsize=6)
    text(ax, 48, 77, "Shared work repeats", ha="center", fontsize=6)
    arrow(ax, (65, 89), (76, 89))
    # An actual conceptual table, explicitly not a heatmap of measured errors.
    for iy in range(4):
        for ix in range(4):
            ax.add_patch(Rectangle((80 + ix * 3.4, 83 + iy * 3), 3.4, 3,
                                   facecolor="#edf2f5", edgecolor="white", lw=.5))
    text(ax, 87, 89, "M[t,k]", ha="center", fontsize=6)
    text(ax, 87, 78, "Source error table", ha="center", fontsize=6)
    arrow(ax, (98, 89), (108, 89))
    text(ax, 123, 94, "Budget search", ha="center")
    # Discrete allocation staircase rather than a module box.
    ax.plot([111, 117, 117, 123, 123, 129, 129, 135], [85, 85, 87, 87, 89, 89, 91, 91], color=BLUE, lw=1.1)
    text(ax, 123, 78, "Choose λ / β / mixture", ha="center", fontsize=6)
    arrow(ax, (138, 89), (148, 89))
    text(ax, 164, 94, "Mixed reconstruction", ha="center")
    tilemap(ax, 153, 83, 22, 7, np.array(d["example"]["rules"]["router"]["0.1"]["map"]).reshape(d["example"]["grid"]))
    text(ax, 164, 78, "Measure delivered loss", ha="center", fontsize=6)
    text(ax, 7, 69, "One-pass alternative", weight="bold", fontsize=6)
    strip(ax, 46, 67, 12, w=2, h=3, color=BLUE)
    for i, x in enumerate([60, 65.4, 70.8, 76.2]):
        arrow(ax, (x, 66), (x, 62), color=DEPTH[i])
    text(ax, 87, 66, "Tap four distinct tiled exits; reuse the prefix", fontsize=6)
    ax.plot([3, 180], [57, 57], color=GRID, lw=.5)

    heading(ax, 3, 52, "b", "Recorded table construction time")
    p = fig.add_axes([.105, .15, .44, .245])
    times = d["runtime"]["stage_medians"]
    # Keys are fixed by the audited archive, never inferred from plot labels.
    vals = [times["table_shipped"], times["table_dedup"], times["table_allexits"]]
    p.barh([2, 1, 0], vals, color=["#98a5ad", "#5d8ca5", BLUE], height=.52)
    for y, v in zip([2, 1, 0], vals):
        p.text(v + 22, y, f"{v:,.0f}", va="center", fontsize=6)
    p.set_yticks([2, 1, 0], ["6 trials", "4 unique exits", "1 shared pass"])
    p.set(xlim=(0, 1640), xlabel="Recorded stage time (ms)")
    p.grid(axis="x", color=GRID, lw=.45); p.set_axisbelow(True)
    heading(ax, 111, 52, "c", "Calibration changes the cost")
    text(ax, 116, 43, "Fixed, held-out β", weight="bold", fontsize=6)
    text(ax, 116, 36, "Router uses decoded features.\nNo per-frame source table is required.", fontsize=6, linespacing=1.5)
    text(ax, 116, 24, "Per-frame source-calibrated β", weight="bold", fontsize=6)
    text(ax, 116, 17, "The plotted budget sweep uses M[t,k].\nA fast router does not remove that cost.", fontsize=6, linespacing=1.5)
    text(ax, 7, 4, "Historical timing: 8 frames × 3 repetitions at QP32. No new end-to-end latency measurement.", fontsize=6)
    audit_and_save(fig, "fig4_decision_cost",
        "The encoder pays to acquire source-error information. The upper trace is schematic; the table and search stages are distinct from latent creation. In the archived benchmark, six trial exits contain only four distinct reachable outputs. The reported table-construction statistic decreases from 1,347.65 ms to 336.87 ms with a shared tiled pass. Bars use the archived upper median of 24 pooled trials (eight frames × three repeats, QP32); raw trial distributions and a contemporaneous load log are unavailable, so no uncertainty bars are invented. The archived GPU search timer can include pending reference-decode work and is excluded here. These stage measurements establish neither end-to-end encode latency nor a router speedup. The source-calibrated budget curves require information unavailable to an autonomous decoder.", book)


def maps(d, book):
    fig, ax = schematic(49)
    ex=d["example"];mshape=ex["grid"];H,W=ex["hw"]
    thumbs=np.load(THUMBNAILS)
    heading(ax,3,45,"a","One source, four allocations at the 0.1 dB target")
    xs=[4,40,76,112,148]; width=30
    for x,label in zip(xs,["Source luma","Search","Router","Bayer dither","Uniform"]):
        text(ax,x+width/2,38,label,ha="center",weight="bold",fontsize=7)
    y=18
    ax.imshow(thumbs[str(ex["index"])],cmap="gray",vmin=0,vmax=255,extent=(xs[0],xs[0]+width,y,y+width*H/W),zorder=1)
    text(ax,xs[0]+width/2,14,ex["seq"].split("_")[0],ha="center",fontsize=6.5)
    text(ax,xs[0]+width/2,10,"QP32",ha="center",fontsize=6.5)
    for x,rule in zip(xs[1:],LABEL):
        v=ex["rules"][rule]["0.1"];m=np.array(v["map"]).reshape(mshape)
        for iy in range(mshape[0]):
            for ix in range(mshape[1]):
                left,right=ix*256,min((ix+1)*256,W);top,bottom=iy*256,min((iy+1)*256,H)
                if right<=left or bottom<=top:continue
                ax.add_patch(Rectangle((x+left/W*width,y+(H-bottom)/W*width),(right-left)/W*width,(bottom-top)/W*width,facecolor=DEPTH[int(m[iy,ix])-2],edgecolor="white",lw=.4))
        text(ax,x+width/2,14,f"{v['saving']:.1f}% MAC saved",ha="center",fontsize=6.5)
        text(ax,x+width/2,10,f"{v['db_rgb']:.3f} dB loss",ha="center",fontsize=6.5)
    depth_key(ax,40,3)
    audit_and_save(fig,"fig5_spatial_decisions",
        "Recorded exit assignments at a nominal 0.1 dB target for videoSRC05 QP32. The source-luma thumbnail is archived evaluator data, not generated imagery or a reconstructed output. Maps are cropped to the valid image extent. Depth colours match the execution diagram. This frame was chosen from five available thumbnails by proximity to the aggregate routing margin. Quality and MAC annotations are recorded mixed-output measurements and modelled arithmetic, respectively. At 0.3 dB the search, router and dither maps for this frame all take depth 6; repeated identical maps are omitted.",book)


def depth_system(d, book):
    fig, ax = schematic(103)
    ax.set_ylim(0,125)
    heading(ax, 3, 119, "a", "Shared early exit: one representation, nested synthesis")
    # A latent stack feeds a single trunk. Departing paths converge on a
    # common canvas; the staircase encodes retained block depth, not runtime.
    tensor(ax, 7, 95, 13, 12, "#E3EEED", grid=4)
    text(ax, 14, 90, "One latent", ha="center", fontsize=6.3)
    arrow(ax, (23,101), (34,101))
    strip(ax, 37, 97, 4, w=2.4, h=8, color=BLUE)
    text(ax, 43, 90, "Shared stem", ha="center", fontsize=6.3)
    arrow(ax, (51,101), (62,101))
    for i, c in enumerate(DEPTH):
        x=65+i*16
        strip(ax,x,97,2,w=3,h=8,color=c)
        if i<3:arrow(ax,(x+8,101),(x+14,101),color=c)
        ax.plot([x+4,x+4,133],[96,88-i*2.6,88-i*2.6],color=c,lw=1.2)
        ax.scatter([x+4],[96],s=9,color=c,zorder=4)
        text(ax,x+4,109,str(6+2*i),ha="center",fontsize=6.3,weight="bold",color=c)
    arrow(ax,(133,81),(147,98),color=BLUE)
    tensor(ax,150,96,15,11,DEPTH[0],grid=4)
    text(ax,157,90,"Common head",ha="center",fontsize=6.3)
    text(ax,178,79,"MEASURED EXIT FAMILY",ha="right",color=BLUE,fontsize=5.8,weight="bold")
    ax.plot([3,180],[75,75],color=GRID,lw=.7)

    heading(ax,3,70,"b","Independent depth controls: each codec learns its own representation")
    xs=[14,45,76,107,138,169]
    for x,depth,c in zip(xs,[2,4,6,8,10,12],DEPTH_ALL):
        text(ax,x,62,f"D{depth}",ha="center",fontsize=7,weight="bold",color=INK)
        # Each expert has its own analysis funnel and latent glyph.
        ax.add_patch(Polygon([[x-8,58],[x+8,58],[x+3,54],[x-3,54]],fc=c,ec=MUTED,lw=.4))
        for j in range(3):ax.add_patch(Rectangle((x-3+j*2.2,50.5),1.5,2,fc=c,ec=MUTED,lw=.3))
        for j in range(depth):ax.add_patch(Rectangle((x-8+j*1.35,44),1.0,4.2,fc=c,ec=MUTED,lw=.3))
        status="TRAINING" if depth<8 else "PLANNED" if depth<12 else "RELEASED"
        text(ax,x,40,status,ha="center",fontsize=5.8,weight="bold",color=MUTED if depth in (8,10) else BLUE if depth<8 else INK)
        if depth in (8,10):ax.add_patch(Rectangle((x-10,43),21,16.5,fill=False,ec=MUTED,ls=(0,(2,2)),lw=.5))
    text(ax,3,34,"Same-recipe D12 is an additional control. The released anchor alone does not isolate depth.",fontsize=6.2)
    ax.plot([3,180],[29,29],color=GRID,lw=.7)

    heading(ax,3,24,"c","Later routing study: select before encoding; signal which expert owns the payload")
    # Feature samples, an actual node graph, batched tiles and a byte tape.
    for i,c in enumerate([DEPTH_ALL[1],DEPTH_ALL[3],DEPTH_ALL[5]]):
        ax.add_patch(Rectangle((7+i*2,9+i),8,7,fc=c,ec="white",lw=.5))
    text(ax,12,4,"RGB patches",ha="center",fontsize=6)
    arrow(ax,(24,13),(35,13))
    nodes=[[(40,10),(40,16)],[(46,8),(46,13),(46,18)],[(52,10),(52,16)]]
    for aa,bb in zip(nodes[:-1],nodes[1:]):
        for x,y in aa:
            for u,v in bb:ax.plot([x,u],[y,v],color="#BBCDCE",lw=.45)
    for col in nodes:
        for x,y in col:ax.scatter(x,y,s=9,c=BLUE,zorder=4)
    text(ax,46,4,"MLP selector",ha="center",fontsize=6)
    arrow(ax,(56,13),(69,13))
    for i,c in enumerate([DEPTH_ALL[1],DEPTH_ALL[3],DEPTH_ALL[5]]):
        for j in reversed(range(3-i)):
            ax.add_patch(Rectangle((74+i*5+j*.5,10+j*.8),4,6,fc=c,ec="white",lw=.4))
    text(ax,81,4,"Expert queues",ha="center",fontsize=6)
    arrow(ax,(91,13),(104,13))
    strip(ax,109,10,6,w=1.4,h=6,color=BLUE)
    text(ax,115,4,"Selected codec",ha="center",fontsize=6)
    arrow(ax,(126,13),(136,13))
    for x,w,c,lab in [(141,6,ORANGE,"ID"),(147,9,TEAL,"z"),(156,20,BLUE,"y")]:
        ax.add_patch(Rectangle((x,9),w,8,fc=c,ec="white",lw=.6))
        text(ax,x+w/2,13,lab,ha="center",color="white",fontsize=6)
    text(ax,159,4,"Control + payload",ha="center",fontsize=6)
    audit_and_save(fig,"fig6_depth_study_system",
        "Shared early exit and the independent DCVC-UF depth study answer different questions. Panel a shows one latent with nested synthesis depths 6/8/10/12; only this family has completed reconstruction comparisons. Panel b shows independent D2/D4/D6 training, planned D8/D10 and the released D12 anchor. The same-recipe D12 control is still needed. Panel c is a proposed subsequent model-bank routing path, selecting before encoding and transmitting expert identity. Network and queue geometry is schematic, not projected quality or throughput.",book)


def main():
    global DATA, OUT, THUMBNAILS
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--data-dir", type=Path, default=DATA)
    ap.add_argument("--output-dir", type=Path, default=OUT)
    ap.add_argument("--thumbnail-file", type=Path, default=THUMBNAILS)
    args = ap.parse_args()
    DATA, OUT, THUMBNAILS = args.data_dir, args.output_dir, args.thumbnail_file
    OUT.mkdir(parents=True, exist_ok=True)
    d = load()
    with PdfPages(OUT / "figure_atlas.pdf", metadata=PDF_META) as book:
        architecture(d, book)
        budget(d, book)
        quality(d, book)
        decision_cost(d, book)
        maps(d, book)
        depth_system(d, book)
        budget_detail(d, book)
    (OUT / "captions.json").write_text(json.dumps(CAPTIONS, indent=2) + "\n")
    (OUT / "layout_audit.json").write_text(json.dumps(AUDIT, indent=2) + "\n")
    manifest = {p.name: hashlib.sha256(p.read_bytes()).hexdigest() for p in sorted(OUT.iterdir()) if p.suffix in {".pdf", ".svg", ".png"}}
    (OUT / "artifact_manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")
    print(json.dumps(dict(figures=len(CAPTIONS), output=str(OUT), text_outside_canvas=0)))


if __name__ == "__main__":
    main()
