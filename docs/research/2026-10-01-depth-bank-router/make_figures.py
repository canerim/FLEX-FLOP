"""CPU-only, reproducible figures for the independent DCVC-UF depth bank.

Only immutable validation/Kodak JSON and manifests are read. The diagrams
are explicitly schematics; measured curves are never inferred from them.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
from collections import defaultdict
from datetime import datetime, timezone
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.patches import Circle, FancyArrowPatch, FancyBboxPatch, Polygon, Rectangle


DEPTHS = (2, 4, 6, 8, 10, 12)
COLORS = {2: "#256DAD", 4: "#D4743A", 6: "#168576", 8: "#795FA3", 10: "#B2537C", 12: "#3C4757"}
INK = "#27313D"
MUTED = "#627081"
GRID = "#E1E6EA"
PALE = "#F5F7F8"
RATES = (0.1, 0.2, 0.4)
RNG = np.random.default_rng(20261001)


def setup() -> None:
    plt.rcParams.update({
        "font.family": "DejaVu Sans", "font.size": 8.7,
        "axes.titlesize": 10.1, "axes.labelsize": 8.8,
        "axes.edgecolor": "#B8C2CA", "axes.labelcolor": INK,
        "xtick.color": MUTED, "ytick.color": MUTED,
        "axes.spines.top": False, "axes.spines.right": False,
        "pdf.fonttype": 42, "ps.fonttype": 42, "svg.fonttype": "none",
        "savefig.facecolor": "white", "figure.facecolor": "white",
    })


def save(fig, output: Path, stem: str) -> None:
    for ext in ("pdf", "svg", "png"):
        fig.savefig(output / f"{stem}.{ext}", dpi=300, bbox_inches="tight", pad_inches=0.12)
    plt.close(fig)


def sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def read_json(path: Path, sources: dict) -> dict:
    sources[str(path)] = sha(path)
    return json.loads(path.read_text())


def interpolate_image(rows: list[dict], rate: float, metric: str) -> float | None:
    ordered = sorted(rows, key=lambda item: item["estimated_bpp"])
    for left, right in zip(ordered, ordered[1:]):
        a, b = left["estimated_bpp"], right["estimated_bpp"]
        if a <= rate <= b:
            t = (math.log(rate) - math.log(a)) / (math.log(b) - math.log(a))
            return left[metric] + t * (right[metric] - left[metric])
    return None


def by_image(data: dict) -> dict[str, list[dict]]:
    groups = defaultdict(list)
    for row in data["rows"]:
        groups[row["image"]].append(row)
    return groups


def monitor_at_rate(data: dict, rate: float = 0.2) -> float | None:
    groups = by_image(data)
    if len(groups) != 4:
        raise ValueError(f"Four monitor images expected, found {len(groups)}")
    values = [interpolate_image(rows, rate, "psnr_yuv611") for rows in groups.values()]
    return float(np.mean(values)) if all(v is not None for v in values) else None


def sampled_history(root: Path, depth: int, sources: dict, n: int = 18) -> tuple[list[dict], int | None]:
    folder = root / f"runs/d{depth}/validation"
    if not folder.exists():
        return [], None
    records = {}
    for path in folder.glob("step_*.json"):
        data = json.loads(path.read_text())
        epoch = data["epoch"]
        value = monitor_at_rate(data)
        if epoch > 0 and value is not None:
            if epoch in records:
                raise ValueError(f"Repeated epoch {epoch} in D{depth}")
            records[epoch] = (path, value)
    if not records:
        return [], None
    available = sorted(records)
    chosen = {available[0], available[-1]}
    chosen.update(e for e in (44, 45, 69, 70, 89, 90, 94, 95, 99, 100, 103, 105) if e in records)
    while len(chosen) < min(n, len(available)):
        e = max((e for e in available if e not in chosen),
                key=lambda e: (min(abs(e - item) for item in chosen), -e))
        chosen.add(e)
    rows = []
    for epoch in sorted(chosen):
        path, value = records[epoch]
        sources[str(path)] = sha(path)
        rows.append({"epoch": epoch, "psnr_yuv611_db": value})
    return rows, available[-1]


def bootstrap_ci(values: np.ndarray, seed: int) -> tuple[float, float]:
    rng = np.random.default_rng(seed)
    resampled = values[rng.integers(0, len(values), (10000, len(values)))].mean(axis=1)
    return tuple(float(x) for x in np.percentile(resampled, (2.5, 97.5)))


def kodak_data(root: Path, sources: dict) -> tuple[dict, dict]:
    paths = {
        2: root / "runs/d2/kodak_final.json",
        4: root / "runs/d4/kodak_final.json",
        12: root / "reference_d12/kodak.json",
    }
    models = {depth: read_json(path, sources) for depth, path in paths.items()}
    for depth, data in models.items():
        if data["dataset"] != "Kodak full resolution" or data["depth"] != depth:
            raise ValueError(f"Unexpected Kodak record for D{depth}")
        if "NOT actual bitstream" not in data["rate_kind"]:
            raise ValueError("Bitrate semantics changed")
        groups = by_image(data)
        if len(groups) != 24 or any(len(rows) != 5 for rows in groups.values()):
            raise ValueError(f"Incomplete Kodak record for D{depth}")
    paired = {}
    grouped = {depth: by_image(data) for depth, data in models.items()}
    for rate in RATES:
        common = []
        per_depth = {depth: {} for depth in models}
        for image in sorted(grouped[12]):
            vals = {depth: interpolate_image(grouped[depth][image], rate, "psnr_yuv611")
                    for depth in models}
            if all(v is not None for v in vals.values()):
                common.append(image)
                for depth, value in vals.items():
                    per_depth[depth][image] = value
        if len(common) < 10:
            raise ValueError(f"Insufficient shared support at {rate} bpp")
        paired[str(rate)] = {"n_images": len(common), "differences": {}}
        for depth in (2, 4):
            differences = np.array([per_depth[depth][im] - per_depth[12][im] for im in common])
            paired[str(rate)]["differences"][str(depth)] = {
                "mean_db": float(np.mean(differences)),
                "image_bootstrap_95ci_db": bootstrap_ci(differences, 20261001 + depth + int(rate * 100)),
            }
    return models, paired


def plot_kodak(models: dict, paired: dict, output: Path) -> None:
    fig = plt.figure(figsize=(7.25, 3.25), layout="constrained")
    gs = fig.add_gridspec(1, 2, width_ratios=(1.25, 1.0), wspace=0.13)
    ax, bx = fig.add_subplot(gs[0]), fig.add_subplot(gs[1])
    for depth in (2, 4, 12):
        means = sorted(models[depth]["means"], key=lambda r: r["estimated_bpp"])
        ax.plot([r["estimated_bpp"] for r in means], [r["psnr_yuv611"] for r in means],
                "-o", color=COLORS[depth], lw=1.9, ms=4.1,
                label="Released D12" if depth == 12 else f"D{depth} · 105 epochs")
    ax.set(xscale="log", xlabel="Estimated bits per pixel", ylabel="YUV 6:1:1 PSNR (dB)")
    ax.set_xticks([.04, .1, .2, .4, .8], [".04", ".1", ".2", ".4", ".8"])
    ax.grid(axis="y", color=GRID, lw=.65)
    ax.legend(frameon=False, fontsize=7.6, loc="lower right")
    ax.set_title("a   Final codec curves · Kodak, n = 24", loc="left", fontweight="bold")

    for depth, shift in ((2, -.065), (4, .065)):
        for idx, rate in enumerate(RATES):
            item = paired[str(rate)]["differences"][str(depth)]
            mean = item["mean_db"]
            lo, hi = item["image_bootstrap_95ci_db"]
            x = idx + shift
            bx.errorbar(x, mean, yerr=[[mean-lo], [hi-mean]], fmt="o", color=COLORS[depth],
                        ms=4.9, capsize=2.6, lw=1.25, label=f"D{depth}" if idx == 0 else None)
    bx.axhline(0, color=MUTED, lw=.85, ls=(0, (3, 3)))
    bx.set(xlim=(-.42, 2.42), xticks=list(range(3)),
           xticklabels=[f"{r:.1f}\n(n={paired[str(r)]['n_images']})" for r in RATES],
           xlabel="Matched estimated bits per pixel",
           ylabel="Δ YUV PSNR vs released D12 (dB)")
    bx.grid(axis="y", color=GRID, lw=.65)
    bx.legend(frameon=False, fontsize=7.8, loc="upper right")
    bx.set_title("b   Paired difference · 95% image bootstrap", loc="left", fontweight="bold")
    fig.text(.015, -.035, "Rate is a deterministic entropy estimate; CIs resample images, not training seeds."
             " Source: full-resolution Kodak; no actual bitstream or router result.",
             color=MUTED, fontsize=7.25)
    save(fig, output, "fig01_kodak_final_depths")


def plot_training(root: Path, sources: dict, output: Path) -> dict:
    fig = plt.figure(figsize=(7.25, 3.82), layout="constrained")
    gs = fig.add_gridspec(1, 2, width_ratios=(1.48, .87), wspace=.13)
    ax, bx = fig.add_subplot(gs[0]), fig.add_subplot(gs[1])
    history = {}
    status = {}
    for depth in DEPTHS:
        p = root / f"runs/d{depth}/status.json"
        status[depth] = read_json(p, sources) if p.exists() else None
        if depth in (2, 4, 6, 8, 10):
            rows, last = sampled_history(root, depth, sources)
            history[depth] = {"sampled": rows, "last_supported_epoch": last}
    ax.axvspan(90, 105, color="#F0F2F4", zorder=0)
    ax.axvline(90, color="#A4AFB8", lw=.9, ls=(0, (3, 3)), zorder=1)
    for depth in (2, 4, 6, 8, 10):
        rows = history[depth]["sampled"]
        if not rows:
            continue
        ax.plot([r["epoch"] for r in rows], [r["psnr_yuv611_db"] for r in rows],
                "-o", color=COLORS[depth], lw=1.65, ms=2.8,
                label=f"D{depth} · epoch {rows[-1]['epoch']}")
    ax.set(xlim=(0, 106), xlabel="Completed training epoch",
           ylabel="YUV PSNR at 0.2 estimated bpp (dB)")
    ax.set_ylim(bottom=34.7)
    ax.grid(axis="y", color=GRID, lw=.65)
    ax.legend(frameon=False, loc="lower right", fontsize=7.35, ncol=1)
    ax.set_title("a   Four-image validation monitor", loc="left", fontweight="bold")

    outer = 27_947_520 / 1e6
    for idx, depth in enumerate(DEPTHS):
        if depth < 12:
            manifest = read_json(root / f"runs/d{depth}/manifest.json", sources)
            total = manifest["parameters"] / 1e6
            decoder = manifest["decoder_parameters"] / 1e6
        else:
            reference = read_json(root / "reference_d12/neural_runtime.json", sources)
            total, decoder = reference["parameters"] / 1e6, reference["decoder_parameters"] / 1e6
        if not math.isclose(total - decoder, outer, abs_tol=1e-7):
            raise ValueError(f"Non-decoder parameter count changed in D{depth}")
        y = len(DEPTHS) - 1 - idx
        bx.barh(y, outer, height=.55, color="#CCD5DB", edgecolor="none")
        bx.barh(y, decoder, left=outer, height=.55, color=COLORS[depth], edgecolor="none")
        run = status[depth]
        progress = min(105, run["global_step"] / run["batches_per_epoch"]) if run else 0.0
        tag = "queued" if run is None else ("complete" if run["state"] == "complete" else f"{progress:.1f}/105")
        bx.text(total + .33, y, tag, va="center", fontsize=7.2,
                color=COLORS[depth] if run else MUTED)
    bx.set(yticks=range(6), yticklabels=[f"D{d}" for d in reversed(DEPTHS)],
           xlim=(0, 56), xlabel="Millions of parameters")
    bx.set_xticks([0, 14, 28, 42], ["0", "14", "28", "42"])
    bx.tick_params(axis="y", length=0)
    bx.grid(axis="x", color=GRID, lw=.65)
    bx.set_axisbelow(True)
    bx.set_title("b   Architecture and live progress", loc="left", fontweight="bold")
    bx.text(.02, -.17, "Gray: all other modules\nColor: synthesis decoder", transform=bx.transAxes,
            color=MUTED, fontsize=7.2, va="top")
    fig.text(.015, -.055, "Fixed four DIV2K center crops; per-image log-rate interpolation."
             " D10 has no 0.2-bpp support yet; D12 has no training record."
             " Gray band: 512-pixel training crop.", color=MUTED, fontsize=7.15)
    save(fig, output, "fig02_training_and_capacity")
    return {"history": history, "status": {str(d):
            ({"state": status[d]["state"], "fractional_epoch": status[d]["global_step"] /
              status[d]["batches_per_epoch"]} if status[d] else {"state": "queued"})
            for d in DEPTHS}}


def box(ax, x, y, w, h, color, radius=.12, alpha=1, edge=None, lw=.8):
    patch = FancyBboxPatch((x, y), w, h,
                           boxstyle=f"round,pad=0,rounding_size={radius}",
                           facecolor=color, edgecolor=edge or "none", lw=lw,
                           alpha=alpha)
    ax.add_patch(patch)
    return patch


def arrow(ax, x0, y0, x1, y1, color=MUTED, lw=1.0, alpha=1):
    ax.add_patch(FancyArrowPatch((x0, y0), (x1, y1), arrowstyle="-|>",
                                 mutation_scale=8.5, color=color, lw=lw, alpha=alpha,
                                 connectionstyle="arc3,rad=0"))


def patch_mosaic(ax, x, y, w, h, rows=4, cols=4, alpha=1):
    shades = ("#C7D7DF", "#B2C5D0", "#D4C8B9", "#AAC0B7",
              "#B9C9D4", "#D8CDBF", "#B9CECA", "#C4BED2",
              "#9CB5C3", "#CABEB4", "#ADC8C3", "#CDBAC8",
              "#D1C6B7", "#B3C9BE", "#B4BED1", "#A4B2BE")
    for rr in range(rows):
        for cc in range(cols):
            color = shades[(rr * cols + cc) % len(shades)]
            ax.add_patch(Rectangle((x+cc*w/cols, y+(rows-1-rr)*h/rows),
                                   w/cols-.018, h/rows-.018, facecolor=color,
                                   edgecolor="white", linewidth=.65, alpha=alpha))
            # Texture strokes make the sample read as image regions, not a legend.
            if (rr+cc) % 3 == 0:
                for k in range(2):
                    xx=x+cc*w/cols+.08+k*.08
                    yy=y+(rows-1-rr)*h/rows+.08
                    ax.plot([xx, xx+.14], [yy, yy+.10], color="#647A88", lw=.75,
                            alpha=.78*alpha, solid_capstyle="round")


def plot_system_diagram(output: Path) -> None:
    fig, ax = plt.subplots(figsize=(11.6, 5.0), layout="constrained")
    ax.set(xlim=(0, 12), ylim=(0, 5.15))
    ax.axis("off")
    ax.text(.2, 4.93, "Independent-codec routing", fontsize=17, fontweight="bold", color=INK)
    ax.text(.2, 4.64, "One source patch selects one complete encoder → entropy → decoder chain",
            fontsize=9.4, color=MUTED)

    # Source image and the compact transmitted expert-ID map.
    patch_mosaic(ax, .33, 1.17, 1.55, 2.78)
    ax.text(.35, 4.12, "RGB source patches", color=INK, fontsize=9.1, fontweight="bold")
    ax.text(.35, .82, "same tile geometry\nfor all six codecs", color=MUTED, fontsize=7.8)
    arrow(ax, 1.98, 2.60, 2.22, 2.60, color=INK, lw=1.35)

    # Feature field and small source-side decision network.
    ax.text(2.28, 4.12, "Encoder-side decision", color=INK, fontsize=9.1, fontweight="bold")
    for yy, name, symbol in ((3.58, "luma / chroma", "Y"), (2.98, "edge / texture", "∇"),
                             (2.38, "target rate", "R"), (1.78, "time budget", "T")):
        box(ax, 2.27, yy-.15, .37, .31, "#E7EDF0", radius=.08)
        ax.text(2.455, yy, symbol, ha="center", va="center", color=INK, fontsize=8, fontweight="bold")
        ax.text(2.76, yy, name, va="center", color=MUTED, fontsize=8)
    for x, ys in ((3.95, (2.1, 2.65, 3.2)), (4.35, (1.95, 2.4, 2.85, 3.3))):
        for yy in ys:
            ax.add_patch(Circle((x, yy), .055, color="#708292"))
    for y0 in (2.1, 2.65, 3.2):
        for y1 in (1.95, 2.4, 2.85, 3.3):
            ax.plot((3.95, 4.35), (y0, y1), color="#B7C2CA", lw=.45, zorder=0)
    ax.text(3.77, 1.48, "tiny MLP", color=MUTED, fontsize=8.1)
    arrow(ax, 4.53, 2.63, 4.82, 2.63, color=INK, lw=1.35)

    # Six distinct lanes. A depth-scaled tail visualizes the ablation without
    # claiming that the encoders or entropy models are shared.
    ax.text(4.91, 4.12, "Six independent codec paths", color=INK, fontsize=9.1, fontweight="bold")
    ax.text(5.29, 3.88, "analysis", fontsize=7.1, color=MUTED)
    ax.text(6.40, 3.88, "entropy", fontsize=7.1, color=MUTED)
    ax.text(7.72, 3.88, "synthesis depth", fontsize=7.1, color=MUTED)
    for idx, depth in enumerate(DEPTHS):
        yy = 3.52 - idx * .48
        cc = COLORS[depth]
        ax.text(4.91, yy, f"D{depth}", va="center", color=cc, fontweight="bold", fontsize=8.4)
        box(ax, 5.29, yy-.13, .82, .26, cc, radius=.055, alpha=.19)
        ax.text(5.70, yy, f"E{depth}", va="center", ha="center", color=INK, fontsize=8.1)
        arrow(ax, 6.14, yy, 6.32, yy, color=cc, lw=.95)
        box(ax, 6.37, yy-.13, .83, .26, cc, radius=.055, alpha=.19)
        ax.text(6.785, yy, f"H{depth}", va="center", ha="center", color=INK, fontsize=8.1)
        arrow(ax, 7.23, yy, 7.43, yy, color=cc, lw=.95)
        for j in range(depth // 2):
            xx = 7.48+j*.22
            box(ax, xx, yy-.13, .16, .26, cc, radius=.045, alpha=.27+.06*(j == 0))
        ax.text(9.06, yy, f"G{depth}", va="center", ha="center", color=cc, fontsize=7.9)
        ax.plot([9.28, 9.38], [yy, yy], color=cc, lw=1)
    ax.text(5.30, .34, "Independent weights and latents at every depth.",
            fontsize=8.0, color=MUTED)

    # Serialized mode map + payload and the decoder-side reconstruction.
    arrow(ax, 9.38, 2.34, 9.67, 2.34, color=INK, lw=1.4)
    ax.text(9.76, 4.12, "Transmitted stream", color=INK, fontsize=9.1, fontweight="bold")
    box(ax, 9.73, 3.34, 1.70, .38, "#E6EBEF", radius=.08)
    ax.text(10.58, 3.53, "model IDs + QP", ha="center", va="center", color=INK, fontsize=7.6)
    for idx, dep in enumerate((2, 4, 4, 6, 8, 12)):
        box(ax, 9.74+idx*.28, 2.67, .23, .40, COLORS[dep], radius=.05, alpha=.82)
    ax.text(9.76, 2.36, "entropy payloads", color=MUTED, fontsize=7.8)
    arrow(ax, 10.56, 2.12, 10.56, 1.89, color=INK, lw=1.2)
    patch_mosaic(ax, 9.79, .58, 1.59, 1.12, rows=4, cols=4, alpha=.8)
    ax.text(9.73, .30, "decoded + assembled image", color=MUTED, fontsize=7.6)

    fig.text(.018, .007, "Schematic only. Source-aware routing occurs at the encoder; the decoder reads the transmitted model ID."
             " End-to-end time must include routing, entropy, transfers and assembly.",
             color=MUTED, fontsize=7.25)
    save(fig, output, "fig03_router_system")


def plot_training_diagram(output: Path) -> None:
    fig, ax = plt.subplots(figsize=(11.6, 5.0), layout="constrained")
    ax.set(xlim=(0, 12), ylim=(0, 5.1))
    ax.axis("off")
    ax.text(.2, 4.88, "Learning decisions, then paying for only one path", fontsize=16.6,
            fontweight="bold", color=INK)
    ax.text(.2, 4.59, "Candidate outcomes are measured offline; deployment sees only the source and budget.",
            fontsize=9.2, color=MUTED)
    ax.plot([6.0, 6.0], [.40, 4.30], color="#B7C1C8", lw=1, ls=(0, (3, 4)))

    # OFFLINE: original image -> six actual codec outputs -> objective.
    ax.text(.35, 4.22, "01   OFFLINE TARGETS", color=COLORS[8], fontweight="bold", fontsize=9.2)
    patch_mosaic(ax, .45, 2.29, 1.22, 1.35, rows=4, cols=4, alpha=.75)
    ax.text(.46, 2.01, "router-train image", color=MUTED, fontsize=7.6)
    for idx, depth in enumerate(DEPTHS):
        yy = 3.47 - idx*.39
        ax.plot([1.86, 2.42], [2.97, yy], color=COLORS[depth], alpha=.48, lw=1.0)
        box(ax, 2.43, yy-.12, .54, .23, COLORS[depth], radius=.06, alpha=.75)
        ax.text(2.70, yy, f"D{depth}", ha="center", va="center", fontsize=7.2, color="white")
        ax.plot([2.99, 3.35], [yy, yy], color=COLORS[depth], alpha=.6, lw=.9)
    ax.text(3.39, 3.76, "candidate outcomes", color=INK, fontsize=8.3, fontweight="bold")
    ax.plot([3.48, 4.65], [1.98, 1.98], color="#8795A0", lw=.8)
    ax.plot([3.48, 3.48], [1.98, 3.50], color="#8795A0", lw=.8)
    ax.plot([4.16, 4.16], [2.05, 3.46], color="#A5AFB8", lw=.8, ls=(0, (2, 3)))
    ax.text(4.17, 3.48, "B", color=MUTED, fontsize=7.2)
    ax.text(3.51, 1.75, "actual bytes →", color=MUTED, fontsize=7.2)
    ax.text(3.18, 2.80, "error", rotation=90, color=MUTED, fontsize=7.2)
    dots = [(2, 3.65, 3.13, .052), (4, 3.84, 2.85, .058),
            (6, 4.00, 2.76, .061), (8, 4.18, 2.53, .066),
            (10, 4.37, 2.37, .071), (12, 4.55, 2.27, .077)]
    for depth, xx, yy, radius in dots:
        ax.add_patch(Circle((xx, yy), radius, color=COLORS[depth], alpha=.9))
    ax.text(3.42, 1.48, "bubble size: time (schematic)", color=MUTED, fontsize=7.0)
    arrow(ax, 4.64, 2.72, 4.90, 2.72, color=INK, lw=1.2)
    box(ax, 4.96, 2.14, .73, 1.15, "#E9EDF1", radius=.12)
    ax.text(5.325, 2.76, "budget\noracle", ha="center", va="center", color=INK,
            fontsize=8.2, fontweight="bold")
    ax.text(.48, .94, "Measure actual bytes, Y/U/V error and time for all six paths.",
            color=MUTED, fontsize=7.8)
    ax.text(.48, .61, "Labels use the assembled image; split train/calibration/test by source image.",
            color=MUTED, fontsize=7.8)

    # ONLINE: cheap source statistics -> MLP -> predicted six cost vectors -> one execution.
    ax.text(6.33, 4.22, "02   DEPLOYED DECISION", color=COLORS[6], fontweight="bold", fontsize=9.2)
    for ix, label in enumerate(("luma", "texture", "color", "QP")):
        box(ax, 6.35+ix*.74, 3.25, .64, .38, "#E8EDF0", radius=.07)
        ax.text(6.67+ix*.74, 3.44, label, ha="center", va="center", color=MUTED, fontsize=7.3)
    arrow(ax, 9.38, 3.44, 9.68, 3.44, color=INK, lw=1.15)
    box(ax, 9.74, 3.12, .68, .63, "#DDF0EC", radius=.12)
    ax.text(10.08, 3.44, "MLP", va="center", ha="center", fontsize=8.7,
            color=COLORS[6], fontweight="bold")
    arrow(ax, 10.45, 3.44, 10.74, 3.44, color=INK, lw=1.15)
    for ix, depth in enumerate(DEPTHS):
        yy = 3.83 - ix*.41
        ax.text(10.76, yy, f"{depth}", va="center", color=COLORS[depth], fontsize=7.8)
        box(ax, 10.96, yy-.07, .20+.43*(1-ix/6), .14, COLORS[depth], radius=.04,
            alpha=.62 if depth != 4 else 1)
    ax.text(10.62, 1.13, "one selected\ncodec path", color=INK,
            fontsize=8.2, ha="left", fontweight="bold")
    ax.text(6.36, 2.65, "MLP predicts  R̂ₖ, ŜYₖ, ŜUₖ, ŜVₖ  for each expert k",
            color=INK, fontsize=8.4)
    ax.text(6.36, 2.37, "time comes from a device / batch-size lookup",
            color=MUTED, fontsize=7.7)
    ax.text(6.36, 2.10, "score = predicted distortion + λ·rate + β·time",
            color=COLORS[6], fontsize=9.2, fontweight="bold")
    ax.text(6.36, 1.57, "calibrate on held-out images; penalize mode-map fragmentation",
            color=MUTED, fontsize=7.8)
    ax.text(6.36, 1.23, "group patches by selected model before codec execution",
            color=MUTED, fontsize=7.8)

    # A ribbon emphasizes measured cost components, not fictional gains.
    box(ax, .43, .20, 11.12, .24, "#EFF2F4", radius=.06)
    ax.text(5.99, .32,
            "ACCOUNTING: router + selected encoder + entropy bytes + model IDs + selected decoder + tile assembly + transfers",
            ha="center", va="center", fontsize=7.65, color=INK)
    fig.text(.02, .008, "Algorithm schematic; bar lengths and colors are illustrative, not measured quality or runtime."
             " Full six-codec search is an offline upper bound and must be charged if run at inference.",
             color=MUTED, fontsize=7.25)
    save(fig, output, "fig04_router_learning")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", type=Path, default=Path("/data10/shareddata/can_karsal/dcvcuf_depth_20260927"))
    parser.add_argument("--output", type=Path, default=Path(__file__).resolve().parent)
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=True)
    setup()
    sources = {}
    models, paired = kodak_data(args.root, sources)
    plot_kodak(models, paired, args.output)
    training = plot_training(args.root, sources, args.output)
    plot_system_diagram(args.output)
    plot_training_diagram(args.output)
    evidence = {
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "metric": "weighted mean of plane PSNR: (6Y+U+V)/8; YCbCr 4:4:4",
        "rate": "deterministic entropy estimate, not actual bytes",
        "kodak": {"n_images": 24, "final_depths": [2, 4], "released_anchor": 12,
                   "paired_yuv_vs_released": paired},
        "training": training,
        "sources_sha256": sources,
        "gpu_inference_performed": False,
        "training_process_modified": False,
        "diagrams_are_schematic": True,
    }
    (args.output / "figure_evidence.json").write_text(json.dumps(evidence, indent=2) + "\n")
    print(json.dumps({"generated_at_utc": evidence["generated_at_utc"],
                      "kodak": evidence["kodak"], "training_status": training["status"],
                      "files": sorted(p.name for p in args.output.glob("fig*.pdf"))}, indent=2))


if __name__ == "__main__":
    main()
