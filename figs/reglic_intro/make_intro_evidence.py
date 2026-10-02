"""Render the introduction figure from archived source luma and exit maps."""

from pathlib import Path
import hashlib
import json

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.colors import BoundaryNorm, ListedColormap
from matplotlib.patches import Rectangle
import numpy as np


HERE = Path(__file__).resolve().parent
DATA = HERE.parents[1] / "data" / "gallery20260928"
meta = json.loads((DATA / "gallery.json").read_text())
archive_path = DATA / "source_luma.npz"
assert hashlib.sha256(archive_path.read_bytes()).hexdigest() == meta["luma_archive_sha256"]
archive = np.load(archive_path)

plt.rcParams.update({
    "font.family": "DejaVu Sans",
    "font.size": 7.1,
    "pdf.fonttype": 42,
    "svg.fonttype": "none",
    "savefig.facecolor": "white",
})
colors = ["#4a6a8b", "#51a69a", "#e2b650", "#bb6674"]
cmap = ListedColormap(colors)
norm = BoundaryNorm([5, 7, 9, 11, 13], cmap.N)

fig = plt.figure(figsize=(3.39, 3.15), facecolor="white")
gs = fig.add_gridspec(2, 2, left=.025, right=.975, top=.91, bottom=.165,
                      wspace=.035, hspace=.13)
fig.text(.025, .952, "SOURCE LUMA", fontsize=6.6, weight="bold", color="#27384b")
fig.text(.525, .952, "EXECUTED DEPTH", fontsize=6.6, weight="bold", color="#27384b")

# Fixed first and last examples from the predeclared eight-source gallery.
for ri, gallery_index in enumerate((0, 7)):
    row = meta["rows"][gallery_index]
    src = archive[str(gallery_index)]
    gh, gw = row["grid"]
    depth = 2 * (np.asarray(row["rules"]["router"]["0.1"]["map"])
                 .reshape(gh, gw) + 1)
    assert src.shape == tuple(row["hw"]) and depth.shape == (gh, gw)
    h, w = src.shape

    ax_src = fig.add_subplot(gs[ri, 0])
    ax_src.imshow(src, cmap="gray", vmin=0, vmax=255, interpolation="nearest")
    ax_src.set(xlim=(0, w), ylim=(h, 0))

    ax_map = fig.add_subplot(gs[ri, 1])
    ax_map.imshow(src, cmap="gray", vmin=0, vmax=255, interpolation="nearest")
    ax_map.imshow(depth, cmap=cmap, norm=norm, interpolation="nearest",
                  extent=(0, w, h, 0), alpha=.79)
    for x in np.linspace(0, w, gw + 1):
        ax_map.plot([x, x], [0, h], color="white", lw=.32, alpha=.85)
    for y in np.linspace(0, h, gh + 1):
        ax_map.plot([0, w], [y, y], color="white", lw=.32, alpha=.85)
    ax_map.set(xlim=(0, w), ylim=(h, 0))

    for ax in (ax_src, ax_map):
        ax.set_xticks([])
        ax.set_yticks([])
        for spine in ax.spines.values():
            spine.set_color("#d7dce3")
            spine.set_linewidth(.45)
    label = "UVG / YachtRide" if gallery_index == 0 else "HEVC-E / FourPeople"
    ax_src.text(.022, .94, label, transform=ax_src.transAxes, va="top",
                fontsize=6.2, color="white", weight="bold",
                bbox={"facecolor": "#1d2a36", "alpha": .83, "edgecolor": "none", "pad": 2})

fig.text(.025, .098, "Synthesis blocks executed", fontsize=6.5, color="#26384c")
for i, depth in enumerate((6, 8, 10, 12)):
    x = .48 + i * .115
    fig.add_artist(Rectangle((x, .089), .021, .021, transform=fig.transFigure,
                             facecolor=colors[i], edgecolor="none"))
    fig.text(x + .027, .092, str(depth), fontsize=6.7, color="#26384c")

for ext in ("pdf", "svg", "png"):
    dest = HERE / f"intro_evidence.{ext}"
    fig.savefig(dest, dpi=300)
    if ext == "svg":
        dest.write_text("\n".join(line.rstrip() for line in dest.read_text().splitlines()) + "\n")
plt.close(fig)
