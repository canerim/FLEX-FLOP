"""Nature's figure conventions, applied once so every figure inherits them.

From Nature's author guidelines (nature.com/nature/for-authors/final-submission
and the final-artwork guide): sans-serif throughout, Helvetica or Arial, the same
face in every figure; panel labels 8 pt bold lower-case a, b, c; all other text
at most 7 pt and never below 5 pt; line weight at least 0.25 pt; column widths
89 mm single, 183 mm double, 120 mm for the 1.5-column case; light background,
minimal decoration; colour-blind-safe palettes, and red/green pairs avoided.

The palette below is Okabe-Ito, which is designed to stay separable under
deuteranopia and protanopia -- the standard choice when a journal asks for
colour-blind-safe and does not supply one.
"""
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

MM = 1 / 25.4
W1, W15, W2 = 89 * MM, 120 * MM, 183 * MM      # Nature column widths, inches

# Okabe-Ito, colour-blind safe. Deliberately no red/green pairing.
BLUE, ORANGE, SKY, GREEN = "#0072B2", "#E69F00", "#56B4E9", "#009E73"
YELLOW, VERM, PURPLE, BLACK = "#F0E442", "#D55E00", "#CC79A7", "#000000"
SERIES = [BLUE, ORANGE, GREEN, VERM, PURPLE, SKY]
INK, INK2, GRID = "#000000", "#4d4d4d", "#d9d9d9"

def apply():
    plt.rcParams.update({
        "font.family": "sans-serif",
        "font.sans-serif": ["DejaVu Sans", "Helvetica", "Arial"],
        "font.size": 7, "axes.labelsize": 7, "axes.titlesize": 7,
        "xtick.labelsize": 6, "ytick.labelsize": 6, "legend.fontsize": 6,
        "axes.linewidth": 0.5, "grid.linewidth": 0.4,
        "xtick.major.width": 0.5, "ytick.major.width": 0.5,
        "xtick.major.size": 2, "ytick.major.size": 2,
        "lines.linewidth": 1.0, "lines.markersize": 3,
        "figure.facecolor": "white", "axes.facecolor": "white",
        "savefig.facecolor": "white", "savefig.dpi": 300,
        "axes.grid": True, "grid.color": GRID, "axes.axisbelow": True,
        "axes.edgecolor": INK2, "text.color": INK,
        "axes.labelcolor": INK, "xtick.color": INK2, "ytick.color": INK2,
        "legend.frameon": False, "axes.spines.top": False,
        "axes.spines.right": False, "pdf.fonttype": 42, "ps.fonttype": 42,
    })

def panel(ax, letter, dx=-0.16, dy=1.06):
    """Nature panel label: 8 pt bold, lower case, upright.

    dx is a fraction of the axes width, so the same offset clears the tick
    labels on a wide panel and lands on top of them on a narrow one. The label
    is marked here and nudged left at save time if it turns out to collide;
    see _place_panels below.
    """
    t = ax.text(dx, dy, letter, transform=ax.transAxes, fontsize=8,
                fontweight="bold", va="top", ha="left", color=INK)
    t._ns_panel = True
    return t


# ---------------------------------------------------------------------------
# Collision and clipping audit, run at save time.
#
# The author's complaint was concrete: labels sitting on top of each other, and
# things that do not show up because they fall outside the axes. Neither is
# visible in the source and neither survives a reviewer. Both are measurable at
# the moment the figure is written, when a renderer exists and every artist has
# a position in pixels, so the audit hooks savefig rather than asking anyone to
# look. Every figure this repository produces goes through naturestyle.apply(),
# so hooking it once covers all of them.
# ---------------------------------------------------------------------------

import json as _json
import os as _os
import sys as _sys
from pathlib import Path as _Path

_AUDIT = _Path(__file__).resolve().parent.parent / "results/figure_audit.json"
_MIN_PX = 3.0        # ignore boxes thinner than this: empty or whitespace text
_FRAC = 0.15         # overlap counts once it covers this much of the smaller box


def _boxes(fig, rend):
    """Every visible piece of text in the figure, with its pixel box and role.

    Two things have to be filtered out or the audit is all noise. Matplotlib
    keeps tick objects for locations outside the current view and leaves them
    visible -- they are clipped at draw time, not hidden -- so their labels sit
    at the ends of the axis and collide with everything. And ticks belonging to
    one axis are laid out by matplotlib, so a pair of them is never a defect
    the script can fix.
    """
    from matplotlib.text import Text
    out = []
    for ai, ax in enumerate(fig.axes):
        if not ax.get_visible():
            continue
        managed = {}
        for which, axis, lim in (("xtick", ax.xaxis, ax.get_xlim()),
                                 ("ytick", ax.yaxis, ax.get_ylim())):
            lo, hi = min(lim), max(lim)
            for tk in axis.get_major_ticks() + axis.get_minor_ticks():
                loc = tk.get_loc()
                inside = loc is not None and lo - 1e-9 <= loc <= hi + 1e-9
                for lab in (tk.label1, tk.label2):
                    managed[id(lab)] = (which, inside)
        seen = set()
        for t in ax.findobj(Text):
            if id(t) in seen:
                continue
            seen.add(id(t))
            if not t.get_visible() or not (t.get_text() or "").strip():
                continue
            role, inside = managed.get(id(t), ("text", True))
            if not inside:
                continue                       # clipped by the axes; not drawn
            try:
                bb = t.get_window_extent(renderer=rend)
            except Exception:
                continue
            if bb.width < _MIN_PX or bb.height < _MIN_PX:
                continue
            out.append((f"{ai}:{role}", t.get_text().strip()[:40], bb, t))
    return out


def _overlaps(items):
    """Pairs of texts that share pixels, ignoring a text inside its own frame."""
    bad = []
    for i in range(len(items)):
        for j in range(i + 1, len(items)):
            (r1, s1, b1, t1), (r2, s2, b2, t2) = items[i], items[j]
            # Ticks of one axis are matplotlib's layout, not the script's.
            if r1 == r2 and r1.split(":")[1] in ("xtick", "ytick"):
                continue
            # A tick label and the axis label it sits under belong to different
            # artists but are laid out by matplotlib and never collide by
            # accident; the collisions worth reporting are between things a
            # script placed itself.
            x = min(b1.x1, b2.x1) - max(b1.x0, b2.x0)
            y = min(b1.y1, b2.y1) - max(b1.y0, b2.y0)
            if x <= 0 or y <= 0:
                continue
            area = x * y
            small = min(b1.width * b1.height, b2.width * b2.height)
            if small > 0 and area / small >= _FRAC:
                bad.append({"a": s1, "b": s2,
                            "frac": round(area / small, 2)})
    return bad


def _outside(fig):
    """Plotted points that fall outside the axes and so are never seen."""
    from matplotlib.lines import Line2D
    bad = []
    for ax in fig.axes:
        if not ax.get_visible():
            continue
        (x0, x1), (y0, y1) = ax.get_xlim(), ax.get_ylim()
        x0, x1 = min(x0, x1), max(x0, x1)
        y0, y1 = min(y0, y1), max(y0, y1)
        for ln in ax.get_lines():
            if not isinstance(ln, Line2D) or not ln.get_visible():
                continue
            # Legend handles and axis-fraction guides are Line2D too, and their
            # coordinates are not data, so comparing them to the view limits
            # reports every one of them as invisible.
            if ln.get_transform() is not ax.transData:
                continue
            try:
                xs, ys = ln.get_xdata(orig=False), ln.get_ydata(orig=False)
            except Exception:
                continue
            n = len(xs)
            if n == 0 or len(ys) != n:
                continue
            out = sum(1 for a, b in zip(xs, ys)
                      if not (x0 <= a <= x1 and y0 <= b <= y1))
            # A curve that runs off the frame is usually a deliberate zoom --
            # tradeoff.png crops its frontier at 0.30 dB on purpose. What is
            # never deliberate is a series with nothing inside the frame at
            # all, which is a limit set against the wrong data and a line the
            # reader never sees.
            if out == n:
                bad.append({"label": (ln.get_label() or "")[:30],
                            "outside": out, "n": n})
    return bad


def audit_enabled():
    return _os.environ.get("NS_NO_AUDIT") != "1"


def _record(path, over, out):
    try:
        db = _json.loads(_AUDIT.read_text()) if _AUDIT.exists() else {}
    except Exception:
        db = {}
    key = _Path(path).name
    # Every audited figure is recorded, clean or not, so that a checker can
    # tell a figure that passed from a figure nobody has looked at.
    db[key] = {"overlaps": over[:8], "outside": out[:8]}
    if over or out:
        print(f"  FIGURE AUDIT {key}: {len(over)} text collision(s), "
              f"{len(out)} series outside the axes", file=_sys.stderr)
        for o in over[:3]:
            print(f"     \"{o['a']}\" over \"{o['b']}\" ({o['frac']:.0%})",
                  file=_sys.stderr)
        for o in out[:3]:
            print(f"     {o['outside']}/{o['n']} points of "
                  f"\"{o['label']}\" outside", file=_sys.stderr)
    _AUDIT.parent.mkdir(parents=True, exist_ok=True)
    _AUDIT.write_text(_json.dumps(db, indent=2, sort_keys=True))


def _place_panels(fig, rend):
    """Move a panel label left until it clears the axis furniture beside it.

    Only labels that actually collide move, and they move in whole tenths of
    the axes width, so a figure whose panels were already clear is written
    byte for byte as before.
    """
    from matplotlib.text import Text
    moved = 0
    for ax in fig.axes:
        labs = [t for t in ax.findobj(Text)
                if getattr(t, "_ns_panel", False) and t.get_visible()]
        if not labs:
            continue
        others = [t for t in ax.findobj(Text)
                  if t.get_visible() and (t.get_text() or "").strip()
                  and not getattr(t, "_ns_panel", False)]
        for t in labs:
            for _ in range(6):
                try:
                    bb = t.get_window_extent(renderer=rend)
                except Exception:
                    break
                hit = False
                for o in others:
                    try:
                        ob = o.get_window_extent(renderer=rend)
                    except Exception:
                        continue
                    x = min(bb.x1, ob.x1) - max(bb.x0, ob.x0)
                    y = min(bb.y1, ob.y1) - max(bb.y0, ob.y0)
                    if x > 0 and y > 0:
                        hit = True
                        break
                if not hit:
                    break
                px, py = t.get_position()
                t.set_position((px - 0.05, py))
                moved += 1
                fig.canvas.draw()
                rend = fig.canvas.get_renderer()
    return moved


def _install():
    from matplotlib.figure import Figure
    if getattr(Figure.savefig, "_ns_audited", False):
        return
    orig = Figure.savefig

    def savefig(self, fname, *a, **kw):
        if audit_enabled() and isinstance(fname, (str, _Path)) \
                and str(fname).endswith(".png"):
            try:
                self.canvas.draw()
                _place_panels(self, self.canvas.get_renderer())
            except Exception as e:
                print(f"  panel placement skipped for {fname}: {e}",
                      file=_sys.stderr)
        r = orig(self, fname, *a, **kw)
        if audit_enabled() and isinstance(fname, (str, _Path)) \
                and str(fname).endswith(".png"):
            try:
                # Draw again before measuring. savefig(bbox_inches="tight")
                # leaves the axis labels positioned against the expanded bbox,
                # and measuring in that state put window.png's x-axis label
                # 54 pixels above where it is drawn. A plain draw restores the
                # geometry the reader sees.
                self.canvas.draw()
                rend = self.canvas.get_renderer()
                _record(fname, _overlaps(_boxes(self, rend)), _outside(self))
            except Exception as e:                      # never break a build
                print(f"  figure audit skipped for {fname}: {e}",
                      file=_sys.stderr)
        return r

    savefig._ns_audited = True
    Figure.savefig = savefig


_install()
