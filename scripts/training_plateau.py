"""The loss stops moving long before the method does.

Reading the training loss of this run would tell you to stop at the first
epoch: its median is flat to within a few hundredths from epoch 0 onward. The
saving the paper reports climbs by a third over the same span. Those two facts
are consistent, and the reason is the objective's shape. The RD loss is
dominated by the deepest exit, which is a fine-tune of an already converged
decoder and has almost nothing left to gain; what keeps improving is the
agreement between the exits, which no term in the loss reports on its own and
which is the whole of what the saving depends on.

  a  the two curves that disagree: median training loss per epoch, flat, and
     the median ladder span over the same epochs, falling
  b  what that buys, measured: mean saving at the 0.1 dB budget on the 53 CTC
     intra frames, one point per epoch, with the reported checkpoint circled

Panel b is the honest version of the limitation the paper states in prose. The
run has not plateaued in the quantity that matters, so every number in the
paper is a lower bound on what this recipe reaches.

    python scripts/training_plateau.py [--layout column] [--out PATH]
"""
from __future__ import annotations

import argparse
import json
import statistics as st
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "scripts"))

import naturestyle as ns          # noqa: E402
import matplotlib.pyplot as plt   # noqa: E402
from savings import epoch_series  # noqa: E402




def panel_loss_and_span(ax, eps, loss, span):
    """What the optimiser sees, against what the method needs."""
    ax.plot(eps, loss, "-o", color=ns.VERM, lw=1.2, ms=4,
            label="training loss (median)")
    ax.set_xlabel("epoch")
    ax.set_ylabel("training loss", color=ns.VERM)
    ax.tick_params(axis="y", colors=ns.VERM)
    # A fixed floor at zero so a flat curve looks flat. Autoscaled, a range of
    # 0.03 fills the panel and the loss appears to be doing something.
    ax.set_ylim(0, max(loss) * 1.6)

    ax2 = ax.twinx()
    ax2.plot(eps, span, "--s", color=ns.BLUE, lw=1.2, ms=4,
             label="ladder span (median)")
    ax2.set_ylabel("span, deepest − shallowest (dB)", color=ns.BLUE)
    ax2.tick_params(axis="y", colors=ns.BLUE)
    ax2.set_ylim(0, max(span) * 1.25)
    ax2.grid(False)
    h1, l1 = ax.get_legend_handles_labels()
    h2, l2 = ax2.get_legend_handles_labels()
    ax.legend(h1 + h2, l1 + l2, loc="lower left", handlelength=1.8,
              labelspacing=0.35)


def panel_saving(ax, ser, label=True):
    """The measured saving per epoch, with the reported checkpoint circled."""
    if not ser:
        return
    xs, ys = list(ser), [ser[e] for e in ser]
    ax.plot(xs, ys, "-o", color=ns.GREEN, lw=1.3, ms=4.5)
    paper = max(ser)
    ax.plot([paper], [ser[paper]], "o", ms=9, mfc="none", mec=ns.BLACK,
            mew=1.2, zorder=4)
    if label:
        ax.annotate("reported", (paper, ser[paper]),
                    textcoords="offset points", xytext=(-6, -13),
                    fontsize=6, ha="right", color=ns.BLACK)
    ax.set_ylim(0, max(ys) * 1.35)
    ax.set_xticks(xs)
    ax.set_xlabel("epoch")
    ax.set_ylabel("decoder MACs saved (%)")


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--log", default="runs/RECIPE512/train_log.jsonl")
    ap.add_argument("--out", default="docs/figures/training_plateau.png")
    # The main paper caps a column figure at 1.25 inches tall, and a two-panel
    # full-width figure squeezed into one column renders its 8 pt labels at
    # under 4. The paper gets the panel that carries the claim; the supplement
    # gets the mechanism beside it.
    ap.add_argument("--layout", choices=("wide", "column"), default="wide")
    a = ap.parse_args()
    ns.apply()
    wide = a.layout == "wide"

    rows = [json.loads(l) for l in
            (ROOT / a.log).read_text().splitlines() if l.strip()]
    # Only complete epochs: a partial one is a median over a different part of
    # the schedule and is not comparable with the rest.
    n_full = max(len([r for r in rows if r["epoch"] == e])
                 for e in {r["epoch"] for r in rows})
    eps = sorted(e for e in {r["epoch"] for r in rows}
                 if len([r for r in rows if r["epoch"] == e]) >= n_full)
    loss = [st.median([r["loss"] for r in rows if r["epoch"] == e])
            for e in eps]
    span = [st.median([r["spread_dB"] for r in rows if r["epoch"] == e
                       and r.get("spread_dB") is not None]) for e in eps]
    ser = epoch_series()

    if wide:
        fig, ax = plt.subplots(1, 2, figsize=(ns.W2, 2.4))
        panel_loss_and_span(ax[0], eps, loss, span)
        panel_saving(ax[1], ser)
        ns.panel(ax[0], "a")
        ns.panel(ax[1], "b", dx=-0.20)
        fig.tight_layout(w_pad=2.6)
    else:
        fig, ax = plt.subplots(1, 1, figsize=(ns.W1, 1.35))
        panel_saving(ax, ser, label=False)
        fig.tight_layout()

    out = ROOT / a.out
    out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out, dpi=500, bbox_inches="tight", pad_inches=0.02,
                facecolor="white")
    plt.close(fig)
    print(f"  -> {a.out}")
    print(f"     complete epochs {eps}")
    print(f"     loss   {[round(v, 3) for v in loss]}")
    print(f"     span   {[round(v, 3) for v in span]}")
    print(f"     saving {{{', '.join(f'{e}: {v:.1f}%' for e, v in ser.items())}}}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
