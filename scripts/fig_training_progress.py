"""Ladder spread and deepest-exit quality against training step, for the live runs.

Spread on its own is not a health signal, and reading it as one is the trap this
figure exists to avoid. `spread_dB` is deepest minus shallowest, so it falls
either because the shallow exits improved -- what we want -- or because the
deepest exit was dragged down, which would destroy the whole premise, since the
deepest exit IS released DCVC-UF at initialisation and the saving is quoted
against it.

Panel b was first drawn as the deepest exit's absolute PSNR, and that was
useless: it swung over 7 dB from batch to batch, because each point carries the
content of one random crop far more than the state of the model, and any drift
was invisible underneath. `anchor_mse` is the right quantity -- our deepest exit
against the frozen released decoder ON THE SAME BATCH -- so the content largely
cancels and what remains is drift. Reported as fidelity, 10 log10(1/mse), since
the network operates on [-0.5, 0.5] and the peak-to-peak range is 1.

VERBATIM has no curve in panel b, and that is correct rather than missing data:
it is Microsoft's recipe with NONE of the additions, so it carries no anchor
term and nothing was ever computed.

Still an indication, not a verdict: these are training batches. The definitive
statement is anchor_drift.py on a checkpoint, over fixed CTC frames.

The four runs also differ in more than one thing at a time -- BEST carries
distillation, scaled adapters, anchor weight 10 and a joint router, CONTROL
none of them -- so the gap between them is the effect of that BUNDLE, not of
any single ingredient. The isolations are separate experiments.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

R = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(R / "scripts"))
import naturestyle as ns  # noqa: E402

ns.apply()
STEPS_PER_EPOCH = 47451
RUNS = [("BEST", ns.BLUE), ("CONTROL", ns.ORANGE), ("VERBATIM", ns.GREEN),
        ("RECIPE512", ns.PURPLE), ("BEST128", ns.SKY), ("FINE12", ns.VERM)]


def trajectory(tag: str):
    """(cumulative step, spread, deepest) with restart rewinds removed.

    A relaunched run writes its step counter from zero again, so the raw log is
    not monotone in training progress. Keeping only the final monotone tail
    means the figure shows the run that is actually alive rather than splicing
    two together.
    """
    p = R / "runs" / tag / "train_log.jsonl"
    if not p.exists():
        return None
    rs = [json.loads(l) for l in p.read_text().splitlines() if l.strip()]
    if not rs:
        return None
    g = lambda r: r["epoch"] * STEPS_PER_EPOCH + r["step"]  # noqa: E731
    tail = [rs[-1]]
    for r in reversed(rs[:-1]):
        if g(r) < g(tail[0]):
            tail.insert(0, r)
        else:
            break
    return (np.array([g(r) for r in tail]),
            np.array([r["spread_dB"] for r in tail]),
            [r.get("anchor_mse") for r in tail])


def rolling(y, w=15):
    """Median over a centred window. Median, not mean: one batch of flat sky is
    an outlier in PSNR, not a trend, and a mean would let it move the curve."""
    if len(y) < 3:
        return y
    w = min(w, len(y) | 1)
    pad = w // 2
    yy = np.pad(y, pad, mode="edge")
    return np.array([np.median(yy[i:i + w]) for i in range(len(y))])


fig, ax = plt.subplots(1, 2, figsize=(ns.W2, 2.1))
any_drawn = False
for tag, c in RUNS:
    t = trajectory(tag)
    if t is None or len(t[0]) < 3:
        continue
    any_drawn = True
    step, spread, amse = t
    ax[0].plot(step / 1000, rolling(spread), color=c, lw=1.0, label=tag)
    keep = [i for i, v in enumerate(amse) if v]        # 0.0 at step 0 is not a
    if len(keep) >= 3:                                 # measurement, it is init
        fid = 10 * np.log10(1.0 / np.array([amse[i] for i in keep]))
        ax[1].plot(step[keep] / 1000, rolling(fid), color=c, lw=1.0, label=tag)

if not any_drawn:
    raise SystemExit("no run has enough log points yet")

ax[0].set_ylabel("Ladder spread (dB)")
ax[0].set_xlabel("Training step (thousands)")
ax[1].set_ylabel("Fidelity to released UF (dB)")
ax[1].set_xlabel("Training step (thousands)")
ax[0].legend(loc="upper right", ncol=2)
ns.panel(ax[0], "a")
ns.panel(ax[1], "b")
ax[0].set_title("Lower is a tighter ladder", fontsize=6, color=ns.INK2, loc="left")
ax[1].set_title("Higher is less drift from the reference decoder", fontsize=6,
                color=ns.INK2, loc="left")
ax[1].legend(loc="lower right", ncol=2, fontsize=5.5)
fig.tight_layout()
out = R / "results" / "nf_progress.png"
fig.savefig(out, dpi=300, bbox_inches="tight")
print(f"wrote {out.relative_to(R)}")
