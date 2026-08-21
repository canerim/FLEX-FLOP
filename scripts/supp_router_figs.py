"""Two figures for the supplement's router section, from results/ alone.

Nothing here opens a checkpoint, a CUDA context or a dataset. Both figures are
read out of the JSON files named in `SOURCES` below, so they carry the same
provenance as the tables beside them and can be redrawn on a laptop.

    ./.venv/bin/python scripts/supp_router_figs.py

writes paper/figures/supp_router_inputs.png and
paper/figures/supp_router_frontier.png.
"""
import json
import sys
from pathlib import Path

import numpy as np

R = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(R / "scripts"))
import naturestyle as ns                                        # noqa: E402
ns.apply()
import matplotlib.pyplot as plt                                 # noqa: E402

RES = R / "results"
OUT = R / "paper" / "figures"

SOURCES = [
    "router_ablation.json",
    "raterank_RECIPE512_b01.json", "raterank_RECIPE512_b03.json",
    "router_RECIPE512_b01.json", "router_RECIPE512_b03.json",
    "saturation_RECIPE512_ctc53.json",
]

#: The order the ablation is read in: the two heads that tie first, then the
#: single-group runs by descending agreement, then the no-per-tile control.
ORDER = ["stem", "all", "latent", "scales", "bits", "qp"]
SHORT = {"stem": "stem", "all": "all five", "latent": "latent",
         "scales": "scales", "bits": "bits", "qp": "q only"}


def J(name):
    return json.loads((RES / name).read_text())


def joint(tag):
    """The jointly trained head's curve at one budget.

    Written first as router_RECIPE512_b0*_PAPER.json and later renamed to
    _jointhead; both names are accepted so the figure redraws either side of
    that rename. scripts/supp/f_router.py resolves it the same way.
    """
    for pat in (f"router_RECIPE512_{tag}_jointhead.json",
                f"router_RECIPE512_{tag}_PAPER.json"):
        if (RES / pat).exists():
            return J(pat)
    raise FileNotFoundError(f"no jointly trained head curve for {tag}")


def sv(d, qp):
    for r in d["rows"]:
        if r["qp"] == qp and r.get("budget_reachable", True):
            return r["saving_pct_vs_release"]
    return np.nan


def inputs_figure():
    A = J("router_ablation.json")
    V = {x["label"]: x for x in A["variants"]}
    floor = V["stem"]["constant_best_agree"]

    fig, (a, b) = plt.subplots(1, 2, figsize=(ns.W1 * 1.42, 1.55),
                               gridspec_kw={"width_ratios": [1.0, 1.12],
                                            "wspace": 0.42})
    y = np.arange(len(ORDER))[::-1]

    a.barh(y, [V[l]["agree"] for l in ORDER],
           xerr=[V[l]["stderr"] for l in ORDER],
           color=[ns.BLUE if l in ("stem", "all") else ns.SKY for l in ORDER],
           height=0.62, error_kw=dict(lw=0.6, capsize=1.6, ecolor=ns.INK2))
    a.axvline(floor, color=ns.VERM, lw=0.8, ls="--", zorder=3)
    a.text(floor + 0.008, y[0] + 0.55, "best constant exit", fontsize=6,
           color=ns.VERM, va="center")
    a.set_yticks(y, [SHORT[l] for l in ORDER])
    a.set_xlim(0.40, 0.83)
    a.set_xlabel("agreement with the oracle")
    a.grid(axis="y", visible=False)
    ns.panel(a, "a", dx=-0.34)

    ex = [2, 3, 4, 5]
    labels = ["oracle"] + [SHORT[l] for l in ORDER]
    oh = np.array(V["stem"]["oracle_hist"], dtype=float)
    mat = [100 * oh[2:] / oh.sum()]
    for l in ORDER:
        ph = np.array(V[l]["pred_hist"], dtype=float)
        mat.append(100 * ph[2:] / ph.sum())
    mat = np.array(mat)
    yy = np.arange(len(labels))[::-1]
    left = np.zeros(len(labels))
    shades = ["#08306b", "#2171b5", "#6baed6", "#c6dbef"]
    for i, e in enumerate(ex):
        b.barh(yy, mat[:, i], left=left, height=0.62, color=shades[i],
               label=f"exit {e}", edgecolor="white", linewidth=0.3)
        left += mat[:, i]
    b.set_yticks(yy, labels)
    b.set_xlim(0, 100)
    b.set_xlabel("share of tiles (%)")
    b.grid(visible=False)
    b.legend(ncol=4, fontsize=6, loc="lower center",
             bbox_to_anchor=(0.5, 1.0), handlelength=1.0, columnspacing=0.8,
             handletextpad=0.4)
    ns.panel(b, "b", dx=-0.30)

    OUT.mkdir(parents=True, exist_ok=True)
    fig.savefig(OUT / "supp_router_inputs.png", dpi=400, bbox_inches="tight")
    plt.close(fig)
    print("  -> paper/figures/supp_router_inputs.png")


def frontier_figure():
    ceil = J("saturation_RECIPE512_ctc53.json")["ceiling_pct"]
    qs = [0, 16, 32, 48, 63]
    x = np.arange(len(qs))
    panels = [("0.1 dB", "b01"), ("0.3 dB", "b03")]

    fig, axes = plt.subplots(1, 2, figsize=(ns.W1 * 1.42, 1.6),
                             sharey=True, gridspec_kw={"wspace": 0.10})
    for ax, (title, tag) in zip(axes, panels):
        rr = J(f"raterank_RECIPE512_{tag}.json")
        hj = joint(tag)
        hf = J(f"router_RECIPE512_{tag}.json")
        ax.axhline(ceil, color=ns.INK2, lw=0.6, ls=":", zorder=1)
        ax.plot(x, [rr["rows"][i]["oracle_saving_pct_vs_release"]
                    for i in range(len(qs))], "-", color=ns.INK2, lw=0.8,
                marker="_", label="oracle")
        ax.plot(x, [sv(rr, q) for q in qs], "-o", color=ns.BLUE,
                label="calibrated bit rule")
        ax.plot(x, [sv(hj, q) for q in qs], "-s", color=ns.ORANGE,
                label="head, joint")
        ax.plot(x, [sv(hf, q) for q in qs], "-^", color=ns.GREEN,
                label="head, frozen")
        ax.set_xticks(x, [str(q) for q in qs])
        ax.set_xlabel("quality index $q$")
        ax.set_title(title, fontsize=6.5)
    axes[0].set_ylabel("saved (% of released decode)")
    axes[1].text(len(qs) - 1.05, ceil + 0.8, "ceiling", fontsize=6,
                 color=ns.INK2, ha="right")
    axes[0].legend(ncol=2, fontsize=6, loc="lower left", handlelength=1.3,
                   columnspacing=0.8, handletextpad=0.4)
    OUT.mkdir(parents=True, exist_ok=True)
    fig.savefig(OUT / "supp_router_frontier.png", dpi=400, bbox_inches="tight")
    plt.close(fig)
    print("  -> paper/figures/supp_router_frontier.png")


if __name__ == "__main__":
    missing = [s for s in SOURCES if not (RES / s).exists()]
    if missing:
        sys.exit(f"missing results files: {', '.join(missing)}")
    inputs_figure()
    frontier_figure()
