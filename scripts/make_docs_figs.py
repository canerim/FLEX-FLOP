"""Figures for docs/: topology, rate-distortion, and the training loss.

Separate from make_figs_nature.py because these serve a different reader. Those
figures argue a result to someone who already knows the system; these explain
the system to someone meeting it.

    python scripts/make_docs_figs.py
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.patches import FancyBboxPatch, FancyArrowPatch

R = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(R / "scripts"))
sys.path.insert(0, str(R))
import naturestyle as ns  # noqa: E402

ns.apply()
OUT = R / "docs" / "figures"
OUT.mkdir(parents=True, exist_ok=True)
STEPS_PER_EPOCH = 47451


def J(name):
    p = R / "results" / name
    return json.loads(p.read_text()) if p.exists() else None


def deepest_cost(tag="BEST"):
    """Cost of the run's own deepest exit, in units of one stock decode.

    Not 1.0: with seam_repair="grid" it is 1.0095. Curves divide their saving
    by this, which measures early exiting against our own full-depth path; the
    figures claim "against the released DCVC-UF decoder", whose denominator is
    exactly 1. The two differ by 0.6-0.9 points, and NOT uniformly across runs
    -- VERBATIM runs seam_repair="none" and so has denominator 1.0 already, so
    the uncorrected numbers quietly favoured every run except VERBATIM in
    exactly the cross-run comparison the CONTROL/VERBATIM question turns on.
    """
    from flexuf.config import FlexUFConfig
    from flexuf.cost import exit_costs
    m = R / "runs" / tag / "meta.json"
    if not m.exists():
        return 1.0095
    return float(exit_costs(FlexUFConfig(**json.loads(m.read_text())["config"]),
                            "head")[-1])


def svr(row, D):
    """Saving against the release: stored if present, else exact algebra."""
    if row.get("saving_pct_vs_release") is not None:
        return row["saving_pct_vs_release"]
    s = row.get("saving_pct")
    return None if s is None else 100.0 - (100.0 - s) * D


def save(fig, name):
    # Both directories. build_pdf reads paper/figures, and a figure written
    # only here goes stale there without anything saying so: tradeoff.png sat
    # in paper/figures printing a 41.9% ceiling for days after the FFN
    # accounting fixed it to 39.1, next to a caption that said 38.3.
    for d in (OUT, R / "paper" / "figures"):
        d.mkdir(parents=True, exist_ok=True)
        fig.savefig(d / name, dpi=500, bbox_inches="tight", pad_inches=0.02,
                    facecolor="white")
    plt.close(fig)
    print(f"  {name}")


# ----------------------------------------------------------------- topology
def topology():
    """The decoder, its split, its exits, and which parts carry gradients.

    Drawn rather than described because three facts only land visually: the
    trunk is twelve identical blocks, the split puts the first j of them
    outside the per-tile region, and the router reads from the shared stem --
    so its input exists whether or not it is used.
    """
    from flexuf.config import FlexUFConfig
    from flexuf.cost import exit_costs
    from flexuf.model import FlexUFIntra

    cfg = FlexUFConfig(**json.loads((R / "runs/BEST/meta.json").read_text())["config"])
    cost = exit_costs(cfg, "head").tolist()
    # Read the adapter type off the built model instead of labelling every exit
    # "1x1". adapter_kind="scaled" deliberately gives the exits that stand in
    # for four or more skipped blocks an FFN and the rest a 1x1, so a uniform
    # label misdescribes four of the five adapters and understates their cost.
    _ad = getattr(FlexUFIntra(cfg).dec, "adapters", None)
    ADAPTER = {0: "FFN", 1: "1×1"}
    adapter_name = ([("FFN" if type(m).__name__.startswith("FFN") else "1×1")
                     for m in _ad] if _ad is not None else [])
    K, j = cfg.num_exits, cfg.split_depth
    per_exit = 12 // K

    fig, a = plt.subplots(figsize=(ns.W2, 3.0))
    a.set_xlim(0, 104); a.set_ylim(0, 50); a.axis("off")

    TRAIN, FROZEN = ns.BLUE, "#b8b8b8"
    y = 26
    a.add_patch(FancyBboxPatch((2, y), 9, 6, boxstyle="round,pad=0.3",
                               fc=FROZEN, ec="none"))
    a.text(6.5, y + 3, "encoder\nhyperprior\nentropy", ha="center", va="center",
           fontsize=6, color="#333333")
    a.text(6.5, y - 2.2, "FROZEN\n27.9M", ha="center", va="top", fontsize=6,
           color="#666666")

    a.add_patch(FancyBboxPatch((14, y), 7, 6, boxstyle="round,pad=0.3",
                               fc=TRAIN, ec="none"))
    a.text(17.5, y + 3, "upsample", ha="center", va="center", fontsize=6,
           color="white")
    a.text(17.5, y - 2.2, "8.2% of\ndecode MAC", ha="center", va="top",
           fontsize=6, color="#666666")

    # the twelve blocks
    x0, bw, gap = 24, 4.6, 0.7
    for b in range(12):
        x = x0 + b * (bw + gap)
        shared = b < j * per_exit
        a.add_patch(FancyBboxPatch((x, y), bw, 6, boxstyle="round,pad=0.2",
                                   fc=TRAIN, ec="none",
                                   alpha=1.0 if shared else 0.72))
        a.text(x + bw / 2, y + 3, str(b), ha="center", va="center",
               fontsize=6, color="white")
    trunk_end = x0 + 12 * (bw + gap) - gap

    a.add_patch(FancyBboxPatch((trunk_end + 2, y), 6, 6,
                               boxstyle="round,pad=0.3", fc=TRAIN, ec="none"))
    a.text(trunk_end + 5, y + 3, "head", ha="center", va="center", fontsize=6,
           color="white")
    a.text(trunk_end + 5, y - 2.2, "2.4%", ha="center", va="top", fontsize=6,
           color="#666666")

    # the split
    xs = x0 + j * per_exit * (bw + gap) - gap / 2
    a.plot([xs, xs], [y - 4, y + 11], color=ns.VERM, lw=1.0, ls=(0, (3, 2)))
    # Two labels, two heights: side by side at 5.5 pt they overlap at the split.
    a.text(xs - 1.5, y + 8.4, f"groups 0–{j-1}  ·  FULL-FRAME", ha="right",
           fontsize=6, color=ns.INK2)
    a.text(xs + 1.5, y + 11.6, f"groups {j}–{K-1}  ·  PER TILE  ·  the skippable "
           f"part, 89.4% of decode MAC", fontsize=6, color=ns.ORANGE)

    # Exits 0..j-1 exist in the ladder but end INSIDE the full-frame stem,
    # which runs once for the whole frame whether or not any tile stops early.
    # Their cost is therefore identical to exit j -- the allocator can never
    # prefer them, and a histogram that counts them separately understates
    # where the tiles actually sit. Drawing them is the point: the usable
    # ladder is K - j rungs, not K.
    for k in range(j):
        x = x0 + (k + 1) * per_exit * (bw + gap) - gap
        a.add_patch(FancyArrowPatch((x - bw / 2, y), (x - bw / 2, y - 4),
                                    arrowstyle="-|>", mutation_scale=6,
                                    color="#999999", lw=0.7))
        a.text(x - bw / 2, y - 5.4, f"exit {k}", ha="center", fontsize=6,
               color="#999999")
    a.text(x0, y - 16.6, f"exits 0–{j-1} end inside the full-frame stem: same\n"
           f"cost as exit {j}, so the allocator never picks them",
           fontsize=6, color="#999999", va="top")

    # exits
    for k in range(j, K):
        x = x0 + (k + 1) * per_exit * (bw + gap) - gap
        a.add_patch(FancyArrowPatch((x - bw / 2, y), (x - bw / 2, y - 7),
                                    arrowstyle="-|>", mutation_scale=7,
                                    color=ns.VERM, lw=0.8))
        sv = 100 * (1 - cost[k] / cost[-1])
        a.text(x - bw / 2, y - 8.5, f"exit {k}", ha="center", fontsize=6,
               color=ns.VERM)
        a.text(x - bw / 2, y - 11.2, f"{sv:.0f}% saved", ha="center", fontsize=6,
               color="#666666")
        nm = (adapter_name[k] if k < len(adapter_name) else None)
        a.text(x - bw / 2, y - 13.0,
               f"{nm} adapter" if nm else "no adapter\n(raw feature)",
               ha="center", va="top", fontsize=6, color="#888888")

    # router
    # The router reads the stem, i.e. the output of the last SHARED group, so
    # the arrow starts at the split rather than at an arbitrary block.
    a.add_patch(FancyBboxPatch((x0 + 1, y + 15.5), 21, 5,
                               boxstyle="round,pad=0.3", fc=ns.GREEN, ec="none"))
    a.text(x0 + 11.5, y + 18, "router head — 8.7K params", ha="center",
           va="center", fontsize=6, color="white")
    a.add_patch(FancyArrowPatch((x0 + 11.5, y + 15.5), (xs - 1, y + 6.4),
                                arrowstyle="-|>", mutation_scale=7,
                                color=ns.GREEN, lw=0.8,
                                connectionstyle="arc3,rad=-0.15"))
    # 384->16 1x1 over the 240x136 stem map at 1080p = 200.6 MMAC, against the
    # 453,540 MMAC decode. The MLP is 2,496 MAC x 40 tiles and rounds to zero.
    a.text(x0 + 24, y + 19.4, "reads the shared 240x136 stem map: 200.6 MMAC, "
           "0.044% of the 453,540 MMAC decode.",
           fontsize=6, color="#555555", va="center")
    # 94 bits per frame, measured -- results/signalled_*.json map_bits. Divided
    # by 1920x1080; an earlier caption quoted 1.2e-4, which is the 720p figure.
    a.text(x0 + 24, y + 16.8, "In the SHIPPED system the encoder picks the "
           "assignment instead and signals it: 94 bits/frame = 4.5e-5 bpp at 1080p.",
           fontsize=6, color="#555555", va="center")

    # "byte-identical bitstream" was too strong: it holds for the file only in
    # the decoder-side-router configuration. The SHIPPED system signals a ~94
    # bit map per frame, so the coded payload is identical and the file is not.
    a.text(2, 4, "blue = trained (decoder 17.49M + router 8.7K = 38.5% of the "
           "model)   ·   grey = frozen, so the coded payload is bit-identical "
           "to the release", fontsize=6, color="#444444")
    a.text(2, 1, f"K = {K} exits over 12 blocks, split depth j = {j}, "
           f"{cfg.rgb_patch}×{cfg.rgb_patch} px tiles", fontsize=6,
           color="#444444")
    save(fig, "topology.png")


# ------------------------------------------------------------- rate/quality
def rd_curve():
    """Absolute PSNR against bitrate, one curve per compute budget.

    The usual way this project reports -- dB BELOW the release -- hides how
    small the quality difference is next to the rate axis a codec paper is read
    on. Here the release and three compute budgets are plotted in the same
    units DCVC-UF's own results use.
    """
    anc = J("anchor_BEST_5qp.json") or J("anchor_BEST.json")
    why = J("why_qp.json")
    pc = J("curve_BEST.json")
    if not (anc and why and pc):
        print("  rd_curve: missing inputs")
        return
    rel = {r["qp"]: r["stock_psnr"] for r in anc["rows"]}
    bpp = {r["qp"]: r["bpp"] for r in why["rows"]}
    qps = sorted(set(rel) & set(bpp))
    if len(qps) < 3:
        print(f"  rd_curve: only {len(qps)} rates have both PSNR and bpp")
        return

    DEEPEST = deepest_cost("BEST")

    def saving_at(qp, db):
        pts = sorted((r.get("db_vs_uf_per_frame", r["db_vs_uf"]), svr(r, DEEPEST))
                     for r in pc["rows"] if r["qp"] == qp)
        D = [p[0] for p in pts]; S = [p[1] for p in pts]
        return float(np.interp(db, D, S, left=np.nan, right=np.nan))

    fig, ax = plt.subplots(1, 2, figsize=(ns.W2, 2.4))
    ax[0].plot([bpp[q] for q in qps], [rel[q] for q in qps], marker="o", ms=4,
               color=ns.INK, lw=1.2, label="released DCVC-UF (100% compute)")
    # Savings below are quoted against that black curve, denominator 1.0.
    for c, db in ((ns.BLUE, 0.05), (ns.ORANGE, 0.10), (ns.VERM, 0.19)):
        sv = [saving_at(q, db) for q in qps]
        y = [rel[q] - db for q in qps]
        lab = (f"{db:.2f} dB budget → "
               f"{np.nanmean(sv):.0f}% saved on average")
        ax[0].plot([bpp[q] for q in qps], y, marker="s", ms=3.5, color=c, lw=1.0,
                   label=lab)
    ax[0].set_xlabel("bitrate (bpp)")
    ax[0].set_ylabel("PSNR (dB), 6:1:1 in 4:2:0")
    ax[0].legend(loc="lower right", fontsize=6)
    ns.panel(ax[0], "a")
    ax[0].set_title("Rate-quality, 40 CTC sequences", fontsize=6, color=ns.INK2,
                    loc="left")

    # zoom: the quality axis a 0.1 dB budget actually spans
    ax[1].plot([bpp[q] for q in qps], [0.0] * len(qps), marker="o", ms=4,
               color=ns.INK, lw=1.2)
    for c, db in ((ns.BLUE, 0.05), (ns.ORANGE, 0.10), (ns.VERM, 0.19)):
        sv = [saving_at(q, db) for q in qps]
        ax[1].plot([bpp[q] for q in qps], [-db] * len(qps), color=c, lw=1.0,
                   ls=(0, (4, 2)))
        for q, s in zip(qps, sv):
            if s == s:
                ax[1].annotate(f"{s:.0f}%", (bpp[q], -db), fontsize=6, color=c,
                               ha="center", va="bottom",
                               textcoords="offset points", xytext=(0, 2))
    ax[1].set_xlabel("bitrate (bpp)")
    ax[1].set_ylabel("dB relative to the release")
    ax[1].set_ylim(-0.24, 0.05)
    ns.panel(ax[1], "b", dx=-0.18)
    ax[1].set_title("Same data, quality axis expanded; labels are compute saved",
                    fontsize=6, color=ns.INK2, loc="left")
    fig.tight_layout()
    save(fig, "rate_quality.png")


# ----------------------------------------------------------------- training
def loss_curves(tag="BEST"):
    """Loss and per-exit quality against step, with the epoch boundary marked.

    The question this answers is whether one epoch is enough -- whether the
    curve has flattened. Both panels are needed for that: the loss mixes rate
    and distortion over randomly sampled QPs, so it can flatten while the
    ladder is still moving.
    """
    p = R / "runs" / tag / "train_log.jsonl"
    if not p.exists():
        print(f"  loss_curves: no log for {tag}")
        return
    rs = [json.loads(l) for l in p.read_text().splitlines() if l.strip()]
    # Drop the pre-restart prefix: a relaunched run rewrites its step counter.
    tail = [rs[-1]]
    g = lambda r: r["epoch"] * STEPS_PER_EPOCH + r["step"]  # noqa: E731
    for r in reversed(rs[:-1]):
        if g(r) < g(tail[0]):
            tail.insert(0, r)
        else:
            break
    x = np.array([g(r) for r in tail]) / 1000.0
    loss = np.array([r["loss"] for r in tail])

    def roll(y, w=15):
        if len(y) < 3:
            return y
        w = min(w, len(y) | 1); pad = w // 2
        yy = np.pad(y, pad, mode="edge")
        return np.array([np.median(yy[i:i + w]) for i in range(len(y))])

    fig, ax = plt.subplots(1, 3, figsize=(ns.W2, 2.2))
    ax[0].plot(x, loss, color="#cccccc", lw=0.5)
    ax[0].plot(x, roll(loss), color=ns.BLUE, lw=1.2)
    ax[0].set_xlabel("training step (thousands)")
    ax[0].set_ylabel("loss (λ·MSE + bpp)")
    ax[0].set_title("Loss: flat after ~5k steps", fontsize=6, color=ns.INK2,
                    loc="left")
    # The loss is lambda*MSE + bpp and bpp swings twofold between batches --
    # 0.32 to 0.61 within ten consecutive records -- so it tracks the content
    # of the crop as much as the state of the model. A flat loss curve is
    # therefore weak evidence of convergence, and panel c is the answer to
    # "has it converged" that this panel cannot give.
    r_ = np.corrcoef(loss, [q["bpp"] for q in tail])[0, 1]
    ax[0].text(0.03, 0.06, f"corr(loss, batch bpp) = {r_:.2f}", fontsize=6,
               color=ns.VERM, transform=ax[0].transAxes)
    ns.panel(ax[0], "a")

    K = len(tail[0]["psnr_per_exit"])
    cols = [ns.SKY, ns.SKY, ns.GREEN, ns.YELLOW, ns.ORANGE, ns.VERM][:K]
    for k in range(2, K):
        y = roll(np.array([r["psnr_per_exit"][k] for r in tail]))
        ax[1].plot(x, y, color=cols[k], lw=1.0,
                   label="deepest" if k == K - 1 else f"exit {k}")
    ax[1].set_xlabel("training step (thousands)")
    ax[1].set_ylabel("PSNR on the batch (dB)")
    ax[1].legend(loc="lower right", fontsize=6, ncol=2)
    ax[1].set_title("Per-exit on the training batch", fontsize=6, color=ns.INK2,
                    loc="left")
    ns.panel(ax[1], "b", dx=-0.18)

    # c: the measurement that is not content-dominated -- fixed CTC frames, one
    # point per checkpoint. This is what "converged" has to mean here, and it
    # answers a question panel a cannot: the loss is flat from 5k steps on, but
    # the quantity the project reports is still climbing at 20k. Runs other
    # than `tag` are included because BEST has only reached one measured
    # checkpoint; the trajectory question is about the recipe, not the run.
    import glob as _g
    QP = 63  # the rate where the movement is largest, so a plateau would show
    runs = {}
    for f in sorted(_g.glob(str(R / "results/signalled_*.json"))):
        d = json.loads(Path(f).read_text())
        e, st = d.get("ckpt_epoch"), d.get("ckpt_step")
        if e is None:
            continue  # no provenance -> cannot be placed on a step axis
        r = Path(f).stem.split("_")[1]
        cum = (e + 1) * STEPS_PER_EPOCH if st is None else e * STEPS_PER_EPOCH + st
        v = next((svr(q, deepest_cost(r)) for q in d["rows"] if q["qp"] == QP),
                 None)
        if v is not None:
            runs.setdefault(r, {})[cum / 1000.0] = v  # dict: later file wins
    for c, (r, pts) in zip(ns.SERIES, sorted(runs.items())):
        xs = sorted(pts)
        ax[2].plot(xs, [pts[k] for k in xs], marker="o", ms=3.5, lw=1.0,
                   color=c, label=r, zorder=3 if r == tag else 2)
    ax[2].legend(loc="lower left", fontsize=6, ncol=1, frameon=False)
    ax[2].set_xlabel("cumulative training step (thousands)")
    ax[2].set_ylabel(f"saved vs release at 0.1 dB, qp {QP} (%)")
    # Not "still climbing": CONTROL falls, which is the open question of the
    # project and must not be papered over by a caption chosen for the runs
    # that behave.
    ax[2].set_title("Measured on fixed CTC frames", fontsize=6,
                    color=ns.INK2, loc="left")
    ns.panel(ax[2], "c", dx=-0.20)

    for a_ in ax[:2]:
        for e in range(1, 1 + int(x[-1] * 1000 // STEPS_PER_EPOCH)):
            a_.axvline(e * STEPS_PER_EPOCH / 1000, color=ns.INK2, lw=0.6,
                       ls=(0, (3, 2)))
            a_.text(e * STEPS_PER_EPOCH / 1000, a_.get_ylim()[1],
                    f" epoch {e}", fontsize=6, color=ns.INK2, va="top")
    fig.suptitle(f"{tag}: one epoch is 47,451 steps", fontsize=6, color=ns.INK2,
                 x=0.005, ha="left")
    fig.tight_layout(rect=(0, 0, 1, 0.95), w_pad=2.4)
    save(fig, f"training_{tag}.png")


# ------------------------------------------------------------- A/B decision
def ab_compare():
    """Encoder search versus decoder prediction, on one checkpoint.

    The two differ only in where the exit assignment comes from. Everything
    else -- checkpoint, frames, budget, dB convention, denominator -- is held
    identical, because the whole point is to price one design choice.
    """
    A = J("signalled_BEST_0817_1542.json")
    B = J("router_BEST_v2.json")
    if not (A and B):
        print("  ab_compare: missing a curve")
        return
    D = deepest_cost("BEST")
    a = {r["qp"]: svr(r, D) for r in A["rows"] if r.get("saving_pct") is not None}
    b = {r["qp"]: svr(r, D) for r in B["rows"] if r.get("saving_pct") is not None}
    qs = sorted(set(a) & set(b))

    fig, ax = plt.subplots(1, 2, figsize=(ns.W2, 2.4))
    x = np.arange(len(qs))
    ax[0].bar(x - 0.19, [a[q] for q in qs], 0.36, color=ns.BLUE,
              label="A · encoder searches, map signalled (+94 bit/frame)")
    ax[0].bar(x + 0.19, [b[q] for q in qs], 0.36, color=ns.ORANGE,
              label="B · decoder predicts, nothing sent (byte-identical)")
    ax[0].axhline(30, color=ns.INK2, lw=0.7, ls=(0, (3, 2)))
    ax[0].text(len(qs) - 0.5, 30.5, "30% target", fontsize=6, color=ns.INK2,
               ha="right")
    ax[0].set_xticks(x); ax[0].set_xticklabels([f"qp {q}" for q in qs])
    ax[0].set_ylabel("decode compute saved at 0.1 dB (%)")
    ax[0].legend(loc="upper right", fontsize=6, frameon=False)
    ax[0].set_ylim(0, 42)
    ax[0].set_title("Same checkpoint, same frames, same budget", fontsize=6,
                    color=ns.INK2, loc="left")
    ns.panel(ax[0], "a")

    gap = [a[q] - b[q] for q in qs]
    ax[1].plot(x, gap, marker="o", ms=4, color=ns.VERM, lw=1.2)
    for i, g in enumerate(gap):
        ax[1].annotate(f"{g:.1f}", (i, g), fontsize=6, color=ns.VERM,
                       ha="center", va="bottom",
                       textcoords="offset points", xytext=(0, 3))
    # The router's own arithmetic is already inside B, so it is a floor under
    # the gap rather than a term to add: even a perfect predictor cannot close
    # the last 0.163 points.
    ax[1].axhline(0.1629, color=ns.INK2, lw=0.7, ls=(0, (1, 2)))
    ax[1].text(0, 0.35, "0.163% — the router's own compute, already charged",
               fontsize=6, color=ns.INK2)
    ax[1].set_xticks(x); ax[1].set_xticklabels([f"qp {q}" for q in qs])
    ax[1].set_ylabel("points of saving given up (A − B)")
    ax[1].set_ylim(0, 7.5)
    ax[1].set_title("The price of an unchanged bitstream — an upper bound",
                    fontsize=6, color=ns.INK2, loc="left")
    ns.panel(ax[1], "b", dx=-0.18)
    fig.tight_layout(w_pad=2.2)
    save(fig, "ab_decision.png")


# ------------------------------------------------- MACs against wall-clock
def latency():
    """What the MAC saving is actually worth on a GPU, and at what price.

    Every other figure in this project counts MACs. This one is the reality
    check: the same allocations timed end to end. It exists because the field
    this work targets measures frames per second, and a MAC count cannot see
    kernel launches, tile gather/scatter, or the arithmetic intensity lost when
    a group runs on 300 tiles instead of 1500.
    """
    BD = {0.1: 0.884, 0.3: 2.511, 0.5: 3.357}   # scripts/paper_metrics.py
    FILES = {0.1: "latency_BEST.json", 0.3: "latency_BEST_b03.json",
             0.5: "latency_BEST_b05.json"}
    got = {b: J(f) for b, f in FILES.items() if J(f)}
    if not got:
        print("  latency: nothing measured")
        return

    fig, ax = plt.subplots(1, 3, figsize=(ns.W2, 2.4))
    buds = sorted(got)
    for c, b in zip(ns.SERIES, buds):
        r = got[b]["rows"]
        qs = [x["qp"] for x in r]
        ax[0].plot(qs, [x["predicted_saving_pct"] for x in r], marker="o", ms=4,
                   color=c, lw=1.3, label=f"{b:g} dB, MACs")
        ax[0].plot(qs, [x["realised_saving_pct"] for x in r], marker="s", ms=3.5,
                   color=c, lw=1.0, ls=(0, (3, 2)),
                   label=f"{b:g} dB, wall-clock")
    ax[0].set_xticks(qs); ax[0].set_xlabel("qp")
    ax[0].set_ylabel("decode saved (%)")
    ax[0].legend(loc="lower left", fontsize=6, ncol=2, frameon=False)
    ax[0].set_title("solid = MACs, dashed = measured", fontsize=6,
                    color=ns.INK2, loc="left")
    ns.panel(ax[0], "a")

    # b: how much of the predicted saving is actually realised
    for c, b in zip(ns.SERIES, buds):
        r = got[b]["rows"]
        ax[1].plot([x["qp"] for x in r],
                   [x["realised_saving_pct"] / x["predicted_saving_pct"]
                    for x in r], marker="o", ms=4, color=c, lw=1.3,
                   label=f"{b:g} dB")
    ax[1].axhline(1.0, color=ns.INK2, lw=0.7, ls=(0, (1, 2)))
    ax[1].text(0, 1.02, "what the MAC model promises", fontsize=6,
               color=ns.INK2)
    ax[1].set_xticks(qs); ax[1].set_xlabel("qp")
    ax[1].set_ylabel("realised / predicted")
    ax[1].set_ylim(0, 1.1)
    ax[1].legend(loc="lower right", fontsize=6, frameon=False, title="budget",
                 title_fontsize=6)
    ax[1].set_title("A looser budget cashes in better", fontsize=6,
                    color=ns.INK2, loc="left")
    ns.panel(ax[1], "b", dx=-0.18)

    # c: the exchange rate, against the one DCVC-UF's own model sizes imply
    xs, ys = [], []
    for b in buds:
        m = np.mean([x["realised_saving_pct"] for x in got[b]["rows"]])
        xs.append(1 / (1 - m / 100)); ys.append(BD[b])
    ax[2].plot(xs, ys, marker="o", ms=5, color=ns.BLUE, lw=1.3,
               label="FLEX-UF (this work)")
    for x, y, b in zip(xs, ys, buds):
        ax[2].annotate(f"{b:g} dB", (x, y), fontsize=6, color=ns.BLUE,
                       textcoords="offset points", xytext=(4, -5))
    # DCVC-UF ships two model sizes; the line between them is what the field
    # currently pays for speed. Table 1 (BD-Rate) and Table 3 (FPS) of
    # arXiv:2606.04410.
    ax[2].plot([1.0, 1.657], [0.0, 10.6], marker="s", ms=4, color=ns.VERM,
               lw=1.3, ls=(0, (4, 2)),
               label="DCVC-UF HT-L → HT-S (their Table 1+3)")
    ax[2].set_xlabel("decode speedup (×)")
    ax[2].set_ylabel("BD-Rate given up (%)")
    ax[2].legend(loc="upper left", fontsize=6, frameon=False)
    ax[2].set_title("Lower is cheaper speed", fontsize=6, color=ns.INK2,
                    loc="left")
    ns.panel(ax[2], "c", dx=-0.20)
    fig.tight_layout(w_pad=2.0)
    save(fig, "latency.png")


# ------------------------------------------------------- A/B across budgets
def ab_budgets():
    """How the cost of not signalling depends on how tight the budget is.

    At 0.1 dB the exit assignment is a real decision and the oracle's access to
    the source frame is worth several points. As the budget loosens the
    allocation degenerates toward "everything at the cheapest exit", the decision
    stops mattering, and the two configurations converge -- at the ceiling the
    only difference left is the router's own arithmetic.
    """
    from flexuf.config import FlexUFConfig
    from flexuf.cost import exit_costs
    cfg = FlexUFConfig(**json.loads((R / "runs/BEST/meta.json").read_text())["config"])
    C = exit_costs(cfg, "head")
    D, CEIL = float(C[-1]), 100 * (1 - float(C[cfg.split_depth]))

    SETS = {
        0.1: ("signalled_BEST_0817_1542.json", "router_BEST_v2.json",
              "router_BEST_v2_lowlam.json"),
        0.3: ("signalled_BEST_b03.json", "router_BEST_b03_lam1.3e-5.json",
              "router_BEST_b03_lam4.1e-6.json"),
        0.5: ("signalled_BEST_b05.json", "router_BEST_b05_lam1.3e-5.json",
              "router_BEST_b05_lam4.1e-6.json"),
    }
    QPS = [0, 16, 32, 48, 63]

    def get(f):
        d = J(f)
        if not d:
            return None
        return {r["qp"]: svr(r, D) for r in d["rows"]
                if r.get("saving_pct") is not None}

    # A budget counts only when BOTH sides are measured. Requiring just the
    # signalled file let a half-finished sweep through: the 0.5 dB signalled
    # result landed while its routers were still queued, and the gap panel got
    # an empty series.
    have = {b: t for b, t in SETS.items()
            if get(t[0]) and any(get(f) for f in t[1:])}
    if not have:
        print("  ab_budgets: nothing measured")
        return
    fig, ax = plt.subplots(1, 2, figsize=(ns.W2, 2.5))
    for c, (bud, (fa, f1, f2)) in zip(ns.SERIES, sorted(have.items())):
        A = get(fa)
        Bs = [b for b in (get(f1), get(f2)) if b]
        best = {q: max(b[q] for b in Bs if q in b) for q in QPS
                if any(q in b for b in Bs)}
        ax[0].plot(QPS, [A.get(q) for q in QPS], marker="o", ms=4, color=c,
                   lw=1.3, label=f"A, {bud:g} dB")
        ax[0].plot(QPS, [best.get(q) for q in QPS], marker="s", ms=3.5,
                   color=c, lw=1.0, ls=(0, (3, 2)), label=f"B, {bud:g} dB")
        qq = [q for q in QPS if q in best and q in A]
        ax[1].plot(qq, [A[q] - best[q] for q in qq],
                   marker="o", ms=4, color=c, lw=1.3, label=f"{bud:g} dB")
    ax[0].axhline(CEIL, color=ns.VERM, lw=0.8, ls=(0, (1, 2)))
    ax[0].text(0, CEIL - 1.0, f"ladder ceiling {CEIL:.1f}%", fontsize=6,
               color=ns.VERM, va="top")
    ax[0].set_xticks(QPS); ax[0].set_xlabel("qp")
    ax[0].set_ylabel("intra decode saved vs release (%)")
    ax[0].legend(loc="lower left", fontsize=6, ncol=2, frameon=False)
    ax[0].set_title("solid = A signalled, dashed = B predicted", fontsize=6,
                    color=ns.INK2, loc="left")
    ns.panel(ax[0], "a")

    # The router's own compute is a hard floor under the gap: even a perfect
    # predictor still has to run.
    ax[1].axhline(0.1629, color=ns.INK2, lw=0.7, ls=(0, (1, 2)))
    ax[1].text(0, 0.22, "0.163% — the router's own compute", fontsize=6,
               color=ns.INK2)
    ax[1].set_xticks(QPS); ax[1].set_xlabel("qp")
    ax[1].set_ylabel("points given up by not signalling")
    ax[1].legend(loc="upper right", fontsize=6, frameon=False, title="budget",
                 title_fontsize=6)
    ax[1].set_title("Seeing the source frame is worth most at a tight budget",
                    fontsize=6, color=ns.INK2, loc="left")
    ns.panel(ax[1], "b", dx=-0.18)
    fig.tight_layout(w_pad=2.2)
    save(fig, "ab_budgets.png")


# ---------------------------------------------------------------- trade-off
def tradeoff():
    """The trade-off read in both directions, and the ceiling that bounds it.

    Every other figure fixes a dB budget and reports the saving. A deployment
    question is usually the other way round -- "I need 30% of the decode back,
    what does it cost me" -- and that reading also exposes the ceiling, which
    the fixed-budget view hides completely.
    """
    pc = J("curve_BEST.json")
    if not pc:
        print("  tradeoff: missing curve_BEST.json")
        return
    from flexuf.config import FlexUFConfig
    from flexuf.cost import exit_costs
    cfg = FlexUFConfig(**json.loads((R / "runs/BEST/meta.json").read_text())["config"])
    C = exit_costs(cfg, "head")
    DEEP = float(C[-1])
    # The most the ladder can ever save: every tile at the shallowest exit that
    # is not dominated, i.e. exit j. Identical at every rate because it is set
    # by the split, not by training.
    CEIL = 100 * (1 - float(C[cfg.split_depth]))

    QPS = sorted({r["qp"] for r in pc["rows"]})

    def frontier(qp):
        pts = sorted((r.get("db_vs_uf_per_frame", r["db_vs_uf"]), svr(r, DEEP))
                     for r in pc["rows"] if r["qp"] == qp)
        return np.array([p[0] for p in pts]), np.array([p[1] for p in pts])

    fig, ax = plt.subplots(1, 2, figsize=(ns.W2, 2.5))
    for c, q in zip(ns.SERIES, QPS):
        d, v = frontier(q)
        ax[0].plot(d, v, color=c, lw=1.2, label=f"qp {q}")
        ax[1].plot(v, d, color=c, lw=1.2)
    ax[0].axvline(0.1, color=ns.INK2, lw=0.7, ls=(0, (3, 2)))
    ax[0].text(0.102, 2, "0.1 dB budget", fontsize=6, color=ns.INK2, rotation=90,
               va="bottom")
    ax[0].axhline(CEIL, color=ns.VERM, lw=0.8, ls=(0, (1, 2)))
    ax[0].text(0.005, CEIL - 0.9, f"ceiling {CEIL:.1f}%: every tile at exit "
               f"{cfg.split_depth}", fontsize=6, color=ns.VERM, va="top")
    ax[0].set_xlabel("quality given up (dB below the release)")
    ax[0].set_ylabel("decode compute saved (%)")
    ax[0].set_xlim(0, 0.30)
    ax[0].legend(loc="lower right", fontsize=6, ncol=2, frameon=False)
    ns.panel(ax[0], "a")

    ax[1].axvline(30, color=ns.INK2, lw=0.7, ls=(0, (3, 2)))
    ax[1].text(29.4, 0.27, "30% target", fontsize=6, color=ns.INK2, rotation=90,
               ha="right", va="top")
    ax[1].axvline(CEIL, color=ns.VERM, lw=0.8, ls=(0, (1, 2)))
    ax[1].text(CEIL - 0.6, 0.27, f"ceiling {CEIL:.1f}%", fontsize=6,
               color=ns.VERM, rotation=90, ha="right", va="top")
    ax[1].set_xlabel("decode compute saved (%)")
    ax[1].set_ylabel("quality that costs (dB)")
    ax[1].set_xlim(0, CEIL + 1.5)
    ax[1].set_ylim(0, 0.30)
    ns.panel(ax[1], "b", dx=-0.18)

    fig.tight_layout(w_pad=2.2)
    save(fig, "tradeoff.png")


# ---------------------------------------------------------------- run tree
def run_tree():
    """Lineage of every training run: what it descends from and how it differs.

    Reads docs/runs.json rather than describing the runs from memory -- the
    launch flags exist only in the process argv, and four of them separate
    BEST from CONTROL, two runs whose second epochs go in opposite directions.
    """
    f = R / "docs" / "runs.json"
    if not f.exists():
        print("  run_tree: run scripts/snapshot_runs.py first")
        return
    D = json.loads(f.read_text())
    runs, diff = D["runs"], D["differing_flags"]

    # The flags worth printing on a box: the ones a reader would need to
    # reproduce the run, not every value that happens to vary.
    SHOW = ["num_exits", "split_depth", "latent_patch", "anchor_weight",
            "adapter_kind", "distill_weight", "joint_router", "seam_repair",
            "new_lr_scale", "epoch_offset", "min_crop", "e", "n", "batch_size",
            "grad_accum"]
    SHOW = [k for k in SHOW if k in diff]
    SHORT = {"num_exits": "K", "split_depth": "j", "latent_patch": "lp",
             "anchor_weight": "anchor", "adapter_kind": "adapter",
             "distill_weight": "distill", "joint_router": "joint router",
             "seam_repair": "seam", "new_lr_scale": "lr×", "min_crop": "crop",
             "epoch_offset": "sched", "e": "epochs", "n": "workers",
             "batch_size": "batch", "grad_accum": "accum"}

    order = sorted(runs, key=lambda t: (int(runs[t]["argv"].get("num_exits", 6)),
                                        t))
    fig, a = plt.subplots(figsize=(ns.W2, 0.46 * len(order) + 1.3))
    a.set_axis_off()
    a.set_xlim(0, 100)
    a.set_ylim(-1.3, len(order) + 0.5)

    def box(x, y, w, h, txt, fc, ec, fs=5.5, tc="white", weight="normal"):
        # pad is in DATA units here, and the y axis spans only ~8 rows: the
        # default 0.3 pad grew every box by more than half a row and welded the
        # column into one solid block.
        a.add_patch(FancyBboxPatch((x, y), w, h, boxstyle="round,pad=0.06",
                                   fc=fc, ec=ec, lw=0.8, zorder=2))
        a.text(x + w / 2, y + h / 2, txt, ha="center", va="center",
               fontsize=fs, color=tc, zorder=3, weight=weight)

    top = len(order)
    box(1, top - 0.85, 15, 0.7, "Microsoft\nDCVC-UF release", "#666666",
        "#666666", fs=5)
    # Two warm starts, because K decides how the twelve blocks are grouped and
    # a K=12 run cannot be quoted against the K=6 remap.
    ws_y = {}
    for i, (k, lab) in enumerate([(6, "warm start K=6"), (12, "warm start K=12")]):
        yy = top - 0.85 - i * 1.15
        box(19, yy, 15, 0.7, f"{lab}\nbit-exact to release", ns.GREEN,
            ns.GREEN, fs=5)
        ws_y[k] = yy + 0.35
        a.annotate("", (18.9, yy + 0.35), (16.1, top - 0.5),
                   arrowprops=dict(arrowstyle="-", lw=0.7, color="#999999"))

    for i, t in enumerate(order):
        r, y = runs[t], len(order) - 1 - i
        K = int(r["argv"].get("num_exits", 6))
        live = r.get("live")
        box(38, y + 0.16, 11, 0.52, t, ns.BLUE if live else "#bbbbbb",
            ns.BLUE if live else "#bbbbbb", fs=5.5, weight="bold")
        # An elbow, not a straight line: six straight lines to two boxes a
        # row apart converge into a fan whose apex reads as one origin, and
        # the K=12 box happened to sit under it -- the figure said FINE12 and
        # the K=6 runs share a warm start, which is exactly what the two
        # separate remaps exist to prevent.
        wy = ws_y.get(K, top - 0.5)
        a.plot([34.4, 35.6, 35.6, 37.6], [wy, wy, y + 0.42, y + 0.42],
               lw=0.7, color=ns.GREEN if K == 12 else "#bbbbbb",
               solid_joinstyle="miter", zorder=1)
        flags = "  ".join(
            f"{SHORT[k]}={r['argv'][k]}" for k in SHOW
            if k in r["argv"] and r["argv"][k] not in (None, False))
        a.text(50.5, y + 0.56, flags, fontsize=6, color=ns.INK2, va="center")
        ep, st = r.get("epoch", 0), r.get("step", 0)
        done = ep + st / STEPS_PER_EPOCH
        a.text(50.5, y + 0.22, f"{done:.2f} epochs done"
               + ("" if live else "  ·  not running"),
               fontsize=6, color=ns.VERM if live else "#999999", va="center")
        # progress against the 4-epoch selection point, not the -e 16 the
        # launcher was given: nothing here will be trained for sixteen epochs.
        a.add_patch(plt.Rectangle((78, y + 0.16), 20, 0.14, fc="#e8e8e8",
                                  ec="none"))
        a.add_patch(plt.Rectangle((78, y + 0.16), 20 * min(done / 4, 1), 0.14,
                                  fc=ns.VERM if live else "#bbbbbb", ec="none"))
    a.text(78, len(order) + 0.05, "progress toward the 4-epoch selection point",
           fontsize=6, color=ns.INK2)
    a.text(1, -1.0, "blue = training now  ·  every run warm-starts from the "
           "release and freezes the encoder, so all of them emit the identical "
           "bitstream\nno further architectures are planned: the four live "
           "recipes run to 4 epochs and are selected on BD-saving over the "
           "common rate interval", fontsize=6, color=ns.INK2, va="top")
    save(fig, "run_tree.png")


if __name__ == "__main__":
    topology()
    run_tree()
    rd_curve()
    ab_compare()
    ab_budgets()
    latency()
    tradeoff()
    loss_curves("BEST")
