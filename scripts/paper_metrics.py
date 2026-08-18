"""Evaluate in the units DCVC-UF's own paper reports, so the numbers compose.

The project measures "compute saved at a dB budget". The DCVC-UF paper (Li et
al., arXiv:2606.04410) reports BD-Rate against a VTM anchor, MACs/frame, params
and FPS, on UVG / MCL-JCV / HEVC B-E in YUV420. Two of those we can produce
exactly, one partially, one not at all -- and saying which is which is the point
of this script.

BD-Rate is the natural translation
----------------------------------
Early exit does not change the bitstream, so its cost shows up entirely as lost
PSNR at a fixed rate. BD-Rate converts that into the equivalent rate increase --
"how many more bits would you have to spend to get this quality back" -- which
is exactly how a codec paper prices a quality loss. It also makes our number
directly comparable to the paper's own table, where DCVC-UF (LD) is -9.5% and
(HT-L) is -42.2% against VTM.

What this script CANNOT produce
-------------------------------
* **HEVC B, C and D** were absent when this script was written and are now on
  disk, so the set is the full 53. The per-dataset breakdown below reads whatever
  the result file recorded, which is how a partial set could never be mistaken
  for a complete one. UVG (7),
  MCL-JCV (30) and HEVC E (3) are, and are reported per dataset.
* **FPS.** The paper measures wall-clock decoding speed. This project measures
  MACs. A tiled early-exit decoder with per-tile depths has real scheduling
  overhead that a MAC count does not capture, and quoting MAC savings as a speed
  claim would be exactly the kind of substitution this project keeps catching.
* **Whole-video BD-Rate.** Ours is the INTRA decoder. At the paper's
  intra-period of -1 there is one intra frame per sequence, so the intra path is
  ~2.8% of a sequence's decode MACs. The translation is printed rather than
  hidden: a 30% intra saving is 30% of an image decode, 8.4% of a video at
  intra-period 8, and 0.8% at intra-period -1.

    python scripts/paper_metrics.py
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "scripts"))

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import naturestyle as ns  # noqa: E402

ns.apply()
OUT = ROOT / "docs" / "figures"
OUT.mkdir(parents=True, exist_ok=True)

INTRA_GMAC = 453.5          # DMCI at 1080p, scripts/mac_audit.py
INTER_GMAC = 167.0          # implied by Table 3's 170 GMAC average for LD
QPS = [0, 16, 32, 48, 63]


def J(*names):
    """First of `names` that exists.

    Called with several candidates so a corrected re-measurement supersedes an
    older file without every call site being edited. The measurement that
    forced this: results taken before flexuf/eval.py landed were computed on a
    full-frame per-exit table and the tiling penalty cancelled out of them, so
    the old files must never be picked when a newer one is present.
    """
    for name in names:
        p = ROOT / "results" / name
        if p.exists():
            return json.loads(p.read_text())
    return None


def deepest_cost(tag="BEST"):
    from flexuf.config import FlexUFConfig
    from flexuf.cost import exit_costs
    cfg = FlexUFConfig(**json.loads((ROOT / "runs" / tag / "meta.json")
                                    .read_text())["config"])
    return float(exit_costs(cfg, "head")[-1])


def svr(row, D):
    if row.get("saving_pct_vs_release") is not None:
        return row["saving_pct_vs_release"]
    s = row.get("saving_pct")
    return None if s is None else 100.0 - (100.0 - s) * D


def bd_rate(r1, p1, r2, p2) -> float:
    """Bjontegaard rate difference (%), curve 2 against anchor curve 1.

    Cubic in (log rate, PSNR), integrated over the overlapping PSNR range --
    the standard formulation (Bjontegaard, VCEG-M33), and the same one the
    DCVC-UF paper's BD-Rate column uses.
    """
    lr1, lr2 = np.log(r1), np.log(r2)
    p_lo, p_hi = max(min(p1), min(p2)), min(max(p1), max(p2))
    if p_hi <= p_lo:
        return float("nan")
    c1 = np.polyfit(p1, lr1, 3)
    c2 = np.polyfit(p2, lr2, 3)
    i1 = np.polyval(np.polyint(c1), [p_lo, p_hi])
    i2 = np.polyval(np.polyint(c2), [p_lo, p_hi])
    return 100.0 * (np.exp((i2[1] - i2[0] - (i1[1] - i1[0])) / (p_hi - p_lo)) - 1)


def main():
    anc = J("anchor_RECIPE512_ctc53.json", "anchor_BEST_5qp.json")
    why = J("why_qp.json")
    if not (anc and why):
        raise SystemExit("need results/anchor_BEST_5qp.json and why_qp.json")
    rel_psnr = {r["qp"]: r["stock_psnr"] for r in anc["rows"]}
    bpp = {r["qp"]: r["bpp"] for r in why["rows"]}
    D = deepest_cost("BEST")

    # Every configuration measured, at every budget. Each entry is a list of
    # candidate files, newest first: a --budgets run stores all three budgets in
    # one file, older runs stored one budget each.
    CONFIGS = [
        ("A signalled", ["signalled_RECIPE512_ctc53.json",
                         "signalled_RECIPE512_b135.json",
                         "signalled_BEST_0817_1542.json"], 0.1),
        ("A signalled", ["signalled_RECIPE512_ctc53.json",
                         "signalled_RECIPE512_b135.json",
                         "signalled_BEST_b03.json"], 0.3),
        ("A signalled", ["signalled_RECIPE512_ctc53.json",
                         "signalled_RECIPE512_b135.json",
                         "signalled_BEST_b05.json"], 0.5),
        ("B router", ["router_RECIPE512_b01.json",
                      "router_RECIPE512_lam1.3e-5.json",
                      "router_BEST_v2.json"], 0.1),
        ("B router", ["router_RECIPE512_b03.json"], 0.3),
        ("B router", ["router_RECIPE512_b05.json"], 0.5),
    ]

    print("=" * 78)
    print("BD-Rate against the RELEASED DCVC-UF intra decoder, YUV420 PSNR")
    print("(positive = we need more bits for the same quality; the release is 0)")
    print("=" * 78)
    print(f"\n  {'configuration':<22}{'budget':>8}{'BD-Rate':>10}"
          f"{'mean saving':>13}{'bits added':>12}")
    rows_for_fig = []
    for label, fnames, budget in CONFIGS:
        d = J(*fnames)
        if not d:
            print(f"  {label:<22}{budget:>8.2f}   (not measured yet: {fnames[0]})")
            continue
        # A multi-budget file holds every budget; filter to the one asked for.
        rs = [r for r in d["rows"] if r.get("saving_pct") is not None
              and (r.get("budget_db") is None
                   or abs(r["budget_db"] - budget) < 1e-9)]
        if not rs:
            print(f"  {label:<22}{budget:>8.2f}   (budget absent from "
                  f"{fnames[0]})")
            continue
        got = {r["qp"]: r for r in rs}
        qs = [q for q in QPS if q in got and q in rel_psnr and q in bpp]
        if len(qs) < 4:
            print(f"  {label:<22}{budget:>8.2f}   (only {len(qs)} rates)")
            continue
        # Our PSNR at each rate is the release's minus the measured dB loss.
        ours = [rel_psnr[q] - abs(got[q]["db_vs_uf"]) for q in qs]
        rate = [bpp[q] for q in qs]
        # A also spends bits on the exit map; B spends none.
        extra = [got[q].get("bpp_added", 0.0) or 0.0 for q in qs]
        rate_ours = [r + e for r, e in zip(rate, extra)]
        bd = bd_rate(np.array(rate), np.array([rel_psnr[q] for q in qs]),
                     np.array(rate_ours), np.array(ours))
        sv = float(np.mean([svr(got[q], D) for q in qs]))
        mb = float(np.mean([got[q].get("map_bits", 0) or 0 for q in qs]))
        print(f"  {label:<22}{budget:>8.2f}{bd:>9.3f}%{sv:>12.2f}%{mb:>11.0f}")
        rows_for_fig.append((label, budget, bd, sv, qs, rate_ours, ours))

    print(f"\n  For scale, the paper's own Table 1 (BD-Rate vs VTM-17.0 LD):")
    print(f"    DCVC-UF (LD) -9.5%   (HT-S) -31.6%   (HT-L) -42.2%")
    print(f"  Ours is a decode-compute method, so its BD-Rate is the COST of")
    print(f"  the compute it saves, not a compression gain.")

    # ---- per dataset, the paper's grouping ------------------------------
    print("\n" + "=" * 78)
    print("Per dataset, as recorded in the result file.")
    print("=" * 78)
    pc = J("curve_RECIPE512_ctc53.json", "curve_BEST.json")
    if pc and pc.get("op_points"):
        # Read the class from DCVC's own test config rather than guessing from
        # the filename. The guessing version silently mislabelled every HEVC B
        # sequence as UVG the moment classes B, C and D landed on disk -- both
        # are 1920x1080 -- and reported "UVG, n=20".
        def _families():
            import sys as _s
            _s.path.insert(0, str(Path.home() / "DCVC"))
            _s.path.insert(0, str(ROOT))
            import ctc_intra as _C
            seqs, _ = _C.discover([])
            return {x["name"]: x["cls"] for x in seqs}
        FAM = _families()

        def fam(n):
            return FAM.get(n, "unknown")
        # the operating point closest to the 0.1 dB budget, per qp
        # The op_points are a coarse grid and their target_db is in the POOLED
        # convention, so the one nearest 0.1 actually lands at ~0.14 dB
        # per-frame. Labelling it "0.1 dB" would put a number 4 points high
        # next to the headline table -- the exact confusion the two dB
        # conventions have caused before. The realised dB is printed instead.
        by, realised = {}, {}
        for q in QPS:
            ops = [o for o in pc["op_points"] if o["qp"] == q
                   and o.get("per_sequence")]
            if not ops:
                continue
            o = min(ops, key=lambda o: abs(o.get("target_db", 9) - 0.1))
            realised[q] = o.get("db_vs_uf_per_frame")
            for s in o["per_sequence"]:
                by.setdefault(fam(s["seq"]), {}).setdefault(q, []).append(
                    100 - (100 - s["saving_pct"]) * D)
        print(f"\n  saving by dataset, at the nearest measured operating point\n")
        print(f"  {'dataset':<10}{'n':>4}" + "".join(f"{q:>9}" for q in QPS))
        for f in ("UVG", "MCL-JCV", "HEVC_B", "HEVC_C", "HEVC_D", "HEVC_E"):
            if f not in by:
                continue
            n = len(next(iter(by[f].values())))
            print(f"  {f:<10}{n:>4}" + "".join(
                f"{np.mean(by[f][q]):>9.2f}" if q in by[f] else f"{'-':>9}"
                for q in QPS))
        print(f"  {'(dB there)':<10}{'':>4}" + "".join(
            f"{realised[q]:>9.3f}" if q in realised else f"{'-':>9}"
            for q in QPS))
        print("\n  Not the 0.1 dB budget: these are the nearest points on the")
        print("  op_point grid, and they sit at ~0.14 dB. Cross-dataset SPREAD")
        print("  is what this table is for, not the absolute level.")

    # ---- complexity, the paper's Table 3 shape ---------------------------
    print("\n" + "=" * 78)
    print("Complexity, in the shape of the paper's Table 3")
    print("=" * 78)
    from flexuf.config import FlexUFConfig
    from flexuf.model import FlexUFIntra
    import torch
    cfg = FlexUFConfig(**json.loads((ROOT / "runs/BEST/meta.json")
                                    .read_text())["config"])
    net = FlexUFIntra(cfg)
    tot = sum(p.numel() for p in net.parameters())
    print(f"\n  {'model':<34}{'intra MACs/frame':>18}{'params':>10}")
    print(f"  {'released DCVC-UF intra decoder':<34}{INTRA_GMAC:>17.1f}G"
          f"{tot/1e6:>9.1f}M")
    a = J("signalled_RECIPE512_ctc53.json", "signalled_BEST_0817_1542.json")
    b = J("router_RECIPE512_b01.json", "router_RECIPE512_lam1.3e-5.json",
          "router_BEST_v2.json")
    for lab, d in (("FLEX-UF A, signalled, 0.1 dB", a),
                   ("FLEX-UF B, router, 0.1 dB", b)):
        if not d:
            continue
        s = np.mean([svr(r, D) for r in d["rows"]
                     if r.get("saving_pct") is not None])
        print(f"  {lab:<34}{INTRA_GMAC*(1-s/100):>17.1f}G"
              f"{(tot + (144024 if 'router' in lab else 0))/1e6:>9.1f}M")
    print(f"\n  The paper's whole-video averages, for context:")
    print(f"    DCVC-UF (LD) 170G   (HT-S) 211G   (HT-L) 343G per frame")

    print("\n" + "=" * 78)
    print("What an intra saving is worth over a whole sequence")
    print("=" * 78)
    print(f"\n  {'intra period':>13}{'intra MAC share':>18}"
          f"{'30% intra ->':>15}")
    for N in (1, 8, 16, 32, 64, 96):
        share = INTRA_GMAC / (INTRA_GMAC + (N - 1) * INTER_GMAC)
        print(f"  {N if N != 96 else '-1 (~96)':>13}{100*share:>17.1f}%"
              f"{30*share:>14.1f}%")
    print("\n  So the headline is an IMAGE/intra-decoder result. Quoting it as a")
    print("  video decode saving without the intra period attached would be wrong.")
    return rows_for_fig


def figures(rows, rel_psnr, bpp, D):
    """The paper's Figure 5 layout, plus the trade-off it is really about."""
    qs = sorted(rel_psnr)
    R = [bpp[q] for q in qs]
    P = [rel_psnr[q] for q in qs]

    # --- Figure 5 style: full range, then the two quality-range zooms ----
    fig, ax = plt.subplots(1, 3, figsize=(ns.W2, 2.3))
    lo = [q for q in qs if rel_psnr[q] <= 39]
    hi = [q for q in qs if rel_psnr[q] >= 39]

    def short(l):
        return l.replace("B router", "B").replace("A signalled", "A")

    BCOL = {0.1: ns.VERM, 0.3: ns.ORANGE, 0.5: ns.GREEN}
    for a_, sel, title in ((ax[0], qs, "All rates"),
                           (ax[1], lo, "Lower quality range"),
                           (ax[2], hi, "Higher quality range")):
        a_.plot([bpp[q] for q in sel], [rel_psnr[q] for q in sel],
                marker="o", ms=4, color=ns.INK, lw=1.3,
                label="released DCVC-UF intra")
        for label, budget, bd, sv, qq, rr, pp in rows:
            idx = [i for i, q in enumerate(qq) if q in sel]
            if not idx:
                continue
            isA = label.startswith("A")
            a_.plot([rr[i] for i in idx], [pp[i] for i in idx],
                    marker="o" if isA else "s", ms=3, lw=1.0,
                    ls="-" if isA else (0, (3, 1.5)), color=BCOL[budget],
                    label=f"{short(label)} @ {budget:g} dB  ({sv:.0f}% saved)")
        a_.set_xlabel("bpp")
        a_.set_title(title, fontsize=6, color=ns.INK2, loc="left")
    ax[0].set_ylabel("PSNR (dB), YUV420")
    ax[0].legend(loc="lower right", fontsize=4.4, frameon=False)
    for i, l in enumerate("abc"):
        ns.panel(ax[i], l, dx=-0.20)
    fig.tight_layout(w_pad=1.8)
    fig.savefig(OUT / "paper_rd.png", dpi=300, bbox_inches="tight",
                facecolor="white")
    plt.close(fig)
    print(f"  docs/figures/paper_rd.png")

    # --- the trade-off the DCVC-UF paper is organised around -------------

    fig, ax = plt.subplots(1, 2, figsize=(ns.W2, 2.4))
    for label, budget, bd, sv, qq, rr, pp in rows:
        ax[0].scatter(bd, sv, s=38, color=BCOL[budget], zorder=3,
                      marker="o" if label.startswith("A") else "s",
                      label=f"{short(label)} @ {budget:g} dB")
    ax[0].scatter(0, 0, s=38, color=ns.INK, marker="*", zorder=3,
                  label="released decoder")
    ax[0].set_xlabel("BD-Rate cost (%), lower is better")
    ax[0].set_ylabel("intra decode MACs saved (%)")
    ax[0].legend(loc="lower right", fontsize=4.6, frameon=False)
    ax[0].set_title("What the compute costs in rate", fontsize=6,
                    color=ns.INK2, loc="left")
    ns.panel(ax[0], "a")

    # MACs/frame, the paper's Table 3 column, drawn
    labels = ["released"] + [f"{short(l)}, {b:g} dB" for l, b, *_ in rows]
    macs = [INTRA_GMAC] + [INTRA_GMAC * (1 - sv / 100)
                           for _, _, _, sv, *_ in rows]
    cols = [ns.INK] + [BCOL[b] for _, b, *_ in rows]
    ax[1].barh(range(len(macs)), macs, color=cols, height=0.6)
    for i, m in enumerate(macs):
        ax[1].text(m + 6, i, f"{m:.0f}G", va="center", fontsize=5,
                   color=ns.INK2)
    ax[1].set_yticks(range(len(macs)))
    ax[1].set_yticklabels(labels, fontsize=4.6)
    ax[1].invert_yaxis()
    ax[1].set_xlabel("intra decode GMAC per 1080p frame")
    ax[1].set_xlim(0, INTRA_GMAC * 1.18)
    ax[1].set_title("Complexity, the paper's Table 3 column", fontsize=6,
                    color=ns.INK2, loc="left")
    ns.panel(ax[1], "b", dx=-0.30)
    fig.tight_layout(w_pad=2.0)
    fig.savefig(OUT / "paper_rdc.png", dpi=300, bbox_inches="tight",
                facecolor="white")
    plt.close(fig)
    print(f"  docs/figures/paper_rdc.png")


if __name__ == "__main__":
    rows = main()
    anc, why = (J("anchor_RECIPE512_ctc53.json", "anchor_BEST_5qp.json"),
                J("why_qp.json"))
    print()
    figures(rows, {r["qp"]: r["stock_psnr"] for r in anc["rows"]},
            {r["qp"]: r["bpp"] for r in why["rows"]}, deepest_cost("BEST"))
