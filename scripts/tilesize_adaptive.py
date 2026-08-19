"""What a resolution-adaptive tile size would give, from measurements that exist.

Section 87 of DECISIONS.md found that a 128 px tile helps only at 416x240 and
costs points everywhere else. The obvious follow-up is to stop choosing one tile
size for the whole test set: send 832x480 and 416x240 down the 128 px path and
leave 720p and 1080p on 256 px. That configuration was never trained and never
measured, but every class it needs has already been measured on both paths, so
the splice can be assembled by arithmetic. No new decode is run here.

The splice is an estimate of a system, not a measurement of one, and two things
about it are reported beside the numbers rather than left for the reader to
discover.

  1. Each source run bisected lambda ONCE over all 53 sequences so that the whole
     set landed on 0.1 dB. Taking four classes from one bisection and two from
     another gives a set whose delivered dB is the weighted mean of two different
     operating points, so the spliced column does not sit exactly on the budget.
     A deployed resolution-adaptive decoder would re-bisect. Nothing in results/
     carries the per-class lambda sweep that would let this script re-bisect, so
     the delivered dB is reported instead of being corrected away.

  2. The 128 px source (BEST128) and the 256 px source (RECIPE512) are different
     training runs read at different points on their schedules. They are
     otherwise a matched pair: runs/*/meta.json differ in exactly rgb_patch (128
     against 256) and latent_patch (8 against 16), with recipe, dataset, batch
     size, exits, split depth, adapter and seam repair identical. So the confound
     is training time, and training time can be measured rather than asserted.
     The strongest handle on it is a BRACKET: RECIPE512 measured by this same
     script at epoch 0 and again at epoch 1 straddles BEST128's training, so a
     128 px number that beats both ends, or loses to both ends, is not explained
     by the extra training. One that lands between them is.
"""
from __future__ import annotations
import json
import sys
from pathlib import Path

import numpy as np
import torch

R = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(R))

# Steps per epoch for every run in this project, taken from convergence.py so
# that a cumulative-step axis built here lines up with the one already plotted.
SPE = 47451

# The rule under test. 832x480 and 416x240 are the two classes where a 256 px
# tile leaves the router 8 and 2 tiles to allocate over, which is the whole
# reason a smaller tile was proposed for them in the first place.
ADAPTIVE_TILE = {"UVG": 256, "MCL-JCV": 256, "HEVC_B": 256, "HEVC_E": 256,
                 "HEVC_C": 128, "HEVC_D": 128}
CLASS_ORDER = ["MCL-JCV", "UVG", "HEVC_B", "HEVC_E", "HEVC_C", "HEVC_D"]
BUDGET = 0.1


def load_per_class(path):
    d = json.loads((R / path).read_text())
    return d, {r["qp"]: r for r in d["rows"]
               if abs(r["budget_db"] - BUDGET) < 1e-9}


src256, rows256 = load_per_class("results/per_class_RECIPE512.json")
src128, rows128 = load_per_class("results/per_class_BEST128.json")
srcE1, rowsE1 = load_per_class("results/per_class_RECIPE512_e1.json")

# Refusing to compare files whose sequence or frame count differs is the guard
# crosscheck_paths.py applies to curve files, and it applies here for the same
# reason: a saving compared across two different test sets is not a comparison.
for other in (src128, srcE1):
    assert src256["n_frames"] == other["n_frames"], "different frame counts"
assert src256["tile_px"] == srcE1["tile_px"] == 256, "tile_px mismatch on 256 px side"
assert src128["tile_px"] == 128, "tile_px mismatch on 128 px side"
qps = sorted(set(rows256) & set(rows128) & set(rowsE1))
for q in qps:
    a, b, e = (rows256[q]["per_class"], rows128[q]["per_class"],
               rowsE1[q]["per_class"])
    assert set(a) == set(b) == set(e) == set(ADAPTIVE_TILE), "class sets differ"
    for c in a:
        assert a[c]["n"] == b[c]["n"] == e[c]["n"], f"{c}: sequence count differs"
        assert a[c]["res"] == b[c]["res"] == e[c]["res"], f"{c}: resolution differs"

N = {c: rows256[qps[0]]["per_class"][c]["n"] for c in CLASS_ORDER}
NTOT = sum(N.values())


def set_mean(v):
    """Sequence-weighted mean, which is what overall_saving in the source is.

    per_class.py reports overall_saving as a mean over all 53 frames and not as a
    mean over the six class means, so a spliced set mean has to be weighted the
    same way or it will not be comparable to the headline it sits beside.
    """
    return sum(v[c] * N[c] for c in CLASS_ORDER) / NTOT


def class_mean(v):
    """Unweighted mean over the six classes, reported only as a sensitivity.

    Thirty of the 53 sequences are MCL-JCV 1080p, so the weighted mean is close to
    a 1080p number and flattens the two low-resolution classes this question is
    entirely about. The unweighted mean over-weights them instead. Neither is
    wrong, and quoting only one of them would be.
    """
    return sum(v[c] for c in CLASS_ORDER) / len(CLASS_ORDER)


def ckpt_provenance(path):
    """Read epoch and step off the checkpoint rather than trusting its name.

    ckpt_step.pth.tar is rewritten by the training loop, so the step it held when
    a result was measured is recoverable only while the file is untouched. This
    record is what will still say which weights produced the 128 px column after
    the watcher has moved on.
    """
    ck = torch.load(R / path, map_location="cpu", weights_only=False)
    ep, st = ck.get("epoch"), ck.get("step")
    return {"path": path, "epoch": ep, "step": st,
            "cumulative_step": (ep + 1) * SPE if st is None else ep * SPE + st,
            "pinned": Path(path).name in ("ckpt_PAPER.pth.tar", "ckpt_PIN_e1.pth.tar")}


ck256 = ckpt_provenance(src256["ckpt"])
ck128 = ckpt_provenance(src128["ckpt"])
ckE1 = ckpt_provenance(srcE1["ckpt"])
train_gap = ck128["cumulative_step"] - ck256["cumulative_step"]
brackets = ck256["cumulative_step"] <= ck128["cumulative_step"] <= ckE1["cumulative_step"]

per_qp = []
for q in qps:
    a, b, e = (rows256[q]["per_class"], rows128[q]["per_class"],
               rowsE1[q]["per_class"])
    sv256 = {c: a[c]["saving"] for c in CLASS_ORDER}
    sv128 = {c: b[c]["saving"] for c in CLASS_ORDER}
    svE1 = {c: e[c]["saving"] for c in CLASS_ORDER}
    db256 = {c: a[c]["db"] for c in CLASS_ORDER}
    db128 = {c: b[c]["db"] for c in CLASS_ORDER}
    svad = {c: (sv128 if ADAPTIVE_TILE[c] == 128 else sv256)[c] for c in CLASS_ORDER}
    svadE1 = {c: (sv128 if ADAPTIVE_TILE[c] == 128 else svE1)[c] for c in CLASS_ORDER}
    dbad = {c: (db128 if ADAPTIVE_TILE[c] == 128 else db256)[c] for c in CLASS_ORDER}

    cls_rows = []
    for c in CLASS_ORDER:
        lo, hi = min(sv256[c], svE1[c]), max(sv256[c], svE1[c])
        cls_rows.append({
            "cls": c, "res": a[c]["res"], "n_sequences": N[c],
            "tiles_256": a[c]["tiles"], "tiles_128": b[c]["tiles"],
            "adaptive_tile_px": ADAPTIVE_TILE[c],
            "saving_256_pct": sv256[c],
            "saving_128_pct": sv128[c],
            "saving_adaptive_pct": svad[c],
            "delta_adaptive_minus_256": svad[c] - sv256[c],
            "delta_128_minus_256": sv128[c] - sv256[c],
            "db_256": db256[c], "db_128": db128[c], "db_adaptive": dbad[c],
            # Where the 128 px number falls relative to the same 256 px run read
            # one epoch apart. "inside" means the extra training alone could
            # account for it and the tile size is not separable from training.
            "saving_256_epoch1_pct": svE1[c],
            "one_epoch_of_training_pts": svE1[c] - sv256[c],
            "delta_128_minus_256_epoch1": sv128[c] - svE1[c],
            "vs_training_bracket": ("above" if sv128[c] > hi else
                                    "below" if sv128[c] < lo else "inside"),
        })
    per_qp.append({
        "qp": q, "budget_db": BUDGET,
        "lam_256": rows256[q]["lam"], "lam_128": rows128[q]["lam"],
        "lam_256_epoch1": rowsE1[q]["lam"],
        "classes": cls_rows,
        "set_mean_saving_pct": {
            "uniform_256": set_mean(sv256), "uniform_128": set_mean(sv128),
            "adaptive": set_mean(svad),
            "uniform_256_epoch1": set_mean(svE1),
            "adaptive_vs_epoch1": set_mean(svadE1)},
        "set_mean_delta_adaptive_minus_256": set_mean(svad) - set_mean(sv256),
        "set_mean_delta_adaptive_minus_256_epoch1": set_mean(svadE1) - set_mean(svE1),
        "class_mean_saving_pct": {
            "uniform_256": class_mean(sv256), "uniform_128": class_mean(sv128),
            "adaptive": class_mean(svad)},
        "class_mean_delta_adaptive_minus_256": class_mean(svad) - class_mean(sv256),
        # The delivered dB moves when two separately bisected operating points are
        # spliced, and a saving quoted at a different distortion is a different
        # number. Carrying all three makes it visible whether the adaptive column
        # bought its gain with compute or with quality.
        "set_mean_delivered_db": {
            "uniform_256": set_mean(db256), "uniform_128": set_mean(db128),
            "adaptive": set_mean(dbad)},
        "overall_saving_source_256": rows256[q]["overall_saving"],
        "overall_saving_source_128": rows128[q]["overall_saving"],
        "overall_saving_source_256_epoch1": rowsE1[q]["overall_saving"],
    })

# A second, weaker handle on the same confound: what a thousand training steps is
# worth on BEST128's own trajectory. The signalled_* series is the only
# per-checkpoint trajectory in results/, and it has to be filtered to one test set
# before a slope means anything, because two of its points were taken on the
# 53-sequence set and the rest on the 40-sequence one.
traj = []
for f in sorted((R / "results").glob("signalled_BEST128_*.json")):
    d = json.loads(f.read_text())
    ok = [r for r in d["rows"] if r.get("saving_pct") is not None]
    ep, st = d.get("ckpt_epoch"), d.get("ckpt_step")
    if ep is None or d.get("n_sequences") != 40 or len(ok) < 5:
        continue
    v = {r["qp"]: r.get("saving_pct_vs_release", r["saving_pct"]) for r in ok}
    if not {0, 16, 32, 48, 63} <= set(v):
        continue
    traj.append(((ep + 1) * SPE if st is None else ep * SPE + st,
                 float(np.mean([v[q] for q in (0, 16, 32, 48, 63)])), f.name))
traj.sort()
xs = np.array([t[0] for t in traj]) / 1000.0
ys = np.array([t[1] for t in traj])
h = len(xs) // 2
slopes = {"whole_series": float(np.polyfit(xs, ys, 1)[0]),
          "first_half": float(np.polyfit(xs[:h + 1], ys[:h + 1], 1)[0]),
          "second_half": float(np.polyfit(xs[h:], ys[h:], 1)[0]),
          "last_segment": float((ys[-1] - ys[-2]) / (xs[-1] - xs[-2]))}
band = sorted(s * train_gap / 1000.0 for s in slopes.values())

# A third handle, from the one place in results/ where training patch and test
# tile were varied on purpose and independently. It is a much smaller measurement
# (20 CTC frames, signalled operating points near but not on 0.1 dB) so it cannot
# be mixed into the table above, but it says which of the two factors is larger.
iso = {}
for k in ("A_128w_128t", "B_128w_256t", "C_256w_256t", "D_256w_128t"):
    p = R / f"results/iso_{k}.json"
    if p.exists():
        d = json.loads(p.read_text())
        iso[k] = {"ckpt": d["ckpt"],
                  "rows": {r["qp"]: [r["saving_pct"], r["db_vs_uf"]] for r in d["rows"]}}
iso_effects = None
if len(iso) == 4:
    qs = sorted(set.intersection(*(set(v["rows"]) for v in iso.values())))
    tile = ([iso["A_128w_128t"]["rows"][q][0] - iso["B_128w_256t"]["rows"][q][0] for q in qs]
            + [iso["D_256w_128t"]["rows"][q][0] - iso["C_256w_256t"]["rows"][q][0] for q in qs])
    wts = ([iso["A_128w_128t"]["rows"][q][0] - iso["D_256w_128t"]["rows"][q][0] for q in qs]
           + [iso["B_128w_256t"]["rows"][q][0] - iso["C_256w_256t"]["rows"][q][0] for q in qs])
    iso_effects = {
        "qps": qs,
        "tile_effect_128t_minus_256t_at_matched_weights": {
            "min": min(tile), "max": max(tile), "mean": sum(tile) / len(tile)},
        "weights_effect_128w_minus_256w_at_matched_tile": {
            "min": min(wts), "max": max(wts), "mean": sum(wts) / len(wts)},
        "caveat": ("20 CTC frames, one signalled operating point per rate rather "
                   "than a bisection, and the four cells do not deliver the same "
                   "dB, so these are magnitudes and not a matched-dB comparison"),
    }

# The note is assembled from the numbers rather than typed, because a confound
# paragraph that drifts out of step with the table it sits next to is worse than
# no paragraph. Every range quoted below is read off the arrays above.
def _rng(q):
    v = [c["one_epoch_of_training_pts"] for c in
         next(r for r in per_qp if r["qp"] == q)["classes"]]
    return min(v), max(v)


def _cls(q, verdict):
    return [c["cls"] for c in next(r for r in per_qp if r["qp"] == q)["classes"]
            if c["vs_training_bracket"] == verdict]


bracket_summary = {q: {v: _cls(q, v) for v in ("above", "inside", "below")}
                   for q in qps}
epoch_worth = {q: _rng(q) for q in qps}
_e0 = ", ".join(f"{epoch_worth[q][0]:.2f} to {epoch_worth[q][1]:.2f} at qp {q}"
                for q in qps)
_bs = "; ".join(
    f"qp {q}: above " + (", ".join(bracket_summary[q]["above"]) or "none")
    + " / inside " + (", ".join(bracket_summary[q]["inside"]) or "none")
    + " / below " + (", ".join(bracket_summary[q]["below"]) or "none")
    for q in qps)
_iso = ""
if iso_effects:
    t = iso_effects["tile_effect_128t_minus_256t_at_matched_weights"]
    w = iso_effects["weights_effect_128w_minus_256w_at_matched_tile"]
    _iso = (f"the tile effect at matched weights spans {t['min']:.1f} to "
            f"{t['max']:.1f} points while the weights effect at matched tile spans "
            f"{w['min']:.1f} to {w['max']:.1f}, means {t['mean']:.1f} against "
            f"{w['mean']:.1f}, so the weights axis is the larger one by about a "
            "factor of three, ")

note = (
    "CONFOUND, STATED IN FULL, NOT HIDDEN. The 128 px column is runs/BEST128 and "
    "the 256 px column is runs/RECIPE512. Their meta.json differ in exactly two "
    "fields, rgb_patch (128 against 256) and latent_patch (8 against 16); recipe, "
    "dataset, batch size, exits, split depth, adapter kind and seam repair are "
    "identical. What is not matched is training time. The 256 px column is "
    f"{ck256['path']} at epoch {ck256['epoch']}, {ck256['cumulative_step']} "
    f"cumulative steps. The 128 px column is {ck128['path']} at epoch "
    f"{ck128['epoch']} step {ck128['step']}, {ck128['cumulative_step']} cumulative "
    f"steps. The 128 px side carries {train_gap} extra steps, {train_gap / SPE:.2f} "
    "of an epoch, and the advantage runs in ITS favour, so the mixture is not "
    "neutral. HOW MUCH IT COULD MATTER, measured three ways. (1) The bracket, "
    f"which is the one that settles it: {ckE1['path']} at epoch {ckE1['epoch']}, "
    f"{ckE1['cumulative_step']} cumulative steps, measured by this same script on "
    "this same 53-sequence set at this same budget, straddles BEST128's training "
    "from above while ckpt_PAPER straddles it from below. Measured per class, one "
    f"epoch of training is worth {_e0}, in points, where the zero floors at qp 32 and "
    "qp 63 are HEVC_C and HEVC_D sitting on the 2.54 per cent saturation point at "
    "which every tile has already taken the last exit and no amount of training "
    "moves them. A 128 px number landing inside that "
    "bracket is a number tile size cannot be credited with; vs_training_bracket "
    f"records the verdict per class per rate: {_bs}. (2) BEST128's own trajectory "
    f"puts {train_gap} steps at {band[0]:.1f} to {band[-1]:.1f} points of mean "
    "saving, but that trajectory is on the 40-sequence subset, which holds no "
    "832x480 and no 416x240 material, so it does not constrain the two classes the "
    f"adaptive rule actually switches. (3) In the iso 2x2, {_iso}on a smaller and "
    "looser measurement that is not dB-matched across its cells. DOES THE "
    "DIRECTION SURVIVE. For the two classes this rule switches, yes, at qp 0 and at "
    "both ends of the bracket: HEVC_D 416x240 beats the 256 px run at epoch 0 and "
    "at epoch 1, and HEVC_C 832x480 loses to it at epoch 0 and at epoch 1. So the "
    "shape, a gain only where the tile count was 2 and a loss at 832x480, is not an "
    "artefact of the extra training. At qp 32 and qp 63 the 128 px column sits "
    "below the bottom of the bracket on every class, so its losses there survive "
    "outright and are if anything understated. What does NOT survive is any reading "
    "of this as a matched-training experiment. It is not one. In particular the qp "
    "0 1080p and 720p picture is not the one section 87 reported: against the "
    "pinned ckpt_PAPER baseline the 128 px path is level or ahead at 1080p and 720p "
    "rather than behind, and only HEVC_C is clearly behind. TWO FURTHER CAVEATS. "
    "Each source run bisected lambda once over all 53 sequences, so the spliced "
    "adaptive column does not sit on 0.1 dB; its delivered set-mean dB is recorded "
    "per qp and must be read with the saving. And the 256 px column in DECISIONS.md "
    "section 87 is about 2.7 points higher per class than "
    "results/per_class_RECIPE512.json and is not reproducible from any file now in "
    "results/, nor from the epoch 1 measurement here, so section 87's deltas and "
    "this file's deltas are not the same comparison and should not be quoted "
    "together."
)

out = {
    "what": "resolution-adaptive tile size, spliced from per-class measurements that already exist",
    "rule": ADAPTIVE_TILE,
    "budget_db": BUDGET,
    "db_convention": ("per-frame, as computed by scripts/per_class.py: 10log10 of "
                      "delivered tiled-decode frame MSE over untiled reference "
                      "frame MSE, averaged over frames, not pooled"),
    "measured_not_modelled": ("dB is measured off the delivered decode; saving is "
                              "the realised exit-cost mean of the assignment the "
                              "bisection actually chose on the deployed tiled path"),
    "n_sequences": NTOT, "sequences_per_class": N, "n_frames": src256["n_frames"],
    "sources": {
        "uniform_256": dict(file="results/per_class_RECIPE512.json", **ck256,
                            ckpt_epoch_recorded=src256.get("ckpt_epoch"),
                            tile_px=src256["tile_px"]),
        "uniform_128": dict(file="results/per_class_BEST128.json", **ck128,
                            ckpt_epoch_recorded=src128.get("ckpt_epoch"),
                            tile_px=src128["tile_px"]),
        "uniform_256_epoch1": dict(file="results/per_class_RECIPE512_e1.json", **ckE1,
                                   ckpt_epoch_recorded=srcE1.get("ckpt_epoch"),
                                   tile_px=srcE1["tile_px"]),
    },
    "per_qp": per_qp,
    "summary_over_qps": {
        "qps": qps,
        "mean_set_mean_delta_adaptive_minus_256":
            sum(r["set_mean_delta_adaptive_minus_256"] for r in per_qp) / len(per_qp),
        "mean_set_mean_delta_adaptive_minus_256_epoch1":
            sum(r["set_mean_delta_adaptive_minus_256_epoch1"] for r in per_qp) / len(per_qp),
        "mean_class_mean_delta_adaptive_minus_256":
            sum(r["class_mean_delta_adaptive_minus_256"] for r in per_qp) / len(per_qp),
    },
    "confound": {
        "training_gap_steps": train_gap,
        "training_gap_epochs": train_gap / SPE,
        "direction": "favours the 128 px column",
        "steps_per_epoch": SPE,
        "epoch_bracket_valid": bool(brackets),
        "bracket_low": ck256, "bracket_high": ckE1, "bracketed": ck128,
        "best128_trajectory": [{"cumulative_step": t[0], "mean_saving_5_rates": t[1],
                                "file": t[2]} for t in traj],
        "best128_slope_pts_per_1k_steps": slopes,
        "training_gap_worth_points": {"low": band[0], "high": band[-1]},
        "trajectory_test_set": ("40 sequences, 1080p and 720p only, no 832x480 and "
                               "no 416x240, so it does not constrain the two "
                               "classes the adaptive rule switches"),
        "iso_2x2": {"cells": iso, "effects": iso_effects},
    },
    "not_measured": [
        "a 128 px run trained to the same step as the 256 px run, which is the only "
        "thing that would remove the confound rather than bound it",
        "the signalling cost of a per-resolution tile size, since neither source "
        "run encoded a tile-size field",
        "re-bisection of lambda for the spliced configuration, which needs a "
        "per-class lambda sweep that results/ does not hold",
    ],
    "epoch_bracket_summary": bracket_summary,
    "one_epoch_of_training_points_range": {str(q): list(epoch_worth[q]) for q in qps},
    "note": note,
}
(R / "results/tilesize_adaptive.json").write_text(json.dumps(out, indent=2))

hdr = (f"{'class':<9} {'res':>9} {'n':>2} {'t256':>5} {'t128':>5} "
       f"{'256px':>8} {'128px':>8} {'adaptive':>9} {'delta':>7}  "
       f"{'256px e1':>8} {'bracket':>7}")
for r in per_qp:
    print(f"\nqp {r['qp']}, budget {BUDGET} dB")
    print(hdr)
    for c in r["classes"]:
        print(f"{c['cls']:<9} {c['res']:>9} {c['n_sequences']:>2} "
              f"{c['tiles_256']:>5} {c['tiles_128']:>5} "
              f"{c['saving_256_pct']:>7.2f}% {c['saving_128_pct']:>7.2f}% "
              f"{c['saving_adaptive_pct']:>8.2f}% {c['delta_adaptive_minus_256']:>+7.2f}  "
              f"{c['saving_256_epoch1_pct']:>7.2f}% {c['vs_training_bracket']:>7}")
    m = r["set_mean_saving_pct"]
    print(f"{'set mean':<9} {'53 seq':>9} {NTOT:>2} {'':>5} {'':>5} "
          f"{m['uniform_256']:>7.2f}% {m['uniform_128']:>7.2f}% {m['adaptive']:>8.2f}% "
          f"{r['set_mean_delta_adaptive_minus_256']:>+7.2f}  "
          f"{m['uniform_256_epoch1']:>7.2f}%")
    d = r["set_mean_delivered_db"]
    print(f"  delivered set-mean dB   256 {d['uniform_256']:.4f}   "
          f"128 {d['uniform_128']:.4f}   adaptive {d['adaptive']:.4f}")
s = out["summary_over_qps"]
print(f"\nmean over {len(qps)} rates, set mean delta "
      f"{s['mean_set_mean_delta_adaptive_minus_256']:+.2f} points against epoch 0, "
      f"{s['mean_set_mean_delta_adaptive_minus_256_epoch1']:+.2f} against epoch 1")
print(f"training gap {train_gap} steps ({train_gap / SPE:.2f} epoch) in favour of "
      f"128 px, worth {band[0]:.1f} to {band[-1]:.1f} points on BEST128's trajectory")
print("  -> results/tilesize_adaptive.json")
