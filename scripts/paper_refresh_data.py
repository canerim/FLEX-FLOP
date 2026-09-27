"""Audited, CPU-only source data for the September 2026 manuscript refresh.

This does not run a codec or change any training artifact. All intervals are
paired sequence-cluster bootstrap intervals at one fixed codec/router pair.
"""
from __future__ import annotations

import csv
import hashlib
import json
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "paper" / "data" / "refresh20260927"
ARCHIVE = ROOT / "docs/reviews/encoder-routing-2026-09-24/data"
BUDGETS = [.05, .1, .15, .2, .3, .5]
RULES = ("oracle", "router", "dither", "uniform")
SEED, DRAWS = 20260927, 5000
SOURCES: dict[str, str] = {}


def sha(path):
    h = hashlib.sha256()
    with Path(path).open("rb") as f:
        for block in iter(lambda: f.read(1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()


def read(rel):
    path = ROOT / rel
    SOURCES[str(rel)] = sha(path)
    return json.loads(path.read_text())


def ci(values, groups):
    """Resample sequences, retaining their available QPs and paired methods."""
    values = np.asarray(values, dtype=float)
    unique, inv = np.unique(groups, return_inverse=True)
    sums = np.bincount(inv, weights=values)
    counts = np.bincount(inv)
    draw = np.random.default_rng(SEED).integers(0, len(unique), (DRAWS, len(unique)))
    means = sums[draw].sum(1) / counts[draw].sum(1)
    return dict(mean=float(values.mean()), lo=float(np.quantile(means, .025)),
                hi=float(np.quantile(means, .975)))


def write_csv(name, rows):
    with (OUT / name).open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0]), lineterminator="\n")
        writer.writeheader()
        writer.writerows(rows)


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    # The earlier analysis is useful secondary evidence only while its raw
    # sources remain byte-identical. Never silently consume a stale summary.
    archive_manifest = read(ARCHIVE.relative_to(ROOT) / "source_manifest.json")
    for rel, expected in archive_manifest["sources"].items():
        actual = sha(ROOT / rel)
        if actual != expected:
            raise RuntimeError(f"Earlier analysis has a changed source: {rel}")
        SOURCES[rel] = actual
    raw = read("flexplus/results/eval_rules_ctc_e15.json")
    rows, costs = raw["rows"], np.array(raw["cost"])
    assert len(rows) == 265 and len({r["seq"] for r in rows}) == 53
    assert len({(r["seq"], r["qp"]) for r in rows}) == len(rows)
    assert sorted({r["qp"] for r in rows}) == [0, 16, 32, 48, 63]
    assert raw["j"] == 2 and raw["K"] == 6
    for r in rows:
        for rule in RULES:
            for value in r["rules"][rule].values():
                if value is None:
                    continue
                k = np.array(value["map"])
                assert len(k) == int(np.prod(r["grid"]))
                assert set(k) <= {2, 3, 4, 5}
                assert np.bincount(k, minlength=6).tolist() == value["hist"]
                assert abs(100 * (1 - costs[k].mean()) - value["saving"]) < 1e-9
                assert np.isfinite([value[f] for f in ("saving", "db_rgb", "db_611")]).all()

    stable = [r for r in rows if all(r["rules"][k].get(str(b)) is not None
                                   for b in BUDGETS for k in RULES)]
    summary, contrasts, samples, by_qp = [], [], [], []
    for b in BUDGETS:
        key = str(b)
        common = [r for r in rows if all(r["rules"][k].get(key) is not None for k in RULES)]
        groups = [r["seq"] for r in common]
        for rule in RULES:
            vals = [r["rules"][rule][key] for r in common]
            interval = ci([v["saving"] for v in vals], groups)
            stable_vals = [r["rules"][rule][key]["saving"] for r in stable]
            summary.append(dict(budget=b, rule=rule, n=len(common),
                                n_sequences=len(set(groups)), **interval,
                                rgb_mean=float(np.mean([v["db_rgb"] for v in vals])),
                                yuv_mean=float(np.mean([v["db_611"] for v in vals])),
                                rgb_max=max(v["db_rgb"] for v in vals),
                                yuv_max=max(v["db_611"] for v in vals),
                                rgb_over=sum(v["db_rgb"] > b + 1e-4 for v in vals),
                                yuv_over=sum(v["db_611"] > b + 1e-4 for v in vals),
                                stable_n=len(stable), stable_mean=float(np.mean(stable_vals))))
            for r, v in zip(common, vals):
                samples.append(dict(sequence=r["seq"], qp=r["qp"], group=r["cls"],
                                    budget=b, rule=rule, saving=v["saving"],
                                    rgb_loss=v["db_rgb"], yuv_loss=v["db_611"],
                                    tiles=len(v["map"])))
        for a, c in [("router", "dither"), ("oracle", "router"), ("oracle", "dither")]:
            vals = [r["rules"][a][key]["saving"] - r["rules"][c][key]["saving"] for r in common]
            fixed = [r["rules"][a][key]["saving"] - r["rules"][c][key]["saving"] for r in stable]
            contrasts.append(dict(budget=b, contrast=f"{a}_minus_{c}", n=len(common),
                                  **ci(vals, groups), stable_mean=float(np.mean(fixed))))
        for qp in [0, 16, 32, 48, 63]:
            subset = [r for r in common if r["qp"] == qp]
            v = [r["rules"]["router"][key]["saving"] - r["rules"]["dither"][key]["saving"] for r in subset]
            by_qp.append(dict(budget=b, qp=qp, n=len(subset), **ci(v, [r["seq"] for r in subset])))

    archived = read(ARCHIVE.relative_to(ROOT) / "analysis.json")
    # Independently recomputed point estimates must agree with the prior audit.
    # CI seed has intentionally changed; intervals need not match exactly.
    for r in summary:
        old = next(x for x in archived["dcvcuf"] if x["budget"] == r["budget"] and x["rule"] == r["rule"])
        assert r["n"] == old["n"] and abs(r["mean"] - old["saving"]) < 1e-10
        assert r["rgb_over"] == old["over_rgb"] and r["yuv_over"] == old["over_611"]
    # Choose one illustrative example among archived thumbnails, by a stated
    # representativeness criterion rather than by maximising the gain.
    thumb_path = ROOT / "flexplus/results/eval_rules_thumbs.npz"
    thumbs = np.load(thumb_path)
    q32 = [r for r in rows if r["qp"] == 32]
    candidates = [(int(k), q32[int(k)]) for k in thumbs.files
                  if all(q32[int(k)]["rules"][rule].get(str(b)) is not None
                         for rule in RULES for b in [.1, .3])]
    premium = next(r["mean"] for r in contrasts if r["budget"] == .1 and r["contrast"] == "router_minus_dither")
    idx, example = min(candidates, key=lambda p: abs(p[1]["rules"]["router"]["0.1"]["saving"] -
                                                   p[1]["rules"]["dither"]["0.1"]["saving"] - premium))
    output = dict(protocol=dict(n_sequences=53, n_frame_qp=265, first_frame_only=True,
                               qp=[0, 16, 32, 48, 63], budgets=BUDGETS,
                               bootstrap_draws=DRAWS, bootstrap_seed=SEED,
                               ci_scope="Sequence-cluster resampling; fixed codec and router; not training-seed uncertainty",
                               checkpoint=raw["ckpt"], router=raw["router2"],
                               budget_calibration="Per-frame source-informed selection on padded RGB tables",
                               measurement="Actual mixed reconstructions cropped to the source extent; losses relative to full-frame fine-tuned e15, despite raw field psnr_release",
                               selection_reference="Separate released-weight warm-start on padded source; differs from reporting anchor and support",
                               over_target_counts="Reported losses above nominal numeric target; not an isolated guarantee test under a common reference",
                               cost_scope="Decoder MAC model; excludes router, signalling and runtime overhead",
                               inference="No new GPU evaluation in this manuscript refresh"),
                  costs=costs.tolist(), ceiling_pct=100 * (1 - float(costs[2])),
                  summary=summary, contrasts=contrasts, by_qp=by_qp,
                  example=dict(index=idx, selection="Closest router–dither premium to the aggregate at 0.1 dB among five archived thumbnails", **example),
                  runtime=archived["runtime"], other_decoders=archived["other_decoders"],
                  perception=archived["perception"])
    for name, vals in [("summary.csv", summary), ("contrasts.csv", contrasts),
                       ("samples.csv", samples), ("by_qp.csv", by_qp)]:
        write_csv(name, vals)
    (OUT / "analysis.json").write_text(json.dumps(output, indent=2) + "\n")
    SOURCES[str(Path(__file__).relative_to(ROOT))] = sha(__file__)
    # Checkpoint hashes establish present provenance, not historical execution.
    for rel in [raw["ckpt"], raw["router2"], "flexuf/backbone/decoder.py",
                "flexuf/router/head2.py", "flexplus/eval_rules_ctc.py"]:
        SOURCES[rel] = sha(ROOT / rel)
    (OUT / "source_manifest.json").write_text(json.dumps(dict(sources=SOURCES,
        caveat="Checkpoint/code hashes recorded at refresh time; evaluation JSON did not pin these hashes at execution time."), indent=2) + "\n")
    macros = {}
    for b, token in [(.05, "Tight"), (.1, "Main"), (.2, "Medium"), (.3, "Loose"), (.5, "Saturated")]:
        for rule in RULES:
            r = next(r for r in summary if r["budget"] == b and r["rule"] == rule)
            macros[token + rule.title() + "Saving"] = f"{r['mean']:.2f}"
        c = next(r for r in contrasts if r["budget"] == b and r["contrast"] == "router_minus_dither")
        macros[token + "Premium"] = f"{c['mean']:.2f}"
        macros[token + "PremiumLow"] = f"{c['lo']:.2f}"
        macros[token + "PremiumHigh"] = f"{c['hi']:.2f}"
        macros[token + "N"] = str(c["n"])
    main = next(r for r in summary if r["budget"] == .1 and r["rule"] == "router")
    macros.update(MainRgbOver=str(main["rgb_over"]), MainYuvOver=str(main["yuv_over"]),
                  MainRgbMax=f"{main['rgb_max']:.3f}", MainYuvMax=f"{main['yuv_max']:.3f}",
                  MacCeiling=f"{output['ceiling_pct']:.2f}")
    (OUT / "macros.tex").write_text("% Generated by scripts/paper_refresh_data.py; do not edit.\n" +
        "\n".join("\\newcommand{\\" + k + "}{" + v + "}" for k, v in macros.items()) + "\n")
    print(json.dumps(dict(output=str(OUT), rows=len(samples), sources=len(SOURCES),
                          example=example["seq"], all_checks_passed=True)))


if __name__ == "__main__":
    main()
