"""Reproduce descriptive bank diagnostics and calibration sample-size checks.

Uses already inspected epoch20/30 CPU real-byte evaluations. No model training,
GPU execution, learned routing, external-test claim, or latency estimate.
Run from anywhere with Python + NumPy; outputs stay inside the paper bundle.
"""
from __future__ import annotations

import hashlib
import json
import math
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "data/ablation20260929"
DEPTHS = (2, 4, 6, 12)
METRICS = ("psnr_rgb", "psnr_yuv611")
RATES = (0.1, 0.2, 0.4)


def at_rate(rows, rate, metric):
    """Interpolate quality in log(payload-bpp), never extrapolate."""
    grouped = {}
    for row in rows:
        grouped.setdefault((row["image"], row["depth"]), []).append(row)
    result = {}
    for (image, depth), curve in sorted(grouped.items()):
        curve.sort(key=lambda r: r["payload_bpp"])
        x = np.array([r["payload_bpp"] for r in curve], dtype=float)
        y = np.array([r[metric] for r in curve], dtype=float)
        if len(x) != 5 or not np.all(np.isfinite(y)):
            raise ValueError("Incomplete or nonfinite RD curve")
        if not np.all(np.diff(x) > 0) or x[0] <= 0:
            raise ValueError("Non-increasing measured rate support")
        if x[0] <= rate <= x[-1]:
            result.setdefault(image, {})[depth] = float(
                np.interp(math.log(rate), np.log(x), y)
            )
    return result


def diagnostics(q, ids):
    """q is image by shallow depth; all selections here inspect test outcomes."""
    means = q.mean(axis=0)
    best_fixed = int(means.argmax())
    chosen = q.argmax(axis=1)  # ties deterministically prefer smaller depth
    best = q[np.arange(len(q)), chosen]
    margin = np.sort(q, axis=1)[:, -1] - np.sort(q, axis=1)[:, -2]
    options = {}
    for j, depth in enumerate(DEPTHS[:3]):
        without = np.delete(q, j, axis=1).max(axis=1)
        options[str(depth)] = {
            "mean_loss_without_expert_db": float((best - without).mean()),
            "images_losing_more_than_0p01_db": int(np.sum(best - without > 0.01)),
        }
    return {
        "mean_quality_db": {str(d): float(means[j]) for j, d in enumerate(DEPTHS[:3])},
        "cohort_selected_best_fixed_depth": DEPTHS[best_fixed],
        "source_informed_winner_count": {str(d): int(np.sum(chosen == j)) for j, d in enumerate(DEPTHS[:3])},
        "within_0p01_db_of_best_count_nonexclusive": {
            str(d): int(np.sum(best - q[:, j] <= 0.01)) for j, d in enumerate(DEPTHS[:3])
        },
        "source_informed_gain_over_cohort_selected_fixed_db": float((best - q[:, best_fixed]).mean()),
        "winner_margin_p10_p50_p90_db": np.quantile(margin, [0.1, 0.5, 0.9]).tolist(),
        "winner_margin_at_most_0p01_db_count": int(np.sum(margin <= 0.01)),
        "d6_minus_d4_mean_db": float((q[:, 2] - q[:, 1]).mean()),
        "d6_minus_d4_p10_p50_p90_db": np.quantile(q[:, 2] - q[:, 1], [0.1, 0.5, 0.9]).tolist(),
        "leave_one_expert_out_no_cost_constraint": options,
        "winners_by_image": {i: DEPTHS[int(j)] for i, j in zip(ids, chosen)},
    }


def zero_failure_bound(n, hypotheses, alpha=0.05):
    """One-sided exact binomial bound, Bonferroni across frozen hypotheses."""
    return -math.expm1(math.log(alpha / hypotheses) / n)


def minimum_zero_failure_n(risk, hypotheses, alpha=0.05):
    return math.ceil(math.log(alpha / hypotheses) / math.log1p(-risk))


def main():
    paths = {e: ROOT / f"data/research20260927/div2k100_epoch{e:03d}/analysis.json" for e in (20, 30)}
    data = {e: json.loads(p.read_text()) for e, p in paths.items()}
    if any(len(d["rows"]) != 2000 for d in data.values()):
        raise ValueError("Expected 100 images x 4 depths x 5 QPs per milestone")
    reference_checks = 0
    results = []
    for rate in RATES:
        quality = {(e, m): at_rate(d["rows"], rate, m) for e, d in data.items() for m in METRICS}
        ids = sorted(set.intersection(*[
            {i for i, values in q.items() if all(d in values for d in DEPTHS)}
            for q in quality.values()
        ]))
        if not ids:
            raise ValueError("No common rate support")
        # Verify the interpolation convention against the original RGB analysis.
        for e in data:
            archived = next(r for r in data[e]["matched_rate"] if
                            (r["rate_field"], r["interpolator"], r["target_bpp"]) ==
                            ("payload_bpp", "linear", rate))["quality_per_image"]
            for i in ids:
                for depth in DEPTHS:
                    if not math.isclose(quality[e, "psnr_rgb"][i][depth], archived[i][str(depth)], abs_tol=1e-10):
                        raise AssertionError("Interpolation differs from archived analysis")
                    reference_checks += 1
        q = {(e, m): np.array([[v[i][d] for d in DEPTHS] for i in ids]) for (e, m), v in quality.items()}
        record = {"payload_bpp": rate, "n": len(ids), "images": ids, "epochs": {}, "temporal_transfer": {}, "metric_transfer": {}}
        for e in data:
            record["epochs"][str(e)] = {}
            for m in METRICS:
                values = q[e, m]
                summary = diagnostics(values[:, :3], ids)
                summary["mean_gap_to_released_d12_db"] = {
                    str(d): float((values[:, j] - values[:, 3]).mean()) for j, d in enumerate(DEPTHS[:3])
                }
                record["epochs"][str(e)][m] = summary
            a = q[e, "psnr_rgb"][:, :3].argmax(axis=1)
            b = q[e, "psnr_yuv611"][:, :3].argmax(axis=1)
            yuv = q[e, "psnr_yuv611"][:, :3]
            record["metric_transfer"][str(e)] = {
                "different_rgb_yuv_winner_count": int(np.sum(a != b)),
                "yuv_mean_regret_using_rgb_winner_db": float((yuv.max(axis=1) - yuv[np.arange(len(ids)), a]).mean()),
            }
        for m in METRICS:
            old, new = q[20, m][:, :3], q[30, m][:, :3]
            a, b = old.argmax(axis=1), new.argmax(axis=1)
            old_choice_new = new[np.arange(len(ids)), a]
            old_advantage = old[np.arange(len(ids)), a] - old[np.arange(len(ids)), b]
            new_advantage = new[np.arange(len(ids)), b] - old_choice_new
            record["temporal_transfer"][m] = {
                "winner_changed_count": int(np.sum(a != b)),
                "winner_changed_with_both_pairwise_margins_over_0p01_db_count": int(np.sum((a != b) & (old_advantage > 0.01) & (new_advantage > 0.01))),
                "epoch30_regret_from_epoch20_source_winners_db": float((new.max(axis=1) - old_choice_new).mean()),
                "epoch20_winners_epoch30_mean_quality_db": float(old_choice_new.mean()),
                "epoch30_best_fixed_mean_quality_db": float(new.mean(axis=0).max()),
                "note": "Same already-observed images; this is checkpoint-label stability, NOT held-out predictor transfer.",
            }
        if any(not np.allclose(q[20, m][:, 3], q[30, m][:, 3], rtol=0, atol=1e-10) for m in METRICS):
            raise AssertionError("Released anchor changed")
        results.append(record)
    risk = [{"frozen_hypotheses": h, "target_exceedance_probability": r,
             "minimum_independent_units_if_zero_failures": minimum_zero_failure_n(r, h),
             "upper_bound_if_zero_failures_n51": zero_failure_bound(51, h),
             "upper_bound_if_zero_failures_n100": zero_failure_bound(100, h)}
            for h in (1, 6, 60) for r in (0.05, 0.10)]
    for row in risk:
        n = row["minimum_independent_units_if_zero_failures"]
        h, r = row["frozen_hypotheses"], row["target_exceedance_probability"]
        assert zero_failure_bound(n, h) <= r
        assert n == 1 or zero_failure_bound(n - 1, h) > r
    output = {
        "scope": "Post-hoc descriptive whole-crop bank diagnostics on fixed epoch20/30 actual-payload RD curves. No spatial routing or runtime result.",
        "selection_bias": "All winner and best-fixed selections use this observed cohort. Gains are optimistic outcome-informed diagnostics, not out-of-sample estimates or deployable policy performance.",
        "metric": "Mean per-image PSNR. psnr_yuv611 uses weighted per-plane PSNR on clipped YCbCr 4:4:4 in this bank evaluator, NOT legacy shared-exit 4:2:0. No conversion of weighted PSNR to additive MSE.",
        "cost": "Unconstrained quality comparisons only. Leave-one-out does not measure a matched-latency or matched-depth frontier.",
        "risk_scope": "Prospective exact binomial zero-failure sample-size arithmetic. Assumes independent identically distributed units and policies fixed independently of calibration outcomes; M includes every tested policy/budget. No guarantee for existing development-used 51 clusters, no observed zero-failure claim.",
        "source_sha256": {str(p.relative_to(ROOT)): hashlib.sha256(p.read_bytes()).hexdigest() for p in paths.values()},
        "script_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        "checks": {"archived_rgb_values_reproduced": reference_checks, "released_anchor_equal_across_epochs": True, "zero_failure_minimum_n_boundaries": True},
        "results": results, "calibration_sample_sizes": risk,
    }
    OUT.mkdir(parents=True, exist_ok=True)
    (OUT / "readiness_audit.json").write_text(json.dumps(output, indent=2, allow_nan=False) + "\n")
    print(json.dumps({"checks": output["checks"], "rates_n": [[r["payload_bpp"], r["n"]] for r in results], "output": str(OUT / "readiness_audit.json")}, indent=2))


if __name__ == "__main__":
    main()
