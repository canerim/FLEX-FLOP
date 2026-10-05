"""Summarise the locked QP32 sparse-gate scan after all 53 cases finish."""
from __future__ import annotations

import hashlib
import json
from pathlib import Path
import random

ROOT = Path(__file__).resolve().parents[2]
SOURCE = ROOT / "proof/cpu_early_exit/results/sparse_repair_gate_full_20261005.json"
OUT = ROOT / "proof/cpu_early_exit/results/sparse_repair_gate_full_summary_20261005.json"


def digest(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def bootstrap_ci(values: list[float]) -> list[float]:
    rng = random.Random(20261005)
    count = len(values)
    means = sorted(sum(values[rng.randrange(count)] for _ in range(count)) / count
                   for _ in range(5000))
    return [means[124], means[4874]]


def describe(rows: list[dict], threshold: float) -> dict:
    by_name = {}
    for row in rows:
        by_name.setdefault(row["sequence"], {})[row["threshold"]] = row
    assert all(set(x) == {0.0, .25, .30, 1.0} for x in by_name.values())
    pairs = [(v[0.0], v[threshold]) for v in by_name.values()]
    loss = [r["rgb_psnr_loss_vs_dense_repair_db"] - b["rgb_psnr_loss_vs_dense_repair_db"]
            for b, r in pairs]
    extra = [r["projected_exact_conv_mac_saving_pct"] -
             b["projected_exact_conv_mac_saving_pct"] for b, r in pairs]
    avg = lambda v: sum(v) / len(v)
    return {
        "n_frames": len(pairs), "threshold": threshold,
        "pointwise_active_fraction": avg([r["active_pointwise_fraction"] for _, r in pairs]),
        "trained_mean_exact_conv_mac_saving_pct": avg([b["projected_exact_conv_mac_saving_pct"] for b, _ in pairs]),
        "sparse_mean_projected_exact_conv_mac_saving_pct": avg([r["projected_exact_conv_mac_saving_pct"] for _, r in pairs]),
        "extra_projected_conv_mac_saving_points": avg(extra),
        "mean_additional_rgb_psnr_loss_db": avg(loss),
        "mean_additional_rgb_psnr_loss_ci95_db": bootstrap_ci(loss),
        "worst_additional_rgb_psnr_loss_db": max(loss),
        "quality_improved_count": sum(x < -1e-5 for x in loss),
        "quality_worsened_count": sum(x > 1e-5 for x in loss),
        "near_equal_count": sum(abs(x) <= 1e-5 for x in loss),
        "trained_over_0p1_db_count": sum(b["rgb_loss_vs_e15_dense_db"] > .1 for b, _ in pairs),
        "sparse_over_0p1_db_count": sum(r["rgb_loss_vs_e15_dense_db"] > .1 for _, r in pairs),
    }


def main() -> None:
    source = json.loads(SOURCE.read_text())
    rows = source["rows"]
    pilot = set(source["pilot_sequences"])
    extension = set(source["extension_sequences"])
    assert len(rows) == 53 * 4 and len(pilot) == 3 and len(extension) == 50
    assert not pilot & extension
    assert {r["sequence"] for r in rows} == pilot | extension
    assert all(r["active_pointwise_fraction"] == rows[0]["active_pointwise_fraction"]
               for r in rows if r["threshold"] == 0.0)
    result = {
        "scope": "Locked thresholds 0.25 and 0.30 chosen on three geometry-stratified pilot frames; primary extension has the other 50 QP32 CTC frames. Same frozen checkpoint, latent and Q90 map per frame.",
        "limitations": "The 50 extension frames were unused for threshold selection, but the codec and router were developed on CTC. MACs are projected for a sparse pointwise kernel; no runtime or other-QP quality is measured.",
        "bootstrap": "5000 paired image draws, seed 20261005; fixed checkpoint/cohort",
        "source_sha256": digest(SOURCE),
        "extension_50": [describe([r for r in rows if r["sequence"] in extension], t)
                         for t in (.25, .30, 1.0)],
        "all_53": [describe(rows, t) for t in (.25, .30, 1.0)],
    }
    OUT.write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps(result["extension_50"], indent=2))


if __name__ == "__main__":
    main()
