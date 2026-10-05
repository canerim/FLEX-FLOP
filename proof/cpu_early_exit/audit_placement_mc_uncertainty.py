"""Sensitivity of the QP32 placement intervention to its three permutations.

This is a nested bootstrap of recorded outcomes, not a claim about unseen
permutations or an independent test set. The three permutation indices are
resampled together for router and dither within each sampled image.
"""
from __future__ import annotations

import hashlib
import json
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
RAW = Path("/data10/shareddata/can_karsal/dcvcuf_depth_20260927/research/placement_replay_qp32/cases")
PUBLISHED = ROOT / "cvpr2027/data/research20260927/shared_crossfit_qp32/placement_replay.json"
OUT = ROOT / "docs/research/2026-10-05-placement-seed-sensitivity"
POLICIES = ("router", "dither")
SEEDS = (202609280, 202609281, 202609282)
N_DRAWS = 20_000


def sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def interval(values: np.ndarray) -> list[float]:
    return np.quantile(values, [.025, .975]).tolist()


def main() -> None:
    published = json.loads(PUBLISHED.read_text())
    paths = sorted(RAW.glob("[0-9][0-9].json"))
    assert len(paths) == published["n_sequences"] == 53
    names, original, shuffled = [], [], []
    for path in paths:
        case = json.loads(path.read_text())
        assert case["baseline_reproduced"] is True
        rows = {(r["policy"], r["variant"], r["seed"]): r for r in case["rows"]}
        expected = {(policy, "original", None) for policy in POLICIES} | {
            (policy, "permuted", seed) for policy in POLICIES for seed in SEEDS}
        assert len(rows) == len(case["rows"]) == 8 and set(rows) == expected
        names.append(case["sequence"])
        original.append([rows[policy, "original", None]["mse_rgb"] for policy in POLICIES])
        shuffled.append([[rows[policy, "permuted", seed]["mse_rgb"] for seed in SEEDS]
                         for policy in POLICIES])
    assert len(names) == len(set(names)) == 53
    original = np.asarray(original, dtype=np.float64)  # image, policy
    shuffled = np.asarray(shuffled, dtype=np.float64)  # image, policy, seed
    assert np.isfinite(original).all() and np.isfinite(shuffled).all()
    assert (original > 0).all() and (shuffled > 0).all()
    gain = 10 * np.log10(shuffled.mean(2) / original)
    expected = {(r["sequence"], r["policy"]): r["rgb_placement_gain_db"]
                for r in published["per_sequence"]}
    assert len(expected) == 106
    for i, name in enumerate(names):
        for j, policy in enumerate(POLICIES):
            assert abs(gain[i, j] - expected[name, policy]) < 1e-12

    rng = np.random.default_rng(20261005)
    image_draws = rng.integers(0, 53, size=(N_DRAWS, 53))
    seed_draws = rng.integers(0, 3, size=(N_DRAWS, 53, 3))
    nested_mse = np.take_along_axis(shuffled[image_draws], seed_draws[:, :, None, :], axis=3)
    nested_gain = 10 * np.log10(nested_mse.mean(3) / original[image_draws])
    nested_means = nested_gain.mean(1)
    image_means = gain[image_draws].mean(1)
    seed_only_mse = np.take_along_axis(shuffled[None], seed_draws[:, :, None, :], axis=3)
    seed_only_gain = 10 * np.log10(seed_only_mse.mean(3) / original[None])
    seed_only_means = seed_only_gain.mean(1)

    def describe(draws: np.ndarray) -> dict:
        return {"router_ci95_db": interval(draws[:, 0]),
                "dither_ci95_db": interval(draws[:, 1]),
                "router_minus_dither_ci95_db": interval(draws[:, 0] - draws[:, 1]),
                "paired_positive_fraction": float(np.mean(draws[:, 0] - draws[:, 1] > 0))}

    result = {
        "scope": "Sensitivity of frozen 53-image QP32 Q90 actual RGB reconstruction placement gains to the three recorded stratified permutations. No new codec evaluation or external-test claim.",
        "estimand": "Equal-image mean of 10log10(mean of three shuffled RGB MSE / original RGB MSE), paired router and dither.",
        "limitations": "Resampling three observed permutations is an approximate Monte Carlo sensitivity check. It cannot measure bias from their finite draw, unobserved permutation tails, training seeds or development-cohort reuse.",
        "bootstrap": {"draws": N_DRAWS, "seed": 20261005,
                      "nested": "sample 53 images with replacement, then three of each sampled image's recorded permutation indices with replacement; same indices for both policies"},
        "published_sha256": sha(PUBLISHED),
        "case_sha256": {path.name: sha(path) for path in paths},
        "n_images": len(names), "names": names,
        "point_estimate_db": {"router": float(gain[:, 0].mean()),
                              "dither": float(gain[:, 1].mean()),
                              "router_minus_dither": float((gain[:, 0] - gain[:, 1]).mean())},
        "image_only_conditional": describe(image_means),
        "permutation_only": describe(seed_only_means),
        "nested_images_and_permutations": describe(nested_means),
    }
    OUT.mkdir(parents=True, exist_ok=True)
    (OUT / "analysis.json").write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps({k: result[k] for k in ("point_estimate_db", "image_only_conditional",
                                                "permutation_only", "nested_images_and_permutations")}, indent=2))


if __name__ == "__main__":
    main()
