"""Summarise frozen Kodak placement replay with image-cluster uncertainty."""
from __future__ import annotations

import hashlib
import json
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
BASE = HERE / "results/kodak24_qp5_placement_20261005"
QPS = (0, 16, 32, 48, 63)


def digest(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def ci(x: np.ndarray) -> list[float]:
    return np.quantile(x, [.025, .975]).tolist()


def main() -> None:
    manifest = json.loads((BASE / "manifest.json").read_text())
    files = sorted(BASE.glob("kodim*_qp*.json"))
    assert len(files) == 120
    rows = {(r["image"], r["qp"]): r for p in files if (r := json.loads(p.read_text()))}
    assert len(rows) == 120
    images = [f"kodim{i:02d}.png" for i in range(1, 25)]
    original = np.empty((24, 5, 2), dtype=np.float64)
    shuffled = np.empty((24, 5, 2, 3), dtype=np.float64)
    map_changed = np.zeros((24, 5), dtype=bool)
    for i, image in enumerate(images):
        for j, qp in enumerate(QPS):
            r = rows[image, qp]
            assert r["manifest_sha256"] == digest(BASE / "manifest.json")
            assert [x["seed"] for x in r["permutations"]] == manifest["seeds"]
            assert len(r["exit_map"]) == 6 and len(r["exit_counts"]) == 6
            assert sum(r["exit_counts"]) == 6
            assert r["original"]["mse_rgb"] > 0 and r["original"]["mse_444"] > 0
            map_changed[i, j] = len(set(r["exit_map"])) > 1
            for k, metric in enumerate(("mse_rgb", "mse_444")):
                original[i, j, k] = r["original"][metric]
                for s, item in enumerate(r["permutations"]):
                    shuffled[i, j, k, s] = item[metric]
                    assert shuffled[i, j, k, s] > 0
    gain = 10 * np.log10(shuffled.mean(-1) / original)
    rng = np.random.default_rng(20261005)
    indices = rng.integers(0, 24, size=(10_000, 24))
    image_means = gain[indices].mean(axis=1)  # draw, QP, metric
    seed_draw = rng.integers(0, 3, size=(10_000, 24, 5, 1, 3))
    nested_mse = np.take_along_axis(shuffled[indices], seed_draw, axis=-1).mean(-1)
    nested_means = (10 * np.log10(nested_mse / original[indices])).mean(axis=1)

    def describe(v: np.ndarray, scope: str) -> dict:
        return {"scope": scope,
                "n_image_qp": int(np.prod(v.shape[:-1])),
                "mean_rgb_gain_db": float(v[..., 0].mean()),
                "mean_444_gain_db": float(v[..., 1].mean())}

    result = {
        "scope": manifest["scope"], "limitations": manifest["limitation"],
        "manifest_sha256": digest(BASE / "manifest.json"),
        "case_sha256": {p.name: digest(p) for p in files},
        "bootstrap": "10,000 paired draws. Resample 24 Kodak images as clusters; each retains five QPs. Nested sensitivity also resamples three of the recorded permutation indices for each selected image-QP. Intervals do not cover unseen permutations, checkpoint seeds or other image populations.",
        "overall": {**describe(gain, "24 images x 5 QPs, equal image-QP weight"),
                    "rgb_image_cluster_ci95_db": ci(image_means[..., 0].mean(1)),
                    "rgb_nested_ci95_db": ci(nested_means[..., 0].mean(1)),
                    "nonuniform_map_cases": int(map_changed.sum())},
        "per_qp": {str(qp): {**describe(gain[:, j], f"24 Kodak images at QP {qp}"),
                              "rgb_image_cluster_ci95_db": ci(image_means[:, j, 0]),
                              "rgb_nested_ci95_db": ci(nested_means[:, j, 0]),
                              "nonuniform_map_cases": int(map_changed[:, j].sum())}
                   for j, qp in enumerate(QPS)},
        "per_image_qp": [{"image": image, "qp": qp,
                          "rgb_gain_db": float(gain[i, j, 0]),
                          "ycbcr444_gain_db": float(gain[i, j, 1]),
                          "nonuniform_map": bool(map_changed[i, j])}
                         for i, image in enumerate(images) for j, qp in enumerate(QPS)],
    }
    (BASE / "summary.json").write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps({"overall": result["overall"], "per_qp": result["per_qp"]}, indent=2))


if __name__ == "__main__":
    main()
