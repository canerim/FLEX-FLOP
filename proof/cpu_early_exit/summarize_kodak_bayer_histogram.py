"""Image-cluster summary of the matched-histogram Kodak Bayer control."""
from __future__ import annotations

import hashlib
import json
from pathlib import Path

import numpy as np

BASE = Path(__file__).resolve().parent / "results/kodak24_qp5_bayer_histogram_20261005"
RANDOM_SUMMARY = Path(__file__).resolve().parent / "results/kodak24_qp5_placement_20261005/summary.json"
QPS = (0, 16, 32, 48, 63)


def digest(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main() -> None:
    files = sorted(BASE.glob("kodim*_qp*.json"))
    assert len(files) == 120
    manifest_path = BASE / "manifest.json"
    rows = {(r["image"], r["qp"]): r for path in files if
            (r := json.loads(path.read_text()))}
    assert len(rows) == 120
    gain = np.empty((24, 5, 2), dtype=np.float64)
    same = np.empty((24, 5), dtype=bool)
    per_case = []
    for i in range(24):
        image = f"kodim{i+1:02d}.png"
        for j, qp in enumerate(QPS):
            r = rows[image, qp]
            assert r["manifest_sha256"] == digest(manifest_path)
            assert (r["router_map"] == r["bayer_map"]) == r["same_map"]
            assert r["exit_counts"] == [r["bayer_map"].count(k) for k in range(6)]
            for m, metric in enumerate(("mse_rgb", "mse_444")):
                assert r["router"][metric] > 0 and r["bayer"][metric] > 0
                gain[i, j, m] = 10 * np.log10(r["bayer"][metric] / r["router"][metric])
            same[i, j] = r["same_map"]
            per_case.append({"image": image, "qp": qp,
                             "rgb_gain_db": float(gain[i, j, 0]),
                             "ycbcr444_gain_db": float(gain[i, j, 1]),
                             "same_map": bool(same[i, j])})
    rng = np.random.default_rng(20261005)
    draw = rng.integers(0, 24, size=(10_000, 24))
    boot = gain[draw].mean(1)
    random_summary = json.loads(RANDOM_SUMMARY.read_text())
    random_rows = {(r["image"], r["qp"]): r for r in random_summary["per_image_qp"]}
    assert len(random_rows) == 120
    random_gain = np.asarray([[random_rows[f"kodim{i+1:02d}.png", qp]["rgb_gain_db"]
                               for qp in QPS] for i in range(24)])
    random_minus_bayer = random_gain - gain[..., 0]
    contrast_boot = random_minus_bayer[draw].mean((1, 2))
    def summary(v, ci_values, same_values):
        return {"n": int(np.prod(v.shape[:-1])),
                "mean_router_gain_rgb_db": float(v[..., 0].mean()),
                "mean_router_gain_444_db": float(v[..., 1].mean()),
                "rgb_image_cluster_ci95_db": np.quantile(ci_values, [.025, .975]).tolist(),
                "router_better_cases": int((v[..., 0] > 1e-9).sum()),
                "bayer_better_cases": int((v[..., 0] < -1e-9).sum()),
                "same_map_cases": int(same_values.sum())}
    result = {
        "scope": "Actual Kodak24 x QP5 RGB and YCbCr444 reconstruction; fixed bitstream, e15 weights, routed depth counts and conv MAC. Positive gain means routed placement improves over the fixed Bayer-rank assignment of the same depths.",
        "limitations": "This is a deterministic matched-histogram placement control, not the archived scalar dithering policy; Kodak images are distinct from the CTC controller calibration but the codec and router were previously developed. No runtime inference.",
        "manifest_sha256": digest(manifest_path),
        "random_control_summary_sha256": digest(RANDOM_SUMMARY),
        "case_sha256": {p.name: digest(p) for p in files},
        "bootstrap": "10,000 draws of 24 images with replacement, retaining five QPs from each image; fixed checkpoint and maps.",
        "overall": summary(gain, boot[..., 0].mean(1), same),
        "random_minus_bayer_gain": {
            "mean_db": float(random_minus_bayer.mean()),
            "image_cluster_ci95_db": np.quantile(contrast_boot, [.025, .975]).tolist(),
            "interpretation": "Positive means the frozen router's advantage over random permutations is larger than its advantage over fixed Bayer; the interval includes zero.",
        },
        "per_qp": {str(qp): summary(gain[:, j], boot[:, j, 0], same[:, j])
                   for j, qp in enumerate(QPS)},
        "per_image_qp": per_case,
    }
    (BASE / "summary.json").write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps({"overall": result["overall"], "per_qp": result["per_qp"]}, indent=2))


if __name__ == "__main__":
    main()
