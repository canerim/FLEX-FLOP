"""Paired image-cluster summary of the frozen seam-regularized Kodak maps."""
from __future__ import annotations

import hashlib
import json
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
BASE = HERE / "results/kodak_seam_regularized_quality_20261005"
PLAN = HERE / "results/kodak_seam_regularized_plan_20261005.json"
QPS = (0, 16, 32, 48, 63)


def digest(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main() -> None:
    files = sorted(BASE.glob("kodim*_qp*.json"))
    assert len(files) == 120
    plan = json.loads(PLAN.read_text())
    manifest_path = BASE / "manifest.json"
    manifest = json.loads(manifest_path.read_text())
    assert manifest["plan_sha256"] == digest(PLAN)
    rows = {(r["image"], r["qp"]): r for path in files if
            (r := json.loads(path.read_text()))}
    assert len(rows) == 120
    gain = np.zeros((24, 5, 2), dtype=np.float64)
    changed = np.zeros((24, 5), dtype=bool)
    records = []
    for i in range(24):
        image = f"kodim{i+1:02d}.png"
        for j, qp in enumerate(QPS):
            r = rows[image, qp]
            assert r["manifest_sha256"] == digest(manifest_path)
            assert r["plan_sha256"] == digest(PLAN)
            assert r["exit_counts"] == [r["original_map"].count(k) for k in range(6)]
            assert r["exit_counts"] == [r["selected_map"].count(k) for k in range(6)]
            changed[i, j] = r["changed"]
            assert changed[i, j] == (r["original_map"] != r["selected_map"])
            for m, metric in enumerate(("mse_rgb", "mse_444")):
                a, b = r["original"][metric], r["seam_regularized"][metric]
                assert a > 0 and b > 0
                gain[i, j, m] = 10 * np.log10(a / b)
            records.append({"image": image, "qp": qp,
                            "changed": bool(changed[i, j]),
                            "rgb_improvement_db": float(gain[i, j, 0]),
                            "ycbcr444_improvement_db": float(gain[i, j, 1])})
    assert changed.sum() == plan["changed_cases"] == 14
    assert (gain[~changed] == 0).all()
    rng = np.random.default_rng(20261005)
    draws = rng.integers(0, 24, (10_000, 24))
    boot = gain[draws].mean(1)
    def describe(a, mask, ci):
        rgb = a[..., 0]
        return {"n": int(rgb.size), "changed": int(mask.sum()),
                "mean_rgb_improvement_db": float(rgb.mean()),
                "rgb_image_cluster_ci95_db": np.quantile(ci, [.025, .975]).tolist(),
                "mean_444_improvement_db": float(a[..., 1].mean()),
                "improved": int((rgb > 1e-9).sum()),
                "worsened": int((rgb < -1e-9).sum()),
                "unchanged": int((abs(rgb) <= 1e-9).sum()),
                "changed_only_mean_rgb_improvement_db": float(rgb[mask].mean())
                if mask.any() else 0.0,
                "worst_changed_rgb_improvement_db": float(rgb[mask].min())
                if mask.any() else 0.0}
    result = {
        "scope": "Frozen decoder-visible 0.5-TV regularization of router placement at exactly the same exit histogram and analytical synthesis MAC, Kodak24 x QP5. Positive improvement means regularized map has lower RGB distortion.",
        "limitations": "The 0.5 score-space penalty was chosen after inspecting router-score margins and map-change counts on Kodak but before quality replay. This is exploratory on a cohort used elsewhere in the project, not held-out validation. No bit-rate or runtime change measured.",
        "plan_sha256": digest(PLAN),
        "manifest_sha256": digest(manifest_path),
        "case_sha256": {p.name: digest(p) for p in files},
        "bootstrap": "10,000 paired image-cluster draws, seed 20261005; fixed weights, plan and codec.",
        "overall": describe(gain, changed, boot[..., 0].mean(1)),
        "per_qp": {str(qp): describe(gain[:, j], changed[:, j], boot[:, j, 0])
                   for j, qp in enumerate(QPS)},
        "rows": records,
    }
    (BASE / "summary.json").write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps({"overall": result["overall"], "per_qp": result["per_qp"]}, indent=2))


if __name__ == "__main__":
    main()
