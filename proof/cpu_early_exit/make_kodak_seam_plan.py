"""Freeze a score-plus-seam reassignment before any quality replay.

For each Kodak image/QP, enumerate unique permutations of the six depths
selected by the frozen router. Maximize the sum of the router's adjusted
scores minus 0.5 times the sum of adjacent exit-index differences. This
uses decoder-visible scores, QP and geometry only; no source distortion.
"""
from __future__ import annotations

import hashlib
import itertools
import json
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
SCAN = ROOT / "proof/cpu_early_exit/results/kodak24_qp5"
OUT = ROOT / "proof/cpu_early_exit/results/kodak_seam_regularized_plan_20261005.json"
LAMBDA_TV = 0.5


def digest(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main() -> None:
    manifest_path = SCAN / "manifest.json"
    manifest = json.loads(manifest_path.read_text())
    assert manifest["cases"] == 120
    rows = []
    source_hashes = {}
    for image in manifest["images"]:
        for qp in manifest["qps"]:
            stem = f"{Path(image).stem}_qp{qp}"
            path = SCAN / f"{stem}.json"
            record = json.loads(path.read_text())
            assert record["manifest_sha256"] == digest(manifest_path)
            source_hashes[path.name] = digest(path)
            h, w = record["shape"]
            nh, nw = h // 256, w // 256
            assert h * w == 6 * 256**2 and nh * nw == 6
            score = np.asarray(record["route_log_probs"], dtype=np.float64)
            cost = np.asarray(record["router_costs"], dtype=np.float64)
            price = float(manifest["calibrated_beta"][str(qp)])
            adjusted = score - price * cost[2:][None, :]
            assert adjusted.shape == (6, 4)
            original = tuple((adjusted.argmax(1) + 2).tolist())
            assert [original.count(k) for k in range(6)] == record["exit_counts"]
            edges = [(i, i + 1) for i in range(6) if i % nw < nw - 1]
            edges += [(i, i + nw) for i in range(6) if i // nw < nh - 1]
            assert len(edges) == 7
            def utility(mode: tuple[int, ...]) -> float:
                return float(sum(adjusted[i, k - 2] for i, k in enumerate(mode)))
            def variation(mode: tuple[int, ...]) -> int:
                return sum(abs(mode[i] - mode[j]) for i, j in edges)
            candidates = sorted(set(itertools.permutations(original)))
            assert 1 <= len(candidates) <= 720
            selected = max(candidates, key=lambda mode: (
                utility(mode) - LAMBDA_TV * variation(mode),
                mode == original, mode))
            assert sorted(selected) == sorted(original)
            rows.append({
                "image": image, "qp": qp, "shape": [h, w],
                "scan_result_sha256": source_hashes[path.name],
                "original": list(original), "selected": list(selected),
                "changed": selected != original,
                "candidate_count": len(candidates),
                "original_score": utility(original),
                "selected_score": utility(selected),
                "original_tv": variation(original),
                "selected_tv": variation(selected),
                "mac_saved_pct": record["mac_saved_pct"],
            })
    assert len(rows) == 120
    result = {
        "scope": "Pre-quality frozen Kodak24 x QP5 map plan. Exact six-depth multiset and conv MAC per image/QP. Decoder-visible router scores, QP and tile geometry only; no source distortion is used in plan selection.",
        "lambda_tv": LAMBDA_TV,
        "score": "Sum over tiles of log_softmax_router_score minus calibrated_beta[QP] times per-exit synthesis cost; same scores as archived decision.",
        "variation": "Sum of absolute differences in exit index across the seven 4-neighbor edges of the 2x3 or 3x2 tile grid.",
        "selection": "Exact enumeration of unique permutations of the router's six selected exits. Maximize score minus lambda_tv*variation; prefer original on exact tie, then lexicographic order.",
        "script_sha256": digest(Path(__file__)),
        "scan_manifest_sha256": digest(manifest_path),
        "scan_case_sha256": source_hashes,
        "n_cases": len(rows),
        "changed_cases": sum(r["changed"] for r in rows),
        "changed_by_qp": {str(q): sum(r["changed"] for r in rows if r["qp"] == q)
                          for q in manifest["qps"]},
        "rows": rows,
    }
    OUT.write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps({k: result[k] for k in ("n_cases", "changed_cases", "changed_by_qp")}, indent=2))


if __name__ == "__main__":
    main()
