"""Audit frozen router scores against bitstream replayed placement quality.

Only the three preselected random maps and the fixed Bayer map enter each
within-case comparison. The router map itself is excluded: by construction it
maximizes its own separable score and would bias the rank test upward.
"""
from __future__ import annotations

import hashlib
import json
import math
from collections import defaultdict
from pathlib import Path

import numpy as np


ROOT = Path(__file__).resolve().parents[2]
BASE = ROOT / "proof/cpu_early_exit/results"
SCAN = BASE / "kodak24_qp5"
RANDOM = BASE / "kodak24_qp5_placement_20261005"
BAYER = BASE / "kodak24_qp5_bayer_histogram_20261005"
OUT = BASE / "kodak_score_ranking_20261005.json"


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def pair_counts(candidates: list[tuple[float, float]],
                min_gap_db: float = 0.0) -> tuple[int, int, int]:
    concordant = discordant = tied = 0
    for i, (score_i, mse_i) in enumerate(candidates):
        for score_j, mse_j in candidates[i + 1 :]:
            ds, dm = score_i - score_j, mse_j - mse_i
            if abs(10 * math.log10(mse_i / mse_j)) < min_gap_db:
                continue
            if abs(ds) < 1e-10 or abs(dm) < 1e-12:
                tied += 1
            elif ds * dm > 0:
                concordant += 1
            else:
                discordant += 1
    return concordant, discordant, tied


def main() -> None:
    paths = sorted(RANDOM.glob("kodim*_qp*.json"))
    assert len(paths) == 120
    cases = []
    by_image = defaultdict(list)
    hashes = {}
    for path in paths:
        scan_path, bayer_path = SCAN / path.name, BAYER / path.name
        hashes[path.name] = {"scan": sha256(scan_path), "random": sha256(path),
                             "bayer": sha256(bayer_path)}
        scan = json.loads(scan_path.read_text())
        random = json.loads(path.read_text())
        bayer = json.loads(bayer_path.read_text())
        assert scan["image"] == random["image"] == bayer["image"]
        assert scan["qp"] == random["qp"] == bayer["qp"]
        assert random["stream_sha256"] == bayer["stream_sha256"] == scan["stream_sha256"]
        assert random["exit_map"] == bayer["router_map"]
        scores = np.asarray(scan["route_log_probs"], dtype=np.float64)
        assert scores.shape == (6, 4)

        # Deduplicate identical maps, excluding the router's actual map.
        maps = {tuple(p["map"]): float(p["mse_rgb"]) for p in random["permutations"]}
        bayer_map = tuple(bayer["bayer_map"])
        if bayer_map in maps:
            assert abs(maps[bayer_map] - bayer["bayer"]["mse_rgb"]) < 1e-12
        maps[bayer_map] = float(bayer["bayer"]["mse_rgb"])
        maps.pop(tuple(random["exit_map"]), None)
        assert all(sorted(mode) == sorted(random["exit_map"]) for mode in maps)
        candidate_rows = [
            {"map": list(mode),
             "score": float(sum(scores[t, depth - 2] for t, depth in enumerate(mode))),
             "mse_rgb": mse}
            for mode, mse in sorted(maps.items())
        ]
        pair_data = [(x["score"], x["mse_rgb"]) for x in candidate_rows]
        win, loss, tie = pair_counts(pair_data)
        material_win, material_loss, material_tie = pair_counts(pair_data, 0.01)
        row = {"image": scan["image"], "qp": scan["qp"],
               "blind_maps": len(candidate_rows), "concordant": win,
               "discordant": loss, "tied": tie,
               "material_concordant": material_win,
               "material_discordant": material_loss,
               "material_tied": material_tie,
               "candidates": candidate_rows}
        cases.append(row)
        by_image[scan["image"]].append((win, loss))

    assert len(by_image) == 24 and all(len(v) == 5 for v in by_image.values())
    wins, losses, ties = (sum(r[k] for r in cases) for k in
                           ("concordant", "discordant", "tied"))
    mat_wins, mat_losses = (sum(r[k] for r in cases) for k in
                            ("material_concordant", "material_discordant"))
    assert wins + losses > 0
    rng = np.random.default_rng(20261005)
    image_pairs = np.asarray([[sum(x[0] for x in by_image[im]),
                               sum(x[1] for x in by_image[im])]
                              for im in sorted(by_image)], dtype=np.int64)
    draws = rng.integers(0, 24, size=(10000, 24))
    sampled = image_pairs[draws].sum(axis=1)
    boot = sampled[:, 0] / sampled.sum(axis=1)
    per_qp = {}
    for qp in (0, 16, 32, 48, 63):
        subset = [r for r in cases if r["qp"] == qp]
        w = sum(r["concordant"] for r in subset)
        l = sum(r["discordant"] for r in subset)
        per_qp[str(qp)] = {"concordant": w, "discordant": l,
                           "accuracy": w / (w + l)}
    result = {
        "scope": "Kodak24 x QP5 score-quality rank audit using only preselected blind maps; same bitstreams and exit histogram within each case. Diagnostic, not independent held-out validation.",
        "definition": "For every pair of distinct non-router blind maps in one image-QP case, concordant means greater sum of frozen per-tile log probabilities yields lower replayed RGB MSE. Equal-score or <1e-12 MSE pairs are ties. Duplicated maps are removed.",
        "source_sha256": hashes,
        "bootstrap": "10000 resamples of 24 images, retaining all five QPs and their blind-map pairs per image; fixed checkpoint and candidate maps.",
        "overall": {"cases": len(cases), "cases_with_pairs": sum(r["concordant"] + r["discordant"] > 0 for r in cases),
                    "concordant": wins, "discordant": losses, "tied": ties,
                    "pair_accuracy": wins / (wins + losses),
                    "image_cluster_ci95": np.quantile(boot, [0.025, 0.975]).tolist(),
                    "material_pair_threshold_db": 0.01,
                    "material_concordant": mat_wins,
                    "material_discordant": mat_losses,
                    "material_pair_accuracy": mat_wins / (mat_wins + mat_losses)},
        "per_qp": per_qp,
        "cases": cases,
    }
    OUT.write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps({"overall": result["overall"], "per_qp": per_qp}, indent=2))


if __name__ == "__main__":
    main()
