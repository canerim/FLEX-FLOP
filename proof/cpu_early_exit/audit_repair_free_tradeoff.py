"""Matched frozen-weight repair ablation on the completed QP32 CPU replay.

No model execution is performed. The same checkpoint, stream, and routed map
produce both outputs in the source intervention archive. This script changes
only the convolution MAC count to account for omitting GridSeamRepair.
"""
from __future__ import annotations

import hashlib
import json
from pathlib import Path
import random
import sys

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "proof/early_exit_vs_released"))
from mac_latency_audit import CHANNELS, FEATURE_STRIDE, case_macs

SOURCE = ROOT / "cvpr2027/data/research20260927/component_interventions/analysis.json"
OUT = ROOT / "proof/cpu_early_exit/results/repair_free_tradeoff_20261005.json"


def digest(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def interval(values: list[float]) -> list[float]:
    rng = random.Random(20261005)
    n = len(values)
    draws = sorted(sum(values[rng.randrange(n)] for _ in range(n)) / n
                   for _ in range(5000))
    return [draws[124], draws[4874]]


def main() -> None:
    source = json.loads(SOURCE.read_text())
    assert source["n_sequences"] == 53 and len(source["cases"]) == 53
    assert source["n_outputs"] == 212
    rows = []
    for case in source["cases"]:
        assert case["baseline_matches_complete_replay"]
        assert all(case["head_replay_exact"])
        height = (case["height"] + 255) // 256 * 256
        width = (case["width"] + 255) // 256 * 256
        counts = [case["map"].count(k) for k in range(6)]
        assert counts[:2] == [0, 0] and sum(counts) == height * width // 256**2
        m = case_macs({"padded_shape": [height, width], "tile_counts": counts})
        feature_pixels = height * width // FEATURE_STRIDE**2
        repair_macs = feature_pixels * (CHANNELS**2 + 9 * CHANNELS)
        variants = {row["variant"]: row for row in case["rows"]}
        assert set(variants) == {"trained", "repair_identity", "adapters_identity", "both_identity"}
        trained, skipped = variants["trained"], variants["repair_identity"]
        quality_loss = trained["psnr_rgb"] - skipped["psnr_rgb"]
        assert abs(quality_loss - skipped["loss_vs_original_db"]) < 1e-9
        trained_saving = 100 * m["e15_routed_conv_mac_saving_fraction"]
        skipped_saving = 100 * (1 - (m["e15_routed_conv_macs"] - repair_macs) /
                                m["released_conv_macs"])
        assert skipped_saving >= trained_saving
        rows.append({
            "sequence": case["sequence"], "qp": 32,
            "padded_shape": [height, width], "tile_counts": counts,
            "trained_exact_conv_mac_saving_pct": trained_saving,
            "repair_free_exact_conv_mac_saving_pct": skipped_saving,
            "incremental_exact_conv_mac_saving_points": skipped_saving - trained_saving,
            "repair_macs": repair_macs,
            "released_full_synthesis_macs": m["released_conv_macs"],
            "trained_rgb_psnr_db": trained["psnr_rgb"],
            "repair_free_rgb_psnr_db": skipped["psnr_rgb"],
            "repair_removal_rgb_psnr_loss_db": quality_loss,
            "trained_rgb_loss_vs_e15_dense_db": trained["loss_vs_dense_anchor_db"],
            "repair_free_rgb_loss_vs_e15_dense_db": skipped["loss_vs_dense_anchor_db"],
        })
    assert len({r["sequence"] for r in rows}) == 53
    mean = lambda name: sum(r[name] for r in rows) / len(rows)
    quality = [r["repair_removal_rgb_psnr_loss_db"] for r in rows]
    result = {
        "scope": "Frozen e15 QP32 Q90 router map, same 53 frames/checkpoint/stream; repair_on versus identity substitution; cropped RGB PSNR against source.",
        "cost_scope": "Exact padded-frame convolution MAC against released D12 full synthesis; repair-free path subtracts one full-frame 1x1 plus 3x3 depthwise repair. No timing, memory, entropy or router cost is inferred.",
        "interpretation": "Frozen-weight inference ablation, not a retrained repair-free model. The quality trade-off is on development-used frames and does not establish safe deployment or a full-codec speedup.",
        "n_frames": len(rows),
        "summary": {
            "trained_exact_conv_mac_saving_pct": mean("trained_exact_conv_mac_saving_pct"),
            "repair_free_exact_conv_mac_saving_pct": mean("repair_free_exact_conv_mac_saving_pct"),
            "incremental_exact_conv_mac_saving_points": mean("incremental_exact_conv_mac_saving_points"),
            "mean_repair_removal_rgb_psnr_loss_db": sum(quality) / len(quality),
            "mean_repair_removal_rgb_psnr_loss_ci95_db": interval(quality),
            "max_repair_removal_rgb_psnr_loss_db": max(quality),
            "trained_above_0p1_db_count": sum(r["trained_rgb_loss_vs_e15_dense_db"] > .1 for r in rows),
            "repair_free_above_0p1_db_count": sum(r["repair_free_rgb_loss_vs_e15_dense_db"] > .1 for r in rows),
        },
        "bootstrap": "5000 paired frame draws, seed 20261005; fixed checkpoint and development cohort",
        "source_sha256": digest(SOURCE),
        "cost_script_sha256": digest(ROOT / "proof/early_exit_vs_released/mac_latency_audit.py"),
        "script_sha256": digest(Path(__file__)),
        "rows": rows,
    }
    OUT.parent.mkdir(parents=True, exist_ok=True)
    OUT.write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps(result["summary"], indent=2))


if __name__ == "__main__":
    main()
