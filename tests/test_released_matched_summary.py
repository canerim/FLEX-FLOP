"""Matched-kernel claims must be recomputable from the paired raw trials."""
import copy
import glob
import json
from pathlib import Path

import pytest

from proof.early_exit_vs_released.summarize import summarize


DATA = Path(__file__).resolve().parents[1] / "proof/early_exit_vs_released/results/cohort_20261003"


def _matched(tmp_path):
    paths = []
    for source in sorted(glob.glob(str(DATA / "*qp*.json"))):
        row = json.loads(Path(source).read_text())
        row["schema"] = 3
        row["released_triton_patches"] = {
            "fused_ffn": 14, "fused_plain_wsilu": 14, "fused_blocks": 14}
        row["released_stock_vs_triton_max_abs"] = 0.0
        row["e15_all_deep_stock_vs_triton_max_abs"] = 0.0
        row["samples"]["released_d12_triton"] = copy.deepcopy(row["samples"]["released_d12"])
        row["samples"]["e15_all_deep_stock"] = copy.deepcopy(row["samples"]["e15_stock"])
        row["samples"]["e15_all_deep_triton"] = copy.deepcopy(row["samples"]["e15_triton"])
        row["median_speedup_released_triton_vs_e15_triton_wall"] = row["median_speedup_wall"]
        row["median_speedup_e15_all_deep_vs_routed_stock_wall"] = 1.0
        row["median_speedup_e15_all_deep_vs_routed_triton_wall"] = 1.0
        target = tmp_path / Path(source).name
        target.write_text(json.dumps(row))
        paths.append(target)
    return paths


def test_matched_summary_uses_raw_paired_samples(tmp_path):
    summary = summarize(_matched(tmp_path))
    assert summary["matched_kernels_median_of_scenario_paired_medians"] == pytest.approx(
        summary["median_of_scenario_paired_medians"])
    assert summary["routing_stock_median_of_scenario_paired_medians"] == pytest.approx(1.0)
    assert summary["routing_triton_median_of_scenario_paired_medians"] == pytest.approx(1.0)


def test_matched_summary_rejects_unsupported_claim(tmp_path):
    paths = _matched(tmp_path)
    row = json.loads(paths[0].read_text())
    row["median_speedup_released_triton_vs_e15_triton_wall"] = 9.0
    paths[0].write_text(json.dumps(row))
    with pytest.raises(ValueError, match="disagrees"):
        summarize(paths)
