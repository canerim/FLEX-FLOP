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
        row["schema"] = 2
        row["released_triton_patches"] = {
            "fused_ffn": 14, "fused_plain_wsilu": 14, "fused_blocks": 14}
        row["released_stock_vs_triton_max_abs"] = 0.0
        row["samples"]["released_d12_triton"] = copy.deepcopy(row["samples"]["released_d12"])
        row["median_speedup_released_triton_vs_e15_triton_wall"] = row["median_speedup_wall"]
        target = tmp_path / Path(source).name
        target.write_text(json.dumps(row))
        paths.append(target)
    return paths


def test_matched_summary_uses_raw_paired_samples(tmp_path):
    summary = summarize(_matched(tmp_path))
    assert summary["matched_kernels_median_of_scenario_paired_medians"] == pytest.approx(
        summary["median_of_scenario_paired_medians"])


def test_matched_summary_rejects_unsupported_claim(tmp_path):
    paths = _matched(tmp_path)
    row = json.loads(paths[0].read_text())
    row["median_speedup_released_triton_vs_e15_triton_wall"] = 9.0
    paths[0].write_text(json.dumps(row))
    with pytest.raises(ValueError, match="disagrees"):
        summarize(paths)
