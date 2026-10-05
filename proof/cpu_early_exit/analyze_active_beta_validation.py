"""Paired analysis of the calibration-locked active-context beta replay.

Validation outcomes only evaluate the already frozen QP-specific beta policy.
No choice or adjustment is made in this script.
"""
from __future__ import annotations

import hashlib
import json
from pathlib import Path

import numpy as np
from exact_active_conv_mac import exact_active_no_repair_saving_pct

HERE = Path(__file__).resolve().parent
BASE = HERE / 'results/div2k_beta/quality_floor'
RAW = BASE / 'active_replicate_beta_validation24.json'
POLICY = BASE / 'active_replicate_beta_locked_policy.json'
FRONTIER = BASE / 'active_replicate_beta_calibration24.json'
OUT = BASE / 'active_replicate_beta_validation24_analysis.json'
QPS = (0, 16, 32, 48, 63)


def sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def summarize(rows: list[dict], field: str, rng: np.random.Generator) -> dict:
    images = sorted({r['image'] for r in rows})
    n_qp = len(rows)//len(images)
    assert len(images) == 24 and n_qp in (1, 5)
    matrix = np.asarray([[r[field] for r in rows if r['image'] == image]
                         for image in images], dtype=float)
    assert matrix.shape == (24, n_qp)
    means = matrix[rng.integers(0, 24, size=(10000, 24))].mean(axis=(1, 2))
    values = matrix.ravel()
    return {'mean': float(values.mean()),
            'image_cluster_ci95': np.quantile(means, [.025, .975]).tolist(),
            'min': float(values.min()), 'max': float(values.max()),
            'higher': int((values > 1e-9).sum()),
            'lower': int((values < -1e-9).sum()),
            'equal': int((np.abs(values) <= 1e-9).sum())}


def main() -> None:
    raw = json.loads(RAW.read_text())
    policy = json.loads(POLICY.read_text())
    frontier = json.loads(FRONTIER.read_text())
    assert raw['complete'] and len(raw['rows']) == raw['expected_cases'] == 120
    assert frontier['complete'] and policy['source_sha256']['frontier'] == sha(FRONTIER)
    assert raw['provenance']['frontier_sha256'] == sha(FRONTIER)
    assert raw['provenance']['policy_sha256'] == sha(POLICY)
    keys = {(r['image'], r['qp']) for r in raw['rows']}
    assert len(keys) == 120 and len({image for image, _ in keys}) == 24
    assert all({qp for image, qp in keys if image == im} == set(QPS)
               for im in {image for image, _ in keys})
    rows = []
    for r in raw['rows']:
        assert r['selected_beta'] == policy['beta'][str(r['qp'])]
        quality_gain = (r['old_active_no_repair_delta444_db']-
                        r['new_delta444_db'])
        saving_gain = (r['new_conv_mac_saving_pct']-
                       r['old_active_no_repair_conv_mac_saving_pct'])
        new_exact = exact_active_no_repair_saving_pct(r['exit_map'])
        old_exact = exact_active_no_repair_saving_pct(r['old_exit_map'])
        rows.append({**r, 'new_vs_old_quality_gain_db': quality_gain,
                     'new_vs_old_conv_mac_saving_points': saving_gain,
                     'new_exact_conv_mac_saving_pct':new_exact,
                     'old_exact_conv_mac_saving_pct':old_exact,
                     'new_vs_old_exact_conv_mac_saving_points':new_exact-old_exact})
    rng = np.random.default_rng(20261007)
    fields = ('new_delta444_db', 'old_active_no_repair_delta444_db',
              'new_conv_mac_saving_pct',
              'old_active_no_repair_conv_mac_saving_pct',
              'new_exact_conv_mac_saving_pct','old_exact_conv_mac_saving_pct',
              'new_vs_old_exact_conv_mac_saving_points',
              'new_vs_old_quality_gain_db',
              'new_vs_old_conv_mac_saving_points')
    all_stats = {field: summarize(rows, field, rng) for field in fields}
    per_qp = {str(qp): {field: summarize([r for r in rows if r['qp'] == qp],
                                         field, rng) for field in fields}
              for qp in QPS}
    thresholds = {
        'new_over_0p1': sum(r['new_delta444_db'] > .1 for r in rows),
        'old_over_0p1': sum(r['old_active_no_repair_delta444_db'] > .1
                           for r in rows),
        'rescued': sum(r['old_active_no_repair_delta444_db'] > .1 and
                       r['new_delta444_db'] <= .1 for r in rows),
        'newly_failed': sum(r['old_active_no_repair_delta444_db'] <= .1 and
                            r['new_delta444_db'] > .1 for r in rows),
    }
    per_qp_target = {str(qp): per_qp[str(qp)]['new_delta444_db']['mean'] <= .1
                     for qp in QPS}
    output = {
        'scope': 'Exploratory disjoint DIV2K validation24 x 5QP of calibration-locked beta with active-replicate/no-repair; same streams and e15, untimed CPU FP32. Analytical synthesis-conv MAC only.',
        'source_sha256': {'raw': sha(RAW), 'policy': sha(POLICY),
                          'frontier': sha(FRONTIER), 'script': sha(Path(__file__)),
                          'exact_conv_formula':sha(HERE/'exact_active_conv_mac.py')},
        'bootstrap': '10000 image-cluster draws, retaining five QPs per sampled image',
        'selected_beta': policy['beta'], 'all': all_stats, 'per_qp': per_qp,
        'validation_mean_delta444_within_0p1_by_qp': per_qp_target,
        'thresholds': thresholds, 'rows': rows,
    }
    OUT.write_text(json.dumps(output, indent=2)+'\n')
    print(json.dumps({'selected_beta': output['selected_beta'],
                      'mean_loss': all_stats['new_delta444_db']['mean'],
                      'mean_legacy_saving_pct': all_stats['new_conv_mac_saving_pct']['mean'],
                      'mean_exact_saving_pct': all_stats['new_exact_conv_mac_saving_pct']['mean'],
                      'saving_change_points': all_stats['new_vs_old_conv_mac_saving_points']['mean'],
                      'quality_change_db': all_stats['new_vs_old_quality_gain_db']['mean'],
                      'per_qp_target': per_qp_target, 'thresholds': thresholds}, indent=2))


if __name__ == '__main__':
    main()
