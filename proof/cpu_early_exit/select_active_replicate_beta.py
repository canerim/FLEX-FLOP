"""Select frozen QP-specific beta from the complete active-context calibration.

No validation or Kodak file is read. Among archived candidates satisfying
mean calibration Delta444 <= 0.1 dB, maximize analytical synthesis-conv MAC
saving; ties minimize mean loss, then beta. This reproduces the earlier
deployed-policy selection rule on a new, explicitly exploratory decoder arm.
"""
from __future__ import annotations

import hashlib
import json
from pathlib import Path


HERE = Path(__file__).resolve().parent
FRONTIER = HERE / 'results/div2k_beta/quality_floor/active_replicate_beta_calibration24.json'
MANIFEST = HERE / 'results/div2k_beta/manifest.json'
LOCKED = HERE / 'results/div2k_beta/locked_policy.json'
OUT = HERE / 'results/div2k_beta/quality_floor/active_replicate_beta_locked_policy.json'
QPS = (0, 16, 32, 48, 63)
TARGET = .1


def sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main() -> None:
    frontier = json.loads(FRONTIER.read_text())
    manifest = json.loads(MANIFEST.read_text())
    deployed_policy = json.loads(LOCKED.read_text())
    assert frontier['complete'] and len(frontier['rows']) == frontier['expected_cases'] == 120
    assert frontier['provenance']['manifest_sha256'] == sha(MANIFEST)
    assert deployed_policy['manifest_sha256'] == sha(MANIFEST)
    assert manifest['target_delta444_db_mean'] == TARGET
    by_key = {(r['image'], r['qp']): r for r in frontier['rows']}
    assert len(by_key) == 120
    images = sorted({image for image, _ in by_key})
    assert len(images) == 24
    assert all({qp for image, qp in by_key if image == im} == set(QPS) for im in images)

    tables = {}
    selected = {}
    original = {}
    for qp in QPS:
        cases = [by_key[image, qp] for image in images]
        betas = manifest['candidates'][str(qp)]
        assert all([c['beta'] for c in case['candidates']] == betas for case in cases)
        table = []
        for index, beta in enumerate(betas):
            maps = [case['maps'][case['candidates'][index]['map_id']]
                    for case in cases]
            values = [m['active_delta444_db'] for m in maps]
            savings = [m['active_no_repair_conv_saving_pct'] for m in maps]
            table.append({
                'beta': beta, 'mean_delta444_db': sum(values)/24,
                'mean_conv_mac_saving_pct': sum(savings)/24,
                'over_0p1_count': sum(value > TARGET for value in values),
                'eligible': sum(values)/24 <= TARGET,
            })
        eligible = [row for row in table if row['eligible']]
        assert eligible, f'No feasible calibration beta at QP{qp}'
        winner = max(eligible, key=lambda row: (
            row['mean_conv_mac_saving_pct'], -row['mean_delta444_db'], -row['beta']))
        locked_beta = deployed_policy['beta'][str(qp)]
        matched = [row for row in table if row['beta'] == locked_beta]
        assert len(matched) == 1
        tables[str(qp)] = table
        selected[str(qp)] = winner
        original[str(qp)] = matched[0]
    output = {
        'scope': 'Exploratory active-replicate/no-repair QP beta calibrated only on 24 DIV2K calibration images at fixed FUFREF2 streams and e15 weights; no validation outcome used.',
        'source_sha256': {'frontier': sha(FRONTIER), 'manifest': sha(MANIFEST),
                          'old_locked_policy': sha(LOCKED), 'script': sha(Path(__file__))},
        'selection_rule': 'For each QP, among archived candidates with calibration mean Delta444 <=0.1 dB, maximize mean analytical synthesis-conv MAC saving; ties minimize mean Delta444, then beta.',
        'target_delta444_db_mean': TARGET,
        'beta': {qp: selected[qp]['beta'] for qp in selected},
        'selected': selected, 'original_beta_on_active_decoder': original,
        'calibration_tables': tables,
        'validation_status': 'not evaluated by this selector',
    }
    OUT.write_text(json.dumps(output, indent=2) + '\n')
    print(json.dumps({'beta': output['beta'], 'selected': selected,
                      'original_beta_on_active_decoder': original}, indent=2))


if __name__ == '__main__':
    main()
