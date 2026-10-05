"""Freeze a small quality-budget sweep from calibration only.

Targets are predeclared in the source. The selector never reads validation
or Kodak outcomes. A target with no feasible candidate at any QP is marked
infeasible rather than silently relaxing its constraint.
"""
from __future__ import annotations

import hashlib
import json
from pathlib import Path

HERE = Path(__file__).resolve().parent
BASE = HERE / 'results/div2k_beta'
FRONTIER = BASE / 'quality_floor/active_replicate_beta_calibration24.json'
MANIFEST = BASE / 'manifest.json'
PRIMARY = BASE / 'quality_floor/active_replicate_beta_locked_policy.json'
OUT = BASE / 'quality_floor/active_replicate_budget_sweep_policy.json'
QPS = (0, 16, 32, 48, 63)
TARGETS = (0.075, 0.1, 0.125, 0.15)


def sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main() -> None:
    frontier = json.loads(FRONTIER.read_text())
    manifest = json.loads(MANIFEST.read_text())
    primary = json.loads(PRIMARY.read_text())
    assert frontier['complete'] and len(frontier['rows']) == 120
    assert frontier['provenance']['manifest_sha256'] == sha(MANIFEST)
    assert primary['source_sha256']['frontier'] == sha(FRONTIER)
    by_key = {(r['image'], r['qp']): r for r in frontier['rows']}
    assert len(by_key) == 120
    images = sorted({image for image, _ in by_key})
    assert len(images) == 24
    assert all({qp for image, qp in by_key if image == im} == set(QPS)
               for im in images)
    tables = {}
    for qp in QPS:
        cases = [by_key[image, qp] for image in images]
        betas = manifest['candidates'][str(qp)]
        assert all([c['beta'] for c in case['candidates']] == betas
                   for case in cases)
        entries = []
        for i, beta in enumerate(betas):
            maps = [case['maps'][case['candidates'][i]['map_id']]
                    for case in cases]
            entries.append({
                'beta': beta,
                'mean_delta444_db': sum(m['active_delta444_db'] for m in maps)/24,
                'mean_conv_mac_saving_pct': sum(
                    m['active_no_repair_conv_saving_pct'] for m in maps)/24,
            })
        tables[str(qp)] = entries
    policies = {}
    for target in TARGETS:
        selected = {}
        for qp in QPS:
            feasible = [row for row in tables[str(qp)]
                        if row['mean_delta444_db'] <= target]
            if not feasible:
                selected[str(qp)] = None
                continue
            selected[str(qp)] = max(feasible, key=lambda row: (
                row['mean_conv_mac_saving_pct'], -row['mean_delta444_db'],
                -row['beta']))
        policies[str(target)] = {
            'target_delta444_db_mean_per_qp': target,
            'complete_five_qp_policy': all(row is not None for row in selected.values()),
            'selected': selected,
            'beta': {qp: row['beta'] if row is not None else None
                     for qp, row in selected.items()},
        }
    assert policies['0.1']['complete_five_qp_policy']
    assert policies['0.1']['beta'] == primary['beta']
    output = {
        'scope': 'Exploratory 0.075/0.1/0.125/0.15 dB per-QP calibration mean Delta444 budgets, active-replicate/no-repair, frozen original FUFREF2 streams and e15. Validation never read.',
        'source_sha256': {'frontier': sha(FRONTIER), 'manifest': sha(MANIFEST),
                          'primary_policy': sha(PRIMARY),
                          'selector': sha(Path(__file__))},
        'selection_rule': 'For each QP and target, maximize calibrated analytical synthesis-conv MAC saving among beta candidates with mean Delta444 no greater than target; ties minimize loss then beta. If any QP has no feasible candidate, label whole policy infeasible.',
        'targets': list(TARGETS), 'qps': list(QPS), 'policies': policies,
        'calibration_tables': tables,
        'validation_status': 'not evaluated by this selector',
    }
    OUT.write_text(json.dumps(output, indent=2)+'\n')
    print(json.dumps({'policies': policies}, indent=2))


if __name__ == '__main__':
    main()
