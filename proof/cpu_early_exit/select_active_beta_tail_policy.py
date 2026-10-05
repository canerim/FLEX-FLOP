"""Freeze a tail-constrained active-context beta using calibration only."""
from __future__ import annotations

import hashlib
import json
from pathlib import Path

HERE = Path(__file__).resolve().parent
BASE = HERE / 'results/div2k_beta'
FRONTIER = BASE / 'quality_floor/active_replicate_beta_calibration24.json'
MANIFEST = BASE / 'manifest.json'
PRIMARY = BASE / 'quality_floor/active_replicate_beta_locked_policy.json'
OUT = BASE / 'quality_floor/active_replicate_tail_locked_policy.json'
QPS = (0, 16, 32, 48, 63)
MEAN_TARGET_DB = .1
MAX_VIOLATIONS_PER_QP = 2


def sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main() -> None:
    frontier = json.loads(FRONTIER.read_text())
    manifest = json.loads(MANIFEST.read_text())
    primary = json.loads(PRIMARY.read_text())
    assert frontier['complete'] and len(frontier['rows']) == 120
    assert frontier['provenance']['manifest_sha256'] == sha(MANIFEST)
    assert primary['source_sha256']['frontier'] == sha(FRONTIER)
    assert manifest['target_delta444_db_mean'] == MEAN_TARGET_DB
    by_key = {(r['image'], r['qp']): r for r in frontier['rows']}
    assert len(by_key) == 120
    images = sorted({image for image, _ in by_key})
    assert len(images) == 24
    assert all({qp for image, qp in by_key if image == im} == set(QPS)
               for im in images)
    tables = {}
    selected = {}
    for qp in QPS:
        cases = [by_key[image, qp] for image in images]
        betas = manifest['candidates'][str(qp)]
        assert all([c['beta'] for c in case['candidates']] == betas
                   for case in cases)
        table = []
        for i, beta in enumerate(betas):
            maps = [case['maps'][case['candidates'][i]['map_id']]
                    for case in cases]
            losses = [m['active_delta444_db'] for m in maps]
            savings = [m['active_no_repair_conv_saving_pct'] for m in maps]
            violations = sum(loss > MEAN_TARGET_DB for loss in losses)
            mean_loss = sum(losses)/24
            table.append({
                'beta': beta, 'mean_delta444_db': mean_loss,
                'over_0p1_count': violations,
                'mean_conv_mac_saving_pct': sum(savings)/24,
                'eligible': (mean_loss <= MEAN_TARGET_DB and
                             violations <= MAX_VIOLATIONS_PER_QP),
            })
        eligible = [row for row in table if row['eligible']]
        assert eligible, f'No tail-feasible beta at QP{qp}'
        winner = max(eligible, key=lambda row: (
            row['mean_conv_mac_saving_pct'], -row['mean_delta444_db'],
            -row['over_0p1_count'], -row['beta']))
        tables[str(qp)] = table
        selected[str(qp)] = winner
    output = {
        'scope': 'Exploratory active-replicate/no-repair beta selected only on DIV2K calibration24 to satisfy mean Delta444<=0.1 dB and at most two individual losses>0.1 dB per QP. Validation never read.',
        'source_sha256': {'frontier': sha(FRONTIER),
                          'manifest': sha(MANIFEST),
                          'primary_policy': sha(PRIMARY),
                          'selector': sha(Path(__file__))},
        'mean_target_delta444_db': MEAN_TARGET_DB,
        'max_calibration_violations_per_qp': MAX_VIOLATIONS_PER_QP,
        'selection_rule': 'Per QP, maximize calibrated analytical synthesis-conv MAC saving among beta candidates with mean Delta444<=0.1 dB and at most 2/24 individual losses>0.1 dB; ties minimize mean loss, violations, then beta.',
        'beta': {qp: row['beta'] for qp,row in selected.items()},
        'selected': selected,
        'primary_beta': primary['beta'],
        'calibration_tables': tables,
        'validation_status': 'not evaluated by this selector',
    }
    OUT.write_text(json.dumps(output, indent=2)+'\n')
    print(json.dumps({'beta': output['beta'], 'selected': selected}, indent=2))


if __name__ == '__main__':
    main()
