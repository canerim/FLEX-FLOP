"""Decompose active-context gain into context and old-repair components.

All 120 DIV2K validation cases use identical FUFREF2 streams, e15 weights,
maps and full-frame e15 references. Both the isolated repair-off arm and the
two active-neighbour arms must be complete before analysis. This is an
exploratory post-result intervention, untimed CPU FP32.
"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np

from compare_canvas_div2k_transfer import describe, sha, QPS


HERE = Path(__file__).resolve().parent
ROOT = HERE / 'results/div2k_beta/quality_floor'
FILES = {
    'exact': ROOT / 'exact_context_validation24.json',
    'isolated': ROOT / 'isolated_no_repair_validation24.json',
    'zero': ROOT / 'active_canvas_validation24_zero.json',
    'replicate': ROOT / 'active_canvas_validation24_replicate.json',
}
OUT = ROOT / 'isolated_no_repair_decomposition.json'


def main() -> None:
    inputs = {name: json.loads(path.read_text()) for name, path in FILES.items()}
    exact = inputs['exact']
    assert len(exact['rows']) == 120 and len(exact['images']) == 24
    by_arm = {}
    for name, data in inputs.items():
        if name != 'exact':
            assert data['complete'] and len(data['rows']) == data['expected_cases'] == 120
            assert data['provenance']['exact_sha256'] == sha(FILES['exact'])
        by_arm[name] = {(r['image'], r['qp']): r for r in data['rows']}
    keys = set(by_arm['exact'])
    assert len(keys) == 120 and all(set(rows) == keys for rows in by_arm.values())
    assert len({image for image, _ in keys}) == 24
    assert all({qp for image, qp in keys if image == im} == set(QPS)
               for im in {image for image, _ in keys})
    for key in ('manifest_sha256', 'released_sha256', 'e15_sha256'):
        assert len({inputs[name]['provenance'][key] for name in ('isolated','zero','replicate')}) == 1

    rows = []
    for image, qp in sorted(keys):
        reference = by_arm['exact'][image, qp]
        isolated = by_arm['isolated'][image, qp]
        zero = by_arm['zero'][image, qp]
        replicate = by_arm['replicate'][image, qp]
        for case in (isolated, zero, replicate):
            assert case['exit_map'] == reference['exit_map']
            assert case['stream_sha256'] == reference['stream_sha256']
            assert case['case_sha256'] == reference['case_sha256']
        deployed = reference['deployed_delta444_db']
        assert abs(isolated['isolated_with_repair_delta444_db'] - deployed) < 1e-6
        assert abs(zero['deployed_delta444_db'] - deployed) < 1e-12
        assert abs(replicate['deployed_delta444_db'] - deployed) < 1e-12
        assert abs(isolated['exact_context_delta444_db'] - reference['exact_context_delta444_db']) < 1e-12
        row = {'image': image, 'qp': qp, 'stream_sha256': reference['stream_sha256'],
               'exit_map': reference['exit_map'],
               'deployed_with_repair_delta444_db': deployed,
               'isolated_no_repair_delta444_db': isolated['isolated_no_repair_delta444_db'],
               'active_zero_with_repair_delta444_db': zero['coupled']['with_repair']['delta444_db'],
               'active_zero_no_repair_delta444_db': zero['coupled']['no_repair']['delta444_db'],
               'active_replicate_with_repair_delta444_db': replicate['coupled']['with_repair']['delta444_db'],
               'active_replicate_no_repair_delta444_db': replicate['coupled']['no_repair']['delta444_db'],
               'exact_context_delta444_db': reference['exact_context_delta444_db']}
        row['isolated_repair_removal_gain_db'] = deployed-row['isolated_no_repair_delta444_db']
        for arm in ('zero', 'replicate'):
            active_on = row[f'active_{arm}_with_repair_delta444_db']
            active_off = row[f'active_{arm}_no_repair_delta444_db']
            row[f'{arm}_context_gain_repair_on_db'] = deployed-active_on
            row[f'{arm}_context_gain_repair_off_db'] = row['isolated_no_repair_delta444_db']-active_off
            row[f'{arm}_repair_removal_gain_in_active_db'] = active_on-active_off
            row[f'{arm}_total_gain_vs_deployed_db'] = deployed-active_off
            row[f'{arm}_context_repair_interaction_db'] = (
                row[f'{arm}_context_gain_repair_off_db']-
                row[f'{arm}_context_gain_repair_on_db'])
            assert abs(row[f'{arm}_total_gain_vs_deployed_db']-
                       row['isolated_repair_removal_gain_db']-
                       row[f'{arm}_context_gain_repair_off_db']) < 1e-12
        rows.append(row)

    rng = np.random.default_rng(20261006)
    fields = ['isolated_repair_removal_gain_db']
    for arm in ('zero','replicate'):
        fields.extend((f'{arm}_context_gain_repair_on_db',
                       f'{arm}_context_gain_repair_off_db',
                       f'{arm}_repair_removal_gain_in_active_db',
                       f'{arm}_total_gain_vs_deployed_db',
                       f'{arm}_context_repair_interaction_db'))
    summary = {field: describe(rows, field, rng) for field in fields}
    by_qp = {str(qp): {field: describe([r for r in rows if r['qp'] == qp],
                                        field, rng) for field in fields}
             for qp in QPS}
    threshold = {arm: sum(r[f'{arm}_delta444_db'] > .1 for r in rows)
                 for arm in ('deployed_with_repair','isolated_no_repair',
                             'active_zero_with_repair','active_zero_no_repair',
                             'active_replicate_with_repair','active_replicate_no_repair',
                             'exact_context')}
    result = {
        'scope': 'Exploratory same-stream DIV2K validation24 x QP5 factorial context/repair decomposition; CPU FP32, no latency or rate claim.',
        'source_sha256': {name: sha(path) for name, path in FILES.items()},
        'bootstrap': '10000 image-cluster draws, retaining all five QPs per sampled image',
        'summary': summary, 'by_qp': by_qp,
        'over_0p1_delta444': threshold, 'rows': rows,
    }
    OUT.write_text(json.dumps(result, indent=2) + '\n')
    print(json.dumps({'summary': summary,
                      'over_0p1_delta444': threshold}, indent=2))


if __name__ == '__main__':
    main()
