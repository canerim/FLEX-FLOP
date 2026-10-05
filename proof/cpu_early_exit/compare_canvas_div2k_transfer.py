"""Compare completed, frozen DIV2K active-neighbour fallback replays.

This is a paired untimed reconstruction analysis. It deliberately refuses
partial runs and cannot estimate rate changes from the available Delta444
records. The 24 images, rather than their 120 correlated QP cases, are the
bootstrap units.
"""
from __future__ import annotations

import hashlib
import json
from pathlib import Path

import numpy as np


HERE = Path(__file__).resolve().parent
ROOT = HERE / 'results/div2k_beta/quality_floor'
EXACT = ROOT / 'exact_context_validation24.json'
ARMS = {
    'zero': ROOT / 'active_canvas_validation24_zero.json',
    'replicate': ROOT / 'active_canvas_validation24_replicate.json',
}
OUT = ROOT / 'active_canvas_validation24_transfer_comparison.json'
QPS = (0, 16, 32, 48, 63)


def sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def describe(rows: list[dict], key: str, rng: np.random.Generator) -> dict:
    images = sorted({row['image'] for row in rows})
    values = np.asarray([row[key] for row in rows], dtype=float)
    clusters = np.asarray([[row[key] for row in rows if row['image'] == image]
                           for image in images], dtype=float)
    assert clusters.shape == (len(images), len(rows) // len(images))
    assert clusters.shape[1] in (1, 5)
    draws = rng.integers(0, len(images), size=(10000, len(images)))
    means = clusters[draws].mean(axis=(1, 2))
    return {
        'cases': len(rows), 'images': len(images),
        'mean_db': float(values.mean()),
        'image_cluster_ci95_db': np.quantile(means, [.025, .975]).tolist(),
        'better': int((values > 1e-7).sum()),
        'worse': int((values < -1e-7).sum()),
        'nearly_equal': int((abs(values) <= 1e-7).sum()),
        'min_db': float(values.min()), 'max_db': float(values.max()),
    }


def main() -> None:
    exact = json.loads(EXACT.read_text())
    arms = {name: json.loads(path.read_text()) for name, path in ARMS.items()}
    assert len(exact['rows']) == 120
    exact_by_key = {(r['image'], r['qp']): r for r in exact['rows']}
    assert len(exact_by_key) == 120
    by_arm = {}
    for name, data in arms.items():
        assert data['fallback'] == name
        assert data['complete'] and data['expected_cases'] == len(data['rows']) == 120, name
        assert data['provenance']['exact_sha256'] == sha(EXACT)
        by_arm[name] = {(r['image'], r['qp']): r for r in data['rows']}
        assert set(by_arm[name]) == set(exact_by_key)
    images = {image for image, _ in exact_by_key}
    assert len(images) == 24
    assert all({qp for image, qp in exact_by_key if image == im} == set(QPS)
               for im in images)

    paired = []
    for image, qp in sorted(exact_by_key):
        reference = exact_by_key[image, qp]
        zero = by_arm['zero'][image, qp]
        replicate = by_arm['replicate'][image, qp]
        for case in (zero, replicate):
            assert case['exit_map'] == reference['exit_map']
            assert case['stream_sha256'] == reference['stream_sha256']
            assert case['case_sha256'] == reference['case_sha256']
            assert abs(case['deployed_delta444_db'] -
                       reference['deployed_delta444_db']) < 1e-12
            assert abs(case['exact_context_delta444_db'] -
                       reference['exact_context_delta444_db']) < 1e-12
        row = {'image': image, 'qp': qp,
               'stream_sha256': reference['stream_sha256'],
               'exit_map': reference['exit_map'],
               'deployed_delta444_db': reference['deployed_delta444_db'],
               'exact_context_delta444_db': reference['exact_context_delta444_db']}
        for repair in ('no_repair', 'with_repair'):
            for name, case in (('zero', zero), ('replicate', replicate)):
                delta = case['coupled'][repair]['delta444_db']
                row[f'{name}_{repair}_delta444_db'] = delta
                row[f'{name}_{repair}_gain_db'] = row['deployed_delta444_db'] - delta
            row[f'replicate_minus_zero_{repair}_db'] = (
                row[f'replicate_{repair}_gain_db'] -
                row[f'zero_{repair}_gain_db'])
        paired.append(row)

    rng = np.random.default_rng(20261006)
    keys = [f'{name}_{repair}_gain_db' for name in ARMS
            for repair in ('no_repair', 'with_repair')]
    keys += [f'replicate_minus_zero_{repair}_db'
             for repair in ('no_repair', 'with_repair')]
    summary = {key: describe(paired, key, rng) for key in keys}
    per_qp = {str(qp): {key: describe([r for r in paired if r['qp'] == qp],
                                       key, rng) for key in keys}
              for qp in QPS}
    threshold = {'deployed': sum(r['deployed_delta444_db'] > .1 for r in paired),
                 'exact_context': sum(r['exact_context_delta444_db'] > .1 for r in paired)}
    transitions = {}
    for name in ARMS:
        for repair in ('no_repair', 'with_repair'):
            field = f'{name}_{repair}_delta444_db'
            threshold[f'{name}_{repair}'] = sum(r[field] > .1 for r in paired)
            transitions[f'{name}_{repair}'] = {
                'rescued': sum(r['deployed_delta444_db'] > .1 and r[field] <= .1
                               for r in paired),
                'newly_failed': sum(r['deployed_delta444_db'] <= .1 and r[field] > .1
                                    for r in paired),
            }
    result = {
        'scope': 'DIV2K validation24 x QP5 frozen FUFREF2 reconstruction; same streams, checkpoint, routes and seam repair. Untimed CPU FP32; no rate or latency claim.',
        'input_sha256': {'exact': sha(EXACT),
                         **{name: sha(path) for name, path in ARMS.items()}},
        'bootstrap': '10000 image-cluster draws, preserving five QPs per image',
        'summary': summary, 'per_qp': per_qp,
        'over_0p1_delta444': threshold,
        'threshold_transitions_vs_deployed': transitions,
        'rows': paired,
    }
    OUT.write_text(json.dumps(result, indent=2) + '\n')
    print(json.dumps({'summary': summary, 'over_0p1_delta444': threshold,
                      'threshold_transitions_vs_deployed': transitions}, indent=2))


if __name__ == '__main__':
    main()
