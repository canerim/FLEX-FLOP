"""Predeclared paired seam-distance summary of frozen context replays."""
from __future__ import annotations

import hashlib
import json
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
RAW = HERE / 'results/div2k_beta/quality_floor/context_seam_locality_validation24.json'
OUT = HERE / 'results/div2k_beta/quality_floor/context_seam_locality_analysis.json'
QPS = (0, 16, 32, 48, 63)


def sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def stats(rows: list[dict], key, rng: np.random.Generator) -> dict:
    images = sorted({r['image'] for r in rows})
    n_qp = len(rows)//len(images)
    assert len(images) == 24 and n_qp in (1, 5)
    matrix = np.asarray([[key(r) for r in rows if r['image'] == image]
                         for image in images], dtype=float)
    assert matrix.shape == (24, n_qp)
    boot = matrix[rng.integers(0, 24, size=(10000, 24))].mean(axis=(1, 2))
    return {'mean': float(matrix.mean()),
            'image_cluster_ci95': np.quantile(boot, [.025, .975]).tolist(),
            'min': float(matrix.min()), 'max': float(matrix.max())}


def main() -> None:
    raw = json.loads(RAW.read_text())
    assert raw['complete'] and len(raw['rows']) == raw['expected_cases'] == 120
    rows = raw['rows']
    keys = {(r['image'], r['qp']) for r in rows}
    assert len(keys) == 120 and len({im for im, _ in keys}) == 24
    assert all({q for im, q in keys if im == image} == set(QPS)
               for image in {im for im, _ in keys})
    assert all(abs(r['full_image_gain_db']-r['archived_full_image_gain_db'])<1e-5
               for r in rows)
    labels = [(b['lower_px'], b['upper_px']) for b in rows[0]['bins']]
    assert labels == [(0,8),(8,16),(16,32),(32,64),(64,None)]
    assert all([(b['lower_px'],b['upper_px']) for b in r['bins']] == labels
               for r in rows)
    for r in rows:
        h,w = r['crop_hw']
        assert sum(b['pixels'] for b in r['bins']) == h*w
        iso = sum(b['isolated_mse444']*b['pixels'] for b in r['bins'])/(h*w)
        active = sum(b['active_mse444']*b['pixels'] for b in r['bins'])/(h*w)
        assert abs(10*np.log10(iso/active)-r['full_image_gain_db'])<1e-5
    rng = np.random.default_rng(20261008)
    net_reduction = [sum((r['bins'][i]['isolated_mse444']-
                          r['bins'][i]['active_mse444'])*r['bins'][i]['pixels']
                         for r in rows) for i in range(len(labels))]
    assert sum(net_reduction)>0
    total_pixels = sum(r['crop_hw'][0]*r['crop_hw'][1] for r in rows)
    aggregate = []
    for index, (lower, upper) in enumerate(labels):
        key = lambda r, i=index: r['bins'][i]['active_gain_db']
        values = stats(rows, key, rng)
        per_qp = {str(qp): stats([r for r in rows if r['qp']==qp], key, rng)
                  for qp in QPS}
        aggregate.append({'lower_px': lower, 'upper_px': upper,
                          'mean_pixels_per_case': sum(r['bins'][index]['pixels'] for r in rows)/120,
                          'pixel_fraction': sum(r['bins'][index]['pixels'] for r in rows)/total_pixels,
                          'share_of_net_mse_reduction': net_reduction[index]/sum(net_reduction),
                          'gain_db': values, 'per_qp_gain_db': per_qp})
    output = {
        'scope': 'Exploratory seam-distance localization at frozen original beta on DIV2K validation24 x QP5; no latency or native rate inference.',
        'raw_sha256': sha(RAW), 'script_sha256': sha(Path(__file__)),
        'bootstrap': '10000 image-cluster draws retaining five QPs per image',
        'full_image_gain_db': stats(rows, lambda r: r['full_image_gain_db'], rng),
        'bins': aggregate,
    }
    OUT.write_text(json.dumps(output, indent=2)+'\n')
    print(json.dumps({'full_image_gain_db': output['full_image_gain_db'],
                      'bins': [{'range':[r['lower_px'],r['upper_px']],
                                'mean_gain_db':r['gain_db']['mean']}
                               for r in aggregate]}, indent=2))


if __name__ == '__main__':
    main()
