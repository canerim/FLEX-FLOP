"""Paired image-cluster summary of the frozen Kodak canvas-coupling replay."""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
DEFAULT_SOURCE = HERE/'results/kodak_canvas_coupling_20261005.json'


def sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def describe(rows: list[dict], rng: np.random.Generator) -> dict:
    images = sorted({r['image'] for r in rows})
    assert all(len([r for r in rows if r['image']==im]) == len(rows)//len(images)
               for im in images)
    result = {'cases': len(rows), 'images': len(images), 'per_variant': {}}
    for key in ('no_repair', 'with_repair'):
        has_yuv=all('yuv611_db' in r['coupled'][key] and 'deployed_yuv611_db' in r
                    for r in rows)
        yuv_gain=(np.asarray([r['coupled'][key]['yuv611_db']-r['deployed_yuv611_db']
                              for r in rows]) if has_yuv else None)
        d444_gain = np.asarray([r['deployed_delta444_db']-r['coupled'][key]['delta444_db']
                                for r in rows])
        case_values = {(r['image'],r['qp']):r['deployed_delta444_db']-r['coupled'][key]['delta444_db']
                       for r in rows}
        qp_list = sorted({r['qp'] for r in rows})
        image_matrix = np.asarray([[case_values[(im,qp)] for qp in qp_list] for im in images])
        draws = rng.integers(0,len(images),size=(10000,len(images)))
        means = image_matrix[draws].mean(axis=(1,2))
        result['per_variant'][key] = {
            'mean_delta444_gain_db': float(d444_gain.mean()),
            'image_cluster_ci95_delta444_db': np.quantile(means,[.025,.975]).tolist(),
            'mean_yuv611_gain_db': float(yuv_gain.mean()) if has_yuv else None,
            'better_cases': int((d444_gain>1e-7).sum()),
            'worse_cases': int((d444_gain<-1e-7).sum()),
            'nearly_equal_cases': int((abs(d444_gain)<=1e-7).sum()),
            'delta444_over_0p1_count': int(sum(r['coupled'][key]['delta444_db']>0.1 for r in rows)),
            'per_qp_delta444_gain_db': {str(qp):float(np.mean([case_values[(im,qp)] for im in images]))
                                        for qp in qp_list},
        }
    deployed=np.asarray([r['deployed_delta444_db'] for r in rows])
    exact=np.asarray([r['exact_context_delta444_db'] for r in rows])
    result['reference']={
        'mean_deployed_delta444_db':float(deployed.mean()),
        'mean_exact_context_delta444_db':float(exact.mean()),
        'deployed_over_0p1_count':int((deployed>0.1).sum()),
        'exact_context_over_0p1_count':int((exact>0.1).sum()),
        'mean_deployed_minus_exact_db':float((deployed-exact).mean()),
    }
    return result


def main() -> None:
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--source',type=Path,default=DEFAULT_SOURCE)
    p.add_argument('--output',type=Path)
    args=p.parse_args()
    output=args.output or args.source.with_name(args.source.stem+'_summary.json')
    data=json.loads(args.source.read_text())
    assert data['complete'] and len(data['rows'])==data['expected_cases']==120
    assert len({(r['image'],r['qp']) for r in data['rows']})==120
    rng=np.random.default_rng(20261005)
    result={'scope':'Frozen Kodak24 x QP5 paired bitstream reconstruction; no retraining or GPU/runtime measurement',
            'source_sha256':sha(args.source), 'bootstrap':'10000 resamples of 24 images retaining all five QPs',
            'all':describe(data['rows'],rng),
            'by_qp':{str(qp):describe([r for r in data['rows'] if r['qp']==qp],rng)
                     for qp in (0,16,32,48,63)},
            'mixed_map_cases':sum(len(set(r['exit_map']))>1 for r in data['rows'])}
    output.write_text(json.dumps(result,indent=2)+'\n')
    print(json.dumps(result['all'],indent=2))


if __name__=='__main__':main()
