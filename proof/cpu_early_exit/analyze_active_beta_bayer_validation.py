"""Paired image-cluster analysis of active-beta fixed-histogram placement."""
from __future__ import annotations

import hashlib
import json
from pathlib import Path

import numpy as np

HERE=Path(__file__).resolve().parent
RAW=HERE/'results/div2k_beta/quality_floor/active_beta_bayer_validation24.json'
OUT=HERE/'results/div2k_beta/quality_floor/active_beta_bayer_validation24_analysis.json'
QPS=(0,16,32,48,63)


def sha(path:Path)->str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def describe(x:np.ndarray,draw:np.ndarray)->dict:
    assert x.ndim==2 and x.shape[0]==24
    boot=x[draw].mean(axis=(1,2))
    return {'n_images':24,'n_cases':int(x.size),'mean_db':float(x.mean()),
            'median_db':float(np.median(x)),
            'image_cluster_ci95_db':np.quantile(boot,[.025,.975]).tolist(),
            'router_better_cases':int((x>1e-9).sum()),
            'bayer_better_cases':int((x< -1e-9).sum()),
            'tied_cases':int((abs(x)<=1e-9).sum())}


def main()->None:
    raw=json.loads(RAW.read_text())
    assert raw['complete'] and len(raw['rows'])==raw['expected_cases']==120
    rows={(r['image'],r['qp']):r for r in raw['rows']}
    assert len(rows)==120
    images=sorted({im for im,_ in rows})
    assert len(images)==24
    assert all({qp for im,qp in rows if im==image}==set(QPS) for image in images)
    for r in rows.values():
        assert r['exit_counts']==[r['router_map'].count(k) for k in range(6)]
        assert r['exit_counts']==[r['bayer_map'].count(k) for k in range(6)]
        assert r['same_map']==(r['router_map']==r['bayer_map'])
        assert abs(r['placement_gain_db']-(r['bayer_delta444_db']-r['router_delta444_db']))<1e-9
    x=np.asarray([[rows[image,qp]['placement_gain_db'] for qp in QPS]
                  for image in images],dtype=float)
    assert np.isfinite(x).all()
    rng=np.random.default_rng(20261005)
    draw=rng.integers(0,24,size=(10000,24))
    output={
        'scope':'DIV2K validation24 x QP5 active-replicate/no-repair, frozen calibration-only beta and maps; fixed Bayer-rank placement has identical exit histogram and analytical synthesis-conv MAC. Positive gain favours router.',
        'limitations':'A matched-histogram position control, not the independently calibrated scalar dithering policy; development-used data, untimed FUFREF2 research streams, fixed checkpoint. No native bitrate or full-codec speedup inference.',
        'source_sha256':{'raw':sha(RAW),'script':sha(Path(__file__))},
        'bootstrap':'10000 draws of 24 images with replacement, retaining all five QPs per image.',
        'overall':describe(x,draw),
        'per_qp':{str(qp):describe(x[:,j:j+1],draw) for j,qp in enumerate(QPS)},
        'per_case':[{'image':im,'qp':qp,'placement_gain_db':rows[im,qp]['placement_gain_db'],
                     'same_map':rows[im,qp]['same_map']}
                    for im in images for qp in QPS],
    }
    OUT.write_text(json.dumps(output,indent=2)+'\n')
    print(json.dumps({'overall':output['overall'],'per_qp':output['per_qp']},indent=2))


if __name__=='__main__':main()
