"""Per-image YUV 6:1:1 BD-rate for frozen Kodak canvas-coupling arms.

All arms decode exactly the same FUFREF2 streams, so the five QP rates are
identical within each image. BD-rate is a quality summary at matched rate,
not evidence of a changed bitstream or measured execution time.
"""
from __future__ import annotations

import hashlib
import json
from pathlib import Path

import bjontegaard
import numpy as np

HERE=Path(__file__).resolve().parent
BASE=HERE/'results'
SCAN=BASE/'kodak24_qp5'
ARMS={
    'stale':BASE/'kodak_canvas_coupling_20261005.json',
    'active_zero':BASE/'kodak_active_canvas_20261005.json',
    'active_replicate':BASE/'kodak_active_canvas_replicate_20261005.json',
}
OUT=BASE/'kodak_canvas_bdrate_20261005.json'
QPS=(0,16,32,48,63)


def sha(path:Path)->str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def bd(anchor_rate:list[float],anchor_quality:list[float],
       test_rate:list[float],test_quality:list[float])->float:
    assert len(anchor_rate)==len(anchor_quality)==len(test_rate)==len(test_quality)==5
    assert np.all(np.diff(anchor_rate)>0) and np.all(np.diff(test_rate)>0)
    assert np.all(np.diff(anchor_quality)>0) and np.all(np.diff(test_quality)>0)
    return float(bjontegaard.bd_rate(anchor_rate,anchor_quality,
                                     test_rate,test_quality,method='pchip'))


def describe(rows:list[dict],key:str,rng:np.random.Generator)->dict:
    x=np.asarray([r[key] for r in rows])
    draws=rng.integers(0,len(x),size=(10000,len(x)))
    return {'n_images':len(x),'mean_pct':float(x.mean()),
            'median_pct':float(np.median(x)),
            'image_bootstrap_ci95_pct':np.quantile(x[draws].mean(axis=1),[.025,.975]).tolist(),
            'minimum_pct':float(x.min()),'maximum_pct':float(x.max())}


def main()->None:
    raw={name:json.loads(path.read_text()) for name,path in ARMS.items()}
    for data in raw.values():
        assert data['complete'] and len(data['rows'])==120
    lookup={name:{(r['image'],r['qp']):r for r in data['rows']}
            for name,data in raw.items()}
    images=sorted({im for im,_ in lookup['stale']})
    assert len(images)==24
    rows=[]
    scan_hashes={}
    for image in images:
        scans=[]
        for qp in QPS:
            path=SCAN/f'{Path(image).stem}_qp{qp}.json'
            scan_hashes[path.name]=sha(path)
            scans.append(json.loads(path.read_text()))
        assert [r['qp'] for r in scans]==list(QPS)
        assert all(lookup[arm][image,qp]['stream_sha256']==s['stream_sha256']
                   for arm in ARMS for qp,s in zip(QPS,scans))
        h,w=scans[0]['shape']
        rates=[8*s['stream_bytes']/(h*w) for s in scans]
        released=[s['released_yuv611_db'] for s in scans]
        deployed=[lookup['stale'][image,qp]['deployed_yuv611_db'] for qp in QPS]
        exact=[lookup['stale'][image,qp]['exact_context_yuv611_db'] for qp in QPS]
        curves={'released':released,'deployed':deployed,'exact_context':exact}
        for arm in ARMS:
            for repair in ('no_repair','with_repair'):
                curves[f'{arm}_{repair}']=[lookup[arm][image,qp]['coupled'][repair]['yuv611_db']
                                               for qp in QPS]
        row={'image':image,'qps':list(QPS),'rate_bpp':rates,'quality_yuv611_db':curves}
        for arm,quality in curves.items():
            if arm=='released':continue
            row[f'{arm}_bdrate_vs_released_pct']=bd(rates,released,rates,quality)
            if arm!='deployed':
                row[f'{arm}_bdrate_vs_deployed_pct']=bd(rates,deployed,rates,quality)
        rows.append(row)
    rng=np.random.default_rng(20261005)
    keys=sorted(k for k in rows[0] if k.endswith('_pct'))
    result={'scope':'Kodak24 five-QP YUV 6:1:1 PCHIP BD-rate from frozen research FUFREF2 streams; same rate per arm, no latency or native CUDA stream.',
            'metric':'Positive BD-rate means more bits would be required at equal quality than the named anchor.',
            'method':'bjontegaard Python package, method=pchip; integrate over common PSNR range for each image, then image-bootstrap 24 paired values.',
            'input_sha256':{'scan_cases':scan_hashes,
                            **{name:sha(path) for name,path in ARMS.items()}},
            'summary':{k:describe(rows,k,rng) for k in keys},
            'per_image':rows}
    OUT.write_text(json.dumps(result,indent=2)+'\n')
    print(json.dumps({k:v['mean_pct'] for k,v in result['summary'].items()},indent=2))


if __name__=='__main__':main()
