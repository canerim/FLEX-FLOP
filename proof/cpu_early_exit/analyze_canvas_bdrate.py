"""Per-image YUV 6:1:1 BD-rate for frozen Kodak canvas-coupling arms.

All arms decode exactly the same FUFREF2 streams, so the five QP rates are
identical within each image. BD-rate is a quality summary at matched rate,
not evidence of a changed bitstream or measured execution time.
"""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import bjontegaard
import numpy as np
from scipy.interpolate import PchipInterpolator

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
    library=float(bjontegaard.bd_rate(anchor_rate,anchor_quality,
                                      test_rate,test_quality,method='pchip'))
    # Independent integral of log(rate) over the common PSNR support.
    lo=max(min(anchor_quality),min(test_quality))
    hi=min(max(anchor_quality),max(test_quality))
    assert hi>lo
    anchor=PchipInterpolator(anchor_quality,np.log(anchor_rate))
    test=PchipInterpolator(test_quality,np.log(test_rate))
    direct=100*(np.exp((test.integrate(lo,hi)-anchor.integrate(lo,hi))/(hi-lo))-1)
    assert abs(library-direct)<1e-10
    return library


def describe(rows:list[dict],key:str,rng:np.random.Generator)->dict:
    x=np.asarray([r[key] for r in rows])
    draws=rng.integers(0,len(x),size=(10000,len(x)))
    return {'n_images':len(x),'mean_pct':float(x.mean()),
            'median_pct':float(np.median(x)),
            'image_bootstrap_ci95_pct':np.quantile(x[draws].mean(axis=1),[.025,.975]).tolist(),
            'minimum_pct':float(x.min()),'maximum_pct':float(x.max())}


def main()->None:
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--arms',nargs='+',choices=tuple(ARMS),
                   default=list(ARMS),help='Completed arms to include; stale is the common anchor')
    p.add_argument('--output',type=Path,default=OUT)
    args=p.parse_args()
    selected=dict.fromkeys(args.arms)
    assert 'stale' in selected and len(selected)>=2
    raw={name:json.loads(ARMS[name].read_text()) for name in selected}
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
            scan=json.loads(path.read_text())
            stream=path.with_suffix('.fufref2')
            assert stream.stat().st_size==scan['stream_bytes']
            assert sha(stream)==scan['stream_sha256']
            scans.append(scan)
        assert [r['qp'] for r in scans]==list(QPS)
        assert all(lookup[arm][image,qp]['stream_sha256']==s['stream_sha256']
                   for arm in selected for qp,s in zip(QPS,scans))
        h,w=scans[0]['shape']
        rates=[8*s['stream_bytes']/(h*w) for s in scans]
        released=[s['released_yuv611_db'] for s in scans]
        deployed=[lookup['stale'][image,qp]['deployed_yuv611_db'] for qp in QPS]
        exact=[lookup['stale'][image,qp]['exact_context_yuv611_db'] for qp in QPS]
        curves={'released':released,'deployed':deployed,'exact_context':exact}
        for arm in selected:
            for repair in ('no_repair','with_repair'):
                curves[f'{arm}_{repair}']=[lookup[arm][image,qp]['coupled'][repair]['yuv611_db']
                                               for qp in QPS]
        row={'image':image,'qps':list(QPS),'rate_bpp':rates,'quality_yuv611_db':curves}
        for arm,quality in curves.items():
            if arm=='released':continue
            row[f'{arm}_bdrate_vs_released_pct']=bd(rates,released,rates,quality)
            if arm!='deployed':
                row[f'{arm}_bdrate_vs_deployed_pct']=bd(rates,deployed,rates,quality)
        if 'active_zero' in selected and 'active_replicate' in selected:
            for repair in ('no_repair','with_repair'):
                zero=curves[f'active_zero_{repair}']
                replicate=curves[f'active_replicate_{repair}']
                row[f'active_replicate_{repair}_bdrate_vs_active_zero_pct']=(
                    bd(rates,zero,rates,replicate))
        rows.append(row)
    rng=np.random.default_rng(20261005)
    keys=sorted(k for k in rows[0] if k.endswith('_pct'))
    result={'scope':'Kodak24 five-QP YUV 6:1:1 PCHIP BD-rate from frozen research FUFREF2 streams; same rate per arm, no latency or native CUDA stream.',
            'metric':'Positive BD-rate means more bits would be required at equal quality than the named anchor.',
            'method':'bjontegaard Python package, method=pchip, independently cross-checked by log-rate PCHIP integration over common PSNR support for each image; then image-bootstrap 24 paired values.',
            'script_sha256':sha(Path(__file__)),
            'input_sha256':{'scan_cases':scan_hashes,
                            **{name:sha(ARMS[name]) for name in selected}},
            'selected_arms':list(selected),
            'summary':{k:describe(rows,k,rng) for k in keys},
            'per_image':rows}
    args.output.write_text(json.dumps(result,indent=2)+'\n')
    print(json.dumps({k:v['mean_pct'] for k,v in result['summary'].items()},indent=2))


if __name__=='__main__':main()
