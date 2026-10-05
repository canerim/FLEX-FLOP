"""Paired Kodak comparison of stale and stage-synchronous tile context."""
from __future__ import annotations

import hashlib
import json
from pathlib import Path

import numpy as np

HERE=Path(__file__).resolve().parent
BASE=HERE/'results'
FILES={
    'stale':BASE/'kodak_canvas_coupling_20261005.json',
    'active_zero':BASE/'kodak_active_canvas_20261005.json',
    'active_replicate':BASE/'kodak_active_canvas_replicate_20261005.json',
}
SCAN=BASE/'kodak24_qp5'
OUT=BASE/'kodak_canvas_arms_comparison_20261005.json'


def sha(path:Path)->str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def interval(rows:list[dict],key:str,rng:np.random.Generator)->list[float]:
    images=sorted({r['image'] for r in rows})
    per_image=np.asarray([[sum(r[key] for r in rows if r['image']==im),
                           sum(r['image']==im for r in rows)] for im in images],dtype=float)
    draws=rng.integers(0,len(images),size=(10000,len(images)))
    sampled=per_image[draws].sum(axis=1)
    return np.quantile(sampled[:,0]/sampled[:,1],[.025,.975]).tolist()


def describe(rows:list[dict],key:str,rng:np.random.Generator)->dict:
    x=np.asarray([r[key] for r in rows],dtype=float)
    return {'cases':len(rows),'images':len({r['image'] for r in rows}),
            'mean_gain_db':float(x.mean()),'image_cluster_ci95_db':interval(rows,key,rng),
            'better':int((x>1e-7).sum()),'worse':int((x<-1e-7).sum()),
            'nearly_equal':int((abs(x)<=1e-7).sum()),
            'min_gain_db':float(x.min()),'max_gain_db':float(x.max())}


def main()->None:
    arms={name:json.loads(path.read_text()) for name,path in FILES.items()}
    for data in arms.values():
        assert data['complete'] and len(data['rows'])==data['expected_cases']==120
    by_arm={name:{(r['image'],r['qp']):r for r in data['rows']}
            for name,data in arms.items()}
    keys=sorted(by_arm['stale'])
    assert len(keys)==120 and all(set(table)==set(keys) for table in by_arm.values())
    rows=[]
    for image,qp in keys:
        first=by_arm['stale'][(image,qp)]
        scan_path=SCAN/f'{Path(image).stem}_qp{qp}.json'
        scan=json.loads(scan_path.read_text())
        assert first['scan_sha256']==sha(scan_path)
        h,w=scan['shape'];nh,nw=h//256,w//256
        assert nh*nw==6
        edges=[(i,i+1) for i in range(6) if i%nw<nw-1]
        edges.extend((i,i+nw) for i in range(6) if i//nw<nh-1)
        assert len(edges)==7
        mode=first['exit_map']
        row={'image':image,'qp':qp,'exit_map':mode,
             'mixed_edge_count':sum(mode[i]!=mode[j] for i,j in edges),
             'depth_total_variation':sum(abs(mode[i]-mode[j]) for i,j in edges),
             'deployed_delta444_db':first['deployed_delta444_db'],
             'exact_context_delta444_db':first['exact_context_delta444_db']}
        for arm,table in by_arm.items():
            rec=table[(image,qp)]
            assert rec['exit_map']==mode
            assert rec['stream_sha256']==first['stream_sha256']
            assert abs(rec['deployed_delta444_db']-row['deployed_delta444_db'])<1e-12
            assert abs(rec['exact_context_delta444_db']-row['exact_context_delta444_db'])<1e-12
            assert rec['scan_sha256']==first['scan_sha256']
            for repair in ('no_repair','with_repair'):
                row[f'{arm}_{repair}_gain_db']=(row['deployed_delta444_db']-
                                               rec['coupled'][repair]['delta444_db'])
                row[f'{arm}_{repair}_delta444_db']=rec['coupled'][repair]['delta444_db']
        row['replicate_minus_zero_no_repair_db']=(row['active_replicate_no_repair_gain_db']-
                                                  row['active_zero_no_repair_gain_db'])
        row['replicate_minus_zero_with_repair_db']=(row['active_replicate_with_repair_gain_db']-
                                                    row['active_zero_with_repair_gain_db'])
        for repair in ('no_repair','with_repair'):
            for arm in ('active_zero','active_replicate'):
                row[f'{arm}_minus_stale_{repair}_db']=(
                    row[f'{arm}_{repair}_gain_db']-row[f'stale_{repair}_gain_db'])
        rows.append(row)
    rng=np.random.default_rng(20261005)
    groups={'all':rows,'uniform':[r for r in rows if r['mixed_edge_count']==0],
            'mixed':[r for r in rows if r['mixed_edge_count']>0]}
    summary={group:{key:describe(subset,key,rng) for key in
                    ('stale_no_repair_gain_db','active_zero_no_repair_gain_db',
                     'active_replicate_no_repair_gain_db','active_zero_with_repair_gain_db',
                     'active_replicate_with_repair_gain_db',
                     'stale_with_repair_gain_db',
                     'active_zero_minus_stale_no_repair_db',
                     'active_replicate_minus_stale_no_repair_db',
                     'active_zero_minus_stale_with_repair_db',
                     'active_replicate_minus_stale_with_repair_db',
                     'replicate_minus_zero_no_repair_db',
                     'replicate_minus_zero_with_repair_db')}
             for group,subset in groups.items()}
    result={'scope':'Kodak24 x QP5 paired actual bitstream reconstruction, unchanged exit maps; exploratory because pilot preceded full cohort. No latency or native CUDA stream.',
            'input_sha256':{name:sha(path) for name,path in FILES.items()},
            'bootstrap':'10000 image-cluster draws, retaining the observed QP cases within each sampled image',
            'overall':summary['all'],'by_edge_class':{k:v for k,v in summary.items() if k!='all'},
            'per_qp':{str(q):{key:describe([r for r in rows if r['qp']==q],key,rng)
                               for key in ('stale_no_repair_gain_db','active_zero_no_repair_gain_db',
                                           'active_replicate_no_repair_gain_db',
                                           'stale_with_repair_gain_db',
                                           'active_zero_with_repair_gain_db',
                                           'active_replicate_with_repair_gain_db')}
                      for q in (0,16,32,48,63)},
            'by_qp_edge_class':{
                str(q):{group:{key:describe(subset,key,rng) for key in
                               ('active_zero_minus_stale_with_repair_db',
                                'active_replicate_minus_stale_with_repair_db')}
                        for group,subset in (
                            ('uniform',[r for r in rows if r['qp']==q and r['mixed_edge_count']==0]),
                            ('mixed',[r for r in rows if r['qp']==q and r['mixed_edge_count']>0]))}
                for q in (0,16,32,48,63)},
            'over_0p1_delta444':{'deployed':sum(r['deployed_delta444_db']>0.1 for r in rows),
                                'exact_context':sum(r['exact_context_delta444_db']>0.1 for r in rows),
                                **{f'{arm}_{repair}':sum(r[f'{arm}_{repair}_delta444_db']>0.1 for r in rows)
                                   for arm in FILES for repair in ('no_repair','with_repair')}},
            'rows':rows}
    OUT.write_text(json.dumps(result,indent=2)+'\n')
    print(json.dumps({'overall':result['overall'],'over_0p1_delta444':result['over_0p1_delta444']},indent=2))


if __name__=='__main__':main()
