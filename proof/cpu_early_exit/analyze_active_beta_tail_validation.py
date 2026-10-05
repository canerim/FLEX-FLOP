"""Paired mean/tail beta transfer analysis on frozen DIV2K validation24."""
from __future__ import annotations

import hashlib
import json
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
BASE = HERE/'results/div2k_beta/quality_floor'
RAW = BASE/'active_replicate_tail_validation24.json'
POLICY = BASE/'active_replicate_tail_locked_policy.json'
OUT = BASE/'active_replicate_tail_validation24_analysis.json'
QPS = (0,16,32,48,63)


def sha(path:Path)->str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def summary(rows:list[dict],key:str,rng:np.random.Generator)->dict:
    images=sorted({r['image'] for r in rows})
    n_qp=len(rows)//len(images)
    assert len(images)==24 and n_qp in (1,5)
    matrix=np.asarray([[r[key] for r in rows if r['image']==image]
                       for image in images],dtype=float)
    assert matrix.shape==(24,n_qp)
    boot=matrix[rng.integers(0,24,size=(10000,24))].mean(axis=(1,2))
    return {'mean':float(matrix.mean()),
            'image_cluster_ci95':np.quantile(boot,[.025,.975]).tolist(),
            'min':float(matrix.min()),'max':float(matrix.max())}


def main()->None:
    raw=json.loads(RAW.read_text())
    policy=json.loads(POLICY.read_text())
    assert raw['complete'] and len(raw['rows'])==raw['expected_cases']==120
    assert raw['provenance']['policy_sha256']==sha(POLICY)
    rows=raw['rows']
    keys={(r['image'],r['qp']) for r in rows}
    assert len(keys)==120 and len({im for im,_ in keys})==24
    assert all({qp for im,qp in keys if im==image}==set(QPS)
               for image in {im for im,_ in keys})
    assert all(r['tail_beta']==policy['beta'][str(r['qp'])] for r in rows)
    extended=[]
    for r in rows:
        extended.append({**r,
                         'tail_minus_primary_saving_points':
                             r['tail_conv_mac_saving_pct']-r['primary_conv_mac_saving_pct'],
                         'tail_minus_primary_quality_gain_db':
                             r['primary_delta444_db']-r['tail_delta444_db']})
    rng=np.random.default_rng(20261010)
    fields=('tail_delta444_db','primary_delta444_db',
            'tail_conv_mac_saving_pct','primary_conv_mac_saving_pct',
            'tail_minus_primary_saving_points',
            'tail_minus_primary_quality_gain_db')
    all_stats={field:summary(extended,field,rng) for field in fields}
    by_qp={str(qp):{field:summary([r for r in extended if r['qp']==qp],field,rng)
                     for field in fields} for qp in QPS}
    thresholds={
        'tail_over_0p1':sum(r['tail_delta444_db']>.1 for r in rows),
        'primary_over_0p1':sum(r['primary_delta444_db']>.1 for r in rows),
        'rescued':sum(r['primary_delta444_db']>.1 and r['tail_delta444_db']<=.1
                      for r in rows),
        'newly_failed':sum(r['primary_delta444_db']<=.1 and r['tail_delta444_db']>.1
                            for r in rows),
        'per_qp':{str(qp):{
            'tail':sum(r['tail_delta444_db']>.1 for r in rows if r['qp']==qp),
            'primary':sum(r['primary_delta444_db']>.1 for r in rows if r['qp']==qp),
        } for qp in QPS},
    }
    output={
        'scope':'Exploratory calibration-only tail-vs-mean beta transfer on DIV2K validation24 x five-QP, same e15/FUFREF2 and active-replicate/no-repair; untimed analytical MAC.',
        'source_sha256':{'raw':sha(RAW),'policy':sha(POLICY),
                         'script':sha(Path(__file__))},
        'bootstrap':'10000 image-cluster draws retaining all five QPs per image',
        'beta':policy['beta'],'all':all_stats,'per_qp':by_qp,
        'thresholds':thresholds,'rows':extended,
    }
    OUT.write_text(json.dumps(output,indent=2)+'\n')
    print(json.dumps({'mean_quality_gain_db':all_stats['tail_minus_primary_quality_gain_db']['mean'],
                      'mean_saving_change_points':all_stats['tail_minus_primary_saving_points']['mean'],
                      'thresholds':thresholds},indent=2))


if __name__=='__main__':main()
