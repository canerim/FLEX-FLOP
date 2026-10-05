"""Reconcile historical Kodak active-canvas MAC with exact conv arithmetic."""
from __future__ import annotations

import hashlib
import json
from pathlib import Path

import numpy as np

from exact_active_conv_mac import exact_active_no_repair_saving_pct
from mac_latency_audit import case_macs

HERE=Path(__file__).resolve().parent
ROOT=HERE.parents[1]
OLD=HERE/'results/kodak_active_canvas_cost_20261005.json'
MAPS=HERE/'results/div2k_beta/quality_floor/exact_context_kodak24.json'
OUT=HERE/'results/kodak_active_canvas_exact_conv_cost_20261005.json'


def sha(path:Path)->str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main()->None:
    old=json.loads(OLD.read_text())
    maps=json.loads(MAPS.read_text())
    assert len(old['rows'])==len(maps['rows'])==120
    assert old['source_sha256']['exact']==sha(MAPS)
    keyed={(r['image'],r['qp']):r for r in maps['rows']}
    assert len(keyed)==120
    rows=[]
    for prior in old['rows']:
        image,qp=prior['image'],prior['qp']
        archived=keyed[image,qp]
        assert prior['exit_map']==archived['exit_map']
        exact=exact_active_no_repair_saving_pct(prior['exit_map'])
        legacy=prior['coupled_no_repair_conv_saving_pct']
        rows.append({'image':image,'qp':qp,'exit_map':prior['exit_map'],
                     'legacy_active_no_repair_saving_pct':legacy,
                     'exact_active_no_repair_saving_pct':exact,
                     'legacy_minus_exact_points':legacy-exact})
    by_qp={str(qp):[r for r in rows if r['qp']==qp] for qp in (0,16,32,48,63)}
    assert all(len(x)==24 for x in by_qp.values())
    def stats(group):
        x=np.asarray([r['legacy_minus_exact_points'] for r in group])
        exact=np.asarray([r['exact_active_no_repair_saving_pct'] for r in group])
        return {'n':len(x),'exact_mean_saving_pct':float(exact.mean()),
                'legacy_minus_exact_mean_points':float(x.mean()),
                'legacy_minus_exact_min_points':float(x.min()),
                'legacy_minus_exact_max_points':float(x.max())}
    result={
        'scope':'Kodak24 x QP5 frozen old-beta maps, corrected exact synthesis Conv2d MAC for active-replicate/no-repair. Same FUFREF2/e15; no new reconstruction, quality, latency or native bitrate result.',
        'reason':'Active depthwise F.conv2d takes (P+2)^2 input but emits P^2 outputs; input halo does not add convolution MAC. Historical frame_relative_cost also normalizes using rounded shares and an 8C^2 block proxy, while the real DepthConvBlock costs 7C^2+9C. Exact arithmetic follows mac_latency_audit.case_macs and removes GridSeamRepair.',
        'source_sha256':{'old':sha(OLD),'maps':sha(MAPS),
                         'formula':sha(HERE/'exact_active_conv_mac.py'),
                         'mac_latency_audit':sha(ROOT/'proof/early_exit_vs_released/mac_latency_audit.py'),
                         'coupler':sha(HERE/'active_canvas_replicate.py'),
                         'script':sha(Path(__file__))},
        'overall':stats(rows),'per_qp':{qp:stats(group) for qp,group in by_qp.items()},
        'rows':rows,
    }
    deployed={}
    for cohort in ('calibration24','validation24','kodak24'):
        source=HERE/f'results/div2k_beta/quality_floor/exact_context_{cohort}.json'
        data=json.loads(source.read_text())
        assert len(data['rows'])==120
        savings=[]
        by_image={}
        for row in data['rows']:
            route=row['exit_map']
            count=[route.count(k) for k in range(6)]
            accounting=case_macs({'padded_shape':[512,768],'tile_counts':count})
            saving=100*accounting['e15_routed_conv_mac_saving_fraction']
            savings.append(saving)
            by_image.setdefault(row['image'],{})[row['qp']]=saving
        assert len(by_image)==24 and all(set(v)=={0,16,32,48,63} for v in by_image.values())
        matrix=np.asarray([[by_image[image][qp] for qp in (0,16,32,48,63)]
                           for image in sorted(by_image)])
        rng=np.random.default_rng(20261005)
        draw=rng.integers(0,24,size=(10000,24))
        boot=matrix[draw].mean(axis=(1,2))
        deployed[cohort]={'n':len(savings),'exact_mean_saving_pct':float(np.mean(savings)),
                          'image_cluster_ci95_pct':np.quantile(boot,[.025,.975]).tolist(),
                          'min_saving_pct':float(np.min(savings)),
                          'max_saving_pct':float(np.max(savings)),
                          'source_sha256':sha(source)}
    result['deployed_exact_conv_by_cohort']=deployed
    OUT.write_text(json.dumps(result,indent=2)+'\n')
    print(json.dumps({'overall':result['overall'],'per_qp':result['per_qp']},indent=2))


if __name__=='__main__':main()
