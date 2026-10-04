"""Ideal shared-context cell-area proxy for the frozen DIV2K beta cohorts.

The proxy shares each exact-context suffix feature cell perfectly across
overlapping tiles and charges adapters on tile cores. It is neither an
implemented decoder nor an exact-context quality measurement.
"""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import sys

import numpy as np
import torch

ROOT=Path(__file__).resolve().parents[2]
sys.path.insert(0,str(ROOT))
from flexuf.config import FlexUFConfig
from flexuf.cost import SHARE_HEAD,SHARE_TRUNK,SHARE_UPSAMPLE,adapter_vs_block


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def ideal_cost_grid(route,cfg,nh,nw):
    patch=cfg.feature_patch
    h,w=nh*patch,nw*patch
    if len(route)!=nh*nw:
        raise RuntimeError('Exit map does not match feature grid')
    cells=[]
    for block in range((cfg.num_exits-cfg.split_depth)*cfg.blocks_per_exit):
        mask=np.zeros((h,w),dtype=bool)
        for tile,mode in enumerate(route):
            remaining=(mode-cfg.split_depth+1)*cfg.blocks_per_exit-block
            if remaining<=0:
                continue
            top,left=(tile//nw)*patch,(tile%nw)*patch
            mask[max(0,top-remaining):min(h,top+patch+remaining),
                 max(0,left-remaining):min(w,left+patch+remaining)]=True
        cells.append(int(mask.sum()))
    per=SHARE_TRUNK/(cfg.num_exits*cfg.blocks_per_exit)
    adapters=sum(adapter_vs_block(cfg.adapter_kind,mode,cfg)
                 for mode in route if mode<cfg.num_exits-1)/len(route)*per
    return (SHARE_UPSAMPLE+cfg.split_depth*cfg.blocks_per_exit*per+
            SHARE_HEAD+per*sum(cells)/(h*w)+adapters),cells


def ideal_cost(route,cfg):
    return ideal_cost_grid(route,cfg,2,3)


def aggregate(rows):
    selected=np.asarray([r['ideal_shared_context_mac_saved_pct'] for r in rows])
    loss=np.asarray([r['deployed_delta444_db'] for r in rows])
    good=selected[loss<=.1]; bad=selected[loss>.1]
    return {'cases':len(rows),'mean_ideal_mac_saved_pct':float(selected.mean()),
            'median_ideal_mac_saved_pct':float(np.median(selected)),
            'ideal_saving_at_least_10pct_count':int((selected>=10).sum()),
            'deployed_over_0p1_count':len(bad),
            'mean_ideal_saving_on_deployed_within_budget_pct':float(good.mean()),
            'mean_ideal_saving_on_deployed_over_budget_pct':float(bad.mean()),
            'interpretation':'Quality groups describe deployed zero-halo outputs, not exact-context outputs.'}


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--manifest',type=Path,required=True)
    p.add_argument('--policy',type=Path,required=True)
    p.add_argument('--cal-dir',type=Path,required=True)
    p.add_argument('--val-dir',type=Path,required=True)
    p.add_argument('--e15',type=Path,required=True)
    p.add_argument('--output',type=Path,required=True)
    args=p.parse_args()
    manifest=json.loads(args.manifest.read_text())
    policy=json.loads(args.policy.read_text())
    manifest_sha=sha(args.manifest)
    if policy['manifest_sha256']!=manifest_sha:
        raise RuntimeError('Policy/manifest mismatch')
    cfg=FlexUFConfig(**torch.load(args.e15,map_location='cpu',weights_only=False)['config'])
    if cfg.split_depth!=2 or cfg.blocks_per_exit!=2 or cfg.feature_patch!=32:
        raise RuntimeError('Unexpected e15 decoder geometry')
    outputs={}; summaries={}
    for split,dirpath in [('calibration',args.cal_dir),('validation',args.val_dir)]:
        rows=[]
        for item in manifest['rows'][split]:
            for qp in manifest['qps']:
                path=dirpath/f'{Path(item["image"]).stem}_qp{qp}.json'
                case=json.loads(path.read_text())
                if (case['manifest_sha256']!=manifest_sha or
                    case['source_sha256']!=item['source_sha256'] or
                    case['image']!=item['image'] or case['qp']!=qp):
                    raise RuntimeError(f'Case provenance mismatch: {path}')
                candidates=[c for c in case['candidates']
                            if c['beta']==policy['beta'][str(qp)]]
                if len(candidates)!=1:
                    raise RuntimeError(f'Missing locked map: {path}')
                c=candidates[0]
                route=c['exit_map']
                if len(route)!=6 or any(mode<cfg.split_depth or mode>=cfg.num_exits for mode in route):
                    raise RuntimeError(f'Unexpected map: {path}')
                cost,cells=ideal_cost(route,cfg)
                rows.append({'image':item['image'],'qp':qp,'case_sha256':sha(path),
                             'exit_map':route,'per_suffix_block_union_feature_cells':cells,
                             'ideal_shared_context_relative_mac':cost,
                             'ideal_shared_context_mac_saved_pct':100*(1-cost),
                             'deployed_delta444_db':c['delta444_db'],
                             'deployed_zero_halo_mac_saved_pct':c['mac_saved_pct']})
        if len(rows)!=120:
            raise RuntimeError(f'Expected 120 cases for {split}')
        outputs[split]=rows
        summaries[split]={'all_qps':aggregate(rows),
                          'per_qp':{str(q):aggregate([r for r in rows if r['qp']==q])
                                    for q in manifest['qps']}}
    result={'schema':1,'scope':__doc__.strip(),'manifest_sha256':manifest_sha,
            'policy_sha256':sha(args.policy),'e15_sha256':sha(args.e15),
            'summaries':summaries,'rows':outputs}
    args.output.parent.mkdir(parents=True,exist_ok=True)
    args.output.write_text(json.dumps(result,indent=2)+'\n')
    print(json.dumps(summaries,indent=2))


if __name__=='__main__':
    main()
