"""Optimistic cell-area cost for sharing overlapping exact-context windows.

For each suffix block, take the union of feature cells required by every
tile's exact full-frame dependency cone. Charge each distinct cell only once,
plus the shared stem, full-frame head, and core-only exit adapters. This is an
analytical optimistic cost for a hypothetical masked scheduler, not a measured
kernel, a global lower bound for all algorithms, or a latency prediction.
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


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--e15',type=Path,required=True)
    p.add_argument('--case',type=Path,action='append',required=True)
    p.add_argument('--output',type=Path,required=True)
    args=p.parse_args()
    cfg=FlexUFConfig(**torch.load(args.e15,map_location='cpu',weights_only=False)['config'])
    if cfg.split_depth!=2 or cfg.blocks_per_exit!=2 or cfg.feature_patch!=32:
        raise RuntimeError('This audit assumes the frozen 2+2 blocks/32-cell geometry')
    per_block=SHARE_TRUNK/12
    rows=[]
    for case in args.case:
        d=json.loads(case.read_text())
        _,_,rgb_w,rgb_h=d['crop_xywh']
        if (rgb_w,rgb_h)!=(768,512):
            raise RuntimeError(f'Unexpected crop in {case}')
        h,w=rgb_h//8,rgb_w//8
        patch=cfg.feature_patch
        if (h,w)!=(64,96):
            raise RuntimeError('Unexpected feature canvas')
        for name,c in d['controls'].items():
            route=c['exit_map']
            if len(route)!=6:
                raise RuntimeError('Expected six tile modes')
            union_areas=[]
            for block in range(8):
                mask=np.zeros((h,w),dtype=np.bool_)
                for tile,mode in enumerate(route):
                    remaining=(mode-cfg.split_depth+1)*cfg.blocks_per_exit-block
                    if remaining<=0:
                        continue
                    top,left=(tile//3)*patch,(tile%3)*patch
                    mask[max(0,top-remaining):min(h,top+patch+remaining),
                         max(0,left-remaining):min(w,left+patch+remaining)]=True
                union_areas.append(int(mask.sum()))
            adapter_cost=sum(adapter_vs_block(cfg.adapter_kind,mode,cfg)
                             for mode in route if mode<cfg.num_exits-1)/len(route)*per_block
            cost=(SHARE_UPSAMPLE+cfg.split_depth*cfg.blocks_per_exit*per_block+
                  SHARE_HEAD+per_block*sum(union_areas)/(h*w)+adapter_cost)
            rows.append({'image':d['image'],'qp':d['qp'],'map_kind':name,
                         'case_sha256':sha(case),'exit_map':route,
                         'per_suffix_block_union_feature_cells':union_areas,
                         'ideal_shared_context_relative_mac':cost,
                         'exact_context_delta444_db':c['exact_context_no_repair_delta444_db'],
                         'deployed_zero_halo_delta444_db':c['zero_halo_deployed_delta444_db']})
    result={'schema':1,'scope':__doc__.strip(),'e15_sha256':sha(args.e15),
            'full_frame_relative_mac':1.0,'rows':rows}
    args.output.parent.mkdir(parents=True,exist_ok=True)
    args.output.write_text(json.dumps(result,indent=2)+'\n')
    for r in rows:
        print(f"{r['image']} QP{r['qp']:02d} {r['map_kind']}: "
              f"ideal MAC {r['ideal_shared_context_relative_mac']:.5f}, "
              f"exact Δ444 {r['exact_context_delta444_db']:.5f} dB")


if __name__=='__main__':
    main()
