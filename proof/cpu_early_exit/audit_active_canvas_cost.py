"""Analytical conv-MAC correction for frozen active-neighbour coupling maps.

The extra (P+2)^2 depthwise area is counted exactly against the project's
full-synthesis conv-MAC normalization. This does not count tile packing,
gather/scatter, memory traffic, or scheduling, so it is not a latency claim.
"""
from __future__ import annotations

import hashlib
import json
import sys
from pathlib import Path

import torch

ROOT=Path(__file__).resolve().parents[2]
sys.path.insert(0,str(ROOT))
from flexuf.config import FlexUFConfig
from flexuf.cost import (_BLOCK_MACPX, _C, N_TRUNK_BLOCKS, SHARE_TRUNK,
                         frame_relative_cost, seam_repair_share)

HERE=Path(__file__).resolve().parent
EXACT=HERE/'results/div2k_beta/quality_floor/exact_context_kodak24.json'
POLICY_SUMMARY=HERE/'results/div2k_beta/per_qp_policy_summary.json'
CHECKPOINT=ROOT/'proof/early_exit_vs_released/artifacts/e15_epoch15.pth.tar'
OUT=HERE/'results/kodak_active_canvas_cost_20261005.json'


def sha(path:Path)->str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main()->None:
    torch.set_num_threads(1)
    exact=json.loads(EXACT.read_text())
    policy_summary=json.loads(POLICY_SUMMARY.read_text())
    ckpt=torch.load(CHECKPOINT,map_location='cpu',weights_only=False)
    cfg=FlexUFConfig(**ckpt['config'])
    assert cfg.trunk_halo==0 and cfg.tile_coupling is False
    assert cfg.seam_repair=='grid' and cfg.feature_patch==32
    # One 3x3 depthwise runs per trunk block on an extra one-cell ring.
    per_block_extra=(9*_C/_BLOCK_MACPX)*(SHARE_TRUNK/N_TRUNK_BLOCKS)*(
        (cfg.feature_patch+2)**2/cfg.feature_patch**2-1)
    repair_share=seam_repair_share(cfg.seam_repair)
    rows=[]
    for r in exact['rows']:
        route=torch.tensor(r['exit_map'],dtype=torch.long)
        blocks=float(((route-cfg.split_depth+1)*cfg.blocks_per_exit).float().mean())
        base=frame_relative_cost(route,cfg)
        overhead=blocks*per_block_extra
        rows.append({'image':r['image'],'qp':r['qp'],'exit_map':r['exit_map'],
                     'mean_executed_suffix_blocks_per_tile':blocks,
                     'deployed_conv_saving_pct':100*(1-base),
                     'coupled_with_repair_conv_saving_pct':100*(1-base-overhead),
                     'coupled_no_repair_conv_saving_pct':100*(1-base+repair_share-overhead),
                     'extra_depthwise_halo_pct_points':100*overhead,
                     'removed_grid_repair_pct_points':100*repair_share})
    assert len(rows)==120
    mean_deployed=sum(r['deployed_conv_saving_pct'] for r in rows)/120
    assert abs(mean_deployed-policy_summary['kodak_transfer']['equal_qp_mean']['mac_saved_pct']['mean'])<1e-10
    result={'scope':'Frozen Kodak24 x QP5 convolution arithmetic only; active-neighbour and stale-neighbour coupling have the same conv count. Excludes tile packing, memory, router, entropy and latency.',
            'source_sha256':{'exact':sha(EXACT),'checkpoint':sha(CHECKPOINT),
                             'cost_model':sha(ROOT/'flexuf/cost.py'),
                             'locked_policy_summary':sha(POLICY_SUMMARY)},
            'formula':'Deployed frame_relative_cost + mean executed suffix blocks per tile × [(9C)/(block MAC per feature pixel)] × [trunk share/12] × [((P+2)^2/P^2)-1]; no-repair subtracts grid seam-repair share.',
            'per_block_extra_fraction':per_block_extra,'grid_repair_fraction':repair_share,
            'mean':{key:sum(r[key] for r in rows)/120 for key in
                    ('deployed_conv_saving_pct','coupled_with_repair_conv_saving_pct',
                     'coupled_no_repair_conv_saving_pct','extra_depthwise_halo_pct_points',
                     'removed_grid_repair_pct_points')},
            'rows':rows}
    OUT.write_text(json.dumps(result,indent=2)+'\n')
    print(json.dumps(result['mean'],indent=2))


if __name__=='__main__':main()
