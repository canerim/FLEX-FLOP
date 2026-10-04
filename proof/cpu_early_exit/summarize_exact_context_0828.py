"""Summarize five frozen DIV2K context counterfactuals and their area MAC proxy."""
from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import torch

from flexuf.config import FlexUFConfig
from flexuf.cost import (SHARE_HEAD, SHARE_TRUNK, SHARE_UPSAMPLE,
                         adapter_vs_block, frame_relative_cost,
                         seam_repair_share)


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--input-dir', type=Path, required=True)
    p.add_argument('--e15', type=Path, required=True)
    p.add_argument('--output-dir', type=Path, required=True)
    args = p.parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    cfg = FlexUFConfig(**torch.load(args.e15, map_location='cpu', weights_only=False)['config'])
    b, j, k = cfg.blocks_per_exit, cfg.split_depth, cfg.num_exits
    per_block = SHARE_TRUNK / (k*b)
    result = {'scope': 'Analytical synthesis MAC share only; context windows charge every pointwise and depthwise operation over their full area; no latency measurement',
              'cases': {}}
    for qp in (0,16,32,48,63):
        data = json.loads((args.input_dir/f'exact_context_0828_qp{qp}.json').read_text())
        if data['image'] != '0828.png' or data['qp'] != qp:
            raise RuntimeError('Wrong case')
        total = cfg.feature_patch**2 * len(data['controls']['all_deep']['exit_map'])
        cases = {}
        for name, c in data['controls'].items():
            route = torch.tensor(c['exit_map'], dtype=torch.long)
            standard = frame_relative_cost(route, cfg, 'head')
            windows = [(r-t)*(right-left) for t,left,r,right in c['tile_windows_tlbr']]
            adapter_area = sum(
                adapter_vs_block(cfg.adapter_kind, int(mode), cfg)*area
                for mode,area in zip(c['exit_map'], windows) if mode < k-1)
            context_base = (SHARE_UPSAMPLE + j*b*per_block + SHARE_HEAD +
                            b*per_block*sum(c['per_suffix_group_feature_cell_areas'])/total +
                            per_block*adapter_area/total)
            cases[name] = {
                'quality_delta444_db': {key:c[key] for key in (
                    'zero_halo_deployed_delta444_db',
                    'exact_context_no_repair_delta444_db',
                    'exact_context_with_repair_delta444_db',
                    'canvas_coupled_no_repair_delta444_db',
                    'canvas_coupled_with_repair_delta444_db')},
                'zero_halo_deployed_relative_mac': standard,
                'exact_context_no_repair_relative_mac': context_base,
                'exact_context_with_repair_relative_mac': context_base+seam_repair_share(cfg.seam_repair),
                'suffix_feature_cell_area_multiplier': (
                    sum(c['per_suffix_group_feature_cell_areas']) /
                    sum(c['zero_halo_suffix_feature_cell_areas'])),
            }
            if 'tapered_suffix_block_feature_cell_areas' in c:
                taper_adapter = sum(
                    adapter_vs_block(cfg.adapter_kind, int(mode), cfg)*cfg.feature_patch**2
                    for mode in c['exit_map'] if mode < k-1)
                cases[name]['tapered_context_no_repair_relative_mac'] = (
                    SHARE_UPSAMPLE + j*b*per_block + SHARE_HEAD +
                    per_block*sum(c['tapered_suffix_block_feature_cell_areas'])/total +
                    per_block*taper_adapter/total)
                cases[name]['tapered_vs_static_suffix_area_ratio'] = (
                    sum(c['tapered_suffix_block_feature_cell_areas']) /
                    (b*sum(c['per_suffix_group_feature_cell_areas'])))
        result['cases'][str(qp)] = cases
    (args.output_dir/'context_counterfactual_summary.json').write_text(json.dumps(result,indent=2)+'\n')

    plt.rcParams.update({'font.family':'DejaVu Sans','font.size':8,'pdf.fonttype':42,
                         'axes.linewidth':.7})
    fig, axes = plt.subplots(1,2,figsize=(7.2,2.7),sharey=False,constrained_layout=True)
    labels = ['Deployed\nzero halo','Exact context\nno repair','Coupled DW\nno repair']
    keys = ['zero_halo_deployed_delta444_db',
            'exact_context_no_repair_delta444_db',
            'canvas_coupled_no_repair_delta444_db']
    colors = ['#B76345','#297E87','#7B5792']
    for ax, qp in zip(axes,(0,63)):
        rows = result['cases'][str(qp)]
        for idx,name in enumerate(('all_deep','primary_route')):
            values = [rows[name]['quality_delta444_db'][key] for key in keys]
            xx = [jdx+idx*.35 for jdx in range(3)]
            ax.bar(xx,values,width=.33,color=colors,alpha=1 if idx else .48,
                   hatch=None if idx else '//',linewidth=.2,edgecolor='#263238')
        ax.set_xticks([v+.175 for v in range(3)],labels)
        ax.set_ylabel('Δ444 versus e15 full (dB)')
        ax.set_title(f"{'a' if qp==0 else 'b'}  DIV2K 0828 · QP{qp}",loc='left',fontweight='bold')
        ax.axhline(.1,color='#33454D',lw=.9,ls=(0,(3,2)))
        ax.grid(axis='y',color='#E3E8EA',lw=.55,zorder=0)
        ax.spines[['top','right']].set_visible(False)
        ax.tick_params(axis='x',length=0)
        ax.set_ylim(0,.83 if qp==0 else .18)
        ax.text(.98,.96,'hatched: all-deep  |  solid: fixed router',
                ha='right',va='top',transform=ax.transAxes,fontsize=6.2,color='#50616B')
    for ext in ('pdf','png'):
        fig.savefig(args.output_dir/f'context_counterfactual.{ext}',dpi=220,bbox_inches='tight')
    plt.close(fig)
    print(json.dumps(result,indent=2))


if __name__ == '__main__':
    main()
