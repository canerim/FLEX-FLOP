"""Arithmetic feasibility of cheap continuation for exited early-exit tiles.

Inspired by Mosaic's low-cost context path, but this does not instantiate or
evaluate that path. It only counts one dense 3x3 depthwise convolution per
skipped suffix block on each exited tile, plus the already-audited active
one-cell halo. A learned continuation would need its own quality, movement,
memory and latency evaluation.
"""
from __future__ import annotations

import hashlib
import json
from pathlib import Path

ROOT=Path(__file__).resolve().parents[2]
HERE=Path(__file__).resolve().parent
SOURCE=HERE/'results/kodak_active_canvas_cost_20261005.json'
OUT=HERE/'results/kodak_ghost_context_cost_20261005.json'
FEATURE_PATCH=32
FULL_SUFFIX_BLOCKS=8


def sha(path:Path)->str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main()->None:
    source=json.loads(SOURCE.read_text())
    rows=source['rows']
    assert len(rows)==120 and source['source_sha256']['cost_model']==sha(ROOT/'flexuf/cost.py')
    # The active-canvas audit charges only the extra halo area. Undo that
    # area multiplier to recover the full-tile depthwise share per block.
    halo_area_ratio=((FEATURE_PATCH+2)/FEATURE_PATCH)**2-1
    dw_share=source['per_block_extra_fraction']/halo_area_ratio
    assert dw_share>0
    out=[]
    for row in rows:
        executed=row['mean_executed_suffix_blocks_per_tile']
        skipped=FULL_SUFFIX_BLOCKS-executed
        assert 0<=skipped<=FULL_SUFFIX_BLOCKS
        ghost_points=100*skipped*dw_share
        out.append({'image':row['image'],'qp':row['qp'],
                    'mean_skipped_suffix_blocks_per_tile':skipped,
                    'ghost_depthwise_cost_pct_points':ghost_points,
                    'ghost_plus_active_halo_cost_pct_points':(
                        ghost_points+row['extra_depthwise_halo_pct_points']),
                    'hypothetical_with_repair_saving_pct':(
                        row['coupled_with_repair_conv_saving_pct']-ghost_points)})
    mean=lambda key:sum(r[key] for r in out)/len(out)
    result={'scope':'Kodak24 x QP5 frozen routes; hypothetical one full-tile 3x3 DW per skipped suffix block. No trained ghost path, quality result, memory accounting, kernel, or latency.',
            'source_sha256':sha(SOURCE),'script_sha256':sha(Path(__file__)),
            'feature_patch':FEATURE_PATCH,
            'full_suffix_blocks':FULL_SUFFIX_BLOCKS,
            'per_full_tile_depthwise_block_fraction':dw_share,
            'mean':{key:mean(key) for key in
                    ('mean_skipped_suffix_blocks_per_tile',
                     'ghost_depthwise_cost_pct_points',
                     'ghost_plus_active_halo_cost_pct_points',
                     'hypothetical_with_repair_saving_pct')},
            'rows':out}
    OUT.write_text(json.dumps(result,indent=2)+'\n')
    print(json.dumps(result['mean'],indent=2))


if __name__=='__main__':main()
