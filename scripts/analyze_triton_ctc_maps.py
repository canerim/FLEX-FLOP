"""CPU-only workload audit for the 53-sequence × 5-QP CTC router cohort."""
from __future__ import annotations

from collections import defaultdict
import json
import math
from pathlib import Path
import statistics

ROOT=Path(__file__).resolve().parents[1]
rows=json.loads((ROOT/'flexplus/results/eval_rules_ctc_e15.json').read_text())['rows']


def summarize(records):
    groups=defaultdict(list)
    for r in records:
        groups[r['qp']].append(r)
    result={}
    for qp,cases in sorted(groups.items()):
        routed=[r for r in cases if r['rules']['router']['0.1'] is not None]
        hist=[0]*6
        entropy=[]
        for r in routed:
            h=r['rules']['router']['0.1']['hist']
            assert sum(h)==len(r['rules']['router']['0.1']['map'])==r['grid'][0]*r['grid'][1]
            assert all(h[k]==r['rules']['router']['0.1']['map'].count(k) for k in range(6))
            for i,count in enumerate(h):hist[i]+=count
            n=sum(h)
            entropy.append(-sum((v/n)*math.log2(v/n) for v in h if v))
        total=sum(hist)
        mean_depth=sum(hist[k]*2*(k+1) for k in range(6))/total
        # Per tile, four stem blocks are shared; suffix count is 2,4,6,8
        # for the reachable D6,D8,D10,D12 exits (indices 2..5).
        mean_suffix=sum(hist[k]*2*(k-1) for k in range(2,6))/total
        result[str(qp)]={
            'n_cases':len(cases),'n_routed_cases':len(routed),
            'n_dense_fallback_cases':len(cases)-len(routed),
            'aggregate_hist':hist,'total_tiles':total,
            'exit_fraction':[v/total for v in hist],
            'mean_retained_blocks':mean_depth,
            'mean_suffix_blocks_per_tile':mean_suffix,
            'mean_map_entropy_bits_per_tile':statistics.mean(entropy),
            'all_one_exit_maps':sum(sum(v>0 for v in r['rules']['router']['0.1']['hist'])==1
                                    for r in routed),
        }
    return {'scope':'archived source-calibrated router maps; CPU-only workload audit, not latency',
            'n_rows':len(records),'n_sequences':len({r['seq'] for r in records}),
            'by_qp':result}


if __name__=='__main__':
    out=summarize(rows)
    path=ROOT/'results/triton_ctc_map_workload.json'
    path.write_text(json.dumps(out,indent=2)+'\n')
    for qp,r in out['by_qp'].items():
        print('QP',qp,'hist',r['aggregate_hist'],'mean blocks',round(r['mean_retained_blocks'],2),
              'missing maps',r['n_dense_fallback_cases'])
    print('saved',path)
