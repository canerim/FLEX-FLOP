"""Join a source-informed uniform-depth counterfactual to its fixed baseline."""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path


def main() -> None:
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--base',type=Path,required=True)
    p.add_argument('--oracle',type=Path,required=True)
    p.add_argument('--output',type=Path,required=True)
    args=p.parse_args()
    base=json.loads(args.base.read_text())
    oracle=json.loads(args.oracle.read_text())
    rows=base['rows']
    failures={(r['image'],r['qp']) for r in rows if r['exact_context_delta444_db']>.1}
    joined={(r['image'],r['qp']):r for r in oracle['rows']}
    if len(rows)!=120 or len(failures)!=24 or set(joined)!=failures:
        raise RuntimeError('Full baseline / oracle mismatch')
    if oracle['provenance']['base_sha256']!=hashlib.sha256(args.base.read_bytes()).hexdigest():
        raise RuntimeError('Baseline hash mismatch')
    def avg(values):return sum(values)/len(values)
    base_savings=[r['ideal_shared_saving_pct'] for r in rows]
    all_deep_fallback=[]
    uniform_fallback=[]
    final_losses=[]
    increments={1:0,2:0,3:0}
    for row in rows:
        key=(row['image'],row['qp'])
        if key in joined:
            item=joined[key]['chosen']
            increments[item['uniform_increment']]+=1
            all_deep_fallback.append(0.)
            uniform_fallback.append(item['ideal_shared_saving_pct'])
            final_losses.append(item['exact_context_delta444_db'])
        else:
            all_deep_fallback.append(row['ideal_shared_saving_pct'])
            uniform_fallback.append(row['ideal_shared_saving_pct'])
            final_losses.append(row['exact_context_delta444_db'])
    summary={'schema':1,'scope':'Source-informed oracle with reference-error feedback; unattainable deployment upper bound; ideal synthesis arithmetic only',
             'cases':len(rows),'baseline_violations':len(failures),
             'base_mean_ideal_saving_pct':avg(base_savings),
             'all_deep_on_failures_mean_ideal_saving_pct':avg(all_deep_fallback),
             'uniform_increment_on_failures_mean_ideal_saving_pct':avg(uniform_fallback),
             'uniform_increment_counts_on_failures':increments,
             'uniform_final_over_0p1_count':sum(v>.1 for v in final_losses),
             'uniform_final_mean_delta444_db':avg(final_losses),
             'additional_ideal_saving_vs_all_deep_fallback_pct_points':avg(uniform_fallback)-avg(all_deep_fallback),
             'source_feedback_required':True,'latency_measured':False}
    args.output.parent.mkdir(parents=True,exist_ok=True)
    args.output.write_text(json.dumps(summary,indent=2)+'\n')
    print(json.dumps(summary,indent=2))


if __name__=='__main__':main()
