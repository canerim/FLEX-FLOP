"""Sequence-cluster uncertainty for an isolated 53×5 CTC synthesis benchmark."""
from __future__ import annotations

import argparse
from collections import defaultdict
import json
from pathlib import Path

import numpy as np

ROOT=Path(__file__).resolve().parents[1]
QPS=(0,16,32,48,63)


def ratio_and_ci(rows, rng, boot=5000):
    """Bootstrap whole sequences; preserve their QP observations as a cluster."""
    if not rows:
        raise ValueError('cannot summarize an empty cohort')
    seqs=sorted({r['sequence'] for r in rows})
    per={s:[r for r in rows if r['sequence']==s] for s in seqs}
    stock=np.array([sum(r['stock_median_ms'] for r in per[s]) for s in seqs])
    fast=np.array([sum(r['fast_median_ms'] for r in per[s]) for s in seqs])
    assert np.all(stock>0) and np.all(fast>0)
    draw=rng.integers(0,len(seqs),(boot,len(seqs)))
    ratios=stock[draw].sum(1)/fast[draw].sum(1)
    return {'n_sequences':len(seqs),'ratio':float(stock.sum()/fast.sum()),
            'ci95':[float(x) for x in np.quantile(ratios,[.025,.975])],
            'mean_stock_ms':float(stock.sum()/len(rows)),
            'mean_fast_ms':float(fast.sum()/len(rows))}


def summarize(rows, *, complete=True):
    if not rows:
        raise ValueError('cannot summarize an empty cohort')
    keys=[(r['sequence'],r['qp']) for r in rows]
    if len(set(keys))!=len(keys):
        raise ValueError('duplicate sequence/QP row')
    seqs={s for s,_ in keys}
    if complete and (len(rows)!=265 or len(seqs)!=53 or
                     any({r['qp'] for r in rows if r['sequence']==s}!=set(QPS) for s in seqs)):
        raise ValueError('expected all 53 sequences at all five QPs')
    rng=np.random.default_rng(20261002)
    overall=ratio_and_ci(rows,rng)
    if all('stock_wall_median_ms' in r and 'fast_wall_median_ms' in r for r in rows):
        wall_rows=[dict(r,stock_median_ms=r['stock_wall_median_ms'],
                        fast_median_ms=r['fast_wall_median_ms']) for r in rows]
        overall_wall=ratio_and_ci(wall_rows,np.random.default_rng(20261002))
    else:
        overall_wall=None
    by_qp={str(q):ratio_and_ci([r for r in rows if r['qp']==q],rng)
           for q in QPS if any(r['qp']==q for r in rows)}
    return {'scope':'decoder synthesis only, isolated GPU if preflight was not overridden',
            'n_frame_qp':len(rows),'n_sequences':len(seqs),
            'dense_fallbacks':sum(bool(r['dense_fallback']) for r in rows),
            'max_abs_output_error':max(r['max_abs_output_error'] for r in rows),
            'max_abs_delta_yuv611_db':max(abs(r['delta_yuv611_db']) for r in rows),
            'overall':overall,'overall_wall':overall_wall,'by_qp':by_qp,
            'bootstrap':'5000 sequence-cluster resamples, seed 20261002; fixed checkpoint and maps'}


def main():
    p=argparse.ArgumentParser()
    p.add_argument('--input',default='results/triton_ctc_cohort.jsonl')
    p.add_argument('--out',default='results/triton_ctc_cohort_summary.json')
    p.add_argument('--allow-partial',action='store_true')
    a=p.parse_args()
    rows=[json.loads(line) for line in (ROOT/a.input).read_text().splitlines() if line.strip()]
    result=summarize(rows,complete=not a.allow_partial)
    out=ROOT/a.out
    out.write_text(json.dumps(result,indent=2)+'\n')
    print('overall',result['overall'])
    for q,val in result['by_qp'].items():
        print('QP',q,val)
    print('saved',out)


if __name__=='__main__':main()
