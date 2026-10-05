"""Recount archived CTC early-exit maps against released D12 convolution MACs.

No model execution. This complements, rather than rewrites, the historical
normalized cost model in the frozen source-informed budget archive.
"""
from __future__ import annotations

from collections import defaultdict
import hashlib
import json
from pathlib import Path
import sys

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT/'proof/early_exit_vs_released'))
from mac_latency_audit import case_macs

SOURCE = ROOT/'cvpr2027/data/bd_rate_20261002/eval_rules_ctc_e15.json'
OUTPUT = ROOT/'docs/research/2026-10-05-ctc-exact-mac'
BUDGETS = ('0.05','0.1','0.15','0.2','0.3','0.5')
POLICIES = ('uniform','dither','router','oracle')


def summary(values, rng):
    arr = np.asarray(values, dtype=float)
    sampled = arr[rng.integers(0,len(arr),size=(5000,len(arr)))].mean(axis=1)
    return {'mean':float(arr.mean()),
            'ci95':[float(v) for v in np.quantile(sampled,(.025,.975))]}


def main():
    archive = json.loads(SOURCE.read_text())
    assert len(archive['rows']) == 265
    grouped = defaultdict(list)
    per_case = []
    for row in archive['rows']:
        grouped[row['seq']].append(row)
        h,w = row['hw']
        gh,gw = row['grid']
        assert gh*256 >= h and gw*256 >= w
        assert gh*gw == len(next(v['map'] for p in POLICIES for v in row['rules'][p].values() if v is not None))
        for budget in BUDGETS:
            for p in POLICIES:
                old = row['rules'][p][budget]
                if old is None:
                    continue
                counts = [old['map'].count(k) for k in range(6)]
                mac = case_macs({'padded_shape':[gh*256,gw*256],
                                 'tile_counts':counts})
                exact = 100*mac['e15_routed_conv_mac_saving_fraction']
                per_case.append({'seq':row['seq'],'qp':row['qp'],
                                 'budget':budget,'policy':p,
                                 'exact_conv_saving_pct':exact,
                                 'legacy_model_saving_pct':old['saving'],
                                 'delta_pct_points':exact-old['saving'],
                                 'released_conv_gmac':mac['released_conv_macs']/1e9,
                                 'routed_conv_gmac':mac['e15_routed_conv_macs']/1e9})
    lookup = {(r['seq'],r['qp'],r['budget'],r['policy']):r for r in per_case}
    assert len(lookup) == len(per_case)
    complete = {budget:sorted(seq for seq, rows in grouped.items()
                              if len(rows)==5 and all((seq,r['qp'],budget,p) in lookup
                                                     for r in rows for p in POLICIES))
                for budget in BUDGETS}
    fixed = sorted(set.intersection(*(set(v) for v in complete.values())))
    assert [len(complete[b]) for b in BUDGETS] == [25,51,53,53,53,53]
    assert len(fixed) == 25
    rng = np.random.default_rng(20261005)
    results = []
    for cohort in ('fixed25','budget_complete'):
        for budget in BUDGETS:
            seqs = fixed if cohort=='fixed25' else complete[budget]
            by_policy = {}
            for p in POLICIES:
                exact = []
                old = []
                for seq in seqs:
                    qps = sorted(r['qp'] for r in grouped[seq])
                    exact.append(np.mean([lookup[(seq,qp,budget,p)]['exact_conv_saving_pct'] for qp in qps]))
                    old.append(np.mean([lookup[(seq,qp,budget,p)]['legacy_model_saving_pct'] for qp in qps]))
                by_policy[p] = {'exact':exact,'old':old}
            paired = np.asarray(by_policy['router']['exact'])-np.asarray(by_policy['dither']['exact'])
            old_paired = np.asarray(by_policy['router']['old'])-np.asarray(by_policy['dither']['old'])
            results.append({'cohort':cohort,'budget':budget,'n_sequences':len(seqs),
                            'policy':{p:{'exact_conv_saving_pct':summary(by_policy[p]['exact'],rng),
                                         'legacy_model_saving_pct':summary(by_policy[p]['old'],rng)}
                                      for p in POLICIES},
                            'router_minus_dither_exact_pct_points':summary(paired,rng),
                            'router_minus_dither_legacy_pct_points':summary(old_paired,rng)})
    result = {'schema':1,
              'scope':'Archived source-calibrated CTC maps, exact convolution count vs released D12 full synthesis; no new quality or latency measurement',
              'input_sha256':hashlib.sha256(SOURCE.read_bytes()).hexdigest(),
              'accounting':'Padded frame, convolution MACs only; routed adapters and seam repair included; router, entropy, memory and signalling excluded',
              'fixed_cohort_sequences':fixed,'results':results,'per_case':per_case}
    OUTPUT.mkdir(parents=True,exist_ok=True)
    (OUTPUT/'analysis.json').write_text(json.dumps(result,indent=2)+'\n')
    compact={'schema':1,'input_sha256':result['input_sha256'],
             'accounting':result['accounting'],
             'rows':[{k:r[k] for k in ('seq','qp','budget','policy','exact_conv_saving_pct')}
                     for r in per_case]}
    (OUTPUT/'exact_mac_by_case.json').write_text(json.dumps(compact,separators=(',',':'))+'\n')
    lines=['# Exact conv-MAC check of the CTC early-exit budget archive','',
           'The frozen source-calibrated 53-sequence × five-QP CTC archive is recounted against the released D12 full-synthesis convolution count on each **padded** frame. The routed count includes exit adapters and seam repair. No model is run and neither quality, mode-map bytes nor latency is remeasured. This is an arithmetic correction to the prior normalized cost model. The 25-sequence fixed cohort is complete at every budget; the budget-complete cohort has 25/51/53/53/53/53 sequences.','',
           '| Nominal Δ444 target | n fixed | Router exact saving | Dither exact saving | Router − dither exact | Router − dither old model |',
           '|---:|---:|---:|---:|---:|---:|']
    for r in results[:6]:
        rv=r['policy']['router']['exact_conv_saving_pct']['mean']
        dv=r['policy']['dither']['exact_conv_saving_pct']['mean']
        diff=r['router_minus_dither_exact_pct_points']
        old=r['router_minus_dither_legacy_pct_points']['mean']
        lines.append(f'| {r["budget"]} dB | {r["n_sequences"]} | {rv:.2f}% | {dv:.2f}% | {diff["mean"]:+.2f} pp [{diff["ci95"][0]:+.2f}, {diff["ci95"][1]:+.2f}] | {old:+.2f} pp |')
    lines.extend(['','The corresponding complete-cohort result at 0.10 dB is '
                  f'**{results[7]["policy"]["router"]["exact_conv_saving_pct"]["mean"]:.2f}%** exact conv saving for router '
                  f'and **{results[7]["policy"]["dither"]["exact_conv_saving_pct"]["mean"]:.2f}%** for dither '
                  f'({results[7]["router_minus_dither_exact_pct_points"]["mean"]:+.2f} pp paired). '
                  'The archived output PSNR and BD-rate proxy remain unchanged; the table only replaces the arithmetic denominator and block accounting. '
                  'The source archive was calibrated using source quality, so this is not a deployable policy result. '
                  'The 95% intervals resample complete sequences and retain their five QPs.','',
                  'Reproduce with `python3 proof/cpu_early_exit/audit_ctc_exact_conv_mac.py`. '
                  'The [analysis JSON](analysis.json) records every case and source SHA256.',''])
    (OUTPUT/'REPORT.md').write_text('\n'.join(lines))
    print('\n'.join(lines[4:12]))


if __name__=='__main__':
    main()
