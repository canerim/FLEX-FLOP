"""Separate scalar-routing loss from the all-deep tiled reconstruction floor.

Reads precomputed calibration candidates only. No model load or GPU access.
"""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path


def sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--manifest', type=Path, required=True)
    parser.add_argument('--policy', type=Path, required=True)
    parser.add_argument('--cal-dir', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    import numpy as np
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt

    manifest = json.loads(args.manifest.read_text())
    policy = json.loads(args.policy.read_text())
    if policy['manifest_sha256'] != sha(args.manifest):
        raise RuntimeError('Policy does not belong to calibration manifest')
    qps = manifest['qps']
    cases = []
    for item in manifest['rows']['calibration']:
        for qp in qps:
            path = args.cal_dir/f'{Path(item["image"]).stem}_qp{qp}.json'
            row = json.loads(path.read_text())
            if (row['manifest_sha256'] != sha(args.manifest) or
                row['source_sha256'] != item['source_sha256'] or
                row['image'] != item['image'] or row['qp'] != qp):
                raise RuntimeError(f'Case provenance mismatch: {path}')
            full = [c for c in row['candidates']
                    if all(exit_index == 5 for exit_index in c['exit_map'])]
            selected = [c for c in row['candidates']
                        if c['beta'] == policy['beta'][str(qp)]]
            if not full or len(selected) != 1:
                raise RuntimeError(f'Missing all-deep or selected candidate: {path}')
            floor = full[0]
            selected = selected[0]
            feasible = [c for c in row['candidates'] if c['delta444_db'] <= .1]
            oracle = max(feasible,key=lambda c:c['mac_saved_pct']) if feasible else floor
            cases.append({'image':item['image'],'qp':qp,
                          'all_deep_floor_delta444_db':floor['delta444_db'],
                          'all_deep_mac_saved_pct':floor['mac_saved_pct'],
                          'selected_delta444_db':selected['delta444_db'],
                          'selected_mac_saved_pct':selected['mac_saved_pct'],
                          'has_feasible_scalar_beta':bool(feasible),
                          'source_informed_candidate_oracle_delta444_db':oracle['delta444_db'],
                          'source_informed_candidate_oracle_mac_saved_pct':oracle['mac_saved_pct']})
    if len(cases) != 120:
        raise RuntimeError('Expected 24 calibration images at five QPs')

    q90 = {}
    for qp in qps:
        subset = [r for r in cases if r['qp']==qp]
        source_records = [json.loads((args.cal_dir/f'{Path(r["image"]).stem}_qp{qp}.json').read_text())
                          for r in subset]
        choices = []
        for i,beta in enumerate(manifest['candidates'][str(qp)]):
            candidates = [r['candidates'][i] for r in source_records]
            losses = np.asarray([r['delta444_db'] for r in candidates])
            saving = float(np.mean([r['mac_saved_pct'] for r in candidates]))
            p90 = float(np.quantile(losses,.9))
            if p90 <= .1:
                choices.append((saving,-float(losses.mean()),beta,p90,int((losses>.1).sum())))
        if not choices:
            raise RuntimeError(f'No Q90-feasible beta at QP{qp}')
        best = max(choices)
        q90[str(qp)]={'beta':best[2], 'calibration_mac_saved_pct':best[0],
                      'calibration_mean_delta444_db':-best[1],
                      'calibration_q90_delta444_db':best[3],
                      'calibration_over_0p1_cases':best[4]}
    summary={'schema':1,'manifest_sha256':sha(args.manifest),
             'primary_policy_sha256':sha(args.policy),'cases':len(cases),
             'all_deep_floor_over_0p1_count':sum(r['all_deep_floor_delta444_db']>.1 for r in cases),
             'primary_over_0p1_count':sum(r['selected_delta444_db']>.1 for r in cases),
             'no_feasible_beta_cases':[(r['image'],r['qp']) for r in cases
                                       if not r['has_feasible_scalar_beta']],
             'primary_mean_mac_saved_pct':float(np.mean([r['selected_mac_saved_pct'] for r in cases])),
             'source_informed_candidate_oracle_mean_mac_saved_pct':float(np.mean(
                 [r['source_informed_candidate_oracle_mac_saved_pct'] for r in cases])),
             'source_informed_candidate_oracle_over_0p1_count':sum(
                 r['source_informed_candidate_oracle_delta444_db']>.1 for r in cases),
             'per_qp_q90_beta_on_calibration':q90,
             'q90_rule_mean_mac_saved_pct_on_calibration':float(np.mean(
                 [x['calibration_mac_saved_pct'] for x in q90.values()])),
             'metric_note':'Delta444 relative to e15 full frame; all-deep tiled floor includes seam repair. Candidate oracle consults source quality and is not deployable. Q90 rule is calibration-only exploratory.',
             'rows':cases}
    args.output.mkdir(parents=True,exist_ok=True)
    (args.output/'beta_quality_floor.json').write_text(json.dumps(summary,indent=2)+'\n')

    plt.rcParams.update({'font.family':'DejaVu Sans','font.size':8,
                         'axes.linewidth':.7,'pdf.fonttype':42})
    colors={0:'#327D86',16:'#5A9A72',32:'#BCA050',48:'#D48848',63:'#A54361'}
    fig,(ax,bx)=plt.subplots(1,2,figsize=(7.15,3.2),
                              gridspec_kw={'width_ratios':[1.08,1]},
                              constrained_layout=True)
    for qp in qps:
        subset=[r for r in cases if r['qp']==qp]
        ax.scatter([r['all_deep_floor_delta444_db'] for r in subset],
                   [r['selected_delta444_db'] for r in subset],
                   s=15,alpha=.62,color=colors[qp],edgecolor='none',label=f'QP{qp}')
    ax.axhline(.1,color='#6B8571',lw=.8,ls=(0,(3,2)))
    ax.axvline(.1,color='#6B8571',lw=.8,ls=(0,(3,2)))
    ax.annotate('0828 · QP0',(0.1148300636332754,0.7918083678455878),
                xytext=(-76,-12),textcoords='offset points',fontsize=7,
                color='#963E5B',arrowprops={'arrowstyle':'-','lw':.6,'color':'#963E5B'})
    ax.set(xlabel='All-deep tiled floor, Δ444 (dB)',
           ylabel='Primary policy loss, Δ444 (dB)',xlim=(-.02,.19),ylim=(-.04,.86))
    ax.set_title('a  Scalar price cannot remove the tile floor',
                 loc='left',fontweight='bold',pad=5)
    ax.legend(frameon=False,ncol=3,fontsize=6.7,loc='upper left',
              handletextpad=.2,columnspacing=.7)
    example=[next(r for r in cases if r['image']=='0828.png' and r['qp']==q)
             for q in qps]
    xx=np.arange(5); width=.34
    bx.bar(xx-width/2,[r['all_deep_floor_delta444_db'] for r in example],
           width,color='#98A8AC',label='All-deep tiles')
    bx.bar(xx+width/2,[r['selected_delta444_db'] for r in example],
           width,color='#A54361',label='Primary β')
    bx.axhline(.1,color='#6B8571',lw=.8,ls=(0,(3,2)))
    bx.set_xticks(xx,[str(q) for q in qps]); bx.set(xlabel='QP',ylabel='Δ444 (dB)')
    bx.set_title('b  No scalar β meets 0.10 dB: image 0828',
                 loc='left',fontweight='bold',pad=5)
    bx.legend(frameon=False,fontsize=7,loc='upper right')
    for axis in (ax,bx):
        axis.spines[['top','right']].set_visible(False)
        axis.tick_params(direction='out',length=3,width=.7)
        axis.grid(axis='y',color='#E8ECED',lw=.6,zorder=0)
    for ext in ('pdf','png'):
        fig.savefig(args.output/f'beta_quality_floor.{ext}',dpi=220,bbox_inches='tight')
    plt.close(fig)
    print(json.dumps({k:v for k,v in summary.items() if k!='rows'},indent=2))


if __name__=='__main__':
    main()
