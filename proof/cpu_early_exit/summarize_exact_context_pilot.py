"""Summarize a completed, fixed-prefix exact-context quality pilot."""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--input', type=Path, required=True)
    p.add_argument('--out-dir', type=Path, required=True)
    args = p.parse_args()
    data = json.loads(args.input.read_text())
    rows = data['rows']
    expected = [(image, qp) for image in data['images'] for qp in data['qps']]
    if sorted((r['image'], r['qp']) for r in rows) != sorted(expected):
        raise RuntimeError('Pilot is incomplete or has duplicate image/QP pairs')
    def stats(group):
        d = np.array([r['deployed_delta444_db'] for r in group])
        e = np.array([r['exact_context_delta444_db'] for r in group])
        s = np.array([r['ideal_shared_saving_pct'] for r in group])
        return {'cases':len(group), 'mean_deployed_delta444_db':float(d.mean()),
                'mean_exact_context_delta444_db':float(e.mean()),
                'mean_context_improvement_db':float((d-e).mean()),
                'median_context_improvement_db':float(np.median(d-e)),
                'context_worsens_count':int((e>d+1e-5).sum()),
                'deployed_over_0p1_count':int((d>.1).sum()),
                'exact_over_0p1_count':int((e>.1).sum()),
                'rescued_to_within_0p1_count':int(((d>.1)&(e<=.1)).sum()),
                'newly_over_0p1_count':int(((d<=.1)&(e>.1)).sum()),
                'persistent_over_0p1_count':int(((d>.1)&(e>.1)).sum()),
                'mean_ideal_shared_saving_pct':float(s.mean()),
                'exact_within_0p1_and_ideal_saving_ge10_count':int(((e<=.1)&(s>=10)).sum())}
    summary = {'schema':1,'scope':'Descriptive fixed-prefix pilot only; no population CI or latency',
               'input':str(args.input),'images':data['images'],
               'all':stats(rows),
               'per_qp':{str(q):stats([r for r in rows if r['qp']==q]) for q in data['qps']},
               'per_image':{image:stats([r for r in rows if r['image']==image]) for image in data['images']}}
    if len(data['images']) == 24:
        # Paired image-cluster bootstrap: QPs from the same image remain together.
        image_effects=np.array([np.mean([r['deployed_delta444_db']-r['exact_context_delta444_db']
                                         for r in rows if r['image']==image]) for image in data['images']])
        rng=np.random.default_rng(20261004)
        draws=rng.integers(0,len(image_effects),size=(10000,len(image_effects)))
        ci=np.quantile(image_effects[draws].mean(axis=1),[.025,.975])
        split='calibration' if 'calibration' in data['scope'] else 'validation'
        summary['scope']=f'Full 24-image DIV2K {split}, fixed policy; paired image-cluster bootstrap for mean context effect; no latency'
        summary['all']['mean_context_improvement_cluster_bootstrap_95ci_db']=[float(v) for v in ci]
    args.out_dir.mkdir(parents=True,exist_ok=True)
    (args.out_dir/'exact_context_pilot_summary.json').write_text(json.dumps(summary,indent=2)+'\n')

    plt.rcParams.update({'font.family':'DejaVu Sans','font.size':8,
                         'pdf.fonttype':42,'axes.spines.top':False,'axes.spines.right':False})
    colors={0:'#6D477E',16:'#316C91',32:'#278D8D',48:'#B88438',63:'#B95654'}
    fig,(ax,ay)=plt.subplots(1,2,figsize=(7.15,2.8),gridspec_kw={'width_ratios':[1,1.05]})
    for q in data['qps']:
        group=[r for r in rows if r['qp']==q]
        d=np.array([r['deployed_delta444_db'] for r in group])
        e=np.array([r['exact_context_delta444_db'] for r in group])
        s=np.array([r['ideal_shared_saving_pct'] for r in group])
        ax.scatter(d,e,s=30,color=colors[q],edgecolor='white',linewidth=.5,label=f'QP {q}',zorder=3)
        ay.scatter(s,e,s=30,color=colors[q],edgecolor='white',linewidth=.5,zorder=3)
    bound=max(.55,max(max(r['deployed_delta444_db'],r['exact_context_delta444_db']) for r in rows)*1.04)
    ax.plot([0,bound],[0,bound],color='#8A9298',lw=.85,ls=(0,(3,2)))
    ax.axhline(.1,color='#A65050',lw=.8,ls=(0,(2,2)))
    ax.set(xlim=(-.025,bound),ylim=(-.025,bound),xlabel='Deployed loss vs. full-frame e15 (dB)',
           ylabel='Exact-context loss vs. full-frame e15 (dB)')
    ax.text(.02,.97,'a  Context recovery',transform=ax.transAxes,va='top',fontweight='bold')
    ax.legend(frameon=False,loc='lower right',ncol=1,fontsize=6.5,
              labelspacing=.25,handletextpad=.25,borderpad=.2)
    ay.axhline(.1,color='#A65050',lw=.8,ls=(0,(2,2)))
    ay.axvline(10,color='#8A9298',lw=.8,ls=(0,(2,2)))
    ay.set(xlabel='Ideal shared-context synthesis MAC saving (%)',
           ylabel='Exact-context loss vs. full-frame e15 (dB)',ylim=(-.025,bound))
    ay.text(.02,.97,'b  Quality vs. ideal arithmetic',transform=ay.transAxes,va='top',fontweight='bold')
    fig.tight_layout(w_pad=1.7)
    fig.savefig(args.out_dir/'exact_context_pilot.pdf',bbox_inches='tight')
    fig.savefig(args.out_dir/'exact_context_pilot.png',dpi=220,bbox_inches='tight')
    plt.close(fig)
    print(json.dumps(summary['all'],indent=2))


if __name__=='__main__':
    main()
