"""Vector figure and image-cluster intervals for the ideal-context area audit."""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--input',type=Path,required=True)
    p.add_argument('--output-dir',type=Path,required=True)
    args=p.parse_args()
    d=json.loads(args.input.read_text())
    args.output_dir.mkdir(parents=True,exist_ok=True)
    qps=[0,16,32,48,63]
    rng=np.random.default_rng(20261004)
    summaries={}
    for split in ('calibration','validation'):
        rows=d['rows'][split]
        images=sorted(set(r['image'] for r in rows))
        if len(images)!=24:
            raise RuntimeError('Expected 24 images per split')
        lookup={(r['image'],r['qp']):r['ideal_shared_context_mac_saved_pct'] for r in rows}
        matrix=np.asarray([[lookup[(image,qp)] for qp in qps] for image in images])
        draws=rng.integers(0,len(images),size=(4000,len(images)))
        boot=matrix[draws].mean(axis=1)
        summaries[split]={'qps':qps,'mean_pct':matrix.mean(axis=0).tolist(),
                          'image_cluster_bootstrap_95pct':np.quantile(boot,[.025,.975],axis=0).T.tolist()}
    (args.output_dir/'ideal_context_cohort_intervals.json').write_text(json.dumps(summaries,indent=2)+'\n')

    plt.rcParams.update({'font.family':'DejaVu Sans','font.size':8,'axes.linewidth':.7,
                         'pdf.fonttype':42})
    fig,(ax,bx)=plt.subplots(1,2,figsize=(7.2,2.75),constrained_layout=True,
                             gridspec_kw={'width_ratios':[1.1,1]})
    colors={'calibration':'#B46B4A','validation':'#217B84'}
    for split in ('calibration','validation'):
        s=summaries[split]; lo=np.asarray(s['image_cluster_bootstrap_95pct'])
        yy=np.asarray(s['mean_pct'])
        ax.plot(qps,yy,marker='o',lw=1.55,ms=4.2,color=colors[split],
                label='Calibration' if split=='calibration' else 'Disjoint validation')
        ax.fill_between(qps,lo[:,0],lo[:,1],color=colors[split],alpha=.12,lw=0)
    ax.axhline(0,color='#506169',lw=.75)
    ax.set(xticks=qps,xlabel='QP',ylabel='Ideal synthesis MAC saving (%)',ylim=(0,43))
    ax.set_title('a  Perfect context sharing, by rate',loc='left',fontweight='bold')
    ax.legend(frameon=False,fontsize=6.8,loc='upper right')
    val=d['rows']['validation']
    within=np.asarray([r['ideal_shared_context_mac_saved_pct'] for r in val if r['deployed_delta444_db']<=.1])
    over=np.asarray([r['ideal_shared_context_mac_saved_pct'] for r in val if r['deployed_delta444_db']>.1])
    vp=bx.violinplot([within,over],positions=[0,1],widths=.55,
                     showmeans=False,showmedians=False,showextrema=False)
    for body,color in zip(vp['bodies'],['#217B84','#B46B4A']):
        body.set_facecolor(color);body.set_edgecolor('none');body.set_alpha(.35)
    for i,values in enumerate((within,over)):
        x=np.full(len(values),i)+np.random.default_rng(1234+i).uniform(-.16,.16,len(values))
        bx.scatter(x,values,s=9,color=['#217B84','#B46B4A'][i],alpha=.48,edgecolor='none')
        bx.plot([i-.16,i+.16],[np.median(values)]*2,color='#22343C',lw=1.6)
    bx.set(xticks=[0,1],xticklabels=[f'Within 0.1 dB\n(n={len(within)})',
                                      f'Over 0.1 dB\n(n={len(over)})'],
           ylabel='Ideal synthesis MAC saving (%)',ylim=(-4,46),xlim=(-.5,1.5))
    bx.set_title('b  Grouped by deployed loss',loc='left',fontweight='bold')
    bx.text(.5,.98,'Quality is from zero-halo decode',transform=bx.transAxes,
            ha='center',va='top',fontsize=6.3,color='#5B6B71')
    for axis in (ax,bx):
        axis.spines[['top','right']].set_visible(False)
        axis.grid(axis='y',color='#E5E9EB',lw=.6,zorder=0)
        axis.tick_params(direction='out',length=3,width=.7)
    for ext in ('pdf','png'):
        fig.savefig(args.output_dir/f'ideal_context_cohort.{ext}',dpi=220,bbox_inches='tight')
    plt.close(fig)
    print(json.dumps(summaries,indent=2))


if __name__=='__main__':
    main()
