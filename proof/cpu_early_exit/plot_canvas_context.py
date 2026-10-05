"""Publication-scale vector figure for paired active-context diagnostics."""
from __future__ import annotations

import hashlib
import json
from pathlib import Path

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np

HERE=Path(__file__).resolve().parent
ROOT=HERE.parents[1]
SOURCE=HERE/'results/kodak_canvas_arms_comparison_20261005.json'
COST=HERE/'results/kodak_active_canvas_cost_20261005.json'
OUT=ROOT/'docs/figures/active-canvas-20261005'


def sha(path:Path)->str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main()->None:
    data=json.loads(SOURCE.read_text())
    cost=json.loads(COST.read_text())
    assert len(data['rows'])==120 and data['over_0p1_delta444']['deployed']>=0
    assert len(cost['rows'])==120
    plt.rcParams.update({
        'font.family':'DejaVu Sans','font.size':8,'axes.titlesize':9,
        'axes.labelsize':8,'xtick.labelsize':7.5,'ytick.labelsize':7.5,
        'svg.fonttype':'none','pdf.fonttype':42,'axes.linewidth':0.7,
        'savefig.pad_inches':0.03,
    })
    colors={'stale':'#CA6050','active_zero':'#197E8D','active_replicate':'#183B5B'}
    names={'stale':'Stale context','active_zero':'Active / zero',
           'active_replicate':'Active / replicate'}
    markers={'stale':'s','active_zero':'o','active_replicate':'D'}
    fig,axes=plt.subplots(1,2,figsize=(7.18,2.64))
    fig.patch.set_facecolor('white')
    for ax in axes:
        ax.set_facecolor('white')
        ax.axhline(0,color='#77818A',lw=.8,zorder=0)
        ax.grid(axis='y',color='#E9EDF0',lw=.55,zorder=0)
        ax.spines[['top','right']].set_visible(False)
        ax.spines[['left','bottom']].set_color('#667582')
        ax.tick_params(length=2.5,color='#667582')

    ax=axes[0]
    offsets={'active_zero':-.10,'active_replicate':.10}
    for arm in ('active_zero','active_replicate'):
        stats=[data['per_qp'][str(q)][f'{arm}_with_repair_gain_db']
               for q in (0,16,32,48,63)]
        x=np.arange(5)+offsets[arm]
        y=np.asarray([s['mean_gain_db'] for s in stats])
        lo=np.asarray([s['image_cluster_ci95_db'][0] for s in stats])
        hi=np.asarray([s['image_cluster_ci95_db'][1] for s in stats])
        ax.errorbar(x,y,yerr=[y-lo,hi-y],fmt=markers[arm],
                    ms=4.7,elinewidth=1.15,capsize=2.2,lw=1.2,
                    color=colors[arm],mfc='white' if arm=='active_zero' else colors[arm],
                    mec=colors[arm],label=names[arm],zorder=3)
    ax.set_xticks(range(5),['0','16','32','48','63'])
    ax.set_xlim(-.45,4.45)
    ax.set_xlabel('Quality index (QP)')
    ax.set_ylabel('Gain over deployed  $\Delta$444 (dB)')
    ax.set_title('a   Active context recovers quality',loc='left',pad=8,
                 color='#183B5B',fontweight='bold')
    ax.legend(frameon=False,ncol=1,loc='upper left',fontsize=7.2,
              handletextpad=.35,labelspacing=.32)

    ax=axes[1]
    stats=[data['per_qp'][str(q)]['stale_with_repair_gain_db']
           for q in (0,16,32,48,63)]
    x=np.arange(5)
    y=np.asarray([s['mean_gain_db'] for s in stats])
    lo=np.asarray([s['image_cluster_ci95_db'][0] for s in stats])
    hi=np.asarray([s['image_cluster_ci95_db'][1] for s in stats])
    ax.errorbar(x,y,yerr=[y-lo,hi-y],fmt=markers['stale'],
                ms=5,elinewidth=1.15,capsize=2.5,lw=1.2,
                color=colors['stale'],mfc=colors['stale'],
                mec=colors['stale'],zorder=3)
    ax.set_xticks(range(5),['0','16','32','48','63'])
    ax.set_xlim(-.45,4.45)
    ax.set_xlabel('Quality index (QP)')
    ax.set_ylabel('Gain over deployed (dB)')
    ax.set_title('b   Stale context fails at high QP',loc='left',pad=8,
                 color='#183B5B',fontweight='bold')

    # The QP63 stale-context tail is much larger than the active-context
    # gains. Label independent scales rather than flattening the small gain.
    for ax in axes: ax.margins(y=.13)
    fig.subplots_adjust(left=.105,right=.985,bottom=.235,top=.88,wspace=.29)
    OUT.mkdir(parents=True,exist_ok=True)
    for ext in ('pdf','svg','png'):
        fig.savefig(OUT/f'canvas_context.{ext}',dpi=300,facecolor='white')
    plt.close(fig)
    svg=OUT/'canvas_context.svg'
    svg.write_text('\n'.join(line.rstrip() for line in svg.read_text().splitlines())+'\n')
    evidence={'data_sha256':sha(SOURCE),'cost_sha256':sha(COST),
              'script_sha256':sha(Path(__file__)),
              'canvas_mm':[182.37,67.06],
              'n_cases':len(data['rows']),
              'scope':'Paired bitstream quality with matched seam repair, no runtime claim; panel a resolves active gains over deployed and panel b exposes stale QP dependence; panels use independently labelled vertical scales',
              'artifacts':{f'canvas_context.{ext}':sha(OUT/f'canvas_context.{ext}')
                           for ext in ('pdf','svg','png')}}
    (OUT/'figure_evidence.json').write_text(json.dumps(evidence,indent=2)+'\n')
    print(json.dumps(evidence,indent=2))


if __name__=='__main__':main()
