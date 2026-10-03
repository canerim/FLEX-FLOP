"""Compact public audit figure for nine paired idle-GPU workloads."""
from __future__ import annotations

import json
from pathlib import Path

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np

HERE = Path(__file__).resolve().parent
DATA = HERE/'results/cohort_20261003/cohort_summary.json'
OUT = HERE/'results/cohort_20261003/speedup_audit'


def main():
    data = json.loads(DATA.read_text())
    rows = data['cases']
    labels = [('BQMall','480p'), ('FourPeople','720p'), ('videoSRC05','1080p')]
    qps = (16,32,48)
    colors = {16:'#cf6f54',32:'#358d92',48:'#244d67'}
    plt.rcParams.update({'font.family':'DejaVu Sans','font.size':8,'axes.linewidth':.7,
                         'axes.spines.top':False,'axes.spines.right':False,'pdf.fonttype':42})
    fig,(ax,bx)=plt.subplots(1,2,figsize=(7.1,2.8),gridspec_kw={'width_ratios':[1.1,1]})
    xs=np.arange(3)
    for qp in qps:
        values=[next(r['paired_median_speedup'] for r in rows if r['sequence'].startswith(name) and r['qp']==qp)
                for name,_ in labels]
        ax.plot(xs,values,lw=1.5,marker='o',ms=5,color=colors[qp],
                markeredgecolor='white',markeredgewidth=.6,label=f'QP {qp}')
    ax.axhline(1,color='#8998a1',lw=.8,ls='--')
    ax.set(xticks=xs,xticklabels=[label for _,label in labels],ylim=(1.8,3.2),
           ylabel='Released / e15+Triton decoder time (×)')
    ax.grid(axis='y',color='#dbe5e8',lw=.5)
    ax.legend(frameon=False,ncol=3,loc='lower right',fontsize=7.5)
    ax.set_title('a   Speedup across workload',loc='left',fontweight='bold',fontsize=9,pad=10)
    width=.34
    release=[next(r['released_median_ms'] for r in rows if r['sequence'].startswith(name) and r['qp']==32) for name,_ in labels]
    early=[next(r['e15_triton_median_ms'] for r in rows if r['sequence'].startswith(name) and r['qp']==32) for name,_ in labels]
    bx.bar(xs-width/2,release,width,color='#bf6752',label='Released D12')
    bx.bar(xs+width/2,early,width,color='#2d8290',label='e15 + Triton')
    bx.set(xticks=xs,xticklabels=[label for _,label in labels],ylabel='Decoder wall time at QP32 (ms)')
    bx.grid(axis='y',color='#dbe5e8',lw=.5)
    bx.set_axisbelow(True)
    bx.legend(frameon=False,loc='upper left',fontsize=7.5)
    bx.set_title('b   Paired median latency',loc='left',fontweight='bold',fontsize=9,pad=10)
    fig.subplots_adjust(left=.10,right=.98,bottom=.24,top=.82,wspace=.37)
    fig.text(.015,.008,'Three first CTC frames · 20 randomized paired blocks/case · idle A6000 · shared latent · decoder synthesis and host tile planning only.',
             fontsize=6.4,color='#687d86')
    for ext in ('pdf','png'):
        fig.savefig(OUT.with_suffix('.'+ext),dpi=300,bbox_inches='tight',facecolor='white')
    plt.close(fig)
    print(OUT)


if __name__=='__main__':
    main()
