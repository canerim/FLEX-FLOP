"""Vector workload figure; these are router maps, not measured latency."""
from __future__ import annotations

import json
from pathlib import Path

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np

ROOT=Path(__file__).resolve().parents[1]
data=json.loads((ROOT/'results/triton_ctc_map_workload.json').read_text())['by_qp']
qps=[0,16,32,48,63]
fractions=np.array([data[str(q)]['exit_fraction'][2:6] for q in qps])
means=np.array([data[str(q)]['mean_retained_blocks'] for q in qps])
assert np.allclose(fractions.sum(axis=1),1)

plt.rcParams.update({'font.family':'DejaVu Sans','font.size':8,
                     'pdf.fonttype':42,'axes.spines.top':False,
                     'axes.spines.right':False})
fig=plt.figure(figsize=(6.8,3.1),facecolor='#FAFBF9')
ax=fig.add_axes([.105,.24,.64,.54],facecolor='#FAFBF9')
colors=['#6AA4A3','#D6A45E','#B86E7A','#485F85']
names=['D6','D8','D10','D12']
bottom=np.zeros(5)
for i,(name,color) in enumerate(zip(names,colors)):
    bars=ax.bar(qps,fractions[:,i]*100,bottom=bottom*100,width=10.5,
                color=color,label=name,edgecolor='#FAFBF9',linewidth=.6)
    for j,bar in enumerate(bars):
        if fractions[j,i]>.09:
            ax.text(bar.get_x()+bar.get_width()/2,
                    (bottom[j]+fractions[j,i]/2)*100,
                    f'{fractions[j,i]*100:.0f}%',ha='center',va='center',
                    fontsize=7.1,fontweight='bold',
                    color='#173039' if i<2 else '#FFFFFF')
    bottom+=fractions[:,i]
ax.set(xlim=(-8,71),ylim=(0,100),xticks=qps,
       xlabel='Quantization parameter (QP)',
       ylabel='Share of routed tiles (%)')
ax.yaxis.set_major_locator(plt.MultipleLocator(25))
ax.grid(axis='y',color='#DCE5E3',linewidth=.6)
ax.set_axisbelow(True)
ax.legend(loc='lower left',bbox_to_anchor=(0,1.02),ncol=4,
          frameon=False,columnspacing=1.5,handlelength=1.5)

right=fig.add_axes([.79,.24,.17,.54],facecolor='#FAFBF9')
right.plot(means,np.arange(5),color='#244B56',linewidth=1.7,
           marker='o',markersize=4.5)
right.set(ylim=(4.5,-.5),xlim=(6.0,8.7),
          yticks=np.arange(5),yticklabels=[f'QP {q}' for q in qps],
          xticks=[6,7,8],xlabel='Mean blocks')
right.tick_params(axis='y',length=0,labelsize=7)
right.grid(axis='x',color='#DCE5E3',linewidth=.6)
right.set_axisbelow(True)
right.spines['left'].set_visible(False)
fig.text(.105,.955,'Router workload shifts toward deeper exits as QP rises',
         fontsize=10.2,fontweight='bold',color='#183941',va='top')
fig.text(.105,.055,
         '53 CTC sequences per QP · archived source-calibrated maps · two QP63 maps unavailable and excluded · no timing claim',
         fontsize=6.6,color='#63787C')
dest=ROOT/'docs/research/2026-10-02-triton-early-exit'
for ext in ('pdf','png'):
    fig.savefig(dest/f'ctc_workload.{ext}',dpi=350,facecolor=fig.get_facecolor())
plt.close(fig)
