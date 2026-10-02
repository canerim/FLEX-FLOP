"""Publication-ready, explicitly exploratory early-exit kernel audit graphic."""
from __future__ import annotations

import json
from pathlib import Path

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.colors import BoundaryNorm, ListedColormap
import numpy as np

ROOT=Path(__file__).resolve().parents[1]
DATA=json.loads((ROOT/'results/triton_early_exit_paired_videoSRC05_qp32.json').read_text())
EVAL=json.loads((ROOT/'flexplus/results/eval_rules_ctc_e15.json').read_text())
row=next(r for r in EVAL['rows'] if r['seq']=='videoSRC05_1920x1080_25.yuv' and r['qp']==32)
grid=np.array(row['rules']['router']['0.1']['map']).reshape(row['grid'])
assert grid.shape==(5,8)

dest=ROOT/'docs/research/2026-10-02-triton-early-exit'
dest.mkdir(parents=True,exist_ok=True)
plt.rcParams.update({'font.family':'DejaVu Sans','font.size':8,'pdf.fonttype':42,
                     'axes.spines.top':False,'axes.spines.right':False})
fig=plt.figure(figsize=(7.2,4.2),facecolor='#F9FAF8')
ax=fig.add_axes([.23,.29,.39,.52],facecolor='#F9FAF8')
am=fig.add_axes([.67,.35,.28,.40],facecolor='#F9FAF8')
keys=['masked_stock','sorted_stock','sorted_fused','sorted_ffn',
      'sorted_trunk','sorted_full']
names=['Current masked','Sorted tiles','Fused activations',
       'Fused FFN','Fused trunk','Fused adapters']
colors=['#899295','#506974','#43818A','#2F7079','#175D66','#B15B63']
yy=np.arange(len(keys))[::-1]
for y,k,name,c in zip(yy,keys,names,colors):
    raw=np.array(DATA['timing_ms'][k]['samples'])
    med=float(np.median(raw))
    ax.hlines(y,100,med,color=c,lw=2.0,alpha=.67)
    jitter=np.linspace(-.11,.11,len(raw))
    ax.scatter(raw,y+jitter,s=8,color=c,alpha=.35,lw=0,zorder=3)
    ax.scatter([med],[y],s=29,color=c,edgecolors='#F9FAF8',lw=.7,zorder=4)
    ax.text(med+4,y,f'{med:.0f}',va='center',fontsize=7.4,color='#253B42')
ax.set(yticks=yy,yticklabels=names,xlim=(98,345),ylim=(-.6,5.6),
       xlabel='Decoder synthesis latency (ms)')
ax.tick_params(axis='y',length=0,pad=5)
ax.grid(axis='x',color='#DDE4E3',lw=.6)
ax.set_axisbelow(True)
ax.text(0,1.095,'a  Paired CUDA-event timings',transform=ax.transAxes,
        fontsize=9.2,fontweight='bold',color='#20383F')
speed=DATA['paired_speedup']['masked_to_full']['median']
ax.text(0,-.31,f'{speed:.2f}× faster than current masked path',transform=ax.transAxes,
        fontsize=8.2,fontweight='bold',color='#B15B63')

palette=['#D3E6E6','#6AA4A3','#D6A45E','#B86E7A']
cmap=ListedColormap(palette)
am.imshow(grid,cmap=cmap,norm=BoundaryNorm([1.5,2.5,3.5,4.5,5.5],4),
          interpolation='nearest',aspect='equal')
for i in range(5):
    for j in range(8):
        am.text(j,i,str(2*(int(grid[i,j])+1)),ha='center',va='center',
                color='#20383F' if grid[i,j]<=3 else '#FFFFFF',fontsize=6.4,
                fontweight='bold')
am.set(xticks=[],yticks=[],xlim=(-.5,7.5),ylim=(4.5,-.5))
for spine in am.spines.values(): spine.set_visible(False)
for j in np.arange(-.5,8,1): am.axvline(j,color='#F9FAF8',lw=.8)
for i in np.arange(-.5,5,1): am.axhline(i,color='#F9FAF8',lw=.8)
am.text(0,1.16,'b  Routed depth map',transform=am.transAxes,
        fontsize=9.2,fontweight='bold',color='#20383F')
am.text(0,-.13,'Each cell: retained trunk blocks',transform=am.transAxes,
        fontsize=6.9,color='#62757A')
fig.text(.05,.965,'DCVC-UF early exit  |  fused synthesis kernels',
         ha='left',va='top',fontsize=10.5,fontweight='bold',color='#20383F')
fig.text(.05,.075,
         'videoSRC05 · QP32 · 1920×1080 padded to 2048×1280 · 6 paired trials on an A6000 shared with training · decoder only',
         fontsize=6.6,color='#60747A')
fig.text(.05,.043,
         'Raw output max |Δ| = 4.47×10⁻⁷; YUV 6:1:1 PSNR shift = −1.31×10⁻⁷ dB. Independent GPU and full-codec tests remain.',
         fontsize=6.6,color='#60747A')
for ext in ['pdf','png']:
    fig.savefig(dest/f'latency_map.{ext}',dpi=350,facecolor=fig.get_facecolor())
plt.close(fig)
