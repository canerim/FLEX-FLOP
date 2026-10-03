"""Recompute paper-facing GPU timing and fidelity claims from raw records."""
from pathlib import Path
import hashlib
import json
import statistics

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np

PAPER = Path(__file__).resolve().parents[1]
OUT = PAPER/'data/triton_20261003'
FIG = PAPER/'figs/triton_20261003'
RAW = OUT/'raw'
OUT.mkdir(exist_ok=True)
FIG.mkdir(exist_ok=True)
paths = {
    'stages': RAW/'triton_early_exit_paired_videoSRC05_qp32.json',
    'depthwise_macro': RAW/'triton_ctc_src05_qp32_depthwise_shared.jsonl',
    'final_macro': RAW/'triton_ctc_src05_qp32_fused_seam_shared.jsonl',
    'seam_ablation': RAW/'triton_seam_ablation_shared.json',
    'quality': RAW/'triton_early_exit_quality_seam_audit.json',
    'memory': RAW/'triton_early_exit_peak_memory_seam.json',
    'map_workload': RAW/'triton_ctc_map_workload.json',
}
raw = {name: json.loads(path.read_text().splitlines()[0]) if path.suffix=='.jsonl'
       else json.loads(path.read_text()) for name,path in paths.items()}


def med(x):
    return float(statistics.median(x))


stage = raw['stages']
assert stage['device'] == 'NVIDIA RTX A6000' and stage['qp'] == 32
assert stage['map_counts'] == [0,0,13,20,4,3]
assert stage['shape_rgb'] == [1280,2048] and stage['valid_shape'] == [1080,1920]
order = [('masked_stock','Masked stock'),('sorted_stock','Sorted tiles'),
         ('sorted_fused','Fused activations'),('sorted_ffn','Fused FFN'),
         ('sorted_trunk','Fused trunk'),('sorted_full','Fused adapters')]
stages = []
for key,label in order:
    samples = stage['timing_ms'][key]['samples']
    assert len(samples)==6
    assert abs(med(samples)-stage['timing_ms'][key]['median']) < 1e-5
    stages.append({'stage':key,'label':label,'median_ms':med(samples),'samples_ms':samples})
ratios = [a/b for a,b in zip(stage['timing_ms']['masked_stock']['samples'],
                            stage['timing_ms']['sorted_full']['samples'])]
assert abs(med(ratios)-stage['paired_speedup']['masked_to_full']['median']) < 1e-9

macro = raw['depthwise_macro']
assert len(macro['stock_ms']) == len(macro['fast_ms']) == 4
assert macro['map_hist'] == stage['map_counts']
macro_ratios = [a/b for a,b in zip(macro['stock_ms'],macro['fast_ms'])]
assert abs(med(macro_ratios)-macro['paired_speedup_median']) < 1e-9
final = raw['final_macro']
assert final['gpu_exclusive_preflight'] is False
assert final['map_hist'] == stage['map_counts']
final_ratios = [a/b for a,b in zip(final['stock_ms'],final['fast_ms'])]
assert abs(med(final_ratios)-final['paired_speedup_median']) < 1e-9

seam = raw['seam_ablation']
assert len(seam['measurements']['stock_seam']['cuda_ms']) == 16
seam_ratios = [a/b for a,b in zip(seam['measurements']['stock_seam']['cuda_ms'],
                                 seam['measurements']['fused_seam']['cuda_ms'])]
assert abs(med(seam_ratios)-seam['paired_speedup_median']) < 1e-9

quality = raw['quality']['cases']
assert len(quality)==5
max_abs = max(c['max_abs_raw'] for c in quality)
max_yuv = max(abs(c['delta_yuv611_db']) for c in quality)
assert max_abs < 1e-6 and max_yuv < 3e-7
memory = raw['memory']['metrics']
old = memory['masked_stock']['incremental_peak_bytes']
new = memory['sorted_fused_full']['incremental_peak_bytes']
saving = 100*(old-new)/old
assert 35 < saving < 36

audit = {'scope':'shared A6000 GPU, real CTC videoSRC05 QP32 archived router map; decoder synthesis only',
         'source_sha256':{k:hashlib.sha256(v.read_bytes()).hexdigest() for k,v in paths.items()},
         'device':stage['device'],'tf32_enabled':stage['tf32_enabled'],
         'valid_hw':stage['valid_shape'],'padded_hw':stage['shape_rgb'],
         'map_hist':stage['map_counts'],
         'stages':stages,
         'stage_paired_speedup':{'n':6,'median':med(ratios),'min':min(ratios),'max':max(ratios)},
         'depthwise_macro':{'n':4,'stock_median_ms':med(macro['stock_ms']),
                            'fast_median_ms':med(macro['fast_ms']),
                            'paired_speedup_median':med(macro_ratios),
                            'wall_stock_median_ms':med(macro['stock_wall_ms']),
                            'wall_fast_median_ms':med(macro['fast_wall_ms'])},
         'final_macro':{'n':4,'stock_median_ms':med(final['stock_ms']),
                        'fast_median_ms':med(final['fast_ms']),
                        'paired_speedup_median':med(final_ratios),
                        'gpu_exclusive_preflight':False,
                        'interpretation':'stock drift prevents an incremental-gain claim'},
         'seam_ablation':{'n':16,'stock_median_ms':med(seam['measurements']['stock_seam']['cuda_ms']),
                          'fast_median_ms':med(seam['measurements']['fused_seam']['cuda_ms']),
                          'paired_speedup_median':med(seam_ratios)},
         'fidelity':{'n_ctc_qp':len(quality),'max_abs_raw':max_abs,
                     'max_abs_yuv611_delta_db':max_yuv},
         'incremental_peak_memory':{'stock_bytes':old,'fast_bytes':new,'saving_pct':saving},
         'claim_boundary':'exploratory shared-GPU synthesis timing; no exclusive 53x5 or end-to-end codec benchmark'}
(OUT/'audit.json').write_text(json.dumps(audit,indent=2)+'\n')

plt.rcParams.update({'font.family':'DejaVu Sans','font.size':7.0,
                     'pdf.fonttype':42,'axes.spines.top':False,
                     'axes.spines.right':False})
fig,axs=plt.subplots(2,1,figsize=(3.45,3.15),
                    gridspec_kw={'height_ratios':[1.25,1]})
navy,teal,rose='#203A4A','#168690','#B66070'
ax=axs[0]
for i,s in enumerate(stages):
    color=rose if i==len(stages)-1 else (navy if i==0 else teal)
    y=len(stages)-1-i
    ax.plot(s['samples_ms'],[y]*6,'o',ms=2.4,color=color,alpha=.35)
    ax.plot(s['median_ms'],y,'o',ms=5.2,color=color)
    ax.text(s['median_ms']+6,y+.08,f"{s['median_ms']:.0f}",va='center',fontsize=6.7,color=color)
ax.set(yticks=range(6),yticklabels=[s['label'] for s in stages[::-1]],
       xlabel='CUDA-event synthesis latency (ms)',xlim=(90,323),
       title='a  One routed CTC frame')
ax.grid(axis='x',alpha=.14)
ax=axs[1]
for x,(label,ratios,color) in enumerate([('Core fusion',ratios,teal),
                                         ('+ depthwise',macro_ratios,navy),
                                         ('Final seam',final_ratios,rose)]):
    ax.plot([x]*len(ratios),ratios,'o',ms=3,color=color,alpha=.35)
    ax.plot(x,med(ratios),'D',ms=5,color=color)
    ax.text(x,med(ratios)+.12,f'{med(ratios):.2f}×',ha='center',fontsize=7,color=color)
ax.axhline(1,color='#6C777E',lw=.7)
ax.set(xticks=range(3),xticklabels=['Core\nfusion','+ depthwise','Final\nseam'],
       ylim=(1.8,3.05),ylabel='Paired stock / fast ratio',
       title='b  Separate paired runs')
ax.grid(axis='y',alpha=.14)
fig.subplots_adjust(left=.37,right=.97,bottom=.16,top=.91,hspace=.62)
for ext in ('pdf','png'):
    fig.savefig(FIG/f'latency_audit.{ext}',dpi=260)
plt.close(fig)

workload=raw['map_workload']['by_qp']
assert raw['map_workload']['n_rows']==265
qps=[0,16,32,48,63]
palette=['#82B9B6','#D9A655','#BA7180','#476B8A']
fig,ax=plt.subplots(figsize=(3.35,1.65))
bottom=np.zeros(5)
for i,depth in enumerate((6,8,10,12)):
    values=np.array([workload[str(q)]['exit_fraction'][i+2]*100 for q in qps])
    ax.bar(range(5),values,bottom=bottom,width=.68,color=palette[i],
           label=f'Exit {depth}',edgecolor='white',linewidth=.5)
    bottom+=values
ax.set(xticks=range(5),xticklabels=qps,ylim=(0,100),
       ylabel='Routed tiles (%)',xlabel='QP')
ax.legend(frameon=False,ncol=4,loc='upper center',
          bbox_to_anchor=(.5,1.18),fontsize=6.7,columnspacing=.8)
ax.spines[['top','right']].set_visible(False)
ax.grid(axis='y',alpha=.1)
fig.subplots_adjust(left=.17,right=.99,bottom=.27,top=.84)
for ext in ('pdf','png'):
    fig.savefig(FIG/f'map_workload.{ext}',dpi=260)
plt.close(fig)
print(json.dumps({k:audit[k] for k in ('stage_paired_speedup','depthwise_macro','final_macro',
                                      'seam_ablation','fidelity','incremental_peak_memory')},indent=2))
