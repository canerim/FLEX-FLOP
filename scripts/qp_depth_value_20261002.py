"""CPU-only QP interaction of archived uniform-exit tile gains."""
from pathlib import Path
import hashlib
import json
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

ROOT = Path(__file__).resolve().parents[1]
SOURCE = ROOT / 'data/refresh20260927/exit_depth_profile.json'
OUT = ROOT / 'data/qp_depth_20261002'
FIG = ROOT / 'figs/qp_depth_20261002'
OUT.mkdir(exist_ok=True); FIG.mkdir(exist_ok=True)
src = json.loads(SOURCE.read_text())
rows = src['incremental_rows']
qps, stages = (0, 16, 32, 48, 63), (6, 8, 10)
seqs = sorted({r['sequence'] for r in rows})
assert len(seqs) == 53 and len(rows) == 53*5*3
idx = {}
for r in rows:
    key = (r['sequence'], r['qp'], r['from_depth'])
    assert key not in idx and r['qp'] in qps and r['from_depth'] in stages
    assert r['to_depth'] == r['from_depth']+2
    assert np.isfinite(r['median']) and np.isfinite(r['iqr'])
    idx[key] = r
assert len(idx) == len(rows)

rng = np.random.default_rng(20261002)
# One sequence cluster carries all QPs and stages through each bootstrap draw.
draws = rng.integers(0, len(seqs), (5000, len(seqs)))
records = []
for stage in stages:
    for qp in qps:
        vals = np.array([idx[(s, qp, stage)]['median'] for s in seqs])
        boot = vals[draws].mean(axis=1)
        records.append({'from_depth': stage, 'to_depth': stage+2, 'qp': qp,
                        'n_sequences': len(vals), 'mean_frame_median_gain_db': float(vals.mean()),
                        'ci95_db': [float(v) for v in np.quantile(boot,[.025,.975])]})
contrasts = []
for stage in stages:
    values = np.array([idx[(s,63,stage)]['median']-idx[(s,0,stage)]['median'] for s in seqs])
    boot=values[draws].mean(axis=1)
    contrasts.append({'from_depth':stage,'qp63_minus_qp0_mean_db':float(values.mean()),
                      'ci95_db':[float(v) for v in np.quantile(boot,[.025,.975])]})
result={'scope':'Archived uniform-exit output comparisons over 53 sequences × 5 QPs; 265 frame-QP cases, not independent codecs or mixed-map causal tile interventions.',
        'metric':'Mean over frames of within-frame median 10log10(MSE_shallow/MSE_deep) on padded YCbCr 4:4:4 network input; historical source calls it RGB.',
        'bootstrap':'5000 paired sequence-cluster draws, seed 20261002; QPs and stages grouped within sequence.',
        'source_sha256':hashlib.sha256(SOURCE.read_bytes()).hexdigest(),
        'records':records,'qp63_minus_qp0':contrasts}
(OUT/'analysis.json').write_text(json.dumps(result,indent=2)+'\n')

plt.rcParams.update({'font.family':'DejaVu Sans','font.size':7.5,
                     'pdf.fonttype':42,'axes.spines.top':False,'axes.spines.right':False})
fig,ax=plt.subplots(figsize=(3.45,2.35))
colors={6:'#1a7891',8:'#d99c2b',10:'#b86175'}
for stage in stages:
    rr=[r for r in records if r['from_depth']==stage]
    x=np.array([r['qp'] for r in rr]); y=np.array([r['mean_frame_median_gain_db'] for r in rr])
    lo=y-np.array([r['ci95_db'][0] for r in rr]);hi=np.array([r['ci95_db'][1] for r in rr])-y
    ax.errorbar(x,y,yerr=[lo,hi],color=colors[stage],marker='o',markersize=3,
                lw=1.25,capsize=1.7,label=f'{stage}→{stage+2}')
ax.set(xlim=(-3,66),ylim=(0,.165),xticks=qps,xlabel='DCVC-UF quality index (QP)',
       ylabel='Median tile gain (dB)')
ax.grid(axis='y',alpha=.15)
ax.legend(frameon=False,fontsize=7,ncol=3,loc='upper left',title='Added synthesis blocks',title_fontsize=7)
fig.tight_layout()
for ext in ('pdf','png'):
    fig.savefig(FIG/f'qp_depth_gain.{ext}',dpi=250,bbox_inches='tight')
plt.close(fig)
