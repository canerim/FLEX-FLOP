"""Audited CPU-only Kodak RD comparison of completed D2/D4/D6 models."""
from pathlib import Path
import hashlib
import json
import math

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
from scipy.interpolate import PchipInterpolator

ROOT = Path(__file__).resolve().parents[1]
OLD = ROOT / 'data/depth_final_20261002'
OUT = ROOT / 'data/depth_final_20261003'
FIG = ROOT / 'figs/depth_final_20261003'
OUT.mkdir(exist_ok=True)
FIG.mkdir(exist_ok=True)
FILES = {'D2': OLD/'d2_epoch105_kodak.json',
         'D4': OLD/'d4_epoch105_kodak.json',
         'D6': OUT/'d6_epoch105_kodak.json',
         'released D12': OLD/'released_d12_kodak.json'}
QPS = (0, 16, 32, 48, 63)
METRICS = ('psnr_rgb', 'psnr_yuv611')
BOOT, SEED = 5000, 20261003
rng = np.random.default_rng(SEED)


def interval(values):
    x = np.asarray(values, dtype=float)
    draws = x[rng.integers(0, len(x), (BOOT, len(x)))].mean(axis=1)
    return [float(v) for v in np.quantile(draws, (.025, .975))]


raw = {name: json.loads(path.read_text()) for name, path in FILES.items()}
hashes = {name: hashlib.sha256(path.read_bytes()).hexdigest()
          for name, path in FILES.items()}
by = {}
for name, data in raw.items():
    assert data['dataset'] == 'Kodak full resolution'
    assert data['rate_kind'] == 'deterministic entropy estimate; NOT actual bitstream'
    if name.startswith('D'):
        assert data['depth'] == int(name[1:])
    assert len(data['rows']) == 120
    grouped = {}
    for row in data['rows']:
        grouped.setdefault(row['image'], {})[row['qp']] = row
    assert len(grouped) == 24
    assert all(set(rows) == set(QPS) for rows in grouped.values())
    by[name] = grouped
images = sorted(by['D2'])
assert all(sorted(grouped) == images for grouped in by.values())


def rd_curve(rows, metric):
    ordered = sorted(rows.values(), key=lambda r: r[metric])
    q = np.array([r[metric] for r in ordered])
    log_rate = np.log([r['estimated_bpp'] for r in ordered])
    assert np.all(np.diff(q) > 0) and np.all(np.diff(log_rate) > 0)
    return q, PchipInterpolator(q, log_rate, extrapolate=False)


result = {'scope': '24 full-resolution Kodak images, 5 QPs; final 105-epoch independent D2/D4/D6 and historical released D12.',
          'rate': 'deterministic entropy estimate, not emitted bitstream or container bytes',
          'method': 'per-image monotone PCHIP log estimated-bpp vs PSNR; four-model common PSNR support; arithmetic mean of per-image BD-rate percentages; no extrapolation',
          'uncertainty': '5000 paired-image bootstrap draws, fixed checkpoints; no training-seed variation',
          'source_sha256': hashes, 'bd_rate': [], 'matched_rate': [], 'qp_means': {}}
for name, data in raw.items():
    result['qp_means'][name] = {str(row['qp']): {
        'estimated_bpp': row['estimated_bpp'],
        'psnr_rgb': row['psnr_rgb'], 'psnr_yuv611': row['psnr_yuv611']
    } for row in data['means']}

for metric in METRICS:
    per = {name: [] for name in FILES if name != 'released D12'}
    pair = {'D4-D2': [], 'D6-D4': [], 'D6-D2': []}
    for image in images:
        cs = {name: rd_curve(by[name][image], metric) for name in FILES}
        low = max(v[0][0] for v in cs.values())
        high = min(v[0][-1] for v in cs.values())
        assert high > low
        base = cs['released D12'][1]
        for name in per:
            logdiff = (cs[name][1].integrate(low, high) -
                       base.integrate(low, high)) / (high-low)
            per[name].append(100 * math.expm1(logdiff))
        rate = .2
        rates = {name: PchipInterpolator(
            np.log([r['estimated_bpp'] for r in sorted(by[name][image].values(),
                                                       key=lambda r: r['estimated_bpp'])]),
            [r[metric] for r in sorted(by[name][image].values(),
                                       key=lambda r: r['estimated_bpp'])],
            extrapolate=False) for name in FILES}
        if all(min(r['estimated_bpp'] for r in by[name][image].values()) <= rate <=
               max(r['estimated_bpp'] for r in by[name][image].values()) for name in FILES):
            vals = {name: float(fn(math.log(rate))) for name, fn in rates.items()}
            pair['D4-D2'].append(vals['D4']-vals['D2'])
            pair['D6-D4'].append(vals['D6']-vals['D4'])
            pair['D6-D2'].append(vals['D6']-vals['D2'])
    for name, vals in per.items():
        result['bd_rate'].append({'metric': metric, 'model': name,
                                  'reference': 'released D12', 'n_images': len(vals),
                                  'mean_pct': float(np.mean(vals)), 'ci95_pct': interval(vals),
                                  'per_image_pct': dict(zip(images, vals))})
    for contrast, vals in pair.items():
        assert len(vals) >= 20
        result['matched_rate'].append({'metric': metric, 'rate_bpp': .2,
                                       'contrast': contrast, 'n_images': len(vals),
                                       'mean_db': float(np.mean(vals)),
                                       'ci95_db': interval(vals)})

(OUT/'analysis.json').write_text(json.dumps(result, indent=2)+'\n')
plt.rcParams.update({'font.family': 'DejaVu Sans', 'font.size': 8,
                     'pdf.fonttype': 42, 'axes.spines.top': False,
                     'axes.spines.right': False})
fig, axs = plt.subplots(1, 2, figsize=(6.9, 2.25), gridspec_kw={'wspace': .35})
colors = {'D2': '#AF6373', 'D4': '#328A9B', 'D6': '#1A5B78'}
ax = axs[0]
for name in ('D2', 'D4', 'D6'):
    r = next(r for r in result['bd_rate'] if r['model']==name and r['metric']=='psnr_yuv611')
    idx = ('D2','D4','D6').index(name)
    ax.errorbar(r['mean_pct'], idx, xerr=[[r['mean_pct']-r['ci95_pct'][0]],
                                       [r['ci95_pct'][1]-r['mean_pct']]],
                fmt='o', ms=5, lw=1.3, capsize=2, color=colors[name])
    ax.text(r['ci95_pct'][1]+.18, idx+.035, f"+{r['mean_pct']:.2f}%",
            color=colors[name], fontsize=7.5)
ax.axvline(0, color='#6E777D', lw=.8)
ax.set(yticks=range(3), yticklabels=['D2','D4','D6'],
       xlabel='Estimated BD-rate vs released D12 (%)', title='Quality-aligned rate cost',
       xlim=(-.45,10.3))
ax.invert_yaxis(); ax.grid(axis='x', alpha=.14)
ax = axs[1]
for name, color in [('D2',colors['D2']),('D4',colors['D4']),('D6',colors['D6']),
                    ('released D12','#252F39')]:
    rows = [result['qp_means'][name][str(q)] for q in QPS]
    ax.plot([r['estimated_bpp'] for r in rows], [r['psnr_yuv611'] for r in rows],
            marker='o', ms=3, lw=1.2, color=color, label=name)
ax.set(xlabel='Estimated bits per pixel', ylabel='YUV 6:1:1 PSNR (dB)',
       title='Kodak24 mean QP points', xscale='log')
ax.legend(frameon=False, fontsize=6.8); ax.grid(alpha=.13)
fig.subplots_adjust(left=.07, right=.99, bottom=.23, top=.82)
for ext in ('pdf','png'):
    fig.savefig(FIG/f'kodak_three.{ext}', dpi=260)
plt.close(fig)
print(json.dumps({'bd_rate':[{k:v for k,v in r.items() if k!='per_image_pct'} for r in result['bd_rate']],
                  'matched_rate':result['matched_rate']}, indent=2))
