"""Kodak24 five-QP RD points and paired rate-matched gaps; CPU-only."""
from pathlib import Path
import hashlib
import json
import math

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.ticker import FixedLocator, FuncFormatter
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / 'data/depth_final_20261002'
OUT = ROOT / 'figs/depth_final_20261002'
OUT.mkdir(parents=True, exist_ok=True)
FILES = {'D2': 'd2_epoch105_kodak.json',
         'D4': 'd4_epoch105_kodak.json',
         'Released D12': 'released_d12_kodak.json'}
QPS = (0, 16, 32, 48, 63)
RATES = (.1, .2, .4)
METRICS = (('psnr_rgb', 'RGB PSNR'), ('psnr_yuv611', 'YUV 6:1:1 PSNR'))
COLORS = {'D2': '#B45A69', 'D4': '#117E85', 'Released D12': '#203645'}
MARKERS = {'D2': 'o', 'D4': 's', 'Released D12': 'D'}

source, curves, sha = {}, {}, {}
for name, filename in FILES.items():
    path = DATA / filename
    sha[filename] = hashlib.sha256(path.read_bytes()).hexdigest()
    data = json.loads(path.read_text())
    assert data['dataset'] == 'Kodak full resolution'
    assert data['rate_kind'] == 'deterministic entropy estimate; NOT actual bitstream'
    assert len(data['rows']) == 24 * len(QPS)
    grouped = {}
    for row in data['rows']:
        grouped.setdefault(row['image'], {})[row['qp']] = row
    assert len(grouped) == 24
    assert all(tuple(sorted(rows)) == QPS for rows in grouped.values())
    assert all(math.isfinite(row[key]) and row[key] > 0
               for row in data['rows'] for key in ('estimated_bpp', 'psnr_rgb', 'psnr_yuv611'))
    for mean in data['means']:
        assert mean['qp'] in QPS
        for key in ('estimated_bpp', 'psnr_rgb', 'psnr_yuv611'):
            actual = np.mean([row[key] for row in data['rows'] if row['qp'] == mean['qp']])
            assert math.isclose(actual, mean[key], abs_tol=1e-7)
    source[name] = data
    curves[name] = grouped
assert all(set(curves[name]) == set(curves['Released D12']) for name in FILES)


def interpolate(rows, metric, rate):
    ordered = sorted(rows.values(), key=lambda row: row['estimated_bpp'])
    x = np.log([row['estimated_bpp'] for row in ordered])
    y = [row[metric] for row in ordered]
    target = math.log(rate)
    assert np.all(np.diff(x) > 0)
    return float(np.interp(target, x, y)) if x[0] <= target <= x[-1] else None


paired = {}
rng = np.random.default_rng(20261002)
for metric, _ in METRICS:
    for name in ('D2', 'D4'):
        for rate in RATES:
            values = []
            for image in sorted(curves['Released D12']):
                baseline = interpolate(curves['Released D12'][image], metric, rate)
                candidate = interpolate(curves[name][image], metric, rate)
                if baseline is not None and candidate is not None:
                    values.append(candidate - baseline)
            assert len(values) >= 20
            values = np.asarray(values)
            boot = values[rng.integers(0, len(values), (5000, len(values)))].mean(axis=1)
            paired[(metric, name, rate)] = {'n_images': len(values),
                                           'mean_db': float(np.mean(values)),
                                           'ci95_db': [float(v) for v in np.quantile(boot, [.025, .975])]}

plt.rcParams.update({'font.family': 'DejaVu Sans', 'font.size': 8,
                     'pdf.fonttype': 42, 'axes.spines.top': False,
                     'axes.spines.right': False})
fig = plt.figure(figsize=(7.05, 4.8), facecolor='white')
gs = fig.add_gridspec(2, 2, height_ratios=[2.6, 1.25], hspace=.52, wspace=.28,
                      left=.092, right=.985, top=.92, bottom=.15)
axes = [[fig.add_subplot(gs[i, j]) for j in range(2)] for i in range(2)]

for j, (metric, metric_name) in enumerate(METRICS):
    top, bottom = axes[0][j], axes[1][j]
    for name in ('Released D12', 'D4', 'D2'):
        ordered = sorted(source[name]['means'], key=lambda row: row['qp'])
        x = [row['estimated_bpp'] for row in ordered]
        y = [row[metric] for row in ordered]
        top.plot(x, y, color=COLORS[name], marker=MARKERS[name],
                 ms=4.2, lw=1.65, label=name, zorder=3 if name == 'Released D12' else 2,
                 markeredgecolor='white', markeredgewidth=.45)
    top.set_xscale('log')
    top.set_xlim(.026, 1.16)
    top.xaxis.set_major_locator(FixedLocator((.03, .1, .3, 1.0)))
    top.xaxis.set_major_formatter(FuncFormatter(lambda v, _: f'{v:g}'))
    top.set_xlabel('Estimated bitrate (bits/pixel)', labelpad=2)
    top.set_ylabel(metric_name)
    top.grid(axis='both', color='#D9DFE2', lw=.55, alpha=.65)
    top.set_axisbelow(True)
    top.text(0, 1.075, f'{"ab"[j]}  {metric_name}', transform=top.transAxes,
             fontweight='bold', fontsize=9.2, va='bottom')
    if j == 0:
        top.legend(frameon=False, loc='upper left', fontsize=7.2,
                   handlelength=2.1, labelspacing=.35)
        for point in source['Released D12']['means']:
            top.annotate(f'QP {point["qp"]}',
                         (point['estimated_bpp'], point[metric]),
                         xytext=(8, 14) if point['qp'] == 0 else (4, -11),
                         textcoords='offset points', fontsize=6.8,
                         color='#203645')
    for name in ('D2', 'D4'):
        y = [paired[(metric, name, rate)]['mean_db'] for rate in RATES]
        lo = [y[k] - paired[(metric, name, rate)]['ci95_db'][0]
              for k, rate in enumerate(RATES)]
        hi = [paired[(metric, name, rate)]['ci95_db'][1] - y[k]
              for k, rate in enumerate(RATES)]
        bottom.errorbar(RATES, y, yerr=[lo, hi], color=COLORS[name],
                        marker=MARKERS[name], ms=4.2, lw=1.55, capsize=2,
                        markeredgecolor='white', markeredgewidth=.45)
    bottom.axhline(0, color='#596970', lw=.8)
    bottom.set(xlim=(.075, .425), xticks=RATES,
               xlabel='Matched estimated bitrate (bits/pixel)',
               ylabel='PSNR − released D12 (dB)')
    bottom.grid(axis='y', color='#D9DFE2', lw=.55, alpha=.7)
    bottom.set_axisbelow(True)
    bottom.text(0, 1.10, f'{"cd"[j]}  Paired image gap',
                transform=bottom.transAxes, fontweight='bold', fontsize=9.2, va='bottom')

fig.text(.092, .029,
         'Kodak24 × 5 QPs. Deterministic entropy-estimated rate, not payload bits. '
         'Lower panels: paired log-rate interpolation, 95% image bootstrap intervals; '
         'n=24 at 0.1/0.2, n=23 at 0.4 bpp.',
         fontsize=6.45, color='#52646C')
for extension in ('pdf', 'png'):
    fig.savefig(OUT / f'kodak_rd_5qps.{extension}', dpi=300,
                facecolor='white', bbox_inches='tight')
plt.close(fig)

metadata = {'dataset': 'Kodak full resolution', 'models': list(FILES),
            'qps': QPS, 'rates': RATES,
            'rate_kind': 'deterministic entropy estimate; NOT actual bitstream',
            'source_sha256': sha,
            'paired_mean_db': [dict(metric=metric, model=name, rate_bpp=rate, **value)
                               for (metric, name, rate), value in paired.items()]}
(DATA / 'kodak_rd_5qps_metadata.json').write_text(json.dumps(metadata, indent=2) + '\n')
