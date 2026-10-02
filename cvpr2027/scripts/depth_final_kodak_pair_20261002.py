"""Paired matched-estimated-rate Kodak comparison; CPU only, no extrapolation."""
from pathlib import Path
import hashlib
import json
import math

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / 'data/depth_final_20261002'
FIG = ROOT / 'figs/depth_final_20261002'
FIG.mkdir(exist_ok=True)
FILES = {'D2': 'd2_epoch105_kodak.json', 'D4': 'd4_epoch105_kodak.json',
         'released D12': 'released_d12_kodak.json'}
METRICS = {'psnr_rgb': 'RGB PSNR', 'psnr_yuv611': 'YUV 6:1:1 PSNR'}
RATES = (0.1, 0.2, 0.4)
BOOT = 5000
SEED = 20261002

sources, curves, hashes = {}, {}, {}
for model, file in FILES.items():
    path = DATA / file
    hashes[file] = hashlib.sha256(path.read_bytes()).hexdigest()
    data = json.loads(path.read_text())
    assert data['rate_kind'] == 'deterministic entropy estimate; NOT actual bitstream'
    if model != 'released D12':
        assert data['depth'] == int(model[1:])
    assert len(data['rows']) == 120
    seen = set()
    by_image = {}
    for row in data['rows']:
        key = (row['image'], row['qp'])
        assert key not in seen and row['qp'] in (0, 16, 32, 48, 63)
        assert all(math.isfinite(row[k]) for k in ('estimated_bpp', *METRICS))
        assert row['estimated_bpp'] > 0
        seen.add(key)
        by_image.setdefault(row['image'], []).append(row)
    assert len(by_image) == 24 and all(len(v) == 5 for v in by_image.values())
    sources[model] = data
    curves[model] = {image: sorted(rows, key=lambda x: x['estimated_bpp'])
                     for image, rows in by_image.items()}
assert len({frozenset(c) for c in curves.values()}) == 1
images = sorted(curves['D2'])
rng = np.random.default_rng(SEED)
records = []
for metric in METRICS:
    for rate in RATES:
        per_image = {}
        target = math.log(rate)
        for image in images:
            interpolated = {}
            for model in FILES:
                rows = curves[model][image]
                x = np.log([r['estimated_bpp'] for r in rows])
                y = [r[metric] for r in rows]
                assert np.all(np.diff(x) > 0)
                if x[0] <= target <= x[-1]:
                    interpolated[model] = float(np.interp(target, x, y))
            if len(interpolated) == 3:
                per_image[image] = {
                    'D4 minus D2': interpolated['D4'] - interpolated['D2'],
                    'released D12 minus D4': interpolated['released D12'] - interpolated['D4']}
        assert len(per_image) >= 20
        for contrast in ('D4 minus D2', 'released D12 minus D4'):
            vals = np.array([per_image[image][contrast] for image in sorted(per_image)])
            draws = vals[rng.integers(0, len(vals), (BOOT, len(vals)))].mean(axis=1)
            records.append({'metric': metric, 'rate_bpp': rate, 'contrast': contrast,
                            'n_images': len(vals), 'mean_db': float(vals.mean()),
                            'ci95_db': [float(v) for v in np.quantile(draws, [.025, .975])],
                            'per_image_db': {im: per_image[im][contrast] for im in sorted(per_image)}})

result = {'scope': 'Final epoch-105 D2/D4 Kodak full-resolution; released D12 is a historical anchor with different training history.',
          'rate': 'Deterministic entropy estimate, not actual bitstream; per-image linear interpolation in log estimated-bpp; no extrapolation.',
          'metrics': 'RGB PSNR and upstream YUV 6:1:1 PSNR; not YUV 4:4:4 or cropped early-exit loss.',
          'uncertainty': f'{BOOT} paired image bootstrap draws; seed {SEED}; 24 Kodak images; no training-seed uncertainty.',
          'source_sha256': hashes, 'records': records}
(DATA / 'paired_kodak_analysis.json').write_text(json.dumps(result, indent=2) + '\n')

plt.rcParams.update({'font.family': 'DejaVu Sans', 'font.size': 8.5,
                     'pdf.fonttype': 42, 'axes.spines.top': False,
                     'axes.spines.right': False})
fig, axes = plt.subplots(1, 2, figsize=(6.92, 2.6), sharey=True)
colors = {'D4 minus D2': '#1a7891', 'released D12 minus D4': '#bd6b4d'}
markers = {'D4 minus D2': 'o', 'released D12 minus D4': 's'}
for ax, (metric, title) in zip(axes, METRICS.items()):
    for contrast in colors:
        subset = [r for r in records if r['metric'] == metric and r['contrast'] == contrast]
        x = np.array([r['rate_bpp'] for r in subset]); y = np.array([r['mean_db'] for r in subset])
        lo = y - np.array([r['ci95_db'][0] for r in subset]); hi = np.array([r['ci95_db'][1] for r in subset]) - y
        ax.errorbar(x, y, yerr=[lo, hi], color=colors[contrast], marker=markers[contrast],
                    linewidth=1.4, markersize=4.3, capsize=2.3, label=contrast)
    ax.axhline(0, color='#8e969b', lw=.7)
    ax.set(xlim=(.065, .435), xticks=RATES, xlabel='Estimated bits per pixel',
           title=title)
    ax.grid(axis='y', alpha=.17)
axes[0].set_ylabel('Paired PSNR gain (dB)')
axes[0].legend(frameon=False, fontsize=7.3, loc='lower left')
fig.tight_layout(w_pad=2.2)
for ext in ('pdf', 'png'):
    fig.savefig(FIG / f'paired_kodak.{ext}', dpi=250, bbox_inches='tight')
plt.close(fig)
