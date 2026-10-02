"""Audited BD-rate estimates for independent codecs and shared early exits.

Runs on CPU. Independent Kodak rates are entropy estimates. Shared-exit rates
are round-tripped e15 rANS latent payloads, excluding map and control bits.
The two BD-rate percentages therefore answer different questions.
"""
from collections import defaultdict
from pathlib import Path
import hashlib
import json
import math

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.ticker import MultipleLocator
import numpy as np
from scipy.interpolate import PchipInterpolator

ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / 'data/bd_rate_20261002'
FIG = ROOT / 'figs/bd_rate_20261002'
DATA.mkdir(parents=True, exist_ok=True)
FIG.mkdir(parents=True, exist_ok=True)
KODAK = ROOT / 'data/depth_final_20261002'
FILES = {'D2': KODAK / 'd2_epoch105_kodak.json',
         'D4': KODAK / 'd4_epoch105_kodak.json',
         'released D12': KODAK / 'released_d12_kodak.json'}
EVAL = DATA / 'eval_rules_ctc_e15.json'
PAYLOAD = DATA / 'real_bitstream_ctc.json'
QPS = (0, 16, 32, 48, 63)
POLICIES = ('uniform', 'dither', 'router', 'oracle')
BOOT = 5000
SEED = 20261002


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def ci(values, rng):
    values = np.asarray(values, dtype=float)
    assert len(values) >= 2 and np.isfinite(values).all()
    means = values[rng.integers(0, len(values), (BOOT, len(values)))].mean(axis=1)
    return [float(v) for v in np.quantile(means, (.025, .975))]


def curve(rows, metric, rate):
    ordered = sorted(rows, key=lambda r: r[metric])
    x = np.array([r[metric] for r in ordered], dtype=float)
    y = np.log(np.array([r[rate] for r in ordered], dtype=float))
    assert len(x) == 5 and np.isfinite(x).all() and np.isfinite(y).all()
    assert np.all(np.diff(x) > 0) and np.all(np.diff(y) > 0)
    return x, PchipInterpolator(x, y, extrapolate=False)


def bd_rates(curves, baseline):
    # Same quality integration interval across every comparator in this unit.
    low = max(c[0][0] for c in curves.values())
    high = min(c[0][-1] for c in curves.values())
    assert high > low
    base = curves[baseline][1]
    result = {model: 100 * math.expm1((spline.integrate(low, high) -
                                      base.integrate(low, high)) / (high - low))
              for model, (_, spline) in curves.items() if model != baseline}
    return result, [float(low), float(high)]


rng = np.random.default_rng(SEED)
models = {name: json.loads(path.read_text()) for name, path in FILES.items()}
for name, obj in models.items():
    assert obj['dataset'] == 'Kodak full resolution'
    assert obj['rate_kind'] == 'deterministic entropy estimate; NOT actual bitstream'
    assert len(obj['rows']) == 120
    assert len({(r['image'], r['qp']) for r in obj['rows']}) == 120
    assert {r['qp'] for r in obj['rows']} == set(QPS)
images = sorted({r['image'] for r in models['released D12']['rows']})
assert len(images) == 24
by_image = {name: {im: [r for r in obj['rows'] if r['image'] == im]
                   for im in images} for name, obj in models.items()}
assert all(all(len(rows) == 5 for rows in grouped.values())
           for grouped in by_image.values())

independent = {'scope': '24 full-resolution Kodak images; final epoch-105 D2/D4 versus historical released D12.',
               'rate': 'Deterministic entropy estimate, not emitted bitstream/container bytes.',
               'bd_definition': 'Mean of per-image PCHIP BD-rate percentages on the three-model common PSNR interval; never population-mean-curve BD-rate.',
               'bd': [], 'local_rate_tax': []}
for metric in ('psnr_rgb', 'psnr_yuv611'):
    per_image = {name: [] for name in ('D2', 'D4')}
    intervals = []
    for im in images:
        cs = {name: curve(by_image[name][im], metric, 'estimated_bpp')
              for name in FILES}
        result, support = bd_rates(cs, 'released D12')
        intervals.append(support)
        for name in per_image:
            per_image[name].append(result[name])
    for name, vals in per_image.items():
        independent['bd'].append({'metric': metric, 'model': name,
                                  'n_images': len(vals), 'mean_pct': float(np.mean(vals)),
                                  'ci95_pct': ci(vals, rng),
                                  'per_image_pct': dict(zip(images, vals))})
    for qp in QPS:
        values = {'D2': [], 'D4': []}
        covered = []
        for im in images:
            reference = next(r for r in by_image['released D12'][im] if r['qp'] == qp)
            quality = reference[metric]
            cs = {name: curve(by_image[name][im], metric, 'estimated_bpp')
                  for name in FILES}
            if not all(c[0][0] <= quality <= c[0][-1] for c in cs.values()):
                continue
            covered.append(im)
            for name in values:
                values[name].append(100 * math.expm1(float(cs[name][1](quality)) -
                                                     math.log(reference['estimated_bpp'])))
        for name, vals in values.items():
            independent['local_rate_tax'].append({
                'metric': metric, 'model': name, 'reference_qp': qp,
                'n_images': len(vals), 'mean_pct': float(np.mean(vals)) if vals else None,
                'ci95_pct': ci(vals, rng) if len(vals) >= 2 else None,
                'images': covered})

archive = json.loads(EVAL.read_text())
payload_rows = json.loads(PAYLOAD.read_text())
assert len(archive['rows']) == len(payload_rows) == 53 * 5
assert all(r['roundtrip_ok'] for r in payload_rows)
payload_idx = {(r['seq'], r['qp']): r for r in payload_rows}
assert len(payload_idx) == 265
by_seq = defaultdict(list)
for row in archive['rows']:
    assert row['qp'] in QPS
    assert math.isclose(row['bpp_real'], payload_idx[(row['seq'], row['qp'])]['real_bpp'], abs_tol=1e-12)
    by_seq[row['seq']].append(row)
assert len(by_seq) == 53
for rows in by_seq.values():
    assert len(rows) == 5 and {r['qp'] for r in rows} == set(QPS)

# Nominal source-calibrated 0.1 dB target. Keep only sequences with all five
# QPs for all four policies, so policy contrasts are truly paired.
budget = '0.1'
seqs = sorted(seq for seq, rows in by_seq.items()
              if all(r['rules'][policy][budget] is not None
                     for r in rows for policy in POLICIES))
assert len(seqs) == 51
shared = {'scope': '51 complete five-QP CTC first-frame sequences; e15 full-frame versus archived 0.1 dB source-calibrated uniform/dither/router/oracle reconstructions.',
          'rate': 'Round-tripped e15 rANS latent payload only. Every policy reuses the same rate per frame/QP; mode map, calibration and container bits are absent.',
          'bd_definition': 'Mean per-sequence PCHIP BD-rate equivalent versus e15 full-frame; five-policy common PSNR interval. This is not complete-codec BD-rate or a deployable router result.',
          'n_sequences': len(seqs), 'excluded_sequences': sorted(set(by_seq) - set(seqs)),
          'policy': [], 'qp': [], 'paired_router_minus_dither': []}
bd = {p: [] for p in POLICIES}
costs = {p: [] for p in POLICIES}
qp_data = {p: {q: {'loss': [], 'saving': []} for q in QPS} for p in POLICIES}
for seq in seqs:
    rows = sorted(by_seq[seq], key=lambda r: r['qp'])
    curves = {'dense': curve([{'psnr': r['psnr_release'], 'bpp': r['bpp_real']}
                              for r in rows], 'psnr', 'bpp')}
    for policy in POLICIES:
        curves[policy] = curve([{'psnr': r['rules'][policy][budget]['psnr'],
                                 'bpp': r['bpp_real']} for r in rows], 'psnr', 'bpp')
    values, _ = bd_rates(curves, 'dense')
    for policy in POLICIES:
        bd[policy].append(values[policy])
        costs[policy].append(float(np.mean([r['rules'][policy][budget]['saving']
                                            for r in rows])))
        for row in rows:
            q = row['qp']
            qp_data[policy][q]['loss'].append(row['psnr_release'] -
                                              row['rules'][policy][budget]['psnr'])
            qp_data[policy][q]['saving'].append(row['rules'][policy][budget]['saving'])
for policy in POLICIES:
    shared['policy'].append({'policy': policy, 'n_sequences': len(seqs),
                            'bd_proxy_mean_pct': float(np.mean(bd[policy])),
                            'bd_proxy_ci95_pct': ci(bd[policy], rng),
                            'mac_saving_mean_pct': float(np.mean(costs[policy])),
                            'mac_saving_ci95_pct': ci(costs[policy], rng),
                            'per_sequence_bd_proxy_pct': dict(zip(seqs, bd[policy]))})
    for qp in QPS:
        vals = qp_data[policy][qp]
        shared['qp'].append({'policy': policy, 'qp': qp, 'n_sequences': len(seqs),
                             'mean_loss_db': float(np.mean(vals['loss'])),
                             'loss_ci95_db': ci(vals['loss'], rng),
                             'mean_mac_saving_pct': float(np.mean(vals['saving'])),
                             'saving_ci95_pct': ci(vals['saving'], rng)})
for key, label in [('loss', 'router_minus_dither_loss_db'),
                   ('saving', 'router_minus_dither_saving_pctpt')]:
    for qp in QPS:
        a = np.asarray(qp_data['router'][qp][key])
        b = np.asarray(qp_data['dither'][qp][key])
        values = a - b
        shared['paired_router_minus_dither'].append({'qp': qp, 'contrast': label,
                                                       'mean': float(np.mean(values)),
                                                       'ci95': ci(values, rng)})
values = np.asarray(bd['router']) - np.asarray(bd['dither'])
shared['paired_router_minus_dither_bd_proxy'] = {
    'mean_pctpt': float(np.mean(values)), 'ci95_pctpt': ci(values, rng)}

result = {'source_sha256': {str(path.relative_to(ROOT)): sha(path)
                            for path in [*FILES.values(), EVAL, PAYLOAD]},
          'bootstrap': f'{BOOT} image/sequence-cluster draws, seed {SEED}; fixed trained weights; no training-seed uncertainty.',
          'independent': independent, 'shared_exit': shared}
(DATA / 'analysis.json').write_text(json.dumps(result, indent=2) + '\n')

plt.rcParams.update({'font.family': 'DejaVu Sans', 'font.size': 7.5,
                     'pdf.fonttype': 42, 'axes.spines.top': False,
                     'axes.spines.right': False})
fig, axs = plt.subplots(2, 2, figsize=(7.15, 6.2),
                        gridspec_kw={'hspace': .44, 'wspace': .31})
fig.subplots_adjust(left=.09, right=.965, top=.94, bottom=.19)
teal, rose, navy = '#087F87', '#B45A69', '#203645'
policy_color = {'uniform': '#777F86', 'dither': '#C08C3A',
                'router': '#087F87', 'oracle': '#B45A69'}

# a: Local rate tax at the reference model's exact QP quality.
ax = axs[0, 0]
for model, color, marker in [('D2', rose, 'o'), ('D4', teal, 's')]:
    rr = [r for r in independent['local_rate_tax']
          if r['metric'] == 'psnr_yuv611' and r['model'] == model
          and r['reference_qp'] in (16, 32, 48, 63)]
    x = [r['reference_qp'] for r in rr]
    y = [r['mean_pct'] for r in rr]
    lo = [v-r['ci95_pct'][0] for v,r in zip(y,rr)]
    hi = [r['ci95_pct'][1]-v for v,r in zip(y,rr)]
    ax.errorbar(x, y, yerr=[lo, hi], color=color, marker=marker,
                lw=1.55, ms=4, capsize=2, label=model)
ax.set(xlim=(9,69),xticks=QPS, xlabel='Released D12 quality index (QP)',
       ylabel='Extra estimated rate at equal YUV quality (%)')
ax.text(.02,.04,'QP 0: outside common quality support',transform=ax.transAxes,
        fontsize=6.4,color='#596A70')
ax.legend(frameon=False, fontsize=7, loc='upper right')
ax.text(0,1.09,'a  Independent codecs · Kodak24',transform=ax.transAxes,
        fontweight='bold',fontsize=8.5)

# b: Mean of per-image BD-rate, two distinct quality metrics.
ax = axs[0, 1]
positions = {'D2': 1, 'D4': 0}
for model, color in [('D2',rose),('D4',teal)]:
    for metric, offset, marker, label in [('psnr_rgb',-.10,'o','RGB'),
                                          ('psnr_yuv611',.10,'s','YUV 6:1:1')]:
        r = next(z for z in independent['bd'] if z['model']==model and z['metric']==metric)
        x=r['mean_pct']; y=positions[model]+offset
        ax.errorbar(x,y,xerr=[[x-r['ci95_pct'][0]],[r['ci95_pct'][1]-x]],
                    fmt=marker,ms=5,capsize=2,color=color,markerfacecolor='white' if metric=='psnr_rgb' else color)
ax.set(ylim=(-.5,1.5),yticks=(0,1),yticklabels=('D4','D2'),
       xlabel='BD-rate vs released D12 (%)')
ax.axvline(0,lw=.8,color='#79858B')
ax.grid(axis='x',lw=.55,alpha=.2)
ax.text(.02,.04,'○ RGB    ■ YUV 6:1:1  ·  n = 24 images',transform=ax.transAxes,
        fontsize=6.6,color='#596A70')
ax.text(0,1.09,'b  Equal-quality coding penalty',transform=ax.transAxes,
        fontweight='bold',fontsize=8.5)

# c: Shared-exit QP loss at identical latent payload (map bits omitted).
ax = axs[1, 0]
for policy in POLICIES:
    rr = [r for r in shared['qp'] if r['policy']==policy]
    ax.plot(QPS,[r['mean_loss_db'] for r in rr],color=policy_color[policy],
            marker={'uniform':'o','dither':'^','router':'s','oracle':'D'}[policy],
            ms=3.6,lw=1.45,label=policy.capitalize())
ax.set(xlim=(-3,66),xticks=QPS,xlabel='DCVC-UF quality index (QP)',
       ylabel='YUV 6:1:1 PSNR loss vs e15 dense (dB)')
ax.grid(axis='y',lw=.55,alpha=.2)
ax.legend(frameon=False,fontsize=6.6,ncol=2,loc='upper right')
ax.text(0,1.09,'c  Shared early exit · CTC first frames',transform=ax.transAxes,
        fontweight='bold',fontsize=8.5)

# d: Coding-equivalent penalty against modelled compute saving.
ax = axs[1, 1]
for policy in POLICIES:
    r=next(z for z in shared['policy'] if z['policy']==policy)
    x=r['mac_saving_mean_pct'];y=r['bd_proxy_mean_pct']
    ax.errorbar(x,y,xerr=[[x-r['mac_saving_ci95_pct'][0]],
                          [r['mac_saving_ci95_pct'][1]-x]],
                yerr=[[y-r['bd_proxy_ci95_pct'][0]],
                      [r['bd_proxy_ci95_pct'][1]-y]],
                fmt='o',ms=5,capsize=2,color=policy_color[policy])
    offsets={'uniform':(-35,-13),'dither':(-35,8),'router':(7,-8),'oracle':(7,8)}
    ax.annotate(policy.capitalize(),(x,y),xytext=offsets[policy],
                textcoords='offset points',fontsize=6.7,color=policy_color[policy])
ax.set(xlabel='Modelled synthesis MAC saving (%)',
       ylabel='Latent-payload-only BD-rate proxy (%)')
ax.grid(axis='both',lw=.55,alpha=.2)
ax.text(0,1.09,'d  Rate–computation trade-off',transform=ax.transAxes,
        fontweight='bold',fontsize=8.5)

for row in axs:
    for ax in row:
        ax.tick_params(direction='out',length=3,width=.7)
        ax.set_axisbelow(True)
fig.text(.09,.075,'Top: final Kodak24; entropy-estimated bpp, no emitted bytes. '
         'Bottom: 51 CTC sequences, e15 rANS latent payload only; mode-map/control bits excluded.',
         fontsize=6.55,color='#4F6068')
fig.text(.09,.052,'BD-rate: per-image/sequence PCHIP over shared PSNR support; '
         'points are means, bars 95% paired bootstrap. The two rows are not a common benchmark.',
         fontsize=6.55,color='#4F6068')
for ext in ('pdf','png'):
    fig.savefig(FIG/f'bd_rate_two_systems.{ext}',dpi=300,
                facecolor='white',bbox_inches='tight')
plt.close(fig)
