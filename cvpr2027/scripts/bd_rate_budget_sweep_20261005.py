"""CTC early-exit latent-payload BD proxy across six archived nominal budgets.

The fixed 25-sequence cohort is complete at every budget. Per-budget complete
cohorts are also reported. Source-calibrated maps and payload-only rate make
this a diagnostic, not a deployable full-codec BD-rate measurement.
"""
from __future__ import annotations

from collections import defaultdict
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
SRC = ROOT/'data/bd_rate_20261002'
DEST = ROOT/'data/bd_rate_budget_20261005'
FIG = ROOT/'figs/bd_rate_budget_20261005'
POLICIES = ('uniform', 'dither', 'router', 'oracle')
BUDGETS = ('0.05', '0.1', '0.15', '0.2', '0.3', '0.5')
BOOT = 5000


def digest(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def bd(rows: list[dict], budget: str) -> dict[str, float]:
    curves = {}
    for name in ('dense', *POLICIES):
        pairs = sorted((r['psnr_release'] if name == 'dense' else
                        r['rules'][name][budget]['psnr'], r['bpp_real']) for r in rows)
        quality = np.array([v[0] for v in pairs])
        log_rate = np.log(np.array([v[1] for v in pairs]))
        if not (len(pairs) == 5 and np.all(np.diff(quality) > 0)
                and np.all(np.diff(log_rate) > 0)):
            raise ValueError(f'Nonmonotone RD curve: {name}, budget {budget}')
        curves[name] = (quality, PchipInterpolator(quality, log_rate, extrapolate=False))
    low = max(v[0][0] for v in curves.values())
    high = min(v[0][-1] for v in curves.values())
    if high <= low:
        raise ValueError(f'No common quality support at budget {budget}')
    baseline = curves['dense'][1]
    return {name: 100*math.expm1((spline.integrate(low, high) -
                                  baseline.integrate(low, high))/(high-low))
            for name, (_, spline) in curves.items() if name != 'dense'}


def summary(values: list[float], rng: np.random.Generator) -> dict:
    x = np.asarray(values, dtype=float)
    samples = x[rng.integers(0, len(x), size=(BOOT, len(x)))].mean(axis=1)
    return {'mean': float(x.mean()), 'ci95': [float(v) for v in np.quantile(samples, (.025, .975))]}


def main() -> None:
    archive_path = SRC/'eval_rules_ctc_e15.json'
    payload_path = SRC/'real_bitstream_ctc.json'
    archive = json.loads(archive_path.read_text())
    payload = json.loads(payload_path.read_text())
    assert len(archive['rows']) == len(payload) == 265
    assert {str(x) for x in archive['budgets']} == set(BUDGETS)
    payload_by_key = {(r['seq'], r['qp']): r for r in payload}
    assert len(payload_by_key) == 265 and all(r['roundtrip_ok'] for r in payload)
    grouped = defaultdict(list)
    for row in archive['rows']:
        assert math.isclose(row['bpp_real'], payload_by_key[(row['seq'], row['qp'])]['real_bpp'], abs_tol=1e-12)
        grouped[row['seq']].append(row)
    assert len(grouped) == 53
    complete = {budget: sorted(seq for seq, rows in grouped.items()
                               if len(rows) == 5 and all(r['rules'][p][budget] is not None
                                                         for r in rows for p in POLICIES))
                for budget in BUDGETS}
    fixed = sorted(set.intersection(*(set(v) for v in complete.values())))
    assert len(fixed) == 25
    rng = np.random.default_rng(20261005)
    result = {'schema': 1,
              'scope': 'CTC first-frame, five QPs, source-calibrated nominal maps; latent-payload BD-rate proxy',
              'source_sha256': {str(archive_path.relative_to(ROOT)): digest(archive_path),
                                str(payload_path.relative_to(ROOT)): digest(payload_path)},
              'rate_includes_map_or_container_bits': False,
              'bootstrap': f'{BOOT} sequence-cluster draws, seed 20261005',
              'fixed_cohort_sequences': fixed, 'rows': []}
    for cohort in ('fixed25', 'budget_complete'):
        for budget in BUDGETS:
            sequences = fixed if cohort == 'fixed25' else complete[budget]
            per_seq = []
            for seq in sequences:
                rows = sorted(grouped[seq], key=lambda r: r['qp'])
                rates = bd(rows, budget)
                savings = {p: float(np.mean([r['rules'][p][budget]['saving'] for r in rows]))
                           for p in POLICIES}
                losses = {p: float(np.mean([r['psnr_release'] - r['rules'][p][budget]['psnr']
                                            for r in rows])) for p in POLICIES}
                per_seq.append({'seq': seq, 'bd': rates, 'saving': savings, 'loss': losses})
            row = {'cohort': cohort, 'nominal_budget_db': float(budget),
                   'n_sequences': len(sequences), 'n_frame_qp_pairs': 5*len(sequences),
                   'policies': {}, 'paired_router_minus_dither': {}}
            for p in POLICIES:
                row['policies'][p] = {
                    'bd_proxy_pct': summary([x['bd'][p] for x in per_seq], rng),
                    'mac_saving_pct': summary([x['saving'][p] for x in per_seq], rng),
                    'achieved_yuv611_loss_db': summary([x['loss'][p] for x in per_seq], rng)}
            for key, label in [('bd', 'bd_proxy_pct_points'),
                               ('saving', 'mac_saving_pct_points'),
                               ('loss', 'achieved_yuv611_loss_db')]:
                row['paired_router_minus_dither'][label] = summary(
                    [x[key]['router']-x[key]['dither'] for x in per_seq], rng)
            result['rows'].append(row)
    DEST.mkdir(parents=True, exist_ok=True)
    FIG.mkdir(parents=True, exist_ok=True)
    (DEST/'analysis.json').write_text(json.dumps(result, indent=2)+'\n')

    plt.rcParams.update({'font.family': 'DejaVu Sans', 'font.size': 8,
                         'pdf.fonttype': 42, 'axes.spines.top': False,
                         'axes.spines.right': False})
    colors = {'uniform':'#7A8791', 'dither':'#C08C3A',
              'router':'#087F87', 'oracle':'#AE6172'}
    fig, axes = plt.subplots(1, 3, figsize=(10.2, 3.0), constrained_layout=True)
    fixed_rows = [r for r in result['rows'] if r['cohort']=='fixed25']
    x = np.array([r['nominal_budget_db'] for r in fixed_rows])
    for p in POLICIES:
        for ax, metric in ((axes[0], 'bd_proxy_pct'), (axes[1], 'mac_saving_pct')):
            values = [r['policies'][p][metric] for r in fixed_rows]
            y = np.array([v['mean'] for v in values])
            lo = np.array([v['ci95'][0] for v in values])
            hi = np.array([v['ci95'][1] for v in values])
            ax.plot(x, y, marker='o', ms=3.2, lw=1.5, color=colors[p], label=p.title())
            ax.fill_between(x, lo, hi, color=colors[p], alpha=.10, linewidth=0)
    axes[0].set(ylabel='Latent-payload BD proxy (%)', title='a  Quality cost')
    axes[1].set(ylabel='Analytical synthesis MAC saved (%)', title='b  Compute saving')
    for metric, color, label in [('mac_saving_pct_points','#087F87','MAC saving'),
                                 ('bd_proxy_pct_points','#AE6172','BD proxy')]:
        vals = [r['paired_router_minus_dither'][metric] for r in fixed_rows]
        y = np.array([v['mean'] for v in vals])
        lo = np.array([v['ci95'][0] for v in vals])
        hi = np.array([v['ci95'][1] for v in vals])
        axes[2].plot(x, y, marker='o', ms=3.2, lw=1.5, color=color, label=label)
        axes[2].fill_between(x, lo, hi, color=color, alpha=.12, linewidth=0)
    axes[2].axhline(0, color='#65747D', lw=.75, ls='--')
    axes[2].set(ylabel='Router − dither (percentage points)',
                title='c  Paired routing margin')
    for ax in axes:
        ax.set_xlabel('Nominal YCbCr444-loss target (dB)')
        ax.set_xticks(x)
        ax.grid(axis='y', color='#E3E8E9', lw=.6)
        ax.tick_params(direction='out', length=2.5)
    axes[0].legend(frameon=False, fontsize=6.8, ncol=2)
    axes[2].legend(frameon=False, fontsize=6.8)
    fig.text(.5, -.01, 'Fixed 25-sequence cohort complete at all six targets · 5 QPs · 95% sequence bootstrap intervals',
             ha='center', color='#586A73', fontsize=7)
    for suffix in ('pdf', 'png'):
        fig.savefig(FIG/f'bd_rate_budget_sweep.{suffix}', dpi=240, bbox_inches='tight')
    plt.close(fig)
    for row in result['rows']:
        q = row['paired_router_minus_dither']
        print(row['cohort'],row['nominal_budget_db'],row['n_sequences'],
              'router-dither BD',round(q['bd_proxy_pct_points']['mean'],3),
              'MAC',round(q['mac_saving_pct_points']['mean'],3))


if __name__ == '__main__':
    main()
