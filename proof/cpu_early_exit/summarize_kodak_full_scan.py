"""Validate Kodak24 x QP5 audit, report equal-QP means and draw its figure."""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path


def sha(path):
    digest = hashlib.sha256()
    with Path(path).open('rb') as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b''):
            digest.update(chunk)
    return digest.hexdigest()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--cohort', type=Path, required=True)
    parser.add_argument('--fixed-cohort', type=Path,
                        help='Optional actual fixed-beta reconstructions on the same streams')
    parser.add_argument('--out-prefix', type=Path, required=True)
    args = parser.parse_args()
    import numpy as np
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt

    manifest_path = args.cohort/'manifest.json'
    manifest = json.loads(manifest_path.read_text())
    if manifest['cases'] != 120 or len(manifest['images']) != 24 or manifest['qps'] != [0, 16, 32, 48, 63]:
        raise RuntimeError('Expected predeclared Kodak24 x QP5 manifest')
    manifest_hash = sha(manifest_path)
    cases = []
    for name in manifest['images']:
        for qp in manifest['qps']:
            path = args.cohort/f'{Path(name).stem}_qp{qp}.json'
            row = json.loads(path.read_text())
            if (row['manifest_sha256'] != manifest_hash or row['image'] != name or
                    row['qp'] != qp or sorted(row['shape']) != [512, 768] or
                    sum(row['exit_counts']) != 6 or
                    sum(row['fixed_beta_exit_counts']) != 6):
                raise RuntimeError(f'Provenance/geometry mismatch: {path}')
            if row['cpu_timing'] is not None:
                raise RuntimeError('This scan must remain untimed')
            if row['quality_loss_db'] != row['released_yuv611_db']-row['routed_yuv611_db']:
                raise RuntimeError(f'Quality difference inconsistent: {path}')
            if (args.cohort/f'{Path(name).stem}_qp{qp}.fufref2').exists():
                if sha(args.cohort/f'{Path(name).stem}_qp{qp}.fufref2') != row['stream_sha256']:
                    raise RuntimeError(f'Stream hash mismatch: {path}')
            cases.append(row)
    if len(cases) != 120:
        raise RuntimeError('Incomplete cohort')
    fixed_cases = {}
    if args.fixed_cohort:
        for row in cases:
            key = (row['image'], row['qp'])
            source_path = args.cohort/f'{Path(row["image"]).stem}_qp{row["qp"]}.json'
            path = args.fixed_cohort/f'{Path(row["image"]).stem}_qp{row["qp"]}.json'
            fixed = json.loads(path.read_text())
            if (fixed['scan_manifest_sha256'] != manifest_hash or
                    fixed['scan_result_sha256'] != sha(source_path) or
                    fixed['stream_sha256'] != row['stream_sha256'] or
                    fixed['image'] != row['image'] or fixed['qp'] != row['qp']):
                raise RuntimeError(f'Fixed-beta provenance mismatch: {path}')
            fixed_cases[key] = fixed
        if len(fixed_cases) != 120:
            raise RuntimeError('Incomplete fixed-beta cohort')
    qps = manifest['qps']
    by_qp = {q: [r for r in cases if r['qp'] == q] for q in qps}
    rng = np.random.default_rng(20261004)

    def interval(values):
        values = np.asarray(values, dtype=float)
        draws = rng.choice(values, (10000, 24), replace=True).mean(axis=1)
        return [float(np.quantile(draws, .025)), float(np.quantile(draws, .975))]

    per_qp = {}
    for qp in qps:
        group = by_qp[qp]
        mac = [r['mac_saved_pct'] for r in group]
        fixed = [r['fixed_beta_mac_saved_pct'] for r in group]
        loss = [r['quality_loss_db'] for r in group]
        per_qp[str(qp)] = {
            'calibrated_beta': manifest['calibrated_beta'][str(qp)],
            'exit_share': (np.sum([r['exit_counts'] for r in group], axis=0)/144).tolist(),
            'fixed_beta_exit_share': (np.sum([r['fixed_beta_exit_counts'] for r in group], axis=0)/144).tolist(),
            'mac_saved_pct_mean': float(np.mean(mac)),
            'mac_saved_pct_image_bootstrap_95': interval(mac),
            'fixed_beta_mac_saved_pct_mean': float(np.mean(fixed)),
            'fixed_beta_mac_saved_pct_image_bootstrap_95': interval(fixed),
            'quality_loss_db_mean': float(np.mean(loss)),
            'quality_loss_db_median': float(np.median(loss)),
            'quality_loss_db_image_bootstrap_95': interval(loss),
            'quality_loss_db_max': float(max(loss)),
            'released_relative_yuv_loss_over_0p1_count': sum(v > .1 for v in loss),
            'stream_bpp_mean': float(np.mean([r['stream_bytes']*8/(512*768) for r in group])),
            'released_yuv611_db_mean': float(np.mean([r['released_yuv611_db'] for r in group])),
            'routed_yuv611_db_mean': float(np.mean([r['routed_yuv611_db'] for r in group])),
        }
        if fixed_cases:
            fixed_losses = [fixed_cases[(r['image'], qp)]['fixed_beta_loss_db'] for r in group]
            calibrated_delta = [fixed_cases[(r['image'], qp)]['calibrated_delta444_db'] for r in group]
            fixed_delta = [fixed_cases[(r['image'], qp)]['fixed_beta_delta444_db'] for r in group]
            per_qp[str(qp)].update({
                'fixed_beta_quality_loss_db_mean': float(np.mean(fixed_losses)),
                'fixed_beta_quality_loss_db_max': float(max(fixed_losses)),
                'fixed_beta_released_relative_yuv_loss_over_0p1_count': sum(v > .1 for v in fixed_losses),
                'calibrated_delta444_db_mean': float(np.mean(calibrated_delta)),
                'calibrated_delta444_db_image_bootstrap_95': interval(calibrated_delta),
                'calibrated_delta444_db_max': float(max(calibrated_delta)),
                'calibrated_delta444_over_0p1_count': sum(v > .1 for v in calibrated_delta),
                'fixed_beta_delta444_db_mean': float(np.mean(fixed_delta)),
                'fixed_beta_delta444_db_image_bootstrap_95': interval(fixed_delta),
                'fixed_beta_delta444_db_max': float(max(fixed_delta)),
                'fixed_beta_delta444_over_0p1_count': sum(v > .1 for v in fixed_delta),
            })
    per_image_mac = np.array([[next(r['mac_saved_pct'] for r in by_qp[q] if r['image'] == name)
                               for q in qps] for name in manifest['images']])
    per_image_fixed = np.array([[next(r['fixed_beta_mac_saved_pct'] for r in by_qp[q] if r['image'] == name)
                                 for q in qps] for name in manifest['images']])
    image_draws = rng.integers(0, 24, size=(10000, 24))
    aggregate_interval = np.quantile(per_image_mac[image_draws].mean(axis=(1, 2)), [.025, .975])
    policy_gap_interval = np.quantile((per_image_fixed-per_image_mac)[image_draws].mean(axis=(1, 2)), [.025, .975])
    aggregate = {
        'weighting': 'Equal weight for each of 24 Kodak images and each of 5 QPs; all images are 512x768 and have six tiles, so image/tile/pixel weighting coincide within this grid. This is not a deployment bitrate mix.',
        'calibrated_mac_saved_pct_mean': float(np.mean([r['mac_saved_pct'] for r in cases])),
        'calibrated_mac_saved_pct_image_cluster_bootstrap_95': aggregate_interval.tolist(),
        'fixed_beta_mac_saved_pct_mean': float(np.mean([r['fixed_beta_mac_saved_pct'] for r in cases])),
        'fixed_minus_calibrated_mac_points_image_cluster_bootstrap_95': policy_gap_interval.tolist(),
        'calibrated_quality_loss_db_mean': float(np.mean([r['quality_loss_db'] for r in cases])),
        'calibrated_quality_loss_db_max': float(max(r['quality_loss_db'] for r in cases)),
        'released_relative_yuv_loss_over_0p1_count': sum(r['quality_loss_db'] > .1 for r in cases),
        'fixed_beta_quality_status': 'Not measured: fixed-beta numbers are decision/MAC counterfactuals only',
    }
    if fixed_cases:
        fixed_losses = [fixed_cases[(r['image'], r['qp'])]['fixed_beta_loss_db'] for r in cases]
        calibrated_delta = [fixed_cases[(r['image'], r['qp'])]['calibrated_delta444_db'] for r in cases]
        fixed_delta = [fixed_cases[(r['image'], r['qp'])]['fixed_beta_delta444_db'] for r in cases]
        calibrated_delta_by_image = np.array([[fixed_cases[(name, q)]['calibrated_delta444_db']
                                               for q in qps] for name in manifest['images']])
        fixed_delta_by_image = np.array([[fixed_cases[(name, q)]['fixed_beta_delta444_db']
                                          for q in qps] for name in manifest['images']])
        aggregate.update({
            'fixed_beta_quality_status': 'Measured from the same FUFREF2 streams',
            'fixed_beta_quality_loss_db_mean': float(np.mean(fixed_losses)),
            'fixed_beta_quality_loss_db_max': float(max(fixed_losses)),
            'fixed_beta_released_relative_yuv_loss_over_0p1_count': sum(v > .1 for v in fixed_losses),
            'fixed_beta_vs_calibrated_quality_loss_db_mean': float(np.mean(fixed_losses)-np.mean([r['quality_loss_db'] for r in cases])),
            'calibrated_delta444_db_mean': float(np.mean(calibrated_delta)),
            'calibrated_delta444_db_image_cluster_bootstrap_95': np.quantile(calibrated_delta_by_image[image_draws].mean(axis=(1, 2)), [.025, .975]).tolist(),
            'calibrated_delta444_db_max': float(max(calibrated_delta)),
            'calibrated_delta444_over_0p1_count': sum(v > .1 for v in calibrated_delta),
            'fixed_beta_delta444_db_mean': float(np.mean(fixed_delta)),
            'fixed_beta_delta444_db_image_cluster_bootstrap_95': np.quantile(fixed_delta_by_image[image_draws].mean(axis=(1, 2)), [.025, .975]).tolist(),
            'fixed_minus_calibrated_delta444_db_image_cluster_bootstrap_95': np.quantile((fixed_delta_by_image-calibrated_delta_by_image)[image_draws].mean(axis=(1, 2)), [.025, .975]).tolist(),
            'fixed_beta_delta444_db_max': float(max(fixed_delta)),
            'fixed_beta_delta444_over_0p1_count': sum(v > .1 for v in fixed_delta),
        })
    summary = {'schema': 1, 'manifest_sha256': manifest_hash,
               'calibration_sha256': manifest['calibration_sha256'],
               'scope': manifest['scope'], 'limitations': manifest['backend'],
               'interpretation_note': 'Holding beta at its QP16 value isolates the effect of QP-specific control calibration. It does not remove QP from the router head or erase QP-dependent latent statistics; causal decoder-capacity demand is not identified by this audit. Released-relative weighted YUV PSNR loss is not the same as e15-full-frame-referenced Delta444 used for the nominal 0.1 dB target.',
               'per_qp': per_qp, 'aggregate': aggregate,
               'raw_result_sha256': {f'{r["image"]}:qp{r["qp"]}': sha(args.cohort/f'{Path(r["image"]).stem}_qp{r["qp"]}.json') for r in cases}}
    if fixed_cases:
        summary['fixed_result_sha256'] = {f'{image}:qp{qp}': sha(args.fixed_cohort/f'{Path(image).stem}_qp{qp}.json')
                                          for image,qp in fixed_cases}

    plt.rcParams.update({'font.family': 'Liberation Sans', 'font.size': 8.2,
                         'axes.labelcolor': '#20313E', 'text.color': '#20313E',
                         'xtick.color': '#42526A', 'ytick.color': '#42526A',
                         'axes.spines.top': False, 'axes.spines.right': False,
                         'pdf.fonttype': 42})
    fig, axes = plt.subplots(1, 3, figsize=(7.2, 2.72),
                             gridspec_kw={'width_ratios': [1.05, 1.1, 1]})
    x = np.arange(5)
    colors = {2: '#278B92', 3: '#22738A', 4: '#245B7A', 5: '#243F5C'}
    bottom = np.zeros(5)
    for exit_idx in range(2, 6):
        shares = np.array([per_qp[str(q)]['exit_share'][exit_idx]*100 for q in qps])
        axes[0].bar(x, shares, bottom=bottom, color=colors[exit_idx],
                    edgecolor='white', linewidth=.5, width=.68,
                    label=f'exit {exit_idx}')
        bottom += shares
    axes[0].set_ylim(0, 100)
    axes[0].set_ylabel('Tiles at exit (%)')
    axes[0].set_title('a  Routed allocation', loc='left', fontweight='bold', fontsize=9)
    axes[0].legend(loc='upper center', bbox_to_anchor=(.5, -.21), ncol=4,
                   frameon=False, fontsize=6.9, columnspacing=.4)

    for ax in axes[1:]:
        ax.grid(axis='y', color='#E5EAF0', lw=.6)
        ax.axhline(0, color='#AEBCC9', lw=.8)
    mac = [per_qp[str(q)]['mac_saved_pct_mean'] for q in qps]
    fixed = [per_qp[str(q)]['fixed_beta_mac_saved_pct_mean'] for q in qps]
    axes[1].plot(x, mac, '-o', color='#008A96', lw=1.5, ms=4,
                 label='QP-calibrated β', zorder=4)
    axes[1].plot(x, fixed, '--s', color='#9A694B', lw=1.2, ms=3.8,
                 label='fixed β (QP16)', zorder=4)
    for j, qp in enumerate(qps):
        values = np.array([r['mac_saved_pct'] for r in by_qp[qp]])
        axes[1].scatter(np.full(24, j)-.045, values, s=6, color='#9FBFC1',
                        alpha=.48, linewidth=0, zorder=2)
    axes[1].set_ylabel('Synthesis MAC saved (%)')
    axes[1].set_title('b  Compute allocation', loc='left', fontweight='bold', fontsize=9)
    axes[1].legend(loc='upper center', bbox_to_anchor=(.5, -.21), ncol=2,
                   frameon=False, fontsize=6.7, columnspacing=.7)

    for j, qp in enumerate(qps):
        values = np.array([fixed_cases[(r['image'], qp)]['calibrated_delta444_db']
                           for r in by_qp[qp]]) if fixed_cases else np.array([r['quality_loss_db'] for r in by_qp[qp]])
        axes[2].scatter(np.full(24, j), values, s=7, color='#9FBFC1',
                        alpha=.55, linewidth=0, zorder=2)
    quality_key = 'calibrated_delta444_db_mean' if fixed_cases else 'quality_loss_db_mean'
    axes[2].plot(x, [per_qp[str(q)][quality_key] for q in qps],
                 '-o', color='#20313E', lw=1.5, ms=4, zorder=4,
                 label='QP-calibrated β')
    if fixed_cases:
        axes[2].plot(x, [per_qp[str(q)]['fixed_beta_delta444_db_mean'] for q in qps],
                     '--s', color='#9A694B', lw=1.2, ms=3.8, zorder=4,
                     label='fixed β (QP16)')
        axes[2].legend(loc='upper center', bbox_to_anchor=(.5, -.21), ncol=2,
                       frameon=False, fontsize=6.7, columnspacing=.7)
        axes[2].axhline(.1, color='#BB7D56', lw=.8, linestyle=':', zorder=1)
    axes[2].set_ylabel('Δ444 vs e15 full (dB)' if fixed_cases else 'YUV PSNR loss (dB)')
    axes[2].set_title('c  Quality cost', loc='left', fontweight='bold', fontsize=9)
    for ax in axes:
        ax.set_xticks(x, [str(q) for q in qps])
        ax.set_xlabel('QP')
        ax.set_xlim(-.4, 4.4)
    fig.subplots_adjust(left=.075, right=.995, top=.88, bottom=.28, wspace=.46)
    args.out_prefix.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(args.out_prefix.with_suffix('.pdf'), facecolor='white')
    fig.savefig(args.out_prefix.with_suffix('.png'), facecolor='white', dpi=320)
    plt.close(fig)
    summary['figure_script_sha256'] = sha(Path(__file__))
    args.out_prefix.with_suffix('.json').write_text(json.dumps(summary, indent=2)+'\n')
    print(json.dumps({'per_qp': per_qp, 'aggregate': aggregate}, indent=2))


if __name__ == '__main__':
    main()
