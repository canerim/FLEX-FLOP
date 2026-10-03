"""Plot measured Kodak3x3 routing, quality and analytical decoder MAC cost.

Latency is deliberately omitted until a claim-eligible, matched CPU run exists.
"""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))


def sha(path):
    digest = hashlib.sha256()
    with Path(path).open('rb') as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b''):
            digest.update(chunk)
    return digest.hexdigest()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--cohort', type=Path, required=True)
    parser.add_argument('--manifest', type=Path,
                        default=ROOT/'proof/early_exit_vs_released/results/bitstream_kodak3x3/manifest.json')
    parser.add_argument('--checkpoint', type=Path,
                        default=ROOT/'proof/early_exit_vs_released/artifacts/e15_epoch15.pth.tar')
    parser.add_argument('--out-prefix', type=Path,
                        default=ROOT/'cvpr2027/figs/cpu_qp/fig_cpu_qp_route_quality_mac')
    args = parser.parse_args()

    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    import numpy as np
    import torch
    from flexuf.config import FlexUFConfig
    from flexuf.cost import frame_relative_cost

    manifest = json.loads(args.manifest.read_text())
    checkpoint = torch.load(args.checkpoint, map_location='cpu', weights_only=False)
    cfg = FlexUFConfig(**checkpoint['config'])
    expected_checkpoint = manifest['released_checkpoint_sha256']
    records = []
    for item in manifest['rows']:
        path = args.cohort/(Path(item['stream']).stem+'.json')
        if not path.exists():
            raise FileNotFoundError(f'Missing predeclared cohort case: {path}')
        row = json.loads(path.read_text())
        if (row['stream_sha256'] != item['stream_sha256'] or
                row['source_sha256'] != item['image_sha256'] or
                row['released_sha256'] != expected_checkpoint or
                row['qp'] != item['qp'] or
                row['shape'] != item['shape']):
            raise RuntimeError(f'Provenance mismatch for {path}')
        if sum(row['route_counts']) != (item['shape'][0]//256)*(item['shape'][1]//256):
            raise RuntimeError(f'Tile count mismatch for {path}')
        if row['timing_ms_raw']['released_cpu'] or row['timing_ms_raw']['e15_routed_cpu']:
            raise RuntimeError('This figure is the untimed cohort; latency belongs in a separate audit')
        exit_map = torch.repeat_interleave(torch.arange(cfg.num_exits),
                                           torch.tensor(row['route_counts']))
        cost = frame_relative_cost(exit_map, cfg, 'head')
        quality = row['quality']
        records.append({'image': item['image'], 'qp': item['qp'],
                        'route_counts': row['route_counts'],
                        'tiles': int(exit_map.numel()),
                        'released_yuv611_db': quality['released_cpu']['yuv_6_1_1'],
                        'routed_yuv611_db': quality['e15_routed_cpu']['yuv_6_1_1'],
                        'all_deep_yuv611_db': quality['e15_all_deep_cpu']['yuv_6_1_1'],
                        'quality_loss_db': (quality['released_cpu']['yuv_6_1_1'] -
                                            quality['e15_routed_cpu']['yuv_6_1_1']),
                        'relative_mac': float(cost),
                        'mac_saved_pct': 100*(1-float(cost)),
                        'stream_sha256': item['stream_sha256'],
                        'run_commit': row['git_commit'],
                        'run_script_sha256': row['script_sha256']})
    if len(records) != manifest['cases'] or len(records) != 9:
        raise RuntimeError('Expected exactly the predeclared Kodak3x3 cohort')
    if len({r['run_commit'] for r in records}) != 1 or len({r['run_script_sha256'] for r in records}) != 1:
        raise RuntimeError('Cohort mixes different implementation versions')

    qps = sorted({row['qp'] for row in records})
    images = sorted({row['image'] for row in records})
    if qps != [16, 32, 48] or len(images) != 3:
        raise RuntimeError('Unexpected cohort axes')
    groups = {qp: [row for row in records if row['qp'] == qp] for qp in qps}
    per_qp = {}
    for qp in qps:
        group = groups[qp]
        counts = np.array([row['route_counts'] for row in group], dtype=float)
        per_qp[str(qp)] = {
            'exit_share': (counts.sum(axis=0)/counts.sum()).tolist(),
            'quality_loss_db_mean': float(np.mean([r['quality_loss_db'] for r in group])),
            'mac_saved_pct_mean': float(np.mean([r['mac_saved_pct'] for r in group])),
            'quality_loss_db_range': [min(r['quality_loss_db'] for r in group),
                                      max(r['quality_loss_db'] for r in group)],
            'mac_saved_pct_range': [min(r['mac_saved_pct'] for r in group),
                                    max(r['mac_saved_pct'] for r in group)]}

    plt.rcParams.update({'font.family': 'Liberation Sans', 'font.size': 8.2,
                         'axes.labelcolor': '#20313E', 'text.color': '#20313E',
                         'xtick.color': '#42526A', 'ytick.color': '#42526A',
                         'axes.spines.top': False, 'axes.spines.right': False,
                         'pdf.fonttype': 42})
    fig, axes = plt.subplots(1, 3, figsize=(7.2, 2.75),
                             gridspec_kw={'width_ratios': [1.18, 1, 1]})
    ax0, ax1, ax2 = axes
    x = np.arange(3)
    exit_colors = {2: '#278B92', 3: '#22738A', 4: '#245B7A', 5: '#243F5C'}
    bottom = np.zeros(3)
    for exit_idx in range(cfg.split_depth, cfg.num_exits):
        share = np.array([per_qp[str(qp)]['exit_share'][exit_idx]*100 for qp in qps])
        ax0.bar(x, share, bottom=bottom, color=exit_colors[exit_idx],
                width=.61, label=f'exit {exit_idx}', edgecolor='white', linewidth=.65)
        bottom += share
    ax0.set_ylim(0, 100)
    ax0.set_xticks(x, [str(qp) for qp in qps])
    ax0.set_ylabel('Tiles at exit (%)')
    ax0.set_xlabel('QP')
    ax0.set_title('a  Allocation', loc='left', fontweight='bold', pad=9, fontsize=9)
    ax0.legend(loc='upper center', bbox_to_anchor=(.5, -.20), ncol=4,
               frameon=False, fontsize=7.1, columnspacing=.5)

    markers = ['o', 's', 'D']
    image_colors = ['#008A96', '#BF783D', '#81909C']
    for ax, key, ylabel, title in [
        (ax1, 'quality_loss_db', 'YUV PSNR loss (dB)', 'b  Distortion'),
        (ax2, 'mac_saved_pct', 'Synthesis MAC saved (%)', 'c  Compute')]:
        means = [float(np.mean([r[key] for r in groups[qp]])) for qp in qps]
        ax.plot(x, means, color='#20313E', lw=1.5, zorder=2)
        for i, image_name in enumerate(images):
            values = [next(r[key] for r in groups[qp] if r['image'] == image_name)
                      for qp in qps]
            ax.scatter(x + (i-1)*.075, values, s=24, marker=markers[i],
                       color=image_colors[i], edgecolor='white', linewidth=.5,
                       zorder=3, label=image_name.replace('.png', ''))
        ax.axhline(0, color='#B5C1CF', lw=.8, zorder=1)
        ax.set_xticks(x, [str(qp) for qp in qps])
        ax.set_xlim(-.35, 2.35)
        ax.set_xlabel('QP')
        ax.set_ylabel(ylabel)
        ax.set_title(title, loc='left', fontweight='bold', pad=9, fontsize=9)
        ax.grid(axis='y', color='#E5EAF0', lw=.65)
    ax1.legend(loc='upper center', bbox_to_anchor=(.5, -.20), ncol=3,
               frameon=False, fontsize=7.1, columnspacing=.5)
    fig.subplots_adjust(left=.082, right=.99, top=.86, bottom=.27, wspace=.5)
    args.out_prefix.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(args.out_prefix.with_suffix('.pdf'), facecolor='white')
    fig.savefig(args.out_prefix.with_suffix('.png'), dpi=320, facecolor='white')
    plt.close(fig)
    summary = {'schema': 1, 'scope': 'Kodak RGB -> YCbCr 4:4:4, FUFREF2 research stream, synthesis-only quality and analytical synthesis MAC excluding router; no CPU latency',
               'manifest': str(args.manifest), 'checkpoint': str(args.checkpoint),
               'manifest_sha256': sha(args.manifest),
               'checkpoint_sha256': sha(args.checkpoint),
               'cost_model_sha256': sha(ROOT/'flexuf/cost.py'),
               'plot_script_sha256': sha(Path(__file__)),
               'qps': qps, 'images': images, 'per_qp': per_qp, 'rows': records,
               'latency_status': 'unmeasured; host loaded during cohort quality run'}
    args.out_prefix.with_suffix('.json').write_text(json.dumps(summary, indent=2)+'\n')
    print(json.dumps(per_qp, indent=2))


if __name__ == '__main__':
    main()
