"""Publication-scale Kodak beta-transfer comparison with quality-risk panel."""
from __future__ import annotations

import argparse
import json
from pathlib import Path


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--per-qp-summary', type=Path, required=True)
    parser.add_argument('--global-summary', type=Path, required=True)
    parser.add_argument('--fixed-dir', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    import numpy as np
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt

    safe = json.loads(args.per_qp_summary.read_text())
    global_policy = json.loads(args.global_summary.read_text())
    qps = [0, 16, 32, 48, 63]
    fixed = []
    for image_number in range(1, 25):
        image = f'kodim{image_number:02d}'
        per_image = []
        for qp in qps:
            row = json.loads((args.fixed_dir/f'{image}_qp{qp}.json').read_text())
            if row['qp'] != qp or Path(row['image']).stem != image:
                raise RuntimeError('Fixed cohort image/QP mismatch')
            per_image.append(row)
        fixed.append(per_image)
    fixed_delta = np.asarray([[r['fixed_beta_delta444_db'] for r in image]
                              for image in fixed])
    fixed_mac = np.asarray([[r['fixed_beta_mac_saved_pct'] for r in image]
                            for image in fixed])
    old_delta = np.asarray([[r['calibrated_delta444_db'] for r in image]
                            for image in fixed])
    old_mac = np.asarray([[r['calibrated_mac_saved_pct'] for r in image]
                          for image in fixed])
    rng = np.random.default_rng(20261004)

    def point_from_arrays(delta, mac):
        idx = rng.integers(0, 24, size=(10000, 24))
        dx = delta.mean(axis=1)[idx].mean(axis=1)
        my = mac.mean(axis=1)[idx].mean(axis=1)
        return (float(delta.mean()), float(mac.mean()),
                (float(np.quantile(dx, .025)), float(np.quantile(dx, .975))),
                (float(np.quantile(my, .025)), float(np.quantile(my, .975))))

    def point_from_summary(s):
        p = s['kodak_transfer']['equal_qp_mean']
        return (p['delta444_db']['mean'], p['mac_saved_pct']['mean'],
                tuple(p['delta444_db']['bootstrap_image_95']),
                tuple(p['mac_saved_pct']['bootstrap_image_95']))

    points = {
        'CTC β': point_from_arrays(old_delta, old_mac),
        'Fixed QP16 β': point_from_arrays(fixed_delta, fixed_mac),
        'DIV2K QP-safe β': point_from_summary(safe),
        'DIV2K global β': point_from_summary(global_policy),
    }
    curves = {
        'CTC β': old_delta.mean(axis=0),
        'Fixed QP16 β': fixed_delta.mean(axis=0),
        'DIV2K QP-safe β': np.asarray([
            safe['kodak_transfer']['per_qp'][str(q)]['delta444_db']['mean'] for q in qps]),
        'DIV2K global β': np.asarray([
            global_policy['kodak_transfer']['per_qp'][str(q)]['delta444_db']['mean'] for q in qps]),
    }
    colors = {'CTC β': '#84949C', 'Fixed QP16 β': '#D09045',
              'DIV2K QP-safe β': '#327D86', 'DIV2K global β': '#A13E61'}
    markers = {'CTC β': 'D', 'Fixed QP16 β': 's',
               'DIV2K QP-safe β': 'o', 'DIV2K global β': '^'}
    label_offsets = {'CTC β': (5, 5), 'Fixed QP16 β': (5, -17),
                     'DIV2K QP-safe β': (-102, 7), 'DIV2K global β': (5, 8)}
    plt.rcParams.update({'font.family': 'DejaVu Sans', 'font.size': 8,
                         'axes.linewidth': .7, 'pdf.fonttype': 42,
                         'svg.fonttype': 'none'})
    fig, (ax, bx) = plt.subplots(1, 2, figsize=(7.15, 3.12),
                                 gridspec_kw={'width_ratios': [1.05, 1]},
                                 constrained_layout=True)
    ax.axvspan(0, .1, color='#F2F7F4', zorder=0)
    ax.axvline(.1, color='#60866E', lw=.8, ls=(0, (3, 2)))
    ax.fill_betweenx([22, 25], 0, .1, color='#E5F0E9', zorder=0)
    for label, (x, y, xci, yci) in points.items():
        ax.errorbar(x, y, xerr=[[x-xci[0]], [xci[1]-x]],
                    yerr=[[y-yci[0]], [yci[1]-y]],
                    fmt=markers[label], ms=5.7, mew=.75, capsize=2,
                    lw=.9, color=colors[label], mec='white', zorder=4)
        ax.annotate(label, (x, y), xytext=label_offsets[label],
                    textcoords='offset points',
                    fontsize=6.6, color=colors[label])
    ax.set(xlabel='Quality loss vs. e15 full, Δ444 (dB)',
           ylabel='Analytical synthesis MAC saved (%)')
    ax.set_title('a  Kodak transfer', loc='left', fontweight='bold', pad=5)
    ax.set_xlim(-.005, .16)
    ax.set_ylim(12, 34)

    bx.axhline(.1, color='#60866E', lw=.8, ls=(0, (3, 2)))
    for label, values in curves.items():
        bx.plot(qps, values, color=colors[label], lw=1.4,
                marker=markers[label], ms=4.3, label=label)
    bx.set_xticks(qps)
    bx.set(xlabel='QP', ylabel='Mean quality loss, Δ444 (dB)')
    bx.set_title('b  Rate-specific quality risk', loc='left', fontweight='bold', pad=5)
    bx.legend(frameon=False, fontsize=6.5, loc='upper left', ncol=2,
              columnspacing=.8, handlelength=1.4, labelspacing=.3)
    for axis in (ax, bx):
        axis.spines[['top', 'right']].set_visible(False)
        axis.tick_params(direction='out', length=3, width=.7, color='#687680')
        axis.grid(axis='y', color='#E8EBED', lw=.6, zorder=0)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(args.output, bbox_inches='tight')
    plt.close(fig)
    print(json.dumps({label: {'delta444_db': value[0], 'mac_saved_pct': value[1]}
                      for label, value in points.items()}, indent=2))


if __name__ == '__main__':
    main()
