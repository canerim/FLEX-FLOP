"""Create an audit-ready vector plot for independent beta calibration."""
from __future__ import annotations

import argparse
import json
from pathlib import Path


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--summary', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    from matplotlib.ticker import MultipleLocator

    summary = json.loads(args.summary.read_text())
    plt.rcParams.update({'font.family': 'DejaVu Sans', 'font.size': 8,
                         'axes.linewidth': .7, 'pdf.fonttype': 42,
                         'svg.fonttype': 'none', 'savefig.transparent': False})
    fig, axes = plt.subplots(1, 2, figsize=(7.15, 3.05),
                             gridspec_kw={'width_ratios': [1.05, 1]},
                             constrained_layout=True)
    ink = '#273746'
    colors = {'calibration': '#3A7A83', 'validation': '#D28A3D',
              'kodak_transfer': '#A23C58'}
    names = {'calibration': 'DIV2K fit (24)',
             'validation': 'DIV2K held out (24)',
             'kodak_transfer': 'Kodak transfer (24)'}

    ax = axes[0]
    ax.axvspan(-.05, .1, color='#E8F2EC', zorder=0)
    ax.axhspan(22, 25, xmax=.72, color='#EFF2E8', zorder=0)
    ax.axvline(.1, color='#66836C', lw=.8, ls=(0, (3, 2)))
    for name in ['calibration', 'validation', 'kodak_transfer']:
        point = summary[name]['equal_qp_mean']
        xx = point['delta444_db']['mean']
        yy = point['mac_saved_pct']['mean']
        xci = point['delta444_db']['bootstrap_image_95']
        yci = point['mac_saved_pct']['bootstrap_image_95']
        ax.errorbar(xx, yy, xerr=[[xx-xci[0]], [xci[1]-xx]],
                    yerr=[[yy-yci[0]], [yci[1]-yy]],
                    fmt='o', ms=6, mew=.8, capsize=2.2, lw=.9,
                    color=colors[name], mec='white', label=names[name], zorder=4)
    prior = summary['kodak_transfer']['equal_qp_mean']
    ax.scatter(prior['old_ctc_calibrated_delta444_db'],
               prior['old_ctc_calibrated_mac_saved_pct'], marker='D', s=31,
               facecolor='white', edgecolor=ink, linewidth=.9,
               label='Kodak: prior CTC β', zorder=5)
    ax.set(xlabel='Quality loss vs. e15 full, Δ444 (dB)',
           ylabel='Analytical synthesis MAC saved (%)')
    ax.xaxis.set_major_locator(MultipleLocator(.05))
    ax.yaxis.set_major_locator(MultipleLocator(5))
    ax.legend(frameon=False, fontsize=7, loc='best', labelspacing=.4)
    ax.text(.01, .99, 'a  Locked policy transfer', transform=ax.transAxes,
            va='top', ha='left', color=ink, fontweight='bold')

    ax = axes[1]
    qps = [0, 16, 32, 48, 63]
    xs = range(len(qps))
    # Old CTC policy per-QP means are embedded in the transfer summary.
    old_per_qp = summary['kodak_transfer'].get('old_ctc_per_qp_mac_saved_pct')
    if old_per_qp is None:
        raise RuntimeError('Summary lacks per-QP prior CTC MAC values')
    old = [old_per_qp[str(q)] for q in qps]
    new = [summary['kodak_transfer']['per_qp'][str(q)]['mac_saved_pct']['mean']
           for q in qps]
    ax.plot(xs, old, color='#9AA6AC', lw=1.2, marker='o', ms=4,
            mfc='white', label='Prior CTC β')
    ax.plot(xs, new, color=colors['kodak_transfer'], lw=1.5,
            marker='o', ms=4.5, label='DIV2K-locked β')
    ax.set_xticks(list(xs), [f'{q}' for q in qps])
    ax.set(xlabel='QP', ylabel='Analytical synthesis MAC saved (%)')
    ax.yaxis.set_major_locator(MultipleLocator(10))
    ax.legend(frameon=False, fontsize=7, loc='best')
    ax.text(.01, .99, 'b  Rate dependence on Kodak', transform=ax.transAxes,
            va='top', ha='left', color=ink, fontweight='bold')

    for ax in axes:
        ax.spines[['top', 'right']].set_visible(False)
        ax.tick_params(direction='out', length=3, width=.7, color='#697781')
        ax.grid(axis='y', color='#E7EBED', lw=.6, zorder=0)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(args.output, bbox_inches='tight')
    plt.close(fig)


if __name__ == '__main__':
    main()
