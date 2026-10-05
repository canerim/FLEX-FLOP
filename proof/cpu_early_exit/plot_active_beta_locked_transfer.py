"""Plot the calibration-locked beta transfer without selecting on validation."""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib

matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np


HERE = Path(__file__).resolve().parent
BASE = HERE / 'results/div2k_beta/quality_floor'
DATA = BASE / 'active_replicate_beta_validation24_analysis.json'
OUTPUT = BASE / 'fig_active_beta_locked_transfer.pdf'
QPS = (0, 16, 32, 48, 63)


def stat(data: dict, qp: int, field: str) -> tuple[float, tuple[float, float]]:
    record = data['per_qp'][str(qp)][field]
    return float(record['mean']), tuple(record['image_cluster_ci95'])


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--analysis', type=Path, default=DATA)
    parser.add_argument('--output', type=Path, default=OUTPUT)
    args = parser.parse_args()
    data = json.loads(args.analysis.read_text())
    assert len(data['rows']) == 120
    assert set(data['per_qp']) == {str(q) for q in QPS}
    plt.rcParams.update({'font.family': 'DejaVu Sans', 'font.size': 8,
                         'axes.linewidth': .75, 'pdf.fonttype': 42})
    fig, axes = plt.subplots(1, 2, figsize=(7.15, 2.85), constrained_layout=True)
    arms = (
        ('Original price', 'old_active_no_repair_delta444_db',
         'old_exact_conv_mac_saving_pct', '#63778B', 'o', -.85),
        ('Calibration-locked price', 'new_delta444_db',
         'new_exact_conv_mac_saving_pct', '#A45336', 's', .85),
    )
    x = np.arange(len(QPS))
    for label, quality_field, mac_field, color, marker, shift in arms:
        for ax, field in zip(axes, (quality_field, mac_field)):
            series = [stat(data, qp, field) for qp in QPS]
            means = np.asarray([v[0] for v in series])
            intervals = np.asarray([v[1] for v in series])
            errors = np.vstack((means - intervals[:, 0], intervals[:, 1] - means))
            ax.errorbar(x + shift * .08, means, yerr=errors, color=color,
                        marker=marker, markersize=4.3, markeredgecolor='white',
                        markeredgewidth=.6, linewidth=1.35, capsize=2.2,
                        elinewidth=.8, label=label, zorder=3)
    axes[0].axhline(.1, color='#8A6955', linestyle=(0, (3, 2)), linewidth=.85)
    axes[0].text(4.3, .103, '0.1 dB target', fontsize=6.3,
                 color='#8A6955', ha='right', va='bottom')
    axes[0].set_ylabel(r'Mean loss vs. full e15, $\Delta_{444}$ (dB)')
    axes[1].set_ylabel('Exact synthesis-conv MAC saved (%)')
    for i, ax in enumerate(axes):
        ax.set_xticks(x, QPS)
        ax.set_xlabel('Quality parameter (QP)')
        ax.set_xlim(-.38, 4.38)
        ax.set_title(('a  Quality transfer', 'b  Compute transfer')[i],
                     loc='left', fontweight='bold', fontsize=8.2, pad=6)
        ax.spines[['top', 'right']].set_visible(False)
        ax.grid(axis='y', color='#E7E9EB', linewidth=.6)
        ax.tick_params(direction='out', length=3, width=.7)
    axes[0].set_ylim(-.02, .18)
    axes[1].set_ylim(0, 48)
    axes[1].legend(frameon=False, fontsize=6.5, loc='upper right')
    args.output.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(args.output, bbox_inches='tight')
    plt.close(fig)
    print(args.output)


if __name__ == '__main__':
    main()
