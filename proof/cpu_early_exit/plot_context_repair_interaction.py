"""Compact vector interaction plot for the matched DIV2K 2x2 decoder audit."""
from __future__ import annotations

import hashlib
import json
from pathlib import Path

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

ROOT = Path(__file__).resolve().parents[2]
SOURCE = ROOT / 'proof/cpu_early_exit/results/div2k_beta/quality_floor/isolated_no_repair_decomposition.json'
OUT = ROOT / 'docs/figures/active-canvas-transfer-20261005'


def sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main() -> None:
    data = json.loads(SOURCE.read_text())
    assert len(data['rows']) == 120
    assert data['over_0p1_delta444']['deployed_with_repair'] == 35
    stats = data['summary']
    arms = {
        'Repair retained': {
            'color': '#16697A', 'offset': -.045,
            'points': [(0., [0., 0.]),
                       (stats['replicate_context_gain_repair_on_db']['mean_db'],
                        stats['replicate_context_gain_repair_on_db']['image_cluster_ci95_db'])],
        },
        'Repair removed': {
            'color': '#C06C38', 'offset': .045,
            'points': [(stats['isolated_repair_removal_gain_db']['mean_db'],
                        stats['isolated_repair_removal_gain_db']['image_cluster_ci95_db']),
                       (stats['replicate_total_gain_vs_deployed_db']['mean_db'],
                        stats['replicate_total_gain_vs_deployed_db']['image_cluster_ci95_db'])],
        },
    }
    plt.rcParams.update({
        'font.family': 'DejaVu Sans', 'font.size': 7.5,
        'axes.labelsize': 7.5, 'xtick.labelsize': 7.2, 'ytick.labelsize': 7,
        'pdf.fonttype': 42, 'svg.fonttype': 'none',
    })
    fig, ax = plt.subplots(figsize=(3.52, 2.65))
    fig.patch.set_facecolor('white')
    ax.axhline(0, lw=.85, color='#788690', linestyle=(0, (3, 2)), zorder=1)
    for label, arm in arms.items():
        xs = [arm['offset'], 1+arm['offset']]
        ys = [point[0] for point in arm['points']]
        low = [point[0]-point[1][0] for point in arm['points']]
        high = [point[1][1]-point[0] for point in arm['points']]
        ax.plot(xs, ys, lw=1.6, color=arm['color'], zorder=2)
        ax.errorbar(xs, ys, yerr=[low, high], fmt='o', ms=5.1,
                    mfc='white' if label == 'Repair retained' else arm['color'],
                    mec=arm['color'], mew=1.25, ecolor=arm['color'],
                    elinewidth=1.15, capsize=2.4, zorder=3, label=label)
    ax.set_xticks([0, 1], ['Isolated tiles', 'Active neighbours'])
    ax.set_xlim(-.33, 1.33)
    ax.set_ylim(-.026, .038)
    ax.set_yticks([-.02, -.01, 0, .01, .02, .03])
    ax.set_ylabel('Mean gain over deployed, $\Delta$444 PSNR (dB)')
    ax.grid(axis='y', color='#E9EEF1', linewidth=.55, zorder=0)
    ax.spines[['top', 'right']].set_visible(False)
    ax.spines[['left', 'bottom']].set_color('#788690')
    ax.tick_params(length=2.4, color='#788690')
    ax.legend(frameon=False, loc='upper left', fontsize=7.1, handlelength=1.6)
    interaction = stats['replicate_context_repair_interaction_db']
    ax.text(.51, -.0208,
            f"Interaction  +{interaction['mean_db']:.3f} dB",
            color='#173E56', fontsize=7.4, fontweight='bold', ha='center',
            bbox={'boxstyle': 'round,pad=.32', 'facecolor': '#F1F5F7',
                  'edgecolor': 'none'})
    fig.subplots_adjust(left=.185, right=.985, bottom=.16, top=.95)
    OUT.mkdir(parents=True, exist_ok=True)
    for ext in ('pdf', 'svg', 'png'):
        fig.savefig(OUT / f'context_repair_interaction.{ext}', dpi=300,
                    facecolor='white')
    plt.close(fig)
    svg = OUT / 'context_repair_interaction.svg'
    svg.write_text('\n'.join(line.rstrip() for line in svg.read_text().splitlines())+'\n')
    evidence = {
        'scope': 'Exploratory same-stream 2x2 interaction on DIV2K validation24 x 5QP; 24-image clustered intervals, CPU FP32, untimed.',
        'source_sha256': sha(SOURCE), 'script_sha256': sha(Path(__file__)),
        'canvas_mm': [89.41, 67.31], 'cases': 120,
        'interaction_mean_db': interaction['mean_db'],
        'interaction_ci95_db': interaction['image_cluster_ci95_db'],
        'artifacts': {f'context_repair_interaction.{ext}': sha(OUT / f'context_repair_interaction.{ext}')
                      for ext in ('pdf', 'svg', 'png')},
    }
    (OUT / 'context_repair_interaction_evidence.json').write_text(json.dumps(evidence, indent=2)+'\n')


if __name__ == '__main__':
    main()
