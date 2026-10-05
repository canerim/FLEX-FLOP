"""Vector figure for paired Kodak/DIV2K active-context transfer effects."""
from __future__ import annotations

import hashlib
import json
from pathlib import Path

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np

ROOT = Path(__file__).resolve().parents[2]
KODAK = ROOT / 'proof/cpu_early_exit/results/kodak_canvas_arms_comparison_20261005.json'
DIV2K = ROOT / 'proof/cpu_early_exit/results/div2k_beta/quality_floor/active_canvas_validation24_transfer_comparison.json'
OUT = ROOT / 'docs/figures/active-canvas-transfer-20261005'
QPS = (0, 16, 32, 48, 63)


def sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def per_qp(rows: list[dict], key: str, rng: np.random.Generator) -> dict:
    answer = {}
    for qp in QPS:
        cases = [r for r in rows if r['qp'] == qp]
        assert len(cases) == 24 and len({r['image'] for r in cases}) == 24
        values = np.asarray([r[key] for r in cases], dtype=float)
        means = values[rng.integers(0, 24, size=(10000, 24))].mean(axis=1)
        answer[str(qp)] = {'mean_db': float(values.mean()),
                           'image_bootstrap_ci95_db': np.quantile(means, [.025, .975]).tolist()}
    return answer


def main() -> None:
    kodak = json.loads(KODAK.read_text())
    div2k = json.loads(DIV2K.read_text())
    assert len(kodak['rows']) == len(div2k['rows']) == 120
    assert len({r['image'] for r in kodak['rows']}) == 24
    assert len({r['image'] for r in div2k['rows']}) == 24
    rng = np.random.default_rng(20261006)
    panels = {
        'replicate_minus_zero_with_repair_db': {
            'Kodak': per_qp(kodak['rows'], 'replicate_minus_zero_with_repair_db', rng),
            'DIV2K': per_qp(div2k['rows'], 'replicate_minus_zero_with_repair_db', rng),
        },
        'replicate_no_repair_minus_with_repair_db': {
            'Kodak': per_qp([
                dict(r, replicate_no_repair_minus_with_repair_db=(
                    r['active_replicate_no_repair_gain_db']-
                    r['active_replicate_with_repair_gain_db']))
                for r in kodak['rows']], 'replicate_no_repair_minus_with_repair_db', rng),
            'DIV2K': per_qp(div2k['rows'], 'replicate_no_repair_minus_with_repair_db', rng),
        },
    }
    plt.rcParams.update({
        'font.family': 'DejaVu Sans', 'font.size': 8,
        'axes.labelsize': 8, 'axes.titlesize': 8.5,
        'xtick.labelsize': 7.5, 'ytick.labelsize': 7.5,
        'pdf.fonttype': 42, 'svg.fonttype': 'none',
    })
    fig, axes = plt.subplots(1, 2, figsize=(7.18, 2.72))
    fig.patch.set_facecolor('white')
    colors = {'Kodak': '#16697A', 'DIV2K': '#C06C38'}
    markers = {'Kodak': 'o', 'DIV2K': 's'}
    titles = ['a   The active fallback transfers',
              'b   Removing old repair helps']
    ylabels = ['Replicate over zero, $\Delta$444 PSNR (dB)',
               'Gain from removing repair (dB)']
    for ax, (name, series), title, ylabel in zip(axes, panels.items(), titles, ylabels):
        ax.axhline(0, color='#687682', lw=.8, linestyle=(0, (3, 2)), zorder=1)
        for label in ('Kodak', 'DIV2K'):
            x = np.arange(5) + (-.08 if label == 'Kodak' else .08)
            stats = [series[label][str(qp)] for qp in QPS]
            y = np.asarray([r['mean_db'] for r in stats])
            lo = np.asarray([r['image_bootstrap_ci95_db'][0] for r in stats])
            hi = np.asarray([r['image_bootstrap_ci95_db'][1] for r in stats])
            ax.errorbar(x, y, yerr=[y-lo, hi-y], fmt=markers[label],
                        ms=4.7, lw=1.15, elinewidth=1.1, capsize=2.2,
                        color=colors[label], mec=colors[label],
                        mfc='white' if label == 'Kodak' else colors[label],
                        label=label, zorder=3)
        ax.set_xticks(range(5), [str(q) for q in QPS])
        ax.set_xlim(-.42, 4.42)
        ax.set_xlabel('Quality index (QP)')
        ax.set_ylabel(ylabel)
        ax.set_title(title, loc='left', pad=8, color='#173E56', fontweight='bold')
        ax.grid(axis='y', color='#E7EDF0', linewidth=.55, zorder=0)
        ax.spines[['top', 'right']].set_visible(False)
        ax.spines[['left', 'bottom']].set_color('#677681')
        ax.tick_params(length=2.5, color='#677681')
        ax.margins(y=.12)
    axes[0].legend(frameon=False, loc='upper left', fontsize=7.5,
                   ncol=1, labelspacing=.28, handletextpad=.35)
    fig.subplots_adjust(left=.10, right=.985, bottom=.22, top=.88, wspace=.32)
    OUT.mkdir(parents=True, exist_ok=True)
    for ext in ('pdf', 'svg', 'png'):
        fig.savefig(OUT / f'canvas_transfer.{ext}', dpi=300, facecolor='white')
    plt.close(fig)
    svg = OUT / 'canvas_transfer.svg'
    svg.write_text('\n'.join(line.rstrip() for line in svg.read_text().splitlines()) + '\n')
    evidence = {
        'scope': 'Same-stream paired early-exit decoder quality; panel a fixes repair, panel b is exploratory within-active-replicate repair removal. No latency or changed-rate claim.',
        'source_sha256': {'Kodak': sha(KODAK), 'DIV2K': sha(DIV2K)},
        'script_sha256': sha(Path(__file__)),
        'canvas_mm': [182.37, 69.09],
        'cohorts': {'Kodak': 120, 'DIV2K': 120},
        'panels': panels,
        'artifacts': {f'canvas_transfer.{ext}': sha(OUT / f'canvas_transfer.{ext}')
                      for ext in ('pdf', 'svg', 'png')},
    }
    (OUT / 'figure_evidence.json').write_text(json.dumps(evidence, indent=2) + '\n')
    print(json.dumps({'artifacts': evidence['artifacts'],
                      'source_sha256': evidence['source_sha256']}, indent=2))


if __name__ == '__main__':
    main()
