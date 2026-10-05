"""Vector rate-independent quality/compute frontier for frozen beta budgets."""
from __future__ import annotations

import hashlib
import json
from pathlib import Path

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

ROOT = Path(__file__).resolve().parents[2]
BASE = ROOT / 'proof/cpu_early_exit/results/div2k_beta/quality_floor'
BUDGET = BASE / 'active_replicate_budget_transfer_analysis.json'
PRIMARY = BASE / 'active_replicate_beta_validation24_analysis.json'
OUT = ROOT / 'docs/figures/active-beta-budget-transfer-20261005'


def sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main() -> None:
    budget = json.loads(BUDGET.read_text())
    primary = json.loads(PRIMARY.read_text())
    assert '0.1' in budget['feasible_targets']
    old_loss = primary['all']['old_active_no_repair_delta444_db']
    old_mac = primary['all']['old_active_no_repair_conv_mac_saving_pct']
    colours = {'0.075':'#2B5D76', '0.1':'#16697A',
               '0.125':'#B77742', '0.15':'#A34E37'}
    plt.rcParams.update({
        'font.family':'DejaVu Sans', 'font.size':7.6,
        'axes.labelsize':7.8, 'xtick.labelsize':7.2, 'ytick.labelsize':7.2,
        'pdf.fonttype':42, 'svg.fonttype':'none',
    })
    fig, ax = plt.subplots(figsize=(3.55, 2.87))
    fig.patch.set_facecolor('white')
    ax.axvspan(0, .1, facecolor='#ECF3F3', alpha=.75, zorder=0)
    ax.axvline(.1, color='#708087', lw=.85, linestyle=(0,(3,2)), zorder=1)
    ax.text(.1, .965, '0.10 dB', transform=ax.get_xaxis_transform(),
            ha='right', va='top', color='#61747E', fontsize=6.9)
    points = []
    for target in budget['feasible_targets']:
        row = budget['policies'][target]
        q = row['mean_delta444_db']
        mac = row['mean_conv_mac_saving_pct']
        x, y = q['mean'], mac['mean']
        color = colours[target]
        ax.errorbar(x, y,
                    xerr=[[x-q['image_cluster_ci95'][0]],
                          [q['image_cluster_ci95'][1]-x]],
                    yerr=[[y-mac['image_cluster_ci95'][0]],
                          [mac['image_cluster_ci95'][1]-y]],
                    fmt='o', ms=5.8, color=color, mec=color,
                    mfc='white' if target!='0.1' else color,
                    mew=1.35, capsize=2.1, elinewidth=1.05,
                    label=f'{float(target):.3f} dB calibration budget', zorder=4)
        points.append((float(target),x,y))
    points.sort()
    ax.plot([r[1] for r in points], [r[2] for r in points],
            color='#93A8AD', lw=1.0, zorder=2)
    x,y = old_loss['mean'],old_mac['mean']
    ax.errorbar(x,y,
                xerr=[[x-old_loss['image_cluster_ci95'][0]],
                      [old_loss['image_cluster_ci95'][1]-x]],
                yerr=[[y-old_mac['image_cluster_ci95'][0]],
                      [old_mac['image_cluster_ci95'][1]-y]],
                fmt='s', ms=5.0, color='#333E45', mec='#333E45', mfc='white',
                mew=1.2, capsize=2.1, elinewidth=1.0,
                label='Original beta on same decoder', zorder=5)
    ax.set_xlabel('Loss vs full-frame e15, $\Delta$444 PSNR (dB)')
    ax.set_ylabel('Synthesis conv. MAC saving (%)')
    ax.set_xlim(left=0)
    ax.grid(axis='y', color='#E8EEF1', linewidth=.55, zorder=0)
    ax.spines[['top','right']].set_visible(False)
    ax.spines[['left','bottom']].set_color('#788690')
    ax.tick_params(length=2.4, color='#788690')
    ax.legend(frameon=False, loc='lower right', fontsize=6.65,
              labelspacing=.27, handlelength=1.5, handletextpad=.4)
    fig.subplots_adjust(left=.16, right=.985, bottom=.19, top=.97)
    OUT.mkdir(parents=True, exist_ok=True)
    for ext in ('pdf','svg','png'):
        fig.savefig(OUT/f'active_beta_budget_transfer.{ext}', dpi=300,
                    facecolor='white')
    plt.close(fig)
    svg = OUT/'active_beta_budget_transfer.svg'
    svg.write_text('\n'.join(line.rstrip() for line in svg.read_text().splitlines())+'\n')
    evidence = {
        'scope':'Exploratory DIV2K validation24 x five-QP frozen e15/FUFREF2 beta budget transfer; analytical synthesis conv. MAC, no YUV BD-rate or latency claim.',
        'source_sha256':{'budget_analysis':sha(BUDGET),'primary_analysis':sha(PRIMARY)},
        'script_sha256':sha(Path(__file__)), 'canvas_mm':[90.17,72.90],
        'feasible_targets':budget['feasible_targets'],
        'points':{'original_beta':{'loss':old_loss,'mac':old_mac},
                  **{target:{'loss':budget['policies'][target]['mean_delta444_db'],
                             'mac':budget['policies'][target]['mean_conv_mac_saving_pct']}
                     for target in budget['feasible_targets']}},
        'artifacts':{f'active_beta_budget_transfer.{ext}':sha(OUT/f'active_beta_budget_transfer.{ext}')
                     for ext in ('pdf','svg','png')},
    }
    (OUT/'figure_evidence.json').write_text(json.dumps(evidence,indent=2)+'\n')


if __name__ == '__main__':
    main()
