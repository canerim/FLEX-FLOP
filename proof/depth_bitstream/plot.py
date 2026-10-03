"""Publication-ready visual audit of actual-byte Kodak RD and BD-rate."""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np

HERE = Path(__file__).resolve().parent
COLORS = {'D2':'#ce6655', 'D4':'#dd9d42', 'D6':'#398f91', 'D12':'#284962'}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--analysis', type=Path, default=HERE/'results/kodak_final_verified/analysis.json')
    parser.add_argument('--out', type=Path, default=HERE/'results/kodak_final_verified/actual_byte_rd')
    args = parser.parse_args()
    data = json.loads(args.analysis.read_text())
    metric = data['metrics']['psnr_yuv611']['bd_rate_vs_released_d12']
    plt.rcParams.update({'font.family':'DejaVu Sans', 'font.size':8, 'axes.linewidth':.7,
                         'axes.spines.top':False, 'axes.spines.right':False,
                         'pdf.fonttype':42, 'svg.fonttype':'none'})
    fig, (ax, bx) = plt.subplots(1, 2, figsize=(7.1, 2.8), gridspec_kw={'width_ratios':[1.22,1]})
    fig.patch.set_facecolor('white')
    for name in ('D2','D4','D6','D12'):
        points = data['qp_means'][name]
        qps = (0,16,32,48,63)
        rates = [points[str(q)]['payload_bpp'] for q in qps]
        quality = [points[str(q)]['psnr_yuv611'] for q in qps]
        ax.plot(rates, quality, color=COLORS[name], lw=1.8, marker='o', ms=3.6,
                markeredgecolor='white', markeredgewidth=.45,
                label='Released D12' if name=='D12' else name)
    ax.set(xscale='log', xlabel='Actual rANS payload (bpp)', ylabel='YUV 6:1:1 PSNR (dB)')
    ax.grid(axis='both', color='#dce4e8', lw=.5, alpha=.75)
    ax.legend(frameon=False, loc='lower right', fontsize=7.5)
    ax.set_title('a   Full Kodak RD', loc='left', fontweight='bold', fontsize=9, pad=10)
    names = ('D2','D4','D6')
    ys = np.arange(len(names))[::-1]
    for y, name in zip(ys, names):
        row = metric[name]
        val = row['mean_pct']
        lo, hi = row['image_bootstrap_ci95_pct']
        bx.barh(y, val, height=.5, color=COLORS[name], alpha=.94)
        bx.errorbar(val, y, xerr=[[val-lo],[hi-val]], fmt='none', ecolor='#263843',
                    capsize=2.5, elinewidth=.9)
        bx.text(hi+.14,y,f'{val:+.2f}%',va='center',ha='left',fontsize=8,fontweight='bold',color='#223843')
    bx.set_yticks(ys, names)
    bx.set(xlabel='Actual-byte BD-rate vs released D12 (%)', xlim=(0, max(metric[n]['image_bootstrap_ci95_pct'][1] for n in names)*1.3))
    bx.set_title('b   Rate cost at matched quality', loc='left', fontweight='bold', fontsize=9, pad=10)
    bx.grid(axis='x', color='#dce4e8', lw=.5)
    bx.set_axisbelow(True)
    fig.text(.015,.008,'BD-rate: 24 per-image PCHIP integrations on common four-model PSNR support; bars show paired-image 95% bootstrap intervals. Mean RD curves are visual context only.',
             color='#677986',fontsize=6.3)
    fig.subplots_adjust(left=.10,right=.97,bottom=.24,top=.82,wspace=.35)
    args.out.parent.mkdir(parents=True,exist_ok=True)
    for ext in ('pdf','png'):
        fig.savefig(args.out.with_suffix('.'+ext), dpi=300, bbox_inches='tight', facecolor='white')
    plt.close(fig)
    print(args.out)


if __name__ == '__main__':
    main()
