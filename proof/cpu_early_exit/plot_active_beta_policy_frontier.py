"""Show mean-quality, exact-compute and tail-risk transfer of frozen prices."""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import matplotlib

matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np


HERE = Path(__file__).resolve().parent
BASE = HERE/'results/div2k_beta/quality_floor'


def sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--budget', type=Path,
                        default=BASE/'active_replicate_budget_transfer_analysis.json')
    parser.add_argument('--primary', type=Path,
                        default=BASE/'active_replicate_beta_validation24_analysis.json')
    parser.add_argument('--tail', type=Path,
                        default=BASE/'active_replicate_tail_validation24_analysis.json')
    parser.add_argument('--output', type=Path,
                        default=BASE/'fig_active_beta_policy_frontier.pdf')
    parser.add_argument('--evidence', type=Path,
                        default=BASE/'fig_active_beta_policy_frontier_evidence.json')
    args = parser.parse_args()
    budget = json.loads(args.budget.read_text())
    primary = json.loads(args.primary.read_text())
    tail = json.loads(args.tail.read_text())
    assert len(primary['rows']) == len(tail['rows']) == 120
    assert budget['feasible_targets'] == ['0.075','0.1','0.125','0.15']
    points = []
    for target in budget['feasible_targets']:
        p = budget['policies'][target]
        points.append({'label':f'B={float(target):.3f}',
                       'kind':'budget','budget':float(target),
                       'loss':p['mean_delta444_db']['mean'],
                       'loss_ci':p['mean_delta444_db']['image_cluster_ci95'],
                       'saving':p['mean_exact_conv_mac_saving_pct']['mean'],
                       'saving_ci':p['mean_exact_conv_mac_saving_pct']['image_cluster_ci95'],
                       'violations':p['over_0p1_count']})
    a = primary['all']
    points.append({'label':'Original β','kind':'old','budget':None,
                   'loss':a['old_active_no_repair_delta444_db']['mean'],
                   'loss_ci':a['old_active_no_repair_delta444_db']['image_cluster_ci95'],
                   'saving':a['old_exact_conv_mac_saving_pct']['mean'],
                   'saving_ci':a['old_exact_conv_mac_saving_pct']['image_cluster_ci95'],
                   'violations':primary['thresholds']['old_over_0p1']})
    t = tail['all']
    points.append({'label':'Tail rule','kind':'tail','budget':None,
                   'loss':t['tail_delta444_db']['mean'],
                   'loss_ci':t['tail_delta444_db']['image_cluster_ci95'],
                   'saving':t['tail_exact_conv_mac_saving_pct']['mean'],
                   'saving_ci':t['tail_exact_conv_mac_saving_pct']['image_cluster_ci95'],
                   'violations':tail['thresholds']['tail_over_0p1']})
    colors = {'budget':'#1E7180','old':'#74848C','tail':'#5C8568'}
    plt.rcParams.update({'font.family':'DejaVu Sans','font.size':7.7,
                         'axes.linewidth':.75,'pdf.fonttype':42})
    fig, (ax,bx) = plt.subplots(1,2,figsize=(7.15,3.03),
                                constrained_layout=True)
    bpoints = [p for p in points if p['kind']=='budget']
    ax.axvspan(0,.1,color='#EEF5F2',zorder=0)
    ax.axvline(.1,color='#7F9B8D',ls=(0,(3,2)),lw=.8,zorder=1)
    ax.plot([p['loss'] for p in bpoints],[p['saving'] for p in bpoints],
            color='#A8BEC0',lw=1.1,zorder=2)
    for p in points:
        color = colors[p['kind']]
        x,y = p['loss'],p['saving']
        ax.errorbar(x,y,xerr=[[x-p['loss_ci'][0]],[p['loss_ci'][1]-x]],
                    yerr=[[y-p['saving_ci'][0]],[p['saving_ci'][1]-y]],
                    fmt={'budget':'o','old':'s','tail':'D'}[p['kind']],
                    ms=5.2,mew=.7,mec='white',mfc=color,color=color,
                    capsize=1.6,elinewidth=.75,alpha=.95,zorder=4)
    ax.set(xlim=(0,.175),ylim=(8,34),
           xlabel=r'Mean loss vs. e15, $\Delta_{444}$ (dB)',
           ylabel='Exact synthesis-conv MAC saved (%)')
    ax.set_title('a  Mean quality versus compute',loc='left',
                 fontsize=8.3,fontweight='bold',pad=5)
    label_positions = {
        'B=0.075':(.045,17.3),'B=0.100':(.095,22.0),
        'B=0.125':(.125,24.1),'B=0.150':(.145,31.0),
        'Original β':(.028,22.8),'Tail rule':(.043,12.1),
    }
    for p in points:
        lx,ly = label_positions[p['label']]
        ax.annotate(p['label'],(p['loss'],p['saving']),xytext=(lx,ly),
                    textcoords='data',fontsize=6.7,
                    color=colors[p['kind']],
                    arrowprops={'arrowstyle':'-','lw':.55,
                                'color':colors[p['kind']]} if
                    p['label'] in ('Original β','B=0.075','B=0.125') else None)
    ax.text(.098,9.0,'0.10 dB target',ha='right',va='bottom',
            fontsize=6.4,color='#618474')

    bx.axhspan(0,10,color='#EEF5F2',zorder=0)
    bx.axhline(10,color='#7F9B8D',ls=(0,(3,2)),lw=.8,zorder=1)
    bx.plot([p['saving'] for p in bpoints],
            [p['violations'] for p in bpoints],color='#A8BEC0',lw=1.1,zorder=2)
    for p in points:
        color = colors[p['kind']]
        bx.scatter(p['saving'],p['violations'],s=38,marker={
            'budget':'o','old':'s','tail':'D'}[p['kind']],
            edgecolor='white',linewidth=.6,color=color,zorder=4)
        label_offset = ((-7,7) if p['label']=='B=0.075' else
                        (8,9) if p['kind']=='old' else (0,6))
        bx.annotate(str(p['violations']),
                    (p['saving'],p['violations']),xytext=label_offset,
                    textcoords='offset points',ha='center',va='bottom',
                    fontsize=6.8,color=color)
    bx.set(xlim=(10,33),ylim=(0,69),
           xlabel='Exact synthesis-conv MAC saved (%)',
           ylabel='Cases with loss >0.1 dB (of 120)')
    bx.set_title('b  The quality tail rises with saving',loc='left',
                 fontsize=8.3,fontweight='bold',pad=5)
    bx.text(32,3,'≤10-case tail target',ha='right',va='bottom',
            fontsize=6.4,color='#618474')
    for axis in (ax,bx):
        axis.spines[['top','right']].set_visible(False)
        axis.grid(axis='y',color='#E7E9EB',lw=.55)
        axis.tick_params(direction='out',length=3,width=.7)
    args.output.parent.mkdir(parents=True,exist_ok=True)
    fig.savefig(args.output,bbox_inches='tight')
    plt.close(fig)
    evidence={'scope':'Exploratory calibration-only active-replicate/no-repair policies on 24 disjoint DIV2K validation images x five QPs; exact synthesis-conv MAC, not full-codec latency.',
              'source_sha256':{'budget':sha(args.budget),
                               'primary':sha(args.primary),'tail':sha(args.tail)},
              'script_sha256':sha(Path(__file__)),
              'figure_sha256':sha(args.output),
              'points':points}
    args.evidence.write_text(json.dumps(evidence,indent=2)+'\n')
    print(args.output)


if __name__=='__main__':
    main()
