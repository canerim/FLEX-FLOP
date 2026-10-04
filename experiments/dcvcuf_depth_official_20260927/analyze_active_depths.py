"""CPU-only health audit of D8/D10/D12 official-recipe runs.

Reads archived four-image validation JSON and live status. No model loading,
GPU context, checkpoint rewrite, or interference with training processes.
"""
from __future__ import annotations

import argparse
from datetime import datetime, timezone
import json
from pathlib import Path

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

from plot_live_validation import load_history


COLORS = {8: '#257A83', 10: '#C88437', 12: '#A43D61'}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--runs', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=True)
    now = datetime.now(timezone.utc)
    histories = {d: load_history(args.runs, d) for d in (8,10,12)}
    summary = {'generated_at_utc':now.isoformat(),
               'protocol':'DIV2K first four central 512 crops; log-rate interpolation to 0.2 estimated bpp; diagnostic YUV611 PSNR, not actual bitstreams',
               'training_process_modified':False, 'models':{}}
    for depth, history in histories.items():
        status_path = args.runs/f'd{depth}'/'status.json'
        status = json.loads(status_path.read_text())
        latest = history[-1]
        best = max(history,key=lambda row:row['psnr_yuv611_db'])
        lookup = {row['epoch']:row['psnr_yuv611_db'] for row in history}
        age = (now-datetime.fromisoformat(status['timestamp'])).total_seconds()
        if status['state']!='running' or age>600:
            raise RuntimeError(f'D{depth} status is stale or stopped: {status}')
        summary['models'][str(depth)] = {
            'status_epoch':status['epoch'], 'status_next_batch':status['next_batch'],
            'batches_per_epoch':status['batches_per_epoch'],
            'latest_validation_epoch':latest['epoch'],
            'latest_psnr_yuv611_db':latest['psnr_yuv611_db'],
            'best_validation_epoch':best['epoch'],
            'best_psnr_yuv611_db':best['psnr_yuv611_db'],
            'matched_epoch_40_db':lookup.get(40),
            'matched_epoch_42_db':lookup.get(42),
            'epoch_50_db':lookup.get(50),
            'learning_rate':status['lr'],
            'nonfinite_skips':status['nonfinite_skips'],
            'status_age_seconds':age,
            'validation_records':len(history),
        }
    common_epochs=set.intersection(*({row['epoch'] for row in history}
                                     for history in histories.values()))
    summary['common_validation_epochs'] = [e for e in sorted(common_epochs) if e<=42]
    for depth in (8,10,12):
        if 40 not in {row['epoch'] for row in histories[depth]} or 42 not in {row['epoch'] for row in histories[depth]}:
            raise RuntimeError(f'D{depth} lacks common epoch 40/42 validation')

    plt.rcParams.update({'font.family':'DejaVu Sans','font.size':8,
                         'axes.linewidth':.7,'pdf.fonttype':42})
    fig, (ax, bx) = plt.subplots(1,2,figsize=(7.25,3.15),
                                 gridspec_kw={'width_ratios':[1.15,1]},
                                 constrained_layout=True)
    for axis in (ax,bx):
        axis.axvline(45,color='#81919A',lw=.9,ls=(0,(3,2)),zorder=1)
        axis.axvspan(45,70,color='#F0F3F4',zorder=0)
        axis.spines[['top','right']].set_visible(False)
        axis.grid(axis='y',color='#E5E9EB',lw=.6,zorder=0)
        axis.tick_params(direction='out',length=3,width=.7)
    for depth, history in histories.items():
        xx=[r['epoch'] for r in history]
        yy=[r['psnr_yuv611_db'] for r in history]
        ax.plot(xx,yy,color=COLORS[depth],lw=1.4,label=f'D{depth}',zorder=3)
        ax.scatter(xx[-1],yy[-1],s=19,color=COLORS[depth],zorder=4)
        zoom=[r for r in history if 35<=r['epoch']<=52]
        bx.plot([r['epoch'] for r in zoom],[r['psnr_yuv611_db'] for r in zoom],
                color=COLORS[depth],lw=1.45,marker='o',ms=2.5,
                label=f'D{depth}',zorder=3)
    ax.set(xlim=(1,77),xlabel='Completed training epoch',
           ylabel='YUV611 PSNR at 0.2 estimated bpp (dB)')
    ax.set_title('a  Full training trajectory',loc='left',fontweight='bold',pad=5)
    ax.legend(frameon=False,ncol=3,loc='lower right',fontsize=7)
    bx.set(xlim=(35,52),xlabel='Completed training epoch',
           ylabel='YUV611 PSNR (dB)')
    bx.set_title('b  Shared pre-LR variability',loc='left',fontweight='bold',pad=5)
    bx.text(45.3,bx.get_ylim()[0]+.015,'LR $2\\times10^{-4}\\rightarrow5\\times10^{-5}$',
            fontsize=6.2,color='#62717B',rotation=90,va='bottom')
    for suffix in ('pdf','png'):
        fig.savefig(args.output/f'active_depths_health.{suffix}',dpi=220,bbox_inches='tight')
    plt.close(fig)
    (args.output/'active_depths_health.json').write_text(json.dumps(summary,indent=2)+'\n')
    print(json.dumps(summary,indent=2))


if __name__=='__main__':
    main()
