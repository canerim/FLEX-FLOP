"""Figure and cluster-bootstrap audit of where active context reduces error."""
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
BASE = HERE / 'results/div2k_beta/quality_floor'


def digest(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--raw', type=Path,
                        default=BASE/'context_seam_locality_validation24.json')
    parser.add_argument('--analysis', type=Path,
                        default=BASE/'context_seam_locality_analysis.json')
    parser.add_argument('--output', type=Path,
                        default=BASE/'fig_context_seam_locality.pdf')
    parser.add_argument('--evidence', type=Path,
                        default=BASE/'fig_context_seam_locality_evidence.json')
    args = parser.parse_args()
    raw = json.loads(args.raw.read_text())
    analysis = json.loads(args.analysis.read_text())
    assert raw['complete'] and len(raw['rows']) == 120
    assert analysis['raw_sha256'] == digest(args.raw)
    images = sorted({r['image'] for r in raw['rows']})
    assert len(images) == 24
    reduction = np.asarray([
        [sum((r['bins'][i]['isolated_mse444']-
               r['bins'][i]['active_mse444'])*r['bins'][i]['pixels']
              for r in raw['rows'] if r['image'] == im)
         for i in range(5)] for im in images], dtype=float)
    assert np.all(reduction.sum(axis=1) > 0)
    total = reduction.sum(axis=0)
    shares = total/total.sum()
    assert np.allclose(shares, [b['share_of_net_mse_reduction']
                                for b in analysis['bins']], atol=1e-12)
    rng = np.random.default_rng(20261009)
    draws = reduction[rng.integers(0,24,size=(10000,24))].sum(axis=1)
    assert np.all(draws.sum(axis=1) > 0)
    cumulative = np.cumsum(draws, axis=1)/draws.sum(axis=1, keepdims=True)
    ci = np.quantile(cumulative,[.025,.975],axis=0)
    pixel_cumulative = np.cumsum([b['pixel_fraction'] for b in analysis['bins']])

    plt.rcParams.update({'font.family':'DejaVu Sans','font.size':8,
                         'axes.linewidth':.75,'pdf.fonttype':42})
    fig, (ax,bx) = plt.subplots(1,2,figsize=(7.15,2.82),
                                gridspec_kw={'width_ratios':[1.03,1]},
                                constrained_layout=True)
    x = np.r_[0,pixel_cumulative]
    y = np.r_[0,np.cumsum(shares)]
    ax.plot([0,1],[0,1],color='#AAB2B9',linestyle=(0,(3,2)),linewidth=.8,
            label='Uniform contribution')
    ax.fill_between(x,np.r_[0,ci[0]],np.r_[0,ci[1]],
                    color='#DAEAE9',alpha=.7,linewidth=0)
    ax.plot(x,y,color='#227C82',marker='o',markersize=4.3,
            markeredgecolor='white',markeredgewidth=.6,linewidth=1.6,
            label='Active-context gain')
    ax.annotate('6.7% of pixels\n62.4% of gain',xy=(x[1],y[1]),
                xytext=(.27,.54),textcoords='axes fraction',fontsize=7,
                arrowprops={'arrowstyle':'-','color':'#227C82','lw':.8},
                color='#17636A',ha='left',va='center')
    ax.set(xlim=(0,1),ylim=(0,1.03),xlabel='Cumulative pixel fraction',
           ylabel='Cumulative share of net error reduction')
    ax.set_title('a  Gain concentrates at tile boundaries',loc='left',
                 fontsize=8.1,fontweight='bold',pad=5)
    ax.legend(frameon=False,fontsize=6.4,loc='lower right')

    labels=['0–8','8–16','16–32','32–64','64+']
    means=np.asarray([b['gain_db']['mean'] for b in analysis['bins']])
    intervals=np.asarray([b['gain_db']['image_cluster_ci95'] for b in analysis['bins']])
    errors=np.vstack((means-intervals[:,0],intervals[:,1]-means))
    bx.errorbar(np.arange(5),means,yerr=errors,fmt='s-',color='#A45336',
                markeredgecolor='white',markeredgewidth=.6,markersize=4.5,
                linewidth=1.4,elinewidth=.8,capsize=2.2)
    bx.set_xticks(np.arange(5),labels)
    bx.set(xlabel='Distance from nearest internal seam (px)',
           ylabel='Mean local 4:4:4 PSNR gain (dB)',ylim=(0,.35))
    bx.set_title('b  Local improvement decays with distance',loc='left',
                 fontsize=8.1,fontweight='bold',pad=5)
    for axis in (ax,bx):
        axis.spines[['top','right']].set_visible(False)
        axis.grid(axis='y',color='#E7E9EB',linewidth=.6)
        axis.tick_params(direction='out',length=3,width=.7)
    args.output.parent.mkdir(parents=True,exist_ok=True)
    fig.savefig(args.output,bbox_inches='tight')
    plt.close(fig)
    evidence={
        'scope':'Frozen original-beta isolated/no-repair vs active-replicate/no-repair; DIV2K validation24 x QP5, FUFREF2 research streams; no latency or native BD-rate.',
        'raw_sha256':digest(args.raw),'analysis_sha256':digest(args.analysis),
        'plot_script_sha256':digest(Path(__file__)),
        'figure_sha256':digest(args.output),
        'bootstrap':'10000 image-cluster draws retaining all five QPs per image',
        'all_images_positive':bool(np.all(reduction.sum(axis=1)>0)),
        'first_8px_pixel_fraction':float(analysis['bins'][0]['pixel_fraction']),
        'first_8px_net_gain_fraction':float(shares[0]),
        'first_8px_net_gain_fraction_ci95':ci[:,0].tolist(),
        'cumulative_pixel_fraction':pixel_cumulative.tolist(),
        'cumulative_gain_fraction':np.cumsum(shares).tolist(),
        'cumulative_gain_fraction_ci95':ci.tolist(),
    }
    args.evidence.write_text(json.dumps(evidence,indent=2)+'\n')
    print(json.dumps({k:evidence[k] for k in
                      ('first_8px_pixel_fraction','first_8px_net_gain_fraction',
                       'first_8px_net_gain_fraction_ci95')},indent=2))


if __name__ == '__main__':
    main()
