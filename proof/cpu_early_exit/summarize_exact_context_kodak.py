"""Summarize Kodak exact-context transfer and render mean real-bitstream RD."""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np


def sha(path):return hashlib.sha256(path.read_bytes()).hexdigest()


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--input',type=Path,required=True)
    p.add_argument('--scan-dir',type=Path,required=True)
    p.add_argument('--out-dir',type=Path,required=True)
    args=p.parse_args()
    data=json.loads(args.input.read_text());rows=data['rows']
    if len(rows)!=120 or sorted((r['image'],r['qp']) for r in rows)!=sorted(
        (image,qp) for image in data['images'] for qp in data['qps']):
        raise RuntimeError('Incomplete or duplicate Kodak result')
    for row in rows:
        scan_path=args.scan_dir/f'{Path(row["image"]).stem}_qp{row["qp"]}.json'
        if sha(scan_path)!=row['scan_sha256']:
            raise RuntimeError(f'Scan hash mismatch: {scan_path}')
        scan=json.loads(scan_path.read_text())
        h,w=scan['shape']
        row['bpp']=8*scan['stream_bytes']/(h*w)
    def avg(group,key):return float(np.mean([r[key] for r in group]))
    def stats(group):
        return {'cases':len(group),
                'mean_released_yuv611_db':avg(group,'released_yuv611_db'),
                'mean_deployed_yuv611_db':avg(group,'deployed_yuv611_db'),
                'mean_exact_yuv611_db':avg(group,'exact_context_yuv611_db'),
                'mean_deployed_gap_to_released_yuv611_db':avg(group,'deployed_gap_to_released_yuv611_db'),
                'mean_exact_gap_to_released_yuv611_db':avg(group,'exact_gap_to_released_yuv611_db'),
                'mean_yuv611_context_gain_db':float(np.mean([r['exact_context_yuv611_db']-r['deployed_yuv611_db'] for r in group])),
                'mean_deployed_delta444_db':avg(group,'deployed_delta444_db'),
                'mean_exact_delta444_db':avg(group,'exact_context_delta444_db'),
                'deployed_over_0p1_delta444_count':sum(r['deployed_delta444_db']>.1 for r in group),
                'exact_over_0p1_delta444_count':sum(r['exact_context_delta444_db']>.1 for r in group),
                'mean_ideal_shared_saving_pct':avg(group,'ideal_shared_saving_pct'),
                'mean_bpp':avg(group,'bpp')}
    out={'schema':1,'scope':'Kodak24 real FUFREF2, locked DIV2K beta; means over images, no BD-rate or latency',
         'input_sha256':sha(args.input),'all':stats(rows),
         'per_qp':{str(q):stats([r for r in rows if r['qp']==q]) for q in data['qps']}}
    images=data['images']
    effects=np.array([[np.mean([r['exact_context_yuv611_db']-r['deployed_yuv611_db']
                                for r in rows if r['image']==image]) for image in images],
                      [np.mean([r['deployed_delta444_db']-r['exact_context_delta444_db']
                                for r in rows if r['image']==image]) for image in images]])
    rng=np.random.default_rng(20261004)
    samples=rng.integers(0,len(images),size=(10000,len(images)))
    out['image_cluster_bootstrap_95ci']={
        'mean_yuv611_context_gain_db':[float(v) for v in np.quantile(effects[0,samples].mean(1),[.025,.975])],
        'mean_delta444_context_recovery_db':[float(v) for v in np.quantile(effects[1,samples].mean(1),[.025,.975])]}
    args.out_dir.mkdir(parents=True,exist_ok=True)
    (args.out_dir/'exact_context_kodak_summary.json').write_text(json.dumps(out,indent=2)+'\n')

    plt.rcParams.update({'font.family':'DejaVu Sans','font.size':8,'pdf.fonttype':42,
                         'axes.spines.top':False,'axes.spines.right':False})
    fig,(ax,bx)=plt.subplots(1,2,figsize=(7.2,2.9),gridspec_kw={'width_ratios':[1.18,1]})
    qps=data['qps']
    x=[out['per_qp'][str(q)]['mean_bpp'] for q in qps]
    for key,label,color,marker in [
        ('deployed_gap_to_released_yuv611_db','Deployed early exit','#B26749','s'),
        ('exact_gap_to_released_yuv611_db','Exact context, fixed route','#257D81','D')]:
        yy=[out['per_qp'][str(q)]['mean_'+key] for q in qps]
        lower=[];upper=[]
        for q,mean in zip(qps,yy):
            vals=np.asarray([r[key] for r in rows if r['qp']==q])
            ci=np.quantile(vals[samples].mean(1),[.025,.975])
            lower.append(mean-ci[0]);upper.append(ci[1]-mean)
        ax.errorbar(x,yy,yerr=np.asarray([lower,upper]),lw=1.45,
                    marker=marker,ms=4,color=color,label=label,
                    capsize=2.1,elinewidth=.75)
    ax.axhline(0,color='#505B69',lw=.85,ls=(0,(3,2)))
    ax.set_xscale('log')
    ax.set(xlabel='Actual FUFREF2 bitrate (mean bpp)',ylabel='YUV611 deficit to released D12 (dB)')
    ax.set_title('a  Rate-dependent quality deficit',loc='left',fontweight='bold',pad=5)
    ax.legend(frameon=False,fontsize=6.7,loc='upper left')
    qcolors={0:'#6D477E',16:'#316C91',32:'#278D8D',48:'#B88438',63:'#B95654'}
    gains=np.array([r['exact_context_yuv611_db']-r['deployed_yuv611_db'] for r in rows])
    for i,q in enumerate(qps):
        group=[r for r in rows if r['qp']==q]
        vals=np.array([r['exact_context_yuv611_db']-r['deployed_yuv611_db'] for r in group])
        jitter=np.random.default_rng(20261004+i).uniform(-.15,.15,len(vals))
        bx.scatter(np.full(len(vals),i)+jitter,vals,s=10,alpha=.6,
                   color=qcolors[q],edgecolor='none',zorder=2)
        bx.plot([i-.25,i+.25],[np.median(vals)]*2,color='#1D2933',lw=1.25,zorder=3)
    bx.axhline(0,color='#82919B',lw=.85,ls=(0,(3,2)))
    bx.set(xticks=range(len(qps)),xticklabels=[str(q) for q in qps],
           xlabel='Quantization parameter',ylabel='Exact-context YUV611 gain (dB)')
    bx.set_title('b  Image-level context effect',loc='left',fontweight='bold',pad=5)
    fig.tight_layout(w_pad=2)
    for ext in ('pdf','png'):
        fig.savefig(args.out_dir/f'exact_context_kodak.{ext}',dpi=220,bbox_inches='tight')
    plt.close(fig)
    print(json.dumps({'all':out['all'],'image_cluster_bootstrap_95ci':out['image_cluster_bootstrap_95ci']},indent=2))


if __name__=='__main__':main()
