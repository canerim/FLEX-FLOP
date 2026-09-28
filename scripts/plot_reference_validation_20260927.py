"""Render complete, measured CPU bitstream validation; never fill missing data."""
from __future__ import annotations
import argparse
import hashlib
import json
from pathlib import Path
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages
from matplotlib.lines import Line2D
import numpy as np
try:
    import paper_refresh_figures as F
except ModuleNotFoundError:
    import build_figures as F
from research_figure_paths_20260927 import paths

DEFAULT,DEFAULT_OUT=paths('div2k100_epoch020')
DEPTHS=(2,4,6,12)
COLORS={depth:F.DEPTH_ALL[depth//2-1] for depth in DEPTHS}
MARKERS={2:'o',4:'s',6:'^',12:'D'}


def matched(data,rate):
    return next(r for r in data['matched_rate'] if (r['rate_field'],r['interpolator'],r['target_bpp'])==('payload_bpp','linear',rate))


def pair(row,depth,cohort='four_model_common_support'):
    return next(p for p in row['pairs'] if (p['reference_depth'],p['depth'],p['cohort'])==(2,depth,cohort))


def interval(ax,x,summary,color,marker='o',offset=0):
    if not summary['n']:return
    value=summary['mean'];ci=summary['ci95']
    ax.plot(x+offset,value,marker=marker,color=color,ms=4,ls='none',zorder=4)
    if ci is not None:
        ax.vlines(x+offset,ci[0],ci[1],color=color,lw=1,zorder=3)
        ax.plot([x+offset]*2,ci,marker='_',color=color,ls='none',ms=4,zorder=3)


def footer(fig,text):
    fig.text(.5,.025,text,ha='center',fontsize=6,color=F.MUTED)


def render(data,out):
    if data['n_cases']!=2000 or data['n_images']!=100 or len(data['rows'])!=2000:
        raise ValueError('Complete 100-image, four-model validation required')
    F.OUT=out;F.AUDIT.clear();F.CAPTIONS.clear()
    with PdfPages(out/'reference_validation_atlas.pdf',metadata=F.PDF_META) as book:
        fig,axs=plt.subplots(1,2,figsize=(183*F.MM,77*F.MM))
        fig.subplots_adjust(left=.078,right=.98,bottom=.32,top=.79,wspace=.30)
        F.panel(axs[0],'a','Actual payload rate–distortion')
        F.panel(axs[1],'b','Paired gain at the same payload rate')
        for depth in DEPTHS:
            rr=sorted([r for r in data['native_qp_means'] if r['depth']==depth],key=lambda r:r['payload_bpp'])
            axs[0].plot([r['payload_bpp'] for r in rr],[r['psnr_rgb'] for r in rr],
                color=COLORS[depth],marker=MARKERS[depth],ls='--' if depth==12 else '-',
                label=f'D{depth}'+(' released' if depth==12 else ' · epoch 20'))
        axs[0].set(xlabel='Mean rANS payload (bits/pixel)',ylabel='Mean RGB PSNR (dB)')
        rates=(.1,.2,.4)
        for x,rate in enumerate(rates):
            row=matched(data,rate)
            for depth,offset in ((4,-.18),(6,0),(12,.18)):
                interval(axs[1],x,pair(row,depth)['delta_psnr_db'],COLORS[depth],MARKERS[depth],offset)
        axs[1].axhline(0,color=F.GREY,lw=.7,zorder=0)
        axs[1].set(xlabel='Target payload rate (bits/pixel)',ylabel='PSNR gain relative to D2 (dB)',
            xticks=range(3),xticklabels=[f'{r:.1f}\nn={matched(data,r)["common_n"]}' for r in rates],xlim=(-.45,2.45))
        fig.legend(*axs[0].get_legend_handles_labels(),loc='lower center',bbox_to_anchor=(.5,.08),ncol=4,fontsize=6.3)
        fig.text(.5,.96,'Actual coded rate · frozen epoch-20 models',ha='center',weight='normal',fontsize=6.8)
        footer(fig,'100 DIV2K centre crops · CPU FP32 · D12 has different training provenance · paired-image 95% intervals')
        F.audit_and_save(fig,'fig_reference_rd',
            'Left: means across all 100 crops at each of five native QPs, joined only as visual guides. Right: per-image log-rate linear interpolation and paired quality differences on the four-model common support at each rate; n is explicit and can change with rate. Whiskers are 95% image-bootstrap intervals conditional on the frozen checkpoints and included cohort. D2/D4/D6 are epoch20 of105; D12 is released and is not a matched-training causal depth control. Rates are actual rANS payload, excluding the custom research header.',book)

        fig,axs=plt.subplots(1,2,figsize=(183*F.MM,78*F.MM))
        fig.subplots_adjust(left=.08,right=.98,bottom=.28,top=.78,wspace=.33)
        F.panel(axs[0],'a','Does the four-crop monitor represent the cohort?')
        F.panel(axs[1],'b','How much bitrate support is available?')
        row=matched(data,.2)
        for x,depth in enumerate((4,6,12)):
            p=pair(row,depth)
            for field,offset,marker,color in [('monitor_four',-.13,'o',F.ORANGE),('additional_96',.13,'s',F.BLUE)]:
                summary=p[field];interval(axs[0],x,summary,color,marker,offset)
                if summary['n']:
                    axs[0].annotate(f'n={summary["n"]}',(x+offset,summary['mean']),xytext=(0,8 if field=='monitor_four' else -13),
                        textcoords='offset points',ha='center',fontsize=5.7,color=color)
        axs[0].axhline(0,color=F.GREY,lw=.7)
        axs[0].set(xticks=range(3),xticklabels=['D4 − D2','D6 − D2','D12 − D2'],
            ylabel='PSNR gain at 0.2 payload bpp (dB)',xlim=(-.5,2.5))
        axs[0].margins(y=.3)
        for depth,offset in ((4,-.23),(6,0),(12,.23)):
            counts=[pair(matched(data,r),depth,'pairwise_support')['delta_psnr_db']['n'] for r in rates]
            axs[1].bar(np.arange(3)+offset,counts,.21,color=COLORS[depth],label=f'D{depth} / D2 pair')
        axs[1].plot(range(3),[matched(data,r)['common_n'] for r in rates],color=F.INK,marker='_',ms=15,ls='none',label='All four models')
        axs[1].set(xticks=range(3),xticklabels=['0.1','0.2','0.4'],xlabel='Target payload rate (bits/pixel)',
            ylabel='Images with measured rate support',ylim=(0,106))
        axs[1].legend(loc='lower center',bbox_to_anchor=(.5,-.43),ncol=2,fontsize=5.8)
        axs[0].legend(handles=[Line2D([],[],color=F.ORANGE,marker='o',ls='none',label='Original four'),
            Line2D([],[],color=F.BLUE,marker='s',ls='none',label='Additional 96')],loc='lower center',bbox_to_anchor=(.5,-.43),ncol=2,fontsize=5.8)
        fig.text(.5,.96,'Validation coverage',ha='center',weight='normal',fontsize=6.8)
        footer(fig,'Means use common supported images · no interval estimated for the four-crop subset · cohorts vary with bitrate')
        F.audit_and_save(fig,'fig_reference_coverage',
            'At0.2 actual payload bpp, compare the original monitor images0801–0804 with the remaining validation images, restricting both groups to the four-model common support. The small monitor subset receives no confidence interval. The coverage panel reports pairwise and four-model shared support at each rate, preventing silent extrapolation or an implicit claim that every rate uses all100 images.',book)

        fig,axs=plt.subplots(1,2,figsize=(183*F.MM,73*F.MM))
        fig.subplots_adjust(left=.085,right=.98,bottom=.25,top=.78,wspace=.32)
        F.panel(axs[0],'a','Actual payload differs from entropy estimates')
        F.panel(axs[1],'b','Released-model bitrate gap')
        for depth in DEPTHS:
            rr=sorted([r for r in data['native_qp_means'] if r['depth']==depth],key=lambda r:r['qp'])
            means=np.array([r['payload_minus_estimate']['mean'] for r in rr])*1000
            bounds=np.array([r['payload_minus_estimate']['ci95'] for r in rr])*1000
            axs[0].plot([r['qp'] for r in rr],means,color=COLORS[depth],marker=MARKERS[depth],label=f'D{depth}')
            axs[0].fill_between([r['qp'] for r in rr],bounds[:,0],bounds[:,1],color=COLORS[depth],alpha=.1,lw=0)
        axs[0].axhline(0,color=F.GREY,lw=.7)
        axs[0].set(xlabel='QP',ylabel=r'Payload − estimated rate ($10^{-3}$ bpp)',xticks=[0,16,32,48,63])
        axs[0].legend(ncol=4,loc='lower center',bbox_to_anchor=(.5,-.4),fontsize=6)
        for x,depth in enumerate((2,4,6)):
            s=next(r['summary'] for r in data['bd_rate_vs_released_d12'] if (r['rate_field'],r['depth'])==('payload_bpp',depth))
            interval(axs[1],x,s,COLORS[depth],MARKERS[depth])
        axs[1].axhline(0,color=F.GREY,lw=.7)
        axs[1].set(xticks=range(3),xticklabels=[f'D{d}\nn='+str(next(r['summary']['n'] for r in data['bd_rate_vs_released_d12'] if (r['rate_field'],r['depth'])==('payload_bpp',d))) for d in (2,4,6)],
            ylabel='BD-rate relative to released D12 (%)',xlim=(-.5,2.5))
        fig.text(.5,.96,'Actual payload and the released-model gap',ha='center',weight='normal',fontsize=6.8)
        footer(fig,'Header excluded · BD-rate uses each image’s four-model common PSNR interval · interim vs released comparison')
        F.audit_and_save(fig,'fig_reference_rate_accounting',
            'Left: mean actual rANS payload minus estimated entropy with pointwise95% image-bootstrap intervals. Right: average per-image BD-rate relative to releasedD12; PCHIP log-rate integration is restricted to each image’s four-model common PSNR interval, and nonmonotone curves are rejected. n reports valid images. The custom88-byte header contributes an additional0.00268555bpp per512crop and is not included here. The released gap mixes depth, training progress and provenance.',book)


if __name__=='__main__':
    ap=argparse.ArgumentParser(description=__doc__);ap.add_argument('--data',type=Path,default=DEFAULT/'analysis.json')
    ap.add_argument('--out',type=Path,default=DEFAULT_OUT);args=ap.parse_args()
    data=json.loads(args.data.read_text());args.out.mkdir(parents=True,exist_ok=True);render(data,args.out)
    (args.out/'figure_evidence.json').write_text(json.dumps({'analysis_sha256':hashlib.sha256(args.data.read_bytes()).hexdigest(),
        'script_sha256':hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),'captions':F.CAPTIONS,'layout_audit':F.AUDIT},indent=2)+'\n')
