"""Scientific figures for complete, predeclared fixed-depth patch controls."""
from __future__ import annotations
import hashlib
import json
from pathlib import Path
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages
from PIL import Image
try:
    import paper_refresh_figures as F
except ModuleNotFoundError:
    import build_figures as F
from research_figure_paths_20260927 import paths, BUNDLED

REPO=Path(__file__).resolve().parents[1]
DATA,OUT=paths('patch_control_epoch020')
ROOT=Path('/data10/shareddata/can_karsal/dcvcuf_depth_20260927/research')
COLORS={0:F.ORANGE,32:F.BLUE,64:F.TEAL}
LABELS={0:'Independent 256 cores',32:'32-pixel context halo',64:'64-pixel context halo'}
OFFSETS={0:-.15,32:0,64:.15}
MARKERS={0:'o',32:'s',64:'^'}


def ci(ax,x,summary,color,marker='o',scale=1):
    if not summary['n']:return
    y=summary['mean']*scale
    if summary['ci95'] is None:
        ax.plot(x,y,marker=marker,color=color,mec='white',mew=.35,ms=4)
        return
    lo,hi=np.array(summary['ci95'])*scale
    ax.errorbar(x,y,yerr=[[max(0,y-lo)],[max(0,hi-y)]],fmt=marker,
        color=color,mfc=color,mec='white',mew=.35,ms=4,capsize=2,lw=.8)


def matched(data,book):
    fig=plt.figure(figsize=(183*F.MM,84*F.MM))
    lower=min(p['delta_psnr_db']['ci95'][0] for r in data['matched_rate']
        if (r['rate_field'],r['interpolator'])==('payload_bpp','linear')
        for p in r['comparisons'] if p['delta_psnr_db']['ci95'] is not None)
    for column,depth in enumerate((2,6,12)):
        ax=fig.add_axes([.085+.32*column,.32,.24,.43])
        F.panel(ax,chr(97+column),f'D{depth}'+(' · released' if depth==12 else ' · epoch 20'))
        rows=[r for r in data['matched_rate'] if (r['depth'],r['rate_field'],r['interpolator'])==(depth,'payload_bpp','linear')]
        for pos,r in enumerate(rows):
            for p in r['comparisons']:
                halo=int(p['variant'][4:]);ci(ax,pos+OFFSETS[halo],p['delta_psnr_db'],COLORS[halo], MARKERS[halo])
        ax.axhline(0,color=F.GREY,lw=.8,ls='--')
        ax.set(xticks=range(len(rows)),xticklabels=[f"{r['target_bpp']:.1f}\nn={r['n']}" for r in rows],
            xlabel='Actual payload rate (bpp)',ylabel='RGB PSNR change from full frame (dB)' if column==0 else '')
        ax.set_xlim(-.4,2.4);ax.set_ylim(lower-.12,.15)
    handles=[plt.Line2D([],[],color=COLORS[h],marker=MARKERS[h],ls='none',label=LABELS[h]) for h in (0,32,64)]
    fig.legend(handles=handles,loc='lower center',bbox_to_anchor=(.51,.12),ncol=3,frameon=False,fontsize=5.8)
    fig.text(.5,.96,'The rate–distortion cost of independent patches',ha='center',weight='normal',fontsize=6.8)
    fig.text(.5,.86,'16 predeclared DIV2K crops · paired images within each model and rate',ha='center',fontsize=6.5,color=F.MUTED)
    fig.text(.5,.035,'Common full/halo0/halo32/halo64 support within each depth · no extrapolation · 95% image-bootstrap intervals',ha='center',fontsize=6,color=F.MUTED)
    F.audit_and_save(fig,'fig_patch_matched_rate',
        'Fixed-depth patching at matched actual payload rate. Each marker is the paired-image mean RGB PSNR change from the same model on the full512 crop; intervals use5000 image-bootstrap draws. Cohorts are the common support of all four variants within each depth and rate, with n shown. Rates use log-rate linear interpolation. These are interim epoch20 D2/D6 and released D12 controls, not adaptive routing, final trained-model comparison or GPU runtime. Container and PCHIP sensitivity remain in the analysis.',book)


def mechanism(data,book):
    fig=plt.figure(figsize=(183*F.MM,91*F.MM));axes=[]
    for col in range(3):axes.append(fig.add_axes([.075+.32*col,.29,.235,.43]))
    F.panel(axes[0],'a','Entropy resets and context bytes')
    F.panel(axes[1],'b','Error near the patch boundary')
    F.panel(axes[2],'c','Coded area and research headers')
    for depth,marker in ((2,'o'),(6,'s'),(12,'^')):
        for halo in (0,32,64):
            rows=[r for r in data['same_qp'] if (r['depth'],r['halo'])==(depth,halo)]
            xx=[r['qp'] for r in rows];yy=[r['summaries']['payload_increase_percent']['mean'] for r in rows]
            axes[0].plot(xx,yy,color=COLORS[halo],marker=marker,ms=3,lw=.8,ls='-' if depth==6 else ':',alpha=.95)
    axes[0].axhline(0,color=F.GREY,lw=.7)
    axes[0].set(xlabel='Quality index',ylabel='Same-QP payload increase (%)',xticks=[0,32,63])
    for position,depth in enumerate((2,6,12)):
        for halo in (0,32,64):
            row=next(r for r in data['same_qp'] if (r['depth'],r['halo'],r['qp'])==(depth,halo,32))
            ci(axes[1],position+OFFSETS[halo],row['summaries']['boundary_specific_r4_excess_mse'],COLORS[halo],scale=1e4)
    axes[1].axhline(0,color=F.GREY,lw=.7,ls='--')
    axes[1].set(xticks=range(3),xticklabels=['D2','D6','D12'],xlabel='QP32 · 8-pixel-wide seam band',ylabel=r'Boundary-specific excess MSE ($10^{-4}$)')
    axes[2].bar([0,1,2,3],[1,1,1.5625,1.5625],color=[F.GREY,F.ORANGE,F.BLUE,F.TEAL],width=.6)
    axes[2].set(xticks=[0,1,2,3],xticklabels=['Full\n512','Core\n256','H32\n320','H64\n320'],ylabel='Total coded area / full-frame area',ylim=(0,1.9))
    for x,area,header in zip(range(4),(1,1,1.5625,1.5625),(88,352,352,352)):
        axes[2].text(x,area+.04,f'{area:g}×',ha='center',fontsize=6)
        axes[2].text(x,.08,f'{header} B',ha='center',fontsize=6,color='white')
    handles=[plt.Line2D([],[],color=COLORS[h],lw=1.2,label=LABELS[h]) for h in (0,32,64)]
    handles += [plt.Line2D([],[],color=F.INK,marker=m,ls='none',label=f'D{d}') for d,m in ((2,'o'),(6,'s'),(12,'^'))]
    fig.legend(handles=handles,loc='lower center',bbox_to_anchor=(.5,.12),ncol=3,frameon=False,fontsize=5.6,columnspacing=1.1)
    fig.text(.5,.96,'Rate, boundaries and coded area',ha='center',weight='normal',fontsize=6.8)
    fig.text(.5,.86,'The fixed-depth control separates coded rate, local error and geometric overhead',ha='center',fontsize=6.5,color=F.MUTED)
    fig.text(.5,.035,'Same-QP diagnostics change both rate and quality · coded area is not latency · header bytes belong to this research format',ha='center',fontsize=5.9,color=F.MUTED)
    F.audit_and_save(fig,'fig_patch_mechanism',
        'Left: mean same-QP percentage increase in actual payload bytes for each fixed-depth model and patch variant,16 images. Centre: at predeclaredQP32, patch-minus-full MSE near the central boundaries minus the corresponding interior excess; radius4 around each central seam. Intervals are paired-image bootstrap and measure a local diagnostic, not a matched-rate causal effect. Right: exact coded area after multiple-of64 padding and research-format header bytes. Halo32 requires four288 windows padded to320, while halo64 uses four320 windows directly; both require1.5625 times the full512 area. Halo64 was added before patch outcomes to test more actual context at equal coded area. Neither area nor header size is a measured runtime or model-map signalling cost.',book)


def qualitative(data,book):
    fig=plt.figure(figsize=(183*F.MM,113*F.MM))
    assets=DATA/'qualitative_windows.npz';provenance=DATA/'qualitative_windows.json'
    if not BUNDLED:
        manifest=json.loads((ROOT/'div2k100_center512_rgb/manifest.json').read_text())
        crops={r['image']:r for r in manifest['images']};arrays={};sources={}
        for name in ('0801.png','0880.png'):
            source=Path(crops[name]['crop_path']);arrays[name[:4]+'_source']=np.load(source,allow_pickle=False)[192:320,192:320]
            sources[str(source)]=hashlib.sha256(source.read_bytes()).hexdigest()
            for suffix in ('full','h0','h32','h64'):
                source=ROOT/f'patch_control_epoch020/d6/{Path(name).stem}_qp32_{suffix}.png'
                arrays[name[:4]+'_'+suffix]=np.asarray(Image.open(source))[192:320,192:320]
                sources[str(source)]=hashlib.sha256(source.read_bytes()).hexdigest()
        np.savez_compressed(assets,**arrays)
        provenance.write_text(json.dumps({'scope':'Predeclared128-square RGB display windows; D6epoch20,QP32,0801/0880,rows/cols192:320. Only cropping, no appearance edits.',
            'source_files_sha256':sources,'asset_sha256':hashlib.sha256(assets.read_bytes()).hexdigest()},indent=2)+'\n')
    if hashlib.sha256(assets.read_bytes()).hexdigest()!=json.loads(provenance.read_text())['asset_sha256']:raise ValueError('Qualitative windows changed')
    windows=np.load(assets,allow_pickle=False)
    for row,name in enumerate(('0801.png','0880.png')):
        variants=[windows[name[:4]+'_'+suffix] for suffix in ('source','full','h0','h32','h64')]
        for col,(label,array) in enumerate(zip(('Source','Full frame','Independent cores','32-pixel halo','64-pixel halo'),variants)):
            ax=fig.add_axes([.055+.188*col,.48-.33*row,.17,.29])
            ax.imshow(array,interpolation='nearest');ax.set_axis_off()
            if row==0:ax.set_title(label,fontsize=6.5,pad=5,color=F.INK)
            if col==0:ax.text(-.08,.5,name[:4],transform=ax.transAxes,ha='right',va='center',fontsize=6,color=F.MUTED,rotation=90)
            ax.axhline(63.5,color='white',alpha=.5,lw=.35);ax.axvline(63.5,color='white',alpha=.5,lw=.35)
    fig.text(.5,.96,'Reconstruction at the patch intersection',ha='center',weight='normal',fontsize=6.8)
    fig.text(.5,.87,'D6 epoch20 · QP32 · fixed 128×128 window at the central patch intersection',ha='center',fontsize=6.5,color=F.MUTED)
    fig.text(.5,.035,'Images and view window fixed before outcomes · 8-bit display only · light cross marks the patch boundary',ha='center',fontsize=6,color=F.MUTED)
    F.audit_and_save(fig,'fig_patch_visual_check',
        'Predeclared qualitative checks on DIV2K0801 and0880 with D6 epoch20 atQP32. All panels show the same128-square window (rows/columns192:320) around the central patch intersection of the512 crop. RGB images are rounded to8bit for display; quantitative analysis uses floating-point reconstructions. Source and full-frame panels use identical cross overlays for visual alignment. Images/windows were not selected by favourable outcomes. This same-QP comparison is not matched in bitrate.',book)


def main():
    source=DATA/'analysis.json';data=json.loads(source.read_text())
    if data['cases']!=240 or len(data['images'])!=16:raise ValueError('Complete predeclared patch study required')
    F.OUT=OUT;F.AUDIT.clear();F.CAPTIONS.clear()
    with PdfPages(OUT/'patch_control_atlas.pdf',metadata=F.PDF_META) as book:
        matched(data,book);mechanism(data,book);qualitative(data,book)
    (OUT/'figure_evidence.json').write_text(json.dumps({'analysis_sha256':hashlib.sha256(source.read_bytes()).hexdigest(),
        'script_sha256':hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),'captions':F.CAPTIONS,'layout_audit':F.AUDIT},indent=2)+'\n')


if __name__=='__main__':main()
