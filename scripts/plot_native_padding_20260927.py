"""Visualize paired padding-policy measurements without claiming CUDA parity."""
import hashlib,json
from pathlib import Path
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages
from matplotlib.lines import Line2D
try:
    import paper_refresh_figures as F
except ModuleNotFoundError:
    import build_figures as F
from research_figure_paths_20260927 import paths
DATA,OUT=paths('native_padding')
STYLES={'full':(F.INK,'-','Full 512'),'halo0':(F.GREY,':','Core 256, no halo'),'halo32_pad64':(F.ORANGE,'--','Halo 32, image pad64'),'halo64':('#709F95','-.','Halo 64, aligned'),'halo32_native_shape':(F.BLUE,'-','Halo 32, native geometry')}

def main():
    source=DATA/'analysis.json';d=json.loads(source.read_text())
    if d['n_cases']!=240 or len(d['rows'])!=1200:raise ValueError('Full paired cohort required')
    F.OUT=OUT;F.AUDIT.clear();F.CAPTIONS.clear()
    fig=plt.figure(figsize=(183*F.MM,113*F.MM))
    top_axes=[];bottom_axes=[]
    for col,depth in enumerate((2,6,12)):
        top=fig.add_axes([.075+.325*col,.58,.245,.23]);bottom=fig.add_axes([.075+.325*col,.23,.245,.21])
        top_axes.append(top);bottom_axes.append(bottom)
        F.panel(top,chr(97+col),f'D{depth}'+(' · released' if depth==12 else ' · epoch 20'))
        F.panel(bottom,chr(100+col),'Padding-policy effect at equal rate')
        for variant,(color,ls,label) in STYLES.items():
            means=[]
            for qp in (0,16,32,48,63):
                rows=[r for r in d['rows'] if (r['depth'],r['variant'],r['qp'])==(depth,variant,qp)]
                if len(rows)!=16:raise ValueError('Incomplete RD point')
                means.append((np.mean([r['payload_bpp'] for r in rows]),np.mean([r['psnr_rgb'] for r in rows])))
            top.plot(*zip(*means),color=color,ls=ls,lw=1,marker='o' if variant=='halo32_native_shape' else None,ms=2.5)
        for x,rate in enumerate((.1,.2,.4)):
            row=next(r for r in d['matched_rate'] if (r['depth'],r['rate_field'],r['interpolator'],r['target_bpp'])==(depth,'payload_bpp','linear',rate))
            s=row['native_minus_pad64_psnr_db']
            if s['n']:
                bottom.plot(x,s['mean'],'o',color=F.BLUE,ms=4)
                if s['ci95'] is not None:bottom.vlines(x,*s['ci95'],color=F.BLUE,lw=.9)
            bottom.text(x,.96,f'n={row["n"]}',transform=bottom.get_xaxis_transform(),ha='center',va='top',fontsize=5.7,color=F.MUTED)
        top.set(xlabel='Mean payload (bpp)',ylabel='Mean RGB PSNR (dB)' if col==0 else '')
        bottom.axhline(0,color=F.GREY,lw=.6);bottom.set(xticks=range(3),xticklabels=['0.1','0.2','0.4'],xlabel='Target payload (bpp)',ylabel='PSNR gain over image padding (dB)' if col==0 else '',xlim=(-.4,2.4))
        bottom.margins(y=.25)
    for axes in (top_axes,bottom_axes):
        low=min(ax.get_ylim()[0] for ax in axes);high=max(ax.get_ylim()[1] for ax in axes)
        for ax in axes:ax.set_ylim(low,high)
    xlow=min(ax.get_xlim()[0] for ax in top_axes);xhigh=max(ax.get_xlim()[1] for ax in top_axes)
    for ax in top_axes:ax.set_xlim(xlow,xhigh)
    fig.legend(handles=[Line2D([],[],color=c,ls=s,lw=1,label=l) for c,s,l in STYLES.values()],loc='lower center',bbox_to_anchor=(.5,.08),ncol=3,fontsize=5.8,frameon=False)
    fig.text(.5,.968,'Matched padding-policy comparison',ha='center',weight='normal',fontsize=6.8)
    fig.text(.5,.907,'Identical 16-image cohort · five QPs · 256 cores with 32-pixel source context',ha='center',fontsize=6.4,color=F.MUTED)
    fig.text(.5,.025,'CPU FP32 geometry control only · native-shaped is not native CUDA output, format or latency',ha='center',fontsize=6,color=F.MUTED)
    with PdfPages(OUT/'native_padding_atlas.pdf',metadata=F.PDF_META) as book:
        F.audit_and_save(fig,'fig_native_padding_rd','Top: all16 images at all5 native QPs, plotting mean actual payload and mean explicitly converted/clipped RGB PSNR. Joined points are visual guides, not means of interpolated curves. Bottom: per-image log-rate interpolation, common support across all5 variants within each depth, no extrapolation, paired95% image-bootstrap intervals conditional on the cohort and weights. Native geometry keeps the288 image and pads its18-square latent to20 before hyperanalysis; image-pad64 instead pads the source to320. This changes analysis support and hyperprior context together. Full512, core256 and halo64/context320 are aligned controls reused from the primary protocol; full512 payload and reconstruction equivalence is independently verified. No CUDA FP16 or native-wire equality is claimed.',book)
    (OUT/'figure_evidence.json').write_text(json.dumps({'analysis_sha256':hashlib.sha256(source.read_bytes()).hexdigest(),'script_sha256':hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),'captions':F.CAPTIONS,'layout_audit':F.AUDIT},indent=2)+'\n')
if __name__=='__main__':main()
