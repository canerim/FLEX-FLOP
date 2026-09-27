"""Exact geometry and Conv2d accounting; neither RD nor runtime projections."""
import hashlib
import json
from pathlib import Path
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle,Patch
from matplotlib.backends.backend_pdf import PdfPages
import paper_refresh_figures as F

OUT=Path(__file__).resolve().parents[1]/'docs/research/2026-09-27-six-hour/patch_geometry'


def main():
    source=OUT/'analysis.json';data=json.loads(source.read_text());F.OUT=OUT;F.AUDIT.clear();F.CAPTIONS.clear()
    fig=plt.figure(figsize=(183*F.MM,127*F.MM))
    for i,halo in enumerate((32,64)):
        ax=fig.add_axes([.14+.49*i,.55,.26,.29]);F.panel(ax,chr(97+i),f'{halo}-pixel halo · {256+halo} → 320')
        side=256+halo
        ax.add_patch(Rectangle((0,0),320,320,facecolor='#E8ECEE',edgecolor=F.GREY,lw=.6,hatch='////'))
        ax.add_patch(Rectangle((0,0),side,side,facecolor='#C7E0E0',edgecolor=F.BLUE,lw=.65))
        ax.add_patch(Rectangle((halo,halo),256,256,facecolor='white',edgecolor=F.INK,lw=.8))
        ax.text(halo+128,halo+118,'Retained core',ha='center',va='center',fontsize=7,color=F.INK)
        ax.text(halo+128,halo+151,'256 × 256',ha='center',va='center',fontsize=6,color=F.MUTED)
        ax.plot([0,halo],[-12,-12],color=F.BLUE,lw=.7)
        for edge in (0,halo):ax.plot([edge,edge],[-17,-7],color=F.BLUE,lw=.7)
        ax.text(halo/2,-28,str(halo),ha='center',fontsize=6,color=F.BLUE)
        ax.text(160,352,'Same coded area: 320 × 320',ha='center',fontsize=6,color=F.MUTED)
        ax.set(xlim=(-15,335),ylim=(365,-38),aspect='equal');ax.set_axis_off()
    fields=[('neural_decoder_macs_percent_of_full_d12','Neural decoder'),('encoder_with_reconstruction_macs_percent_of_full_d12','Encoder with reconstruction')]
    for col,(key,title) in enumerate(fields):
        ax=fig.add_axes([.085+.49*col,.14,.365,.25]);F.panel(ax,chr(99+col),title)
        for offset,halo,color in ((-.17,0,F.BLUE),(.17,32,F.ORANGE)):
            rows=[r for r in data['rows'] if r['halo']==halo]
            y=np.arange(len(rows))+offset;values=[r[key] for r in rows]
            ax.barh(y,values,height=.3,color=color,zorder=2)
            for yy,v in zip(y,values):ax.text(v+1.5,yy,f'{v:.1f}',va='center',fontsize=5.2,color=F.MUTED,bbox={'facecolor':'white','edgecolor':'none','pad':.3})
        ax.axvline(100,color=F.INK,lw=.7,ls='--',zorder=3)
        ax.set(yticks=range(6),yticklabels=[f'D{d}' for d in (2,4,6,8,10,12)],xlim=(0,175),xticks=[0,50,100,150],
            xlabel='Conv2d MACs / full-frame D12 (%)')
        ax.invert_yaxis();ax.grid(axis='y',visible=False);ax.grid(axis='x',color=F.GRID,lw=.5,zorder=0)
    fig.legend(handles=[Patch(facecolor='#C7E0E0',edgecolor=F.BLUE,label='Actual source context'),
        Patch(facecolor='#E8ECEE',edgecolor=F.GREY,hatch='////',label='Replicated padding')],
        loc='center',bbox_to_anchor=(.5,.485),ncol=2,frameon=False,fontsize=6)
    fig.legend(handles=[Patch(facecolor=F.BLUE,label='One 512 crop or four 256 cores'),
        Patch(facecolor=F.ORANGE,label='Four patches with halo 32 or 64')],
        loc='lower center',bbox_to_anchor=(.5,.038),ncol=2,frameon=False,fontsize=6)
    fig.text(.5,.975,'PADDING CAN CONSUME THE DEPTH SAVING',ha='center',weight='bold',fontsize=8)
    fig.text(.5,.922,'Bottom-right patch of a 2×2 grid · extra context need not mean extra padded area',ha='center',fontsize=6.4,color=F.MUTED)
    fig.text(.5,.009,'Exact architecture/geometry accounting · no runtime, trained D8/D10 quality, or context-sufficiency claim',ha='center',fontsize=5.8,color=F.MUTED)
    with PdfPages(OUT/'patch_geometry_atlas.pdf',metadata=F.PDF_META) as book:
        F.audit_and_save(fig,'fig_patch_geometry',
            'Top: exact local geometry of the bottom-right256-square core in a512 crop. A32-pixel halo produces a288-square input padded on its bottom/right to320; a64-pixel halo uses320 source pixels directly. Both therefore code1.5625 times the full512 area across four patches. Bottom: Conv2d MACs relative to full-frameD12, including neural entropy recovery in the decoder and source analysis/hyperanalysis in encoder-with-reconstruction. Area scaling was checked by independent meta-tensor traces forD2/D12 at256 and320. Four256 calls share full512 arithmetic but may differ in overhead and rate. Entropy coding, transfers and call overhead are excluded; encoder branch overlap prevents interpreting arithmetic ratios as time ratios. D8/D10 are architectural counts only.',book)
    (OUT/'figure_evidence.json').write_text(json.dumps({'analysis_sha256':hashlib.sha256(source.read_bytes()).hexdigest(),
        'script_sha256':hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),'captions':F.CAPTIONS,'layout_audit':F.AUDIT},indent=2)+'\n')


if __name__=='__main__':main()
