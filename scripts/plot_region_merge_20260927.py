"""Plot the complete paired region-coalescing control with unchanged depth maps."""
import hashlib,json
from pathlib import Path
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages
from matplotlib.lines import Line2D
from matplotlib.patches import Rectangle
try:
    import paper_refresh_figures as F
except ModuleNotFoundError:
    import build_figures as F
from research_figure_paths_20260927 import paths
DATA,OUT=paths('region_merge')
PATTERNS={'vertical':(F.BLUE,'o','Vertical halves'),'horizontal':(F.ORANGE,'s','Horizontal halves')}


def main():
    source=DATA/'analysis.json';d=json.loads(source.read_text())
    if (d['n_cases'],d['n_profiles'],len(d['rows']))!=(80,800,800):raise ValueError('Complete bank cohort required')
    F.OUT=OUT;F.AUDIT.clear();F.CAPTIONS.clear()
    fig=plt.figure(figsize=(183*F.MM,103*F.MM))
    axes=[fig.add_axes(rect) for rect in ([.09,.60,.37,.32],[.60,.60,.37,.32],[.09,.17,.37,.28],[.60,.17,.37,.28])]
    for ax,letter,title in zip(axes,'abcd',('Same map, fewer independent regions','Same-QP payload change','Matched payload quality','Matched complete-container quality')):F.panel(ax,letter,title)
    a,b,c,e=axes;a.set_axis_off();a.set_xlim(0,10);a.set_ylim(0,5)
    for x,merged,label in [(0.7,False,'Four regions'),(6,True,'Two regions')]:
        for col in range(2):
            a.add_patch(Rectangle((x+col*1.25,1.3),1.25,2.5,facecolor=F.DEPTH_ALL[0 if col==0 else 2],edgecolor='white',lw=.7))
        a.add_patch(Rectangle((x,1.3),2.5,2.5,fill=False,ec=F.INK,lw=.6))
        if not merged:a.plot([x,x+2.5],[2.55,2.55],color='white',lw=1.3)
        a.text(x+1.25,.9,label,ha='center',fontsize=6)
        for yy in ([2.55] if merged else [1.925,3.175]):
            a.text(x+.625,yy,'D2',ha='center',va='center',fontsize=6,color=F.INK)
            a.text(x+1.875,yy,'D6',ha='center',va='center',fontsize=6,color=F.INK)
    a.annotate('',xy=(5.75,2.55),xytext=(3.55,2.55),arrowprops=dict(arrowstyle='->',lw=.7,color=F.MUTED))
    a.text(4.65,3.1,'Merge',ha='center',fontsize=6)
    a.text(5,.25,'Identical pixel depths · both phases retained',ha='center',fontsize=5.7,color=F.MUTED)
    cost=d['cost_audit']['paired_merge_changes'][0]['neural_decoder_macs_change_percent']
    a.text(5,4.5,f'Neural decoder MACs: {cost:+.2f}% (not latency)',ha='center',fontsize=5.7,color=F.BLUE)
    for pattern,(color,marker,label) in PATTERNS.items():
        same=[r for r in d['same_qp'] if r['pattern']==pattern]
        b.plot([r['qp'] for r in same],[r['summaries']['payload_change_percent']['mean'] for r in same],color=color,marker=marker,ms=3,lw=1)
        for row in same:
            ci=row['summaries']['payload_change_percent']['ci95']
            if ci is not None:b.vlines(row['qp'],*ci,color=color,lw=.8)
        for axis,field in [(c,'payload_bpp'),(e,'container_bpp')]:
            selected=[r for r in d['matched_rate'] if (r['pattern'],r['rate_field'],r['interpolator'])==(pattern,field,'linear')]
            offset=-.06 if pattern=='vertical' else .06
            for i,row in enumerate(selected):
                s=row['phase_averaged_gain_db']
                if s['n']:
                    axis.plot(i+offset,s['mean'],marker,color=color,ms=3.5)
                    if s['ci95'] is not None:axis.vlines(i+offset,*s['ci95'],color=color,lw=.9)
                axis.text(i,.98 if pattern=='vertical' else .87,f'n={row["n"]}',color=color,ha='center',va='top',fontsize=5.4,transform=axis.get_xaxis_transform())
    b.axhline(0,color=F.GREY,lw=.6);b.set(xlabel='QP',ylabel='Payload change (%)',xticks=[0,16,32,48,63])
    for ax in (c,e):
        ax.axhline(0,color=F.GREY,lw=.6);ax.set(xticks=range(3),xticklabels=['0.1','0.2','0.4'],xlim=(-.35,2.35),xlabel='Target rate (bits/pixel)',ylabel='RGB PSNR gain (dB)');ax.margins(y=.35)
    common_ylim=(min(c.get_ylim()[0],e.get_ylim()[0]),max(c.get_ylim()[1],e.get_ylim()[1]))
    c.set_ylim(*common_ylim);e.set_ylim(*common_ylim)
    fig.legend(handles=[Line2D([],[],color=color,marker=marker,lw=1,label=label) for color,marker,label in PATTERNS.values()],loc='lower center',bbox_to_anchor=(.5,.055),ncol=2,frameon=False,fontsize=6)
    fig.text(.5,.012,'16 images · phase-averaged paired 95% intervals · fixed D2/D6 epoch-20 maps · CPU research streams',ha='center',fontsize=5.8,color=F.MUTED)
    with PdfPages(OUT/'region_merge_atlas.pdf',metadata=F.PDF_META) as book:
        F.audit_and_save(fig,'fig_region_merge','Fixed50/50 D2/D6 assignments with32-pixel clipped context. Panel a depicts one vertical phase schematically; complementary phases and horizontal halves are both retained. Merging changes four256-square regions into two512×256 or256×512 regions without changing any source pixel\'s assigned model. Panel b averages the two phase-specific same-QP relative payload changes within each image, then averages images. Panels c,d use per-image log-rate interpolation on the common support of both phases of both implementations, with no extrapolation; n labels show supported images out of16. Bars are5000 paired-image bootstrap95% intervals conditional on the fixed checkpoints and cohort. Container rate includes embedded FUFREF2 identity/shape headers and the FUFBNK1 fixed-profile framing, not an optimised production format. The unmerged checkerboard controls are retained in bundled data but have no mergeable equal-depth neighbour. This engineering ablation jointly changes context, region geometry and entropy resets; it is not a learned allocation or runtime result.',book)
    (OUT/'figure_evidence.json').write_text(json.dumps({'analysis_sha256':hashlib.sha256(source.read_bytes()).hexdigest(),'script_sha256':hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),'captions':F.CAPTIONS,'layout_audit':F.AUDIT},indent=2)+'\n')


if __name__=='__main__':main()
