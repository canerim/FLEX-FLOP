"""Display exact whole-crop allocation bounds, without claiming a trained router."""
import hashlib
import json
from pathlib import Path
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages
try:
    import paper_refresh_figures as F
except ModuleNotFoundError:
    import build_figures as F
from research_figure_paths_20260927 import paths

DATA,OUT=paths('depth_allocation')


def main():
    path=DATA/'analysis.json';data=json.loads(path.read_text())
    if [r['rate'] for r in data['results']]!=[.1,.2,.4]:raise ValueError('All three predeclared rates required')
    F.OUT=OUT;F.AUDIT.clear();F.CAPTIONS.clear()
    fig=plt.figure(figsize=(183*F.MM,121*F.MM))
    for col,row in enumerate(data['results']):
        top=fig.add_axes([.08+.32*col,.57,.245,.23]);bottom=fig.add_axes([.08+.32*col,.21,.245,.23])
        F.panel(top,chr(97+col),f"{row['rate']:.1f} payload bpp · n={row['n']}")
        F.panel(bottom,chr(100+col),'Separate capacity and placement')
        if not row['n']:
            for ax in (top,bottom):ax.text(.5,.5,'No common rate support',ha='center',va='center',transform=ax.transAxes)
            continue
        curves=row['curves'];x=[r['allowed_mean_neural_decoder_gmac'] for r in curves]
        for key,color,ls in [('all_three',F.BLUE,'-'),('without_d4',F.ORANGE,'--'),('blind_histogram',F.GREY,':')]:
            top.plot(x,[r[key]['pooled_psnr_db'] for r in curves],color=color,ls=ls,lw=1.2)
        bottom.plot(x,[r['d4_option_value_db'] for r in curves],color=F.ORANGE,lw=1.1,label='Value of adding D4')
        bottom.plot(x,[r['placement_premium_db'] for r in curves],color=F.BLUE,lw=1.1,label='Value of source-informed placement')
        middle=curves[row['n']]['allowed_mean_neural_decoder_gmac']
        for ax in (top,bottom):
            ax.axvline(middle,color=F.GREY,lw=.6,ls=':')
            ax.set(xlabel='Allowed neural decoder GMAC / crop',xticks=[46,54,62],xlim=(x[0],x[-1]))
        top.set_ylabel('Pooled-crop RGB PSNR (dB)' if col==0 else '')
        bottom.set_ylabel('Conditional PSNR advantage (dB)' if col==0 else '')
        bottom.set_ylim(bottom=-.002)
    handles=[plt.Line2D([],[],color=c,ls=s,lw=1.2,label=l) for c,s,l in
             [(F.BLUE,'-','Source-informed D2/D4/D6'),(F.ORANGE,'--','Source-informed D2/D6'),(F.GREY,':','Best blind histogram: expected MSE')]]
    fig.legend(handles=handles,loc='lower center',bbox_to_anchor=(.5,.105),ncol=3,frameon=False,fontsize=5.7,columnspacing=1.1)
    handles=[plt.Line2D([],[],color=c,lw=1.2,label=l) for c,l in [(F.ORANGE,'Adding D4: three-depth minus two-depth optimum'),(F.BLUE,'Placement: three-depth optimum minus blind expectation')]]
    fig.legend(handles=handles,loc='lower center',bbox_to_anchor=(.5,.063),ncol=2,frameon=False,fontsize=5.7,columnspacing=1.1)
    fig.text(.5,.965,'DO WE NEED THE MIDDLE EXPERT?',ha='center',weight='bold',fontsize=8)
    fig.text(.5,.9,'Exact allocation on whole 512 crops · matched epoch20 D2/D4/D6 · common actual-rate support',ha='center',fontsize=6.4,color=F.MUTED)
    fig.text(.5,.018,'Source-informed interpolated bound · single seed · conditional diagnostics · no within-image routing or latency claim',ha='center',fontsize=5.9,color=F.MUTED)
    with PdfPages(OUT/'allocation_atlas.pdf',metadata=F.PDF_META) as book:
        F.audit_and_save(fig,'fig_depth_allocation',
            'Exact dynamic programming minimizes total RGB MSE under an aggregate architectural neural-decoder MAC budget across whole512 crops. Quality comes from per-image log-rate interpolation at the indicated actual payload rate, with common D2/D4/D6 support and no extrapolation. Top: full three-depth set, two-depth set excludingD4, and the expected MSE of the best source-calibrated blind histogram. Displayed PSNR is derived from pooled MSE, not average image PSNR or expected PSNR. Bottom separates the incremental option value of D4 from assignment value. Dashed vertical line marks a budget equal to uniformD4. Each point permits unused budget. These are conditional optimistic bounds at epoch20, not a trained spatial router, external-test result, measured codec output or wall-time speedup.',book)
    (OUT/'figure_evidence.json').write_text(json.dumps({'analysis_sha256':hashlib.sha256(path.read_bytes()).hexdigest(),
        'script_sha256':hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),'captions':F.CAPTIONS,'layout_audit':F.AUDIT},indent=2)+'\n')


if __name__=='__main__':main()
