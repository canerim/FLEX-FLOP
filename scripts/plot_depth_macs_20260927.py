"""Show why synthesis savings and whole-decoder arithmetic savings differ."""
import hashlib
import json
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

REPO=Path(__file__).resolve().parents[1]
BUNDLED=(REPO/'data/refresh20260927/analysis.json').exists()
DATA=REPO/('data/depthmacs20260927' if BUNDLED else 'docs/research/2026-09-27-six-hour/depth_macs')
OUT=REPO/'figs/depthmacs20260927' if BUNDLED else DATA


def main():
    path=DATA/'analysis.json';data=json.loads(path.read_text());rows=data['rows'];OUT.mkdir(parents=True,exist_ok=True)
    if [r['depth'] for r in rows]!=[2,4,6,8,10,12]:raise ValueError('Incomplete architectural grid')
    F.OUT=OUT;F.AUDIT.clear();F.CAPTIONS.clear()
    fig,axs=plt.subplots(1,2,figsize=(183*F.MM,76*F.MM))
    fig.subplots_adjust(left=.08,right=.98,bottom=.32,top=.78,wspace=.29)
    F.panel(axs[0],'a','The prior remains when synthesis gets smaller')
    F.panel(axs[1],'b','A smaller denominator inflates the saving')
    depths=np.array([r['depth'] for r in rows]);synthesis=np.array([r['synthesis_macs'] for r in rows])/1e9
    prior=np.array([r['entropy_neural_macs'] for r in rows])/1e9
    axs[0].bar(depths,prior,1.28,color=F.GREY,label='Neural entropy recovery')
    axs[0].bar(depths,synthesis,1.28,bottom=prior,color=F.BLUE,label='Synthesis transform')
    axs[0].set(xticks=depths,xlabel='Retained synthesis blocks',ylabel='Neural decoder Conv2d cost (GMac)',ylim=(0,100))
    for d,total in zip(depths,prior+synthesis):axs[0].text(d,total+2,f'{total:.1f}',ha='center',fontsize=6,color=F.MUTED)
    for key,color,marker,label in [('synthesis_macs',F.ORANGE,'o','Synthesis only'),
            ('neural_decoder_macs',F.BLUE,'s','Neural decoder'),
            ('encoder_with_reconstruction_macs',F.INK,'^','Encoder + reconstruction')]:
        values=[r[key+'_saving_vs_d12_percent'] for r in rows]
        axs[1].plot(depths,values,color=color,marker=marker,label=label)
    axs[1].set(xticks=depths,xlabel='Retained synthesis blocks',ylabel='Conv2d MAC reduction from D12 (%)',ylim=(-3,82))
    axs[0].legend(loc='upper center',bbox_to_anchor=(.5,-.27),ncol=1,fontsize=6)
    axs[1].legend(loc='upper center',bbox_to_anchor=(.5,-.27),ncol=1,fontsize=6)
    fig.text(.5,.96,'DEPTH SAVINGS DEPEND ON WHAT IS COUNTED',ha='center',weight='bold',fontsize=8)
    fig.text(.5,.025,'512 × 512 · architectural Conv2d traces · no latency inference · D8/D10 are architecture-only controls',
        ha='center',fontsize=6,color=F.MUTED)
    with PdfPages(OUT/'depth_macs_atlas.pdf',metadata=F.PDF_META) as book:
        F.audit_and_save(fig,'fig_depth_mac_denominators',
            'All six architectural depths traced with meta tensors, with a D2/64x64 real CPU trace parity check. Neural decoder includes hyper synthesis, prior fusion, prior reduction, all three causal spatial-prior evaluations and synthesis; rANS, elementwise arithmetic and memory movement are excluded. The prior contributes31.079GMac at512x512 regardless of depth. The right panel changes only the accounting denominator, not the depth modification. Encoder+reconstruction includes analysis, hyper analysis, neural priors and synthesis. MAC ratios do not imply wall-time ratios, especially where coding overlaps synthesis. No trained D8/D10 quality is implied.',book)
    (OUT/'figure_evidence.json').write_text(json.dumps({'analysis_sha256':hashlib.sha256(path.read_bytes()).hexdigest(),
        'script_sha256':hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),'captions':F.CAPTIONS,'layout_audit':F.AUDIT},indent=2)+'\n')
    (OUT/'artifact_manifest.json').write_text(json.dumps({p.name:hashlib.sha256(p.read_bytes()).hexdigest()
        for p in sorted(OUT.iterdir()) if p.suffix in ('.pdf','.svg','.png')},indent=2)+'\n')


if __name__=='__main__':main()
