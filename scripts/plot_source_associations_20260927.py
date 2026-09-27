"""Plot every predeclared cheap-feature association; no predictive claims."""
import hashlib
import json
from pathlib import Path
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap
from matplotlib.backends.backend_pdf import PdfPages
import paper_refresh_figures as F

OUT=Path(__file__).resolve().parents[1]/'docs/research/2026-09-27-six-hour/source_features'
FEATURES=('rgb_std','rgb_gradient_energy','rgb_laplacian_energy')


def main():
    path=OUT/'association.json';data=json.loads(path.read_text());rows=data['rows']
    if len(rows)!=18:raise ValueError('All18 predeclared combinations are required')
    lookup={(r['depth'],r['rate'],r['feature']):r for r in rows}
    keys=[(d,r) for d in (4,6) for r in (.1,.2,.4)]
    values=np.array([[np.nan if lookup[d,r,f]['spearman_rho'] is None else lookup[d,r,f]['spearman_rho'] for f in FEATURES] for d,r in keys])
    labels=[f'D{d} − D2 · {r:.1f} bpp · n={lookup[d,r,FEATURES[0]]["n"]}' for d,r in keys]
    F.OUT=OUT;F.AUDIT.clear();F.CAPTIONS.clear()
    fig=plt.figure(figsize=(183*F.MM,91*F.MM))
    left=fig.add_axes([.21,.27,.29,.48]);right=fig.add_axes([.64,.27,.33,.48])
    F.panel(left,'a','All 18 feature–benefit associations')
    F.panel(right,'b','Gradient energy at 0.2 payload bpp')
    cmap=LinearSegmentedColormap.from_list('feature_association',[F.ORANGE,'#F8F9F9',F.BLUE]);cmap.set_bad(F.GRID)
    im=left.imshow(values,aspect='auto',cmap=cmap,vmin=-1,vmax=1,interpolation='none')
    left.set(xticks=range(3),xticklabels=['RGB\nvariation','Gradient\nenergy','Laplacian\nenergy'],yticks=range(6),yticklabels=labels)
    left.grid(False);left.tick_params(length=0);left.spines[['top','right','bottom','left']].set_visible(False)
    left.axhline(2.5,color='white',lw=3)
    for (y,x),v in np.ndenumerate(values):
        left.text(x,y,'—' if not np.isfinite(v) else f'{v:+.2f}',ha='center',va='center',fontsize=6,
            color='white' if abs(v)>.65 else F.INK)
    cb=fig.colorbar(im,cax=fig.add_axes([.24,.14,.23,.022]),orientation='horizontal',ticks=[-1,0,1])
    cb.set_label('Spearman rank correlation',fontsize=6,labelpad=2);cb.ax.tick_params(labelsize=5.7)
    r=lookup[6,.2,'rgb_gradient_energy']
    xx=np.array(r['feature_values']);yy=np.array(r['depth_gain_db'])
    if len(xx):right.scatter(xx,yy,s=10,facecolors=F.BLUE,edgecolors='white',linewidths=.25,alpha=.75,rasterized=False)
    right.axhline(0,color=F.GREY,lw=.7)
    if len(xx) and (xx>0).all():right.set_xscale('log')
    right.set(xlabel='Source RGB gradient energy',ylabel='D6 − D2 RGB PSNR (dB)')
    right.text(.97,.96,f'n={r["n"]}',transform=right.transAxes,ha='right',va='top',fontsize=6,color=F.MUTED)
    fig.text(.5,.96,'VISUAL COMPLEXITY IS A HYPOTHESIS TO TEST',ha='center',weight='bold',fontsize=8)
    fig.text(.5,.88,'Source-only statistics versus additional-depth benefit · frozen epoch20 models',ha='center',fontsize=6.5,color=F.MUTED)
    fig.text(.5,.035,'Exploratory association · no trained router, significance selection, causal claim or held-out prediction',ha='center',fontsize=6,color=F.MUTED)
    with PdfPages(OUT/'source_associations_atlas.pdf',metadata=F.PDF_META) as book:
        F.audit_and_save(fig,'fig_source_depth_association',
            'All18 predeclared combinations of three source-only RGB statistics, two depth comparisons and three actual payload rates. Each coefficient uses images within that pair’s measured rate support, with counts shown; no extrapolation. The predeclared scatter panel shows gradient energy versus D6−D2 at0.2bpp, independently of its observed correlation. These are descriptive associations at epoch20, not significance tests, causal feature effects, an untouched test set or evidence that a cheap router succeeds.',book)
    (OUT/'figure_evidence.json').write_text(json.dumps({'association_sha256':hashlib.sha256(path.read_bytes()).hexdigest(),
        'script_sha256':hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),'captions':F.CAPTIONS,'layout_audit':F.AUDIT},indent=2)+'\n')


if __name__=='__main__':main()
