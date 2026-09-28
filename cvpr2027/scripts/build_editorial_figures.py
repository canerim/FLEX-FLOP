"""Replace manuscript tables with data-backed, editable publication plots."""
import json,hashlib
from pathlib import Path
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from matplotlib.backends.backend_pdf import PdfPages
import build_figures as F
ROOT=Path(__file__).resolve().parents[1]
OUT=ROOT/'figs/editorial20260928'
SOURCES=[]
def load(path):
    SOURCES.append(path);return json.loads((ROOT/path).read_text())
def fixed(book):
    d=load('data/research20260927/shared_crossfit_qp32/analysis.json')
    fig,ax=plt.subplots(figsize=(89*F.MM,65*F.MM));fig.subplots_adjust(left=.19,right=.97,bottom=.25,top=.88)
    for r in d['summaries']:
        k=r['policy'];criterion=r['criterion'];m=r['metrics'];x=m['cropped_rgb_loss_db']['mean'];y=m['saving_points']['mean'];n=r['above_nominal_target']['cropped_rgb_loss_db']
        ax.scatter(x,y,s=37,marker='o' if criterion=='mean' else 's',fc=F.COL[k] if criterion=='mean' else 'white',ec=F.COL[k],lw=1.1,zorder=3)
        dx,dy=(-5,2) if k=='uniform' else (5,-1)
        ax.annotate(f'{n}/53',(x,y),xytext=(dx,dy),textcoords='offset points',ha='right' if dx<0 else 'left',fontsize=7,color=F.COL[k])
    ax.set(xlim=(.039,.103),ylim=(11,29),xticks=[.04,.06,.08,.10],yticks=[12,16,20,24,28],xlabel='Mean achieved RGB loss (dB)',ylabel='Synthesis MAC saving (%)')
    ax.grid(color=F.GRID,lw=.4);ax.set_axisbelow(True)
    ax.text(.02,1.07,'Labels: frames exceeding 0.10 dB loss',transform=ax.transAxes,fontsize=7,color=F.MUTED)
    handles=[Line2D([],[],color=F.COL[k],marker='o',lw=0,label={'router':'Router','dither':'Dither','uniform':'Uniform'}[k]) for k in ['router','dither','uniform']]
    handles += [Line2D([],[],color=F.INK,marker='o',lw=0,label='Mean'),Line2D([],[],color=F.INK,marker='s',mfc='white',lw=0,label='Q90')]
    fig.legend(handles=handles,loc='lower center',bbox_to_anchor=(.52,.015),ncol=5,handletextpad=.3,columnspacing=.8,fontsize=7)
    F.audit_and_save(fig,'fig_fixed_control','Actual cropped RGB losses and synthesis MAC savings for fixed mean/Q90 controls on all53 QP32 frames, e15 full-frame reference. Labels count losses above0.10dB. Controls are fitted on other sequence folds, on development-used data. Equal calibration targets do not imply matched achieved quality. No quality fallback.',book)
def metric(book):
    d=load('data/sharedmetric20260927/analysis.json')
    fig,ax=plt.subplots(figsize=(89*F.MM,51*F.MM));fig.subplots_adjust(left=.18,right=.96,bottom=.24,top=.78)
    xs=np.arange(2)
    for off,key,c,m,label in [(-.12,'archived_db_rgb',F.GREY,'o','Archived'),(0,'unclipped_ycbcr444',F.BLUE,'s','CPU 444'),(.12,'clipped_rgb',F.ORANGE,'D','CPU RGB')]:
        vals=[r[key] if key=='archived_db_rgb' else r[key]['loss_db'] for r in d['rows']]
        ax.scatter(xs+off,vals,c=c,marker=m,s=29,label=label,zorder=3)
    ax.set(xticks=xs,xticklabels=['BasketballPass','BQMall'],xlim=(-.5,1.5),ylim=(.05,.082),yticks=[.05,.06,.07,.08],ylabel='PSNR loss (dB)')
    ax.grid(axis='y',color=F.GRID,lw=.4);ax.legend(loc='upper left',ncol=3,bbox_to_anchor=(-.13,1.23),handletextpad=.3,columnspacing=.8)
    F.audit_and_save(fig,'fig_metric_proof','Two QP32 nominal0.10dB router cases. Archived field and actual CPU unclipped YCbCr444 losses agree within1.1e-6dB. Explicit clipping and RGB conversion change the metric. All are losses relative to e15 full-frame synthesis. Two metric checks, not a pooled quality evaluation.',book)
def schedule(book):
    # Zero-based epoch intervals transcribed from the pinned upstream schedule;
    # this is a protocol diagram, not an experimental learning curve.
    spec={'upstream_commit':'cbdae87a5445','epoch_intervals':[[0,45,256,2e-4],[45,70,256,5e-5],[70,90,256,1e-5],[90,95,512,2e-4],[95,99,512,5e-5],[99,103,512,1e-5],[103,105,512,1e-6]]}
    fig,axs=plt.subplots(1,2,figsize=(183*F.MM,54*F.MM));fig.subplots_adjust(left=.095,right=.98,bottom=.25,top=.78,wspace=.30)
    for ax,rows,letter,title in [(axs[0],spec['epoch_intervals'][:3],'a','256 × 256 crops · epochs 0–89'),(axs[1],spec['epoch_intervals'][3:],'b','512 × 512 crops · epochs 90–104')]:
        F.panel(ax,letter,title)
        xs=[r[0] for r in rows]+[rows[-1][1]];ys=[r[3] for r in rows]+[rows[-1][3]]
        ax.step(xs,ys,where='post',color=F.BLUE,lw=1.8)
        for r in rows:
            ax.axvspan(r[0],r[1],color=F.BLUE,alpha=.04 if rows.index(r)%2==0 else .10)
            ax.scatter([r[0]],[r[3]],s=19,c=F.BLUE,zorder=3)
            label={2e-4:r'$2\times10^{-4}$',5e-5:r'$5\times10^{-5}$',1e-5:r'$10^{-5}$',1e-6:r'$10^{-6}$'}[r[3]]
            ax.annotate(label,((r[0]+r[1])/2,r[3]),xytext=(0,5),textcoords='offset points',ha='center',fontsize=7,color=F.INK)
        ax.set(yscale='log',ylim=(7e-7,4e-4),xlim=(xs[0],xs[-1]),xticks=xs,xlabel='Epoch boundary (zero-based)',ylabel='Learning rate')
        ax.set_yticks([1e-6,1e-5,1e-4],[r'$10^{-6}$',r'$10^{-5}$',r'$10^{-4}$']);ax.minorticks_off()
    F.audit_and_save(fig,'fig_official_schedule','Unshortened105-epoch upstream image-training schedule, zero-based epoch boundaries. Both panels have independently scaled horizontal axes; right panel magnifies the final15epochs. Same seven stages apply to D2/D4/D6. This depicts the protocol, not measured training performance.',book)
    (OUT/'schedule_source.json').write_text(json.dumps(spec,indent=2)+'\n')
def milestone(book):
    d=load('data/refresh20260927/depth_milestone_audit.json')
    fig,ax=plt.subplots(figsize=(89*F.MM,54*F.MM));fig.subplots_adjust(left=.23,right=.96,bottom=.24,top=.82)
    for epoch,c,off,marker in [(20,F.GREY,-.13,'o'),(30,F.BLUE,.13,'s')]:
        rows=[r for r in d['comparisons'] if r['epoch']==epoch]
        for i,r in enumerate(rows):
            v=r['mean'];lo,hi=r['ci95'];ax.errorbar(v,i+off,xerr=[[v-lo],[hi-v]],fmt=marker,c=c,ms=4,capsize=2,lw=1)
    ax.set_yticks([0,1,2],['D4 − D2','D6 − D2','D6 − D4']);ax.set(xlim=(-.035,.225),ylim=(2.55,-.6),xticks=[0,.1,.2],xlabel='RGB PSNR difference (dB)')
    ax.axvline(0,color=F.MUTED,lw=.7);ax.grid(axis='x',color=F.GRID,lw=.4)
    ax.legend(handles=[Line2D([],[],c=c,marker=m,label=f'Epoch {e}',lw=1) for e,c,m in [(20,F.GREY,'o'),(30,F.BLUE,'s')]],loc='lower center',bbox_to_anchor=(.45,1.01),ncol=2)
    F.audit_and_save(fig,'fig_depth_milestones','Epoch20 and30 depth comparisons at0.2 actual payload bpp, same99images on common support. Means and paired95% image-bootstrap intervals at fixed weights; no extrapolation, checkpoint selection or training-seed uncertainty. These are intermediate milestones, not final105-epoch rankings.',book)
def main():
    OUT.mkdir(parents=True,exist_ok=True);F.OUT=OUT;F.AUDIT.clear();F.CAPTIONS.clear()
    with PdfPages(OUT/'editorial_atlas.pdf',metadata=F.PDF_META) as book:
        fixed(book);metric(book);schedule(book);milestone(book)
    for name,obj in [('layout_audit',F.AUDIT),('captions',F.CAPTIONS)]:
        (OUT/(name+'.json')).write_text(json.dumps(obj,indent=2)+'\n')
    sha=lambda p:hashlib.sha256(p.read_bytes()).hexdigest()
    (OUT/'artifact_manifest.json').write_text(json.dumps({p.name:sha(p) for p in sorted(OUT.iterdir()) if p.suffix in {'.pdf','.png','.svg'}},indent=2)+'\n')
    (OUT/'source_manifest.json').write_text(json.dumps({p:sha(ROOT/p) for p in SOURCES+['scripts/build_editorial_figures.py','scripts/editorial_visuals_20260928.py']},indent=2)+'\n')
    print('Four table-free evidence figures generated.')
if __name__=='__main__':main()
