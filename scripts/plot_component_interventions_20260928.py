"""Publication panel for a complete frozen-weight sensitivity experiment."""
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
DATA,OUT=paths('component_interventions')
STATES=[('repair_identity','Repair',F.ORANGE),('adapters_identity','Adapters',F.TEAL),('both_identity','Both',F.BLUE)]


def main():
    source=DATA/'analysis.json';data=json.loads(source.read_text())
    if (data['n_sequences'],data['n_outputs'])!=(53,212):raise ValueError('Complete factorial required')
    F.OUT=OUT;F.AUDIT.clear();F.CAPTIONS.clear()
    fig=plt.figure(figsize=(183*F.MM,78*F.MM))
    a=fig.add_axes([.075,.27,.25,.55]);b=fig.add_axes([.415,.27,.25,.55]);c=fig.add_axes([.77,.27,.20,.55])
    for ax,letter,title in [(a,'a','Frozen-checkpoint sensitivity'),(b,'b','Where does the error change?'),(c,'c','Joint versus separate penalties')]:F.panel(ax,letter,title)
    summaries={s['variant']:s['metrics'] for s in data['summaries']}
    rng=np.random.default_rng(20260928)
    for i,(variant,label,color) in enumerate(STATES):
        rows=[r for r in data['contrasts'] if r['variant']==variant];stats=summaries[variant]['psnr_loss_db']
        a.scatter(i+rng.uniform(-.16,.16,len(rows)),[r['psnr_loss_db'] for r in rows],s=6,facecolors='none',edgecolors=color,alpha=.42,lw=.45)
        a.vlines(i,*stats['ci95'],color=F.INK,lw=1.1,zorder=3);a.plot(i,stats['mean'],'o',color=F.INK,ms=3,zorder=4)
        for offset,key,marker,filled in [(-.12,'boundary_mse_increase_percent','o',True),(.12,'interior_mse_increase_percent','s',False)]:
            s=summaries[variant][key];b.vlines(i+offset,*s['ci95'],color=color,lw=.9)
            b.plot(i+offset,s['mean'],marker,ms=3.5,color=color,mfc=color if filled else 'white',mew=.8)
    a.axhline(0,color=F.GREY,lw=.6);b.axhline(0,color=F.GREY,lw=.6)
    a.text(0,.55,f"Repair mean\n{summaries['repair_identity']['psnr_loss_db']['mean']:+.4f} dB",ha='center',va='bottom',fontsize=5.5,color=F.ORANGE)
    labels=[s[1] for s in STATES]
    a.set(xticks=range(3),xticklabels=labels,xlabel='Replaced by identity',ylabel='RGB PSNR loss from trained model (dB)',xlim=(-.4,2.4))
    b.set(xticks=range(3),xticklabels=labels,xlabel='Replaced by identity',ylabel='MSE increase from trained model (%)',xlim=(-.4,2.4))
    values=np.sort([r['joint_minus_sum_loss_db'] for r in data['interaction']['per_sequence']])
    c.step(values,np.arange(1,len(values)+1)/len(values),where='post',color=F.BLUE,lw=1.05)
    c.axvline(0,color=F.GREY,lw=.6,ls='--');c.set(xlabel='Joint − summed individual\nPSNR losses (dB)',ylabel='Fraction of sequences',ylim=(0,1.02),yticks=[0,.5,1])
    b.legend(handles=[Line2D([],[],marker='o',color=F.INK,lw=0,label='Boundary',ms=3),Line2D([],[],marker='s',color=F.INK,mfc='white',lw=0,label='Interior',ms=3)],loc='upper left',frameon=False,fontsize=5.6)
    fig.text(.5,.11,'Same 53 frames, QP32 and fixed Q90 router maps · dots: sequences · black means and paired 95% intervals',ha='center',fontsize=5.8,color=F.MUTED)
    fig.text(.5,.045,'Weights are frozen: removing a trained component is not a retraining ablation or a runtime measurement',ha='center',fontsize=5.8,color=F.MUTED)
    with PdfPages(OUT/'component_interventions_atlas.pdf',metadata=F.PDF_META) as book:
        F.audit_and_save(fig,'fig_component_interventions','Frozen e15 checkpoint, all53 CTC first frames atQP32, the same pre-existing Q90 router map per frame. Repair, adapters, or both are replaced by identity without retraining or map reselection. Positive PSNR loss in panel a means that intervention harms the trained reconstruction. Open points are individual sequences; black points and intervals show means and5000 paired-sequence bootstrap95% intervals conditional on the fixed weights and development corpus. Panel b averages within-sequence relative RGB MSE changes in an8-pixel-wide band around internal256-pixel tile boundaries and in the remaining valid pixels, with paired95% intervals. The outer image perimeter is not a tile boundary. Panel c shows the distribution of joint PSNR loss minus the sum of the two separate losses; this interaction depends on the metric scale. Exact repair/head replay and the unchanged trained baseline are verified before measurement. These interventions measure dependence of this checkpoint on learned modules, not the quality attainable by separately optimized alternatives, a runtime benefit, or external-test generalization.',book)
    (OUT/'figure_evidence.json').write_text(json.dumps({'analysis_sha256':hashlib.sha256(source.read_bytes()).hexdigest(),'script_sha256':hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),'captions':F.CAPTIONS,'layout_audit':F.AUDIT},indent=2)+'\n')


if __name__=='__main__':main()
