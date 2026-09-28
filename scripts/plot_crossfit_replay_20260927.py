"""Plot complete fixed-control image replay; never equate nominal and achieved quality."""
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
DATA,OUT=paths('shared_crossfit_qp32')
COLORS={'router':F.BLUE,'dither':F.ORANGE,'uniform':F.GREY}

def interval(ax,x,s,color,marker='o',scale=1):
    ax.plot(x,s['mean']*scale,marker=marker,color=color,ms=4,ls='none',mfc=color if marker=='o' else 'white')
    if s['ci95'] is not None:ax.vlines(x,*(np.array(s['ci95'])*scale),color=color,lw=.8)

def main():
    source=DATA/'analysis.json';d=json.loads(source.read_text())
    if len(d['rows'])!=318 or any(s['n']!=53 for s in d['summaries']):raise ValueError('Complete cohort required')
    F.OUT=OUT;F.AUDIT.clear();F.CAPTIONS.clear()
    fig=plt.figure(figsize=(183*F.MM,119*F.MM))
    axes=[fig.add_axes(p) for p in ([.08,.59,.37,.22],[.58,.59,.37,.22],[.08,.23,.37,.22],[.58,.23,.37,.22])]
    titles=['Achieved quality and synthesis arithmetic','Same images, different colour metrics','Nominal target is not a per-image bound','Replaying the archived table quantity']
    for i,(ax,title) in enumerate(zip(axes,titles)):F.panel(ax,chr(97+i),title)
    for s in d['summaries']:
        c,p=s['criterion'],s['policy'];m=s['metrics'];color=COLORS[p];marker='o' if c=='mean' else 's'
        x=m['cropped_rgb_loss_db'];y=m['saving_points']
        axes[0].plot(x['mean'],y['mean'],marker=marker,color=color,ms=4,ls='none',mfc=color if c=='mean' else 'white')
        axes[0].hlines(y['mean'],*x['ci95'],color=color,lw=.65);axes[0].vlines(x['mean'],*y['ci95'],color=color,lw=.65)
        idx=['router','dither','uniform'].index(p);xx=idx+(-.13 if c=='mean' else .13)
        interval(axes[1],xx,m['rgb_minus_444_loss_db'],color,marker,scale=1e3)
        count=s['above_nominal_target']['cropped_rgb_loss_db']
        axes[2].bar(xx,count,width=.22,color=color,alpha=1 if c=='mean' else .45)
        axes[2].text(xx,count+1,str(count),ha='center',fontsize=5.5,color=F.MUTED)
        values=sorted(r['actual_minus_table_padded_loss_db'] for r in d['rows'] if (r['criterion'],r['policy'])==(c,p))
        axes[3].plot(np.array(values)*1e4,np.arange(1,54)/53,color=color,ls='-' if c=='mean' else '--',lw=.9)
    axes[0].set(xlabel='Final cropped RGB loss (dB)',ylabel='Synthesis MAC saving (%)')
    axes[1].axhline(0,color=F.GREY,lw=.6);axes[1].set(xticks=range(3),xticklabels=['Router','Dither','Uniform'],ylabel=r'RGB loss − 444 loss ($10^{-3}$ dB)',xlim=(-.45,2.45))
    axes[2].set(xticks=range(3),xticklabels=['Router','Dither','Uniform'],ylabel='Frames above 0.1 dB RGB loss',ylim=(0,57),yticks=[0,15,30,45,53],xlim=(-.45,2.45))
    axes[3].axvline(0,color=F.GREY,lw=.6);axes[3].set(xlabel=r'Actual − table padded loss ($10^{-4}$ dB)',ylabel='Cumulative fraction',ylim=(0,1.04))
    handles=[Line2D([],[],color=c,lw=1.2,label=p.title()) for p,c in COLORS.items()]
    handles += [Line2D([],[],color=F.INK,marker=m,ls=l,mfc=f,label=t) for m,l,f,t in [('o','-',F.INK,'Mean calibration'),('s','--','white','Q90 calibration')]]
    fig.legend(handles=handles,loc='lower center',bbox_to_anchor=(.5,.085),ncol=5,fontsize=5.8,frameon=False,columnspacing=1.4)
    fig.text(.5,.967,'Fixed controls, actual reconstructions',ha='center',weight='normal',fontsize=6.8)
    fig.text(.5,.911,'All 53 CTC first frames · QP32 · nominal 0.1 dB control · 318 policy replays',ha='center',fontsize=6.4,color=F.MUTED)
    fig.text(.5,.04,'Matched controls do not imply matched achieved quality · cropped losses use the same e15 dense anchor',ha='center',fontsize=6,color=F.MUTED)
    fig.text(.5,.015,'CPU neural replay · source RGB derived from YCbCr420 · no bitstream, native latency or external-test claim',ha='center',fontsize=5.8,color=F.MUTED)
    with PdfPages(OUT/'crossfit_replay_atlas.pdf',metadata=F.PDF_META) as book:
        F.audit_and_save(fig,'fig_crossfit_actual_replay','All six fixed control/policy combinations replayed on every sequence. Controls are cross-fitted across sequences, while router weights are fixed. Panel a reports achieved explicitly converted/clipped RGB loss against e15 full-frame synthesis and architectural synthesis MAC saving, with conditional paired-sequence bootstrap intervals. It is not an equal-quality frontier. The paired router-minus-dither saving intervals include zero: mean control1.87pp[−1.60,5.19], Q90control2.54pp[−0.42,5.54]; these53-sequence results do not establish a positive allocation premium. Panel b reports paired RGB-minus-unclipped444 losses on identical cropped support and reference. Panel c counts RGB losses above the same numeric nominal threshold; control calibration used a different metric, support and reference, so this does not isolate a calibration error. Panel d compares actual padded loss against archived scalar R with the table prediction; backend and historical implementation effects are not separated from mixed-map effects. All53 sequences, including infeasible calibration cases, remain included. This is development-corpus neural replay, not a causal bitstream or native runtime benchmark.',book)
    (OUT/'figure_evidence.json').write_text(json.dumps({'analysis_sha256':hashlib.sha256(source.read_bytes()).hexdigest(),'script_sha256':hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),'captions':F.CAPTIONS,'layout_audit':F.AUDIT},indent=2)+'\n')
if __name__=='__main__':main()
