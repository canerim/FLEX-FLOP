"""Exact-decode evidence and the causal information path, rendered as vectors."""
import hashlib
import json
from pathlib import Path
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle,FancyArrowPatch
from matplotlib.backends.backend_pdf import PdfPages
import numpy as np
import paper_refresh_figures as F

ROOT=Path('/data10/shareddata/can_karsal/dcvcuf_depth_20260927/research')
OUT=Path(__file__).resolve().parents[1]/'docs/research/2026-09-27-six-hour/causal_reference'


def main():
    path=ROOT/'reference_verification_epoch020/verification.json';v=json.loads(path.read_text())
    if not v['all_passed'] or len(v['cases'])!=36:raise ValueError('Complete verification required')
    expected={f'd{d}_{h}x{w}_qp{q}' for d in (2,4,6,12) for h,w in ((64,64),(65,97),(512,512)) for q in (0,32,63)}
    if {r['name'] for r in v['cases']}!=expected:raise ValueError('Unexpected engineering grid')
    for r in v['cases']:
        if not all(r[k] for k in ('fresh_process','exact_reconstruction_match','exact_model_forward_match','exact_latent_and_coder_indexes')):
            raise ValueError('Incomplete correctness evidence')
    OUT.mkdir(parents=True,exist_ok=True);F.OUT=OUT;F.CAPTIONS.clear();F.AUDIT.clear()
    fig=plt.figure(figsize=(183*F.MM,91*F.MM))
    left=fig.add_axes([.08,.36,.32,.42]);right=fig.add_axes([.54,.44,.43,.34])
    F.panel(left,'a','Only decoded information enters the next stage')
    labels=[r'$\hat z$',r'$\hat y_0$',r'$\hat y_1$',r'$\hat y_2$',r'$\hat y_3$']
    for row in range(5):
        for col in range(5):
            fill='#DCECEB' if col<row else 'white'
            left.add_patch(Rectangle((col-.5,row-.5),1,1,fc=fill,ec=F.GRID,lw=.6))
            if col<row:left.scatter(col,row,s=21,color=F.BLUE,zorder=3)
            if col==row:left.add_patch(Rectangle((col-.34,row-.34),.68,.68,fc=F.INK,ec='white',lw=.6))
    left.set(xlim=(-.5,4.5),ylim=(4.5,-.5),xticks=range(5),yticks=range(5),
             xticklabels=labels,yticklabels=['Decode '+s for s in labels],xlabel='Recovered symbol groups')
    left.grid(False);left.spines[['top','right','bottom','left']].set_visible(False);left.tick_params(length=0)
    left.text(.5,-.24,'● already recovered     ■ current stage',transform=left.transAxes,ha='center',fontsize=6,color=F.MUTED)
    left.text(.5,-.35,'Checkpoint + QP provide the fixed starting state.',transform=left.transAxes,ha='center',fontsize=6,color=F.MUTED)
    F.panel(right,'b','36 fresh-process checks, all exact')
    for row,depth in enumerate((2,4,6,12)):
        for col in range(9):
            right.add_patch(Rectangle((col-.47,row-.43),.94,.86,fc=F.DEPTH_ALL[(depth//2)-1],ec='white',lw=.5))
            right.scatter(col,row,s=6,color='white',zorder=3)
    right.set(xlim=(-.55,8.55),ylim=(3.55,-.65),xticks=range(9),xticklabels=['0','32','63']*3,
              yticks=range(4),yticklabels=['D2','D4','D6','D12 release'],xlabel='QP within each input geometry')
    right.grid(False);right.spines[['top','right','bottom','left']].set_visible(False);right.tick_params(length=0)
    for x,label in ((1,'64 × 64'),(4,'65 × 97'),(7,'512 × 512')):
        right.text(x,-.77,label,ha='center',va='bottom',fontsize=6)
    for x in (2.5,5.5):right.axvline(x,color='white',lw=3)
    right.text(.5,-.52,'Equal reconstruction, latents and coder indexes\nOfficial FP32 forward parity also checked',transform=right.transAxes,ha='center',fontsize=6,color=F.MUTED,linespacing=1.4)
    bottom=fig.add_axes([.065,.045,.915,.17],xlim=(0,100),ylim=(0,10));bottom.axis('off')
    bottom.text(0,8,'PROCESS BOUNDARY',fontsize=6,weight='bold',color=F.MUTED)
    bottom.text(0,3.4,'Encoder',fontsize=8,weight='bold')
    bottom.add_patch(FancyArrowPatch((15,3.7),(29,3.7),arrowstyle='-|>',mutation_scale=7,color=F.MUTED,lw=.8))
    bottom.text(38,6.3,'FUFREF1 stream',ha='center',fontsize=7,weight='bold')
    for i,(width,color,label) in enumerate(((9,F.GREY,'88-byte header'),(17,F.BLUE,'rANS payload'))):
        x=29 if i==0 else 38
        bottom.add_patch(Rectangle((x,2.2),width,2.2,fc=color,ec='white',lw=.5))
        bottom.text(x+width/2,.8,label,ha='center',fontsize=5.6,color=F.MUTED)
    bottom.add_patch(FancyArrowPatch((57,3.7),(68,3.7),arrowstyle='-|>',mutation_scale=7,color=F.MUTED,lw=.8))
    bottom.text(70,3.4,'Isolated decoder',fontsize=8,weight='bold')
    bottom.text(100,8,'Model identity checked · source analysis disabled',ha='right',fontsize=6,color=F.MUTED)
    fig.text(.5,.965,'CAUSAL DECODE · CORRECTNESS BEFORE PERFORMANCE',ha='center',weight='bold',fontsize=8)
    fig.text(.5,.89,'CPU FP32 research format · epoch20 D2/D4/D6 and released D12 · no CUDA compatibility or latency claim',ha='center',fontsize=6.3,color=F.MUTED)
    with PdfPages(OUT/'causal_reference_atlas.pdf',metadata=F.PDF_META) as book:
        F.audit_and_save(fig,'fig_causal_reference',
            'The decoder reconstructs each spatial-prior stage using only its resident checkpoint, QP and previously decoded symbols. Lower-triangular dots mark available decoded groups; diagonal squares mark the current group. Four depths, three geometries and three QPs pass all 36 fresh-process checks, including official FP32-forward parity. The stream cartoon is schematic, not a byte-scale partition; the actual custom header is 88 bytes. Correctness is local to the specified CPU numerical convention, not a released-format or speed claim.',book)
    record={'verification_sha256':hashlib.sha256(path.read_bytes()).hexdigest(),'verification_path':str(path),
            'cases':36,'all_passed':True,'script_sha256':hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
            'caption':F.CAPTIONS,'layout_audit':F.AUDIT}
    (OUT/'evidence.json').write_text(json.dumps(record,indent=2)+'\n')


if __name__=='__main__':main()
