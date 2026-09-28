"""Common-support released-reference plot; reconstruction evidence only."""
import hashlib,json
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
DATA,OUT=paths('released_anchor_qp32')


def main():
    source=DATA/'analysis.json';d=json.loads(source.read_text())
    if (d['n_sequences'],d['n_policy_cases'])!=(53,318):raise ValueError('Full common-support cohort required')
    F.OUT=OUT;F.AUDIT.clear();F.CAPTIONS.clear()
    fig=plt.figure(figsize=(183*F.MM,79*F.MM));a=fig.add_axes([.20,.23,.34,.61]);b=fig.add_axes([.68,.23,.28,.61])
    F.panel(a,'a','One released reference for every output');F.panel(b,'b','Fine-tuning changes the anchor')
    rows=[('e15 full frame',d['anchor_summaries']['e15_full_rgb_loss_vs_released_db'],F.INK,True)]
    for criterion in ('mean','q90'):
        for policy in ('router','dither','uniform'):
            summary=next(r for r in d['policy_summaries'] if (r['criterion'],r['policy'])==(criterion,policy))
            rows.append((f'{policy.capitalize()} · {"mean" if criterion=="mean" else "Q90"}',summary['metrics']['cropped_rgb_loss_vs_released_db'],F.COL[policy],criterion=='mean'))
    for y,(name,stats,color,filled) in enumerate(rows):
        a.hlines(y,*stats['ci95'],color=color,lw=1);a.plot(stats['mean'],y,'o' if filled else 's',ms=4,color=color,mfc=color if filled else 'white',mew=.9)
    a.axvline(0,color=F.GREY,lw=.6);a.axvline(.1,color=F.GREY,lw=.6,ls='--')
    a.set(yticks=range(len(rows)),yticklabels=[r[0] for r in rows],ylim=(6.6,-.6),xlabel='RGB PSNR loss versus released D12 (dB)')
    x=np.array([r['e15_full_444_loss_vs_released_db'] for r in d['cases']]);y=np.array([r['e15_full_rgb_loss_vs_released_db'] for r in d['cases']])
    low=min(0,float(x.min()),float(y.min()));high=max(float(x.max()),float(y.max()));margin=max(.003,(high-low)*.08)
    b.plot([low-margin,high+margin],[low-margin,high+margin],color=F.GREY,lw=.6,ls='--')
    b.scatter(x,y,s=11,facecolors='none',edgecolors=F.BLUE,lw=.7,alpha=.75)
    b.axhline(0,color=F.GRID,lw=.5);b.axvline(0,color=F.GRID,lw=.5)
    b.set(xlim=(low-margin,high+margin),ylim=(low-margin,high+margin),xlabel='e15 444 loss versus released (dB)',ylabel='e15 RGB loss versus released (dB)')
    fig.text(.5,.10,'53 frames · QP32 · cropped valid support · CPU FP32 · paired 95% intervals conditional on fixed checkpoints',ha='center',fontsize=5.7,color=F.MUTED)
    fig.text(.5,.045,'The same output can have a different reported loss when its reference changes; paired policy differences remain unchanged',ha='center',fontsize=5.6,color=F.MUTED)
    with PdfPages(OUT/'released_anchor_atlas.pdf',metadata=F.PDF_META) as book:
        F.audit_and_save(fig,'fig_released_anchor','All53 CTC first frames atQP32, using the same replicate-to256 input padding and cropped valid support as the shared-exit replay. Stock DMCI synthesis strictly loads the official releasedD12; all255 analysis/entropy tensors match e15 before latent reuse and the first frame verifies exact stock-versus-remapped synthesis. Panel a reports e15 full-frame and all six fixed-policy outputs against the same released RGB anchor. Means and paired95% sequence-bootstrap intervals are conditional on this development corpus and checkpoints. Filled points use mean control and hollow points Q90; the full-frame point is a reference-drift control. The dashed0.1 line is a numerical comparison, not an RGB guarantee under the original padded444 calibration. Panel b shows per-sequence e15-versus-released anchor drift in444 and explicitly converted/clippedRGB; dashed diagonal denotes metric agreement. Positive loss means lower PSNR than released. This reconstruction audit is neither a native CUDA bitstream benchmark nor a final depth-bank result.',book)
    (OUT/'figure_evidence.json').write_text(json.dumps({'analysis_sha256':hashlib.sha256(source.read_bytes()).hexdigest(),'script_sha256':hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),'captions':F.CAPTIONS,'layout_audit':F.AUDIT},indent=2)+'\n')


if __name__=='__main__':main()
