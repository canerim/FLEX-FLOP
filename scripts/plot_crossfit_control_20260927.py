"""Publication-style figures for the explicitly limited CPU calibration audit."""
import json
from pathlib import Path
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages
from matplotlib.lines import Line2D
import paper_refresh_figures as F

OUT=Path(__file__).resolve().parents[1]/'docs/research/2026-09-27-six-hour/crossfit_control'


def main():
    data=json.loads((OUT/'analysis.json').read_text());F.OUT=OUT;F.AUDIT.clear();F.CAPTIONS.clear()
    summaries=data['summaries']
    with PdfPages(OUT/'calibration_atlas.pdf',metadata=F.PDF_META) as book:
        fig,axs=plt.subplots(1,2,figsize=(183*F.MM,73*F.MM))
        fig.subplots_adjust(left=.075,right=.98,bottom=.26,top=.78,wspace=.32)
        for ax,field,title,letter in zip(axs,('mean_saving_points','fraction_violating'),
                                      ('The risk target changes the compute frontier','A mean target is not a frame guarantee'),('a','b')):
            F.panel(ax,letter,title)
            for criterion,ls in [('mean','-'),('q90','--')]:
                for policy in ('router','dither','uniform'):
                    rows=[r for r in summaries if r['criterion']==criterion and r['policy']==policy and r['budget']<=.3]
                    scale=100 if field=='fraction_violating' else 1
                    ax.plot([r['budget'] for r in rows],[r[field]*scale for r in rows],
                            color=F.COL[policy],marker=F.MARK[policy],ls=ls,ms=2.5,
                            mfc=F.COL[policy] if criterion=='mean' else 'white')
            ax.set(xlabel='Calibration target (table PSNR loss, dB)',xticks=[.05,.1,.15,.2,.3])
            ax.axvline(.1,color=F.GREY,lw=.6,ls=':',zorder=0)
        axs[0].set(ylabel='Modelled decoder MAC saving (%)',ylim=(-3,42))
        axs[1].set(ylabel='Held-out target violations (%)',ylim=(0,53))
        handles=[Line2D([],[],color=F.COL[p],marker=F.MARK[p],label=F.LABEL[p]) for p in ('router','dither','uniform')]
        handles += [Line2D([],[],color=F.INK,ls='-',label='Mean calibration'),Line2D([],[],color=F.INK,ls='--',label='Q90 calibration')]
        fig.legend(handles=handles,loc='lower center',bbox_to_anchor=(.5,.085),ncol=5,handlelength=1.6,columnspacing=1.2,fontsize=6)
        fig.text(.5,.96,'SOURCE-FREE CONTROL · SEQUENCE CROSS-FIT DIAGNOSTIC',ha='center',weight='bold',fontsize=8)
        fig.text(.5,.025,'Padded source-error tables · fixed model · 53 sequences × 5 QPs · no final reconstruction or latency claim',ha='center',fontsize=6.2,color=F.MUTED)
        F.audit_and_save(fig,'fig_crossfit_budget_risk',
            'Five-fold sequence-disjoint calibration of a fixed router, Bayer dither and uniform depth. Solid lines fit the training mean; dashed lines fit its 90th percentile. Every frame-QP remains, including outcomes from infeasible calibration groups. The 0.05 dB Q90 target is infeasible for groups covering 159/265 held-out cases in each policy. These are padded source-error table diagnostics, not delivered mixed-image quality or an untouched test set.',book)
        fig,axs=plt.subplots(1,2,figsize=(183*F.MM,70*F.MM))
        fig.subplots_adjust(left=.075,right=.98,bottom=.27,top=.77,wspace=.34)
        for ax,criterion,letter in zip(axs,('mean','q90'),('a','b')):
            F.panel(ax,letter,'Mean calibration' if criterion=='mean' else 'Q90 calibration')
            for policy in ('router','dither','uniform'):
                rr=[r for r in data['rows'] if r['criterion']==criterion and r['policy']==policy and r['budget']==.1]
                losses=np.sort([r['loss_db'] for r in rr]);assert len(losses)==265
                ax.step(losses,np.arange(1,266)/265,where='post',color=F.COL[policy],label=F.LABEL[policy])
            ax.axvline(.1,color=F.INK,ls='--',lw=.7)
            ax.axhline(.9,color=F.GREY,ls=':',lw=.6)
            ax.set(xlabel='Held-out table PSNR loss (dB)',ylabel='Empirical cumulative fraction',xlim=(-.015,.4),ylim=(0,1.025))
            for policy,yy in [('router',.27),('dither',.15)]:
                row=next(r for r in summaries if r['criterion']==criterion and r['policy']==policy and r['budget']==.1)
                ax.text(.97,yy,f"{F.LABEL[policy]}: {row['violations']}/265 exceed",transform=ax.transAxes,ha='right',fontsize=6,color=F.COL[policy])
        axs[0].legend(loc='lower center',bbox_to_anchor=(1.16,-.39),ncol=3,fontsize=6.5)
        fig.text(.5,.96,'THE SAME 0.1 dB TARGET PRODUCES DIFFERENT TAIL RISKS',ha='center',weight='bold',fontsize=8)
        fig.text(.5,.025,'All 265 cases retained · vertical line: target · horizontal line: 90% · descriptive distributions, not confidence bands',ha='center',fontsize=6,color=F.MUTED)
        F.audit_and_save(fig,'fig_crossfit_loss_distribution',
            'At a 0.1 dB table-loss target, held-out loss distributions differ substantially between training-mean and training-Q90 calibration. Curves are descriptive ECDFs of 265 correlated frame-QP observations, not confidence bands or final cropped reconstruction errors. No source-conditioned test fallback is applied.',book)
    (OUT/'figure_captions.json').write_text(json.dumps(F.CAPTIONS,indent=2)+'\n')
    (OUT/'layout_audit.json').write_text(json.dumps(F.AUDIT,indent=2)+'\n')


if __name__=='__main__':main()
