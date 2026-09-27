"""Matched-epoch/rate audit of the four-crop DCVC-UF training monitor.

No inference, checkpoint selection, extrapolation or final-quality claims.
Compares linear and PCHIP interpolation in log-rate and decomposes the
misleading same-QP quality gap into raw quality and rate-normalisation terms.
"""
from __future__ import annotations
import argparse
import datetime
import hashlib
import json
import math
from pathlib import Path
import sys

import numpy as np
from scipy.interpolate import PchipInterpolator

ROOT=Path('/data10/shareddata/can_karsal/dcvcuf_depth_20260927')
REPO=Path(__file__).resolve().parents[1]
IMAGES=[f'{i:04d}.png' for i in range(801,805)]
QPS=[0,16,32,48,63]
RATES=[.1,.2,.4]
DEPTHS=[2,4,6]


def sha(path):return hashlib.sha256(path.read_bytes()).hexdigest()


def load(path,epoch=None):
    digest=sha(path);v=json.loads(path.read_text())
    expected={(i,q) for i in IMAGES for q in QPS}
    if len(v['rows'])!=20 or {(r['image'],r['qp']) for r in v['rows']}!=expected:
        raise ValueError(f'Incomplete/duplicate fixed-crop grid: {path}')
    if v.get('crop')!=512 or any((r['h'],r['w'])!=(512,512) for r in v['rows']):
        raise ValueError('Different crop geometry')
    if v.get('rate_kind')!='deterministic entropy estimate; NOT actual bitstream':
        raise ValueError('Unexpected rate definition')
    if epoch is not None and (v.get('epoch')!=epoch or v.get('global_step')!=epoch*24049):
        raise ValueError('Epoch/step mismatch')
    for r in v['rows']:
        if not math.isfinite(r['psnr_rgb']) or not math.isfinite(r['estimated_bpp']) or r['estimated_bpp']<=0:
            raise ValueError('Nonfinite or nonpositive observation')
    for q in QPS:
        rows=[r for r in v['rows'] if r['qp']==q]
        summaries=[r for r in v['means'] if r['qp']==q]
        if len(summaries)!=1:raise ValueError('Duplicated/missing QP mean')
        for key in ('psnr_rgb','estimated_bpp'):
            if abs(np.mean([r[key] for r in rows])-summaries[0][key])>1e-9:
                raise ValueError('Stored mean does not match image observations')
    if sha(path)!=digest:raise RuntimeError('Input changed while reading')
    return v,{'path':str(path),'sha256':digest}


def interpolate(v,image,rate,method='linear'):
    rr=sorted((r for r in v['rows'] if r['image']==image),key=lambda r:r['estimated_bpp'])
    rates=np.array([r['estimated_bpp'] for r in rr]);quality=np.array([r['psnr_rgb'] for r in rr])
    if not np.all(np.diff(rates)>0):raise ValueError('Repeated or unordered rate knots')
    if not rates[0]<=rate<=rates[-1]:return None
    if method=='linear':return float(np.interp(np.log(rate),np.log(rates),quality))
    if method=='pchip':return float(PchipInterpolator(np.log(rates),quality,extrapolate=False)(np.log(rate)))
    raise ValueError('Unknown interpolator')


def rate_summary(v,rate,method):
    values={image:interpolate(v,image,rate,method) for image in IMAGES}
    present=[q for q in values.values() if q is not None]
    return {'rate':rate,'method':method,'n':len(present),'mean_psnr':float(np.mean(present)) if len(present)==4 else None,'per_image':values}


def audit(root,epoch):
    if not 1<=epoch<=105:raise ValueError('Epoch outside official recipe')
    history=[];sources={};latest={}
    for e in range(1,epoch+1):
        for d in DEPTHS:
            path=root/f'runs/d{d}/validation/step_{e*24049:09d}.json'
            v,src=load(path,e);sources[f'd{d}_epoch{e:03d}']=src
            for rate in RATES:
                history.append({'depth':d,'epoch':e,**rate_summary(v,rate,'linear')})
            if e==epoch:latest[d]=v
    ref_path=root/'reference_d12/validation/latest.json'
    reference,src=load(ref_path)
    archive=json.loads((root/'reference_d12/source_manifest.json').read_text())
    if src['sha256']!=archive['metric_files']['validation/latest.json']['sha256']:
        raise ValueError('Released reference metric changed since archival')
    sources['released_reference']=src
    current={str(d):{method:[rate_summary(v,r,method) for r in RATES] for method in ('linear','pchip')}
             for d,v in (latest|{12:reference}).items()}
    paired=[]
    for d in (4,6,12):
        for method in ('linear','pchip'):
            for rate in RATES:
                a=next(v for v in current[str(d)][method] if v['rate']==rate)
                b=next(v for v in current['2'][method] if v['rate']==rate)
                delta={i:(a['per_image'][i]-b['per_image'][i]) if a['per_image'][i] is not None and b['per_image'][i] is not None else None for i in IMAGES}
                present=[v for v in delta.values() if v is not None]
                paired.append({'depth':d,'reference_depth':2,'rate':rate,'method':method,'n':len(present),
                               'mean_delta_db':float(np.mean(present)) if len(present)==4 else None,
                               'per_image_delta_db':delta,
                               'leave_one_crop_out_means_db':{i:float(np.mean([delta[j] for j in IMAGES if j!=i])) for i in IMAGES} if len(present)==4 else None})
    qp_decomposition=[]
    for d in (4,6):
        for image in IMAGES:
            base=next(r for r in latest[2]['rows'] if r['image']==image and r['qp']==32)
            row=next(r for r in latest[d]['rows'] if r['image']==image and r['qp']==32)
            matched=interpolate(latest[d],image,base['estimated_bpp'],'linear')
            if matched is None:raise ValueError('QP32 reference rate is outside candidate support')
            raw=row['psnr_rgb']-base['psnr_rgb'];adjustment=matched-row['psnr_rgb']
            qp_decomposition.append({'depth':d,'image':image,'qp':32,'reference_rate':base['estimated_bpp'],
                                     'candidate_rate_at_same_qp':row['estimated_bpp'],
                                     'same_qp_delta_db':raw,'rate_adjustment_db':adjustment,'matched_delta_db':matched-base['psnr_rgb']})
            assert abs(raw+adjustment-(matched-base['psnr_rgb']))<1e-12
    qp_means=[{'depth':d,**{key:float(np.mean([v[key] for v in qp_decomposition if v['depth']==d]))
                           for key in ('same_qp_delta_db','rate_adjustment_db','matched_delta_db')}} for d in (4,6)]
    sensitivity=[]
    for d in (4,6):
        for rate in RATES:
            rr={method:next(v for v in paired if v['depth']==d and v['rate']==rate and v['method']==method) for method in ('linear','pchip')}
            sensitivity.append({'depth':d,'rate':rate,'linear_gap_db':rr['linear']['mean_delta_db'],
                                'pchip_gap_db':rr['pchip']['mean_delta_db'],
                                'change_db':rr['pchip']['mean_delta_db']-rr['linear']['mean_delta_db']})
    ref_gaps=[]
    for d in DEPTHS:
        for rate in RATES:
            a=next(v for v in current['12']['linear'] if v['rate']==rate)
            b=next(v for v in current[str(d)]['linear'] if v['rate']==rate)
            ref_gaps.append({'depth':d,'rate':rate,'n':min(a['n'],b['n']),
                             'released_minus_shallow_db':a['mean_psnr']-b['mean_psnr'] if a['mean_psnr'] is not None and b['mean_psnr'] is not None else None})
    return {'created_utc':datetime.datetime.now(datetime.timezone.utc).isoformat(),'epoch':epoch,
            'scope':'Four fixed DIV2K centre crops during ongoing training. Not final benchmark, confidence interval, or checkpoint selection.',
            'rate_kind':'Deterministic entropy estimate; not coded bytes','interpolation_domain':'PSNR against log estimated bpp, per image; no extrapolation',
            'history':history,'current':current,'paired':paired,'qp32_decomposition':qp_decomposition,
            'qp32_decomposition_means':qp_means,'interpolation_sensitivity':sensitivity,'released_reference_gaps':ref_gaps,
            'source_files':sources,'analysis_script_sha256':sha(Path(__file__)),
            'released_reference_caveat':archive['capture_note'],
            'sources':{'pchip':'https://docs.scipy.org/doc/scipy/reference/generated/scipy.interpolate.PchipInterpolator.html'}}


def figures(result,out):
    import matplotlib.pyplot as plt
    from matplotlib.backends.backend_pdf import PdfPages
    import paper_refresh_figures as F
    F.OUT=out;F.AUDIT.clear();F.CAPTIONS.clear()
    colours={2:'#68AEA7',4:'#007F86',6:'#244457',12:'#8999A3'}
    with PdfPages(out/'learning_atlas.pdf',metadata=F.PDF_META) as book:
        fig,axs=plt.subplots(1,2,figsize=(183*F.MM,71*F.MM))
        fig.subplots_adjust(left=.08,right=.98,bottom=.23,top=.78,wspace=.36)
        ax=axs[0];F.panel(ax,'a','Learning at matched estimated rate')
        for d in DEPTHS:
            rows=[v for v in result['history'] if v['depth']==d and v['rate']==.2]
            ax.plot([r['epoch'] for r in rows],[r['mean_psnr'] for r in rows],color=colours[d],marker={2:'o',4:'s',6:'^'}[d],label=f'D{d}',ms=2.4)
        anchor=next(v['mean_psnr'] for v in result['current']['12']['linear'] if v['rate']==.2)
        ax.axhline(anchor,color=colours[12],ls='--',label='Released D12',lw=.9)
        ax.set(xlabel='Completed training epoch',ylabel='Mean RGB PSNR at 0.2 est. bpp (dB)',xticks=[1,5,10,15,20],xlim=(.5,result['epoch']+.5))
        ax.legend(loc='lower right',ncol=2,fontsize=6.5,handlelength=1.4)
        ax=axs[1];F.panel(ax,'b',f'Per-crop depth benefit at epoch {result["epoch"]}')
        for d,offset,marker in [(4,-.12,'s'),(6,.12,'^')]:
            row=next(v for v in result['paired'] if v['depth']==d and v['rate']==.2 and v['method']=='linear')
            ax.scatter([row['per_image_delta_db'][i] for i in IMAGES],np.arange(4)+offset,color=colours[d],marker=marker,s=17,label=f'D{d} − D2',zorder=3)
        ax.axvline(0,color=F.MUTED,lw=.6);ax.grid(False);ax.grid(axis='x',color=F.GRID,lw=.5)
        ax.set_yticks(range(4),[i.removesuffix('.png') for i in IMAGES])
        ax.set(ylabel='Fixed DIV2K crop',xlabel='RGB PSNR difference at 0.2 est. bpp (dB)',ylim=(3.5,-.5))
        ax.legend(loc='lower right',fontsize=6.5)
        fig.text(.5,.96,'ONGOING TRAINING · FOUR-CROP DIAGNOSTIC',ha='center',weight='bold',fontsize=8)
        fig.text(.5,.035,'Same epoch and per-image rate · no extrapolation · released D12 has different training provenance',ha='center',fontsize=6.5,color=F.MUTED)
        F.audit_and_save(fig,'fig_learning_matched_rate','Interim monitoring only. Per-image log-rate linear interpolation, four fixed centre crops. No confidence interval or final capacity conclusion. The released anchor has a different training history.',book)
        fig,axs=plt.subplots(1,2,figsize=(183*F.MM,66*F.MM))
        fig.subplots_adjust(left=.08,right=.98,bottom=.26,top=.78,wspace=.38)
        ax=axs[0];F.panel(ax,'a','Same QP conceals a rate difference')
        rows=result['qp32_decomposition_means'];x=np.arange(2)
        raw=np.array([v['same_qp_delta_db'] for v in rows]);adj=np.array([v['rate_adjustment_db'] for v in rows])
        ax.bar(x,raw,color=F.GREY,width=.5,label='Same-QP quality gap')
        ax.bar(x,adj,bottom=raw,color=F.BLUE,width=.5,label='Rate-normalisation term')
        for i,row in enumerate(rows):ax.text(i,row['matched_delta_db']+.012,f"{row['matched_delta_db']:.3f}",ha='center',fontsize=7)
        ax.set(xticks=x,xticklabels=['D4 − D2','D6 − D2'],ylabel='Mean quality difference (dB)',ylim=(0,max(r['matched_delta_db'] for r in rows)*1.4))
        ax.legend(loc='upper left',fontsize=6.2,handlelength=1)
        ax=axs[1];F.panel(ax,'b','Check sensitivity to interpolation')
        for d,offset in [(4,-.05),(6,.05)]:
            rr=[v for v in result['interpolation_sensitivity'] if v['depth']==d]
            xx=np.arange(3)+offset
            ax.plot(xx,[r['linear_gap_db'] for r in rr],color=colours[d],marker='s',label=f'D{d} − D2, linear')
            ax.plot(xx,[r['pchip_gap_db'] for r in rr],color=colours[d],marker='o',mfc='white',ls='--',label=f'D{d} − D2, PCHIP')
        ax.set(xticks=range(3),xticklabels=['0.1','0.2','0.4'],xlabel='Matched estimated bitrate (bpp)',ylabel='Mean RGB PSNR advantage (dB)')
        ax.legend(loc='lower center',fontsize=5.8,ncol=2,handlelength=1.2,bbox_to_anchor=(.5,-.43))
        fig.text(.5,.96,f'EPOCH {result["epoch"]} · COMPARISON PROTOCOL MATTERS',ha='center',weight='bold',fontsize=8)
        fig.text(.25,.045,'a: Match each crop to its own D2 QP32 bitrate.',ha='center',fontsize=6,color=F.MUTED)
        F.audit_and_save(fig,'fig_qp_and_interpolation','Panel a decomposes the same-QP32 quality difference into the raw difference and the linear log-rate adjustment to each crop\'s D2 QP32 bitrate. Panel b compares two interpolators at fixed .1/.2/.4 estimated bpp; their difference is sensitivity, not a statistical interval. Both panels use four fixed crops at the same ongoing-training epoch.',book)
    (out/'figure_captions.json').write_text(json.dumps(F.CAPTIONS,indent=2)+'\n')
    (out/'layout_audit.json').write_text(json.dumps(F.AUDIT,indent=2)+'\n')


def report(result,out):
    e=result['epoch'];lines=[f'# {e}. epoch: eşlenmiş derinlik ve bitrate analizi','',
      'Bu sonuçlar dört sabit DIV2K merkez crop üzerinden, devam eden eğitim sırasında hesaplandı. Nihai kalite, codec hızı veya router başarısı değildir.','',
      '| Model | 0,1 tahmini bpp | 0,2 tahmini bpp | 0,4 tahmini bpp |','|---|---:|---:|---:|']
    for d in (2,4,6,12):
        vals=result['current'][str(d)]['linear']
        cells=[f"{r['mean_psnr']:.4f}" if r['mean_psnr'] is not None else f"Kapsam dışı ({r['n']}/4)" for r in vals]
        lines.append('| '+('Released D12' if d==12 else f'D{d}')+' | '+' | '.join(cells)+' |')
    lines+=['','PSNR, her crop için log(tahmini bpp) ekseninde ayrı enterpole edilip dört crop üzerinden ortalanır. Ekstrapolasyon yoktur. Released D12 eğitim geçmişi farklı olduğu için bu fark saf kapasite cezası değildir.','',
      '## Aynı QP neden yanıltıyor?','',
      'Aşağıdaki karşılaştırmada her görüntünün hedef rate değeri o görüntünün D2/QP32 bitrate\'idir; tablodaki sabit 0,2 bpp karşılaştırmasıyla aynı nokta değildir.','',
      '| Fark | Aynı QP32 PSNR farkı | Rate eşleme düzeltmesi | Eş rate toplam fark |','|---|---:|---:|---:|']
    for v in result['qp32_decomposition_means']:
        lines.append(f"| D{v['depth']} − D2 | {v['same_qp_delta_db']:.4f} | {v['rate_adjustment_db']:.4f} | {v['matched_delta_db']:.4f} |")
    max_change=max(abs(v['change_db']) for v in result['interpolation_sensitivity'])
    lines+=['',f'Lineer yerine PCHIP kullanıldığında incelenen altı depth–rate farkının en büyük değişimi {max_change:.4f} dB. Bu bir güven aralığı değil, enterpolasyon duyarlılığıdır. Beş QP arasındaki ara değerler doğrudan ölçüm değildir.',
      '', '## Bundan sonra','',
      '- Dört crop üzerinden nihai derinlik sıralaması veya anlamlılık iddiası kurma; görüntü bazlı farkları koru.',
      '- Ayrı, daha geniş validation değerlendirmesinde aynı checkpoint/epoch ve aynı rate desteğini kullan.',
      '- Gerçek entropy payload, container başlığı ve entropy tahminini ayrı raporla.',
      '- Yakın PSNR\'yi aynı çıktı/model hatası diye yorumlamadan önce blok yürütmesi ve optimizer kapsamı kontrolünü kullan; 20. epoch checkpoint bütünlük denetimi üç modelde geçti.',
      '', '[Öğrenme ve crop farkları](fig_learning_matched_rate.pdf) · [QP ve enterpolasyon etkisi](fig_qp_and_interpolation.pdf)',
      '', '[PCHIP yöntem kaynağı](https://docs.scipy.org/doc/scipy/reference/generated/scipy.interpolate.PchipInterpolator.html).',
      '', 'Ham hesaplar ve kaynak SHA-256 değerleri `analysis.json` dosyasındadır.']
    (out/'REPORT_TR.md').write_text('\n'.join(lines)+'\n')


if __name__=='__main__':
    ap=argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--root',type=Path,default=ROOT);ap.add_argument('--epoch',type=int,default=20)
    ap.add_argument('--out',type=Path,default=REPO/'docs/research/2026-09-27-six-hour/learning_epoch020')
    args=ap.parse_args();args.out.mkdir(parents=True,exist_ok=True)
    data=audit(args.root,args.epoch)
    (args.out/'analysis.json').write_text(json.dumps(data,indent=2,allow_nan=False)+'\n')
    figures(data,args.out);report(data,args.out)
    print(json.dumps({'epoch':args.epoch,'inputs':len(data['source_files']),'qp32_decomposition_means':data['qp32_decomposition_means'],'interpolation_sensitivity':data['interpolation_sensitivity'],'out':str(args.out)},indent=2))
