"""Analyze a complete 100-image CPU reference-codec validation, without extrapolation."""
from __future__ import annotations
import argparse
import hashlib
import json
import math
from pathlib import Path
import numpy as np
from scipy.interpolate import PchipInterpolator

ROOT=Path('/data10/shareddata/can_karsal/dcvcuf_depth_20260927')
REPO=Path(__file__).resolve().parents[1]
DEPTHS=(2,4,6,12);QPS=(0,16,32,48,63);RATES=(.1,.2,.4)
IMAGES=[f'{i:04d}.png' for i in range(801,901)]
FIELDS=('payload_bpp','container_bpp','estimated_bpp','psnr_rgb','psnr_yuv611')


def sha(path):return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def curve(rows,rate_field):
    ordered=sorted(rows,key=lambda r:r[rate_field])
    x=np.array([r[rate_field] for r in ordered]);y=np.array([r['psnr_rgb'] for r in ordered])
    if len(x)!=5 or not np.isfinite(x).all() or not np.isfinite(y).all() or (x<=0).any() or (np.diff(x)<=0).any():
        raise ValueError('Invalid or repeated RD rates')
    return np.log(x),y


def quality_at(rows,rate,rate_field,method='linear'):
    x,y=curve(rows,rate_field);point=math.log(rate)
    if point<x[0] or point>x[-1]:return None
    if method=='linear':return float(np.interp(point,x,y))
    if method=='pchip':return float(PchipInterpolator(x,y,extrapolate=False)(point))
    raise ValueError('Unknown interpolation method')


def bd_rate(reference,candidate,rate_field,interval=None):
    xr,yr=curve(reference,rate_field);xc,yc=curve(candidate,rate_field)
    if (np.diff(yr)<=0).any() or (np.diff(yc)<=0).any():
        return {'percent':None,'reason':'Nonmonotone quality knots; no automatic pruning'}
    lo=max(yr[0],yc[0]);hi=min(yr[-1],yc[-1])
    if interval is not None:lo=max(lo,interval[0]);hi=min(hi,interval[1])
    if not hi>lo:return {'percent':None,'reason':'No common PSNR interval'}
    a=PchipInterpolator(yr,xr,extrapolate=False).antiderivative()
    b=PchipInterpolator(yc,xc,extrapolate=False).antiderivative()
    gap=float((b(hi)-b(lo)-a(hi)+a(lo))/(hi-lo))
    return {'percent':float(100*np.expm1(gap)),'psnr_interval':[float(lo),float(hi)]}


def describe(values):
    a=np.array(values,dtype=float)
    if not len(a):return {'n':0,'mean':None,'ci95':None}
    if not np.isfinite(a).all():raise ValueError('Nonfinite summary input')
    ci=None
    if len(a)>=5:
        rng=np.random.default_rng(20260927)
        boot=a[rng.integers(0,len(a),(5000,len(a)))].mean(axis=1)
        ci=np.quantile(boot,[.025,.975]).tolist()
    return {'n':len(a),'mean':float(a.mean()),'median':float(np.median(a)),
            'min':float(a.min()),'max':float(a.max()),'ci95':ci,
            'ci_scope':'Paired image resampling conditional on fixed checkpoint and included image cohort; not training-seed uncertainty'}


def mathematical_checks():
    # Analytic log-linear RD curve and a multiplicative rate shift.
    reference=[{'payload_bpp':.05*2**i,'psnr_rgb':25+3*i} for i in range(5)]
    candidate=[dict(r,payload_bpp=r['payload_bpp']*1.1) for r in reference]
    assert abs(bd_rate(reference,candidate,'payload_bpp')['percent']-10)<1e-10
    assert abs(quality_at(reference,.05*math.sqrt(2),'payload_bpp')-26.5)<1e-10
    assert quality_at(reference,.049,'payload_bpp') is None
    assert quality_at(reference,.81,'payload_bpp','pchip') is None
    bad=[dict(r) for r in reference];bad[2]['psnr_rgb']=24
    assert bd_rate(reference,bad,'payload_bpp')['percent'] is None


def analyze(folder):
    mathematical_checks()
    progress=json.loads((folder/'progress.json').read_text())
    if progress['state']!='complete' or progress['completed']!=2000:raise ValueError('Complete 2000-case run required')
    manifest=json.loads((folder/'manifest.json').read_text())
    if manifest['epoch_shallow']!=30:raise ValueError('Expected the predeclared epoch30 milestone')
    provenance=json.loads((REPO/'docs/research/2026-09-27-six-hour/reference_epoch030_provenance.json').read_text())
    for path,digest in provenance['files'].items():
        if sha(path)!=digest:raise ValueError('Immutable run input changed: '+path)
    sources={};rows=[];groups={}
    for depth in DEPTHS:
        files=sorted((folder/f'd{depth}/cases').glob('*.json'))
        if len(files)!=500:raise ValueError('Expected 500 cases per depth')
        seen=set()
        for path in files:
            value=json.loads(path.read_text());image=value['image'];qp=value['qp'];key=(image,qp)
            if key in seen or image not in IMAGES or qp not in QPS:raise ValueError('Duplicated/invalid case')
            seen.add(key)
            if value['depth']!=depth or value['checkpoint_sha256']!=manifest['checkpoints'][str(depth)]:raise ValueError('Model mismatch')
            if not value['exact_isolated_decode'] or (value['height'],value['width'])!=(512,512):raise ValueError('Decode/geometry mismatch')
            stream=folder/f'd{depth}/streams/{path.stem}.fufref'
            if sha(stream)!=value['stream_sha256']:raise ValueError('Encoded stream changed')
            if value['container_bytes']!=value['payload_bytes']+88 or stream.stat().st_size!=value['container_bytes']:
                raise ValueError('Byte accounting mismatch')
            if abs(value['payload_bpp']-8*value['payload_bytes']/512**2)>1e-12:raise ValueError('Rate denominator mismatch')
            if any(not math.isfinite(value[k]) for k in FIELDS):raise ValueError('Nonfinite metric')
            sources[str(path)]=sha(path)
            row={'depth':depth,'image':image,'qp':qp,'monitor_subset':value['monitor_subset'],
                 **{k:value[k] for k in FIELDS},'payload_minus_estimate_bpp':value['payload_bpp']-value['estimated_bpp']}
            rows.append(row);groups.setdefault((depth,image),[]).append(row)
        assert seen=={(i,q) for i in IMAGES for q in QPS}
    native=[]
    for depth in DEPTHS:
        for qp in QPS:
            rr=[r for r in rows if r['depth']==depth and r['qp']==qp]
            native.append({'depth':depth,'qp':qp,'n':100,**{k:float(np.mean([r[k] for r in rr])) for k in FIELDS},
                           'payload_minus_estimate':describe([r['payload_minus_estimate_bpp'] for r in rr])})
    matched=[]
    for field in ('payload_bpp','estimated_bpp','container_bpp'):
        for method in ('linear','pchip'):
            for rate in RATES:
                per_image={i:{str(d):quality_at(groups[d,i],rate,field,method) for d in DEPTHS} for i in IMAGES}
                common=[i for i in IMAGES if all(v is not None for v in per_image[i].values())]
                pairs=[]
                for ref,depth in ((2,4),(2,6),(2,12),(12,2),(12,4),(12,6)):
                    available=[i for i in IMAGES if per_image[i][str(ref)] is not None and per_image[i][str(depth)] is not None]
                    for cohort,ids in [('pairwise_support',available),('four_model_common_support',common)]:
                        delta={i:per_image[i][str(depth)]-per_image[i][str(ref)] for i in ids}
                        pairs.append({'reference_depth':ref,'depth':depth,'cohort':cohort,'images':ids,
                                      'delta_psnr_db':describe(list(delta.values())),'per_image_delta_db':delta,
                                      'monitor_four':describe([v for i,v in delta.items() if int(Path(i).stem)<=804]),
                                      'additional_96':describe([v for i,v in delta.items() if int(Path(i).stem)>804])})
                matched.append({'rate_field':field,'interpolator':method,'target_bpp':rate,
                                'common_n':len(common),'common_images':common,'quality_per_image':per_image,'pairs':pairs})
    bd=[]
    for field in ('payload_bpp','estimated_bpp','container_bpp'):
        for depth in (2,4,6):
            values={}
            for image in IMAGES:
                qualities=[curve(groups[d,image],field)[1] for d in DEPTHS]
                interval=(max(v.min() for v in qualities),min(v.max() for v in qualities))
                values[image]=bd_rate(groups[12,image],groups[depth,image],field,interval)
            valid={i:v['percent'] for i,v in values.items() if v['percent'] is not None}
            bd.append({'depth':depth,'reference_depth':12,'rate_field':field,'interval_scope':'Per-image PSNR intersection across all four models',
                       'summary':describe(list(valid.values())),'per_image':values})
    backend=[]
    for depth in DEPTHS:
        path=ROOT/('reference_d12/validation/latest.json' if depth==12 else f'runs/d{depth}/validation/step_{20*24049:09d}.json')
        stored=json.loads(path.read_text());sources[str(path)]=sha(path)
        for old in stored['rows']:
            now=next(r for r in groups[depth,old['image']] if r['qp']==old['qp'])
            backend.append({'depth':depth,'image':old['image'],'qp':old['qp'],
                            'cpu_minus_archived_gpu_psnr_db':now['psnr_rgb']-old['psnr_rgb'],
                            'cpu_minus_archived_gpu_estimated_bpp':now['estimated_bpp']-old['estimated_bpp']})
    return {'scope':'Interim epoch30 CPU reference codec validation; released D12 has different training provenance',
            'n_images':100,'n_cases':2000,'header_bytes':88,'manifest':manifest,'source_files_sha256':sources,
            'native_qp_means':native,'matched_rate':matched,'bd_rate_vs_released_d12':bd,'rows':rows,
            'backend_comparison':backend,'analysis_script_sha256':sha(__file__),
            'statistics':'Means of per-image PSNRs, linear/PCHIP interpolation in log rate; no extrapolation. BD-rate uses PCHIP log-rate vs PSNR on each image four-model common support. 5000 paired image bootstrap draws, seed20260927, conditional on the cohort and frozen checkpoints; no training-seed uncertainty.',
            'mathematical_checks':'Analytic 10% rate scaling, exact log-linear interpolation, out-of-support refusal, nonmonotone BD-rate refusal'}


def report(x,out):
    lines=['# DIV2K100: gerçek payload ile epoch30 derinlik değerlendirmesi','',
           'D2/D4/D6: 30/105 epoch. D12: released model, farklı eğitim geçmişi. CPU FP32 araştırma formatı; GPU hız veya released-format uyumluluğu iddiası yok.',
           '100 merkez512 crop × dört model × beş QP = 2000 ayrı encode/decode vakası. Her reconstruction bağımsız süreçte birebir doğrulandı.',
           '', '| Payload bpp hedefi | Ortak görüntü sayısı | D4−D2 dB | D6−D2 dB | Released D12−D2 dB |',
           '|---:|---:|---:|---:|---:|']
    for rate in RATES:
        r=next(r for r in x['matched_rate'] if (r['rate_field'],r['interpolator'],r['target_bpp'])==('payload_bpp','linear',rate))
        cells=[]
        for depth in (4,6,12):
            s=next(p['delta_psnr_db'] for p in r['pairs'] if (p['reference_depth'],p['depth'],p['cohort'])==(2,depth,'four_model_common_support'))
            cells.append('—' if not s['n'] else f"{s['mean']:+.4f}"+(f" [{s['ci95'][0]:+.4f}, {s['ci95'][1]:+.4f}]" if s['ci95'] else ''))
        lines.append(f"| {rate:.1f} | {r['common_n']}/100 | "+' | '.join(cells)+' |')
    lines+=['','Aralıklar %95 paired görüntü bootstrap aralığıdır; checkpoint ve ortak bitrate desteğine koşulludur. Seed belirsizliği değildir. Her rate noktasında kapsanan görüntüler farklı olabilir; eksik görüntüler için ekstrapolasyon yapılmadı.',
            '', '| Model | Released D12\'ye göre BD-rate (%) | Geçerli görüntü sayısı |', '|---|---:|---:|']
    for r in x['bd_rate_vs_released_d12']:
        if r['rate_field']=='payload_bpp':
            s=r['summary'];value='—' if not s['n'] else f"{s['mean']:+.3f}"
            lines.append(f"| D{r['depth']} | {value} | {s['n']}/100 |")
    lines+=['','BD-rate her görüntüde dört modelin ortak PSNR aralığında hesaplanır; görüntü sonuçları eşit ağırlıkla ortalanır. Released karşılaştırması saf derinlik etkisini ayırmaz.',
            '', 'Research container başlığı 88 byte/crop = 0,00268555 bpp; bunun 64 byte\'ı kimlik/bütünlük hash\'idir. Bu, minimum production header veya router side-bit maliyeti değildir.',
            'Entropy tahmini, rANS payload ve container sonuçları `analysis.json` içinde ayrı bulunur. CPU ve önceki GPU ölçümleri birleştirilmedi; ilk dört crop\'un backend farkları ayrıca kaydedildi.',
            '', 'Dört crop\'luk monitor ile ek96 validation görüntüsünün farkları her eş-rate çiftinde korunur. Veri, checkpoint seçmek için kullanılmadı.']
    (out/'REPORT_TR.md').write_text('\n'.join(lines)+'\n')


if __name__=='__main__':
    ap=argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--input',type=Path,default=ROOT/'research/div2k100_reference_epoch030')
    ap.add_argument('--out',type=Path,default=REPO/'docs/research/2026-09-27-six-hour/div2k100_epoch030')
    ap.add_argument('--check-math-only',action='store_true');args=ap.parse_args()
    if args.check_math_only:mathematical_checks();print('Analytic RD checks passed')
    else:
        result=analyze(args.input);args.out.mkdir(parents=True,exist_ok=True)
        (args.out/'analysis.json').write_text(json.dumps(result,indent=2,allow_nan=False)+'\n');report(result,args.out)
        print(json.dumps({'cases':result['n_cases'],'images':result['n_images'],'out':str(args.out)}))
