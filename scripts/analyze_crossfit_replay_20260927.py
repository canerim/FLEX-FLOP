"""Analyze only the complete53-sequence QP32 fixed-control reconstruction replay."""
import hashlib
import json
from pathlib import Path
import numpy as np
from analyze_reference_validation_20260927 import describe

REPO=Path(__file__).resolve().parents[1]
SOURCE=Path('/data10/shareddata/can_karsal/dcvcuf_depth_20260927/research/shared_crossfit_qp32')
OUT=REPO/'docs/research/2026-09-27-six-hour/shared_crossfit_qp32'
METRICS=('loss_db','saving_points','cropped_ycbcr444_loss_db','cropped_rgb_loss_db',
         'cropped_yuv611_loss_db','actual_minus_table_padded_loss_db','actual_padded_loss_vs_archived_R_db')


def sha(path):return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def main():
    progress=json.loads((SOURCE/'progress.json').read_text())
    if (progress['state'],progress['completed_sequences'],progress['completed_policy_cases'])!=('complete',53,318):
        raise ValueError('Complete cohort required; do not select completed favourable cases')
    manifest=json.loads((SOURCE/'manifest.json').read_text())
    for p,h in manifest['source_sha256'].items():
        if sha(p)!=h:raise ValueError('Replay input changed: '+p)
    paths=sorted((SOURCE/'cases').glob('*.json'))
    if len(paths)!=53:raise ValueError('Wrong sequence count')
    rows=[];sources={};cases=[]
    for p in paths:
        c=json.loads(p.read_text());sources[str(p)]=sha(p);cases.append(c)
        if manifest['names'][c['sequence_index']]!=c['sequence'] or len(c['rows'])!=6:raise ValueError('Unexpected sequence')
        for r in c['rows']:
            if r['sequence']!=c['sequence'] or r['qp']!=32 or r['budget']!=.1:raise ValueError('Unexpected replay configuration')
            if not all(np.isfinite(r[k]) for k in METRICS):raise ValueError('Nonfinite metric')
            if not r['map'] or any(k not in (2,3,4,5) for k in r['map']):raise ValueError('Unreachable exit')
            r['rgb_minus_444_loss_db']=r['cropped_rgb_loss_db']-r['cropped_ycbcr444_loss_db']
            rows.append(r)
    keys={(r['sequence'],r['criterion'],r['policy']) for r in rows}
    if len(keys)!=318:raise ValueError('Duplicate case')
    summaries=[];paired=[]
    for criterion in ('mean','q90'):
        for policy in ('router','dither','uniform'):
            selected=[r for r in rows if (r['criterion'],r['policy'])==(criterion,policy)]
            if {r['sequence'] for r in selected}!=set(manifest['names']):raise ValueError('Missing sequence')
            summary={'criterion':criterion,'policy':policy,'n':53,
                'metrics':{k:describe([r[k] for r in selected]) for k in (*METRICS,'rgb_minus_444_loss_db')},
                'training_infeasible_cases':sum(not r['training_feasible'] for r in selected),
                'above_nominal_target':{k:sum(r[k]>.1+1e-4 for r in selected) for k in
                    ('loss_db','cropped_ycbcr444_loss_db','cropped_rgb_loss_db','cropped_yuv611_loss_db')}}
            summaries.append(summary)
        pairs=[]
        for name in manifest['names']:
            router=next(r for r in rows if (r['sequence'],r['criterion'],r['policy'])==(name,criterion,'router'))
            dither=next(r for r in rows if (r['sequence'],r['criterion'],r['policy'])==(name,criterion,'dither'))
            pairs.append({'sequence':name,**{k:router[k]-dither[k] for k in METRICS}})
        paired.append({'criterion':criterion,'direction':'router minus dither','per_sequence':pairs,
            'summaries':{k:describe([r[k] for r in pairs]) for k in METRICS}})
    result={'scope':'Actual shared-exit image replay at predeclaredQP32 and nominal0.1dB; 53 sequences and318 fixed-policy cases',
        'manifest':manifest,'source_files_sha256':sources,'rows':rows,'summaries':summaries,'paired_router_dither':paired,
        'statistics':'5000 paired-sequence bootstrap draws, seed20260927; one QP per sequence; conditional on fixed checkpoints, controls and development corpus. No training-seed uncertainty or untouched external test.',
        'comparison_warning':'Mean/Q90 controls are calibrated on other sequences in padded444-MSE against a released anchor. Final losses use cropped e15 full-frame reconstruction. Same numeric budget does not imply matched actual quality, and mean/Q90 calibration is not a per-image bound.',
        'metric_warning':'RGB is explicitly converted and clipped; source is RGB converted from archived YCbCr420, not original RGB acquisition. 611 is a weighted mean of plane PSNRs. No equality among those losses is assumed.',
        'table_replay_warning':'Actual-minus-table uses identical padded support and archived scalarR, but cannot by itself isolate mixed-head interactions from numerical/backend or historic implementation effects.',
        'nominal_exceedance_tolerance_db':1e-4,'analysis_script_sha256':sha(__file__)}
    OUT.mkdir(parents=True,exist_ok=True);(OUT/'analysis.json').write_text(json.dumps(result,indent=2,allow_nan=False)+'\n')
    lines=['# Cross-fit kontrolü gerçek görüntüye taşıma','',
        'QP32, nominal0.1dB,53 CTC ilk frame; her birinde mean/Q90 × router/dither/uniform. Hiçbir case elenmedi; haritalar kaynak hata etiketleri kullanılmadan, başka sequence\'lerde seçilmiş sabit kontrollerle üretildi.',
        '', '| Kalibrasyon | Kural | Synthesis MAC tasarrufu % | Final444 kayıp dB | Final RGB kayıp dB | 444/RGB hedef aşımı |',
        '|---|---|---:|---:|---:|---:|']
    for r in summaries:
        m=r['metrics'];v=r['above_nominal_target']
        lines.append(f"| {r['criterion']} | {r['policy']} | {m['saving_points']['mean']:.3f} | {m['cropped_ycbcr444_loss_db']['mean']:.5f} | {m['cropped_rgb_loss_db']['mean']:.5f} | {v['cropped_ycbcr444_loss_db']}/{v['cropped_rgb_loss_db']} /53 |")
    lines += ['', 'Bu tablo aynı gerçekleşen kaliteye eşlenmiş bir üstünlük karşılaştırması değildir. Ortalama/Q90 kalibrasyonu tek-frame garantisi değildir. Yöntem karar anında testM/R kullanmaz ama corpus daha önce geliştirmede görüldüğü için dış test başarısı iddia edilmez.',
        'Padded released-anchor tablo hedefi ile cropped e15-reference raporlama farklıdır. Aynı sayısal0.1 eşik üzerindeki sayılar tek başına hangi farkın sorumlu olduğunu açıklamaz.',
        'Gerçek RGB dönüşümü bu tekrarda açıkça yapılır. Eski265-pair arşivinin db_rgb alanı ise444-MSE idi; bu53-sequence/QP32 tekrarı bütün eskiRGB deneylerinin yeniden yapılması değildir.',
        'CPU neural görüntü üretildi; entropy stream, native GPU latency veya otonom deployment benchmarkı yapılmadı. Paired bootstrap aralıkları ve bütün ham sonuçlar analysis.json içinde.']
    (OUT/'REPORT_TR.md').write_text('\n'.join(lines)+'\n')
    print(json.dumps({'sequences':53,'policy_cases':318,'summaries':summaries,'out':str(OUT)}))


if __name__=='__main__':main()
