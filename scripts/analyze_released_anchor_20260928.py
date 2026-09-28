"""Common-support released/e15 reference comparison from the complete53 frames."""
import hashlib,json,math
from pathlib import Path
from analyze_reference_validation_20260927 import describe
REPO=Path(__file__).resolve().parents[1]
SOURCE=Path('/data10/shareddata/can_karsal/dcvcuf_depth_20260927/research/released_anchor_qp32')
PRIOR=SOURCE.parent/'shared_crossfit_qp32'
OUT=REPO/'docs/research/2026-09-27-six-hour/released_anchor_qp32'


def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()


def main():
    state=json.loads((SOURCE/'progress.json').read_text())
    if (state['state'],state['completed'])!=('complete',53):raise ValueError('Complete53-frame cohort required')
    manifest=json.loads((SOURCE/'manifest.json').read_text())
    for path,digest in manifest['source_sha256'].items():
        if sha(path)!=digest:raise ValueError('Frozen source changed')
    paths=sorted((SOURCE/'cases').glob('*.json'));cases=[json.loads(p.read_text()) for p in paths]
    if len(cases)!=53 or {c['sequence'] for c in cases}!=set(manifest['names']):raise ValueError('Incomplete cohort')
    if sum(c['stock_warmstart_canary_exact'] is True for c in cases)!=1:raise ValueError('Exact canary missing')
    rows=[]
    for case,path in zip(cases,paths):
        original=json.loads((PRIOR/'cases'/path.name).read_text())
        if case['frame_sha256']!=original['first_frame_bytes_sha256']:raise ValueError('Source frame mismatch')
        if abs(case['released_cropped_rgb_psnr']-case['e15_cropped_rgb_psnr']-case['e15_full_rgb_loss_vs_released_db'])>1e-10:raise ValueError('Anchor PSNR identity fails')
        for row in case['policy_rows']:
            before=next(r for r in original['rows'] if (r['criterion'],r['policy'])==(row['criterion'],row['policy']))
            if row['saving_points']!=before['saving_points']:raise ValueError('Costs changed with reference')
            for short,prior_key in [('rgb','cropped_rgb_loss_db'),('444','cropped_ycbcr444_loss_db'),('611','cropped_yuv611_loss_db')]:
                expected=before[prior_key]+case[f'e15_full_{short}_loss_vs_released_db']
                if abs(row[f'cropped_{short}_loss_vs_released_db']-expected)>1e-10:raise ValueError('Reference-shift identity fails')
            rows.append({'sequence':case['sequence'],**row})
    summaries=[]
    for criterion in ('mean','q90'):
        for policy in ('router','dither','uniform'):
            chosen=[r for r in rows if (r['criterion'],r['policy'])==(criterion,policy)]
            if len(chosen)!=53:raise ValueError('Missing policy case')
            keys=('saving_points','cropped_rgb_loss_vs_released_db','cropped_444_loss_vs_released_db','cropped_611_loss_vs_released_db')
            summaries.append({'criterion':criterion,'policy':policy,'n':53,'metrics':{k:describe([r[k] for r in chosen]) for k in keys},
                'rgb_above_nominal_point1':sum(r['cropped_rgb_loss_vs_released_db']>.1+1e-4 for r in chosen)})
    metrics=('released_cropped_rgb_psnr','e15_cropped_rgb_psnr','e15_full_rgb_loss_vs_released_db','e15_full_444_loss_vs_released_db','e15_full_611_loss_vs_released_db','released_padded_vs_archived_R_db')
    result={'scope':manifest['scope'],'manifest':manifest,'n_sequences':53,'n_policy_cases':318,'cases':cases,'policy_rows':rows,'policy_summaries':summaries,
        'anchor_summaries':{k:describe([r[k] for r in cases]) for k in metrics},
        'maximum_absolute_padded_R_discrepancy_db':max(abs(r['released_padded_vs_archived_R_db']) for r in cases),
        'source_files_sha256':{str(p):sha(p) for p in paths},'analysis_script_sha256':sha(__file__),
        'statistics':'Equal sequence weights,5000 paired-sequence bootstrap draws, seed20260927. Conditional on fixed checkpoints and QP32 development corpus; not seed uncertainty or external-test performance.',
        'interpretation':'Positive loss means lower PSNR than the official released synthesis under this shared replay padding/support/precision contract. Re-expressing the same outputs changes absolute-reference loss and threshold counts, but cannot change paired router-minus-dither loss or MAC differences. This is not a native deployed released bitstream/latency test.'}
    OUT.mkdir(parents=True,exist_ok=True);(OUT/'analysis.json').write_text(json.dumps(result,indent=2,allow_nan=False)+'\n')
    a=result['anchor_summaries'];lines=['# Aynı support üzerinde released D12 referansı','',
        '53 CTC ilk frame, QP32. Kaynak önce shared deneydeki gibi256 katına replicate-pad edilir, metrik geçerli kaynak alanına crop edilir. Stock DMCI released ağırlıkları, CPU FP32; native CUDA/bitstream çıktısı değil.',
        '',f"Released RGB PSNR ortalaması: {a['released_cropped_rgb_psnr']['mean']:.5f}dB; e15 full-frame: {a['e15_cropped_rgb_psnr']['mean']:.5f}dB.",
        f"e15 full-frame'in released'e göre RGB kaybı: {a['e15_full_rgb_loss_vs_released_db']['mean']:+.5f}dB; CI {a['e15_full_rgb_loss_vs_released_db']['ci95']}.",
        '', '| Kontrol | Kural | Released RGB kaybı (dB) | 0.1 üzeri /53 |','|---|---|---:|---:|']
    for s in summaries:lines.append(f"| {s['criterion']} | {s['policy']} | {s['metrics']['cropped_rgb_loss_vs_released_db']['mean']:+.5f} | {s['rgb_above_nominal_point1']} |")
    lines+=['','Referans değişimi aynı çıktıların raporunu değiştirir; router ve dither arasındaki paired farkı değiştirmez. Nominal bütçe padded444 ve farklı kalibrasyon hedefiyle seçildiği için bu sayılar ayrı bir RGB garanti testi değildir. Her görüntünün ağırlığı aynı; en kolay görüntüler seçilmedi.']
    (OUT/'REPORT_TR.md').write_text('\n'.join(lines)+'\n');print(json.dumps({'anchors':result['anchor_summaries'],'policies':summaries}))


if __name__=='__main__':main()
