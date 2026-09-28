"""Complete-cohort paired analysis of frozen adapter/repair interventions."""
import hashlib,json,math
from pathlib import Path
from analyze_reference_validation_20260927 import describe
REPO=Path(__file__).resolve().parents[1]
SOURCE=Path('/data10/shareddata/can_karsal/dcvcuf_depth_20260927/research/component_interventions_qp32')
OUT=REPO/'docs/research/2026-09-27-six-hour/component_interventions'
VARIANTS=('repair_identity','adapters_identity','both_identity')


def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()


def main():
    state=json.loads((SOURCE/'progress.json').read_text())
    if (state['state'],state['completed'])!=('complete',53):raise ValueError('All53 sequences required')
    manifest=json.loads((SOURCE/'manifest.json').read_text())
    for path,digest in manifest['source_sha256'].items():
        if sha(path)!=digest:raise ValueError('Source provenance changed')
    paths=sorted((SOURCE/'cases').glob('*.json'));cases=[json.loads(p.read_text()) for p in paths]
    if len(cases)!=53 or {c['sequence'] for c in cases}!=set(manifest['names']):raise ValueError('Cohort incomplete')
    contrasts=[];interactions=[]
    for case in cases:
        if not case['baseline_matches_complete_replay'] or case['head_replay_exact']!=[True,True]:raise ValueError('Exact path check failed')
        lookup={r['variant']:r for r in case['rows']}
        if set(lookup)!=set(manifest['variants']):raise ValueError('Incomplete factorial')
        n=case['height']*case['width'];nb=case['boundary_pixels'];ni=case['interior_pixels']
        if nb+ni!=n or min(nb,ni)<=0:raise ValueError('Support partition differs')
        base=lookup['trained']
        for row in lookup.values():
            if abs(row['psnr_rgb']+10*math.log10(row['mse_rgb']))>1e-10:raise ValueError('PSNR identity fails')
            weighted=(row['boundary_mse_rgb']*nb+row['interior_mse_rgb']*ni)/n
            if abs(weighted-row['mse_rgb'])>8*2**-23*row['mse_rgb']+1e-12:raise ValueError('Boundary partition accounting fails')
        for variant in VARIANTS:
            row=lookup[variant]
            contrast={'sequence':case['sequence'],'variant':variant,'psnr_loss_db':base['psnr_rgb']-row['psnr_rgb'],
                'global_mse_increase_percent':100*(row['mse_rgb']/base['mse_rgb']-1),
                'boundary_mse_increase_percent':100*(row['boundary_mse_rgb']/base['boundary_mse_rgb']-1),
                'interior_mse_increase_percent':100*(row['interior_mse_rgb']/base['interior_mse_rgb']-1),
                'boundary_specific_excess_mse':(row['boundary_mse_rgb']-base['boundary_mse_rgb'])-(row['interior_mse_rgb']-base['interior_mse_rgb'])}
            if abs(contrast['psnr_loss_db']-row['loss_vs_original_db'])>1e-7:raise ValueError('Baseline loss differs')
            contrasts.append(contrast)
        loss={v:base['psnr_rgb']-lookup[v]['psnr_rgb'] for v in VARIANTS}
        interactions.append({'sequence':case['sequence'],'joint_minus_sum_loss_db':loss['both_identity']-loss['repair_identity']-loss['adapters_identity']})
    keys=[k for k in contrasts[0] if k not in ('sequence','variant')]
    summary=[{'variant':variant,'metrics':{k:describe([r[k] for r in contrasts if r['variant']==variant]) for k in keys}} for variant in VARIANTS]
    result={'scope':manifest['scope'],'manifest':manifest,'n_sequences':53,'n_outputs':212,'cases':cases,'contrasts':contrasts,'summaries':summary,
        'interaction':{'definition':'Joint identity-substitution PSNR loss minus the sum of the two individual losses, in dB; interaction is metric-scale dependent.',
            'per_sequence':interactions,'summary':describe([r['joint_minus_sum_loss_db'] for r in interactions])},
        'statistics':'5000 paired-sequence bootstrap draws, seed20260927; equal sequence weight. Intervals conditional on fixed checkpoint, QP, archived maps and development cohort; no training-seed uncertainty.',
        'source_files_sha256':{str(p):sha(p) for p in paths},'analysis_script_sha256':sha(__file__),
        'interpretation':'A removed module leaves the remaining trained parameters fixed. This measures checkpoint sensitivity, not the best attainable quality after retraining without that component. Boundary/interior changes need not imply an effect exclusive to seams. No runtime or bitstream claim.'}
    OUT.mkdir(exist_ok=True,parents=True);(OUT/'analysis.json').write_text(json.dumps(result,indent=2,allow_nan=False)+'\n')
    lines=['# Sabit checkpoint üzerinde adapter ve repair müdahalesi','',
        '53 CTC ilk frame, QP32, aynı Q90 router haritaları. 2×2 identity-substitution kontrolü; yeniden eğitim yapılmadı. Orijinal çıktı ve cache edilen repair/head yolu yeniden doğrulandı.',
        '', '| Kapatılan | RGB PSNR kaybı (dB) | Sınır MSE değişimi (%) | Interior MSE değişimi (%) |','|---|---:|---:|---:|']
    for s in summary:
        m=s['metrics'];lines.append(f"| {s['variant']} | {m['psnr_loss_db']['mean']:+.5f} | {m['boundary_mse_increase_percent']['mean']:+.3f} | {m['interior_mse_increase_percent']['mean']:+.3f} |")
    lines+=['','Pozitif PSNR kaybı, identity müdahalesinin mevcut checkpoint kalitesini düşürdüğünü belirtir. Bir bileşen olmadan yeniden optimize edilmiş ağın kalitesini göstermez. Aynı harita korunur; çıkarılan modüllerin MAC maliyeti burada hızlanma olarak sunulmaz.']
    (OUT/'REPORT_TR.md').write_text('\n'.join(lines)+'\n');print(json.dumps({'summaries':summary,'interaction':result['interaction']['summary']}))


if __name__=='__main__':main()
