"""Summarize a complete fresh-router numerical audit, without quality claims."""
import hashlib,json,math
from pathlib import Path
REPO=Path(__file__).resolve().parents[1]
SOURCE=Path('/data10/shareddata/can_karsal/dcvcuf_depth_20260927/research/router_logit_replay_qp32')
OUT=REPO/'docs/research/2026-09-27-six-hour/router_logit_replay'


def sha(path):return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def main():
    progress=json.loads((SOURCE/'progress.json').read_text())
    if (progress['state'],progress['completed'])!=('complete',53):raise ValueError('Complete53-sequence cohort required')
    manifest=json.loads((SOURCE/'manifest.json').read_text())
    for path,digest in manifest['source_sha256'].items():
        if sha(path)!=digest:raise ValueError('Frozen audit source changed')
    paths=sorted((SOURCE/'cases').glob('*.json'));rows=[json.loads(p.read_text()) for p in paths]
    if len(rows)!=53 or {r['sequence'] for r in rows}!=set(manifest['names']):raise ValueError('Incomplete sequence set')
    policies=[]
    for criterion in ('mean','q90'):
        selected=[(r,next(c for c in r['comparisons'] if c['criterion']==criterion)) for r in rows]
        changed=[]
        for row,c in selected:
            count=sum(a!=b for a,b in zip(c['fresh_map'],c['archived_map']))
            if len(c['fresh_map'])!=c['tile_count'] or len(c['archived_map'])!=c['tile_count'] or count!=c['changed_tiles']:raise ValueError('Map mismatch accounting fails')
            if count:changed.append({'sequence':row['sequence'],'sequence_index':row['sequence_index'],**c})
        policies.append({'criterion':criterion,'n_sequences':53,'n_tiles':sum(c['tile_count'] for r,c in selected),
            'changed_sequences':len(changed),'changed_tiles':sum(c['changed_tiles'] for r,c in selected),'changed_cases':changed,
            'mean_fresh_minus_archived_saving_points':sum(c['fresh_saving_points']-c['archived_saving_points'] for r,c in selected)/53})
    for row in rows:
        if row['router_conv_linear_macs']!=sum(r['macs'] for r in row['router_layers']):raise ValueError('MAC accounting fails')
        if [r['name'] for r in row['router_layers']]!=['proj_stem','mlp.1','mlp.3','mlp.5']:raise ValueError('Unexpected executed modules')
        _,_,h,w=row['padded_shape'];tiles=(h//256)*(w//256)
        expected=h*w//64*384*48+tiles*(177*256+256*256+256*6)
        if expected!=row['router_conv_linear_macs']:raise ValueError('Executed MACs differ from analytic count')
        row['router_conv_linear_macs_per_padded_pixel']=row['router_conv_linear_macs']/(h*w)
    result={'scope':manifest['scope'],'manifest':manifest,'rows':rows,'policies':policies,
        'maximum_absolute_log_probability_difference':max(r['max_absolute_log_probability_difference'] for r in rows),
        'exact_log_probability_frames':sum(r['exact_log_probabilities'] for r in rows),
        'router_macs_per_padded_pixel':sorted({r['router_conv_linear_macs_per_padded_pixel'] for r in rows}),
        'cost_equation':'384*48*(padded_pixels/64) + tile_count*(177*256+256*256+256*6); tile256square',
        'source_files_sha256':{str(p):sha(p) for p in paths},'analysis_script_sha256':sha(__file__),
        'interpretation':'Map equality only validates these frozen controls on these53frames atQP32. It is not all-beta, all-QP, hardware-independent equality, final-image quality or runtime. If changed_cases is nonempty, archived-map quality must not be labelled fresh-router output without decoding the changed maps.'}
    OUT.mkdir(parents=True,exist_ok=True);(OUT/'analysis.json').write_text(json.dumps(result,indent=2,allow_nan=False)+'\n')
    lines=['# Router checkpoint tekrar üretimi','',
        '53 CTC ilk frame, QP32, frozen e15 + ce_soft_6k. CPU FP32 yeniden üretilen log olasılıkları ile tarihsel GPU arşivi karşılaştırıldı. İki cross-fit kontrol aynı kaldı; source hata etiketleri karara girmedi.',
        '',f"En büyük mutlak log-olasılık farkı: {result['maximum_absolute_log_probability_difference']:.8g}.",
        '', '| Kontrol | Değişen görüntü | Değişen tile | Ortalama tasarruf değişimi (pp) |','|---|---:|---:|---:|']
    for p in policies:lines.append(f"| {p['criterion']} | {p['changed_sequences']}/53 | {p['changed_tiles']}/{p['n_tiles']} | {p['mean_fresh_minus_archived_saving_points']:+.8f} |")
    lines+=['',f"Router Conv2d+Linear maliyeti: {result['router_macs_per_padded_pixel']} MAC/padded pixel. Pooling, LayerNorm, aktivasyon, argmax, bellek/dispatch ve kaynak encode maliyeti buna dahil değil. Geçersiz latent/bit branch'leri hâlâ saklanan parametrelerdir; yürütülen convolution yalnız stem projection'dır.",
        'Bu sayılar duvar saati kazancı değildir. Map eşitliği yalnız ölçülen QP/kontrol/cohort içindir; bütün beta veya platformlarda eşitlik iddia edilmez.']
    (OUT/'REPORT_TR.md').write_text('\n'.join(lines)+'\n');print(json.dumps({k:result[k] for k in ('maximum_absolute_log_probability_difference','exact_log_probability_frames','router_macs_per_padded_pixel','policies')}))


if __name__=='__main__':main()
