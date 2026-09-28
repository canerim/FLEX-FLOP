"""Analyze all fixed-map bank profiles; paired phases are never independent samples."""
import hashlib
import json
import math
from pathlib import Path
import sys
import numpy as np
from analyze_reference_validation_20260927 import describe, quality_at, mathematical_checks

REPO=Path(__file__).resolve().parents[1]
ROOT=Path('/data10/shareddata/can_karsal/dcvcuf_depth_20260927/research')
SOURCE=ROOT/'region_merge_epoch020'
OUT=REPO/'docs/research/2026-09-27-six-hour/region_merge'
sys.path.insert(0,str(REPO/'experiments/dcvcuf_region_merge_20260927'))
from layout import description, regions, unpack
FIELDS=('payload_bpp','container_bpp','estimated_bpp','psnr_rgb','mse_rgb',
        'vertical_band_r4_mse','horizontal_band_r4_mse')


def sha(path):return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def main():
    mathematical_checks()
    state=json.loads((SOURCE/'progress.json').read_text())
    if (state['state'],state['completed'])!=('complete',80):raise ValueError('Complete80-case bank study required')
    manifest=json.loads((SOURCE/'manifest.json').read_text())
    for name,digest in manifest['source_sha256'].items():
        if sha(REPO/'experiments/dcvcuf_region_merge_20260927'/name)!=digest:raise ValueError('Bank source changed')
    if sha(ROOT/'patch_native_shape_epoch020/manifest.json')!=manifest['primary_manifest_sha256']:raise ValueError('Prerequisite changed')
    for name,digest in manifest['derived_source_sha256'].items():
        if sha(ROOT/'native_shape_reference/source'/name)!=digest:raise ValueError('Codec source changed')
    crop_manifest=ROOT/'div2k100_center512_rgb/manifest.json'
    if sha(crop_manifest)!=manifest['crop_manifest_sha256']:raise ValueError('Crop manifest changed')
    crops={r['image']:r for r in json.loads(crop_manifest.read_text())['images']}
    rows=[];groups={};case_hashes={};seen=set()
    for path in sorted((SOURCE/'cases').glob('*.json')):
        case=json.loads(path.read_text());image=case['image'];qp=case['qp']
        if (image,qp) in seen or image not in manifest['images'] or qp not in manifest['qps']:raise ValueError('Unexpected bank case')
        seen.add((image,qp))
        if case['checkpoint_sha256']!=manifest['checkpoints'] or case['crop_sha256']!=crops[image]['crop_file_sha256']:raise ValueError('Case identity changed')
        for source,digest in case['source_cases_sha256'].items():
            if sha(source)!=digest:raise ValueError('Reused component source changed')
        if sorted(v['profile'] for v in case['variants'])!=list(range(10)):raise ValueError('Incomplete profile set')
        for variant in case['variants']:
            p=variant['profile'];pattern,merged,phase=description(p)
            if (variant['pattern'],variant['merged'],variant['phase'])!=(pattern,merged,phase):raise ValueError('Profile semantics differ')
            stream=Path(variant['stream_path']);data=stream.read_bytes();profile,parts=unpack(data)
            if profile!=p or sha(stream)!=variant['stream_sha256'] or not variant['exact_independent_decode']:raise ValueError('Bank stream integrity failed')
            if len(parts)!=len(variant['parts']):raise ValueError('Part count differs')
            for (region,blob),saved in zip(parts,variant['parts']):
                if json.loads(json.dumps(region))!=saved['region'] or hashlib.sha256(blob).hexdigest()!=saved['stream_sha256']:raise ValueError('Embedded component differs')
                if len(blob)!=saved['container_bytes'] or len(blob)!=saved['payload_bytes']+88:raise ValueError('Component byte accounting fails')
            payload=sum(v['payload_bytes'] for v in variant['parts']);embedded=sum(len(blob) for _,blob in parts)
            if (payload,embedded,len(data),12+4*len(parts))!=(variant['payload_bytes'],variant['embedded_bytes'],variant['container_bytes'],variant['bank_header_bytes']):raise ValueError('Bank byte accounting fails')
            for field,count in [('payload_bpp',payload),('container_bpp',len(data))]:
                if abs(variant[field]-count*8/512**2)>1e-12:raise ValueError('Pixel denominator differs')
            if any(not math.isfinite(variant[k]) for k in FIELDS):raise ValueError('Nonfinite metric')
            if abs(variant['psnr_rgb']+10*math.log10(max(variant['mse_rgb'],1e-12)))>1e-10:raise ValueError('PSNR definition differs')
            row={'image':image,'qp':qp,'profile':p,'pattern':pattern,'merged':merged,'phase':phase,
                'n_regions':len(parts),'payload_bytes':payload,'embedded_bytes':embedded,'container_bytes':len(data),
                **{k:variant[k] for k in FIELDS}}
            rows.append(row);groups.setdefault((image,p),[]).append(row)
        case_hashes[str(path)]=sha(path)
    if len(seen)!=80 or len(rows)!=800:raise ValueError('Complete80×10 grid required')
    cost_path=REPO/'docs/research/2026-09-27-six-hour/region_macs/analysis.json'
    cost=json.loads(cost_path.read_text())
    if cost['cuda_initialized'] or not all(r['meta_equals_real_cpu'] for r in cost['real_cpu_checks']):raise ValueError('Geometry cost verification missing')
    for path,digest in cost['source_sha256'].items():
        if sha(path)!=digest:raise ValueError('Cost audit source changed')
    native=[];native_images=[];matched=[]
    for pattern,unmerged,merged in [('vertical',(2,3),(4,5)),('horizontal',(6,7),(8,9))]:
        for qp in manifest['qps']:
            values=[]
            for image in manifest['images']:
                lookup={r['profile']:r for r in rows if (r['image'],r['qp'])==(image,qp)}
                contrasts=[]
                for u,m in zip(unmerged,merged):
                    a,b=lookup[u],lookup[m]
                    contrasts.append({'psnr_gain_db':b['psnr_rgb']-a['psnr_rgb'],
                        'payload_change_percent':100*(b['payload_bpp']/a['payload_bpp']-1),
                        'container_change_percent':100*(b['container_bpp']/a['container_bpp']-1),
                        'vertical_band_mse_change':b['vertical_band_r4_mse']-a['vertical_band_r4_mse'],
                        'horizontal_band_mse_change':b['horizontal_band_r4_mse']-a['horizontal_band_r4_mse']})
                averaged={key:float(np.mean([v[key] for v in contrasts])) for key in contrasts[0]}
                values.append(averaged);native_images.append({'image':image,'qp':qp,'pattern':pattern,**averaged})
            native.append({'pattern':pattern,'qp':qp,'summaries':{key:describe([v[key] for v in values]) for key in values[0]}})
        for field in ('payload_bpp','container_bpp'):
            for method in ('linear','pchip'):
                for rate in (.1,.2,.4):
                    qualities={image:{p:quality_at(groups[image,p],rate,field,method) for p in (*unmerged,*merged)} for image in manifest['images']}
                    common=[image for image,q in qualities.items() if all(v is not None for v in q.values())]
                    gain={image:float(np.mean([qualities[image][m]-qualities[image][u] for u,m in zip(unmerged,merged)])) for image in common}
                    matched.append({'pattern':pattern,'rate_field':field,'interpolator':method,'target_bpp':rate,
                        'n':len(common),'images':common,'phase_averaged_gain_db':describe(list(gain.values())),
                        'per_image_gain_db':gain,'quality_per_image':qualities})
    result={'scope':manifest['scope'],'n_cases':80,'n_profiles':800,'n_images':16,'manifest':manifest,'rows':rows,
        'same_qp':native,'same_qp_per_image':native_images,'matched_rate':matched,'source_files_sha256':case_hashes,
        'analysis_script_sha256':sha(__file__),'cost_audit_sha256':sha(cost_path),'cost_audit':cost,
        'statistics':'Both complementary phases averaged within image before5000 paired-image bootstrap draws, seed20260927. Rate support intersects both phases of both implementations within each pattern; no extrapolation. Not training-seed uncertainty.',
        'interpretation':'Identical per-pixel D2/D6 assignments within each merged/unmerged contrast. Region geometry, context and entropy resets change together. Fixed profiles are an engineering control, not a learned router, routing gain or latency result.'}
    OUT.mkdir(parents=True,exist_ok=True);(OUT/'analysis.json').write_text(json.dumps(result,indent=2,allow_nan=False)+'\n')
    lines=['# Aynı derinlik haritasında bölge birleştirme','',
        '16 görüntü ×5 QP ×10 sabit profil. D2/D6 epoch20, yüzde50/yüzde50 piksel dağılımı. Aynı haritanın dört parçalı ve iki birleşik bölgeli uygulamaları; CPU FP32 FUFBNK1 araştırma bitstream\'i.',
        '', '| Harita | Payload bpp | Ortak görüntü | Birleştirme RGB PSNR farkı (dB) |','|---|---:|---:|---:|']
    for row in matched:
        if (row['rate_field'],row['interpolator'])!=('payload_bpp','linear'):continue
        s=row['phase_averaged_gain_db'];value='—' if not s['n'] else f"{s['mean']:+.5f}"
        lines.append(f"| {row['pattern']} | {row['target_bpp']:.1f} | {row['n']}/16 | {value} |")
    lines += ['', 'Tamamlayıcı iki faz görüntü içinde ortalanır;32 bağımsız örnek gibi sayılmaz. Header dahil oranlar ve PCHIP duyarlılığı analysis.json içinde. Checkerboard profilleri korunur, fakat birleşebilir komşuları olmadığı için merged kontrastı yoktur.',
        'Bu, adaptif seçimin başarısını veya runtime kazancını ölçmez. Context ve entropy reset sayısı birlikte değişir. Connected-region uygulaması önceki çalışmalarda vardır; tek başına yenilik iddiası değildir.']
    (OUT/'REPORT_TR.md').write_text('\n'.join(lines)+'\n');print(json.dumps({'cases':80,'profiles':800,'out':str(OUT)}))


if __name__=='__main__':main()
