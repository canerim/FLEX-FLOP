"""Analyze the complete fixed-depth, independently decoded patch control."""
from __future__ import annotations
import argparse
import json
import math
from pathlib import Path
import numpy as np
from analyze_reference_validation_20260927 import sha,describe,quality_at,bd_rate,mathematical_checks

ROOT=Path('/data10/shareddata/can_karsal/dcvcuf_depth_20260927/research')
REPO=Path(__file__).resolve().parents[1]
IMAGES=[f'{i:04d}.png' for i in np.rint(np.linspace(801,900,16)).astype(int)]
DEPTHS=(2,6,12);QPS=(0,16,32,48,63);VARIANTS=('full','halo0','halo32','halo64')
FIELDS=('payload_bpp','container_bpp','estimated_bpp','psnr_rgb','mse_rgb',
        'seam_r4_mse','interior_r4_mse','seam_r16_mse','interior_r16_mse')


def analyze(folder):
    mathematical_checks()
    state=json.loads((folder/'progress.json').read_text())
    if state['state']!='complete' or state['completed']!=240:raise ValueError('Complete 240-case patch control required')
    manifest=json.loads((folder/'manifest.json').read_text())
    if manifest['images']!=IMAGES or manifest['depths']!=list(DEPTHS) or manifest['qps']!=list(QPS):
        raise ValueError('Unexpected experimental grid')
    full=ROOT/'div2k100_reference_epoch020'
    if sha(full/'manifest.json')!=manifest['reference_manifest_sha256']:raise ValueError('Reference manifest changed')
    full_manifest=json.loads((full/'manifest.json').read_text())
    script=REPO/'experiments/dcvcuf_patch_control_20260927/evaluate_patching.py'
    if sha(script)!=manifest['script_sha256']:raise ValueError('Experiment code changed')
    sources={};rows=[];groups={};cases=[]
    for depth in DEPTHS:
        files=sorted((folder/f'd{depth}/cases').glob('*.json'))
        if len(files)!=80:raise ValueError('Expected 80 cases per depth')
        seen=set()
        for path in files:
            case=json.loads(path.read_text());image=case['image'];qp=case['qp']
            if image not in IMAGES or qp not in QPS or (image,qp) in seen:raise ValueError('Case grid invalid')
            seen.add((image,qp));sources[str(path)]=sha(path)
            if case['depth']!=depth or case['checkpoint_sha256']!=full_manifest['checkpoints'][str(depth)]:
                raise ValueError('Checkpoint mismatch')
            original=full/f'd{depth}/cases/{path.name}'
            if sha(original)!=case['reference_case_sha256']:raise ValueError('Reference case changed')
            variants={'full':case['full']}
            for value in case['patch_variants']:
                name=f'halo{value["halo"]}'
                if name not in VARIANTS or name in variants:raise ValueError('Variant mismatch')
                tiles=value['tiles']
                if len(tiles)!=4 or {(t['row'],t['col']) for t in tiles}!={(0,0),(0,1),(1,0),(1,1)}:
                    raise ValueError('Incomplete patch grid')
                for tile in tiles:
                    stream=Path(tile['stream_path'])
                    if not tile['exact_isolated_decode'] or sha(stream)!=tile['stream_sha256']:
                        raise ValueError('Patch decode or stream integrity failed')
                    if stream.stat().st_size!=tile['container_bytes'] or tile['container_bytes']!=tile['payload_bytes']+88:
                        raise ValueError('Patch byte accounting failed')
                    expected=256+value['halo']
                    if tile['bottom']-tile['top']!=expected or tile['right']-tile['left']!=expected:
                        raise ValueError('Unexpected context window')
                if value['payload_bytes']!=sum(t['payload_bytes'] for t in tiles):raise ValueError('Payload sum mismatch')
                if value['container_bytes']!=sum(t['container_bytes'] for t in tiles):raise ValueError('Container sum mismatch')
                expected_pixels=512**2 if value['halo']==0 else 4*320**2
                if value['padded_coded_pixels']!=expected_pixels:raise ValueError('Padding area mismatch')
                variants[name]=value
            if set(variants)!=set(VARIANTS):raise ValueError('Missing variant')
            for name,value in variants.items():
                if any(not math.isfinite(value[k]) for k in FIELDS):raise ValueError('Nonfinite metric')
                if abs(value['payload_bpp']-value['payload_bytes']*8/512**2)>1e-12:raise ValueError('Rate denominator error')
                for radius in (4,16):
                    band=2*radius;pixels=2*512*band-band**2
                    if value[f'seam_r{radius}_pixels']!=pixels:raise ValueError('Seam mask area error')
                    reconstructed=(value[f'seam_r{radius}_mse']*pixels+value[f'interior_r{radius}_mse']*(512**2-pixels))/512**2
                    if abs(reconstructed-value['mse_rgb'])>max(1e-10,abs(value['mse_rgb'])*1e-6):
                        raise ValueError('Seam/interior MSE does not reconstruct full MSE')
                row={'depth':depth,'image':image,'qp':qp,'variant':name,**{k:value[k] for k in FIELDS}}
                rows.append(row);groups.setdefault((depth,image,name),[]).append(row)
            cases.append(case)
        if seen!={(i,q) for i in IMAGES for q in QPS}:raise ValueError('Incomplete image/QP grid')
    same_qp=[]
    for depth in DEPTHS:
        for qp in QPS:
            selected=[c for c in cases if (c['depth'],c['qp'])==(depth,qp)]
            for halo in (0,32,64):
                records=[]
                for c in selected:
                    v=next(v for v in c['patch_variants'] if v['halo']==halo);f=c['full']
                    r={'image':c['image'],'delta_psnr_db':v['psnr_rgb']-f['psnr_rgb'],
                        'delta_payload_bpp':v['payload_bpp']-f['payload_bpp'],
                        'payload_increase_percent':100*(v['payload_bpp']/f['payload_bpp']-1)}
                    for radius in (4,16):
                        r[f'seam_r{radius}_excess_mse']=v[f'seam_r{radius}_mse']-f[f'seam_r{radius}_mse']
                        r[f'interior_r{radius}_excess_mse']=v[f'interior_r{radius}_mse']-f[f'interior_r{radius}_mse']
                        r[f'boundary_specific_r{radius}_excess_mse']=r[f'seam_r{radius}_excess_mse']-r[f'interior_r{radius}_excess_mse']
                    records.append(r)
                same_qp.append({'depth':depth,'qp':qp,'halo':halo,'per_image':records,
                    'summaries':{key:describe([r[key] for r in records]) for key in records[0] if key!='image'}})
    matched=[];bd=[]
    for depth in DEPTHS:
        for field in ('payload_bpp','container_bpp'):
            for method in ('linear','pchip'):
                for rate in (.1,.2,.4):
                    values={i:{v:quality_at(groups[depth,i,v],rate,field,method) for v in VARIANTS} for i in IMAGES}
                    common=[i for i,v in values.items() if all(y is not None for y in v.values())]
                    comparisons=[]
                    for variant in ('halo0','halo32','halo64'):
                        delta={i:values[i][variant]-values[i]['full'] for i in common}
                        comparisons.append({'variant':variant,'delta_psnr_db':describe(list(delta.values())),'per_image_delta':delta})
                    recovery={i:values[i]['halo32']-values[i]['halo0'] for i in common}
                    extra_context={i:values[i]['halo64']-values[i]['halo32'] for i in common}
                    matched.append({'depth':depth,'rate_field':field,'interpolator':method,'target_bpp':rate,
                        'images':common,'n':len(common),'comparisons':comparisons,'quality_per_image':values,
                        'halo32_minus_halo0_db':describe(list(recovery.values())),
                        'halo64_minus_halo32_db':describe(list(extra_context.values()))})
            for variant in ('halo0','halo32','halo64'):
                values={}
                for image in IMAGES:
                    q=[[r['psnr_rgb'] for r in groups[depth,image,v]] for v in VARIANTS]
                    interval=(max(min(y) for y in q),min(max(y) for y in q))
                    values[image]=bd_rate(groups[depth,image,'full'],groups[depth,image,variant],field,interval)
                valid=[v['percent'] for v in values.values() if v['percent'] is not None]
                bd.append({'depth':depth,'rate_field':field,'variant':variant,'summary':describe(valid),'per_image':values})
    return {'scope':'Interim fixed-depth patching control, not adaptive routing or a GPU benchmark',
        'padding_scope':'FUFREF1 CPU image-pad64. Native image-pad16 plus latent-pad4 is different; halo32/64 equal-area result is specific to this reference protocol.',
        'cases':240,'images':IMAGES,'manifest':manifest,'source_files_sha256':sources,'rows':rows,
        'same_qp':same_qp,'matched_rate':matched,'bd_rate':bd,'analysis_script_sha256':sha(__file__),
        'statistics':'5000 paired-image bootstrap draws, seed20260927; intervals conditional on16 selected images and frozen checkpoints, not training seeds. Matched-rate comparisons use common support across all4 full/patch variants within each depth; no extrapolation.',
        'seam_scope':'Same-QP MSE excess relative to the same model full512 image. Boundary-specific excess subtracts interior excess; this is a local distortion diagnostic, not a matched-rate causal seam estimate.',
        'compute_geometry':{'full_padded_pixels':262144,'halo0_padded_pixels':262144,'halo32_padded_pixels':409600,
            'halo32_area_multiplier':1.5625,'halo64_padded_pixels':409600,'halo64_area_multiplier':1.5625,'note':'Coded spatial area only, not a runtime or MAC measurement'},
        'header_note':'Full frame uses88 research-header bytes; four patches use352. The extra264bytes contribute0.008056640625bpp over512². No adaptive expert-map bits in this fixed-depth control.'}


def report(data,out):
    lines=['# Sabit derinlikte patchleme: kalite, gerçek byte ve sınır hatası','',
        'Önceden belirlenen16 DIV2K merkez512 crop; D2/D6 epoch20, released D12; beş QP. CPU FP32 araştırma formatı. Router deneyi veya GPU hız ölçümü değil.',
        '', '| Model | Payload bpp | Ortak görüntü | Halo0−full dB | Halo32−full dB | Halo64−full dB |', '|---|---:|---:|---:|---:|---:|']
    for r in data['matched_rate']:
        if (r['rate_field'],r['interpolator'])!=('payload_bpp','linear'):continue
        vals=[]
        for p in r['comparisons']:
            s=p['delta_psnr_db'];vals.append('—' if not s['n'] else f"{s['mean']:+.4f}")
        lines.append(f"| D{r['depth']} | {r['target_bpp']:.1f} | {r['n']}/16 | "+' | '.join(vals)+' |')
    lines+=['', 'Ortak destek her modelde full/halo0/halo32/halo64 eğrilerinin kesişimidir. Derinlikler arasında kapsanan görüntüler farklı olabilir; bu tablo tek başına derinlik sıralaması için kullanılmaz.',
        'Paired görüntü bootstrap aralıkları ve lineer–PCHIP duyarlılığı analysis.json içindedir. Veriler küçük, önceden belirlenmiş bir ara kontroldür.',
        '', '32-pixel halo ile dört288×288 pencere, codec padding sonrası dört320×320 alan kodlar. Toplam kodlanan alan full512\'nin1,5625 katıdır. Bu geometrik oran hız oranı değildir.',
        '64-pixel halo doğrudan320×320 pencere kullanır: 32-halo ile aynı kodlanan alan, daha fazla gerçek context. Bu varyant hiçbir patch sonucu görülmeden eklendi.',
        'Dört bağımsız stream toplam352 research-header byte taşır; full-frame88 byte. Extra264 byte =0,00805664bpp. Payload ve container ayrı raporlanır.',
        'Seam ölçümleri aynı QP\'de full-frame hatası çıkarılarak yapılır; kalite ve bitrate birlikte değişebilir. Aynı-bitrate global PSNR karşılaştırması yukarıdadır.']
    (out/'REPORT_TR.md').write_text('\n'.join(lines)+'\n')


if __name__=='__main__':
    ap=argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--input',type=Path,default=ROOT/'patch_control_epoch020')
    ap.add_argument('--out',type=Path,default=REPO/'docs/research/2026-09-27-six-hour/patch_control_epoch020')
    args=ap.parse_args();data=analyze(args.input);args.out.mkdir(parents=True,exist_ok=True)
    (args.out/'analysis.json').write_text(json.dumps(data,indent=2,allow_nan=False)+'\n');report(data,args.out)
    print(json.dumps({'cases':data['cases'],'images':len(data['images']),'out':str(args.out)}))
