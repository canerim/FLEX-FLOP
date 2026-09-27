"""Paired CPU padding-policy analysis on the complete predeclared patch cohort."""
import hashlib
import json
import math
from pathlib import Path
import numpy as np
from analyze_reference_validation_20260927 import describe,quality_at,bd_rate

REPO=Path(__file__).resolve().parents[1]
ROOT=Path('/data10/shareddata/can_karsal/dcvcuf_depth_20260927/research')
SOURCE=ROOT/'patch_native_shape_epoch020'
PRIMARY=ROOT/'patch_control_epoch020'
OUT=REPO/'docs/research/2026-09-27-six-hour/native_padding'
VARIANTS=('full','halo0','halo32_pad64','halo64','halo32_native_shape')
FIELDS=('payload_bpp','container_bpp','estimated_bpp','psnr_rgb','mse_rgb',
        'seam_r4_mse','interior_r4_mse','seam_r16_mse','interior_r16_mse')


def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()


def main():
    state=json.loads((SOURCE/'progress.json').read_text())
    if (state['state'],state['completed'])!=('complete',240):raise ValueError('Full240-case native-shaped study required')
    manifest=json.loads((SOURCE/'manifest.json').read_text())
    if sha(PRIMARY/'manifest.json')!=manifest['legacy_manifest_sha256']:raise ValueError('Primary protocol changed')
    script=REPO/'experiments/dcvcuf_native_shape_reference_20260927/evaluate_padding.py'
    if sha(script)!=manifest['script_sha256']:raise ValueError('Native-shaped evaluator changed')
    for name,digest in manifest['source_sha256'].items():
        if sha(ROOT/'native_shape_reference/source'/name)!=digest:raise ValueError('Native-shaped codec source changed')
    rows=[];cases=[];hashes={};groups={}
    for depth in manifest['depths']:
        paths=sorted((SOURCE/f'd{depth}/cases').glob('*.json'))
        if len(paths)!=80:raise ValueError('Expected80 cases per model')
        seen=set()
        for path in paths:
            case=json.loads(path.read_text());image=case['image'];qp=case['qp']
            if (image,qp) in seen or image not in manifest['images'] or qp not in manifest['qps']:raise ValueError('Unexpected case')
            seen.add((image,qp));primary_path=PRIMARY/f'd{depth}/cases/{path.name}'
            if sha(primary_path)!=case['primary_case_sha256']:raise ValueError('Primary case changed')
            primary=json.loads(primary_path.read_text())
            if primary['checkpoint_sha256']!=case['checkpoint_sha256'] or primary['crop_sha256']!=case['crop_sha256']:raise ValueError('Case provenance differs')
            full_path=ROOT/f'div2k100_reference_epoch020/d{depth}/streams/{path.stem}.fufref'
            full2=SOURCE/f'd{depth}/streams/{path.stem}_full_rewrapped.fufref2'
            if sha(full_path)!=case['full_original_stream_sha256'] or sha(full2)!=case['full_rewrapped_stream_sha256']:raise ValueError('Full stream changed')
            if full_path.read_bytes()[8:]!=full2.read_bytes()[8:] or not case['full_psnr_matches_primary']:raise ValueError('Aligned reuse failed')
            native=case['native_shape_halo32'];tiles=native['tiles']
            if len(tiles)!=4 or {(t['row'],t['col']) for t in tiles}!={(0,0),(0,1),(1,0),(1,1)}:raise ValueError('Incomplete grid')
            for t in tiles:
                p=Path(t['stream_path'])
                if not t['exact_isolated_decode'] or sha(p)!=t['stream_sha256']:raise ValueError('Native-shaped stream integrity failed')
                if (t['height'],t['width'])!=(288,288) or t['container_bytes']!=t['payload_bytes']+88 or p.stat().st_size!=t['container_bytes']:raise ValueError('Shape/byte mismatch')
            if native['payload_bytes']!=sum(t['payload_bytes'] for t in tiles) or native['container_bytes']!=sum(t['container_bytes'] for t in tiles):raise ValueError('Payload sum differs')
            variants={'full':primary['full'],'halo32_native_shape':native}
            for v in primary['patch_variants']:variants['halo32_pad64' if v['halo']==32 else f"halo{v['halo']}"]=v
            if set(variants)!=set(VARIANTS):raise ValueError('Variant grid differs')
            for name,v in variants.items():
                if any(not math.isfinite(v[k]) for k in FIELDS):raise ValueError('Nonfinite metric')
                if abs(v['payload_bpp']-v['payload_bytes']*8/512**2)>1e-12:raise ValueError('Rate denominator differs')
                for radius in (4,16):
                    pixels=2*512*(2*radius)-(2*radius)**2
                    combined=(v[f'seam_r{radius}_mse']*pixels+v[f'interior_r{radius}_mse']*(512**2-pixels))/512**2
                    if abs(combined-v['mse_rgb'])>max(1e-10,v['mse_rgb']*1e-6):raise ValueError('MSE decomposition fails')
                r={'depth':depth,'image':image,'qp':qp,'variant':name,**{k:v[k] for k in FIELDS}}
                rows.append(r);groups.setdefault((depth,image,name),[]).append(r)
            old=variants['halo32_pad64'];full=variants['full']
            cases.append({'depth':depth,'image':image,'qp':qp,
                'native_minus_pad64_psnr_db':native['psnr_rgb']-old['psnr_rgb'],
                'native_minus_pad64_payload_bpp':native['payload_bpp']-old['payload_bpp'],
                'native_payload_change_percent':100*(native['payload_bpp']/old['payload_bpp']-1),
                'native_minus_full_psnr_db':native['psnr_rgb']-full['psnr_rgb'],
                'native_minus_pad64_seam_r4_mse':native['seam_r4_mse']-old['seam_r4_mse']})
            hashes[str(path)]=sha(path);hashes[str(primary_path)]=sha(primary_path)
    same=[];matched=[];bd=[]
    for depth in manifest['depths']:
        for qp in manifest['qps']:
            subset=[r for r in cases if (r['depth'],r['qp'])==(depth,qp)]
            same.append({'depth':depth,'qp':qp,'summaries':{k:describe([r[k] for r in subset]) for k in subset[0] if k not in ('depth','image','qp')}})
        for field in ('payload_bpp','container_bpp'):
            for method in ('linear','pchip'):
                for rate in (.1,.2,.4):
                    values={i:{v:quality_at(groups[depth,i,v],rate,field,method) for v in VARIANTS} for i in manifest['images']}
                    common=[i for i,v in values.items() if all(q is not None for q in v.values())]
                    delta={i:values[i]['halo32_native_shape']-values[i]['halo32_pad64'] for i in common}
                    matched.append({'depth':depth,'rate_field':field,'interpolator':method,'target_bpp':rate,'n':len(common),'images':common,
                        'quality_per_image':values,'native_minus_pad64_psnr_db':describe(list(delta.values())),
                        'per_image_native_minus_pad64_db':delta,
                        'versus_full':{v:describe([values[i][v]-values[i]['full'] for i in common]) for v in VARIANTS if v!='full'}})
            for reference in ('full','halo32_pad64'):
                values={}
                for image in manifest['images']:
                    qualities=[[r['psnr_rgb'] for r in groups[depth,image,v]] for v in VARIANTS]
                    interval=(max(min(q) for q in qualities),min(max(q) for q in qualities))
                    values[image]=bd_rate(groups[depth,image,reference],groups[depth,image,'halo32_native_shape'],field,interval)
                bd.append({'depth':depth,'reference_variant':reference,'rate_field':field,
                    'summary':describe([v['percent'] for v in values.values() if v['percent'] is not None]),'per_image':values})
    result={'scope':'Paired CPU FP32 padding-policy ablation. Native-shaped denotes image/latent geometry only; not native CUDA output, bitstream or speed.',
        'n_cases':240,'n_images':16,'manifest':manifest,'rows':rows,'same_qp':same,'same_qp_per_image':cases,
        'matched_rate':matched,'bd_rate':bd,'source_files_sha256':hashes,'analysis_script_sha256':sha(__file__),
        'statistics':'5000 paired-image bootstrap draws; seed20260927; common support across all5 variants within each depth and rate; no extrapolation. Conditional on16 images and interim checkpoints, not seed variation.',
        'interpretation':'Changing the padding location changes both source-analysis support and hyperprior context; the experiment does not attribute the outcome to only one mechanism. Full/core/halo64 aligned cases retain their primary records.'}
    OUT.mkdir(parents=True,exist_ok=True);(OUT/'analysis.json').write_text(json.dumps(result,indent=2,allow_nan=False)+'\n')
    lines=['# Padding nerede uygulanıyor? CPU politika karşılaştırması','',
        'Aynı16 görüntü,5 QP,D2/D6 epoch20 ve releasedD12. Yalnız halo32/context288 tekrar kodlandı. FUFREF1 görüntüyü320\'ye pad eder; FUFREF2 görüntüyü288 tutup y18\'i hyperanalysis için20\'ye pad eder. Native GPU eşdeğerliği iddiası yok.',
        '', '| Model | Payload bpp | Ortak n | Native-shaped − image-pad64 RGB PSNR (dB) |','|---|---:|---:|---:|']
    for r in matched:
        if (r['rate_field'],r['interpolator'])!=('payload_bpp','linear'):continue
        s=r['native_minus_pad64_psnr_db'];value='—' if not s['n'] else f"{s['mean']:+.5f}"
        lines.append(f"| D{r['depth']} | {r['target_bpp']:.1f} | {r['n']}/16 | {value} |")
    lines += ['', 'Bütün beş varyantın ortak desteği kullanılır: full,halo0,halo32-pad64,halo64,halo32-native-shape. Farklırate/derinliklerde cohort değişebilir. Header ve payload ayrı; same-QP değişimler matched-rate tablosunun yerine geçmez.',
        'Bu karşılaştırma padding yerini değiştirir; encoder sınır desteği ve hyperprior context\'i birlikte etkilenir. Tek bir mekanizmanın nedensel katkısı ayrılamaz.']
    (OUT/'REPORT_TR.md').write_text('\n'.join(lines)+'\n');print(json.dumps({'cases':240,'out':str(OUT)}))


if __name__=='__main__':main()
