"""Frozen active-neighbour coupling replay on the disjoint DIV2K 24-image split.

The reference streams, checkpoint, calibrated beta and maps are inherited
from the preceding exact-context study. Only the suffix depthwise boundary
reads change. This is untimed CPU FP32 evaluation, not a speed measurement.
"""
from __future__ import annotations

import argparse
import json
import math
import sys
from dataclasses import replace
from pathlib import Path

HERE=Path(__file__).resolve().parent
ROOT=HERE.parents[1]
PROOF=ROOT/'proof/early_exit_vs_released'
sys.path[:0]=[str(ROOT),str(ROOT/'proof/depth_bitstream'),str(PROOF),str(HERE)]
from audit_exact_context_pilot import sha
from active_canvas_coupling import ActiveCanvasCoupler
from active_canvas_replicate import ActiveCanvasReplicateCoupler


def main()->None:
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--fallback',choices=('zero','replicate'),required=True)
    p.add_argument('--max-new-cases',type=int,default=120)
    p.add_argument('--exact',type=Path,default=HERE/'results/div2k_beta/quality_floor/exact_context_validation24.json')
    p.add_argument('--manifest',type=Path,default=HERE/'results/div2k_beta/manifest.json')
    p.add_argument('--cases',type=Path,default=Path('/tmp/flexplus-div2k-beta-validation'))
    p.add_argument('--source-dir',type=Path,default=ROOT/'data/DIV2K_valid_HR')
    p.add_argument('--upstream',type=Path,default=Path('/tmp/flexplus-proof-clean/proof/early_exit_vs_released/.local/DCVC'))
    p.add_argument('--extension',type=Path,default=Path('/tmp/flexplus-proof-clean/proof/early_exit_vs_released/.local/entropy'))
    p.add_argument('--release',type=Path,default=PROOF/'artifacts/released_cvpr2026_image.pth.tar')
    p.add_argument('--e15',type=Path,default=PROOF/'artifacts/e15_epoch15.pth.tar')
    p.add_argument('--output',type=Path)
    args=p.parse_args()
    if args.output is None:
        args.output=HERE/f'results/div2k_beta/quality_floor/active_canvas_validation24_{args.fallback}.json'

    import numpy as np
    import torch
    from PIL import Image
    from model_io import load_model
    from reference_codec import ReferenceCodec
    from flexuf.config import FlexUFConfig
    from flexuf.model import FlexUFIntra,load_flexuf_state
    import flexuf.backbone.coupling as coupling_module
    sys.path.insert(0,str(args.upstream.resolve()))
    from src.utils.transforms import rgb2ycbcr_np
    torch.set_num_threads(1)
    torch.set_num_interop_threads(1)

    exact=json.loads(args.exact.read_text())
    manifest=json.loads(args.manifest.read_text())
    assert len(exact['rows'])==120 and len(exact['images'])==24
    assert exact['qps']==manifest['qps']==[0,16,32,48,63]
    manifest_images={r['image']:r for r in manifest['rows']['validation']}
    assert set(exact['images'])==set(manifest_images)
    class_path=HERE/('active_canvas_coupling.py' if args.fallback=='zero' else 'active_canvas_replicate.py')
    critical=[ROOT/'flexuf/backbone/decoder.py',ROOT/'flexuf/backbone/coupling.py',
              ROOT/'flexuf/model.py',ROOT/'flexuf/config.py',Path(__file__),class_path]
    provenance={'exact_sha256':sha(args.exact),'manifest_sha256':sha(args.manifest),
                'released_sha256':sha(args.release),'e15_sha256':sha(args.e15),
                'source_code_sha256':{str(x.relative_to(ROOT)):sha(x) for x in critical}}
    assert exact['provenance']['manifest_sha256']==provenance['manifest_sha256']
    assert exact['provenance']['release_sha256']==provenance['released_sha256']
    assert exact['provenance']['e15_sha256']==provenance['e15_sha256']
    result={'schema':1,'scope':f'DIV2K validation24 x QP5 frozen FUFREF2, active-neighbour {args.fallback} fallback, untimed CPU FP32; no retraining',
            'fallback':args.fallback,'expected_cases':120,'complete':False,
            'quality_unit':'Delta444 dB against full-frame e15',
            'provenance':provenance,'rows':[]}
    if args.output.exists():
        old=json.loads(args.output.read_text())
        if old['provenance']!=provenance or old['fallback']!=args.fallback:
            raise RuntimeError('Existing replay has different provenance')
        result['rows']=old['rows']
    done={(r['image'],r['qp']) for r in result['rows']}
    assert len(done)==len(result['rows'])

    released,info=load_model(args.release,12,args.upstream)
    codec=ReferenceCodec(released,info['sha256'],args.extension)
    ckpt=torch.load(args.e15,map_location='cpu',weights_only=False)
    cfg=FlexUFConfig(**ckpt['config'])
    model=FlexUFIntra(cfg).eval();load_flexuf_state(model,ckpt)
    dec=model.dec
    original_cfg,original_repair=dec.cfg,dec.seam_repair
    original_class=coupling_module.CanvasCoupler
    coupling_module.CanvasCoupler=(ActiveCanvasCoupler if args.fallback=='zero'
                                   else ActiveCanvasReplicateCoupler)
    source_cache={}
    added=0
    args.output.parent.mkdir(parents=True,exist_ok=True)
    with torch.inference_mode():
        for archived in exact['rows']:
            image,qp=archived['image'],archived['qp']
            if (image,qp) in done:continue
            if added>=args.max_new_cases:break
            im=manifest_images[image]
            source_path=args.source_dir/image
            assert sha(source_path)==im['source_sha256']
            if image not in source_cache:
                x,y,w,h=im['crop_xywh']
                with Image.open(source_path) as file:
                    rgb=np.asarray(file.convert('RGB').crop((x,y,x+w,y+h)),dtype=np.float32)/255
                source_cache[image]=torch.from_numpy(rgb2ycbcr_np(rgb)-.5).permute(2,0,1)[None].contiguous()
            source=source_cache[image]
            stem=f'{Path(image).stem}_qp{qp}'
            case_path=args.cases/f'{stem}.json'
            stream_path=args.cases/f'{stem}.fufref2'
            case=json.loads(case_path.read_text())
            assert sha(case_path)==archived['case_sha256']
            assert sha(stream_path)==archived['stream_sha256']==case['stream_sha256']
            route=torch.tensor(archived['exit_map'],dtype=torch.long)
            latent,q,_=codec.decode_latent(stream_path.read_bytes())
            full=dec.forward_full(latent,q)
            h,w=source.shape[-2:]
            full_mse=float((source-full[:,:,:h,:w]).square().mean())
            assert math.isclose(full_mse,case['e15_full_mse444'],abs_tol=1e-11)
            dec.cfg=replace(cfg,tile_coupling=True,sorted_tiles=False,trunk_halo=0)
            outputs={}
            for repair in (False,True):
                dec.seam_repair=original_repair if repair else None
                output=dec(latent,q,exit_map=route)
                mse=float((source-output[:,:,:h,:w]).square().mean())
                outputs['with_repair' if repair else 'no_repair']={
                    'delta444_db':10*math.log10(mse/full_mse)}
                if all(k==cfg.num_exits-1 for k in route.tolist()) and not repair:
                    if not torch.allclose(output,full,atol=1e-4,rtol=1e-4):
                        raise RuntimeError(f'All-deep exactness failed: {stem}')
            dec.cfg,dec.seam_repair=original_cfg,original_repair
            result['rows'].append({'image':image,'qp':qp,
                                   'case_sha256':sha(case_path),'stream_sha256':sha(stream_path),
                                   'exit_map':route.tolist(),
                                   'deployed_delta444_db':archived['deployed_delta444_db'],
                                   'exact_context_delta444_db':archived['exact_context_delta444_db'],
                                   'coupled':outputs})
            added+=1
            result['complete']=len(result['rows'])==120
            args.output.write_text(json.dumps(result,indent=2)+'\n')
            print(f'{image} QP{qp}: active-{args.fallback} no-repair '
                  f'{outputs["no_repair"]["delta444_db"]:.4f} dB',flush=True)
    assert provenance['source_code_sha256']=={str(x.relative_to(ROOT)):sha(x) for x in critical}
    coupling_module.CanvasCoupler=original_class
    print(f'complete={result["complete"]}; cases={len(result["rows"])}')


if __name__=='__main__':main()
