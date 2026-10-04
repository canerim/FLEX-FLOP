"""Source-informed CPU diagnostic: uniformly deepen the 24 failing routes.

This looks at ground-truth reconstruction error and cannot be deployed as a
decoder router. It bounds a simple global depth-increase fallback under the
exact-context counterfactual, with ideal arithmetic rather than latency.
"""
from __future__ import annotations

import argparse
import json
import math
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
PROOF = ROOT/'proof/early_exit_vs_released'
sys.path[:0] = [str(ROOT), str(ROOT/'proof/depth_bitstream'), str(PROOF)]

from audit_exact_context_pilot import exact_reconstruct, sha
from analyze_ideal_context_cohort import ideal_cost


def main() -> None:
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--base',type=Path,default=HERE/'results/div2k_beta/quality_floor/exact_context_validation24.json')
    p.add_argument('--cases',type=Path,default=Path('/tmp/flexplus-div2k-beta-validation'))
    p.add_argument('--source-dir',type=Path,default=ROOT/'data/DIV2K_valid_HR')
    p.add_argument('--upstream',type=Path,default=Path('/tmp/flexplus-proof-clean/proof/early_exit_vs_released/.local/DCVC'))
    p.add_argument('--extension',type=Path,default=Path('/tmp/flexplus-proof-clean/proof/early_exit_vs_released/.local/entropy'))
    p.add_argument('--e15',type=Path,default=PROOF/'artifacts/e15_epoch15.pth.tar')
    p.add_argument('--release',type=Path,default=PROOF/'artifacts/released_cvpr2026_image.pth.tar')
    p.add_argument('--output',type=Path,default=HERE/'results/div2k_beta/quality_floor/uniform_depth_oracle_validation24.json')
    args=p.parse_args()

    import numpy as np
    import torch
    from PIL import Image
    from model_io import load_model
    from reference_codec import ReferenceCodec
    from flexuf.config import FlexUFConfig
    from flexuf.model import FlexUFIntra,load_flexuf_state
    sys.path.insert(0,str(args.upstream.resolve()))
    from src.utils.transforms import rgb2ycbcr_np

    torch.set_num_threads(1)
    torch.set_num_interop_threads(1)
    base=json.loads(args.base.read_text())
    if len(base['rows'])!=120 or len(base['images'])!=24:
        raise RuntimeError('Full validation baseline required')
    if base['provenance']['e15_sha256']!=sha(args.e15) or base['provenance']['release_sha256']!=sha(args.release):
        raise RuntimeError('Checkpoint provenance mismatch')
    failures=[r for r in base['rows'] if r['exact_context_delta444_db']>.1]
    provenance={'base_sha256':sha(args.base),'e15_sha256':sha(args.e15),'release_sha256':sha(args.release)}
    result={'schema':1,'scope':'Source-informed uniform depth increase on exact-context validation failures; nondeployable oracle; ideal shared MAC only, no latency',
            'selection':'For each violating fixed route, raise every tile by 1, then 2, then 3 exits (clip to deepest); stop at first Delta444 <=0.1 dB',
            'provenance':provenance,'rows':[]}
    if args.output.exists():
        old=json.loads(args.output.read_text())
        if old['provenance']!=provenance or old['selection']!=result['selection']:
            raise RuntimeError('Existing output has different provenance')
        result['rows']=old['rows']
    done={(r['image'],r['qp']) for r in result['rows']}

    released,info=load_model(args.release,12,args.upstream)
    codec=ReferenceCodec(released,info['sha256'],args.extension)
    ckpt=torch.load(args.e15,map_location='cpu',weights_only=False)
    cfg=FlexUFConfig(**ckpt['config'])
    model=FlexUFIntra(cfg).eval()
    load_flexuf_state(model,ckpt)
    dec=model.dec
    args.output.parent.mkdir(parents=True,exist_ok=True)
    with torch.inference_mode():
        for row in failures:
            key=(row['image'],row['qp'])
            if key in done:continue
            stemname=Path(row['image']).stem
            case_path=args.cases/f'{stemname}_qp{row["qp"]}.json'
            stream_path=args.cases/f'{stemname}_qp{row["qp"]}.fufref2'
            case=json.loads(case_path.read_text())
            source_path=args.source_dir/row['image']
            if (sha(case_path)!=row['case_sha256'] or sha(stream_path)!=row['stream_sha256'] or
                sha(source_path)!=case['source_sha256']):
                raise RuntimeError(f'Case or source mismatch: {case_path}')
            x,y,w,h=case['crop_xywh']
            with Image.open(source_path) as image:
                rgb=np.asarray(image.convert('RGB').crop((x,y,x+w,y+h)),dtype=np.float32)/255
            source=torch.from_numpy(rgb2ycbcr_np(rgb)-.5).permute(2,0,1)[None].contiguous()
            latent,q,_=codec.decode_latent(stream_path.read_bytes())
            ref=dec.forward_full(latent,q)
            mse_ref=float((source-ref[:,:,:h,:w]).square().mean())
            if not math.isclose(mse_ref,case['e15_full_mse444'],abs_tol=1e-11):
                raise RuntimeError(f'Reference MSE mismatch: {case_path}')
            stem=dec.upsample(latent)
            for group in dec.groups[:cfg.split_depth]:stem=group(stem)
            trials=[]
            for step in (1,2,3):
                route=[min(cfg.num_exits-1,v+step) for v in row['exit_map']]
                output=exact_reconstruct(dec,cfg,stem,q,route)
                mse=float((source-output[:,:,:h,:w]).square().mean())
                delta=10*math.log10(mse/mse_ref)
                cost,_=ideal_cost(route,cfg)
                trials.append({'uniform_increment':step,'exit_map':route,
                               'exact_context_delta444_db':delta,
                               'ideal_shared_saving_pct':100*(1-cost)})
                if delta<=.1:break
            if trials[-1]['exact_context_delta444_db']>.1:
                raise RuntimeError(f'All-deep did not recover e15: {case_path}')
            result['rows'].append({'image':row['image'],'qp':row['qp'],
                                   'base_exit_map':row['exit_map'],
                                   'base_exact_context_delta444_db':row['exact_context_delta444_db'],
                                   'base_ideal_shared_saving_pct':row['ideal_shared_saving_pct'],
                                   'trials':trials,'chosen':trials[-1]})
            args.output.write_text(json.dumps(result,indent=2)+'\n')
            print(f'{row["image"]} QP{row["qp"]}: +{trials[-1]["uniform_increment"]} '
                  f'-> {trials[-1]["exact_context_delta444_db"]:.4f} dB, '
                  f'{trials[-1]["ideal_shared_saving_pct"]:.1f}% ideal saved',flush=True)


if __name__=='__main__':main()
