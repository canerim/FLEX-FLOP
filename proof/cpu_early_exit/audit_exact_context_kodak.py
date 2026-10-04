"""Replay Kodak24 FUFREF2 streams with the frozen DIV2K beta and exact context.

All quality uses full-frame e15 centred YCbCr444 as reference. Analytical
shared-context MAC is an optimistic arithmetic bound, not measured latency.
Kodak has been inspected in prior project work; this is external-dataset
diagnosis rather than a pristine untouched benchmark.
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
sys.path[:0]=[str(ROOT),str(ROOT/'proof/depth_bitstream'),str(PROOF)]
from audit_exact_context_pilot import exact_reconstruct,sha
from analyze_ideal_context_cohort import ideal_cost_grid


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--scan-dir',type=Path,default=HERE/'results/kodak24_qp5')
    p.add_argument('--transfer-dir',type=Path,default=Path('/tmp/flexplus-div2k-beta-kodak'))
    p.add_argument('--source-dir',type=Path,default=ROOT/'data/kodak')
    p.add_argument('--policy',type=Path,default=HERE/'results/div2k_beta/locked_policy.json')
    p.add_argument('--upstream',type=Path,default=Path('/tmp/flexplus-proof-clean/proof/early_exit_vs_released/.local/DCVC'))
    p.add_argument('--extension',type=Path,default=Path('/tmp/flexplus-proof-clean/proof/early_exit_vs_released/.local/entropy'))
    p.add_argument('--release',type=Path,default=PROOF/'artifacts/released_cvpr2026_image.pth.tar')
    p.add_argument('--e15',type=Path,default=PROOF/'artifacts/e15_epoch15.pth.tar')
    p.add_argument('--output',type=Path,default=HERE/'results/div2k_beta/quality_floor/exact_context_kodak24.json')
    args=p.parse_args()

    import numpy as np
    import torch
    from PIL import Image
    from model_io import load_model
    from reference_codec import ReferenceCodec
    from flexuf.config import FlexUFConfig
    from flexuf.model import FlexUFIntra,load_flexuf_state
    from flexuf.kernels.planned_decoder import forward_from_stem_with_cpu_map
    from matched_routed_cpu import psnr
    sys.path.insert(0,str(args.upstream.resolve()))
    from src.utils.transforms import rgb2ycbcr_np

    torch.set_num_threads(1)
    torch.set_num_interop_threads(1)
    manifest=json.loads((args.scan_dir/'manifest.json').read_text())
    policy=json.loads(args.policy.read_text())
    images=manifest['images'];qps=manifest['qps']
    if images!=[f'kodim{i:02d}.png' for i in range(1,25)] or qps!=[0,16,32,48,63]:
        raise RuntimeError('Unexpected Kodak24 manifest')
    prov={'scan_manifest_sha256':sha(args.scan_dir/'manifest.json'),
          'policy_sha256':sha(args.policy),'release_sha256':sha(args.release),
          'e15_sha256':sha(args.e15)}
    if manifest['released_sha256']!=prov['release_sha256'] or manifest['e15_sha256']!=prov['e15_sha256']:
        raise RuntimeError('Scan/checkpoint mismatch')
    result={'schema':1,'scope':'Kodak24 x QP5, real FUFREF2, locked DIV2K beta; CPU exact-context diagnosis, no retraining or latency',
            'images':images,'qps':qps,'provenance':prov,
            'quality_unit':'Delta dB = 10log10(MSE_policy/MSE_e15_full), centred YCbCr444',
            'ideal_cost_unit':'optimistic unique feature-cell synthesis MAC, no execution overhead',
            'rows':[]}
    if args.output.exists():
        old=json.loads(args.output.read_text())
        if any(old[k]!=result[k] for k in ('schema','images','qps','provenance')):
            raise RuntimeError('Existing output provenance differs')
        result['rows']=[r for r in old['rows'] if 'exact_context_yuv611_db' in r]
    done={(r['image'],r['qp']) for r in result['rows']}
    released,info=load_model(args.release,12,args.upstream)
    codec=ReferenceCodec(released,info['sha256'],args.extension)
    ckpt=torch.load(args.e15,map_location='cpu',weights_only=False)
    cfg=FlexUFConfig(**ckpt['config'])
    model=FlexUFIntra(cfg).eval();load_flexuf_state(model,ckpt)
    dec=model.dec
    dec.cfg=replace(cfg,sorted_tiles=True)
    args.output.parent.mkdir(parents=True,exist_ok=True)
    with torch.inference_mode():
        for image_name in images:
            source_path=args.source_dir/image_name
            rgb=np.asarray(Image.open(source_path).convert('RGB'),dtype=np.float32)/255
            source=torch.from_numpy(rgb2ycbcr_np(rgb)-.5).permute(2,0,1)[None].contiguous()
            for qp in qps:
                if (image_name,qp) in done:continue
                stemname=f'{Path(image_name).stem}_qp{qp}'
                scan_path=args.scan_dir/f'{stemname}.json'
                transfer_path=args.transfer_dir/f'{stemname}.json'
                stream_path=args.scan_dir/f'{stemname}.fufref2'
                scan=json.loads(scan_path.read_text())
                transfer=json.loads(transfer_path.read_text())
                if (scan['manifest_sha256']!=prov['scan_manifest_sha256'] or
                    scan['source_sha256']!=sha(source_path) or
                    scan['stream_sha256']!=sha(stream_path) or
                    transfer['scan_result_sha256']!=sha(scan_path) or
                    transfer['policy_sha256']!=prov['policy_sha256']):
                    raise RuntimeError(f'Provenance mismatch: {stemname}')
                scores=torch.tensor(scan['route_log_probs'],dtype=torch.float64)
                costs=torch.tensor(scan['router_costs'],dtype=torch.float64)
                route=(scores-float(policy['beta'][str(qp)])*costs[cfg.split_depth:][None,:]).argmax(1)+cfg.split_depth
                if torch.bincount(route,minlength=cfg.num_exits).tolist()!=transfer['locked_exit_counts']:
                    raise RuntimeError(f'Frozen map mismatch: {stemname}')
                latent,q,_=codec.decode_latent(stream_path.read_bytes())
                reference=dec.forward_full(latent,q)
                h,w=scan['shape']
                ref_mse=float((source-reference[:,:,:h,:w]).square().mean())
                if not math.isclose(ref_mse,transfer['e15_full_mse444'],abs_tol=1e-11):
                    raise RuntimeError(f'Reference MSE mismatch: {stemname}')
                stem=dec.upsample(latent)
                for group in dec.groups[:cfg.split_depth]:stem=group(stem)
                deployed=forward_from_stem_with_cpu_map(dec,stem,q,route)
                exact=exact_reconstruct(dec,cfg,stem,q,route.tolist())
                def delta(output):
                    mse=float((source-output[:,:,:h,:w]).square().mean())
                    return 10*math.log10(mse/ref_mse)
                deployed_db=delta(deployed)
                if abs(deployed_db-transfer['locked_beta_delta444_db'])>1e-3:
                    raise RuntimeError(f'Cached/deployed mismatch: {stemname}: {deployed_db}')
                exact_db=delta(exact)
                deployed_yuv=psnr(source,deployed[:,:,:h,:w])['yuv_6_1_1']
                exact_yuv=psnr(source,exact[:,:,:h,:w])['yuv_6_1_1']
                if abs(deployed_yuv-transfer['locked_beta_yuv611_db'])>1e-3:
                    raise RuntimeError(f'Cached/deployed YUV611 mismatch: {stemname}')
                if all(v==cfg.num_exits-1 for v in route.tolist()) and not torch.allclose(exact,reference,atol=1e-4,rtol=1e-4):
                    raise RuntimeError(f'All-deep exact/full mismatch: {stemname}')
                fh,fw=stem.shape[-2:]
                cost,_=ideal_cost_grid(route.tolist(),cfg,fh//cfg.feature_patch,fw//cfg.feature_patch)
                result['rows'].append({'image':image_name,'qp':qp,
                                       'scan_sha256':sha(scan_path),'transfer_sha256':sha(transfer_path),
                                       'stream_sha256':sha(stream_path),'exit_map':route.tolist(),
                                       'deployed_delta444_db':deployed_db,
                                       'exact_context_delta444_db':exact_db,
                                       'released_yuv611_db':scan['released_yuv611_db'],
                                       'deployed_yuv611_db':deployed_yuv,
                                       'exact_context_yuv611_db':exact_yuv,
                                       'deployed_gap_to_released_yuv611_db':scan['released_yuv611_db']-deployed_yuv,
                                       'exact_gap_to_released_yuv611_db':scan['released_yuv611_db']-exact_yuv,
                                       'ideal_shared_saving_pct':100*(1-cost)})
                args.output.write_text(json.dumps(result,indent=2)+'\n')
                print(f'{image_name} QP{qp}: deployed {deployed_db:.4f}, exact {exact_db:.4f} dB',flush=True)


if __name__=='__main__':main()
