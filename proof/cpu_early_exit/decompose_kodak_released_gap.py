"""Decompose Kodak released-to-deployed YUV611 gap on fixed real streams.

Released - deployed = (released - full-frame e15)
                    + (full-frame e15 - exact-context routed e15)
                    + (exact-context routed e15 - deployed e15).
The additive identity is per image/QP, not only a mean approximation.
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

HERE=Path(__file__).resolve().parent
ROOT=HERE.parents[1]
PROOF=ROOT/'proof/early_exit_vs_released'
sys.path[:0]=[str(ROOT),str(ROOT/'proof/depth_bitstream'),str(PROOF)]
from audit_exact_context_pilot import sha


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--exact',type=Path,default=HERE/'results/div2k_beta/quality_floor/exact_context_kodak24.json')
    p.add_argument('--scan-dir',type=Path,default=HERE/'results/kodak24_qp5')
    p.add_argument('--transfer-dir',type=Path,default=Path('/tmp/flexplus-div2k-beta-kodak'))
    p.add_argument('--source-dir',type=Path,default=ROOT/'data/kodak')
    p.add_argument('--upstream',type=Path,default=Path('/tmp/flexplus-proof-clean/proof/early_exit_vs_released/.local/DCVC'))
    p.add_argument('--extension',type=Path,default=Path('/tmp/flexplus-proof-clean/proof/early_exit_vs_released/.local/entropy'))
    p.add_argument('--release',type=Path,default=PROOF/'artifacts/released_cvpr2026_image.pth.tar')
    p.add_argument('--e15',type=Path,default=PROOF/'artifacts/e15_epoch15.pth.tar')
    p.add_argument('--output',type=Path,default=HERE/'results/div2k_beta/quality_floor/kodak_released_gap_decomposition.json')
    args=p.parse_args()

    import numpy as np
    import torch
    from PIL import Image
    from model_io import load_model
    from reference_codec import ReferenceCodec
    from matched_routed_cpu import psnr
    from flexuf.config import FlexUFConfig
    from flexuf.model import FlexUFIntra,load_flexuf_state
    sys.path.insert(0,str(args.upstream.resolve()))
    from src.utils.transforms import rgb2ycbcr_np

    torch.set_num_threads(1);torch.set_num_interop_threads(1)
    exact=json.loads(args.exact.read_text())
    if len(exact['rows'])!=120 or exact['provenance']['release_sha256']!=sha(args.release) or exact['provenance']['e15_sha256']!=sha(args.e15):
        raise RuntimeError('Incomplete or mismatched exact-context cohort')
    result={'schema':1,'scope':'Kodak24 x QP5 real streams; additive YUV611 gap components; CPU untimed',
            'exact_sha256':sha(args.exact),'release_sha256':sha(args.release),
            'e15_sha256':sha(args.e15),'rows':[]}
    if args.output.exists():
        old=json.loads(args.output.read_text())
        if any(old[k]!=result[k] for k in ('schema','exact_sha256','release_sha256','e15_sha256')):
            raise RuntimeError('Existing output provenance mismatch')
        result['rows']=old['rows']
    done={(r['image'],r['qp']) for r in result['rows']}
    released,info=load_model(args.release,12,args.upstream)
    codec=ReferenceCodec(released,info['sha256'],args.extension)
    ckpt=torch.load(args.e15,map_location='cpu',weights_only=False)
    cfg=FlexUFConfig(**ckpt['config'])
    model=FlexUFIntra(cfg).eval();load_flexuf_state(model,ckpt)
    dec=model.dec
    args.output.parent.mkdir(parents=True,exist_ok=True)
    with torch.inference_mode():
        for row in exact['rows']:
            image_name,qp=row['image'],row['qp']
            if (image_name,qp) in done:continue
            stem=f'{Path(image_name).stem}_qp{qp}'
            scan_path=args.scan_dir/f'{stem}.json'
            transfer_path=args.transfer_dir/f'{stem}.json'
            stream_path=args.scan_dir/f'{stem}.fufref2'
            source_path=args.source_dir/image_name
            if (sha(scan_path)!=row['scan_sha256'] or
                sha(transfer_path)!=row['transfer_sha256'] or
                sha(stream_path)!=row['stream_sha256']):
                raise RuntimeError(f'Input hash mismatch: {stem}')
            scan=json.loads(scan_path.read_text())
            transfer=json.loads(transfer_path.read_text())
            if scan['source_sha256']!=sha(source_path):
                raise RuntimeError(f'Source mismatch: {stem}')
            rgb=np.asarray(Image.open(source_path).convert('RGB'),dtype=np.float32)/255
            source=torch.from_numpy(rgb2ycbcr_np(rgb)-.5).permute(2,0,1)[None].contiguous()
            latent,q,_=codec.decode_latent(stream_path.read_bytes())
            full=dec.forward_full(latent,q)
            h,w=scan['shape'];full=full[:,:,:h,:w]
            mse=float((source-full).square().mean())
            if abs(mse-transfer['e15_full_mse444'])>1e-11:
                raise RuntimeError(f'e15 full MSE mismatch: {stem}')
            full_quality=psnr(source,full)['yuv_6_1_1']
            released_gap=row['released_yuv611_db']-full_quality
            route_gap=full_quality-row['exact_context_yuv611_db']
            context_gap=row['exact_context_yuv611_db']-row['deployed_yuv611_db']
            if abs(released_gap+route_gap+context_gap-row['deployed_gap_to_released_yuv611_db'])>1e-9:
                raise RuntimeError(f'Gap fails additive identity: {stem}')
            result['rows'].append({'image':image_name,'qp':qp,'e15_full_yuv611_db':full_quality,
                                   'released_minus_e15_full_db':released_gap,
                                   'e15_full_minus_exact_route_db':route_gap,
                                   'exact_route_minus_deployed_db':context_gap,
                                   'released_minus_deployed_db':row['deployed_gap_to_released_yuv611_db']})
            args.output.write_text(json.dumps(result,indent=2)+'\n')
            print(f'{stem}: checkpoint {released_gap:.4f}, route {route_gap:.4f}, context {context_gap:.4f}',flush=True)


if __name__=='__main__':main()
