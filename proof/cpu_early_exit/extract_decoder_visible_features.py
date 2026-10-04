"""Extract pre-synthesis features from released FUFREF2 latent streams.

Uses only the bitstream, its QP, and the frozen early-exit map. It does not
access the source image or any reconstruction when computing features.
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


def main() -> None:
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--cal',type=Path,default=HERE/'results/div2k_beta/quality_floor/exact_context_calibration24.json')
    p.add_argument('--val',type=Path,default=HERE/'results/div2k_beta/quality_floor/exact_context_validation24.json')
    p.add_argument('--cal-streams',type=Path,default=Path('/tmp/flexplus-div2k-beta-calibration'))
    p.add_argument('--val-streams',type=Path,default=Path('/tmp/flexplus-div2k-beta-validation'))
    p.add_argument('--upstream',type=Path,default=Path('/tmp/flexplus-proof-clean/proof/early_exit_vs_released/.local/DCVC'))
    p.add_argument('--extension',type=Path,default=Path('/tmp/flexplus-proof-clean/proof/early_exit_vs_released/.local/entropy'))
    p.add_argument('--release',type=Path,default=PROOF/'artifacts/released_cvpr2026_image.pth.tar')
    p.add_argument('--output',type=Path,default=HERE/'results/div2k_beta/quality_floor/decoder_visible_features.json')
    args=p.parse_args()

    import numpy as np
    import torch
    from model_io import load_model
    from reference_codec import ReferenceCodec
    torch.set_num_threads(1)
    torch.set_num_interop_threads(1)
    source_sets={split:json.loads(path.read_text()) for split,path in [('calibration',args.cal),('validation',args.val)]}
    for split,data in source_sets.items():
        if len(data['rows'])!=120 or len(data['images'])!=24:
            raise RuntimeError(f'Incomplete {split} exact-context labels')
    released,info=load_model(args.release,12,args.upstream)
    codec=ReferenceCodec(released,info['sha256'],args.extension)
    result={'schema':1,'scope':'Decoder-visible pre-synthesis latent, stream size, QP, and frozen route; exact-context labels joined only for downstream audit',
            'release_sha256':sha(args.release),
            'input_sha256':{'calibration':sha(args.cal),'validation':sha(args.val)},
            'rows':{'calibration':[],'validation':[]}}
    with torch.inference_mode():
        for split,data in source_sets.items():
            directory=args.cal_streams if split=='calibration' else args.val_streams
            for row in data['rows']:
                stem=Path(row['image']).stem
                stream_path=directory/f'{stem}_qp{row["qp"]}.fufref2'
                if sha(stream_path)!=row['stream_sha256']:
                    raise RuntimeError(f'Stream hash mismatch: {stream_path}')
                bits=stream_path.read_bytes()
                latent,q,_=codec.decode_latent(bits)
                a=latent[0].detach().cpu().numpy().astype(np.float64)
                channels,h,w=a.shape
                if h%2 or w%3:
                    raise RuntimeError(f'Unexpected tile geometry: {a.shape}')
                route=np.asarray(row['exit_map'],dtype=np.float64)
                tile_energy=np.asarray([np.mean(np.abs(a[:,(t//3)*h//2:(t//3+1)*h//2,
                                                        (t%3)*w//3:(t%3+1)*w//3]))
                                        for t in range(6)])
                weights=5-route
                abs_a=np.abs(a)
                feature={
                    'qp':float(row['qp']),
                    'stream_bytes':float(len(bits)),
                    'route_mean':float(route.mean()),
                    'route_min':float(route.min()),
                    'route_std':float(route.std()),
                    'route_shallow_fraction':float(np.mean(route<=3)),
                    'route_deep_fraction':float(np.mean(route==5)),
                    'route_adjacent_jumps':float(sum(abs(route[i]-route[j]) for i,j in
                                                       [(0,1),(1,2),(3,4),(4,5),(0,3),(1,4),(2,5)])),
                    'latent_abs_mean':float(abs_a.mean()),
                    'latent_std':float(a.std()),
                    'latent_abs_p90':float(np.quantile(abs_a,.9)),
                    'latent_zero_fraction':float(np.mean(a==0)),
                    'latent_spatial_tv':float(np.mean(np.abs(np.diff(a,axis=1)))+
                                              np.mean(np.abs(np.diff(a,axis=2)))),
                    'tile_energy_mean':float(tile_energy.mean()),
                    'tile_energy_max':float(tile_energy.max()),
                    'tile_energy_std':float(tile_energy.std()),
                    'shallow_weighted_tile_energy':float(np.mean(tile_energy*weights)),
                    'shallow_deep_energy_gap':float(np.mean(tile_energy[route<=3])-
                                                    np.mean(tile_energy[route>3]))
                        if np.any(route<=3) and np.any(route>3) else 0.,
                }
                result['rows'][split].append({'image':row['image'],'qp':row['qp'],
                                              'features':feature,
                                              'label_over_0p1':int(row['exact_context_delta444_db']>.1),
                                              'exact_context_delta444_db':row['exact_context_delta444_db']})
            print(f'{split}: {len(result["rows"][split])} rows',flush=True)
    args.output.parent.mkdir(parents=True,exist_ok=True)
    args.output.write_text(json.dumps(result,indent=2)+'\n')


if __name__=='__main__':main()
