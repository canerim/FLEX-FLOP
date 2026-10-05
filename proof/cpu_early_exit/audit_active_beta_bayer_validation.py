"""Fixed-histogram Bayer placement for frozen active-beta DIV2K validation."""
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
from active_canvas_replicate import ActiveCanvasReplicateCoupler


def main()->None:
    ap=argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--max-new-cases',type=int,default=120)
    ap.add_argument('--output',type=Path,default=HERE/'results/div2k_beta/quality_floor/active_beta_bayer_validation24.json')
    args=ap.parse_args()
    assert 1<=args.max_new_cases<=120

    import numpy as np
    import torch
    from PIL import Image
    from model_io import load_model
    from reference_codec import ReferenceCodec
    from flexuf.config import FlexUFConfig
    from flexuf.model import FlexUFIntra,load_flexuf_state
    from flexuf.cost import frame_relative_cost
    from flexplus.null_models_uf import bayer
    import flexuf.backbone.coupling as coupling_module

    upstream=Path('/tmp/flexplus-proof-clean/proof/early_exit_vs_released/.local/DCVC')
    extension=Path('/tmp/flexplus-proof-clean/proof/early_exit_vs_released/.local/entropy')
    sys.path.insert(0,str(upstream.resolve()))
    from src.utils.transforms import rgb2ycbcr_np
    torch.set_num_threads(1)
    torch.set_num_interop_threads(1)
    base=HERE/'results/div2k_beta'
    validation_path=base/'quality_floor/active_replicate_beta_validation24.json'
    policy_path=base/'quality_floor/active_replicate_beta_locked_policy.json'
    manifest_path=base/'manifest.json'
    exact_path=base/'quality_floor/exact_context_validation24.json'
    released_path=PROOF/'artifacts/released_cvpr2026_image.pth.tar'
    e15_path=PROOF/'artifacts/e15_epoch15.pth.tar'
    cases=Path('/tmp/flexplus-div2k-beta-validation')
    source_dir=ROOT/'data/DIV2K_valid_HR'

    validation=json.loads(validation_path.read_text())
    policy=json.loads(policy_path.read_text())
    manifest=json.loads(manifest_path.read_text())
    exact=json.loads(exact_path.read_text())
    assert validation['complete'] and len(validation['rows'])==120
    assert validation['provenance']['policy_sha256']==sha(policy_path)
    assert validation['provenance']['manifest_sha256']==sha(manifest_path)
    assert validation['provenance']['exact_sha256']==sha(exact_path)
    assert policy['source_sha256']['manifest']==sha(manifest_path)
    assert exact['provenance']['release_sha256']==sha(released_path)
    assert exact['provenance']['e15_sha256']==sha(e15_path)
    assert manifest['qps']==[0,16,32,48,63]
    by_key={(r['image'],r['qp']):r for r in validation['rows']}
    archived={(r['image'],r['qp']):r for r in exact['rows']}
    assert len(by_key)==len(archived)==120 and set(by_key)==set(archived)
    images={r['image']:r for r in manifest['rows']['validation']}
    assert len(images)==24
    critical=[Path(__file__),HERE/'active_canvas_replicate.py',
              ROOT/'flexplus/null_models_uf.py',ROOT/'flexuf/backbone/decoder.py',
              ROOT/'flexuf/backbone/coupling.py',ROOT/'flexuf/model.py',
              ROOT/'flexuf/config.py']
    provenance={'validation_sha256':sha(validation_path),'policy_sha256':sha(policy_path),
                'manifest_sha256':sha(manifest_path),'exact_sha256':sha(exact_path),
                'released_sha256':sha(released_path),'e15_sha256':sha(e15_path),
                'source_code_sha256':{str(p.relative_to(ROOT)):sha(p) for p in critical}}
    result={'scope':'DIV2K validation24 x QP5 frozen active-replicate/no-repair beta: routed versus deterministic Bayer-rank maps at identical exit histogram, FUFREF2 stream and analytical synthesis-conv MAC. CPU FP32 untimed.',
            'expected_cases':120,'complete':False,'provenance':provenance,'rows':[]}
    if args.output.exists():
        old=json.loads(args.output.read_text())
        assert old['provenance']==provenance and old['expected_cases']==120
        result['rows']=old['rows']
    done={(r['image'],r['qp']) for r in result['rows']}
    assert len(done)==len(result['rows'])
    args.output.parent.mkdir(parents=True,exist_ok=True)

    released,info=load_model(released_path,12,upstream)
    codec=ReferenceCodec(released,info['sha256'],extension)
    checkpoint=torch.load(e15_path,map_location='cpu',weights_only=False)
    cfg=FlexUFConfig(**checkpoint['config'])
    assert cfg.rgb_patch==256 and cfg.num_exits==6 and cfg.tile_pad_mode=='replicate'
    assert not cfg.tile_coupling and cfg.seam_repair=='grid' and cfg.trunk_halo==0
    model=FlexUFIntra(cfg).eval()
    load_flexuf_state(model,checkpoint)
    dec=model.dec
    original_cfg,original_repair=dec.cfg,dec.seam_repair
    original_class=coupling_module.CanvasCoupler
    coupling_module.CanvasCoupler=ActiveCanvasReplicateCoupler
    dec.cfg=replace(cfg,tile_coupling=True,sorted_tiles=False,trunk_halo=0)
    dec.seam_repair=None
    rank=bayer(2,3)
    positions=torch.argsort(rank,stable=True)
    assert len(positions)==6
    source_cache={}
    added=0
    try:
        with torch.inference_mode():
            for image,qp in sorted(by_key):
                if (image,qp) in done:continue
                if added>=args.max_new_cases:break
                archived_row=archived[image,qp]
                primary=by_key[image,qp]
                source_path=source_dir/image
                item=images[image]
                assert sha(source_path)==item['source_sha256']
                if image not in source_cache:
                    x,y,w,h=item['crop_xywh']
                    assert (w,h)==(768,512)
                    with Image.open(source_path) as file:
                        rgb=np.asarray(file.convert('RGB').crop((x,y,x+w,y+h)),
                                       dtype=np.float32)/255
                    source_cache[image]=torch.from_numpy(rgb2ycbcr_np(rgb)-.5).permute(2,0,1)[None].contiguous()
                source=source_cache[image]
                stem_name=f'{Path(image).stem}_qp{qp}'
                case_path=cases/f'{stem_name}.json'
                stream_path=cases/f'{stem_name}.fufref2'
                assert sha(case_path)==primary['case_sha256']==archived_row['case_sha256']
                assert sha(stream_path)==primary['stream_sha256']==archived_row['stream_sha256']
                selected=torch.tensor(primary['exit_map'],dtype=torch.long)
                assert len(selected)==6 and primary['selected_beta']==policy['beta'][str(qp)]
                fixed=torch.empty_like(selected)
                fixed[positions]=selected.sort(descending=True).values
                assert torch.equal(torch.bincount(fixed,minlength=cfg.num_exits),
                                   torch.bincount(selected,minlength=cfg.num_exits))
                assert abs(frame_relative_cost(fixed,cfg)-
                           frame_relative_cost(selected,cfg))<1e-7
                latent,quant_step,_=codec.decode_latent(stream_path.read_bytes())
                h,w=source.shape[-2:]
                full=dec.forward_full(latent,quant_step)[:,:,:h,:w]
                full_mse=float((source-full).square().mean())
                assert full_mse>0
                routed=dec(latent,quant_step,exit_map=selected)[:,:,:h,:w]
                routed_mse=float((source-routed).square().mean())
                routed_loss=10*math.log10(routed_mse/full_mse)
                assert abs(routed_loss-primary['new_delta444_db'])<1e-7
                if torch.equal(selected,fixed):
                    bayer_mse=routed_mse
                else:
                    out=dec(latent,quant_step,exit_map=fixed)[:,:,:h,:w]
                    bayer_mse=float((source-out).square().mean())
                assert bayer_mse>0
                result['rows'].append({
                    'image':image,'qp':qp,'case_sha256':sha(case_path),
                    'stream_sha256':sha(stream_path),'source_sha256':item['source_sha256'],
                    'selected_beta':primary['selected_beta'],
                    'router_map':selected.tolist(),'bayer_map':fixed.tolist(),
                    'exit_counts':torch.bincount(selected,minlength=cfg.num_exits).tolist(),
                    'same_map':bool(torch.equal(selected,fixed)),
                    'conv_mac_saving_pct':primary['new_conv_mac_saving_pct'],
                    'router_mse444':routed_mse,'bayer_mse444':bayer_mse,
                    'router_delta444_db':routed_loss,
                    'bayer_delta444_db':10*math.log10(bayer_mse/full_mse),
                    'placement_gain_db':10*math.log10(bayer_mse/routed_mse),
                })
                added+=1
                result['complete']=len(result['rows'])==120
                args.output.write_text(json.dumps(result,indent=2)+'\n')
                print(f'{stem_name}: gain {result["rows"][-1]["placement_gain_db"]:.6f} dB',flush=True)
    finally:
        dec.cfg,dec.seam_repair=original_cfg,original_repair
        coupling_module.CanvasCoupler=original_class
    assert provenance['source_code_sha256']=={str(p.relative_to(ROOT)):sha(p)
                                               for p in critical}
    print(f"complete={result['complete']}; cases={len(result['rows'])}")


if __name__=='__main__':main()
