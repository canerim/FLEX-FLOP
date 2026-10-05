"""Third-cohort CLIC39 transfer of frozen active-context and beta policies.

Each geometry-selected 768x512 crop/QP is encoded once into a FUFREF2
research stream using the released D12 analysis/entropy model. A frozen
e15/router then reconstructs deployed original-beta, active-replicate at
original beta, and active-replicate at calibration-retuned beta from the
same latent. CPU FP32, one thread, untimed; not a native codec benchmark.
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
from active_canvas_replicate import ActiveCanvasReplicateCoupler


def main()->None:
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--max-new-cases',type=int,default=195)
    p.add_argument('--output',type=Path,
                   default=HERE/'results/clic39_active_beta/raw.json')
    p.add_argument('--stream-dir',type=Path,
                   default=Path('/tmp/flexplus-clic39-active-beta'))
    args=p.parse_args()
    assert 1<=args.max_new_cases<=195

    import numpy as np
    import torch
    import torch.nn.functional as F
    from PIL import Image
    from model_io import load_model
    from reference_codec import ReferenceCodec,parse_container,tensor_hash
    from bitstream_benchmark import load_router
    from matched_routed_cpu import psnr
    from flexuf.config import FlexUFConfig
    from flexuf.cost import (_BLOCK_MACPX,_C,N_TRUNK_BLOCKS,SHARE_TRUNK,
                             frame_relative_cost,seam_repair_share)
    from flexuf.model import FlexUFIntra,load_flexuf_state
    import flexuf.backbone.coupling as coupling_module

    manifest_path=HERE/'results/clic39_active_beta/manifest.json'
    old_policy_path=HERE/'results/div2k_beta/locked_policy.json'
    new_policy_path=HERE/'results/div2k_beta/quality_floor/active_replicate_beta_locked_policy.json'
    frontier_path=HERE/'results/div2k_beta/quality_floor/active_replicate_beta_calibration24.json'
    source_dir=ROOT/'data/clic'
    upstream=Path('/tmp/flexplus-proof-clean/proof/early_exit_vs_released/.local/DCVC')
    extension=Path('/tmp/flexplus-proof-clean/proof/early_exit_vs_released/.local/entropy')
    released_path=PROOF/'artifacts/released_cvpr2026_image.pth.tar'
    e15_path=PROOF/'artifacts/e15_epoch15.pth.tar'
    router_path=PROOF/'artifacts/router_stem_qp.pth'
    calibration_path=PROOF/'router_calibration.json'
    sys.path.insert(0,str(upstream.resolve()))
    from src.utils.transforms import rgb2ycbcr_np
    torch.set_num_threads(1)
    torch.set_num_interop_threads(1)

    manifest=json.loads(manifest_path.read_text())
    old_policy=json.loads(old_policy_path.read_text())
    new_policy=json.loads(new_policy_path.read_text())
    frontier=json.loads(frontier_path.read_text())
    assert manifest['expected_cases']==195 and len(manifest['included'])==39
    assert len(manifest['excluded'])==2 and manifest['crop_size_wh']==[768,512]
    assert manifest['qps']==[0,16,32,48,63]
    assert manifest['script_sha256']==sha(HERE/'prepare_clic39_active_beta.py')
    assert frontier['complete'] and len(frontier['rows'])==frontier['expected_cases']==120
    assert frontier['provenance']['manifest_sha256']==sha(HERE/'results/div2k_beta/manifest.json')
    assert new_policy['source_sha256']['frontier']==sha(frontier_path)
    assert new_policy['source_sha256']['script']==sha(HERE/'select_active_replicate_beta.py')
    assert new_policy['source_sha256']['old_locked_policy']==sha(old_policy_path)
    assert set(old_policy['beta'])==set(new_policy['beta'])=={str(q) for q in manifest['qps']}
    critical=[Path(__file__),HERE/'active_canvas_replicate.py',
              ROOT/'flexuf/backbone/decoder.py',ROOT/'flexuf/backbone/coupling.py',
              ROOT/'flexuf/cost.py',ROOT/'flexuf/model.py',ROOT/'flexuf/config.py',
              ROOT/'flexuf/router/head2.py']
    provenance={
        'manifest_sha256':sha(manifest_path),'old_policy_sha256':sha(old_policy_path),
        'new_policy_sha256':sha(new_policy_path),'frontier_sha256':sha(frontier_path),
        'released_sha256':sha(released_path),'e15_sha256':sha(e15_path),
        'router_sha256':sha(router_path),'calibration_sha256':sha(calibration_path),
        'source_code_sha256':{str(x.relative_to(ROOT)):sha(x) for x in critical},
    }
    assert old_policy['manifest_sha256']==new_policy['source_sha256']['manifest']==sha(HERE/'results/div2k_beta/manifest.json')
    result={
        'scope':'Local CLIC39 eligible 768x512 crops x QP5, frozen DIV2K beta policies, same FUFREF2 stream/e15/router across deployed and active-replicate/no-repair arms; CPU FP32 untimed.',
        'expected_cases':195,'complete':False,
        'quality_unit':'Source YUV 6:1:1 PSNR and Delta444 dB against full-frame e15',
        'cost_unit':'Analytical full-synthesis convolution MAC saving %, excluding router/control/memory/entropy',
        'provenance':provenance,'rows':[],
    }
    if args.output.exists():
        old=json.loads(args.output.read_text())
        assert old['provenance']==provenance and old['expected_cases']==195
        result['rows']=old['rows']
    done={(r['image'],r['qp']) for r in result['rows']}
    assert len(done)==len(result['rows'])
    args.output.parent.mkdir(parents=True,exist_ok=True)
    args.stream_dir.mkdir(parents=True,exist_ok=True)

    released,info=load_model(released_path,12,upstream)
    codec=ReferenceCodec(released,info['sha256'],extension)
    checkpoint=torch.load(e15_path,map_location='cpu',weights_only=False)
    cfg=FlexUFConfig(**checkpoint['config'])
    assert cfg.rgb_patch==256 and cfg.tile_pad_mode=='replicate'
    assert not cfg.tile_coupling and cfg.seam_repair=='grid' and cfg.trunk_halo==0
    e15=FlexUFIntra(cfg).eval()
    load_flexuf_state(e15,checkpoint)
    shared={k:v for k,v in released.state_dict().items() if not k.startswith('dec.')}
    assert all(k in e15.state_dict() and torch.equal(v,e15.state_dict()[k])
               for k,v in shared.items())
    class RouterArgs:
        router=router_path
        calibration=calibration_path
    head,cost,ctc_beta,_=load_router(RouterArgs(),cfg,torch.device('cpu'))
    assert set(ctc_beta)==set(manifest['qps'])
    dec=e15.dec
    original_cfg,original_repair=dec.cfg,dec.seam_repair
    original_class=coupling_module.CanvasCoupler
    repair_share=seam_repair_share(cfg.seam_repair)
    per_block_extra=(9*_C/_BLOCK_MACPX)*(SHARE_TRUNK/N_TRUNK_BLOCKS)*(
        (cfg.feature_patch+2)**2/cfg.feature_patch**2-1)
    added=0
    try:
        with torch.inference_mode():
            for item in manifest['included']:
                source_path=source_dir/item['image']
                assert sha(source_path)==item['source_sha256']
                left,top,w,h=item['crop_xywh']
                with Image.open(source_path) as file:
                    rgb=np.asarray(file.convert('RGB').crop((left,top,left+w,top+h)),
                                   dtype=np.float32)/255
                source=torch.from_numpy(rgb2ycbcr_np(rgb)-.5).permute(2,0,1)[None].contiguous()
                assert source.shape==(1,3,512,768)
                for qp in manifest['qps']:
                    if (item['image'],qp) in done:continue
                    if added>=args.max_new_cases:break
                    stem_name=f'{Path(item["image"]).stem}_qp{qp}'
                    stream_path=args.stream_dir/f'{stem_name}.fufref2'
                    if stream_path.exists():
                        stream=stream_path.read_bytes()
                    else:
                        stream=codec.encode(source,qp,audit=False,reconstruct=False).stream
                        stream_path.write_bytes(stream)
                    meta=parse_container(stream,12,info['sha256'])
                    assert [meta['height'],meta['width'],meta['qp']]==[h,w,qp]
                    latent,quant_step,_=codec.decode_latent(stream)
                    released_image=released.dec(latent,quant_step)[:,:,:h,:w]
                    full=dec.forward_full(latent,quant_step)[:,:,:h,:w]
                    full_mse=float((source-full).square().mean())
                    assert full_mse>0
                    stem=dec.upsample(latent)
                    for group in dec.groups[:cfg.split_depth]:
                        stem=group(stem)
                    logits=head(stem,latent,latent,torch.tensor([qp],dtype=torch.int32),
                                cfg.feature_patch,cfg.latent_patch)
                    scores=F.log_softmax(logits[:,cfg.split_depth:],dim=1).double()
                    def route(beta):
                        return (scores-float(beta)*cost[cfg.split_depth:].double()[None,:]).argmax(1).cpu()+cfg.split_depth
                    old_route=route(old_policy['beta'][str(qp)])
                    new_route=route(new_policy['beta'][str(qp)])
                    assert old_route.numel()==new_route.numel()==6
                    dec.cfg=replace(cfg,sorted_tiles=True)
                    deployed=dec(latent,quant_step,exit_map=old_route)[:,:,:h,:w]
                    coupling_module.CanvasCoupler=ActiveCanvasReplicateCoupler
                    dec.cfg=replace(cfg,tile_coupling=True,sorted_tiles=False,trunk_halo=0)
                    dec.seam_repair=None
                    active_old=dec(latent,quant_step,exit_map=old_route)[:,:,:h,:w]
                    active_new=(active_old if torch.equal(old_route,new_route) else
                                dec(latent,quant_step,exit_map=new_route)[:,:,:h,:w])
                    coupling_module.CanvasCoupler=original_class
                    dec.cfg,dec.seam_repair=original_cfg,original_repair
                    outputs={'released_d12':released_image,'e15_full':full,
                             'deployed_old_beta':deployed,
                             'active_old_beta':active_old,
                             'active_new_beta':active_new}
                    quality={name:psnr(source,tensor)['yuv_6_1_1']
                             for name,tensor in outputs.items()}
                    delta={name:10*math.log10(float((source-tensor).square().mean())/full_mse)
                           for name,tensor in outputs.items() if name!='released_d12'}
                    def active_saving(route_map):
                        blocks=float(((route_map-cfg.split_depth+1)*cfg.blocks_per_exit).float().mean())
                        return 100*(1-frame_relative_cost(route_map,cfg)+repair_share-
                                    blocks*per_block_extra)
                    savings={
                        'deployed_old_beta':100*(1-frame_relative_cost(old_route,cfg)),
                        'active_old_beta':active_saving(old_route),
                        'active_new_beta':active_saving(new_route),
                    }
                    if torch.all(new_route==cfg.num_exits-1):
                        assert torch.allclose(active_new,full,rtol=1e-4,atol=1e-4)
                    result['rows'].append({
                        'image':item['image'],'qp':qp,'source_sha256':item['source_sha256'],
                        'crop_xywh':item['crop_xywh'],'stream_sha256':sha(stream_path),
                        'stream_bytes':len(stream),'rate_bpp':8*len(stream)/(w*h),
                        'latent_sha256':tensor_hash(latent),
                        'old_beta':old_policy['beta'][str(qp)],
                        'new_beta':new_policy['beta'][str(qp)],
                        'old_exit_map':old_route.tolist(),'new_exit_map':new_route.tolist(),
                        'yuv611_psnr_db':quality,'delta444_db':delta,
                        'conv_mac_saving_pct':savings,
                    })
                    added+=1
                    result['complete']=len(result['rows'])==195
                    args.output.write_text(json.dumps(result,indent=2)+'\n')
                    print(f'{stem_name}: active new loss {delta["active_new_beta"]:.5f} dB, saving {savings["active_new_beta"]:.2f}%',flush=True)
                if added>=args.max_new_cases:break
    finally:
        dec.cfg,dec.seam_repair=original_cfg,original_repair
        coupling_module.CanvasCoupler=original_class
    assert provenance['source_code_sha256']=={str(x.relative_to(ROOT)):sha(x)
                                               for x in critical}
    print(f"complete={result['complete']}; cases={len(result['rows'])}")


if __name__=='__main__':main()
