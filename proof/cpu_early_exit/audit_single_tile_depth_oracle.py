"""Source-informed tile-subset +1 exit ablation for exact-context failures.

The fixed 120-case DIV2K validation maps remain unchanged except for one
candidate tile at a time. Ground-truth source error chooses the best tile,
so this is a nondeployable spatial headroom audit, not a trained router.
"""
from __future__ import annotations

import argparse
import itertools
import json
import math
import sys
from pathlib import Path

HERE=Path(__file__).resolve().parent
ROOT=HERE.parents[1]
PROOF=ROOT/'proof/early_exit_vs_released'
sys.path[:0]=[str(ROOT),str(ROOT/'proof/depth_bitstream'),str(PROOF)]
from audit_exact_context_pilot import sha
from analyze_ideal_context_cohort import ideal_cost


def tile_core(dec,cfg,stem,tile,mode):
    _,_,h,w=stem.shape
    side=cfg.feature_patch
    nw=w//side
    top,left=(tile//nw)*side,(tile%nw)*side
    halo=(mode-cfg.split_depth+1)*cfg.blocks_per_exit
    t,l=max(0,top-halo),max(0,left-halo)
    b,r=min(h,top+side+halo),min(w,left+side+halo)
    work=stem[:,:,t:b,l:r].contiguous()
    for group in range(cfg.split_depth,mode+1):work=dec.groups[group](work)
    work=dec._at_exit(work,mode)
    return work[:,:,top-t:top-t+side,left-l:left-l+side].contiguous()


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--base',type=Path,default=HERE/'results/div2k_beta/quality_floor/exact_context_validation24.json')
    p.add_argument('--cases',type=Path,default=Path('/tmp/flexplus-div2k-beta-validation'))
    p.add_argument('--source-dir',type=Path,default=ROOT/'data/DIV2K_valid_HR')
    p.add_argument('--upstream',type=Path,default=Path('/tmp/flexplus-proof-clean/proof/early_exit_vs_released/.local/DCVC'))
    p.add_argument('--extension',type=Path,default=Path('/tmp/flexplus-proof-clean/proof/early_exit_vs_released/.local/entropy'))
    p.add_argument('--release',type=Path,default=PROOF/'artifacts/released_cvpr2026_image.pth.tar')
    p.add_argument('--e15',type=Path,default=PROOF/'artifacts/e15_epoch15.pth.tar')
    p.add_argument('--output',type=Path,default=HERE/'results/div2k_beta/quality_floor/single_tile_oracle_validation24.json')
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

    torch.set_num_threads(1);torch.set_num_interop_threads(1)
    base=json.loads(args.base.read_text())
    if len(base['rows'])!=120 or base['provenance']['release_sha256']!=sha(args.release) or base['provenance']['e15_sha256']!=sha(args.e15):
        raise RuntimeError('Full exact-context validation baseline required')
    failures=[r for r in base['rows'] if r['exact_context_delta444_db']>.1]
    if len(failures)!=24:raise RuntimeError('Unexpected failure count')
    provenance={'base_sha256':sha(args.base),'release_sha256':sha(args.release),'e15_sha256':sha(args.e15)}
    result={'schema':1,'scope':'Source-informed single-tile +1 exact-context oracle, DIV2K validation; no deployment or runtime',
            'provenance':provenance,'rows':[]}
    if args.output.exists():
        old=json.loads(args.output.read_text())
        if old['provenance']!=provenance:raise RuntimeError('Existing output mismatch')
        for row in old['rows']:
            if 'minimum_upgraded_tiles' in row:
                result['rows'].append(row)
            elif row['rescued_by_one_tile']:
                row['minimum_upgraded_tiles']=1
                row['best_subset']={**row['best'],'tiles':[row['best']['tile']]}
                result['rows'].append(row)
            elif row.get('rescued_by_two_tiles'):
                row['minimum_upgraded_tiles']=2
                row['best_subset']=row['best_pair']
                result['rows'].append(row)
        for r in result['rows']:
            r.setdefault('best_pair',None)
            r.setdefault('rescued_by_two_tiles',False)
    done={(r['image'],r['qp']) for r in result['rows']}
    released,info=load_model(args.release,12,args.upstream)
    codec=ReferenceCodec(released,info['sha256'],args.extension)
    ckpt=torch.load(args.e15,map_location='cpu',weights_only=False)
    cfg=FlexUFConfig(**ckpt['config'])
    model=FlexUFIntra(cfg).eval();load_flexuf_state(model,ckpt)
    dec=model.dec
    args.output.parent.mkdir(parents=True,exist_ok=True)
    with torch.inference_mode():
        for row in failures:
            image,qp=row['image'],row['qp']
            if (image,qp) in done:continue
            stemname=f'{Path(image).stem}_qp{qp}'
            case_path=args.cases/f'{stemname}.json'
            stream_path=args.cases/f'{stemname}.fufref2'
            source_path=args.source_dir/image
            case=json.loads(case_path.read_text())
            if (sha(case_path)!=row['case_sha256'] or sha(stream_path)!=row['stream_sha256'] or
                sha(source_path)!=case['source_sha256']):
                raise RuntimeError(f'Input hash mismatch: {stemname}')
            x,y,w,h=case['crop_xywh']
            with Image.open(source_path) as im:
                rgb=np.asarray(im.convert('RGB').crop((x,y,x+w,y+h)),dtype=np.float32)/255
            source=torch.from_numpy(rgb2ycbcr_np(rgb)-.5).permute(2,0,1)[None].contiguous()
            latent,q,_=codec.decode_latent(stream_path.read_bytes())
            reference=dec.forward_full(latent,q)
            ref_mse=float((source-reference[:,:,:h,:w]).square().mean())
            if not math.isclose(ref_mse,case['e15_full_mse444'],abs_tol=1e-11):
                raise RuntimeError(f'Reference MSE mismatch: {stemname}')
            stem=dec.upsample(latent)
            for group in dec.groups[:cfg.split_depth]:stem=group(stem)
            route=row['exit_map'];side=cfg.feature_patch;nw=stem.shape[-1]//side
            canvas=torch.empty_like(stem)
            positions=[];cores=[]
            for tile,mode in enumerate(route):
                top,left=(tile//nw)*side,(tile%nw)*side
                core=tile_core(dec,cfg,stem,tile,mode)
                canvas[:,:,top:top+side,left:left+side]=core
                positions.append((top,left));cores.append(core)
            def loss_db():
                output=dec._apply_head(canvas,q)
                mse=float((source-output[:,:,:h,:w]).square().mean())
                return 10*math.log10(mse/ref_mse)
            base_db=loss_db()
            if abs(base_db-row['exact_context_delta444_db'])>1e-4:
                raise RuntimeError(f'Cached exact/base mismatch: {stemname}: {base_db}')
            candidates=[]
            upgraded={}
            for tile,mode in enumerate(route):
                if mode==cfg.num_exits-1:continue
                top,left=positions[tile]
                newcore=tile_core(dec,cfg,stem,tile,mode+1)
                upgraded[tile]=newcore
                canvas[:,:,top:top+side,left:left+side]=newcore
                delta=loss_db()
                canvas[:,:,top:top+side,left:left+side]=cores[tile]
                newroute=list(route);newroute[tile]=mode+1
                cost,_=ideal_cost(newroute,cfg)
                candidates.append({'tile':tile,'exit_map':newroute,
                                   'exact_context_delta444_db':delta,
                                   'ideal_shared_saving_pct':100*(1-cost)})
            rescued=[c for c in candidates if c['exact_context_delta444_db']<=.1]
            best=(max(rescued,key=lambda c:c['ideal_shared_saving_pct']) if rescued
                  else min(candidates,key=lambda c:c['exact_context_delta444_db']))
            pair_candidates=[]
            if not rescued:
                for first,second in itertools.combinations(upgraded,2):
                    for tile in (first,second):
                        top,left=positions[tile]
                        canvas[:,:,top:top+side,left:left+side]=upgraded[tile]
                    delta=loss_db()
                    for tile in (first,second):
                        top,left=positions[tile]
                        canvas[:,:,top:top+side,left:left+side]=cores[tile]
                    newroute=list(route)
                    newroute[first]+=1;newroute[second]+=1
                    cost,_=ideal_cost(newroute,cfg)
                    pair_candidates.append({'tiles':[first,second],'exit_map':newroute,
                                            'exact_context_delta444_db':delta,
                                            'ideal_shared_saving_pct':100*(1-cost)})
            rescued_pairs=[c for c in pair_candidates if c['exact_context_delta444_db']<=.1]
            best_pair=(max(rescued_pairs,key=lambda c:c['ideal_shared_saving_pct']) if rescued_pairs
                       else min(pair_candidates,key=lambda c:c['exact_context_delta444_db'])
                       if pair_candidates else None)
            if rescued:
                minimum_tiles,best_subset=1,{**best,'tiles':[best['tile']]}
            elif rescued_pairs:
                minimum_tiles,best_subset=2,best_pair
            else:
                minimum_tiles,best_subset=None,None
                for size in range(3,len(upgraded)+1):
                    successes=[]
                    for tiles in itertools.combinations(upgraded,size):
                        for tile in tiles:
                            top,left=positions[tile]
                            canvas[:,:,top:top+side,left:left+side]=upgraded[tile]
                        delta=loss_db()
                        for tile in tiles:
                            top,left=positions[tile]
                            canvas[:,:,top:top+side,left:left+side]=cores[tile]
                        if delta<=.1:
                            newroute=list(route)
                            for tile in tiles:newroute[tile]+=1
                            cost,_=ideal_cost(newroute,cfg)
                            successes.append({'tiles':list(tiles),'exit_map':newroute,
                                              'exact_context_delta444_db':delta,
                                              'ideal_shared_saving_pct':100*(1-cost)})
                    if successes:
                        minimum_tiles,best_subset=size,max(successes,key=lambda c:c['ideal_shared_saving_pct'])
                        break
            result['rows'].append({'image':image,'qp':qp,'base_exit_map':route,
                                   'base_exact_context_delta444_db':base_db,
                                   'base_ideal_shared_saving_pct':row['ideal_shared_saving_pct'],
                                   'rescued_by_one_tile':bool(rescued),
                                   'best':best,'candidates':candidates,
                                   'rescued_by_two_tiles':bool(rescued_pairs),
                                   'best_pair':best_pair,'pair_candidates':pair_candidates,
                                   'minimum_upgraded_tiles':minimum_tiles,
                                   'best_subset':best_subset})
            args.output.write_text(json.dumps(result,indent=2)+'\n')
            print(f'{stemname}: minimum +1 tiles={minimum_tiles}, '
                  f'best subset saving={best_subset["ideal_shared_saving_pct"]:.1f}%'
                  if best_subset else f'{stemname}: no feasible +1 subset',flush=True)


if __name__=='__main__':main()
