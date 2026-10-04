"""Separate-data beta calibration for the pinned e15 stem/QP router.

Prepare a deterministic DIV2K-valid split, capture real research bitstreams
and assembled-image quality on calibration images, lock a five-QP beta table,
then evaluate it on disjoint validation images. No Kodak data enters fitting.
All execution is CPU-only and untimed.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import math
from pathlib import Path
import sys

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
PROOF = ROOT/'proof/early_exit_vs_released'
sys.path[:0] = [str(ROOT), str(ROOT/'proof/depth_bitstream'), str(PROOF)]
QPS = [0, 16, 32, 48, 63]


def digest(path):
    value = hashlib.sha256()
    with Path(path).open('rb') as handle:
        for chunk in iter(lambda: handle.read(1024*1024), b''):
            value.update(chunk)
    return value.hexdigest()


def write_json(path, data):
    path.parent.mkdir(parents=True, exist_ok=True)
    temp = path.with_suffix(path.suffix+'.tmp')
    temp.write_text(json.dumps(data, indent=2)+'\n')
    temp.replace(path)


def prepare(args):
    from PIL import Image
    files = sorted(args.source_dir.glob('*.png'))
    if len(files) != 100:
        raise RuntimeError(f'Expected exactly 100 DIV2K validation PNGs, found {len(files)}')
    ordered = sorted(files, key=lambda p: hashlib.sha256(p.name.encode()).hexdigest())
    selected = {'calibration': ordered[:24], 'validation': ordered[24:48]}
    rows = {}
    for split, paths in selected.items():
        rows[split] = []
        for path in paths:
            with Image.open(path) as image:
                w,h = image.size
            if w < 768 or h < 512:
                raise RuntimeError(f'Source too small for locked crop: {path}')
            left,top = (w-768)//2,(h-512)//2
            rows[split].append({'image': path.name, 'source_sha256': digest(path),
                                'source_size_wh': [w,h], 'crop_xywh': [left,top,768,512]})
    calibration = json.loads(args.calibration.read_text())
    baseline = {int(r['qp']): float(r['beta']) for r in calibration['rows'] if r['kind']=='B'}
    if set(baseline) != set(QPS):
        raise RuntimeError('Missing pinned QP calibration')
    candidates = {str(q): sorted(set([float(x) for x in range(-50,65,5)] +
                                      [baseline[q], baseline[16]])) for q in QPS}
    manifest = {
        'schema': 1, 'scope': 'DIV2K validation original PNG -> deterministic centred 768x512 RGB crop -> centred YCbCr444 -> FUFREF2 -> e15 assembled output; CPU FP32 untimed',
        'selection': 'SHA256(filename) order of all 100 DIV2K valid PNGs; first 24 calibration, next 24 validation; fixed before codec evaluation',
        'qps': QPS, 'target_delta444_db_mean': .1,
        'candidate_rule': 'Global beta grid -50..60 inclusive in steps of 5, plus exact CTC baseline beta at each QP and exact QP16 beta; no Kodak-derived candidates',
        'candidates': candidates, 'ctc_beta': {str(q): baseline[q] for q in QPS},
        'calibration_checkpoint_sha256': calibration['router_checkpoint_sha256'],
        'calibration_json_sha256': digest(args.calibration),
        'released_sha256': digest(args.release), 'e15_sha256': digest(args.e15),
        'router_sha256': digest(args.router),
        'cost_model_sha256': digest(ROOT/'flexuf/cost.py'),
        'script_sha256': digest(Path(__file__)), 'rows': rows,
    }
    if manifest['router_sha256'] != manifest['calibration_checkpoint_sha256']:
        raise RuntimeError('Router differs from baseline beta checkpoint')
    if args.manifest.exists() and json.loads(args.manifest.read_text()) != manifest:
        raise RuntimeError('Prepared manifest differs from existing locked manifest')
    write_json(args.manifest, manifest)
    print(json.dumps({'manifest_sha256': digest(args.manifest),
                      'calibration_images': len(rows['calibration']),
                      'validation_images': len(rows['validation']),
                      'candidates_per_qp': {k: len(v) for k,v in candidates.items()}}, indent=2))


def capture(args):
    import numpy as np
    from PIL import Image
    import torch
    import torch.nn.functional as F
    from model_io import load_model
    from reference_codec import ReferenceCodec, parse_container, tensor_hash
    from bitstream_benchmark import load_router
    from flexuf.config import FlexUFConfig
    from flexuf.cost import frame_relative_cost
    from flexuf.model import FlexUFIntra, load_flexuf_state
    from flexuf.backbone.decoder import patchify, unpatchify
    from flexuf.kernels.planned_decoder import forward_from_stem_with_cpu_map

    manifest = json.loads(args.manifest.read_text())
    if manifest['schema'] != 1 or len(manifest['rows'][args.split]) != 24:
        raise RuntimeError('Invalid locked DIV2K manifest')
    manifest_sha = digest(args.manifest)
    if (manifest['released_sha256'] != digest(args.release) or
            manifest['e15_sha256'] != digest(args.e15) or
            manifest['router_sha256'] != digest(args.router)):
        raise RuntimeError('Checkpoint identity changed after preparation')
    locked = None
    if args.split == 'validation':
        if args.policy is None:
            raise RuntimeError('Validation requires a previously locked policy')
        locked = json.loads(args.policy.read_text())
        if locked['manifest_sha256'] != manifest_sha:
            raise RuntimeError('Policy was fitted on another manifest')
    elif args.policy is not None:
        raise RuntimeError('Calibration capture must not consume a fitted policy')

    sys.path.insert(0, str(args.upstream.resolve()))
    from src.utils.transforms import rgb2ycbcr_np
    torch.set_num_threads(1)
    torch.set_num_interop_threads(1)
    released, info = load_model(args.release, 12, args.upstream)
    codec = ReferenceCodec(released, info['sha256'], args.extension)
    ckpt = torch.load(args.e15, map_location='cpu', weights_only=False)
    cfg = FlexUFConfig(**ckpt['config'])
    e15 = FlexUFIntra(cfg).eval()
    load_flexuf_state(e15, ckpt)
    shared = {k:v for k,v in released.state_dict().items() if not k.startswith('dec.')}
    if any(k not in e15.state_dict() or not torch.equal(v,e15.state_dict()[k])
           for k,v in shared.items()):
        raise RuntimeError('Released/e15 analysis and entropy weights differ')
    from dataclasses import replace
    e15.dec.cfg = replace(cfg, sorted_tiles=True)
    head,cost,baseline_beta,_ = load_router(args,cfg,torch.device('cpu'))
    if any(baseline_beta[q] != manifest['ctc_beta'][str(q)] for q in QPS):
        raise RuntimeError('Baseline beta changed')
    args.out_dir.mkdir(parents=True, exist_ok=True)

    def cached_features(stem):
        decoder = e15.dec
        j,K = cfg.split_depth,cfg.num_exits
        tiles,nh,nw = patchify(stem,cfg.feature_patch)
        undo = None
        if cfg.tile_pad_mode != 'zeros':
            undo = decoder._set_tile_padding(cfg.tile_pad_mode,j)
        try:
            work=tiles
            feats=[]
            for g in range(j,K):
                work=decoder.groups[g](work)
                feats.append(decoder._at_exit(work,g).clone())
        finally:
            if undo is not None:
                undo()
            elif cfg.tile_pad_mode != 'zeros':
                decoder._set_tile_padding('zeros',j)
        return torch.stack(feats),nh,nw

    def render_cached(features,nh,nw,route,q):
        n=route.numel()
        selected=features[route-cfg.split_depth,torch.arange(n)]
        image=unpatchify(selected,nh,nw,batch=1)
        if e15.dec.seam_repair is not None:
            image=e15.dec.seam_repair(image)
        return e15.dec._apply_head(image,q)

    for item in manifest['rows'][args.split]:
        path=args.source_dir/item['image']
        if digest(path)!=item['source_sha256']:
            raise RuntimeError(f'Source hash changed: {path}')
        left,top,w,h=item['crop_xywh']
        with Image.open(path) as image:
            rgb=np.asarray(image.convert('RGB').crop((left,top,left+w,top+h)),dtype=np.float32)/255
        source=torch.from_numpy(rgb2ycbcr_np(rgb)-.5).permute(2,0,1)[None].contiguous()
        for qp in QPS:
            stem_name=f'{Path(item["image"]).stem}_qp{qp}'
            out=args.out_dir/f'{stem_name}.json'
            if out.exists():
                old=json.loads(out.read_text())
                if (old['manifest_sha256']==manifest_sha and old['split']==args.split and
                        old.get('policy_sha256')==(digest(args.policy) if locked else None)):
                    print(f'skip {stem_name}',flush=True)
                    continue
                raise RuntimeError(f'Conflicting result: {out}')
            stream_path=args.out_dir/f'{stem_name}.fufref2'
            if stream_path.exists():
                stream=stream_path.read_bytes()
            else:
                stream=codec.encode(source,qp,audit=False,reconstruct=False).stream
                stream_path.write_bytes(stream)
            meta=parse_container(stream,12,info['sha256'])
            if [meta['height'],meta['width'],meta['qp']] != [h,w,qp]:
                raise RuntimeError('Stream geometry/QP differs from crop')
            y,q,_=codec.decode_latent(stream)
            with torch.inference_mode():
                reference=e15.dec.forward_full(y,q)[:,:,:h,:w]
                stem=e15.dec.upsample(y)
                for group in e15.dec.groups[:cfg.split_depth]:
                    stem=group(stem)
                qp_tensor=torch.tensor([qp],dtype=torch.int32)
                logits=head(stem,y,y,qp_tensor,cfg.feature_patch,cfg.latent_patch)
                scores=F.log_softmax(logits[:,cfg.split_depth:],dim=1).double()
                features,nh,nw=cached_features(stem)
                dref=float((source-reference).square().mean())
                if dref<=0:
                    raise RuntimeError('Nonpositive e15 reference distortion')
                betas=(manifest['candidates'][str(qp)] if locked is None else
                       [float(locked['beta'][str(qp)])])
                unique={}
                records=[]
                for beta in betas:
                    route=(scores-beta*cost[cfg.split_depth:].double()[None,:]).argmax(1).cpu()+cfg.split_depth
                    key=tuple(route.tolist())
                    if key not in unique:
                        image=render_cached(features,nh,nw,route,q)[:,:,:h,:w]
                        mse=float((source-image).square().mean())
                        unique[key]=10*math.log10(mse/dref)
                    records.append({'beta':float(beta),'delta444_db':unique[key],
                                    'mac_saved_pct':100*(1-frame_relative_cost(route,cfg,'head')),
                                    'exit_map':list(key)})
                check_route=(scores-baseline_beta[qp]*cost[cfg.split_depth:].double()[None,:]).argmax(1).cpu()+cfg.split_depth
                if args.split=='calibration' and item is manifest['rows'][args.split][0]:
                    direct=forward_from_stem_with_cpu_map(e15.dec,stem,q,check_route)[:,:,:h,:w]
                    cached=render_cached(features,nh,nw,check_route,q)[:,:,:h,:w]
                    max_abs=float((direct-cached).abs().max())
                    if max_abs>2e-5:
                        raise RuntimeError(f'Cached suffix differs from direct decode by {max_abs}')
                else:
                    max_abs=None
            row={'schema':1,'split':args.split,'image':item['image'],'qp':qp,
                 'manifest_sha256':manifest_sha,'policy_sha256':digest(args.policy) if locked else None,
                 'source_sha256':item['source_sha256'],'crop_xywh':item['crop_xywh'],
                 'stream_sha256':hashlib.sha256(stream).hexdigest(),
                 'latent_sha256':tensor_hash(y),'e15_full_mse444':dref,
                 'cached_vs_direct_max_abs_check':max_abs,
                 'unique_exit_maps':len(unique),'candidates':records,
                 'latency_status':'unmeasured; CPU analytical/quality calibration only'}
            write_json(out,row)
            print(json.dumps({'case':stem_name,'split':args.split,'unique_maps':len(unique),
                              'candidate_count':len(records)}),flush=True)


def fit(args):
    manifest=json.loads(args.manifest.read_text())
    manifest_sha=digest(args.manifest)
    chosen={}
    tables={}
    source_hashes={}
    for qp in QPS:
        rows=[]
        for item in manifest['rows']['calibration']:
            path=args.out_dir/f'{Path(item["image"]).stem}_qp{qp}.json'
            row=json.loads(path.read_text())
            if (row['manifest_sha256']!=manifest_sha or row['split']!='calibration' or
                    row['image']!=item['image'] or row['qp']!=qp or
                    row['source_sha256']!=item['source_sha256']):
                raise RuntimeError(f'Calibration result provenance mismatch: {path}')
            rows.append(row)
            source_hashes[path.name]=digest(path)
        if len(rows)!=24:
            raise RuntimeError('Incomplete calibration split')
        grid=manifest['candidates'][str(qp)]
        table=[]
        for j,beta in enumerate(grid):
            if any(row['candidates'][j]['beta']!=beta for row in rows):
                raise RuntimeError('Candidate grid/order mismatch')
            losses=[row['candidates'][j]['delta444_db'] for row in rows]
            savings=[row['candidates'][j]['mac_saved_pct'] for row in rows]
            table.append({'beta':beta,'mean_delta444_db':sum(losses)/24,
                          'mean_mac_saved_pct':sum(savings)/24,
                          'over_0p1_count':sum(x>.1 for x in losses)})
        feasible=[r for r in table if r['mean_delta444_db']<=manifest['target_delta444_db_mean']]
        if not feasible:
            raise RuntimeError(f'No feasible beta on predeclared grid at QP{qp}')
        best=max(feasible,key=lambda r:(r['mean_mac_saved_pct'],-r['mean_delta444_db'],-r['beta']))
        chosen[str(qp)]=best['beta']
        tables[str(qp)]=table
    policy={'schema':1,'scope':'Locked beta chosen on DIV2K calibration images only; 24 disjoint DIV2K validation images and Kodak never consulted during fit',
            'manifest_sha256':manifest_sha,'target_delta444_db_mean':manifest['target_delta444_db_mean'],
            'selection_rule':'Among predeclared grid candidates with calibration mean Delta444 <=0.1 dB, maximize mean analytical synthesis MAC saving; ties minimize mean loss then beta',
            'beta':chosen,'calibration_tables':tables,'calibration_result_sha256':source_hashes}
    if args.policy.exists() and json.loads(args.policy.read_text())!=policy:
        raise RuntimeError('Existing locked policy differs from recalculation')
    write_json(args.policy,policy)
    print(json.dumps({'policy_sha256':digest(args.policy),'beta':chosen,
                      'selected_calibration':{q:next(row for row in tables[q] if row['beta']==chosen[q]) for q in chosen}},indent=2))


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('action',choices=['prepare','capture','fit'])
    parser.add_argument('--source-dir',type=Path,default=ROOT/'data/DIV2K_valid_HR')
    parser.add_argument('--manifest',type=Path,required=True)
    parser.add_argument('--out-dir',type=Path)
    parser.add_argument('--split',choices=['calibration','validation'])
    parser.add_argument('--policy',type=Path)
    parser.add_argument('--upstream',type=Path,default=PROOF/'.local/DCVC')
    parser.add_argument('--extension',type=Path,default=PROOF/'.local/entropy')
    parser.add_argument('--release',type=Path,default=PROOF/'artifacts/released_cvpr2026_image.pth.tar')
    parser.add_argument('--e15',type=Path,default=PROOF/'artifacts/e15_epoch15.pth.tar')
    parser.add_argument('--router',type=Path,default=PROOF/'artifacts/router_stem_qp.pth')
    parser.add_argument('--calibration',type=Path,default=PROOF/'router_calibration.json')
    args=parser.parse_args()
    if args.action=='prepare':
        prepare(args)
    elif args.action=='capture':
        if args.out_dir is None or args.split is None:
            parser.error('capture requires --out-dir and --split')
        capture(args)
    else:
        if args.out_dir is None or args.policy is None:
            parser.error('fit requires --out-dir and --policy')
        fit(args)


if __name__=='__main__':
    main()
