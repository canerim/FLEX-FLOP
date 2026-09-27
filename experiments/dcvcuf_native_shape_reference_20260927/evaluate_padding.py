"""Paired halo32 padding-policy control after the complete CPU-pad64 study.

Only the non64-aligned288 windows need new coding. Full512, core256 and halo64
320 outputs already have identical geometry; aligned format parity is verified.
"""
import json
import math
import os
from pathlib import Path
import subprocess
import sys
import time
from datetime import datetime,timezone
os.environ['CUDA_VISIBLE_DEVICES']=''
import numpy as np
import torch
from prepare import HERE,ROOT,OUT as SOURCE,sha

RESULT=ROOT/'patch_native_shape_epoch020'
LEGACY=ROOT/'patch_control_epoch020'


def utc():return datetime.now(timezone.utc).isoformat()


def atomic(path,value):
    temp=path.with_suffix(path.suffix+'.tmp');temp.write_text(json.dumps(value,indent=2,allow_nan=False)+'\n');temp.replace(path)


def receive(worker):
    line=worker.stdout.readline()
    if not line:raise RuntimeError('Isolated decoder exited')
    return json.loads(line)


def metrics(rec,target):
    error=(rec-target).square().mean(1)[0];mse=float(error.mean())
    r={'mse_rgb':mse,'psnr_rgb':-10*math.log10(max(mse,1e-12))}
    for radius in (4,16):
        seam=torch.zeros((512,512),dtype=torch.bool)
        seam[256-radius:256+radius,:]=True;seam[:,256-radius:256+radius]=True
        r[f'seam_r{radius}_mse']=float(error[seam].mean());r[f'interior_r{radius}_mse']=float(error[~seam].mean())
        r[f'seam_r{radius}_pixels']=int(seam.sum())
    return r


def decode(worker,stream,output):
    worker.stdin.write(json.dumps({'stream':str(stream),'output':str(output)})+'\n');worker.stdin.flush()
    r=receive(worker)
    if not r['ok'] or not r['source_analysis_disabled']:raise ValueError('Decoder isolation failed')
    x=torch.from_numpy(np.load(output,allow_pickle=False));output.unlink();return x,r['diagnostics']


def main():
    torch.set_num_threads(2);script_sha=sha(__file__)
    proof_path=ROOT/'native_shape_verification_epoch020/verification.json';proof=json.loads(proof_path.read_text())
    if not proof['all_passed'] or len(proof['cases'])!=45:raise ValueError('Native-shape engineering proof incomplete')
    for name,digest in proof['derived_source_sha256'].items():
        if sha(SOURCE/name)!=digest:raise ValueError('Derived source changed')
    RESULT.mkdir(parents=True,exist_ok=True)
    progress={'state':'waiting_legacy_patch_control','started_utc':utc(),'pid':os.getpid(),'completed':0,'total':240,
              'script_sha256':script_sha,'scope':'CPU padding-policy ablation only; not CUDA runtime or wire compatibility'}
    atomic(RESULT/'progress.json',progress)
    worker=None
    try:
        while True:
            p=LEGACY/'progress.json'
            if p.exists():
                state=json.loads(p.read_text())
                if state['state']=='failed':raise RuntimeError('Legacy patch control failed')
                if state['state']=='complete':
                    if state['completed']!=240:raise ValueError('Incomplete legacy patch cohort')
                    break
                progress['legacy_completed']=state['completed']
            progress['updated_utc']=utc();atomic(RESULT/'progress.json',progress);time.sleep(30)
        if sha(__file__)!=script_sha:raise ValueError('Queued evaluation script changed')
        sys.path.insert(0,str(SOURCE))
        from model_io import load_model,ROOT as TRAINROOT
        from reference_codec import ReferenceCodec,MAGIC
        def checkpoint(d):return TRAINROOT/('reference_d12/released_cvpr2026_image.pth.tar' if d==12 else f'runs/d{d}/weights_epoch020.pt')
        legacy_manifest=json.loads((LEGACY/'manifest.json').read_text())
        if legacy_manifest['halos']!=[0,32,64]:raise ValueError('Unexpected primary patch protocol')
        crop_manifest=ROOT/'div2k100_center512_rgb/manifest.json'
        crops={c['image']:c for c in json.loads(crop_manifest.read_text())['images']}
        manifest={'format':'FUFREF2','scope':'Paired CPU FP32 padding-policy control: image-pad16 plus latent-pad4 versusFUFREF1 image-pad64; no native CUDA numerical or timing claim',
            'images':legacy_manifest['images'],'depths':legacy_manifest['depths'],'qps':legacy_manifest['qps'],
            'halo':32,'context_side':288,'image_padding_multiple':16,'hyperanalysis_latent_padding_multiple':4,
            'script_sha256':script_sha,'proof_sha256':sha(proof_path),'source_sha256':proof['derived_source_sha256'],
            'legacy_manifest_sha256':sha(LEGACY/'manifest.json'),'crop_manifest_sha256':sha(crop_manifest),
            'selection':'Identical predeclared16-image/5-QP/depth2,6,12 cohort; no outcome selection',
            'reuse':'Full512, halo0/core256 and halo64/context320 geometry is unchanged. Primary stream payloads and metrics remain separate records; full512 is independently decoded after only version-magic rewrapping.'}
        if (RESULT/'manifest.json').exists() and json.loads((RESULT/'manifest.json').read_text())!=manifest:raise ValueError('Output provenance differs')
        atomic(RESULT/'manifest.json',manifest);start=time.perf_counter();progress['state']='running'
        for depth in manifest['depths']:
            net,info=load_model(checkpoint(depth),depth)
            if info['sha256']!=proof['models'][str(depth)]['sha256']:raise ValueError('Wrong checkpoint')
            codec=ReferenceCodec(net,info['sha256'],ROOT/'reference_entropy_v1')
            from src.utils.transforms import rgb2ycbcr_np,ycbcr2rgb
            folder=RESULT/f'd{depth}';(folder/'cases').mkdir(parents=True,exist_ok=True);(folder/'streams').mkdir(exist_ok=True)
            log=(folder/'decoder.stderr.log').open('a')
            worker=subprocess.Popen([sys.executable,str(SOURCE/'decode_worker.py'),'--checkpoint',str(checkpoint(depth)),
                '--depth',str(depth)],stdin=subprocess.PIPE,stdout=subprocess.PIPE,stderr=log,text=True,bufsize=1)
            if not receive(worker)['ready']:raise ValueError('Decoder not ready')
            for image in manifest['images']:
                c=crops[image]
                if sha(c['crop_path'])!=c['crop_file_sha256']:raise ValueError('Source crop changed')
                rgb=np.load(c['crop_path'],allow_pickle=False).astype(np.float32)/255
                target=torch.from_numpy(rgb).permute(2,0,1)[None].contiguous()
                x=torch.from_numpy(rgb2ycbcr_np(rgb)-.5).permute(2,0,1)[None].contiguous()
                for qp in manifest['qps']:
                    name=f'{Path(image).stem}_qp{qp:02d}';path=folder/'cases'/(name+'.json')
                    primary_path=LEGACY/f'd{depth}/cases/{name}.json';primary=json.loads(primary_path.read_text())
                    if path.exists():
                        case=json.loads(path.read_text())
                        if case['primary_case_sha256']!=sha(primary_path) or case['checkpoint_sha256']!=info['sha256']:raise ValueError('Existing case changed')
                        for t in case['native_shape_halo32']['tiles']:
                            if sha(t['stream_path'])!=t['stream_sha256']:raise ValueError('Existing stream changed')
                    else:
                        full_path=ROOT/f'div2k100_reference_epoch020/d{depth}/streams/{name}.fufref'
                        full_bytes=full_path.read_bytes()
                        if full_bytes[:8]!=b'FUFREF1\x00':raise ValueError('Wrong full-frame reference format')
                        full2=folder/'streams'/(name+'_full_rewrapped.fufref2');full2.write_bytes(MAGIC+full_bytes[8:])
                        rec,diag=decode(worker,full2,folder/'decoded.npy')
                        full_metrics=metrics(ycbcr2rgb(rec.clamp(-.5,.5)+.5,clamp=True),target)
                        if abs(full_metrics['psnr_rgb']-primary['full']['psnr_rgb'])>1e-8:raise AssertionError('Aligned full-frame parity failed')
                        canvas=torch.empty_like(x);coverage=torch.zeros((512,512),dtype=torch.int32);tiles=[]
                        for row,y in enumerate((0,256)):
                            for col,z in enumerate((0,256)):
                                top,left=max(0,y-32),max(0,z-32);bottom,right=min(512,y+288),min(512,z+288)
                                encoded=codec.encode(x[:,:,top:bottom,left:right].contiguous(),qp)
                                stream=folder/'streams'/f'{name}_r{row}c{col}.fufref2';stream.write_bytes(encoded.stream)
                                decoded,diag=decode(worker,stream,folder/'decoded.npy')
                                if not torch.equal(decoded,encoded.reconstruction):raise AssertionError('Isolated patch decode differs')
                                for key in ('z_hat_sha256','y_hat_sha256'):
                                    if diag[key]!=encoded.diagnostics[key]:raise AssertionError('Patch latent differs')
                                for a,b in zip(diag['stages'],encoded.diagnostics['stages']):
                                    for key in ('symbols_sha256','indexes_sha256'):
                                        if a[key]!=b[key]:raise AssertionError('Patch symbol/index trace differs')
                                oy,ox=y-top,z-left;canvas[:,:,y:y+256,z:z+256]=decoded[:,:,oy:oy+256,ox:ox+256]
                                coverage[y:y+256,z:z+256]+=1
                                tiles.append({'row':row,'col':col,'top':top,'left':left,'bottom':bottom,'right':right,
                                    'stream_path':str(stream),'stream_sha256':sha(stream),'exact_isolated_decode':True,**encoded.diagnostics})
                        if not torch.equal(coverage,torch.ones_like(coverage)):raise AssertionError('Invalid stitching')
                        assembled=ycbcr2rgb(canvas.clamp(-.5,.5)+.5,clamp=True);result=metrics(assembled,target)
                        payload=sum(t['payload_bytes'] for t in tiles);container=sum(t['container_bytes'] for t in tiles)
                        result.update(tiles=tiles,payload_bytes=payload,container_bytes=container,
                            payload_bpp=payload*8/512**2,container_bpp=container*8/512**2,
                            estimated_bpp=sum(t['estimated_bits_y']+t['estimated_bits_z'] for t in tiles)/512**2)
                        if qp==32 and image in ('0801.png','0880.png'):
                            from PIL import Image
                            Image.fromarray((assembled[0].permute(1,2,0).numpy()*255).round().clip(0,255).astype(np.uint8)).save(folder/(name+'_native_shape_h32.png'))
                        case={'depth':depth,'image':image,'qp':qp,'checkpoint_sha256':info['sha256'],'crop_sha256':c['crop_file_sha256'],
                            'primary_case_sha256':sha(primary_path),'full_rewrapped_stream_sha256':sha(full2),
                            'full_original_stream_sha256':sha(full_path),'full_psnr_matches_primary':True,
                            'native_shape_halo32':result,'finished_utc':utc()}
                        atomic(path,case)
                    progress.update(completed=progress['completed']+1,depth=depth,image=image,qp=qp,updated_utc=utc(),elapsed_seconds=time.perf_counter()-start)
                    atomic(RESULT/'progress.json',progress);print(json.dumps(progress),flush=True)
            worker.stdin.write('{"stop":true}\n');worker.stdin.flush();worker.wait(timeout=30)
            if worker.returncode:raise RuntimeError('Decoder worker failed')
            log.close();worker=None;del net,codec
        if progress['completed']!=240 or sha(__file__)!=script_sha or torch.cuda.is_initialized():raise AssertionError('Run integrity failure')
        for name,digest in proof['derived_source_sha256'].items():
            if sha(SOURCE/name)!=digest:raise ValueError('Derived source changed during run')
        progress.update(state='complete',finished_utc=utc());atomic(RESULT/'progress.json',progress)
    except BaseException as exc:
        progress.update(state='failed',error=repr(exc),updated_utc=utc());atomic(RESULT/'progress.json',progress);raise
    finally:
        if worker is not None and worker.poll() is None:worker.terminate();worker.wait(timeout=30)


if __name__=='__main__':main()
