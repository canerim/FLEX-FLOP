"""Fixed-depth patching control with real bytes and isolated causal decoding.

Compares a 512 crop, four independent 256 cores, and 32/64-pixel context halos.
This is not adaptive routing, not GPU timing, and not final trained-model RD.
"""
from __future__ import annotations
import argparse
import json
import math
import os
from pathlib import Path
import subprocess
import sys
import time
os.environ['CUDA_VISIBLE_DEVICES']=''
os.environ.setdefault('OMP_NUM_THREADS','2')
import numpy as np
from PIL import Image
import torch

HERE=Path(__file__).resolve().parent
REFERENCE=HERE.parent/'dcvcuf_reference_20260927'
sys.path.insert(0,str(REFERENCE))
from model_io import ROOT,load_model,file_sha
from reference_codec import ReferenceCodec
from evaluate_validation import atomic,checkpoint,receive,utc

DEPTHS=(2,6,12)
QPS=(0,16,32,48,63)
IMAGES=[f'{i:04d}.png' for i in np.rint(np.linspace(801,900,16)).astype(int)]


def metrics(rec,target):
    squared=(rec-target).square()
    error=squared.mean(dim=1)[0]
    # Match the frozen whole-frame evaluator's FP32 reduction order exactly.
    mse=float(squared.mean())
    result={'mse_rgb':mse,'psnr_rgb':-10*math.log10(max(mse,1e-12))}
    for radius in (4,16):
        seam=torch.zeros((512,512),dtype=torch.bool)
        seam[256-radius:256+radius,:]=True;seam[:,256-radius:256+radius]=True
        result[f'seam_r{radius}_mse']=float(error[seam].mean())
        result[f'interior_r{radius}_mse']=float(error[~seam].mean())
        result[f'seam_r{radius}_pixels']=int(seam.sum())
    return result


def isolated(worker,stream_path,output_path):
    worker.stdin.write(json.dumps({'stream':str(stream_path),'output':str(output_path)})+'\n');worker.stdin.flush()
    response=receive(worker)
    if not response.get('ok') or not response['source_analysis_disabled']:
        raise RuntimeError('Decoder worker verification failed')
    rec=torch.from_numpy(np.load(output_path,allow_pickle=False));output_path.unlink()
    return rec,response['diagnostics']


def save_rgb(path,rgb):
    Image.fromarray((rgb[0].permute(1,2,0).numpy()*255).round().clip(0,255).astype(np.uint8)).save(path)


def run(args):
    torch.set_num_threads(2)
    full=args.reference
    full_progress=json.loads((full/'progress.json').read_text())
    if full_progress['state']!='complete' or full_progress['completed']!=2000:
        raise ValueError('Wait for the complete 2000-case reference validation')
    full_manifest=json.loads((full/'manifest.json').read_text())
    for name,sha in full_manifest['code_sha256'].items():
        if file_sha(REFERENCE/name)!=sha:raise ValueError('Reference code changed: '+name)
    crop_manifest=ROOT/'research/div2k100_center512_rgb/manifest.json'
    if file_sha(crop_manifest)!=full_manifest['crops_manifest_sha256']:raise ValueError('Crop manifest changed')
    crops={r['image']:r for r in json.loads(crop_manifest.read_text())['images']}
    config={'schema':1,'scope':'Fixed-depth patching/halo control; CPU reference format; interim epoch20 except released D12',
            'images':IMAGES,'selection':'16 evenly spaced DIV2K validation IDs, predeclared independently of quality',
            'depths':DEPTHS,'qps':QPS,'core_size':256,'halos':[0,32,64],
            'halo64_amendment':'Added before any patch outcomes,2026-09-27T17:54UTC: 32 and64 context halos both pad to320 on this2x2 grid; compare useful context at the same coded area',
            'halo_protocol':'Clip context windows to the 512 crop; codec pads each window to a multiple of 64; retain only the 256 core; no blending',
            'rate_denominator':512*512,'rate_note':'Sum every coded patch payload and report research container separately; no model-map bits for fixed-depth control',
            'reference_manifest_sha256':file_sha(full/'manifest.json'),'crop_manifest_sha256':file_sha(crop_manifest),
            'script_sha256':file_sha(__file__),'threads':2,'torch':torch.__version__}
    config=json.loads(json.dumps(config));args.out.mkdir(parents=True,exist_ok=True)
    mp=args.out/'manifest.json'
    if mp.exists() and json.loads(mp.read_text())!=config:raise ValueError('Different output configuration')
    atomic(mp,config)
    progress={'state':'running','started_utc':utc(),'pid':os.getpid(),'completed':0,'total':240}
    atomic(args.out/'progress.json',progress);worker=None;start=time.perf_counter()
    try:
        for depth in DEPTHS:
            net,source=load_model(checkpoint(depth),depth)
            if source['sha256']!=full_manifest['checkpoints'][str(depth)]:raise ValueError('Wrong checkpoint')
            codec=ReferenceCodec(net,source['sha256'],ROOT/'research/reference_entropy_v1')
            from src.utils.transforms import rgb2ycbcr_np,ycbcr2rgb
            folder=args.out/f'd{depth}';(folder/'streams').mkdir(parents=True,exist_ok=True)
            (folder/'cases').mkdir(exist_ok=True)
            log=(folder/'decoder.stderr.log').open('a')
            worker=subprocess.Popen([sys.executable,str(REFERENCE/'decode_worker.py'),'--checkpoint',str(checkpoint(depth)),
                                     '--depth',str(depth)],stdin=subprocess.PIPE,stdout=subprocess.PIPE,stderr=log,text=True,bufsize=1)
            ready=receive(worker)
            if not ready.get('ready') or ready['checkpoint']['sha256']!=source['sha256']:
                raise ValueError('Worker has wrong model')
            for image in IMAGES:
                crop=crops[image];cp=Path(crop['crop_path'])
                if file_sha(cp)!=crop['crop_file_sha256']:raise ValueError('Crop changed')
                rgb=np.load(cp,allow_pickle=False)
                target=torch.from_numpy(rgb.astype(np.float32)/255.).permute(2,0,1)[None].contiguous()
                x=torch.from_numpy(rgb2ycbcr_np(rgb.astype(np.float32)/255.)-.5).permute(2,0,1)[None].contiguous()
                for qp in QPS:
                    name=f'{Path(image).stem}_qp{qp:02d}';casepath=folder/'cases'/(name+'.json')
                    if casepath.exists():
                        record=json.loads(casepath.read_text())
                        if record['checkpoint_sha256']!=source['sha256'] or record['crop_sha256']!=crop['crop_file_sha256']:
                            raise ValueError('Existing case provenance differs')
                        for variant in record['patch_variants']:
                            for tile in variant['tiles']:
                                if file_sha(Path(tile['stream_path']))!=tile['stream_sha256']:raise ValueError('Patch stream changed')
                    else:
                        original_path=full/f'd{depth}/cases/{name}.json'
                        original=json.loads(original_path.read_text());stream=full/f'd{depth}/streams/{name}.fufref'
                        if file_sha(stream)!=original['stream_sha256']:raise ValueError('Full reference stream changed')
                        recon,diag=isolated(worker,stream,folder/(name+'_temporary.npy'))
                        if diag['y_hat_sha256']!=original['y_hat_sha256']:raise AssertionError('Full reference latent differs')
                        fullrgb=ycbcr2rgb(recon.clamp(-.5,.5)+.5,clamp=True);fullmetrics=metrics(fullrgb,target)
                        if abs(fullmetrics['psnr_rgb']-original['psnr_rgb'])>1e-8:raise AssertionError('Full reference PSNR differs')
                        record={'depth':depth,'qp':qp,'image':image,'checkpoint_sha256':source['sha256'],
                                'crop_sha256':crop['crop_file_sha256'],'reference_case_sha256':file_sha(original_path),
                                'full':{**fullmetrics,**{k:original[k] for k in ('payload_bytes','container_bytes','payload_bpp','container_bpp','estimated_bpp')}},
                                'patch_variants':[]}
                        if qp==32 and image in ('0801.png','0880.png'):save_rgb(folder/(name+'_full.png'),fullrgb)
                        for halo in (0,32,64):
                            canvas=torch.empty_like(x);tiles=[];coverage=torch.zeros((512,512),dtype=torch.int32)
                            for row,y in enumerate((0,256)):
                                for col,z in enumerate((0,256)):
                                    top,left=max(0,y-halo),max(0,z-halo)
                                    bottom,right=min(512,y+256+halo),min(512,z+256+halo)
                                    window=x[:,:,top:bottom,left:right].contiguous()
                                    encoded=codec.encode(window,qp)
                                    tilepath=folder/'streams'/f'{name}_h{halo}_r{row}c{col}.fufref';tilepath.write_bytes(encoded.stream)
                                    decoded,diag=isolated(worker,tilepath,folder/(name+'_temporary.npy'))
                                    if not torch.equal(decoded,encoded.reconstruction):raise AssertionError('Patch isolated decoder differs')
                                    for key in ('z_hat_sha256','y_hat_sha256'):
                                        if diag[key]!=encoded.diagnostics[key]:raise AssertionError('Patch latent differs')
                                    for a,b in zip(diag['stages'],encoded.diagnostics['stages']):
                                        for key in ('symbols_sha256','indexes_sha256'):
                                            if a[key]!=b[key]:raise AssertionError('Patch entropy trace differs')
                                    oy,ox=y-top,z-left
                                    canvas[:,:,y:y+256,z:z+256]=decoded[:,:,oy:oy+256,ox:ox+256]
                                    coverage[y:y+256,z:z+256]+=1
                                    tiles.append({'row':row,'col':col,'top':top,'left':left,'bottom':bottom,'right':right,
                                                  'stream_path':str(tilepath),'stream_sha256':file_sha(tilepath),
                                                  'exact_isolated_decode':True,**encoded.diagnostics})
                            if not torch.equal(coverage,torch.ones_like(coverage)):raise AssertionError('Invalid stitching coverage')
                            assembled=ycbcr2rgb(canvas.clamp(-.5,.5)+.5,clamp=True)
                            result=metrics(assembled,target)
                            payload=sum(t['payload_bytes'] for t in tiles);container=sum(t['container_bytes'] for t in tiles)
                            bits=sum(t['estimated_bits_y']+t['estimated_bits_z'] for t in tiles)
                            result.update(halo=halo,tiles=tiles,payload_bytes=payload,container_bytes=container,
                                          payload_bpp=payload*8/(512*512),container_bpp=container*8/(512*512),estimated_bpp=bits/(512*512),
                                          psnr_delta_vs_full_same_qp=result['psnr_rgb']-fullmetrics['psnr_rgb'],
                                          payload_delta_vs_full_same_qp=(payload-original['payload_bytes'])*8/(512*512),
                                          padded_coded_pixels=sum(math.ceil(t['height']/64)*64*math.ceil(t['width']/64)*64 for t in tiles))
                            for radius in (4,16):
                                result[f'seam_r{radius}_excess_mse_vs_full']=result[f'seam_r{radius}_mse']-fullmetrics[f'seam_r{radius}_mse']
                                result[f'interior_r{radius}_excess_mse_vs_full']=result[f'interior_r{radius}_mse']-fullmetrics[f'interior_r{radius}_mse']
                            record['patch_variants'].append(result)
                            if qp==32 and image in ('0801.png','0880.png'):save_rgb(folder/(name+f'_h{halo}.png'),assembled)
                        record['finished_utc']=utc();atomic(casepath,record)
                    progress.update(completed=progress['completed']+1,depth=depth,image=image,qp=qp,updated_utc=utc(),elapsed_seconds=time.perf_counter()-start)
                    atomic(args.out/'progress.json',progress)
                    print(json.dumps(progress),flush=True)
            worker.stdin.write('{"stop":true}\n');worker.stdin.flush();worker.wait(timeout=30);log.close()
            if worker.returncode:raise RuntimeError('Decoder worker failed')
            worker=None;del codec,net
        if file_sha(__file__)!=config['script_sha256'] or torch.cuda.is_initialized():raise AssertionError('Run integrity failure')
        progress.update(state='complete',finished_utc=utc());atomic(args.out/'progress.json',progress)
    except BaseException as exc:
        progress.update(state='failed',error=repr(exc),updated_utc=utc());atomic(args.out/'progress.json',progress);raise
    finally:
        if worker is not None and worker.poll() is None:worker.terminate();worker.wait(timeout=30)


if __name__=='__main__':
    ap=argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--reference',type=Path,default=ROOT/'research/div2k100_reference_epoch020')
    ap.add_argument('--out',type=Path,default=ROOT/'research/patch_control_epoch020')
    run(ap.parse_args())
