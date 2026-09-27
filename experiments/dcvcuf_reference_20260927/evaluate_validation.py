"""Frozen-checkpoint DIV2K validation using actual bytes and an isolated decoder.

CPU research format; neither released CUDA wire compatibility nor latency claims.
"""
from __future__ import annotations
import argparse
import datetime
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
from model_io import ROOT,load_model,file_sha
from reference_codec import ReferenceCodec

HERE=Path(__file__).resolve().parent
QPS=(0,16,32,48,63)
DEPTHS=(2,4,6,12)


def utc():return datetime.datetime.now(datetime.timezone.utc).isoformat()


def atomic(path,value):
    tmp=path.with_suffix(path.suffix+'.tmp')
    tmp.write_text(json.dumps(value,indent=2,allow_nan=False)+'\n');tmp.replace(path)


def checkpoint(depth):
    return ROOT/('reference_d12/released_cvpr2026_image.pth.tar' if depth==12 else f'runs/d{depth}/weights_epoch020.pt')


def receive(worker):
    line=worker.stdout.readline()
    if not line:raise RuntimeError(f'Decoder exited unexpectedly: {worker.poll()}')
    return json.loads(line)


def run(args):
    torch.set_num_threads(2)
    verification_path=ROOT/'research/reference_verification_epoch020/verification.json'
    verification=json.loads(verification_path.read_text())
    if not verification['all_passed'] or len(verification['cases'])!=36:
        raise ValueError('Complete 36-case engineering preflight required')
    for name,sha in verification['code_sha256'].items():
        if file_sha(HERE/name)!=sha:raise ValueError('Preflight code changed: '+name)
    manifest_path=args.crops/'manifest.json'
    crops=json.loads(manifest_path.read_text())['images']
    if [r['image'] for r in crops]!=[f'{i:04d}.png' for i in range(801,901)]:
        raise ValueError('Expected all 100 validation images')
    source_hashes={p.name:file_sha(p) for p in HERE.glob('*.py')}
    config={'schema':1,'epoch_shallow':20,'depths':DEPTHS,'qps':QPS,
            'scope':'Interim CPU actual-payload validation; released D12 has different training history',
            'format':'FUFREF1 CPU FP32 research container, not released CUDA format',
            'header_bytes':88,'header_note':'Includes 64 integrity/identity hash bytes; not minimal production overhead',
            'crops_manifest_sha256':file_sha(manifest_path),'verification_sha256':file_sha(verification_path),
            'code_sha256':source_hashes,'checkpoints':{str(d):file_sha(checkpoint(d)) for d in DEPTHS},
            'torch':torch.__version__,'numpy':np.__version__,'threads':2,'device':'cpu',
            'checkpoint_selection':False,'source_analysis_disabled_in_decoder':True}
    # JSON normalisation allows an identical manifest to be resumed.
    config=json.loads(json.dumps(config));args.out.mkdir(parents=True,exist_ok=True)
    config_path=args.out/'manifest.json'
    if config_path.exists() and json.loads(config_path.read_text())!=config:
        raise ValueError('Output configuration differs; use a new directory')
    atomic(config_path,config)
    progress={'state':'running','started_utc':utc(),'pid':os.getpid(),'completed':0,'total':2000,
              'output':str(args.out),'cuda_initialized':False}
    atomic(args.out/'progress.json',progress)
    worker=None;started=time.perf_counter()
    try:
        for depth in DEPTHS:
            net,source=load_model(checkpoint(depth),depth)
            if source['sha256']!=verification['models'][str(depth)]['sha256']:
                raise ValueError('Checkpoint differs from engineering preflight')
            codec=ReferenceCodec(net,source['sha256'],ROOT/'research/reference_entropy_v1')
            from src.utils.transforms import rgb2ycbcr_np,ycbcr2rgb
            folder=args.out/f'd{depth}';(folder/'streams').mkdir(parents=True,exist_ok=True)
            (folder/'cases').mkdir(exist_ok=True)
            error_log=(folder/'decoder.stderr.log').open('a')
            worker=subprocess.Popen([sys.executable,str(HERE/'decode_worker.py'),'--checkpoint',str(checkpoint(depth)),
                                     '--depth',str(depth)],stdin=subprocess.PIPE,stdout=subprocess.PIPE,
                                    stderr=error_log,text=True,bufsize=1)
            ready=receive(worker)
            if not ready.get('ready') or ready['checkpoint']['sha256']!=source['sha256']:
                raise ValueError('Decoder checkpoint identity mismatch')
            for crop in crops:
                path=Path(crop['crop_path'])
                if file_sha(path)!=crop['crop_file_sha256']:raise ValueError('Crop cache changed')
                rgb=np.load(path,allow_pickle=False)
                if rgb.shape!=(512,512,3) or rgb.dtype!=np.uint8:raise ValueError('Invalid crop')
                target_rgb=torch.from_numpy(rgb.astype(np.float32)/255.).permute(2,0,1)[None].contiguous()
                x=torch.from_numpy(rgb2ycbcr_np(rgb.astype(np.float32)/255.)-.5).permute(2,0,1)[None].contiguous()
                for qp in QPS:
                    name=f'{Path(crop["image"]).stem}_qp{qp:02d}'
                    case_path=folder/'cases'/(name+'.json');stream_path=folder/'streams'/(name+'.fufref')
                    if case_path.exists():
                        case=json.loads(case_path.read_text())
                        if (case['checkpoint_sha256']!=source['sha256'] or case['crop_sha256']!=crop['crop_file_sha256']
                            or case['stream_sha256']!=file_sha(stream_path) or not case['exact_isolated_decode']):
                            raise ValueError('Existing case integrity failed')
                    else:
                        encoded=codec.encode(x,qp);stream_path.write_bytes(encoded.stream)
                        decoded_path=folder/(name+'_decoded.npy')
                        worker.stdin.write(json.dumps({'stream':str(stream_path),'output':str(decoded_path)})+'\n');worker.stdin.flush()
                        response=receive(worker)
                        if not response.get('ok') or not response['source_analysis_disabled']:
                            raise ValueError('Invalid decoder response')
                        decoded=np.load(decoded_path,allow_pickle=False);diag=response['diagnostics']
                        if not np.array_equal(decoded,encoded.reconstruction.numpy()):
                            raise AssertionError(f'Isolated reconstruction mismatch: d{depth}/{name}')
                        for key in ('z_hat_sha256','y_hat_sha256'):
                            if diag[key]!=encoded.diagnostics[key]:raise AssertionError('Latent mismatch: '+key)
                        if len(diag['stages'])!=4:raise AssertionError('Expected four entropy stages')
                        for a,b in zip(diag['stages'],encoded.diagnostics['stages']):
                            for key in ('symbols','symbols_sha256','indexes_sha256'):
                                if a[key]!=b[key]:raise AssertionError('Entropy trace mismatch: '+key)
                        rec=torch.from_numpy(decoded).clamp(-.5,.5)
                        rec_rgb=ycbcr2rgb(rec+.5,clamp=True)
                        mse_rgb=float((rec_rgb-target_rgb).square().mean())
                        channel_mse=(rec-x).square().mean(dim=(0,2,3)).tolist()
                        channel_psnr=[-10*math.log10(max(v,1e-12)) for v in channel_mse]
                        case={'image':crop['image'],'monitor_subset':crop['monitor_subset'],
                              'checkpoint_sha256':source['sha256'],'crop_sha256':crop['crop_file_sha256'],
                              'stream_sha256':file_sha(stream_path),'exact_isolated_decode':True,
                              'decoder_pid':ready['pid'],'psnr_rgb':-10*math.log10(max(mse_rgb,1e-12)),
                              'mse_rgb':mse_rgb,'channel_mse_yuv':channel_mse,
                              'psnr_y':channel_psnr[0],'psnr_u':channel_psnr[1],'psnr_v':channel_psnr[2],
                              'psnr_yuv611':(6*channel_psnr[0]+channel_psnr[1]+channel_psnr[2])/8,
                              **encoded.diagnostics,'decoder_diagnostics':diag,'finished_utc':utc()}
                        atomic(case_path,case);decoded_path.unlink()
                        if qp==32 and crop['image'] in ('0801.png','0820.png','0880.png'):
                            pixels=(rec_rgb[0].permute(1,2,0).numpy()*255).round().clip(0,255).astype(np.uint8)
                            Image.fromarray(pixels).save(folder/(name+'_reconstruction.png'))
                    progress.update(completed=progress['completed']+1,depth=depth,image=crop['image'],qp=qp,
                                    updated_utc=utc(),elapsed_seconds=time.perf_counter()-started)
                    atomic(args.out/'progress.json',progress)
                    if progress['completed']%25==0:print(json.dumps(progress),flush=True)
            worker.stdin.write('{"stop":true}\n');worker.stdin.flush()
            worker.wait(timeout=30);error_log.close()
            if worker.returncode:raise RuntimeError('Decoder process failed')
            worker=None;del codec,net
        if torch.cuda.is_initialized():raise AssertionError('Unexpected CUDA initialization')
        if source_hashes!={p.name:file_sha(p) for p in HERE.glob('*.py')}:
            raise AssertionError('Research code changed during evaluation')
        if file_sha(manifest_path)!=config['crops_manifest_sha256']:raise AssertionError('Crop manifest changed')
        progress.update(state='complete',finished_utc=utc())
        atomic(args.out/'progress.json',progress);print(json.dumps(progress),flush=True)
    except BaseException as exc:
        progress.update(state='failed',error=repr(exc),updated_utc=utc());atomic(args.out/'progress.json',progress)
        raise
    finally:
        if worker is not None and worker.poll() is None:
            worker.terminate();worker.wait(timeout=30)


if __name__=='__main__':
    ap=argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--crops',type=Path,default=ROOT/'research/div2k100_center512_rgb')
    ap.add_argument('--out',type=Path,default=ROOT/'research/div2k100_reference_epoch020')
    run(ap.parse_args())
