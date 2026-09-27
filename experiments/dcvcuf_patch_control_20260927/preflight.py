"""Check new patch geometries, decoder isolation and seam accounting on CPU."""
import json
import os
from pathlib import Path
import subprocess
import sys
os.environ['CUDA_VISIBLE_DEVICES']=''
import numpy as np
import torch
from evaluate_patching import REFERENCE,ROOT,ReferenceCodec,load_model,file_sha,checkpoint,receive,atomic,metrics,utc


def main():
    torch.set_num_threads(2);torch.manual_seed(20260927)
    out=ROOT/'research/patch_control_preflight'
    if (out/'verification.json').exists():raise ValueError('Preflight already exists; inspect before overwrite')
    out.mkdir(parents=True,exist_ok=True)
    net,source=load_model(checkpoint(2),2)
    codec=ReferenceCodec(net,source['sha256'],ROOT/'research/reference_entropy_v1')
    log=(out/'decoder_halo64.stderr.log').open('w')
    worker=subprocess.Popen([sys.executable,str(REFERENCE/'decode_worker.py'),'--checkpoint',str(checkpoint(2)),
        '--depth','2'],stdin=subprocess.PIPE,stdout=subprocess.PIPE,stderr=log,text=True,bufsize=1)
    rows=[]
    try:
        ready=receive(worker);assert ready['ready'] and ready['checkpoint']['sha256']==source['sha256']
        for size in (256,288,320):
            # Smooth deterministic test content keeps symbols within codec range.
            grid=torch.linspace(-.2,.2,size)
            x=(grid[None,None,:,None]+grid[None,None,None,:]).expand(1,3,size,size).contiguous()
            encoded=codec.encode(x,32);stream=out/f'preflight_halo64_size{size}.fufref';stream.write_bytes(encoded.stream)
            target=out/f'preflight_halo64_size{size}.npy'
            worker.stdin.write(json.dumps({'stream':str(stream),'output':str(target)})+'\n');worker.stdin.flush()
            response=receive(worker);assert response['ok'] and response['source_analysis_disabled']
            reconstructed=torch.from_numpy(np.load(target,allow_pickle=False));assert torch.equal(reconstructed,encoded.reconstruction)
            for key in ('z_hat_sha256','y_hat_sha256'):
                assert response['diagnostics'][key]==encoded.diagnostics[key]
            rows.append({'size':size,'qp':32,'depth':2,'exact_fresh_process_decode':True,
                'payload_bytes':encoded.diagnostics['payload_bytes'],'stream_sha256':file_sha(stream)})
        target=torch.zeros((1,3,512,512));result=metrics(target+.25,target)
        assert result['mse_rgb']==.0625
        for radius in (4,16):
            assert result[f'seam_r{radius}_pixels']==2*512*(2*radius)-(2*radius)**2
            assert result[f'seam_r{radius}_mse']==result[f'interior_r{radius}_mse']==.0625
        for halo in (0,32,64):
            cover=torch.zeros((512,512),dtype=torch.int32);area=0
            for y in (0,256):
                for x in (0,256):
                    top,left=max(0,y-halo),max(0,x-halo)
                    bottom,right=min(512,y+256+halo),min(512,x+256+halo)
                    assert bottom-top==right-left==256+halo
                    assert 0<=y-top<=halo and 0<=x-left<=halo
                    cover[y:y+256,x:x+256]+=1
                    area+=((bottom-top+63)//64*64)*((right-left+63)//64*64)
            assert torch.equal(cover,torch.ones_like(cover))
            assert area==(512**2 if halo==0 else 4*320**2)
        assert not torch.cuda.is_initialized()
        result={'cases':rows,'metric_geometry_check':True,'cuda_initialized':False,'completed_utc':utc(),
            'patch_script_sha256':file_sha(Path(__file__).with_name('evaluate_patching.py')),
            'preflight_script_sha256':file_sha(__file__),'checkpoint_sha256':source['sha256'],
            'scope':'D2/QP32 engineering preflight only, not comparative image-quality evidence'}
        atomic(out/'verification.json',result);print(json.dumps(result))
        worker.stdin.write('{"stop":true}\n');worker.stdin.flush();worker.wait(timeout=30)
        assert worker.returncode==0
    finally:
        if worker.poll() is None:worker.terminate();worker.wait(timeout=30)
        log.close()


if __name__=='__main__':main()
