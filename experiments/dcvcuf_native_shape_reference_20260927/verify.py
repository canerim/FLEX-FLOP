"""Fresh-process proof for isolated FUFREF2 native-shaped CPU coding."""
import importlib.util
import json
import os
from pathlib import Path
import subprocess
import sys
os.environ['CUDA_VISIBLE_DEVICES']=''
import numpy as np
import torch
from prepare import HERE,BASE,ROOT,OUT as SOURCE,sha


def main():
    torch.set_num_threads(2)
    prep=json.loads((SOURCE.parent/'preparation.json').read_text())
    for name,digest in prep['derived_source_sha256'].items():
        if sha(SOURCE/name)!=digest:raise ValueError('Derived source changed')
    out=ROOT/'native_shape_verification_epoch020'
    if out.exists():raise ValueError('Inspect existing verification before retry')
    out.mkdir(parents=True)
    sys.path.insert(0,str(SOURCE))
    from model_io import load_model,ROOT as TRAINROOT
    from reference_codec import ReferenceCodec
    spec=importlib.util.spec_from_file_location('reference_codec_pad64',BASE/'reference_codec.py')
    old=importlib.util.module_from_spec(spec);sys.modules[spec.name]=old;spec.loader.exec_module(old)
    def checkpoint(d):return TRAINROOT/('reference_d12/released_cvpr2026_image.pth.tar' if d==12 else f'runs/d{d}/weights_epoch020.pt')
    def receive(worker):
        line=worker.stdout.readline()
        if not line:raise RuntimeError('Fresh decoder exited')
        return json.loads(line)
    crop_manifest=json.loads((ROOT/'div2k100_center512_rgb/manifest.json').read_text())
    crop=next(c for c in crop_manifest['images'] if c['image']=='0801.png')
    if sha(crop['crop_path'])!=crop['crop_file_sha256']:raise ValueError('Crop changed')
    cases=[];models={};worker=None
    try:
        for depth in (2,6,12):
            net,info=load_model(checkpoint(depth),depth);models[str(depth)]=info
            codec=ReferenceCodec(net,info['sha256'],ROOT/'reference_entropy_v1')
            pad64=old.ReferenceCodec(net,info['sha256'],ROOT/'reference_entropy_v1')
            from src.utils.transforms import rgb2ycbcr_np
            rgb=np.load(crop['crop_path'],allow_pickle=False).astype(np.float32)/255
            target=torch.from_numpy(rgb2ycbcr_np(rgb)-.5).permute(2,0,1)[None].contiguous()
            log=(out/f'd{depth}_decoder.stderr.log').open('w')
            worker=subprocess.Popen([sys.executable,str(SOURCE/'decode_worker.py'),'--checkpoint',str(checkpoint(depth)),
                '--depth',str(depth)],stdin=subprocess.PIPE,stdout=subprocess.PIPE,stderr=log,text=True,bufsize=1)
            assert receive(worker)['ready']
            for h,w in ((64,64),(65,97),(288,288),(320,320),(512,512)):
                x=target[:,:,:h,:w].contiguous()
                for qp in (0,32,63):
                    encoded=codec.encode(x,qp);name=f'd{depth}_{h}x{w}_q{qp}'
                    stream=out/(name+'.fufref2');stream.write_bytes(encoded.stream);decoded_path=out/(name+'.npy')
                    worker.stdin.write(json.dumps({'stream':str(stream),'output':str(decoded_path)})+'\n');worker.stdin.flush()
                    response=receive(worker);assert response['ok'] and response['source_analysis_disabled']
                    decoded=torch.from_numpy(np.load(decoded_path,allow_pickle=False));decoded_path.unlink()
                    assert torch.equal(decoded,encoded.reconstruction)
                    for key in ('z_hat_sha256','y_hat_sha256'):assert response['diagnostics'][key]==encoded.diagnostics[key]
                    for a,b in zip(response['diagnostics']['stages'],encoded.diagnostics['stages']):
                        for key in ('symbols_sha256','indexes_sha256'):assert a[key]==b[key]
                    aligned=h%64==w%64==0
                    if aligned:
                        comparison=pad64.encode(x,qp)
                        assert comparison.stream[88:]==encoded.stream[88:]
                        assert torch.equal(comparison.reconstruction,encoded.reconstruction)
                    cases.append({'depth':depth,'qp':qp,'height':h,'width':w,'exact_fresh_process_decode':True,
                        'symbol_and_index_trace_match':True,'pad64_payload_and_reconstruction_equal':True if aligned else None,
                        'payload_bytes':encoded.diagnostics['payload_bytes'],'stream_sha256':sha(stream)})
                    print(json.dumps({'complete':len(cases),'total':45,**cases[-1]}),flush=True)
            worker.stdin.write('{"stop":true}\n');worker.stdin.flush();worker.wait(timeout=30);assert worker.returncode==0
            log.close();worker=None;del net,codec,pad64
        assert len(cases)==45 and not torch.cuda.is_initialized()
        for name,digest in prep['derived_source_sha256'].items():assert sha(SOURCE/name)==digest
        result={'all_passed':True,'cases':cases,'models':models,'source':str(SOURCE),
            'derived_source_sha256':prep['derived_source_sha256'],'preparation_sha256':sha(SOURCE.parent/'preparation.json'),
            'verify_script_sha256':sha(__file__),'cuda_initialized':False,'source_analysis_disabled_in_worker':True,
            'scope':'CPU FP32 image-pad16/latent-pad4 engineering proof. Same payload/reconstruction asFUFREF1 for64-aligned shapes; no native CUDA numerical, wire-format or runtime claim.'}
        (out/'verification.json').write_text(json.dumps(result,indent=2)+'\n')
        (HERE/'verification.json').write_text(json.dumps(result,indent=2)+'\n')
    finally:
        if worker is not None and worker.poll() is None:worker.terminate();worker.wait(timeout=30)


if __name__=='__main__':main()
