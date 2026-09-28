"""Eight predeclared shared-stream correctness cases; no RD or latency claim."""
import datetime,json,os,subprocess,sys,hashlib
from pathlib import Path
from codec import REPO,ROOT,REFERENCE,SNAPSHOT,UPSTREAM,CHECKPOINT,RELEASE,HEADER,load,wrap,unwrap,sha,torch
import numpy as np
import torch.nn.functional as F

HERE=Path(__file__).resolve().parent
OUT=ROOT/'research/shared_stream_preflight'
PRIOR=ROOT/'research/shared_crossfit_qp32'


def utc():return datetime.datetime.now(datetime.timezone.utc).isoformat()
def atomic(path,value):
    tmp=path.with_suffix(path.suffix+'.tmp');tmp.write_text(json.dumps(value,indent=2,allow_nan=False)+'\n');tmp.replace(path)


def bad_stream_checks(data,identities):
    original=list(HEADER.unpack_from(data));body=data[HEADER.size:]
    variants={'truncated':data[:-1],'checksum':data[:-1]+bytes([data[-1]^1])}
    for name,index,value in [('checkpoint',5,bytes(32)),('map_count',3,original[3]+1),('width',2,1),('inner_length',4,2**31)]:
        header=original.copy();header[index]=value;variants[name]=HEADER.pack(*header)+body
    if original[3]%4:
        changed=bytes([body[0]|1])+body[1:];header=original.copy();header[6]=hashlib.sha256(changed).digest();variants['unused_map_bits']=HEADER.pack(*header)+changed
    result={}
    for name,value in variants.items():
        try:unwrap(value,identities['shared_checkpoint_sha256'],identities['released_checkpoint_sha256'])
        except ValueError as error:result[name]=str(error)
        else:raise AssertionError('Malformed input accepted: '+name)
    return result


def main():
    if OUT.exists():raise ValueError('Refusing to overwrite a preflight attempt')
    OUT.mkdir(parents=True);(OUT/'streams').mkdir();(OUT/'decoded').mkdir()
    prior=[json.loads(p.read_text()) for p in sorted((PRIOR/'cases').glob('*.json'))]
    chosen=[]
    for stem,dimensions in [('BasketballPass',(240,416)),('BQMall',(480,832))]:
        matches=[r for r in prior if r['sequence'].startswith(stem+'_')]
        if len(matches)!=1 or (matches[0]['height'],matches[0]['width'])!=dimensions:raise ValueError('Predeclared source differs')
        chosen+=matches
    plans=[]
    for case in chosen:
        n=((case['height']+255)//256)*((case['width']+255)//256)
        maps={c:next(r['map'] for r in case['rows'] if (r['criterion'],r['policy'])==(c,'router')) for c in ('mean','q90')}
        maps.update(deepest=[5]*n,cyclic=[2+i%4 for i in range(n)])
        plans.append({'sequence':case['sequence'],'height':case['height'],'width':case['width'],'frame_sha256':case['first_frame_bytes_sha256'],'maps':maps})
    files=[*sorted(HERE.glob('*.py')),HERE/'PROTOCOL.md',REFERENCE/'reference_codec.py',CHECKPOINT,RELEASE,PRIOR/'manifest.json',*sorted((PRIOR/'cases').glob('*.json'))]
    frozen={str(p):sha(p) for p in files}
    manifest={'scope':__doc__,'declared_utc':utc(),'plans':plans,'qp':32,'code_and_input_sha256':frozen,'backend':'CPU FP32; pinned research rANS, two torch threads','protocol':'FUFEXIT1 outer map/valid-support/e15 identity around released-front-end FUFREF1 entropy bytes. Explicit map transmission, no source-free selection claim.'}
    atomic(OUT/'manifest.json',manifest);progress={'state':'running','pid':os.getpid(),'started_utc':utc(),'completed':0,'total':8};atomic(OUT/'progress.json',progress)
    worker=None
    try:
        codec,synthesis,identities=load();import ctc_intra as C
        seqs,_=C.discover([]);lookup={Path(s['path']).name:s for s in seqs}
        with (OUT/'decoder_stderr.log').open('w') as err:
            worker=subprocess.Popen([sys.executable,str(HERE/'decode_worker.py')],stdin=subprocess.PIPE,stdout=subprocess.PIPE,stderr=err,text=True,bufsize=1,cwd=HERE)
            ready=json.loads(worker.stdout.readline())
            if not ready['ready'] or not ready['identities']['source_analysis_disabled'] or ready['cuda_initialized']:raise ValueError('Independent decoder not ready')
            for key in ('shared_checkpoint_sha256','released_checkpoint_sha256','front_tensors_equal'):
                if ready['identities'][key]!=identities[key]:raise ValueError('Encoder/decoder identity mismatch')
            rows=[];negative=None
            with torch.inference_mode():
                for plan in plans:
                    seq=lookup[plan['sequence']];h,w=plan['height'],plan['width']
                    with Path(seq['path']).open('rb') as f:digest=hashlib.sha256(f.read(h*w*3//2)).hexdigest()
                    if digest!=plan['frame_sha256']:raise ValueError('Source frame identity changed')
                    x,_=C.read_frames(seq['path'],w,h,1,1);xp=F.pad(x,(0,(-w)%256,0,(-h)%256),mode='replicate');inner_hashes=[]
                    for profile,indices in plan['maps'].items():
                        synthesis.set_map(indices);encoded=codec.encode(xp,32)
                        stream=wrap(encoded.stream,h,w,indices,identities['shared_checkpoint_sha256'])
                        if negative is None:negative=bad_stream_checks(stream,identities)
                        key=Path(plan['sequence']).stem+'_'+profile;path=OUT/'streams'/f'{key}.bin';path.write_bytes(stream)
                        output=OUT/'decoded'/f'{key}.npy';worker.stdin.write(json.dumps({'stream':str(path),'output':str(output)})+'\n');worker.stdin.flush()
                        line=worker.stdout.readline()
                        if not line:raise RuntimeError('Independent decoder exited: '+str(worker.poll()))
                        reply=json.loads(line)
                        if not reply['ok'] or not reply['source_analysis_disabled'] or reply['cuda_initialized']:raise ValueError('Decoder scope differs')
                        expected=encoded.reconstruction[:,:,:h,:w].contiguous();decoded=torch.from_numpy(np.load(output,allow_pickle=False))
                        if not torch.equal(expected,decoded):raise AssertionError('Cropped reconstruction differs')
                        for field in ('z_hat_sha256','y_hat_sha256'):
                            if encoded.diagnostics[field]!=reply['trace'][field]:raise AssertionError('Recovered latent differs')
                        for a,b in zip(encoded.diagnostics['stages'],reply['trace']['stages']):
                            for field in ('stage','symbols','symbols_sha256','indexes_sha256'):
                                if a[field]!=b[field]:raise AssertionError('Entropy recovery differs')
                        if reply['meta']['map']!=indices:raise AssertionError('Transmitted map differs')
                        meta=reply['meta'];inner_hash=hashlib.sha256(encoded.stream).hexdigest();inner_hashes.append(inner_hash)
                        row={'sequence':plan['sequence'],'profile':profile,'map':indices,'height':h,'width':w,'qp':32,'container_bytes':len(stream),'map_bytes':meta['map_bytes'],'outer_header_bytes':meta['outer_header_bytes'],'inner_header_bytes':meta['inner_header_bytes'],'payload_bytes':meta['payload_bytes'],'container_bpp_valid':len(stream)*8/(h*w),'inner_stream_sha256':inner_hash,'stream_sha256':sha(path),'decoded_tensor_sha256':hashlib.sha256(decoded.numpy().tobytes()).hexdigest(),'reconstruction_exact':True,'latents_symbols_indexes_exact':True,'source_analysis_disabled':True}
                        assert sum(row[k] for k in ('map_bytes','outer_header_bytes','inner_header_bytes','payload_bytes'))==len(stream)
                        rows.append(row);atomic(OUT/'cases.json',rows);output.unlink()
                        progress.update(completed=len(rows),sequence=plan['sequence'],profile=profile,updated_utc=utc());atomic(OUT/'progress.json',progress);print(json.dumps(progress),flush=True)
                    if len(set(inner_hashes))!=1:raise AssertionError('Synthesis map changed entropy bytes')
            worker.stdin.write(json.dumps({'stop':True})+'\n');worker.stdin.flush();worker.wait(timeout=30)
            if worker.returncode:raise RuntimeError('Decoder exited abnormally')
        for path,digest in frozen.items():
            if sha(path)!=digest:raise ValueError('Frozen source changed during preflight')
        if len(rows)!=8 or torch.cuda.is_initialized():raise AssertionError('Incomplete or unexpected CUDA preflight')
        result={'scope':__doc__,'manifest':manifest,'identities':identities,'decoder_ready':ready,'cases':rows,'malformed_rejections':negative,'all_exact':True,'same_inner_bytes_for_all_four_maps_per_source':True,'cuda_initialized':False,'finished_utc':utc()};atomic(OUT/'verification.json',result)
        progress.update(state='complete',finished_utc=utc(),all_exact=True,cuda_initialized=False);atomic(OUT/'progress.json',progress)
    except BaseException as error:
        if worker is not None and worker.poll() is None:worker.terminate();worker.wait(timeout=30)
        progress.update(state='failed',error=repr(error),updated_utc=utc());atomic(OUT/'progress.json',progress);raise


if __name__=='__main__':main()
