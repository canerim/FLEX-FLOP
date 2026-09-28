"""Fixed-map two-expert coalescing with independently decoded research bank streams."""
from __future__ import annotations
import argparse,datetime,hashlib,json,math,os,subprocess,sys,time
from pathlib import Path
os.environ['CUDA_VISIBLE_DEVICES']='';os.environ.setdefault('OMP_NUM_THREADS','2')
import numpy as np
import torch
HERE=Path(__file__).resolve().parent
ROOT=Path('/data10/shareddata/can_karsal/dcvcuf_depth_20260927');RESEARCH=ROOT/'research'
SOURCE=RESEARCH/'native_shape_reference/source';PRIMARY=RESEARCH/'patch_native_shape_epoch020'
sys.path.insert(0,str(SOURCE))
from model_io import load_model
from reference_codec import ReferenceCodec,parse_container
from layout import pack,regions,description,checks,HEADER
IMAGES=[f'{i:04d}.png' for i in np.rint(np.linspace(801,900,16)).astype(int)]
QPS=(0,16,32,48,63)

def sha(p):
    with Path(p).open('rb') as f:return hashlib.file_digest(f,'sha256').hexdigest()
def utc():return datetime.datetime.now(datetime.timezone.utc).isoformat()
def atomic(path,data):
    temp=path.with_suffix(path.suffix+'.tmp');temp.write_text(json.dumps(data,indent=2,allow_nan=False)+'\n');temp.replace(path)
def receive(worker):
    line=worker.stdout.readline()
    if not line:raise RuntimeError('Independent bank decoder exited; inspect stderr')
    return json.loads(line)
def source_hashes():return {p.name:sha(p) for p in sorted(HERE.glob('*.py'))}

def metrics(rgb,target):
    sq=(rgb-target).square();err=sq.mean(1)[0];mse=float(sq.mean())
    return {'mse_rgb':mse,'psnr_rgb':-10*math.log10(max(mse,1e-12)),
            'vertical_band_r4_mse':float(err[:,252:260].mean()),'horizontal_band_r4_mse':float(err[252:260,:].mean())}

def models():
    codecs={};sources={}
    for depth in (2,6):
        net,info=load_model(ROOT/f'runs/d{depth}/weights_epoch020.pt',depth)
        codecs[depth]=ReferenceCodec(net,info['sha256'],RESEARCH/'reference_entropy_v1');sources[depth]=info
    return codecs,sources

def start_worker(out,sources):
    log=(out/'decoder.stderr.log').open('a')
    p=subprocess.Popen([sys.executable,str(HERE/'decode_worker.py')],stdin=subprocess.PIPE,stdout=subprocess.PIPE,stderr=log,text=True,bufsize=1)
    ready=receive(p)
    if not ready['ready'] or any(ready['checkpoints'][str(d)]['sha256']!=s['sha256'] for d,s in sources.items()):raise ValueError('Bank decoder model identity differs')
    return p,log

def stop_worker(worker,log):
    if worker.poll() is None:worker.stdin.write('{"stop":true}\n');worker.stdin.flush();worker.wait(timeout=30)
    log.close()
    if worker.returncode:raise RuntimeError('Bank decoder did not exit normally')

def one_case(image,qp,crop,out,codecs,sources,worker,reuse):
    from src.utils.transforms import rgb2ycbcr_np,ycbcr2rgb
    path=Path(crop['crop_path'])
    if sha(path)!=crop['crop_file_sha256']:raise ValueError('Source crop changed')
    rgb=np.load(path,allow_pickle=False);target=torch.from_numpy(rgb.astype(np.float32)/255).permute(2,0,1)[None].contiguous()
    x=torch.from_numpy(rgb2ycbcr_np(rgb.astype(np.float32)/255)-.5).permute(2,0,1)[None].contiguous()
    name=f'{image[:4]}_qp{qp:02d}';components={};originals={};source_cases={}
    if reuse:
        for depth in (2,6):
            old=PRIMARY/f'd{depth}/cases/{name}.json';record=json.loads(old.read_text());source_cases[str(old)]=sha(old)
            if record['checkpoint_sha256']!=sources[depth]['sha256'] or record['crop_sha256']!=crop['crop_file_sha256']:raise ValueError('Native-padding source case changed')
            for tile in record['native_shape_halo32']['tiles']:
                originals[depth,tile['row'],tile['col']]=tile
    variants=[]
    for profile in range(10):
        parts=[];meta=[];expected=torch.empty_like(x)
        for region in regions(profile):
            depth=region['depth'];t,l,b,r=region['core'];wt,wl,wb,wr=region['window'];key=(depth,wt,wl,wb,wr)
            if key not in components:
                if reuse and b-t==r-l==256:
                    old=originals[depth,t//256,l//256];stream=Path(old['stream_path']).read_bytes()
                    if hashlib.sha256(stream).hexdigest()!=old['stream_sha256']:raise ValueError('Reused component bytes changed')
                    rec,diag=codecs[depth].decode(stream);measure=old
                else:
                    encoded=codecs[depth].encode(x[:,:,wt:wb,wl:wr].contiguous(),qp)
                    stream,rec,diag=encoded.stream,encoded.reconstruction,encoded.diagnostics;measure=diag
                components[key]=(stream,rec,diag,measure)
            stream,rec,diag,measure=components[key];oy,ox=t-wt,l-wl
            expected[:,:,t:b,l:r]=rec[:,:,oy:oy+b-t,ox:ox+r-l];parts.append(stream)
            meta.append({'region':region,'stream_sha256':hashlib.sha256(stream).hexdigest(),'payload_bytes':measure['payload_bytes'],'container_bytes':len(stream),'estimated_bits':measure['estimated_bits_y']+measure['estimated_bits_z'],'diagnostics':diag})
        bank=pack(profile,parts);stream_path=out/'streams'/f'{name}_p{profile:02d}.fufbank';stream_path.write_bytes(bank)
        temporary=out/'temporary.npy';worker.stdin.write(json.dumps({'stream':str(stream_path),'output':str(temporary)})+'\n');worker.stdin.flush();response=receive(worker)
        if not response['ok'] or not response['source_analysis_disabled'] or response['profile']!=profile:raise ValueError('Independent bank decode failed')
        reconstructed=torch.from_numpy(np.load(temporary,allow_pickle=False));temporary.unlink()
        if not torch.equal(reconstructed,expected):raise AssertionError('Independent bank reconstruction differs')
        for saved,decoded in zip(meta,response['parts']):
            if saved['stream_sha256']!=decoded['stream_sha256']:raise AssertionError('Component identity differs')
            for key in ('z_hat_sha256','y_hat_sha256'):
                if saved['diagnostics'][key]!=decoded[key]:raise AssertionError('Bank latent differs')
            for a,b in zip(saved['diagnostics']['stages'],decoded['stages']):
                for key in ('symbols_sha256','indexes_sha256'):
                    if a[key]!=b[key]:raise AssertionError('Bank entropy trace differs')
        pattern,merged,phase=description(profile);rgb_hat=ycbcr2rgb(reconstructed.clamp(-.5,.5)+.5,clamp=True)
        result=metrics(rgb_hat,target);payload=sum(m['payload_bytes'] for m in meta);embedded=sum(len(p) for p in parts)
        result.update(profile=profile,pattern=pattern,merged=merged,phase=phase,parts=meta,payload_bytes=payload,embedded_bytes=embedded,container_bytes=len(bank),bank_header_bytes=len(bank)-embedded,
            payload_bpp=payload*8/512**2,container_bpp=len(bank)*8/512**2,estimated_bpp=sum(m['estimated_bits'] for m in meta)/512**2,
            stream_path=str(stream_path),stream_sha256=sha(stream_path),exact_independent_decode=True)
        variants.append(result)
        if image in ('0801.png','0880.png') and qp==32:
            from PIL import Image
            Image.fromarray((rgb_hat[0].permute(1,2,0).numpy()*255).round().clip(0,255).astype(np.uint8)).save(out/f'{name}_p{profile:02d}.png')
    return {'image':image,'qp':qp,'crop_sha256':crop['crop_file_sha256'],'checkpoint_sha256':{d:s['sha256'] for d,s in sources.items()},'source_cases_sha256':source_cases,'variants':variants,'finished_utc':utc()}

def main():
    ap=argparse.ArgumentParser(description=__doc__);ap.add_argument('--preflight',action='store_true');args=ap.parse_args();torch.set_num_threads(2)
    out=RESEARCH/('region_merge_preflight' if args.preflight else 'region_merge_epoch020');out.mkdir(exist_ok=True);(out/'streams').mkdir(exist_ok=True);(out/'cases').mkdir(exist_ok=True)
    pinned=source_hashes();progress={'state':'preflight' if args.preflight else 'waiting_native_padding','pid':os.getpid(),'started_utc':utc(),'completed':0,'total':1 if args.preflight else 80,'source_sha256':pinned}
    atomic(out/'progress.json',progress)
    worker=None;log=None
    try:
        proof=json.loads((RESEARCH/'native_shape_verification_epoch020/verification.json').read_text())
        if not proof['all_passed']:raise ValueError('Native-shaped reference proof incomplete')
        for name,digest in proof['derived_source_sha256'].items():
            if sha(SOURCE/name)!=digest:raise ValueError('Derived codec changed')
        crop_file=RESEARCH/'div2k100_center512_rgb/manifest.json';crops={r['image']:r for r in json.loads(crop_file.read_text())['images']}
        if not args.preflight:
            preflight=RESEARCH/'region_merge_preflight/verification.json';verified=json.loads(preflight.read_text())
            if verified['source_sha256']!=pinned or not verified['all_profiles_exact']:raise ValueError('Current bank preflight missing')
            while True:
                path=PRIMARY/'progress.json'
                if path.exists():
                    state=json.loads(path.read_text());progress['native_padding_completed']=state['completed']
                    if state['state']=='failed':raise RuntimeError('Native-padding prerequisite failed')
                    if state['state']=='complete':
                        if state['completed']!=240:raise ValueError('Incomplete prerequisite cohort')
                        break
                progress['updated_utc']=utc();atomic(out/'progress.json',progress);time.sleep(30)
            if source_hashes()!=pinned:raise ValueError('Queued bank source changed')
        codecs,sources=models();worker,log=start_worker(out,sources);check=checks()
        manifest={'scope':'Fixed fifty-fifty D2/D6 maps, merged versus unmerged execution. CPU FP32 research bitstreams, not adaptive routing or runtime.',
            'images':['0801.png'] if args.preflight else IMAGES,'qps':[32] if args.preflight else QPS,'profiles':list(range(10)),'source_sha256':pinned,
            'derived_source_sha256':proof['derived_source_sha256'],'checkpoints':{d:s['sha256'] for d,s in sources.items()},'crop_manifest_sha256':sha(crop_file),'layout_checks':check,
            'decoder_cache':'Bounded cache of at most32 component reconstructions, populated only by independent byte decoding; no runtime claim',
            'primary_manifest_sha256':None if args.preflight else sha(PRIMARY/'manifest.json')}
        manifest=json.loads(json.dumps(manifest));mp=out/'manifest.json'
        if mp.exists() and json.loads(mp.read_text())!=manifest:raise ValueError('Existing bank output provenance differs')
        atomic(mp,manifest);progress['state']='running'
        for image in manifest['images']:
            for qp in manifest['qps']:
                case_path=out/'cases'/f'{image[:4]}_qp{qp:02d}.json'
                if case_path.exists():
                    record=json.loads(case_path.read_text())
                    if record['crop_sha256']!=crops[image]['crop_file_sha256'] or record['checkpoint_sha256']!=manifest['checkpoints']:raise ValueError('Existing bank case changed')
                    for v in record['variants']:
                        if sha(v['stream_path'])!=v['stream_sha256']:raise ValueError('Bank stream changed')
                else:
                    record=one_case(image,qp,crops[image],out,codecs,sources,worker,reuse=not args.preflight);atomic(case_path,record)
                progress.update(completed=progress['completed']+1,image=image,qp=qp,updated_utc=utc());atomic(out/'progress.json',progress);print(json.dumps(progress),flush=True)
        stop_worker(worker,log);worker=None
        if source_hashes()!=pinned or torch.cuda.is_initialized():raise AssertionError('Bank run integrity failure')
        if args.preflight:atomic(out/'verification.json',{'all_profiles_exact':True,'profiles':10,'source_sha256':pinned,'layout_checks':check,'cuda_initialized':False,'completed_utc':utc(),'scope':'One predeclared image/QP engineering proof, not comparative quality evidence'})
        progress.update(state='complete',finished_utc=utc());atomic(out/'progress.json',progress)
    except BaseException as e:
        progress.update(state='failed',error=repr(e),updated_utc=utc());atomic(out/'progress.json',progress);raise
    finally:
        if worker is not None and worker.poll() is None:worker.terminate();worker.wait(timeout=30)
if __name__=='__main__':main()
