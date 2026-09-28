"""Prepare and run isolated native-depth correctness checks on an idle study GPU.

Default --preflight is CPU-only. --run-correctness requires an explicitly selected
GPU from the existing study allocation (1/3/6), and refuses any active compute
process on it. This script has not yet produced GPU validation evidence.
"""
from __future__ import annotations
import argparse,datetime,hashlib,importlib.util,json,os,re,subprocess,sys
from pathlib import Path
HERE=Path(__file__).resolve().parent
ROOT=Path('/data10/shareddata/can_karsal/dcvcuf_depth_20260927')
RESEARCH=ROOT/'research'
REFERENCE=HERE.parent/'dcvcuf_reference_20260927'
QPS=(0,32,63)
SHAPES=((64,64),(65,97),(288,288),(320,320),(512,512),(288,512),(512,288))


def sha(path):
    with Path(path).open('rb') as f:return hashlib.file_digest(f,'sha256').hexdigest()

def utc():return datetime.datetime.now(datetime.timezone.utc).isoformat()

def atomic(path,data):
    path=Path(path);temp=path.with_suffix(path.suffix+'.tmp');temp.write_text(json.dumps(data,indent=2,allow_nan=False)+'\n');temp.replace(path)

def checkpoint(depth):return ROOT/('reference_d12/released_cvpr2026_image.pth.tar' if depth==12 else f'runs/d{depth}/weights_epoch020.pt')

def binary(backend):
    name='compile_report.json' if backend=='patched' else 'stock_compile_report.json'
    report=json.loads((HERE/name).read_text())
    if report['state']!='compiled' or report['returncode']!=0 or len(report['artifacts'])!=1:raise ValueError('Compilation evidence incomplete')
    path,digest=next(iter(report['artifacts'].items()))
    if sha(path)!=digest:raise ValueError('Native binary changed')
    return Path(path),report

def cpu_model(depth):
    sys.path.insert(0,str(REFERENCE))
    from model_io import load_model
    from reference_codec import install_entropy_path
    install_entropy_path(RESEARCH/'reference_entropy_v1')
    net,provenance=load_model(checkpoint(depth),depth)
    actual=sorted({int(m.group(1)) for k in net.state_dict() if (m:=re.match(r'^dec\.dec_1\.(\d+)\.',k)) and int(m.group(1))>0})
    if actual!=list(range(1,depth+1)):raise ValueError('Checkpoint prefix does not match declared model depth')
    net.update(0)
    return net,provenance

def preflight(out):
    # Hide devices before importing torch or any extension. The native CUDA
    # binaries are only hashed, not imported, in this mode.
    os.environ['CUDA_VISIBLE_DEVICES']='';os.environ['OMP_NUM_THREADS']='2'
    import torch
    torch.set_num_threads(2)
    binaries={}
    for backend in ('stock','patched'):
        path,report=binary(backend);binaries[backend]={'path':str(path),'sha256':sha(path),'setup_sha256':report['setup_sha256']}
    if binaries['stock']['setup_sha256']!=binaries['patched']['setup_sha256']:raise ValueError('Toolchain setup differs')
    rows=[]
    for depth in (2,4,6,12):
        net,p=cpu_model(depth);cdf=net.add_cdf_to_state_dict({})
        if any(t.device.type!='cpu' for t in cdf.values()):raise ValueError('CDF unexpectedly on GPU')
        rows.append({'depth':depth,'checkpoint':p,'retained_blocks_verified':depth,'cdf':{k:{'shape':list(t.shape),'dtype':str(t.dtype),'sha256':hashlib.sha256(t.contiguous().numpy().tobytes()).hexdigest()} for k,t in cdf.items()}})
        del net
    if torch.cuda.is_initialized():raise AssertionError('CPU preflight initialized CUDA')
    result={'state':'cpu_preflight_complete','completed_utc':utc(),'binaries':binaries,'models':rows,'cuda_initialized':False,'script_sha256':sha(__file__),'scope':'Strict model/prefix/CDF/binary preparation only. GPU exactness, graph recapture, residency and runtime are not tested.'}
    out.mkdir(parents=True,exist_ok=True);atomic(out/'cpu_preflight.json',result);print(json.dumps({'out':str(out),'models':len(rows),'cuda_initialized':False}))

def idle_device(index):
    if index not in (1,3,6):raise ValueError('GPU is outside the current study allocation')
    rows=subprocess.check_output(['nvidia-smi','--query-gpu=index,uuid,name,compute_cap','--format=csv,noheader,nounits'],text=True).strip().splitlines()
    fields=next((line.split(', ') for line in rows if int(line.split(',')[0])==index),None)
    if fields is None or fields[3]!='8.6':raise ValueError('Expected an SM86 study GPU')
    active=subprocess.check_output(['nvidia-smi','--query-compute-apps=gpu_uuid,pid','--format=csv,noheader,nounits'],text=True)
    if any(line.split(',')[0].strip()==fields[1] for line in active.splitlines()):raise RuntimeError('Selected GPU has an active compute process; no job was launched')
    return {'index':index,'uuid':fields[1],'name':fields[2],'compute_capability':fields[3]}

def load_native(backend):
    import torch
    path,report=binary(backend)
    if 'inference_extensions_cuda' in sys.modules:raise RuntimeError('Native extension already imported')
    spec=importlib.util.spec_from_file_location('inference_extensions_cuda',path)
    module=importlib.util.module_from_spec(spec);spec.loader.exec_module(module);sys.modules[spec.name]=module
    return module

def worker(args):
    import numpy as np
    import torch
    # Match the upstream inference precision, layout, CDF update and RNG setup.
    os.environ['CUBLAS_WORKSPACE_CONFIG']=':4096:8'
    module=load_native(args.backend)
    net,provenance=cpu_model(args.depth)
    from src.utils.common import set_torch_env
    from src.utils.transforms import rgb2ycbcr_np
    set_torch_env();device=torch.device('cuda:0')
    stream=torch.cuda.Stream(device=device,priority=-1);torch.cuda.set_stream(stream)
    net=net.half().to(device).to(memory_format=torch.channels_last)
    net.proxy=module.DMCIProxy();net.proxy.set_param(net.add_cdf_to_state_dict(net.state_dict()),0)
    request=json.loads(args.request.read_text());rows=[]
    args.output.mkdir(parents=True,exist_ok=True)
    if args.role=='decode':
        # No source paths or input tensors are supplied to this process.
        allowed={'id','height','width','qp','stream','stream_sha256','ec_parallel','checkpoint_sha256'}
        if any(set(r)!=allowed for r in request['cases']):raise ValueError('Decoder request contains unexpected information')
        def forbidden(*unused,**kwargs):raise RuntimeError('Source-side encoder disabled in decoder worker')
        net.compress=forbidden;net.enc.forward=forbidden;net.hyper_enc.forward=forbidden
    with torch.inference_mode():
        for i,case in enumerate(request['cases']):
            identity=case['id'];h,w,qp=case['height'],case['width'],case['qp']
            if args.role=='encode':
                if sha(case['crop'])!=case['crop_sha256']:raise ValueError('Source crop changed')
                rgb=np.load(case['crop'],allow_pickle=False)[:h,:w]
                x=torch.from_numpy(rgb2ycbcr_np(rgb.astype(np.float32)/255)-.5).permute(2,0,1)[None].contiguous().to(device=device,dtype=torch.float16,memory_format=torch.channels_last)
                pad_r,pad_b=net.get_padding_size(h,w,16)
                torch.cuda.synchronize();result=net.compress(x,qp,pad_b,pad_r);torch.cuda.synchronize()
                payload=result['bit_stream'];stream_path=args.output/(identity+'.bin');stream_path.write_bytes(payload)
                metadata={'stream':str(stream_path),'stream_sha256':sha(stream_path),'payload_bytes':len(payload),'ec_parallel':result['ec_parallel']}
            else:
                if case['checkpoint_sha256']!=provenance['sha256'] or sha(case['stream'])!=case['stream_sha256']:raise ValueError('Decoder stream/model identity differs')
                payload=Path(case['stream']).read_bytes();torch.cuda.synchronize()
                result=net.decompress(payload,{'height':h,'width':w},qp,case['ec_parallel']);torch.cuda.synchronize();metadata={}
            # A native output may borrow pool memory. Retain a CPU copy before
            # another call or shape recapture; never compare two aliases.
            reconstruction=result['x_hat'].detach().cpu().contiguous().numpy().copy()
            if not np.isfinite(reconstruction).all():raise ValueError('Nonfinite native reconstruction')
            path=args.output/(identity+'.npy');np.save(path,reconstruction,allow_pickle=False)
            rows.append({'id':identity,'height':h,'width':w,'qp':qp,'reconstruction':str(path),'reconstruction_sha256':sha(path),'shape':list(reconstruction.shape),'checkpoint_sha256':provenance['sha256'],**metadata})
            atomic(args.output/'progress.json',{'state':'running','completed':i+1,'total':len(request['cases']),'updated_utc':utc()})
    record={'state':'complete','role':args.role,'backend':args.backend,'depth':args.depth,'checkpoint':provenance,'binary_sha256':sha(binary(args.backend)[0]),'cases':rows,'precision':'FP16 channels_last; CDF built from FP32 CPU parameters; skip0','source_analysis_disabled':args.role=='decode','request_sha256':sha(args.request),'script_sha256':sha(__file__),'completed_utc':utc()}
    atomic(args.output/'manifest.json',record)

def correctness(args):
    import numpy as np
    device=idle_device(args.gpu)
    if args.output.exists():raise ValueError('Use a fresh output directory to preserve previous evidence')
    args.output.mkdir(parents=True)
    crop_manifest=json.loads((RESEARCH/'div2k100_center512_rgb/manifest.json').read_text())
    crops={c['image']:c for c in crop_manifest['images']};cases=[]
    for image in ('0801.png','0880.png'):
        c=crops[image]
        for h,w in SHAPES:
            for qp in QPS:
                cases.append({'id':f'{image[:4]}_{h}x{w}_q{qp}','height':h,'width':w,'qp':qp,'crop':c['crop_path'],'crop_sha256':c['crop_file_sha256']})
    # Return to the first geometry and quality after shape changes.
    cases.append({**cases[0],'id':'repeat_first_after_recapture'})
    encode_request=args.output/'encode_request.json';atomic(encode_request,{'cases':cases})
    progress={'state':'running','device':device,'started_utc':utc(),'completed_models':[],'scope':'Native payload/reconstruction correctness; synchronized calls are not a latency benchmark'}
    atomic(args.output/'progress.json',progress)
    outputs={};env=dict(os.environ,CUDA_VISIBLE_DEVICES=device['uuid'],OMP_NUM_THREADS='1',OPENBLAS_NUM_THREADS='1')
    try:
        for backend,depth in [('stock',12),('patched',12),('patched',2),('patched',4),('patched',6)]:
            # The parent never creates a CUDA context; check the allocated GPU
            # again before each sequential worker, rather than stealing capacity.
            idle_device(args.gpu)
            folder=args.output/f'{backend}_d{depth}';folder.mkdir()
            manifests={}
            for role in ('encode','decode'):
                idle_device(args.gpu)
                target=folder/role;request=encode_request if role=='encode' else folder/'decode_request.json'
                cmd=[sys.executable,str(Path(__file__).resolve()),'--role',role,'--backend',backend,'--depth',str(depth),'--request',str(request),'--output',str(target)]
                with (folder/(role+'.log')).open('w') as log:subprocess.run(cmd,env=env,stdout=log,stderr=subprocess.STDOUT,check=True)
                manifests[role]=json.loads((target/'manifest.json').read_text())
                if role=='encode':
                    fields=('id','height','width','qp','stream','stream_sha256','ec_parallel','checkpoint_sha256')
                    atomic(folder/'decode_request.json',{'cases':[{k:r[k] for k in fields} for r in manifests[role]['cases']]})
            for enc,dec in zip(manifests['encode']['cases'],manifests['decode']['cases']):
                if enc['id']!=dec['id'] or not np.array_equal(np.load(enc['reconstruction']),np.load(dec['reconstruction'])):raise AssertionError('Independent native decoder reconstruction differs')
            first,last=manifests['encode']['cases'][0],manifests['encode']['cases'][-1]
            if first['stream_sha256']!=last['stream_sha256'] or not np.array_equal(np.load(first['reconstruction']),np.load(last['reconstruction'])):raise AssertionError('Shape/QP recapture changes repeated case')
            outputs[backend,depth]=manifests
            progress['completed_models'].append({'backend':backend,'depth':depth,'exact_independent_decode_cases':len(cases),'repeat_after_recapture_equal':True})
            atomic(args.output/'progress.json',progress)
        for stock,patched in zip(outputs['stock',12]['encode']['cases'],outputs['patched',12]['encode']['cases']):
            if stock['stream_sha256']!=patched['stream_sha256'] or stock['ec_parallel']!=patched['ec_parallel'] or not np.array_equal(np.load(stock['reconstruction']),np.load(patched['reconstruction'])):raise AssertionError('Patched D12 differs from stock native control')
        progress.update(state='complete',finished_utc=utc(),stock_patched_d12_exact=True,script_sha256=sha(__file__),resident_expert_switching='not tested',latency='not measured',cases_per_model=len(cases))
        atomic(args.output/'progress.json',progress)
    except BaseException as e:
        progress.update(state='failed',error=repr(e),updated_utc=utc());atomic(args.output/'progress.json',progress);raise

def main():
    ap=argparse.ArgumentParser(description=__doc__);ap.add_argument('--preflight',action='store_true');ap.add_argument('--run-correctness',action='store_true');ap.add_argument('--gpu',type=int);ap.add_argument('--output',type=Path,default=RESEARCH/'native_depth_preflight');ap.add_argument('--role',choices=['encode','decode']);ap.add_argument('--backend',choices=['stock','patched']);ap.add_argument('--depth',type=int);ap.add_argument('--request',type=Path);args=ap.parse_args()
    if args.role:
        if not args.backend or args.depth not in (2,4,6,12) or args.request is None:ap.error('Worker requires backend, supported depth and request')
        worker(args)
    elif args.run_correctness:
        if args.preflight or args.gpu not in (1,3,6):ap.error('Correctness requires --gpu1/3/6 and cannot also request CPU preflight')
        correctness(args)
    else:preflight(args.output)
if __name__=='__main__':main()
