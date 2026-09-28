"""Bank decoder receives only a container path and output path, never source pixels."""
import hashlib,json,os,sys
from collections import OrderedDict
from pathlib import Path
os.environ['CUDA_VISIBLE_DEVICES']=''
os.environ.setdefault('OMP_NUM_THREADS','2')
ROOT=Path('/data10/shareddata/can_karsal/dcvcuf_depth_20260927')
SOURCE=ROOT/'research/native_shape_reference/source'
sys.path.insert(0,str(SOURCE))
import numpy as np
import torch
from model_io import load_model
from reference_codec import ReferenceCodec,parse_container
from layout import unpack

def main():
    torch.set_num_threads(2);codecs={};sources={};cache=OrderedDict()
    for depth in (2,6):
        net,info=load_model(ROOT/f'runs/d{depth}/weights_epoch020.pt',depth)
        codecs[depth]=ReferenceCodec(net,info['sha256'],ROOT/'research/reference_entropy_v1');sources[depth]=info
        def forbidden(*args,**kwargs):raise AssertionError('Source analysis forbidden in independent bank decoder')
        net.enc.forward=forbidden;net.hyper_enc.forward=forbidden;net.forward_one_frame=forbidden
    print(json.dumps({'ready':True,'pid':os.getpid(),'checkpoints':sources,'cuda_initialized':False}),flush=True)
    for line in sys.stdin:
        request=json.loads(line)
        if request.get('stop'):break
        if set(request)!={'stream','output'}:raise ValueError('Only bank stream/output paths accepted')
        profile,parts=unpack(Path(request['stream']).read_bytes());canvas=torch.empty((1,3,512,512));coverage=torch.zeros((512,512),dtype=torch.int8);diagnostics=[];qps=set()
        for region,stream in parts:
            depth=region['depth'];meta=parse_container(stream,depth,sources[depth]['sha256']);qps.add(meta['qp'])
            t,l,b,r=region['core'];wt,wl,wb,wr=region['window']
            if (meta['height'],meta['width'])!=(wb-wt,wr-wl):raise ValueError('Embedded stream geometry differs from profile')
            key=hashlib.sha256(stream).hexdigest();cached=key in cache
            if cached:rec,diag=cache.pop(key)
            else:rec,diag=codecs[depth].decode(stream)
            cache[key]=(rec,diag)
            while len(cache)>32:cache.popitem(last=False)
            oy,ox=t-wt,l-wl;canvas[:,:,t:b,l:r]=rec[:,:,oy:oy+b-t,ox:ox+r-l];coverage[t:b,l:r]+=1
            diagnostics.append({'stream_sha256':key,'depth':depth,'cached_from_prior_byte_decode':cached,**diag})
        if len(qps)!=1 or not torch.all(coverage==1):raise ValueError('Bank QP or coverage mismatch')
        if torch.cuda.is_initialized():raise AssertionError('Unexpected CUDA initialization')
        np.save(request['output'],canvas.numpy(),allow_pickle=False)
        print(json.dumps({'ok':True,'profile':profile,'parts':diagnostics,'source_analysis_disabled':True,'cuda_initialized':False}),flush=True)
if __name__=='__main__':main()
