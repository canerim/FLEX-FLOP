"""Persistent CPU decoder worker. Requests contain stream/output paths only."""
import argparse
import json
import os
from pathlib import Path
import sys
os.environ['CUDA_VISIBLE_DEVICES']=''
os.environ.setdefault('OMP_NUM_THREADS','2')
import numpy as np
import torch
from model_io import ROOT,load_model
from reference_codec import ReferenceCodec


def main():
    ap=argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--checkpoint',type=Path,required=True)
    ap.add_argument('--depth',type=int,required=True)
    ap.add_argument('--extension-dir',type=Path,default=ROOT/'research/reference_entropy_v1')
    args=ap.parse_args();torch.set_num_threads(2)
    net,source=load_model(args.checkpoint,args.depth)
    codec=ReferenceCodec(net,source['sha256'],args.extension_dir)
    def forbidden(*unused):raise AssertionError('Source analysis called inside isolated decoder')
    net.enc.forward=forbidden;net.hyper_enc.forward=forbidden;net.forward_one_frame=forbidden
    print(json.dumps({'ready':True,'checkpoint':source,'pid':os.getpid(),'cuda_initialized':torch.cuda.is_initialized()}),flush=True)
    for line in sys.stdin:
        request=json.loads(line)
        if request.get('stop'):break
        if set(request)!={'stream','output'}:raise ValueError('Only stream/output paths are accepted')
        recon,diagnostics=codec.decode(Path(request['stream']).read_bytes())
        output=Path(request['output']);output.parent.mkdir(parents=True,exist_ok=True)
        np.save(output,recon.numpy(),allow_pickle=False)
        if torch.cuda.is_initialized():raise AssertionError('Unexpected CUDA initialization')
        print(json.dumps({'ok':True,'output':str(output),'diagnostics':diagnostics,
                          'source_analysis_disabled':True,'cuda_initialized':False}),flush=True)


if __name__=='__main__':main()
