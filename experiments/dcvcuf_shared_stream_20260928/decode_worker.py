"""Only stream/output paths enter this independent shared-exit decoder."""
import json,os,sys
from pathlib import Path
from codec import load,decode,torch
import numpy as np


def main():
    codec,synthesis,identities=load(source_free=True)
    print(json.dumps({'ready':True,'pid':os.getpid(),'identities':identities,'cuda_initialized':torch.cuda.is_initialized()}),flush=True)
    for line in sys.stdin:
        request=json.loads(line)
        if request=={'stop':True}:break
        if set(request)!={'stream','output'}:raise ValueError('Only stream/output paths accepted')
        out,trace,meta=decode(codec,synthesis,Path(request['stream']).read_bytes(),identities)
        np.save(request['output'],out.numpy(),allow_pickle=False)
        if torch.cuda.is_initialized():raise AssertionError('Unexpected CUDA initialization')
        print(json.dumps({'ok':True,'trace':trace,'meta':meta,'source_analysis_disabled':True,'cuda_initialized':False}),flush=True)


if __name__=='__main__':main()
