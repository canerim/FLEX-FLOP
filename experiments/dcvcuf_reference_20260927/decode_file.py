"""Decode a reference stream in a fresh CPU process; no source image accepted."""
from __future__ import annotations
import argparse
import json
import os
from pathlib import Path
os.environ['CUDA_VISIBLE_DEVICES']=''
os.environ.setdefault('OMP_NUM_THREADS','2')
import numpy as np
import torch
from model_io import ROOT,UPSTREAM,load_model
from reference_codec import ReferenceCodec


def main():
    ap=argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--checkpoint',type=Path,required=True)
    ap.add_argument('--depth',type=int,required=True)
    ap.add_argument('--stream',type=Path,required=True)
    ap.add_argument('--output',type=Path,required=True)
    ap.add_argument('--upstream',type=Path,default=UPSTREAM)
    ap.add_argument('--extension-dir',type=Path,default=ROOT/'research/reference_entropy_v1')
    args=ap.parse_args()
    torch.set_num_threads(2)
    net,source=load_model(args.checkpoint,args.depth,args.upstream)
    codec=ReferenceCodec(net,source['sha256'],args.extension_dir)
    # These modules must never run in a bitstream-only decoder.
    def forbidden(*unused):raise AssertionError('Decoder attempted to run source analysis')
    net.enc.forward=forbidden
    net.hyper_enc.forward=forbidden
    net.forward_one_frame=forbidden
    recon,diagnostics=codec.decode(args.stream.read_bytes())
    args.output.parent.mkdir(parents=True,exist_ok=True)
    np.save(args.output,recon.numpy(),allow_pickle=False)
    diagnostics.update(checkpoint=source,cuda_initialized=torch.cuda.is_initialized(),
                       decoder_has_no_source_argument=True,source_analysis_modules_disabled=True)
    if diagnostics['cuda_initialized']:raise AssertionError('Unexpected CUDA initialization')
    args.output.with_suffix('.json').write_text(json.dumps(diagnostics,indent=2)+'\n')
    print(json.dumps({'decoded':str(args.output),'cuda_initialized':False,'source_analysis_modules_disabled':True}))


if __name__=='__main__':main()
