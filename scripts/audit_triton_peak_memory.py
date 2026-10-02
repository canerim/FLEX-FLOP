"""One-pass CUDA allocator peak of stock and opt-in fast e15 synthesis.

The two decoders stay resident; reported bytes are *incremental active
allocations above the pre-decode baseline*, excluding model weights and input
latent. This avoids conflating a fused kernel's temporary footprint with the
process' CUDACachingAllocator reservation or other GPU users' memory.
"""
from __future__ import annotations

import copy
import json
from pathlib import Path
import sys

import torch
import torch.nn.functional as F

ROOT=Path(__file__).resolve().parents[1]
sys.path[:0]=[str(ROOT),str(Path.home()/'DCVC')]
import ctc_intra as C
from flexuf.config import FlexUFConfig
from flexuf.kernels import enable_fast_inference
from flexuf.model import FlexUFIntra,load_flexuf_state


def main():
    torch.cuda.set_device(0)
    torch.backends.cudnn.allow_tf32=False
    torch.backends.cuda.matmul.allow_tf32=False
    dev=torch.device('cuda:0')
    ck=torch.load(ROOT/'runs/RECIPE512/ckpt_PIN_e15.pth.tar',map_location='cpu',weights_only=False)
    cfg=FlexUFConfig(**ck['config'])
    net=FlexUFIntra(cfg).to(dev).eval();load_flexuf_state(net,ck)
    fast=copy.deepcopy(net.dec).eval();enable_fast_inference(fast)
    seq=next(s for s in C.discover([])[0] if s['name']=='videoSRC05_1920x1080_25.yuv')
    x,_=C.read_frames(seq['path'],seq['w'],seq['h'],1,1)
    x=F.pad(x.to(dev),(0,(-seq['w'])%cfg.rgb_patch,0,(-seq['h'])%cfg.rgb_patch),mode='replicate')
    row=next(r for r in json.load(open(ROOT/'flexplus/results/eval_rules_ctc_e15.json'))['rows']
             if r['seq']==seq['name'] and r['qp']==32)
    em=torch.tensor(row['rules']['router']['0.1']['map'],device=dev)
    with torch.inference_mode():
        y,q,_=net._encode_to_latent(x,torch.tensor([32],device=dev))
        # Warm both paths so lazy cuDNN/Triton work is outside the peak.
        for dec in (net.dec,fast):
            z=dec(y,q,exit_map=em)
            del z
        torch.cuda.synchronize()
        metrics={}
        for name,dec in [('masked_stock',net.dec),('sorted_fused_full',fast)]:
            torch.cuda.reset_peak_memory_stats(dev)
            start=torch.cuda.memory_allocated(dev)
            out=dec(y,q,exit_map=em)
            torch.cuda.synchronize()
            peak=torch.cuda.max_memory_allocated(dev)
            metrics[name]={'active_before_bytes':start,
                           'incremental_peak_bytes':peak-start,
                           'output_bytes':out.numel()*out.element_size()}
            del out
            torch.cuda.synchronize()
    result={'scope':'single CTC QP32 decode; incremental active allocator peak, not reserved GPU memory',
            'device':torch.cuda.get_device_name(dev),'shape_rgb':[1280,2048],
            'map_hist':row['rules']['router']['0.1']['hist'],'metrics':metrics}
    out=ROOT/'results/triton_early_exit_peak_memory.json'
    out.write_text(json.dumps(result,indent=2)+'\n')
    print(json.dumps(result,indent=2))


if __name__=='__main__':main()
