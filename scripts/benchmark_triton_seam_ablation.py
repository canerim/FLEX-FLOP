"""Pair the final fast decoder against itself with only seam fusion disabled."""
from __future__ import annotations

import copy
import json
from pathlib import Path
import random
import statistics
import sys
import time

import torch
import torch.nn.functional as F

ROOT=Path(__file__).resolve().parents[1]
sys.path[:0]=[str(ROOT),str(Path.home()/'DCVC')]
import ctc_intra as C
from flexuf.config import FlexUFConfig
from flexuf.kernels import enable_fast_inference
from flexuf.kernels.fused_seam import FusedGridSeamRepair
from flexuf.model import FlexUFIntra,load_flexuf_state


def timed(fn):
    torch.cuda.synchronize()
    a,b=torch.cuda.Event(enable_timing=True),torch.cuda.Event(enable_timing=True)
    start=time.perf_counter()
    a.record();fn();b.record();b.synchronize()
    return a.elapsed_time(b),(time.perf_counter()-start)*1000


def main():
    torch.cuda.set_device(0)
    torch.backends.cudnn.allow_tf32=False
    torch.backends.cuda.matmul.allow_tf32=False
    dev=torch.device('cuda:0')
    ck=torch.load(ROOT/'runs/RECIPE512/ckpt_PIN_e15.pth.tar',map_location='cpu',weights_only=False)
    cfg=FlexUFConfig(**ck['config'])
    net=FlexUFIntra(cfg).to(dev).eval();load_flexuf_state(net,ck)
    fused=copy.deepcopy(net.dec).eval();enable_fast_inference(fused)
    stock_seam=copy.deepcopy(fused).eval()
    assert isinstance(stock_seam.seam_repair,FusedGridSeamRepair)
    stock_seam.seam_repair=stock_seam.seam_repair.original
    seq=next(s for s in C.discover([])[0] if s['name']=='videoSRC05_1920x1080_25.yuv')
    x,_=C.read_frames(seq['path'],seq['w'],seq['h'],1,1)
    x=F.pad(x.to(dev),(0,(-seq['w'])%cfg.rgb_patch,0,(-seq['h'])%cfg.rgb_patch),mode='replicate')
    ar=json.loads((ROOT/'flexplus/results/eval_rules_ctc_e15.json').read_text())
    row=next(r for r in ar['rows'] if r['seq']==seq['name'] and r['qp']==32)
    em=torch.tensor(row['rules']['router']['0.1']['map'],device=dev)
    with torch.inference_mode():
        y,q,_=net._encode_to_latent(x,torch.tensor([32],device=dev))
        arms={'stock_seam':lambda:stock_seam(y,q,exit_map=em),
              'fused_seam':lambda:fused(y,q,exit_map=em)}
        ref=arms['stock_seam']();candidate=arms['fused_seam']()
        error=float((ref-candidate).abs().max())
        for _ in range(2):
            for fn in arms.values():fn()
        vals={name:{'cuda_ms':[],'wall_ms':[]} for name in arms}
        rng=random.Random(20261002)
        for _ in range(16):
            order=list(arms)
            rng.shuffle(order)
            for name in order:
                cuda_ms,wall_ms=timed(arms[name])
                vals[name]['cuda_ms'].append(cuda_ms)
                vals[name]['wall_ms'].append(wall_ms)
    paired=[s/f for s,f in zip(vals['stock_seam']['cuda_ms'],vals['fused_seam']['cuda_ms'])]
    out=ROOT/'results/triton_seam_ablation_shared.json'
    out.write_text(json.dumps({'scope':'shared GPU7; paired, same weights/map/decoder except seam wrapper',
                               'max_abs_output_error':error,
                               'stock_seam_median_ms':statistics.median(vals['stock_seam']['cuda_ms']),
                               'fused_seam_median_ms':statistics.median(vals['fused_seam']['cuda_ms']),
                               'paired_speedup_median':statistics.median(paired),
                               'measurements':vals},indent=2)+'\n')
    print('stock seam',statistics.median(vals['stock_seam']['cuda_ms']),
          'fused seam',statistics.median(vals['fused_seam']['cuda_ms']),
          'paired ratio',statistics.median(paired),'max error',error)
    print('saved',out)


if __name__=='__main__':main()
