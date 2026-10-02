"""One-pass operator profile of the optional e15 Triton decoder.

This is an exploratory profile on a shared GPU, never a publication latency
measurement. It writes aggregate CUDA/CPU operator timings for prioritisation.
"""
from __future__ import annotations

import argparse
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
    p=argparse.ArgumentParser()
    p.add_argument('--gpu',type=int,default=0)
    p.add_argument('--sequence',default='videoSRC05_1920x1080_25.yuv')
    p.add_argument('--qp',type=int,default=32)
    p.add_argument('--out',default='results/triton_decoder_profile_shared.json')
    a=p.parse_args()
    torch.cuda.set_device(a.gpu)
    torch.backends.cudnn.allow_tf32=False
    torch.backends.cuda.matmul.allow_tf32=False
    dev=torch.device(f'cuda:{a.gpu}')
    ck=torch.load(ROOT/'runs/RECIPE512/ckpt_PIN_e15.pth.tar',map_location='cpu',weights_only=False)
    cfg=FlexUFConfig(**ck['config'])
    net=FlexUFIntra(cfg).to(dev).eval()
    load_flexuf_state(net,ck)
    counts=enable_fast_inference(net.dec)
    seq=next(s for s in C.discover([])[0] if s['name']==a.sequence)
    x,_=C.read_frames(seq['path'],seq['w'],seq['h'],1,1)
    x=x.to(dev)
    x=F.pad(x,(0,(-seq['w'])%cfg.rgb_patch,0,(-seq['h'])%cfg.rgb_patch),mode='replicate')
    ar=json.loads((ROOT/'flexplus/results/eval_rules_ctc_e15.json').read_text())
    row=next(r for r in ar['rows'] if r['seq']==a.sequence and r['qp']==a.qp)
    mode=row['rules']['router']['0.1']
    if mode is None:
        raise ValueError('this profiler requires an archived router map')
    em=torch.tensor(mode['map'],device=dev)
    with torch.inference_mode():
        y,q,_=net._encode_to_latent(x,torch.tensor([a.qp],device=dev))
        for _ in range(2):
            net.dec(y,q,exit_map=em)
        torch.cuda.synchronize()
        with torch.profiler.profile(
                activities=[torch.profiler.ProfilerActivity.CPU,
                            torch.profiler.ProfilerActivity.CUDA],
                record_shapes=False,profile_memory=False) as prof:
            net.dec(y,q,exit_map=em)
            torch.cuda.synchronize()
    rows=[]
    for ev in prof.key_averages():
        rows.append({'op':ev.key,'calls':ev.count,
                     'cuda_total_us':float(ev.device_time_total),
                     'cpu_total_us':float(ev.cpu_time_total),
                     'cpu_self_us':float(ev.self_cpu_time_total)})
    rows.sort(key=lambda r:r['cuda_total_us'],reverse=True)
    result={'scope':'one-pass shared-GPU operator profile; not a throughput benchmark',
            'sequence':a.sequence,'qp':a.qp,'map_hist':torch.bincount(em,minlength=cfg.num_exits).tolist(),
            'fused_counts':counts,'rows':rows}
    out=ROOT/a.out
    out.parent.mkdir(parents=True,exist_ok=True)
    out.write_text(json.dumps(result,indent=2)+'\n')
    for r in rows[:20]:
        print(f"{r['cuda_total_us']/1000:8.2f} ms {r['calls']:4d} {r['op']}")
    print('saved',out)


if __name__=='__main__':main()
