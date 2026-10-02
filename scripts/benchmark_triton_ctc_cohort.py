"""Dedicated-GPU CTC-wide synthesis benchmark for the opt-in Triton decoder.

Default preflight REFUSES any other compute process on the selected GPU. The
command is intentionally prepared now and run after the depth jobs free a card.
Results append as JSONL, so an interrupted multi-hour run can resume safely.
Only decoder synthesis is timed; latent preparation is outside the events.
"""
from __future__ import annotations

import argparse
import copy
from datetime import datetime, timezone
import json
import os
from pathlib import Path
import random
import statistics
import subprocess
import sys
import time

import torch
import torch.nn.functional as F

ROOT=Path(__file__).resolve().parents[1]
sys.path[:0]=[str(ROOT),str(Path.home()/'DCVC')]
import ctc_intra as C
from flexuf.config import FlexUFConfig
from flexuf.kernels import enable_fast_inference
from flexuf.model import FlexUFIntra,load_flexuf_state


def gpu_uuid_for_visible_index(index):
    visible=os.environ.get('CUDA_VISIBLE_DEVICES')
    physical=int(visible.split(',')[index]) if visible else index
    p=subprocess.run(['nvidia-smi','--query-gpu=index,uuid',
                      '--format=csv,noheader,nounits'],capture_output=True,
                     text=True,check=True)
    mapping={int(line.split(',',1)[0].strip()):line.split(',',1)[1].strip()
             for line in p.stdout.splitlines() if line.strip()}
    return mapping[physical]


def assert_gpu_exclusive(uuid):
    p=subprocess.run(['nvidia-smi','--query-compute-apps=pid,gpu_uuid',
                      '--format=csv,noheader,nounits'],capture_output=True,
                     text=True,check=True)
    others=[]
    for line in p.stdout.splitlines():
        if not line.strip():
            continue
        pid,app_uuid=[x.strip() for x in line.split(',',1)]
        if app_uuid==uuid and int(pid)!=os.getpid():
            others.append(int(pid))
    if others:
        raise RuntimeError(f'GPU {uuid} has other compute PIDs {others}; refusing shared-GPU benchmark')


def timed(fn):
    torch.cuda.synchronize()
    a,b=torch.cuda.Event(enable_timing=True),torch.cuda.Event(enable_timing=True)
    wall_start=time.perf_counter()
    a.record();fn();b.record();b.synchronize()
    wall_ms=(time.perf_counter()-wall_start)*1000
    return a.elapsed_time(b),wall_ms


def main():
    p=argparse.ArgumentParser()
    p.add_argument('--gpu',type=int,required=True,help='CUDA-visible GPU index')
    p.add_argument('--repeats',type=int,default=12)
    p.add_argument('--max-cases',type=int,default=0)
    p.add_argument('--sequence',default=None,help='Restrict to one named CTC sequence')
    p.add_argument('--qp',type=int,choices=(0,16,32,48,63),default=None)
    p.add_argument('--out',default='results/triton_ctc_cohort.jsonl')
    p.add_argument('--dry-run',action='store_true')
    p.add_argument('--resume',action='store_true')
    p.add_argument('--allow-shared',action='store_true',
                   help='Exploratory timing only; default requires an idle GPU')
    a=p.parse_args()
    assert a.repeats>=2
    rows=json.loads((ROOT/'flexplus/results/eval_rules_ctc_e15.json').read_text())['rows']
    pairs=[(r['seq'],r['qp']) for r in rows]
    assert len(pairs)==len(set(pairs))==265
    if a.sequence is not None:
        pairs=[pair for pair in pairs if pair[0]==a.sequence]
    if a.qp is not None:
        pairs=[pair for pair in pairs if pair[1]==a.qp]
    if not pairs:
        raise ValueError('no CTC sequence/QP matches the filters')
    seqs={s['name']:s for s in C.discover([])[0]}
    assert all(name in seqs for name,_ in pairs)
    if a.max_cases:
        pairs=pairs[:a.max_cases]
    out=ROOT/a.out
    if out.exists() and not a.resume and not a.dry_run:
        raise FileExistsError(f'{out} exists; pass --resume or choose a new --out')
    completed=set()
    if a.resume and out.exists():
        for line in out.read_text().splitlines():
            if line.strip():
                rec=json.loads(line)
                completed.add((rec['sequence'],rec['qp']))
    todo=[pair for pair in pairs if pair not in completed]
    if a.dry_run:
        print(json.dumps({'cases_total':len(pairs),'already_complete':len(completed),
                          'to_run':len(todo),'sequences':len({s for s,_ in pairs}),
                          'qp':sorted({q for _,q in pairs}),
                          'missing_router_maps':sum(r['rules']['router']['0.1'] is None
                                                    for r in rows if (r['seq'],r['qp']) in pairs)},indent=2))
        return
    if not todo:
        print('all selected cases already complete')
        return
    uuid=gpu_uuid_for_visible_index(a.gpu)
    if not a.allow_shared:
        assert_gpu_exclusive(uuid)
    torch.cuda.set_device(a.gpu)
    torch.backends.cudnn.allow_tf32=False
    torch.backends.cuda.matmul.allow_tf32=False
    dev=torch.device(f'cuda:{a.gpu}')
    ck=torch.load(ROOT/'runs/RECIPE512/ckpt_PIN_e15.pth.tar',
                  map_location='cpu',weights_only=False)
    cfg=FlexUFConfig(**ck['config'])
    net=FlexUFIntra(cfg).to(dev).eval()
    load_flexuf_state(net,ck)
    fast=copy.deepcopy(net.dec).eval()
    counts=enable_fast_inference(fast)
    by_pair={(r['seq'],r['qp']):r for r in rows}
    cached_frame=None
    out.parent.mkdir(parents=True,exist_ok=True)
    with torch.inference_mode(),out.open('a') as stream:
        for idx,(name,qp) in enumerate(todo):
            if not a.allow_shared and idx%10==0:
                assert_gpu_exclusive(uuid)
            seq=seqs[name]
            if cached_frame is None or cached_frame[0]!=name:
                x,planes=C.read_frames(seq['path'],seq['w'],seq['h'],1,1)
                cached_frame=(name,x,planes[0])
            _,frame,plane=cached_frame
            H,W=seq['h'],seq['w']
            x=frame.to(dev)
            xp=F.pad(x,(0,(-W)%cfg.rgb_patch,0,(-H)%cfg.rgb_patch),mode='replicate')
            y,q,_=net._encode_to_latent(xp,torch.tensor([qp],device=dev))
            archived=by_pair[(name,qp)]['rules']['router']['0.1']
            if archived is None:
                n=(xp.shape[-2]//cfg.rgb_patch)*(xp.shape[-1]//cfg.rgb_patch)
                mode=[cfg.num_exits-1]*n
                fallback=True
            else:
                mode=archived['map']
                fallback=False
            em=torch.tensor(mode,dtype=torch.long,device=dev)
            arm={'stock':lambda:net.dec(y,q,exit_map=em),
                 'fast':lambda:fast(y,q,exit_map=em)}
            reference=arm['stock']()
            optimized=arm['fast']()
            max_error=float((reference-optimized).abs().max())
            psnr_stock=C.psnr_611_420(reference[:,:,:H,:W],plane)
            psnr_fast=C.psnr_611_420(optimized[:,:,:H,:W],plane)
            for _ in range(2):
                arm['stock']();arm['fast']()
            timing={'stock':[],'fast':[]}
            wall={'stock':[],'fast':[]}
            rng=random.Random(20261002+idx)
            for _ in range(a.repeats):
                order=['stock','fast']
                rng.shuffle(order)
                for key in order:
                    cuda_ms,wall_ms=timed(arm[key])
                    timing[key].append(cuda_ms)
                    wall[key].append(wall_ms)
            ratio=[s/f for s,f in zip(timing['stock'],timing['fast'])]
            record={'timestamp_utc':datetime.now(timezone.utc).isoformat(),
                    'sequence':name,'qp':qp,'valid_hw':[H,W],
                    'padded_hw':list(xp.shape[-2:]),
                    'map_hist':torch.bincount(em,minlength=cfg.num_exits).tolist(),
                    'dense_fallback':fallback,
                    'stock_ms':timing['stock'],'fast_ms':timing['fast'],
                    'stock_median_ms':statistics.median(timing['stock']),
                    'fast_median_ms':statistics.median(timing['fast']),
                    'stock_wall_ms':wall['stock'],'fast_wall_ms':wall['fast'],
                    'stock_wall_median_ms':statistics.median(wall['stock']),
                    'fast_wall_median_ms':statistics.median(wall['fast']),
                    'paired_speedup_median':statistics.median(ratio),
                    'max_abs_output_error':max_error,
                    'delta_yuv611_db':psnr_fast-psnr_stock,
                    'fused_counts':counts,'gpu_uuid':uuid,
                    'scope':'synthesis only, TF32 disabled, dedicated GPU required unless allow-shared'}
            stream.write(json.dumps(record)+'\n')
            stream.flush()
            if idx%10==0 or idx==len(todo)-1:
                print(f'{idx+1}/{len(todo)} {name} qp{qp} '
                      f'{record["paired_speedup_median"]:.2f}x',flush=True)
            del xp,y,q,em,reference,optimized,x


if __name__=='__main__':main()
