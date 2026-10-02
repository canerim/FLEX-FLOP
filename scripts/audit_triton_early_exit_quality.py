"""Five real CTC/archived-map checks of the optional fast synthesis path.

One decoder pass per case, no timing loop. This verifies output and the official
YUV 6:1:1 metric while live training remains undisturbed.
"""
from __future__ import annotations

import argparse
import copy
from datetime import datetime, timezone
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

CASES=[('videoSRC05_1920x1080_25.yuv',0),
       ('videoSRC01_1920x1080_30.yuv',32),
       ('videoSRC05_1920x1080_25.yuv',32),
       ('videoSRC10_1920x1080_30.yuv',32),
       ('videoSRC05_1920x1080_25.yuv',63)]


def main():
    p=argparse.ArgumentParser()
    p.add_argument('--gpu',type=int,default=0)
    p.add_argument('--out',default='results/triton_early_exit_quality_audit.json')
    a=p.parse_args()
    torch.cuda.set_device(a.gpu)
    torch.backends.cudnn.allow_tf32=False
    torch.backends.cuda.matmul.allow_tf32=False
    dev=torch.device(f'cuda:{a.gpu}')
    ck=torch.load(ROOT/'runs/RECIPE512/ckpt_PIN_e15.pth.tar',map_location='cpu',weights_only=False)
    cfg=FlexUFConfig(**ck['config'])
    net=FlexUFIntra(cfg).to(dev).eval()
    load_flexuf_state(net,ck)
    fast=copy.deepcopy(net.dec).eval()
    counts=enable_fast_inference(fast)
    archive=json.loads((ROOT/'flexplus/results/eval_rules_ctc_e15.json').read_text())
    ar={(r['seq'],r['qp']):r for r in archive['rows']}
    sequences={s['name']:s for s in C.discover([])[0]}
    rows=[]
    with torch.inference_mode():
        for name,qp in CASES:
            seq=sequences[name]
            x,planes=C.read_frames(seq['path'],seq['w'],seq['h'],1,1)
            x=x.to(dev)
            H,W=seq['h'],seq['w']
            xp=F.pad(x,(0,(-W)%cfg.rgb_patch,0,(-H)%cfg.rgb_patch),mode='replicate')
            y,q,_=net._encode_to_latent(xp,torch.tensor([qp],device=dev))
            em=torch.tensor(ar[(name,qp)]['rules']['router']['0.1']['map'],device=dev)
            base=net.dec(y,q,exit_map=em)[:,:,:H,:W]
            opt=fast(y,q,exit_map=em)[:,:,:H,:W]
            p0=C.psnr_611_420(base,planes[0])
            p1=C.psnr_611_420(opt,planes[0])
            rows.append({'sequence':name,'qp':qp,
                         'hist':torch.bincount(em,minlength=cfg.num_exits).tolist(),
                         'max_abs_raw':float((base-opt).abs().max()),
                         'mse_raw':float((base-opt).square().mean()),
                         'stock_yuv611_db':p0,'fast_yuv611_db':p1,
                         'delta_yuv611_db':p1-p0})
            print(name,qp,'max',rows[-1]['max_abs_raw'],'delta dB',p1-p0,flush=True)
    result={'timestamp_utc':datetime.now(timezone.utc).isoformat(),
            'scope':'real CTC first frames, archived source-calibrated router maps, output equivalence only; no runtime claim',
            'tf32_enabled':False,'checkpoint':'runs/RECIPE512/ckpt_PIN_e15.pth.tar',
            'fused_counts':counts,'cases':rows}
    out=ROOT/a.out
    out.parent.mkdir(parents=True,exist_ok=True)
    out.write_text(json.dumps(result,indent=2)+'\n')
    print('saved',out)


if __name__=='__main__':
    main()
