"""Interleaved decoder-only benchmark that resists shared-GPU timing drift.

Both models are resident simultaneously and each block shuffles arm order. This
still is not an isolated GPU result when the depth training is active; raw times
and per-block paired ratios are saved for an honest follow-up audit.
"""
from __future__ import annotations

import argparse
import copy
from dataclasses import replace
from datetime import datetime, timezone
import json
import math
from pathlib import Path
import random
import statistics
import sys

import numpy as np
from PIL import Image, ImageOps
import torch
import torch.nn.functional as F

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT), str(Path.home() / 'DCVC')]

from flexuf.config import FlexUFConfig
from flexuf.kernels.planned_decoder import forward_with_cpu_map
from flexuf.kernels.wsilu_chunkadd import install_fused_plain_wsilu, install_fused_wsilu
from flexuf.kernels.fused_ffn import install_fused_ffn
from flexuf.kernels.fused_pwout import install_fused_trunk_blocks
from flexuf.kernels.fused_adapters import install_fused_adapters
from flexuf.model import FlexUFIntra, load_flexuf_state
import ctc_intra as C


def timed(fn):
    torch.cuda.synchronize()
    a, b = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
    a.record()
    fn()
    b.record()
    b.synchronize()
    return a.elapsed_time(b)


def summary(vals):
    return {'median': statistics.median(vals), 'min': min(vals), 'max': max(vals),
            'samples': vals}


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--checkpoint', default='runs/RECIPE512/ckpt_PIN_e15.pth.tar')
    p.add_argument('--image', default='data/kodak/kodim01.png')
    p.add_argument('--ctc-seq', default=None,
                   help='Use the actual first CTC frame and its archived QP32 router map')
    p.add_argument('--gpu', type=int, default=0)
    p.add_argument('--qp', type=int, default=32)
    p.add_argument('--width', type=int, default=512)
    p.add_argument('--height', type=int, default=512)
    p.add_argument('--blocks', type=int, default=12)
    p.add_argument('--out', default='results/triton_early_exit_paired.json')
    a = p.parse_args()
    assert a.width % 256 == 0 and a.height % 256 == 0
    torch.cuda.set_device(a.gpu)
    torch.backends.cudnn.allow_tf32 = False
    torch.backends.cuda.matmul.allow_tf32 = False
    dev = torch.device(f'cuda:{a.gpu}')
    ck = torch.load(ROOT/a.checkpoint, map_location='cpu', weights_only=False)
    cfg = FlexUFConfig(**ck['config'])
    net = FlexUFIntra(cfg).to(dev).eval()
    load_flexuf_state(net,ck)
    if a.ctc_seq:
        seq = next(s for s in C.discover([])[0] if s['name'] == a.ctc_seq)
        valid, planes = C.read_frames(seq['path'], seq['w'], seq['h'], 1, 1)
        valid = valid.to(dev)
        h_valid,w_valid = seq['h'],seq['w']
        a.width = math.ceil(w_valid/cfg.rgb_patch)*cfg.rgb_patch
        a.height = math.ceil(h_valid/cfg.rgb_patch)*cfg.rgb_patch
        x = F.pad(valid,(0,a.width-w_valid,0,a.height-h_valid),mode='replicate')
        archive = json.loads((ROOT/'flexplus/results/eval_rules_ctc_e15.json').read_text())
        row = next(r for r in archive['rows'] if r['seq']==a.ctc_seq and r['qp']==a.qp)
        em_cpu = torch.tensor(row['rules']['router']['0.1']['map'],dtype=torch.long)
    else:
        img = ImageOps.fit(Image.open(ROOT/a.image).convert('YCbCr'),
                           (a.width,a.height), method=Image.Resampling.LANCZOS)
        x = torch.from_numpy(np.asarray(img).copy()).permute(2,0,1)[None].to(dev).float()/255-0.5
        h_valid,w_valid = a.height,a.width
    with torch.inference_mode():
        y,q,_ = net._encode_to_latent(x,torch.tensor([a.qp],device=dev))
    stock = net.dec
    masked = copy.deepcopy(stock).eval()
    fused = copy.deepcopy(stock).eval()
    ffn = copy.deepcopy(stock).eval()
    trunk = copy.deepcopy(stock).eval()
    chunk_count = install_fused_wsilu(fused)
    plain_count = install_fused_plain_wsilu(fused)
    ffn_count = install_fused_ffn(ffn)
    ffn_plain_count = install_fused_plain_wsilu(ffn)
    trunk_ffn_count = install_fused_ffn(trunk)
    trunk_plain_count = install_fused_plain_wsilu(trunk)
    trunk_block_count = install_fused_trunk_blocks(trunk)
    full = copy.deepcopy(trunk).eval()
    adapter_count = install_fused_adapters(full)
    stock.cfg = replace(cfg,sorted_tiles=True)
    fused.cfg = replace(cfg,sorted_tiles=True)
    ffn.cfg = replace(cfg,sorted_tiles=True)
    trunk.cfg = replace(cfg,sorted_tiles=True)
    full.cfg = replace(cfg,sorted_tiles=True)
    masked.cfg = replace(cfg,sorted_tiles=False)
    n=(a.width//cfg.rgb_patch)*(a.height//cfg.rgb_patch)
    if a.ctc_seq:
        assert em_cpu.numel()==n
        em=em_cpu.to(dev)
    else:
        choices=torch.tensor([2]*50+[3]*30+[4]*15+[5]*5,device=dev)
        em=choices[torch.arange(n,device=dev)*37%100].long()
        em_cpu=em.cpu()
    arms={
        'masked_stock':lambda:masked(y,q,exit_map=em),
        'sorted_stock':lambda:stock(y,q,exit_map=em),
        'sorted_fused':lambda:fused(y,q,exit_map=em),
        'sorted_ffn':lambda:ffn(y,q,exit_map=em),
        'sorted_trunk':lambda:trunk(y,q,exit_map=em),
        'sorted_full':lambda:full(y,q,exit_map=em),
        'planned_stock':lambda:forward_with_cpu_map(stock,y,q,em_cpu),
        'planned_fused':lambda:forward_with_cpu_map(fused,y,q,em_cpu),
        'planned_ffn':lambda:forward_with_cpu_map(ffn,y,q,em_cpu),
        'planned_trunk':lambda:forward_with_cpu_map(trunk,y,q,em_cpu),
        'planned_full':lambda:forward_with_cpu_map(full,y,q,em_cpu),
    }
    with torch.inference_mode():
        ref=arms['sorted_stock']()
        test=arms['sorted_fused']()
        ffn_output=arms['sorted_ffn']()
        trunk_output=arms['sorted_trunk']()
        full_output=arms['sorted_full']()
        pstock=arms['planned_stock']()
        pfused=arms['planned_fused']()
        pffn=arms['planned_ffn']()
        ptrunk=arms['planned_trunk']()
        pfull=arms['planned_full']()
        masked_output=arms['masked_stock']()
        errors={'fused_max_abs':float((ref-test).abs().max()),
                'fused_mse':float((ref-test).square().mean()),
                'ffn_max_abs':float((ref-ffn_output).abs().max()),
                'ffn_mse':float((ref-ffn_output).square().mean()),
                'trunk_max_abs':float((ref-trunk_output).abs().max()),
                'trunk_mse':float((ref-trunk_output).square().mean()),
                'full_max_abs':float((ref-full_output).abs().max()),
                'full_mse':float((ref-full_output).square().mean()),
                'planned_stock_max_abs':float((ref-pstock).abs().max()),
                'planned_fused_max_abs':float((test-pfused).abs().max()),
                'planned_ffn_max_abs':float((ffn_output-pffn).abs().max()),
                'planned_trunk_max_abs':float((trunk_output-ptrunk).abs().max()),
                'planned_full_max_abs':float((full_output-pfull).abs().max()),
                'masked_stock_max_abs':float((ref-masked_output).abs().max())}
        source_mse=lambda z:float((z[:,:,:h_valid,:w_valid].clamp(-.5,.5)-x[:,:,:h_valid,:w_valid]).square().mean())
        errors['source_psnr_delta_db']=10*math.log10(source_mse(ref)/source_mse(test))
        errors['source_psnr_ffn_delta_db']=10*math.log10(source_mse(ref)/source_mse(ffn_output))
        errors['source_psnr_trunk_delta_db']=10*math.log10(source_mse(ref)/source_mse(trunk_output))
        errors['source_psnr_full_delta_db']=10*math.log10(source_mse(ref)/source_mse(full_output))
        if a.ctc_seq:
            errors['source_psnr_611_stock_db']=C.psnr_611_420(ref[:,:,:h_valid,:w_valid],planes[0])
            errors['source_psnr_611_fused_db']=C.psnr_611_420(test[:,:,:h_valid,:w_valid],planes[0])
            errors['source_psnr_611_ffn_db']=C.psnr_611_420(ffn_output[:,:,:h_valid,:w_valid],planes[0])
            errors['source_psnr_611_trunk_db']=C.psnr_611_420(trunk_output[:,:,:h_valid,:w_valid],planes[0])
            errors['source_psnr_611_full_db']=C.psnr_611_420(full_output[:,:,:h_valid,:w_valid],planes[0])
        for _ in range(3):
            for fn in arms.values(): fn()
        vals={name:[] for name in arms}
        rng=random.Random(20261002)
        orders=[]
        for _ in range(a.blocks):
            order=list(arms)
            rng.shuffle(order)
            orders.append(order)
            for name in order:
                vals[name].append(timed(arms[name]))
    ratio_sorted=[s/f for s,f in zip(vals['sorted_stock'],vals['sorted_fused'])]
    ratio_planned=[s/f for s,f in zip(vals['planned_stock'],vals['planned_fused'])]
    ratio_total=[s/f for s,f in zip(vals['masked_stock'],vals['sorted_fused'])]
    ratio_ffn=[s/f for s,f in zip(vals['sorted_stock'],vals['sorted_ffn'])]
    ratio_ffn_total=[s/f for s,f in zip(vals['masked_stock'],vals['sorted_ffn'])]
    ratio_trunk=[s/f for s,f in zip(vals['sorted_stock'],vals['sorted_trunk'])]
    ratio_trunk_total=[s/f for s,f in zip(vals['masked_stock'],vals['sorted_trunk'])]
    ratio_full=[s/f for s,f in zip(vals['sorted_stock'],vals['sorted_full'])]
    ratio_full_total=[s/f for s,f in zip(vals['masked_stock'],vals['sorted_full'])]
    result={'timestamp_utc':datetime.now(timezone.utc).isoformat(),
            'scope':'paired, shuffled, decoder-only CUDA-event timing on shared GPU; no entropy/router execution',
            'device':torch.cuda.get_device_name(dev),'tf32_enabled':False,
            'image':a.image if not a.ctc_seq else a.ctc_seq,
            'qp':a.qp,
            'map_provenance':f'archived CTC router QP{a.qp} budget0.1' if a.ctc_seq else 'synthetic',
            'shape_rgb':[a.height,a.width],'valid_shape':[h_valid,w_valid],'n_tiles':n,
            'map_counts':torch.bincount(em,minlength=cfg.num_exits).tolist(),
            'checkpoint':a.checkpoint,'fused_chunk_count':chunk_count,
            'fused_plain_count':plain_count,'fused_ffn_count':ffn_count,
            'ffn_plain_count':ffn_plain_count,
            'trunk_ffn_count':trunk_ffn_count,'trunk_plain_count':trunk_plain_count,
            'fused_trunk_block_count':trunk_block_count,'errors':errors,
            'fused_adapter_count':adapter_count,
            'timing_ms':{name:summary(v) for name,v in vals.items()},
            'paired_speedup':{'sorted':summary(ratio_sorted),'planned':summary(ratio_planned),
                              'masked_to_fused':summary(ratio_total),
                              'sorted_to_ffn':summary(ratio_ffn),
                              'masked_to_ffn':summary(ratio_ffn_total),
                              'sorted_to_trunk':summary(ratio_trunk),
                              'masked_to_trunk':summary(ratio_trunk_total),
                              'sorted_to_full':summary(ratio_full),
                              'masked_to_full':summary(ratio_full_total)},
            'order_per_block':orders}
    out=ROOT/a.out
    out.parent.mkdir(parents=True,exist_ok=True)
    out.write_text(json.dumps(result,indent=2)+'\n')
    print(json.dumps({'errors':errors,'medians_ms':{k:statistics.median(v) for k,v in vals.items()},
                      'paired_speedup_median':{'sorted':statistics.median(ratio_sorted),
                                               'planned':statistics.median(ratio_planned),
                                               'masked_to_fused':statistics.median(ratio_total),
                                               'sorted_to_ffn':statistics.median(ratio_ffn),
                                               'masked_to_ffn':statistics.median(ratio_ffn_total),
                                               'sorted_to_trunk':statistics.median(ratio_trunk),
                                               'masked_to_trunk':statistics.median(ratio_trunk_total),
                                               'sorted_to_full':statistics.median(ratio_full),
                                               'masked_to_full':statistics.median(ratio_full_total)}},indent=2))
    print('saved',out)


if __name__=='__main__':
    main()
