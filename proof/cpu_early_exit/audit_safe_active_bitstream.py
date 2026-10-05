"""One frozen mixed-depth bitstream integration check for NaN-safe masking.

Runs kodim01/QP63 with the separate SafeActiveCanvasCoupler. The temporary
audit output is not a publication result because the original replay script
does not hash this replacement class. Only exact agreement with the pinned
120-case active-zero record is retained.
"""
from __future__ import annotations

import hashlib
import json
import sys
import tempfile
from dataclasses import replace
from pathlib import Path

HERE=Path(__file__).resolve().parent
ROOT=HERE.parents[1]
sys.path.insert(0,str(HERE))
sys.path.insert(0,str(ROOT))

from proof.cpu_early_exit import audit_active_canvas_kodak as audit
from proof.cpu_early_exit.active_canvas_coupling import ActiveCanvasCoupler
from proof.cpu_early_exit.active_canvas_safe import SafeActiveCanvasCoupler

RAW=HERE/'results/kodak_active_canvas_20261005.json'
OUT=HERE/'results/safe_active_bitstream_replay_20261005.json'


def sha(path:Path)->str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def compare_pixels(target:dict)->dict:
    import torch
    from model_io import load_model
    from reference_codec import ReferenceCodec
    from flexuf.config import FlexUFConfig
    from flexuf.model import FlexUFIntra,load_flexuf_state
    import flexuf.backbone.coupling as coupling_module

    stream=HERE/'results/kodak24_qp5/kodim01_qp63.fufref2'
    release=ROOT/'proof/early_exit_vs_released/artifacts/released_cvpr2026_image.pth.tar'
    e15=ROOT/'proof/early_exit_vs_released/artifacts/e15_epoch15.pth.tar'
    upstream=Path('/tmp/flexplus-proof-clean/proof/early_exit_vs_released/.local/DCVC')
    extension=Path('/tmp/flexplus-proof-clean/proof/early_exit_vs_released/.local/entropy')
    assert sha(stream)==target['stream_sha256']
    released,info=load_model(release,12,upstream)
    codec=ReferenceCodec(released,info['sha256'],extension)
    latent,q,_=codec.decode_latent(stream.read_bytes())
    ckpt=torch.load(e15,map_location='cpu',weights_only=False)
    cfg=FlexUFConfig(**ckpt['config'])
    model=FlexUFIntra(cfg).eval()
    load_flexuf_state(model,ckpt)
    dec=model.dec
    dec.cfg=replace(cfg,tile_coupling=True,sorted_tiles=False,trunk_halo=0)
    route=torch.tensor(target['exit_map'],dtype=torch.long)
    original_class=coupling_module.CanvasCoupler
    original_repair=dec.seam_repair
    checks={}
    try:
        with torch.inference_mode():
            for repair in (False,True):
                dec.seam_repair=original_repair if repair else None
                coupling_module.CanvasCoupler=ActiveCanvasCoupler
                frozen=dec(latent,q,exit_map=route)
                coupling_module.CanvasCoupler=SafeActiveCanvasCoupler
                safe=dec(latent,q,exit_map=route)
                delta=(frozen-safe).abs()
                checks['with_repair' if repair else 'no_repair']={
                    'pixel_exact':bool(torch.equal(frozen,safe)),
                    'max_abs_difference':float(delta.max())}
                assert torch.equal(frozen,safe)
    finally:
        coupling_module.CanvasCoupler=original_class
        dec.seam_repair=original_repair
        dec.cfg=cfg
    return checks


def main()->None:
    baseline=json.loads(RAW.read_text())
    assert baseline['complete'] and len(baseline['rows'])==120
    target=baseline['rows'][4]
    assert (target['image'],target['qp'])==('kodim01.png',63)
    with tempfile.TemporaryDirectory(prefix='safe-active-bitstream-') as name:
        path=Path(name)/'safe.json'
        partial={**baseline,'rows':baseline['rows'][:4],'complete':False}
        path.write_text(json.dumps(partial))
        audit.ActiveCanvasCoupler=SafeActiveCanvasCoupler
        sys.argv=['audit_active_canvas_kodak.py','--output',str(path),
                  '--max-new-cases','1']
        audit.main()
        result=json.loads(path.read_text())
    assert len(result['rows'])==5 and not result['complete']
    actual=result['rows'][4]
    assert (actual['image'],actual['qp'])==('kodim01.png',63)
    for key in ('exit_map','stream_sha256','scan_sha256'):
        assert actual[key]==target[key]
    for variant in ('no_repair','with_repair'):
        for metric in ('delta444_db','yuv611_db'):
            assert actual['coupled'][variant][metric]==target['coupled'][variant][metric]
    pixel_checks=compare_pixels(target)
    check={'scope':'One actual mixed-depth Kodak QP63 bitstream; full-decoder pixel equality and exact recorded metrics for NaN-safe versus frozen finite-input active-zero implementations. Not a whole-cohort proof or latency result.',
           'image':'kodim01.png','qp':63,'source_sha256':sha(RAW),
           'safe_implementation_sha256':sha(HERE/'active_canvas_safe.py'),
           'reference_implementation_sha256':sha(HERE/'active_canvas_coupling.py'),
           'recorded_metrics':target['coupled'],'metrics_exact':True,
           'full_decoder_pixel_checks':pixel_checks}
    OUT.write_text(json.dumps(check,indent=2)+'\n')
    print(json.dumps(check,indent=2))


if __name__=='__main__':main()
