"""Fixed QP32 cross-fit controls replayed on actual shared-exit reconstructions.

All 53 archived sequences, six fixed policy/calibration combinations. Selection
uses stored logits and cross-fit controls, never held-out distortion labels.
This CPU neural replay is not a bitstream or latency experiment.
"""
from __future__ import annotations
import hashlib
import json
import math
import os
from pathlib import Path
import subprocess
import sys
import time
from datetime import datetime, timezone
os.environ['CUDA_VISIBLE_DEVICES']=''
os.environ.setdefault('OMP_NUM_THREADS','2')
import numpy as np
import torch
import torch.nn.functional as F

REPO=Path(__file__).resolve().parents[1]
BASE=Path('/data10/shareddata/can_karsal/dcvcuf_depth_20260927/research')
OUT=BASE/'shared_crossfit_qp32'
SNAPSHOT=BASE/'shared_metric_audit/source'
UPSTREAM=Path('/home/can_karsal/DCVC')
BAYER=np.array([[0,32,8,40,2,34,10,42],[48,16,56,24,50,18,58,26],
 [12,44,4,36,14,46,6,38],[60,28,52,20,62,30,54,22],
 [3,35,11,43,1,33,9,41],[51,19,59,27,49,17,57,25],
 [15,47,7,39,13,45,5,37],[63,31,55,23,61,29,53,21]],dtype=float)


def sha(path):
    with Path(path).open('rb') as f:return hashlib.file_digest(f,'sha256').hexdigest()


def utc():return datetime.now(timezone.utc).isoformat()


def atomic(path,value):
    temp=path.with_suffix(path.suffix+'.tmp')
    temp.write_text(json.dumps(value,indent=2,allow_nan=False)+'\n');temp.replace(path)


def make_map(frame,policy,control,cost,j):
    # Do not read frame M or R in the decision path.
    if policy=='router':return (frame['lp'].double().numpy()-control*cost[None,j:]).argmax(1)+j
    nh,nw=frame['grid'];rank=np.tile((BAYER+.5)/64,(math.ceil(nh/8),math.ceil(nw/8)))[:nh,:nw].ravel()
    if policy=='dither':
        rung=min(int(math.floor(control)),3)
        return np.where(rank<control-rung,min(rung+1,3),rung)+j
    if policy=='uniform':return np.full(len(rank),int(control)+j,dtype=np.int64)
    raise ValueError(policy)


def main():
    torch.set_num_threads(2);torch.manual_seed(42)
    if subprocess.check_output(['git','-C',str(UPSTREAM),'rev-parse','HEAD'],text=True).strip()!='819c219b24db34310bbd15c51a720aaaf5eb2e7d':
        raise ValueError('Shared-exit upstream revision changed')
    if subprocess.check_output(['git','-C',str(UPSTREAM),'diff','HEAD','--','src']):raise ValueError('Upstream source changed')
    proof=json.loads((REPO/'docs/research/2026-09-27-six-hour/shared_metric_audit/analysis.json').read_text())
    for path,h in proof['source_sha256'].items():
        if path.startswith(str(SNAPSHOT)) or path.startswith(str(UPSTREAM)):
            if sha(path)!=h:raise ValueError('Metric-proof source changed: '+path)
    cp=REPO/'runs/RECIPE512/ckpt_PIN_e15.pth.tar'
    dump_path=REPO/'flexplus/results/router_dump_e15_ce_soft.pt'
    calibration_path=REPO/'docs/research/2026-09-27-six-hour/crossfit_control/analysis.json'
    calibration=json.loads(calibration_path.read_text())
    if sha(dump_path)!=calibration['source_sha256']:raise ValueError('Router dump changed')
    if sha(cp)!=proof['source_sha256'][str(cp)]:raise ValueError('Checkpoint changed')
    dump=torch.load(dump_path,map_location='cpu',weights_only=False)
    names=dump['names'];cost=dump['cost'].double().numpy();j=dump['j']
    if names!=calibration['names'] or len(names)!=53:raise ValueError('Unexpected sequence cohort')
    controls=[r for r in calibration['rows'] if r['qp']==32 and r['budget']==.1]
    if len(controls)!=318:raise ValueError('Expected all 53 x 6 policy/calibration cases')
    # Freeze every map before opening any target image. Table labels are read
    # below only to audit agreement with the already completed cross-fit study.
    maps={}
    for r in controls:
        index=r['sequence_index'];frame=dump['frames'][32][index]
        em=make_map(frame,r['policy'],r['control'],cost,j)
        no_labels={k:v for k,v in frame.items() if k not in ('M','R')}
        assert np.array_equal(em,make_map(no_labels,r['policy'],r['control'],cost,j))
        maps[index,r['criterion'],r['policy']]=em
        expected=10*math.log10(float(frame['M'].double().numpy()[np.arange(len(em)),em].mean())/frame['R'])
        if abs(expected-r['loss_db'])>1e-10:raise ValueError('Map does not reproduce selected table loss')
        if abs(100*(1-cost[em].mean())-r['saving_points'])>1e-10:raise ValueError('Map saving differs')
    sys.path.insert(0,str(SNAPSHOT));sys.path.insert(0,str(UPSTREAM))
    from flexuf.config import FlexUFConfig
    from flexuf.model import FlexUFIntra
    import ctc_intra as C
    from src.utils.transforms import ycbcr2rgb
    ck=torch.load(cp,map_location='cpu',weights_only=False);cfg=FlexUFConfig(**ck['config'])
    net=FlexUFIntra(cfg).eval();net.load_state_dict(ck.get('state_dict',ck.get('net',ck)),strict=True)
    sequences,_=C.discover([]);lookup={s['name']:s for s in sequences}
    files=[Path(__file__),cp,dump_path,calibration_path,*SNAPSHOT.rglob('*.py'),UPSTREAM/'src/utils/transforms.py']
    manifest={'scope':'Fixed-router sequence-cross-fit actual mixed reconstruction replay; CPU, no real bitstream or runtime claim',
        'qp':32,'nominal_budget_db':.1,'names':names,'criteria':['mean','q90'],'policies':['router','dither','uniform'],
        'total_sequences':53,'total_policy_cases':318,'map_selection':'Frozen cross-fit controls and stored logits only; no held-out M/R accessed by make_map',
        'checkpoint_config':ck['config'],'strict_checkpoint_load':True,'source_sha256':{str(p):sha(p) for p in files},
        'torch':torch.__version__,'threads':2,'metric':'Unclipped equal-channel YCbCr444 MSE ratio and explicitly clipped RGB MSE ratio, both cropped and relative to same e15 full-frame output. Additional padded ratio against archived released MSE R isolates table prediction discrepancy.',
        'interpretation':'QP32 is predeclared; all53 sequences retained. This development corpus is not an untouched external test. No final-image threshold filtering, fallback or map reselection.'}
    OUT.mkdir(parents=True,exist_ok=True);(OUT/'cases').mkdir(exist_ok=True)
    if (OUT/'manifest.json').exists() and json.loads((OUT/'manifest.json').read_text())!=manifest:raise ValueError('Existing output has different provenance')
    atomic(OUT/'manifest.json',manifest)
    progress={'state':'running','pid':os.getpid(),'started_utc':utc(),'completed_sequences':0,'total_sequences':53,'completed_policy_cases':0}
    atomic(OUT/'progress.json',progress);start=time.perf_counter()
    # Small-to-large order checks the replay early without choosing by outcome.
    order=sorted(range(53),key=lambda i:(lookup[names[i]]['w']*lookup[names[i]]['h'],names[i]))
    try:
        with torch.inference_mode():
            for index in order:
                name=names[index];case=lookup[name];path=OUT/'cases'/f'{index:02d}.json'
                h,w=case['h'],case['w']
                with Path(case['path']).open('rb') as f:frame_hash=hashlib.sha256(f.read(w*h*3//2)).hexdigest()
                if path.exists():
                    rec=json.loads(path.read_text())
                    if rec['sequence']!=name or rec['first_frame_bytes_sha256']!=frame_hash:raise ValueError('Existing frame provenance changed')
                else:
                    x,planes=C.read_frames(case['path'],w,h,1,1)
                    xp=F.pad(x,(0,(-w)%cfg.rgb_patch,0,(-h)%cfg.rgb_patch),mode='replicate')
                    y,q,_=net._encode_to_latent(xp,torch.tensor([32],dtype=torch.int32))
                    dense=net.dec.forward_full(y,q);densecrop=dense[:,:,:h,:w]
                    targetrgb=ycbcr2rgb(x+.5,clamp=True)
                    densergb=ycbcr2rgb(densecrop.clamp(-.5,.5)+.5,clamp=True)
                    anchor444=float((densecrop-x).square().mean());anchorrgb=float((densergb-targetrgb).square().mean())
                    anchor611=C.psnr_611_420(densecrop,planes[0]);cache={};rows=[]
                    for r in [r for r in controls if r['sequence_index']==index]:
                        em=maps[index,r['criterion'],r['policy']];key=tuple(em.tolist())
                        if key not in cache:
                            out=net.dec(y,q,exit_map=torch.tensor(em,dtype=torch.long))
                            crop=out[:,:,:h,:w]
                            rgb=ycbcr2rgb(crop.clamp(-.5,.5)+.5,clamp=True)
                            mse444=float((crop-x).square().mean());msergb=float((rgb-targetrgb).square().mean())
                            padded=float((out-xp).square().mean())
                            frame=dump['frames'][32][index]
                            cache[key]={'cropped_ycbcr444_mse':mse444,'cropped_rgb_mse':msergb,
                                'cropped_ycbcr444_loss_db':10*math.log10(mse444/anchor444),
                                'cropped_rgb_loss_db':10*math.log10(msergb/anchorrgb),
                                'cropped_yuv611_loss_db':anchor611-C.psnr_611_420(crop,planes[0]),
                                'actual_padded_mse':padded,'actual_padded_loss_vs_archived_R_db':10*math.log10(padded/frame['R'])}
                        metrics=cache[key]
                        rows.append({**r,'map':list(key),**metrics,
                            'actual_minus_table_padded_loss_db':metrics['actual_padded_loss_vs_archived_R_db']-r['loss_db']})
                    rec={'sequence':name,'sequence_index':index,'qp':32,'height':h,'width':w,
                        'first_frame_bytes_sha256':frame_hash,'unique_maps':len(cache),
                        'anchor_cropped_ycbcr444_mse':anchor444,'anchor_cropped_rgb_mse':anchorrgb,
                        'anchor_cropped_yuv611_psnr':anchor611,'rows':rows,'finished_utc':utc()}
                    atomic(path,rec)
                progress.update(completed_sequences=progress['completed_sequences']+1,
                    completed_policy_cases=progress['completed_policy_cases']+len(rec['rows']),sequence=name,
                    updated_utc=utc(),elapsed_seconds=time.perf_counter()-start)
                atomic(OUT/'progress.json',progress);print(json.dumps(progress),flush=True)
        if torch.cuda.is_initialized():raise AssertionError('Unexpected GPU initialization')
        for p,h in manifest['source_sha256'].items():
            if sha(p)!=h:raise ValueError('Run input changed: '+p)
        if progress['completed_policy_cases']!=318:raise AssertionError('Incomplete policy cohort')
        progress.update(state='complete',finished_utc=utc());atomic(OUT/'progress.json',progress)
    except BaseException as exc:
        progress.update(state='failed',error=repr(exc),updated_utc=utc());atomic(OUT/'progress.json',progress);raise


if __name__=='__main__':main()
