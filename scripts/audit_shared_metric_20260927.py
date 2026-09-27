"""Replay two archived shared-exit cases to identify the actual loss colour space."""
from __future__ import annotations
import hashlib
import io
import json
import math
import os
from pathlib import Path
import subprocess
import sys
import tarfile
os.environ['CUDA_VISIBLE_DEVICES']=''
os.environ.setdefault('OMP_NUM_THREADS','2')
import torch
import torch.nn.functional as F

REPO=Path(__file__).resolve().parents[1]
ROOT=Path('/data10/shareddata/can_karsal/dcvcuf_depth_20260927/research/shared_metric_audit')
UPSTREAM=Path('/home/can_karsal/DCVC')
UPSTREAM_COMMIT='819c219b24db34310bbd15c51a720aaaf5eb2e7d'
REPO_COMMIT='9e17209'
OUT=REPO/'docs/research/2026-09-27-six-hour/shared_metric_audit'


def sha(path):
    with Path(path).open('rb') as f:return hashlib.file_digest(f,'sha256').hexdigest()


def ratio(candidate,reference,target):
    a=float((candidate-target).square().mean());b=float((reference-target).square().mean())
    return {'candidate_mse':a,'reference_mse':b,'loss_db':10*math.log10(a/b)}


def main():
    if subprocess.check_output(['git','-C',str(UPSTREAM),'rev-parse','HEAD'],text=True).strip()!=UPSTREAM_COMMIT:
        raise ValueError('Original shared-exit upstream changed')
    if subprocess.check_output(['git','-C',str(UPSTREAM),'diff','HEAD','--','src']):raise ValueError('Upstream sources dirty')
    ROOT.mkdir(parents=True,exist_ok=True);snapshot=ROOT/'source'
    if snapshot.exists():raise ValueError('Inspect existing audit before retry')
    snapshot.mkdir()
    archive=subprocess.check_output(['git','archive',REPO_COMMIT,'flexuf','ctc_intra.py'],cwd=REPO)
    with tarfile.open(fileobj=io.BytesIO(archive)) as tar:tar.extractall(snapshot,filter='data')
    sys.path.insert(0,str(snapshot));sys.path.insert(0,str(UPSTREAM))
    from flexuf.config import FlexUFConfig
    from flexuf.model import FlexUFIntra
    import ctc_intra as C
    from src.utils.transforms import ycbcr2rgb
    torch.set_num_threads(2);torch.manual_seed(42)
    checkpoint=REPO/'runs/RECIPE512/ckpt_PIN_e15.pth.tar'
    source=REPO/'flexplus/results/eval_rules_ctc_e15.json'
    ck=torch.load(checkpoint,map_location='cpu',weights_only=False);cfg=FlexUFConfig(**ck['config'])
    net=FlexUFIntra(cfg).eval();net.load_state_dict(ck.get('state_dict',ck.get('net',ck)),strict=True)
    records=json.loads(source.read_text());sequences,_=C.discover([]);lookup={s['name']:s for s in sequences}
    names=('BasketballPass_416x240_50.yuv','BQMall_832x480_60.yuv');rows=[]
    with torch.inference_mode():
        for name in names:
            old=next(r for r in records['rows'] if (r['seq'],r['qp'])==(name,32));case=lookup[name]
            x,planes=C.read_frames(case['path'],case['w'],case['h'],1,1);h,w=x.shape[-2:]
            xp=F.pad(x,(0,(-w)%cfg.rgb_patch,0,(-h)%cfg.rgb_patch),mode='replicate')
            qp=torch.tensor([32],dtype=torch.int32);y,q,_=net._encode_to_latent(xp,qp)
            dense=net.dec.forward_full(y,q)[:,:,:h,:w]
            saved=old['rules']['router']['0.1'];em=torch.tensor(saved['map'],dtype=torch.long)
            routed=net.dec(y,q,exit_map=em)[:,:,:h,:w]
            direct=ratio(routed,dense,x)
            unclipped=ratio(ycbcr2rgb(routed+.5,clamp=False),ycbcr2rgb(dense+.5,clamp=False),ycbcr2rgb(x+.5,clamp=False))
            clipped=ratio(ycbcr2rgb(routed.clamp(-.5,.5)+.5,clamp=True),ycbcr2rgb(dense.clamp(-.5,.5)+.5,clamp=True),ycbcr2rgb(x+.5,clamp=True))
            yuv611=C.psnr_611_420(dense,planes[0])-C.psnr_611_420(routed,planes[0])
            row={'sequence':name,'qp':32,'rule':'router','nominal_budget':.1,'map':saved['map'],
                'archived_db_rgb':saved['db_rgb'],'archived_db_611':saved['db_611'],
                'unclipped_ycbcr444':direct,'unclipped_rgb':unclipped,'clipped_rgb':clipped,
                'yuv611_420_loss_db':yuv611,'direct_minus_archive_db':direct['loss_db']-saved['db_rgb'],
                'first_frame_bytes_sha256':hashlib.sha256(Path(case['path']).open('rb').read(w*h*3//2)).hexdigest()}
            if abs(row['direct_minus_archive_db'])>2e-4:raise AssertionError('CPU replay does not reproduce archived direct-tensor metric')
            rows.append(row);print(json.dumps(row),flush=True)
    assert not torch.cuda.is_initialized()
    files=[checkpoint,source,Path(__file__),snapshot/'ctc_intra.py',snapshot/'flexuf/model.py',snapshot/'flexuf/eval.py',
        snapshot/'flexuf/backbone/decoder.py',UPSTREAM/'src/utils/transforms.py',UPSTREAM/'src/models/image_model.py']
    result={'finding':'Archived db_rgb and padded source-MSE tables use equal-channel unclipped YCbCr444 tensor MSE, not RGB MSE. The misleading field name must not determine the reported metric.',
        'evidence':'Input reader upsamples source420 planes to444 and centres by0.5; synthesis returns that representation; evaluation subtracts tensors directly with no RGB conversion. Two fixed-map CPU replays verify the archived ratio within2e-4dB.',
        'scope':'Two metric-provenance checks only; not a new full-cohort RGB evaluation, bitstream test or latency benchmark.',
        'repo_commit':subprocess.check_output(['git','rev-parse',REPO_COMMIT],cwd=REPO,text=True).strip(),
        'upstream_commit':UPSTREAM_COMMIT,'checkpoint_config':ck['config'],'strict_checkpoint_load':True,
        'torch':torch.__version__,'cuda_initialized':False,'rows':rows,'source_sha256':{str(p):sha(p) for p in files}}
    OUT.mkdir(parents=True,exist_ok=True);(OUT/'analysis.json').write_text(json.dumps(result,indent=2,allow_nan=False)+'\n')


if __name__=='__main__':main()
