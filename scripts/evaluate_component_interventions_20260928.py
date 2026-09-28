"""Frozen-checkpoint 2x2 adapter/repair intervention; not a retraining ablation.

All53 QP32 frames, one fixed Q90 router map per frame from the completed
cross-fit replay. No map reselection, weight update or outcome-based exclusions.
"""
import datetime,hashlib,json,math,os,subprocess,sys
from pathlib import Path
os.environ['CUDA_VISIBLE_DEVICES']=''
os.environ['OMP_NUM_THREADS']='2'
os.environ['OPENBLAS_NUM_THREADS']='1'
import torch
import torch.nn.functional as F
REPO=Path(__file__).resolve().parents[1]
BASE=Path('/data10/shareddata/can_karsal/dcvcuf_depth_20260927/research')
ORIGINAL=BASE/'shared_crossfit_qp32'
OUT=BASE/'component_interventions_qp32'
SNAPSHOT=BASE/'shared_metric_audit/source'
UPSTREAM=Path('/home/can_karsal/DCVC')


def sha(p):
    with Path(p).open('rb') as f:return hashlib.file_digest(f,'sha256').hexdigest()
def utc():return datetime.datetime.now(datetime.timezone.utc).isoformat()
def atomic(path,value):
    temp=path.with_suffix(path.suffix+'.tmp');temp.write_text(json.dumps(value,indent=2,allow_nan=False)+'\n');temp.replace(path)


def main():
    torch.set_num_threads(2);torch.manual_seed(42)
    progress=json.loads((ORIGINAL/'progress.json').read_text())
    if progress['state']!='complete' or progress['completed_policy_cases']!=318:raise ValueError('Complete baseline replay required')
    original=json.loads((ORIGINAL/'manifest.json').read_text())
    for path,digest in original['source_sha256'].items():
        if sha(path)!=digest:raise ValueError('Baseline source changed')
    if subprocess.check_output(['git','-C',str(UPSTREAM),'rev-parse','HEAD'],text=True).strip()!='819c219b24db34310bbd15c51a720aaaf5eb2e7d':raise ValueError('Shared upstream changed')
    cases=[json.loads(p.read_text()) for p in sorted((ORIGINAL/'cases').glob('*.json'))]
    if len(cases)!=53 or {c['sequence'] for c in cases}!=set(original['names']):raise ValueError('Incomplete source cohort')
    fixed={c['sequence']:next(r for r in c['rows'] if (r['criterion'],r['policy'])==('q90','router')) for c in cases}
    cp=REPO/'runs/RECIPE512/ckpt_PIN_e15.pth.tar'
    sys.path.insert(0,str(SNAPSHOT));sys.path.insert(0,str(UPSTREAM))
    from flexuf.config import FlexUFConfig
    from flexuf.model import FlexUFIntra
    from src.utils.transforms import ycbcr2rgb
    import ctc_intra as C
    ck=torch.load(cp,map_location='cpu',weights_only=False);cfg=FlexUFConfig(**ck['config'])
    if not cfg.full_frame_head or cfg.seam_repair!='grid':raise ValueError('Unexpected reconstruction path')
    net=FlexUFIntra(cfg).eval();net.load_state_dict(ck.get('state_dict',ck.get('net',ck)),strict=True)
    adapters=net.dec.adapters;identity=torch.nn.ModuleList(torch.nn.Identity() for _ in adapters)
    captured=[]
    def capture(module,args):captured.append(args[0].detach().clone())
    handle=net.dec.seam_repair.register_forward_pre_hook(capture)
    seqs,_=C.discover([]);lookup={Path(s['path']).name:s for s in seqs}
    sources={str(p):sha(p) for p in [Path(__file__),ORIGINAL/'manifest.json',cp,*sorted((ORIGINAL/'cases').glob('*.json'))]}
    manifest={'scope':__doc__,'source_sha256':sources,'names':original['names'],'qp':32,'control':'fixed archived-map Q90 router, nominal0.1dB',
        'variants':['trained','repair_identity','adapters_identity','both_identity'],
        'boundary':'RGB pixels within4pixels of any interior256pixel tile boundary; outer image perimeter excluded as a boundary; cropped valid support only',
        'metrics':'Explicitly converted/clipped RGB MSE and PSNR plus unclipped444 MSE. Baseline anchors and exit maps come from the complete verified replay. Positive ablation PSNR loss means the intervention harmed reconstruction.',
        'reuse':'For each adapter state, run the decoder with trained repair once and retain the repair input. Reapply the same full-frame head to that input to obtain repair-identity output. No synthesis-trunk change between the two repair states.',
        'limitation':'Frozen-weight intervention, not separately optimized models or an estimate of retrained component value. CPU neural output, not bitstream or latency. This development corpus is not untouched external test.'}
    OUT.mkdir(parents=True,exist_ok=True);(OUT/'cases').mkdir(exist_ok=True)
    if (OUT/'manifest.json').exists() and json.loads((OUT/'manifest.json').read_text())!=manifest:raise ValueError('Resume provenance differs')
    atomic(OUT/'manifest.json',manifest)
    status={'state':'running','pid':os.getpid(),'started_utc':utc(),'completed':0,'total':53};atomic(OUT/'progress.json',status)
    try:
        with torch.inference_mode():
            for case in sorted(cases,key=lambda c:(c['height']*c['width'],c['sequence'])):
                name=case['sequence'];h,w=case['height'],case['width'];path=OUT/'cases'/f"{case['sequence_index']:02d}.json"
                with Path(lookup[name]['path']).open('rb') as f:frame_hash=hashlib.sha256(f.read(h*w*3//2)).hexdigest()
                if frame_hash!=case['first_frame_bytes_sha256']:raise ValueError('Baseline frame changed')
                if path.exists():
                    record=json.loads(path.read_text())
                    if record['frame_sha256']!=frame_hash or record['sequence']!=name:raise ValueError('Saved frame differs')
                else:
                    x,_=C.read_frames(lookup[name]['path'],w,h,1,1);xp=F.pad(x,(0,(-w)%cfg.rgb_patch,0,(-h)%cfg.rgb_patch),mode='replicate')
                    y,q,_=net._encode_to_latent(xp,torch.tensor([32],dtype=torch.int32))
                    em=torch.tensor(fixed[name]['map'],dtype=torch.long)
                    target=ycbcr2rgb(x+.5,clamp=True)
                    band=torch.zeros((h,w),dtype=torch.bool)
                    for boundary in range(cfg.rgb_patch,h,cfg.rgb_patch):band[max(0,boundary-4):min(h,boundary+4),:]=True
                    for boundary in range(cfg.rgb_patch,w,cfg.rgb_patch):band[:,max(0,boundary-4):min(w,boundary+4)]=True
                    if not band.any() or band.all():raise ValueError('Need both boundary and interior pixels')
                    rows=[];baseline_tensor=None;head_replay_exact=[]
                    for adapter_on in (True,False):
                        net.dec.adapters=adapters if adapter_on else identity;captured.clear()
                        repaired=net.dec(y,q,exit_map=em)
                        if len(captured)!=1:raise ValueError('Unexpected repair execution count')
                        feat=captured.pop()
                        # Exact path check validates cached-head reuse before measuring the intervention.
                        replay=net.dec._apply_head(net.dec.seam_repair(feat),q)
                        head_replay_exact.append(torch.equal(replay,repaired));captured.clear()
                        if not head_replay_exact[-1]:raise ValueError('Replayed repair/head differs')
                        del replay
                        if adapter_on:baseline_tensor=repaired.detach().clone()
                        elif bool((em==cfg.num_exits-1).all()) and not torch.equal(repaired,baseline_tensor):raise ValueError('Deepest-only adapter control changed')
                        for repair_on,out in ((True,repaired),(False,net.dec._apply_head(feat,q))):
                            crop=out[:,:,:h,:w];rgb=ycbcr2rgb(crop.clamp(-.5,.5)+.5,clamp=True)
                            error=(rgb-target).square();mse=float(error.mean());plane=error.mean(dim=(0,1));mse444=float((crop-x).square().mean())
                            variant=('trained' if repair_on else 'repair_identity') if adapter_on else ('adapters_identity' if repair_on else 'both_identity')
                            rows.append({'variant':variant,'adapters_on':adapter_on,'repair_on':repair_on,'mse_rgb':mse,'psnr_rgb':-10*math.log10(mse),
                                'mse_444':mse444,'boundary_mse_rgb':float(plane[band].mean()),'interior_mse_rgb':float(plane[~band].mean()),
                                'loss_vs_original_db':10*math.log10(mse/fixed[name]['cropped_rgb_mse']),
                                'loss_vs_dense_anchor_db':10*math.log10(mse/case['anchor_cropped_rgb_mse'])})
                        del feat,repaired
                    net.dec.adapters=adapters
                    baseline=rows[0]
                    if abs(baseline['mse_rgb']-fixed[name]['cropped_rgb_mse'])>1e-12 or abs(baseline['mse_444']-fixed[name]['cropped_ycbcr444_mse'])>1e-12:raise ValueError('Baseline does not reproduce prior replay')
                    record={'sequence':name,'sequence_index':case['sequence_index'],'frame_sha256':frame_hash,'height':h,'width':w,
                        'map':em.tolist(),'boundary_pixels':int(band.sum()),'interior_pixels':int((~band).sum()),'head_replay_exact':head_replay_exact,
                        'baseline_matches_complete_replay':True,'rows':rows,'finished_utc':utc()}
                    atomic(path,record)
                status.update(completed=status['completed']+1,sequence=name,updated_utc=utc());atomic(OUT/'progress.json',status);print(json.dumps(status),flush=True)
        for path,digest in sources.items():
            if sha(path)!=digest:raise ValueError('Run source changed')
        if torch.cuda.is_initialized():raise AssertionError('Unexpected CUDA context')
        status.update(state='complete',cuda_initialized=False,finished_utc=utc());atomic(OUT/'progress.json',status)
    except BaseException as error:
        status.update(state='failed',error=repr(error),updated_utc=utc());atomic(OUT/'progress.json',status);raise
    finally:
        net.dec.adapters=adapters;handle.remove()


if __name__=='__main__':main()
