"""Actual QP32 placement intervention at fixed depth histograms.

Predeclared before execution: all53 completed cross-fit QP32 frames, Q90
router and Bayer maps, three deterministic spatial permutations per map, stratified by valid tile extent.
Same permutation indices for both policies on each frame. No outcome-based
map choice or exclusions. Frozen e15 CPU neural replay, no timing/stream claim.
"""
import datetime, hashlib, json, math, os, random, subprocess, sys
from pathlib import Path
os.environ['CUDA_VISIBLE_DEVICES']=''
os.environ['OMP_NUM_THREADS']='2'
os.environ['OPENBLAS_NUM_THREADS']='1'
import torch
import torch.nn.functional as F
REPO=Path(__file__).resolve().parents[1]
BASE=Path('/data10/shareddata/can_karsal/dcvcuf_depth_20260927/research')
ORIGINAL=BASE/'shared_crossfit_qp32'
OUT=BASE/'placement_replay_qp32'
SNAPSHOT=BASE/'shared_metric_audit/source'
UPSTREAM=Path('/home/can_karsal/DCVC')
SEEDS=(202609280,202609281,202609282)


def sha(path):
    with Path(path).open('rb') as f:
        return hashlib.file_digest(f,'sha256').hexdigest()

def utc():return datetime.datetime.now(datetime.timezone.utc).isoformat()
def atomic(path,data):
    temp=path.with_suffix(path.suffix+'.tmp')
    temp.write_text(json.dumps(data,indent=2,allow_nan=False)+'\n');temp.replace(path)


def main():
    torch.set_num_threads(2);torch.manual_seed(42)
    progress=json.loads((ORIGINAL/'progress.json').read_text())
    if progress['state']!='complete' or progress['completed_policy_cases']!=318:
        raise ValueError('Complete original replay required')
    original=json.loads((ORIGINAL/'manifest.json').read_text())
    for path,digest in original['source_sha256'].items():
        if sha(path)!=digest:raise ValueError('Baseline source changed: '+path)
    upstream=subprocess.check_output(['git','-C',str(UPSTREAM),'rev-parse','HEAD'],text=True).strip()
    if upstream!='819c219b24db34310bbd15c51a720aaaf5eb2e7d':raise ValueError('Upstream changed')
    cases=[json.loads(p.read_text()) for p in sorted((ORIGINAL/'cases').glob('*.json'))]
    if len(cases)!=53 or {c['sequence'] for c in cases}!=set(original['names']):raise ValueError('Wrong cohort')
    cp=REPO/'runs/RECIPE512/ckpt_PIN_e15.pth.tar'
    sources={str(p):sha(p) for p in [Path(__file__),ORIGINAL/'manifest.json',cp,*sorted((ORIGINAL/'cases').glob('*.json'))]}
    manifest={'scope':__doc__,'source_sha256':sources,'upstream_commit':upstream,'names':original['names'],
        'qp':32,'criterion':'q90','nominal_budget_db':.1,'policies':['router','dither'],'permutation_seeds':list(SEEDS),
        'permutation':'Permute within strata of identical valid tile height and width. Python Random seed is int(SHA256(sequence+colon+seed),16). Same indices for both policies.',
        'primary':'For each sequence/policy, 10log10(mean of three shuffled RGB MSEs / original RGB MSE). Then equal-sequence average. Positive means original placement is better. Sequence bootstrap conditional on the three sampled permutations.',
        'secondary':'Mean per-permutation PSNR loss; unclipped444 counterparts; paired router-minus-dither placement gain.',
        'metric':'Same conversion as original replay: source ycbcr2rgb(x+0.5,clamp=True); reconstruction ycbcr2rgb(crop.clamp(-0.5,0.5)+0.5,clamp=True). Also raw equal-channel444 MSE.',
        'geometry':'Preserve depth histograms separately within each valid-height/valid-width stratum of the padded256 grid. Thus depth counts and depth-weighted valid area both stay fixed. Reconstruction interactions and boundary locations can still change.',
        'limitation':'Development-used cohort, fixed Q90 controls and checkpoint, onlyQP32, three Monte Carlo permutations. No retraining, source-informed reselection, actual bitstream or latency. Conditional uncertainty is not external generalisation or exhaustive permutation expectation.',
        'python':sys.version,'torch':torch.__version__}
    OUT.mkdir(parents=True,exist_ok=True);(OUT/'cases').mkdir(exist_ok=True)
    if (OUT/'manifest.json').exists() and json.loads((OUT/'manifest.json').read_text())!=manifest:raise ValueError('Resume manifest differs')
    atomic(OUT/'manifest.json',manifest)
    sys.path.insert(0,str(SNAPSHOT));sys.path.insert(0,str(UPSTREAM))
    from flexuf.config import FlexUFConfig
    from flexuf.model import FlexUFIntra
    from src.utils.transforms import ycbcr2rgb
    import ctc_intra as C
    ck=torch.load(cp,map_location='cpu',weights_only=False);cfg=FlexUFConfig(**ck['config'])
    net=FlexUFIntra(cfg).eval();net.load_state_dict(ck.get('state_dict',ck.get('net',ck)),strict=True)
    seqs,_=C.discover([]);lookup={Path(s['path']).name:s for s in seqs}
    status={'state':'running','pid':os.getpid(),'started_utc':utc(),'completed_sequences':0,'total_sequences':53,'scheduled_map_cases':424}
    atomic(OUT/'progress.json',status)
    try:
        with torch.inference_mode():
            for case in sorted(cases,key=lambda c:(c['height']*c['width'],c['sequence'])):
                name=case['sequence'];h,w=case['height'],case['width'];path=OUT/'cases'/f"{case['sequence_index']:02d}.json"
                with Path(lookup[name]['path']).open('rb') as f:frame_sha=hashlib.sha256(f.read(h*w*3//2)).hexdigest()
                if frame_sha!=case['first_frame_bytes_sha256']:raise ValueError('Frame changed')
                if path.exists():
                    record=json.loads(path.read_text())
                    if record['sequence']!=name or record['frame_sha256']!=frame_sha:raise ValueError('Saved identity differs')
                else:
                    x,_=C.read_frames(lookup[name]['path'],w,h,1,1)
                    xp=F.pad(x,(0,(-w)%cfg.rgb_patch,0,(-h)%cfg.rgb_patch),mode='replicate')
                    y,q,_=net._encode_to_latent(xp,torch.tensor([32],dtype=torch.int32))
                    target=ycbcr2rgb(x+.5,clamp=True)
                    rows=[];cache={};permutations=None
                    for policy in ('router','dither'):
                        base=next(r for r in case['rows'] if (r['criterion'],r['policy'])==('q90',policy))
                        em=torch.tensor(base['map'],dtype=torch.long);flat=em.reshape(-1);n=flat.numel()
                        if permutations is None:
                            permutations=[]
                            for seed in SEEDS:
                                order=list(range(n));rng=random.Random(int(hashlib.sha256(f'{name}:{seed}'.encode()).hexdigest(),16))
                                nx=(w+cfg.rgb_patch-1)//cfg.rgb_patch;ny=(h+cfg.rgb_patch-1)//cfg.rgb_patch
                                if n!=nx*ny:raise ValueError('Tile geometry disagrees with map')
                                strata={}
                                for i in range(n):
                                    r,c=divmod(i,nx);extent=(min(cfg.rgb_patch,h-r*cfg.rgb_patch),min(cfg.rgb_patch,w-c*cfg.rgb_patch))
                                    strata.setdefault(extent,[]).append(i)
                                for group in strata.values():
                                    shuffled=group.copy();rng.shuffle(shuffled)
                                    for destination,source in zip(group,shuffled):order[destination]=source
                                permutations.append(order)
                        hist=torch.bincount(flat,minlength=cfg.num_exits)
                        maps=[('original',None,em)]+[('permuted',seed,flat[order].reshape_as(em)) for seed,order in zip(SEEDS,permutations)]
                        for variant,seed,m in maps:
                            if not torch.equal(torch.bincount(m.reshape(-1),minlength=cfg.num_exits),hist):raise ValueError('Histogram changed')
                            key=tuple(m.reshape(-1).tolist())
                            if key not in cache:
                                out=net.dec(y,q,exit_map=m);crop=out[:,:,:h,:w]
                                rgb=ycbcr2rgb(crop.clamp(-.5,.5)+.5,clamp=True)
                                cache[key]={'mse_rgb':float((rgb-target).square().mean()),'mse_444':float((crop-x).square().mean())}
                                del out,crop,rgb
                            metrics=cache[key]
                            if variant=='original' and (abs(metrics['mse_rgb']-base['cropped_rgb_mse'])>1e-12 or abs(metrics['mse_444']-base['cropped_ycbcr444_mse'])>1e-12):raise ValueError('Original map does not reproduce baseline')
                            rows.append({'policy':policy,'variant':variant,'seed':seed,'map':m.tolist(),'tile_histogram':hist.tolist(),
                                'saving_points':base['saving_points'],**metrics,
                                'rgb_loss_vs_original_db':10*math.log10(metrics['mse_rgb']/base['cropped_rgb_mse']),
                                '444_loss_vs_original_db':10*math.log10(metrics['mse_444']/base['cropped_ycbcr444_mse'])})
                    record={'sequence':name,'sequence_index':case['sequence_index'],'height':h,'width':w,'frame_sha256':frame_sha,
                        'permutation_indices':permutations,'distinct_maps_executed':len(cache),'baseline_reproduced':True,
                        'rows':rows,'finished_utc':utc()}
                    atomic(path,record)
                status.update(completed_sequences=status['completed_sequences']+1,sequence=name,updated_utc=utc());atomic(OUT/'progress.json',status);print(json.dumps(status),flush=True)
        for path,digest in sources.items():
            if sha(path)!=digest:raise ValueError('Run source changed')
        if torch.cuda.is_initialized():raise AssertionError('CUDA context created')
        status.update(state='complete',finished_utc=utc(),cuda_initialized=False);atomic(OUT/'progress.json',status)
    except BaseException as error:
        status.update(state='failed',error=repr(error),updated_utc=utc());atomic(OUT/'progress.json',status);raise


if __name__=='__main__':main()
