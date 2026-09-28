"""Released-D12 anchor on the same53 QP32 frames and valid support as e15 replay.

CPU FP32 neural reconstruction, not native CUDA or bitstream evaluation.
All source identities and unchanged analysis/entropy weights are verified.
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
PRIOR=BASE/'shared_crossfit_qp32';OUT=BASE/'released_anchor_qp32'
SNAPSHOT=BASE/'shared_metric_audit/source';UPSTREAM=Path('/home/can_karsal/DCVC')


def sha(path):
    with Path(path).open('rb') as f:return hashlib.file_digest(f,'sha256').hexdigest()
def utc():return datetime.datetime.now(datetime.timezone.utc).isoformat()
def atomic(path,data):
    tmp=path.with_suffix(path.suffix+'.tmp');tmp.write_text(json.dumps(data,indent=2,allow_nan=False)+'\n');tmp.replace(path)


def main():
    torch.set_num_threads(2);torch.manual_seed(42)
    progress=json.loads((PRIOR/'progress.json').read_text())
    if (progress['state'],progress['completed_policy_cases'])!=('complete',318):raise ValueError('Complete baseline cohort required')
    prior=json.loads((PRIOR/'manifest.json').read_text())
    for path,digest in prior['source_sha256'].items():
        if sha(path)!=digest:raise ValueError('Frozen baseline changed')
    if subprocess.check_output(['git','-C',str(UPSTREAM),'rev-parse','HEAD'],text=True).strip()!='819c219b24db34310bbd15c51a720aaaf5eb2e7d':raise ValueError('Upstream changed')
    if subprocess.check_output(['git','-C',str(UPSTREAM),'diff','HEAD','--','src']):raise ValueError('Upstream source modified')
    audit=json.loads((REPO/'cvpr2027/data/refresh20260927/reference_audit.json').read_text())
    release_path=BASE.parent/'reference_d12/released_cvpr2026_image.pth.tar'
    cp=REPO/'runs/RECIPE512/ckpt_PIN_e15.pth.tar';warm_path=REPO/'runs/warmstart/ckpt_warmstart.pth.tar'
    for path in (release_path,cp,warm_path):
        key=str(path.relative_to(REPO)) if path.is_relative_to(REPO) else str(path)
        if sha(path)!=audit['hashes'][key]:raise ValueError('Anchor checkpoint identity differs')
    sys.path.insert(0,str(SNAPSHOT));sys.path.insert(0,str(UPSTREAM))
    from flexuf.config import FlexUFConfig
    from flexuf.model import FlexUFIntra
    from flexuf.backbone.warmstart import remap_decoder_state
    from src.models.image_model import DMCI
    from src.utils.transforms import ycbcr2rgb
    import ctc_intra as C
    ck=torch.load(cp,map_location='cpu',weights_only=False);cfg=FlexUFConfig(**ck['config'])
    net=FlexUFIntra(cfg).eval();net.load_state_dict(ck['state_dict'],strict=True)
    release=torch.load(release_path,map_location='cpu',weights_only=False);release=release.get('state_dict',release)
    stock=DMCI().eval();stock.load_state_dict(release,strict=True)
    shared={k:v for k,v in release.items() if not k.startswith('dec.')}
    if len(shared)!=255 or any(not torch.equal(v,net.state_dict()[k]) for k,v in shared.items()):raise ValueError('Analysis/entropy sharing not exact')
    mapped,unknown=remap_decoder_state({k[4:]:v for k,v in release.items() if k.startswith('dec.')},cfg.blocks_per_exit)
    warm=torch.load(warm_path,map_location='cpu',weights_only=False);warm=warm.get('state_dict',warm.get('net',warm))
    if unknown or any(not torch.equal(v,warm['dec.'+k]) for k,v in mapped.items()):raise ValueError('Released decoder remapping differs')
    # This model is only a canary for the exact stock-versus-remapped forward path.
    warm_model=FlexUFIntra(cfg).eval()
    missing,unexpected=warm_model.dec.load_state_dict(mapped,strict=False)
    if unexpected or any(not k.startswith(('adapters.','seam_repair.','pad_coef')) for k in missing):raise ValueError('Unexpected warm model mismatch')
    dump_path=REPO/'flexplus/results/router_dump_e15_ce_soft.pt';dump=torch.load(dump_path,map_location='cpu',weights_only=False)
    cases=[json.loads(p.read_text()) for p in sorted((PRIOR/'cases').glob('*.json'))]
    if len(cases)!=53 or {c['sequence'] for c in cases}!=set(prior['names']):raise ValueError('Incomplete baseline cases')
    seqs,_=C.discover([]);lookup={Path(s['path']).name:s for s in seqs}
    files=[Path(__file__),cp,warm_path,release_path,PRIOR/'manifest.json',dump_path,*sorted((PRIOR/'cases').glob('*.json')),SNAPSHOT/'flexuf/backbone/warmstart.py',UPSTREAM/'src/models/image_model.py',UPSTREAM/'src/layers/layers.py']
    manifest={'scope':__doc__,'source_sha256':{str(p):sha(p) for p in files},'names':prior['names'],'qp':32,
        'shared_frontend_tensors_exact':len(shared),'released_decoder_tensors_exact_to_warmstart':len(mapped),
        'protocol':'Reuse the latent computed by the frozen e15 front end only after all255 analysis/entropy tensors including quality scales match the official release. Decode with strictly loaded stock DMCI synthesis. First source case additionally requires exact stock-versus-remapped warm-start output.',
        'support':'Same replicate-to256 padded source as archived shared replay; crop to valid source pixels for explicit clipped RGB and unclipped444. This is not a native image-pad16 evaluation.',
        'report':'Compare released and e15 full-frame anchors, then re-express all318 previously verified fixed-policy outputs against the released anchor using their unchanged MSEs. No new maps, exclusions, quality filtering or updates.'}
    OUT.mkdir(parents=True,exist_ok=True);(OUT/'cases').mkdir(exist_ok=True)
    if (OUT/'manifest.json').exists() and json.loads((OUT/'manifest.json').read_text())!=manifest:raise ValueError('Resume provenance differs')
    atomic(OUT/'manifest.json',manifest);status={'state':'running','pid':os.getpid(),'started_utc':utc(),'completed':0,'total':53};atomic(OUT/'progress.json',status)
    try:
        with torch.inference_mode():
            for case in sorted(cases,key=lambda c:(c['height']*c['width'],c['sequence'])):
                name=case['sequence'];h,w=case['height'],case['width'];idx=case['sequence_index'];path=OUT/'cases'/f'{idx:02d}.json'
                with Path(lookup[name]['path']).open('rb') as f:frame_hash=hashlib.sha256(f.read(h*w*3//2)).hexdigest()
                if frame_hash!=case['first_frame_bytes_sha256']:raise ValueError('Source frame changed')
                if path.exists():
                    record=json.loads(path.read_text())
                    if record['frame_sha256']!=frame_hash or record['sequence']!=name:raise ValueError('Existing source differs')
                else:
                    x,planes=C.read_frames(lookup[name]['path'],w,h,1,1);xp=F.pad(x,(0,(-w)%cfg.rgb_patch,0,(-h)%cfg.rgb_patch),mode='replicate')
                    y,q,_=net._encode_to_latent(xp,torch.tensor([32],dtype=torch.int32));released=stock.dec(y,q)
                    canary=None
                    if status['completed']==0:
                        mapped_out=warm_model.dec.forward_full(y,q);canary=torch.equal(mapped_out,released)
                        if not canary:raise ValueError('Stock/remapped released output differs')
                        del mapped_out
                    crop=released[:,:,:h,:w];rgb=ycbcr2rgb(crop.clamp(-.5,.5)+.5,clamp=True);target=ycbcr2rgb(x+.5,clamp=True)
                    mse444=float((crop-x).square().mean());msergb=float((rgb-target).square().mean());padded=float((released-xp).square().mean())
                    released611=C.psnr_611_420(crop,planes[0]);archived_R=float(dump['frames'][32][idx]['R'])
                    shifted=[]
                    for row in case['rows']:
                        shifted.append({'criterion':row['criterion'],'policy':row['policy'],'saving_points':row['saving_points'],
                            'cropped_rgb_loss_vs_released_db':10*math.log10(row['cropped_rgb_mse']/msergb),
                            'cropped_444_loss_vs_released_db':10*math.log10(row['cropped_ycbcr444_mse']/mse444),
                            'cropped_611_loss_vs_released_db':row['cropped_yuv611_loss_db']+released611-case['anchor_cropped_yuv611_psnr']})
                    record={'sequence':name,'sequence_index':idx,'frame_sha256':frame_hash,'height':h,'width':w,'stock_warmstart_canary_exact':canary,
                        'released_cropped_rgb_mse':msergb,'released_cropped_444_mse':mse444,'released_cropped_rgb_psnr':-10*math.log10(msergb),
                        'released_cropped_611_psnr':released611,'released_padded_444_mse':padded,'archived_released_R':archived_R,
                        'released_padded_vs_archived_R_db':10*math.log10(padded/archived_R),
                        'e15_cropped_rgb_psnr':-10*math.log10(case['anchor_cropped_rgb_mse']),
                        'e15_full_rgb_loss_vs_released_db':10*math.log10(case['anchor_cropped_rgb_mse']/msergb),
                        'e15_full_444_loss_vs_released_db':10*math.log10(case['anchor_cropped_ycbcr444_mse']/mse444),
                        'e15_full_611_loss_vs_released_db':released611-case['anchor_cropped_yuv611_psnr'],
                        'policy_rows':shifted,'finished_utc':utc()}
                    atomic(path,record)
                status.update(completed=status['completed']+1,sequence=name,updated_utc=utc());atomic(OUT/'progress.json',status);print(json.dumps(status),flush=True)
        if torch.cuda.is_initialized():raise AssertionError('Unexpected CUDA context')
        for path,digest in manifest['source_sha256'].items():
            if sha(path)!=digest:raise ValueError('Input changed during run')
        status.update(state='complete',finished_utc=utc(),cuda_initialized=False);atomic(OUT/'progress.json',status)
    except BaseException as error:
        status.update(state='failed',error=repr(error),updated_utc=utc());atomic(OUT/'progress.json',status);raise


if __name__=='__main__':main()
