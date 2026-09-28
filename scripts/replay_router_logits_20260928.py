"""Fresh CPU router inference versus archived logits on all53 QP32 frames.

This is a numerical/provenance audit, not a latency or new quality experiment.
No source-error label enters either fixed-control map construction.
"""
import datetime,hashlib,json,os,subprocess,sys,time
from pathlib import Path
os.environ['CUDA_VISIBLE_DEVICES']=''
os.environ['OMP_NUM_THREADS']='2'
os.environ['OPENBLAS_NUM_THREADS']='1'
import numpy as np
import torch
import torch.nn.functional as F
REPO=Path(__file__).resolve().parents[1]
BASE=Path('/data10/shareddata/can_karsal/dcvcuf_depth_20260927/research')
OUT=BASE/'router_logit_replay_qp32'
SNAPSHOT=BASE/'shared_metric_audit/source'
UPSTREAM=Path('/home/can_karsal/DCVC')


def sha(p):
    with Path(p).open('rb') as f:return hashlib.file_digest(f,'sha256').hexdigest()
def utc():return datetime.datetime.now(datetime.timezone.utc).isoformat()
def atomic(path,obj):
    tmp=path.with_suffix(path.suffix+'.tmp');tmp.write_text(json.dumps(obj,indent=2,allow_nan=False)+'\n');tmp.replace(path)


def main():
    torch.set_num_threads(2);torch.manual_seed(42)
    if subprocess.check_output(['git','-C',str(UPSTREAM),'rev-parse','HEAD'],text=True).strip()!='819c219b24db34310bbd15c51a720aaaf5eb2e7d':raise ValueError('Shared upstream changed')
    proof=json.loads((REPO/'docs/research/2026-09-27-six-hour/shared_metric_audit/analysis.json').read_text())
    for p,h in proof['source_sha256'].items():
        if p.startswith(str(SNAPSHOT)) or p.startswith(str(UPSTREAM)):
            if sha(p)!=h:raise ValueError('Frozen source changed')
    cp=REPO/'runs/RECIPE512/ckpt_PIN_e15.pth.tar';dp=REPO/'flexplus/results/router_dump_e15_ce_soft.pt'
    calibration=REPO/'docs/research/2026-09-27-six-hour/crossfit_control/analysis.json'
    controls=json.loads(calibration.read_text());dump=torch.load(dp,map_location='cpu',weights_only=False)
    hp=REPO/dump['router2'];h=torch.load(hp,map_location='cpu',weights_only=False)
    if sha(dp)!=controls['source_sha256'] or sha(hp)!=controls['router_checkpoint_sha256']:raise ValueError('Archived router provenance changed')
    if sha(cp)!=proof['source_sha256'][str(cp)]:raise ValueError('Shared checkpoint changed')
    if h['inputs']!='stem,qp' or h['context']!=0:raise ValueError('Unexpected live head')
    sys.path.insert(0,str(SNAPSHOT));sys.path.insert(0,str(UPSTREAM))
    from flexuf.config import FlexUFConfig,LATENT_CH,TRUNK_CH
    from flexuf.model import FlexUFIntra
    from flexuf.router.head2 import StemRouterHeadV2
    import ctc_intra as C
    ck=torch.load(cp,map_location='cpu',weights_only=False);cfg=FlexUFConfig(**ck['config'])
    net=FlexUFIntra(cfg).eval();net.load_state_dict(ck.get('state_dict',ck.get('net',ck)),strict=True)
    head=StemRouterHeadV2(TRUNK_CH,LATENT_CH,cfg.num_exits,min_exit=cfg.split_depth,inputs=h['inputs'],with_bits=h.get('with_bits'),r_stem=h.get('r_stem',48),r_lat=h.get('r_lat',32),hidden=h.get('hidden',256)).eval()
    head.load_state_dict(h['router_head_v2'],strict=True)
    layers=[]
    for name,module in head.named_modules():
        if isinstance(module,(torch.nn.Conv2d,torch.nn.Linear)):
            def count(m,inputs,out,name=name):
                macs=out.numel()*m.in_features if isinstance(m,torch.nn.Linear) else out.numel()*(m.in_channels//m.groups)*m.kernel_size[0]*m.kernel_size[1]
                layers.append({'name':name,'input':list(inputs[0].shape),'output':list(out.shape),'macs':macs})
            module.register_forward_hook(count)
    seqs,_=C.discover([]);lookup={Path(s['path']).name:s for s in seqs};names=dump['names'];j=dump['j'];cost=dump['cost'].double().numpy()
    if len(names)!=53 or names!=controls['names'] or any(n not in lookup for n in names):raise ValueError('Sequence grid differs')
    selected=[r for r in controls['rows'] if r['qp']==32 and r['budget']==.1 and r['policy']=='router']
    if len(selected)!=106:raise ValueError('Expected two fixed controls per sequence')
    sources={str(p):sha(p) for p in (Path(__file__),cp,hp,dp,calibration,SNAPSHOT/'flexuf/router/head2.py',SNAPSHOT/'flexuf/model.py',SNAPSHOT/'flexuf/backbone/decoder.py',SNAPSHOT/'ctc_intra.py')}
    manifest={'scope':__doc__,'source_sha256':sources,'names':names,'qp':32,'budget':.1,'head_inputs':h['inputs'],'head_parameters':sum(p.numel() for p in head.parameters()),
        'comparison':'Record every logit/map difference; do not loosen a tolerance or exclude discrepant frames. Archived logits were generated on a historical GPU path; current inference is frozen-source CPU FP32.',
        'cost_scope':'Executed Conv2d and Linear MACs only. Pooling, normalization, activation, decision, transfer and dispatch are excluded; no wall-time conversion.'}
    OUT.mkdir(parents=True,exist_ok=True);(OUT/'cases').mkdir(exist_ok=True)
    if (OUT/'manifest.json').exists() and json.loads((OUT/'manifest.json').read_text())!=manifest:raise ValueError('Resume provenance differs')
    atomic(OUT/'manifest.json',manifest);progress={'state':'running','pid':os.getpid(),'started_utc':utc(),'completed':0,'total':53};atomic(OUT/'progress.json',progress)
    order=sorted(range(53),key=lambda i:(lookup[names[i]]['h']*lookup[names[i]]['w'],names[i]))
    try:
        with torch.inference_mode():
            for index in order:
                name=names[index];case=lookup[name];path=OUT/'cases'/f'{index:02d}.json';height,width=case['h'],case['w']
                with Path(case['path']).open('rb') as f:source_hash=hashlib.sha256(f.read(height*width*3//2)).hexdigest()
                if path.exists():
                    record=json.loads(path.read_text())
                    if record['sequence']!=name or record['frame_sha256']!=source_hash:raise ValueError('Saved case source differs')
                else:
                    x,_=C.read_frames(case['path'],width,height,1,1);xp=F.pad(x,(0,(-width)%cfg.rgb_patch,0,(-height)%cfg.rgb_patch),mode='replicate')
                    qp=torch.tensor([32],dtype=torch.int32);y,q,aux=net._encode_to_latent(xp,qp)
                    stem=net.dec.upsample(y)
                    for group in range(j):stem=net.dec.groups[group](stem)
                    layers.clear();logits=head(stem,y,aux['scales_hat'],qp,cfg.feature_patch,cfg.latent_patch)
                    current=F.log_softmax(logits[:,j:],1).double().numpy();old=dump['frames'][32][index]['lp'].double().numpy()
                    if current.shape!=old.shape or not np.isfinite(current).all():raise ValueError('Logit shape/numerics invalid')
                    comparisons=[]
                    for control in [r for r in selected if r['sequence_index']==index]:
                        score=current-control['control']*cost[None,j:];old_score=old-control['control']*cost[None,j:]
                        em=score.argmax(1)+j;old_em=old_score.argmax(1)+j;sorted_scores=np.sort(score,axis=1)
                        comparisons.append({'criterion':control['criterion'],'control':control['control'],'tile_count':len(em),'changed_tiles':int(np.sum(em!=old_em)),
                            'fresh_map':em.tolist(),'archived_map':old_em.tolist(),'fresh_saving_points':float(100*(1-cost[em].mean())),
                            'archived_saving_points':float(100*(1-cost[old_em].mean())),'minimum_fresh_top2_margin':float(np.min(sorted_scores[:,-1]-sorted_scores[:,-2]))})
                    delta=current-old
                    record={'sequence':name,'sequence_index':index,'frame_sha256':source_hash,'height':height,'width':width,'padded_shape':list(xp.shape),
                        'max_absolute_log_probability_difference':float(np.abs(delta).max()),'mean_absolute_log_probability_difference':float(np.abs(delta).mean()),
                        'exact_log_probabilities':bool(np.array_equal(current,old)),'comparisons':comparisons,'router_conv_linear_macs':sum(r['macs'] for r in layers),
                        'router_layers':list(layers),'finished_utc':utc()}
                    atomic(path,record)
                progress.update(completed=progress['completed']+1,sequence=name,updated_utc=utc());atomic(OUT/'progress.json',progress);print(json.dumps(progress),flush=True)
        if torch.cuda.is_initialized():raise AssertionError('Unexpected CUDA context')
        progress.update(state='complete',finished_utc=utc(),cuda_initialized=False);atomic(OUT/'progress.json',progress)
    except BaseException as e:
        progress.update(state='failed',error=repr(e),updated_utc=utc());atomic(OUT/'progress.json',progress);raise


if __name__=='__main__':main()
