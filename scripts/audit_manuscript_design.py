"""CPU provenance for the DCVC-UF-only manuscript and analytical scenarios.

No quality or runtime values are generated for uncompleted experiments.
The speedup curves are an explicit Amdahl sensitivity model, not predictions.
"""
from pathlib import Path
import hashlib,json
import torch
ROOT=Path(__file__).resolve().parents[1]
DEPTH_ROOT=Path('/data10/shareddata/can_karsal/dcvcuf_depth_20260927')
OUT=ROOT/'paper/data/refresh20260927'

def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()

def main():
    router_path=ROOT/'runs/RECIPE512/routers2/ablation_e15/ce_soft_6k.pth'
    head=torch.load(router_path,map_location='cpu',mmap=True,weights_only=False)
    router_record_path=ROOT/'results/router_e15/ce_soft_6k.json'
    router_record=json.loads(router_record_path.read_text())
    assert router_record['router_ckpt']==str(router_path.relative_to(ROOT))
    for key in ('inputs','lam','ckpt','ckpt_epoch','with_bits'):
        assert router_record[key]==head[key]
    assert router_record['eval']==head['eval']
    meta_path=ROOT/'runs/RECIPE512/meta.json'
    archive_meta=json.loads(meta_path.read_text())
    manifests={d:json.loads((DEPTH_ROOT/f'runs/d{d}/manifest.json').read_text()) for d in (2,4,6)}
    assert head['inputs']=='stem,qp' and head['target']=='ce_soft' and head['context']==0
    assert all(m['num_images']==384795 and m['all_parameters_trainable'] for m in manifests.values())
    step=(manifests[4]['decoder_parameters']-manifests[2]['decoder_parameters'])//2
    fixed=manifests[2]['decoder_parameters']-2*step
    nond=manifests[2]['parameters']-manifests[2]['decoder_parameters']
    assert step==1038720 and fixed==1767168 and nond==27947520
    for d,m in manifests.items():
        assert m['decoder_parameters']==fixed+d*step
        assert m['parameters']==fixed+d*step+nond
    depths=[2,4,6,8,10,12]
    params=[dict(depth=d,decoder_parameters=fixed+d*step,total_parameters=fixed+d*step+nond,
                 training_status='ongoing' if d in (2,4,6) else 'planned' if d in (8,10) else 'released reference; same-recipe control pending') for d in depths]
    scenarios=[]
    for f in (.5,.7,.9):
        for h in (0.,.03):
            for d in depths:
                relative=(1-f)+f*d/12+h
                scenarios.append(dict(depth=d,assumed_trunk_time_fraction=f,
                    assumed_extra_overhead_fraction=h,relative_time=relative,speedup=1/relative))
    sources=[router_path,router_record_path,meta_path,ROOT/'scripts/train_router2.py',ROOT/'flexuf/router/head2.py',
             ROOT/'scripts/launch_recipe512.sh',ROOT/'train_flexuf_image.py',Path(__file__)]
    sources += [DEPTH_ROOT/f'runs/d{d}/manifest.json' for d in manifests]
    result=dict(scope='DCVC-UF intra only; CPU checkpoint metadata inspection and analytic capacity/sensitivity calculations',
        router={k:v for k,v in head.items() if k not in ('router_head_v2','heldout_agree','eval')},
        router_stored_tensor_elements=sum(v.numel() for v in head['router_head_v2'].values()),
        router_training_record=router_record,
        shared_latent_training=archive_meta,
        caveat='Historical metadata and current code identify the inspected configuration; historical executable hashes were not captured at original run time.',
        parameters=dict(per_trunk_block=step,fixed_synthesis=fixed,non_synthesis=nond,rows=params,
            interpretation='Exact architecture counts derived from the repeated block structure and verified D2/D4/D6 manifests; not trained quality or runtime.'),
        projection=dict(label='ANALYTICAL SCENARIO — NOT MEASURED',
            equation='T(d)/T(12) = (1-f) + f*d/12 + h; speedup is its reciprocal',
            assumptions='The full-depth trunk consumes a hypothetical fraction f of baseline time; trunk work scales linearly with retained depth; h is extra overhead relative to full baseline time. No calibration to device timings.',
            excluded='No predicted PSNR, bitrate, learned routing benefit, confidence interval, or measured speedup.',rows=scenarios),
        hashes={str(p.relative_to(ROOT)) if p.is_relative_to(ROOT) else str(p):sha(p) for p in sources})
    (OUT/'design_audit.json').write_text(json.dumps(result,indent=2)+'\n')
    print(json.dumps(dict(router_inputs=head['inputs'],router_target=head['target'],depth_parameters=params),indent=2))
    assert not torch.cuda.is_initialized()

if __name__=='__main__':main()
