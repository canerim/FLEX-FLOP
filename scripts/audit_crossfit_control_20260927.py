"""Sequence-disjoint scalar calibration on archived DCVC-UF tile-error tables.

This is a cross-fit diagnostic of a fixed router, not a new decoder experiment
or an untouched external test. Every policy decision on a held-out sequence
uses only its stored logits, QP and a control fitted on other sequences.
"""
from __future__ import annotations
import hashlib
import json
import os
from pathlib import Path
import sys
import time
os.environ['CUDA_VISIBLE_DEVICES']=''
os.environ.setdefault('OMP_NUM_THREADS','1')
import numpy as np
import torch

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT/'flexplus'))
from null_models_uf import bayer
OUT=ROOT/'docs/research/2026-09-27-six-hour/crossfit_control'
BUDGETS=(.05,.1,.15,.2,.3,.5)


def sha(path):return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def choose(loss,saving,budget,criterion):
    risk=loss.mean(axis=1) if criterion=='mean' else np.quantile(loss,.9,axis=1,method='linear')
    feasible=np.flatnonzero(risk<=budget+1e-12)
    if len(feasible):
        best=min(feasible,key=lambda i:(-saving[i].mean(),risk[i],i));ok=True
    else:
        # No source-aware fallback at test time. The least training-risk
        # control is explicitly marked infeasible, and all test cases remain.
        best=min(range(len(risk)),key=lambda i:(risk[i],-saving[i].mean(),i));ok=False
    return int(best),bool(ok),float(risk[best])


def main():
    torch.set_num_threads(1);start=time.perf_counter();OUT.mkdir(parents=True,exist_ok=True)
    path=ROOT/'flexplus/results/router_dump_e15_ce_soft.pt'
    source_hash=sha(path);dump=torch.load(path,map_location='cpu',weights_only=False)
    names=dump['names'];assert len(names)==53 and len(set(names))==53
    # Split fixed by names only; no loss/rate/logit contributes to assignment.
    order=sorted(range(53),key=lambda i:hashlib.sha256(('FLEX-CAL-20260927:'+names[i]).encode()).hexdigest())
    folds=np.empty(53,dtype=np.int32)
    for rank,index in enumerate(order):folds[index]=rank%5
    assert sorted(np.bincount(folds).tolist())==[10,10,11,11,11]
    j=dump['j'];cost=dump['cost'].double().numpy()[j:]
    assert len(cost)==4 and np.all(np.diff(cost)>0)
    controls={'router':np.linspace(-256,256,4097),'dither':np.arange(385,dtype=float)/128,
              'uniform':np.arange(4,dtype=float)}
    qps=sorted(map(int,dump['frames']))
    tables={p:{'loss':np.empty((len(qps),len(c),53)),'saving':np.empty((len(qps),len(c),53))}
            for p,c in controls.items()}
    for qi,qp in enumerate(qps):
        frames=dump['frames'][qp]
        for fi,frame in enumerate(frames):
            lp=frame['lp'].double().numpy();M=frame['M'].double().numpy()[:,j:]
            assert lp.shape==M.shape and np.isfinite(lp).all() and np.isfinite(M).all() and (M>0).all()
            rank=bayer(*frame['grid']).double().numpy()
            for policy,probes in controls.items():
                if policy=='router':
                    maps=(lp[None]-probes[:,None,None]*cost[None,None]).argmax(2)
                    assert (maps[0]==3).all() and (maps[-1]==0).all(),'Grid does not reach both endpoints'
                elif policy=='dither':
                    rung=np.minimum(np.floor(probes).astype(int),3)
                    maps=np.where(rank[None]<probes[:,None]-rung[:,None],np.minimum(rung[:,None]+1,3),rung[:,None])
                else:maps=np.broadcast_to(probes.astype(int)[:,None],(4,len(M)))
                errors=M[np.arange(len(M))[None],maps].mean(axis=1)
                tables[policy]['loss'][qi,:,fi]=10*np.log10(errors/frame['R'])
                tables[policy]['saving'][qi,:,fi]=100*(1-cost[maps].mean(axis=1))
        print(json.dumps({'qp':qp,'tables_complete':True}),flush=True)
    selections=[];rows=[]
    for criterion in ('mean','q90'):
        for budget in BUDGETS:
            for policy in controls:
                for qi,qp in enumerate(qps):
                    loss=tables[policy]['loss'][qi];saving=tables[policy]['saving'][qi]
                    for fold in range(5):
                        train=np.flatnonzero(folds!=fold);test=np.flatnonzero(folds==fold)
                        assert not set(train)&set(test) and len(train)+len(test)==53
                        selected,feasible,trainrisk=choose(loss[:,train],saving[:,train],budget,criterion)
                        selections.append({'criterion':criterion,'budget':budget,'policy':policy,'qp':qp,'fold':fold,
                                           'control':float(controls[policy][selected]),'training_feasible':feasible,
                                           'train_risk_db':trainrisk,'train_indices':train.tolist(),'test_indices':test.tolist()})
                        for fi in test:
                            rows.append({'criterion':criterion,'budget':budget,'policy':policy,'qp':qp,'fold':fold,
                                         'sequence':names[fi],'sequence_index':int(fi),'control':float(controls[policy][selected]),
                                         'training_feasible':feasible,'loss_db':float(loss[selected,fi]),
                                         'saving_points':float(saving[selected,fi]),
                                         'exceeds_target':bool(loss[selected,fi]>budget+1e-12)})
    summaries=[]
    for criterion in ('mean','q90'):
        for budget in BUDGETS:
            for policy in controls:
                rr=[r for r in rows if (r['criterion'],r['budget'],r['policy'])==(criterion,budget,policy)]
                assert len(rr)==265 and len({(r['qp'],r['sequence']) for r in rr})==265
                losses=np.array([r['loss_db'] for r in rr]);savings=np.array([r['saving_points'] for r in rr])
                summaries.append({'criterion':criterion,'budget':budget,'policy':policy,'n':265,
                                  'mean_loss_db':float(losses.mean()),'q90_loss_db':float(np.quantile(losses,.9)),
                                  'max_loss_db':float(losses.max()),'mean_saving_points':float(savings.mean()),
                                  'violations':int((losses>budget+1e-12).sum()),
                                  'fraction_violating':float((losses>budget+1e-12).mean()),
                                  'training_infeasible_cases':sum(not r['training_feasible'] for r in rr)})
    # Deliberately perturb only held-out labels: fitted control must not change.
    test=np.flatnonzero(folds==0);train=np.flatnonzero(folds!=0)
    loss=tables['router']['loss'][0].copy();saving=tables['router']['saving'][0]
    original=choose(loss[:,train],saving[:,train],.1,'q90')
    loss[:,test]=1e9
    assert choose(loss[:,train],saving[:,train],.1,'q90')==original
    assert sha(path)==source_hash and not torch.cuda.is_initialized()
    result={'scope':'Fixed-router sequence-cross-fit calibration diagnostic on padded MSE tables; no mixed-image reconstruction, external test, or latency claim',
            'source_sha256':source_hash,'source':str(path),'script_sha256':sha(__file__),
            'bayer_source_sha256':sha(ROOT/'flexplus/null_models_uf.py'),
            'router_checkpoint_sha256':sha(ROOT/dump['router2']),
            'numpy':np.__version__,'torch':torch.__version__,
            'split_rule':'SHA256(FLEX-CAL-20260927:sequence_name), sorted rank modulo five; all QPs grouped',
            'names':names,'folds':folds.tolist(),'budgets':BUDGETS,'qps':qps,
            'control_grids':{p:c.tolist() for p,c in controls.items()},
            'calibration':'Per-QP training mean or linearly interpolated 90th percentile of table PSNR loss; maximise training mean MAC saving among feasible controls',
            'infeasible_handling':'Select least training-risk control; flag infeasible and retain all test outcomes; no source-informed fallback',
            'selection_checks':{'disjoint_sequences':True,'every_sequence_qp_once':True,'heldout_label_perturbation_invariant':True,
                                'full_depth_and_shallow_grid_endpoints':True},
            'interpretation':'Mean or q90 calibration is not a per-frame guarantee. Data have previously informed development. Cross-fit results are conditional on this fixed checkpoint and grid, not nested model selection.',
            'summaries':summaries,'selections':selections,'rows':rows,'wall_seconds':time.perf_counter()-start}
    (OUT/'analysis.json').write_text(json.dumps(result,indent=2,allow_nan=False)+'\n')
    np.savez_compressed(OUT/'control_tables.npz',**{p+'_'+k:v for p,t in tables.items() for k,v in t.items()})
    lines=['# Kaynak görüntüsüz kontrol: sequence-disjoint cross-fit teşhisi','',
           'Sabit e15 router logits ve padded tile-MSE tabloları kullanıldı. Yeni reconstruction veya hız deneyi değildir.',
           '53 sequence hash ile beş gruba bölündü; aynı sequence\'in beş QP\'si aynı grupta. Her QP için kontrol diğer dört grupta seçildi.',
           'Bu veri daha önce geliştirmede incelendiği için dış test başarısı iddia edilmiyor. Test kaynağı karar anında kullanılmıyor.',
           '', '| Kalibrasyon | Hedef dB | Policy | MAC tasarrufu % | Ortalama kayıp dB | Q90 kayıp dB | Hedef aşımı /265 | Kalibrasyon uygunsuz /265 |',
           '|---|---:|---|---:|---:|---:|---:|---:|']
    for r in summaries:
        lines.append(f"| {r['criterion']} | {r['budget']:.2f} | {r['policy']} | {r['mean_saving_points']:.3f} | {r['mean_loss_db']:.4f} | {r['q90_loss_db']:.4f} | {r['violations']} | {r['training_infeasible_cases']} |")
    lines += ['', 'Ortalama hedef kontrolü tek görüntü garantisi değildir. Q90 kontrolü de %100 güvence sağlamaz; başarısız kalibrasyonlar filtrelenmedi.',
              'Eş MAC veya eş gerçekleşen kalite karşılaştırması yapılmadan tasarruf farkı doğrudan router üstünlüğü diye okunmamalı.',
              'Sonraki aşama: kontrol değerlerini dondurup final mixed reconstruction üzerinde, eğitimde/kalibrasyonda görülmeyen ayrı veriyle ölçmek.']
    (OUT/'REPORT_TR.md').write_text('\n'.join(lines)+'\n')
    print(json.dumps({'done':True,'rows':len(rows),'out':str(OUT),'wall_seconds':result['wall_seconds']}),flush=True)


if __name__=='__main__':main()
