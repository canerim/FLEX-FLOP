"""Profile the actual DCVC-UF uniform-exit reconstructions in frozen M tables.

The columns were obtained with one uniform depth over the padded frame.
Their mean is exactly that uniform output's padded RGB MSE. Mixed-map quality
is not inferred by recombining columns; independent codec depths are unrelated.
"""
from pathlib import Path
import hashlib,json
import numpy as np
import torch
ROOT=Path(__file__).resolve().parents[1];OUT=ROOT/'paper/data/refresh20260927'

def ci(values,names):
    _,idx=np.unique(names,return_inverse=True);sums=np.bincount(idx,weights=values);n=np.bincount(idx)
    ix=np.random.default_rng(20260927).integers(0,len(n),(5000,len(n)))
    boot=sums[ix].sum(1)/n[ix].sum(1)
    return dict(mean=float(np.mean(values)),lo=float(np.quantile(boot,.025)),hi=float(np.quantile(boot,.975)))

def main():
    torch.set_num_threads(2)
    p=ROOT/'flexplus/results/router_dump_e15_ce_soft.pt'
    d=torch.load(p,map_location='cpu',weights_only=False)
    uniform=[];incremental=[]
    for qp,frames in d['frames'].items():
        for i,e in enumerate(frames):
            M=e['M'].double().numpy();ref=e['R'];name=d['names'][i]
            assert np.all(M>0) and ref>0
            assert np.array_equal(M[:,0],M[:,2]) and np.array_equal(M[:,1],M[:,2])
            for k in range(2,6):
                loss=10*np.log10(M[:,k].mean()/ref)
                uniform.append(dict(sequence=name,qp=int(qp),depth=(k+1)*2,padded_rgb_loss_db=float(loss),
                    mac_saving=100*(1-float(d['cost'][k])),tiles=len(M)))
            for k in range(2,5):
                gain=10*np.log10(M[:,k]/M[:,k+1])
                incremental.append(dict(sequence=name,qp=int(qp),from_depth=(k+1)*2,to_depth=(k+2)*2,
                    tile_gain_db=gain.tolist(),negative_fraction=float(np.mean(gain<0)),
                    median=float(np.median(gain)),iqr=float(np.quantile(gain,.75)-np.quantile(gain,.25))))
    assert len(uniform)==265*4
    summary=[]
    for qp in [0,16,32,48,63,None]:
        for depth in (6,8,10,12):
            rr=[r for r in uniform if r['depth']==depth and (qp is None or r['qp']==qp)]
            summary.append(dict(qp=qp,depth=depth,n=len(rr),mac_saving=rr[0]['mac_saving'],
                **ci([r['padded_rgb_loss_db'] for r in rr],[r['sequence'] for r in rr])))
    gains=[]
    for depth in (6,8,10):
        rr=[r for r in incremental if r['from_depth']==depth]
        # Every frame–QP gets equal weight; tile populations differ by size.
        gains.append(dict(from_depth=depth,to_depth=depth+2,frames=len(rr),
            mean_negative_tile_fraction=ci([r['negative_fraction'] for r in rr],[r['sequence'] for r in rr]),
            mean_within_frame_iqr=ci([r['iqr'] for r in rr],[r['sequence'] for r in rr]),
            mean_within_frame_median=ci([r['median'] for r in rr],[r['sequence'] for r in rr])))
    sources=[p,ROOT/'flexuf/eval.py',ROOT/'flexplus/dump_router_lp.py',Path(__file__)]
    result=dict(scope='Archived uniform-map reconstructions, padded RGB support, released-weight full-frame reference; not cropped mixed-output or independent D2/D4/D6 quality.',
        interpretation='Mean of equal-area tile MSEs equals the padded full-frame MSE of that uniform-depth reconstruction. Adjacent tile gains compare two different uniform-depth contexts, not a causal one-tile intervention in a mixed map.',
        bootstrap='5000 sequence-cluster draws; QPs grouped by sequence. Frame–QP weighting, not pooled tile weighting.',
        uniform_rows=uniform,incremental_rows=incremental,uniform_summary=summary,incremental_summary=gains,
        hashes={str(x.relative_to(ROOT)):hashlib.sha256(x.read_bytes()).hexdigest() for x in sources})
    (OUT/'exit_depth_profile.json').write_text(json.dumps(result,indent=2)+'\n')
    print(json.dumps(dict(uniform=[r for r in summary if r['qp'] is None],incremental=gains),indent=2))
    assert not torch.cuda.is_initialized()

if __name__=='__main__':main()
