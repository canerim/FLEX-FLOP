"""CPU-only exploratory risk score from frozen exit maps and QP.

Fit on 24 DIV2K calibration images and evaluate once on 24 disjoint validation
images. Labels use source reconstruction quality during analysis, but inference
features use only the decoder-known QP and tile exit map. This does not change
the router or guarantee a quality-constrained codec.
"""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import numpy as np
from scipy.optimize import minimize
from scipy.special import expit
from scipy.stats import rankdata


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def extract(directory, items, qps, beta, manifest_sha):
    rows = []
    for item in items:
        for qp in qps:
            path = directory/f'{Path(item["image"]).stem}_qp{qp}.json'
            record = json.loads(path.read_text())
            if (record['manifest_sha256'] != manifest_sha or
                record['image'] != item['image'] or record['qp'] != qp or
                record['source_sha256'] != item['source_sha256']):
                raise RuntimeError(f'Case provenance mismatch: {path}')
            candidates = [x for x in record['candidates']
                          if x['beta'] == beta[str(qp)]]
            if len(candidates) != 1:
                raise RuntimeError(f'Frozen policy candidate missing: {path}')
            chosen = candidates[0]
            modes = np.asarray(chosen['exit_map'],dtype=float)
            if modes.size != 6 or not np.all((modes>=2)&(modes<=5)):
                raise RuntimeError(f'Unexpected six-tile geometry: {path}')
            grid = modes.reshape(2,3)
            edges = np.concatenate((np.abs(np.diff(grid,axis=0)).ravel(),
                                    np.abs(np.diff(grid,axis=1)).ravel()))
            # Fixed, interpretable decoder-side features. No image pixels,
            # reconstruction quality, latent activations, or fitted router logits.
            feat = [float(qp/63), float(modes.mean()-2),
                    float(modes.min()-2), float((modes==5).mean()),
                    float((modes<=3).mean()),float(edges.mean()),
                    float((edges>0).mean())]
            rows.append({'image':item['image'],'qp':qp,
                         'case_sha256':sha(path),'exit_map':modes.astype(int).tolist(),
                         'features':feat,'loss_db':chosen['delta444_db'],
                         'over_budget':bool(chosen['delta444_db']>.1),
                         'mac_saved_pct':chosen['mac_saved_pct']})
    return rows


def auc(y,score):
    y=np.asarray(y,bool)
    npos=int(y.sum()); nneg=len(y)-npos
    return float((rankdata(score)[y].sum()-npos*(npos+1)/2)/(npos*nneg))


def average_precision(y,score):
    y=np.asarray(y,bool)
    order=np.argsort(-score,kind='stable')
    yy=y[order]
    return float((np.cumsum(yy)/np.arange(1,len(y)+1))[yy].mean())


def metrics(y,prob,threshold):
    y=np.asarray(y,bool); pred=prob>=threshold
    tp=int((pred&y).sum()); fp=int((pred&~y).sum())
    fn=int((~pred&y).sum()); tn=int((~pred&~y).sum())
    return {'auc':auc(y,prob),'average_precision':average_precision(y,prob),
            'brier':float(np.mean((prob-y)**2)),
            'threshold':float(threshold),'tp':tp,'fp':fp,'fn':fn,'tn':tn,
            'recall':tp/(tp+fn),'precision':tp/(tp+fp) if tp+fp else None,
            'flagged_fraction':(tp+fp)/len(y)}


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--manifest',type=Path,required=True)
    p.add_argument('--policy',type=Path,required=True)
    p.add_argument('--cal-dir',type=Path,required=True)
    p.add_argument('--val-dir',type=Path,required=True)
    p.add_argument('--output',type=Path,required=True)
    args=p.parse_args()
    manifest=json.loads(args.manifest.read_text())
    policy=json.loads(args.policy.read_text())
    manifest_sha=sha(args.manifest)
    if policy['manifest_sha256']!=manifest_sha:
        raise RuntimeError('Policy/manifest mismatch')
    qps=manifest['qps']
    cal=extract(args.cal_dir,manifest['rows']['calibration'],qps,policy['beta'],manifest_sha)
    val=extract(args.val_dir,manifest['rows']['validation'],qps,policy['beta'],manifest_sha)
    if len(cal)!=120 or len(val)!=120 or set(r['image'] for r in cal)&set(r['image'] for r in val):
        raise RuntimeError('Expected disjoint 24-image calibration/validation sets')
    names=['qp_scaled','mean_depth_minus_2','min_depth_minus_2',
           'fraction_deepest','fraction_shallow','mean_edge_jump',
           'fraction_unequal_edges']
    xc=np.asarray([r['features'] for r in cal]); xv=np.asarray([r['features'] for r in val])
    yc=np.asarray([r['over_budget'] for r in cal],dtype=float)
    yv=np.asarray([r['over_budget'] for r in val],dtype=float)
    mean=xc.mean(0); std=xc.std(0); std=np.where(std<1e-8,1,std)
    A=np.column_stack((np.ones(len(cal)),(xc-mean)/std))
    B=np.column_stack((np.ones(len(val)),(xv-mean)/std))
    # Fixed ridge strength; no validation tuning or feature search.
    lam=1.0
    def objective(w):
        z=A@w
        loss=np.logaddexp(0,z).sum()-yc@z+lam*np.dot(w[1:],w[1:])/2
        grad=A.T@(expit(z)-yc)+np.r_[0,lam*w[1:]]
        return loss,grad
    fit=minimize(objective,np.r_[np.log(yc.mean()/(1-yc.mean())),np.zeros(len(names))],
                 jac=True,method='L-BFGS-B',options={'maxiter':1000,'ftol':1e-12})
    if not fit.success:
        raise RuntimeError(f'Risk fit failed: {fit.message}')
    pc=expit(A@fit.x); pv=expit(B@fit.x)
    # Smallest flagged set that catches at least 80% of positives on calibration.
    order=np.argsort(-pc,kind='stable')
    needed=int(np.ceil(.8*yc.sum()))
    threshold=float(pc[order[np.flatnonzero(np.cumsum(yc[order])>=needed)[0]]])
    rng=np.random.default_rng(20261004)
    validation_images=sorted(set(r['image'] for r in val))
    image_indices={image:np.asarray([i for i,r in enumerate(val) if r['image']==image])
                   for image in validation_images}
    auc_samples=[]; difference_samples=[]; recall_samples=[]
    for _ in range(2000):
        draw=rng.choice(validation_images,size=len(validation_images),replace=True)
        ix=np.concatenate([image_indices[image] for image in draw])
        y=yv[ix].astype(bool)
        if y.sum()==0 or y.sum()==len(y):
            continue
        score=auc(y,pv[ix]); baseline=auc(y,-xv[ix,1])
        auc_samples.append(score)
        difference_samples.append(score-baseline)
        recall_samples.append(float(((pv[ix]>=threshold)&y).sum()/y.sum()))
    interval=lambda values:[float(x) for x in np.quantile(values,[.025,.975])]
    result={'schema':1,'scope':'Post-hoc exploratory route/QP-only risk fit; source-quality labels, no source-quality input; no deployment claim',
            'manifest_sha256':manifest_sha,'policy_sha256':sha(args.policy),
            'feature_names':names,'regularization_l2':lam,
            'feature_mean_calibration':mean.tolist(),'feature_std_calibration':std.tolist(),
            'coefficients_intercept_first':fit.x.tolist(),
            'threshold_rule':'Calibration score cutoff for at least 80% violation recall, selected without validation labels',
            'calibration':metrics(yc,pc,threshold),
            'validation':metrics(yv,pv,threshold),
            'baseline_validation_mean_depth_auc':auc(yv,-xv[:,1]),
            'baseline_validation_qp_auc':auc(yv,xv[:,0]),
            'validation_image_cluster_bootstrap':{
                'seed':20261004,'draws_requested':2000,'draws_valid':len(auc_samples),
                'risk_auc_95pct':interval(auc_samples),
                'risk_minus_mean_depth_auc_95pct':interval(difference_samples),
                'recall_at_locked_threshold_95pct':interval(recall_samples)},
            'validation_rows':[{**r,'risk_probability':float(prob),'flagged':bool(prob>=threshold)}
                               for r,prob in zip(val,pv)]}
    args.output.parent.mkdir(parents=True,exist_ok=True)
    args.output.write_text(json.dumps(result,indent=2)+'\n')
    print(json.dumps({k:v for k,v in result.items() if k!='validation_rows'},indent=2))


if __name__=='__main__':
    main()
