"""Exploratory exact-context risk fit using decoder-visible latent features.

All model parameters and the 80%-recall threshold are fit on the 24-image
calibration split. Evaluation uses the disjoint 24-image DIV2K split, which
has already informed research decisions and is not a pristine external test.
"""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import numpy as np
from scipy.optimize import minimize
from scipy.special import expit
from audit_route_only_risk import metrics,auc


ROUTE=['qp','route_mean','route_min','route_std','route_shallow_fraction',
       'route_deep_fraction','route_adjacent_jumps']
LATENT=['stream_bytes','latent_abs_mean','latent_std','latent_abs_p90',
        'latent_zero_fraction','latent_spatial_tv','tile_energy_mean',
        'tile_energy_max','tile_energy_std','shallow_weighted_tile_energy',
        'shallow_deep_energy_gap']


def fit_and_score(cal,val,names,lam=10.):
    xc=np.asarray([[r['features'][name] for name in names] for r in cal],float)
    xv=np.asarray([[r['features'][name] for name in names] for r in val],float)
    yc=np.asarray([r['label_over_0p1'] for r in cal],float)
    yv=np.asarray([r['label_over_0p1'] for r in val],float)
    mean=xc.mean(0);std=np.where(xc.std(0)<1e-8,1,xc.std(0))
    A=np.column_stack((np.ones(len(cal)),(xc-mean)/std))
    B=np.column_stack((np.ones(len(val)),(xv-mean)/std))
    def objective(w):
        z=A@w
        return (np.logaddexp(0,z).sum()-yc@z+lam*np.dot(w[1:],w[1:])/2,
                A.T@(expit(z)-yc)+np.r_[0,lam*w[1:]])
    fit=minimize(objective,np.r_[np.log(yc.mean()/(1-yc.mean())),np.zeros(len(names))],
                 jac=True,method='L-BFGS-B',options={'maxiter':1000,'ftol':1e-12})
    if not fit.success:raise RuntimeError(f'Fit failed: {fit.message}')
    pc,pv=expit(A@fit.x),expit(B@fit.x)
    order=np.argsort(-pc,kind='stable')
    needed=int(np.ceil(.8*yc.sum()))
    threshold=float(pc[order[np.flatnonzero(np.cumsum(yc[order])>=needed)[0]]])
    return {'features':names,'lambda_l2':lam,'feature_mean':mean.tolist(),
            'feature_std':std.tolist(),'coefficients_intercept_first':fit.x.tolist(),
            'threshold':'smallest calibration flagged set reaching at least 80% recall',
            'calibration':metrics(yc,pc,threshold),'validation':metrics(yv,pv,threshold),
            'validation_probabilities':pv.tolist()}


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--features',type=Path,required=True)
    p.add_argument('--output',type=Path,required=True)
    args=p.parse_args()
    data=json.loads(args.features.read_text())
    cal,val=data['rows']['calibration'],data['rows']['validation']
    if len(cal)!=120 or len(val)!=120 or ({r['image'] for r in cal}&{r['image'] for r in val}):
        raise RuntimeError('Expected disjoint 24-image splits')
    route=fit_and_score(cal,val,ROUTE)
    expanded=fit_and_score(cal,val,ROUTE+LATENT)
    y=np.asarray([r['label_over_0p1'] for r in val],bool)
    sr=np.asarray(route['validation_probabilities'])
    se=np.asarray(expanded['validation_probabilities'])
    images=sorted({r['image'] for r in val})
    ix={image:np.asarray([i for i,r in enumerate(val) if r['image']==image]) for image in images}
    rng=np.random.default_rng(20261004)
    diffs=[];auc_expanded=[]
    for _ in range(5000):
        draw=rng.choice(images,size=len(images),replace=True)
        ids=np.concatenate([ix[image] for image in draw])
        yy=y[ids]
        if not yy.any() or yy.all():continue
        a,b=auc(yy,sr[ids]),auc(yy,se[ids])
        diffs.append(b-a);auc_expanded.append(b)
    result={'schema':1,'scope':'Exploratory decoder-visible latent risk audit; source labels for fitting/evaluation only; no source pixels in features; no deployment or latency claim',
            'features_sha256':hashlib.sha256(args.features.read_bytes()).hexdigest(),
            'calibration_positive_count':sum(r['label_over_0p1'] for r in cal),
            'validation_positive_count':int(y.sum()),
            'route_only':route,'route_plus_latent':expanded,
            'validation_cluster_bootstrap':{'draws':len(diffs),
                'expanded_auc_95ci':[float(v) for v in np.quantile(auc_expanded,[.025,.975])],
                'expanded_minus_route_auc_95ci':[float(v) for v in np.quantile(diffs,[.025,.975])]},
            'interpretation':'Validation is disjoint from fit, but was inspected in earlier mechanism work; this is an exploratory transfer check.'}
    args.output.parent.mkdir(parents=True,exist_ok=True)
    args.output.write_text(json.dumps(result,indent=2)+'\n')
    print(json.dumps({'calibration_positive_count':result['calibration_positive_count'],
                      'validation_positive_count':result['validation_positive_count'],
                      'route_only':{'calibration':route['calibration'],'validation':route['validation']},
                      'route_plus_latent':{'calibration':expanded['calibration'],'validation':expanded['validation']},
                      'validation_cluster_bootstrap':result['validation_cluster_bootstrap']},indent=2))


if __name__=='__main__':main()
