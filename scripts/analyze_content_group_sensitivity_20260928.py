"""Post-hoc cluster sensitivity; groups fixed by the outcome-blind source audit.

Existing policies, outputs, primary sequence-bootstrap intervals and folds
are unchanged. Resample51 conservative content groups, retaining all members
of each sampled group; the mean continues to give equal weight to sequences.
"""
import hashlib,json
from pathlib import Path
import numpy as np
ROOT=Path(__file__).resolve().parents[1]/'docs/research/2026-09-27-six-hour'
OUT=ROOT/'cohort_similarity'


def main():
    paths={k:ROOT/v for k,v in {'groups':'cohort_similarity/proposed_content_groups.json','replay':'shared_crossfit_qp32/analysis.json','components':'component_interventions/analysis.json','anchor':'released_anchor_qp32/analysis.json'}.items()}
    inputs={k:json.loads(p.read_text()) for k,p in paths.items()};groups=inputs['groups']['groups']
    names=sorted(n for group in groups for n in group)
    if len(groups)!=51 or len(names)!=53 or len(set(names))!=53:raise ValueError('Expected conservative51-group partition')
    specs=[]
    for criterion in ('mean','q90'):
        lookup={(r['sequence'],r['policy']):r for r in inputs['replay']['rows'] if r['criterion']==criterion}
        for field,unit in [('saving_points','percentage points'),('cropped_rgb_loss_db','dB')]:
            values={n:lookup[n,'router'][field]-lookup[n,'dither'][field] for n in names}
            specs.append((criterion+'_router_minus_dither_'+field,unit,values))
    specs.append(('e15_full_rgb_loss_vs_released_db','dB',{r['sequence']:r['e15_full_rgb_loss_vs_released_db'] for r in inputs['anchor']['cases']}))
    for variant in ('repair_identity','adapters_identity','both_identity'):
        specs.append((variant+'_psnr_loss_db','dB',{r['sequence']:r['psnr_loss_db'] for r in inputs['components']['contrasts'] if r['variant']==variant}))
    # Same draws for every paired endpoint, with a declared fixed seed.
    draws=np.random.default_rng(20260928).integers(0,len(groups),size=(5000,len(groups)))
    group_sizes=np.array([len(g) for g in groups],dtype=float);denominator=group_sizes[draws].sum(axis=1)
    rows=[]
    for name,unit,values in specs:
        if set(values)!=set(names):raise ValueError('Metric cohort differs')
        sums=np.array([sum(values[n] for n in g) for g in groups]);samples=sums[draws].sum(axis=1)/denominator
        rows.append({'metric':name,'unit':unit,'n_sequences':53,'n_clusters':51,'mean':float(np.mean([values[n] for n in names])),'cluster_ci95':[float(x) for x in np.quantile(samples,[.025,.975])],'per_sequence':values})
    result={'scope':__doc__,'rows':rows,'bootstrap':{'draws':5000,'seed':20260928,'scheme':'Sample51 groups with replacement; include every sequence in each selected group, then average sequences. Group sizes vary, so each draw can contain a different number of sequence rows.'},'source_sha256':{k:hashlib.sha256(p.read_bytes()).hexdigest() for k,p in paths.items()},'script_sha256':hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),'interpretation':'Descriptive sensitivity to the two conservatively joined candidate pairs. Does not repair the original fold assignment, establish complete content independence, supply training-seed uncertainty or create a new held-out performance test.'}
    (OUT/'cluster_sensitivity.json').write_text(json.dumps(result,indent=2,allow_nan=False)+'\n')
    print(json.dumps([{k:v for k,v in r.items() if k!='per_sequence'} for r in rows],indent=2))


if __name__=='__main__':main()
