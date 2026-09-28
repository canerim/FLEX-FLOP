"""Generate manuscript numbers and compact tables from bundled DCVC-UF data."""
from pathlib import Path
import hashlib
import json
import math
import numpy as np
ROOT=Path(__file__).resolve().parents[1]
DATA=ROOT/'data/refresh20260927'


def milestone_table():
    """Keep image support fixed across both milestones; never select a checkpoint."""
    paths=[ROOT/f'data/research20260927/div2k100_epoch{epoch:03d}/analysis.json' for epoch in (20,30)]
    data={epoch:json.loads(path.read_text()) for epoch,path in zip((20,30),paths)}
    for epoch,d in data.items():
        assert d['manifest']['epoch_shallow']==epoch and d['n_cases']==2000 and d['n_images']==100
        keys={(r['image'],r['depth'],r['qp']) for r in d['rows']}
        assert len(keys)==len(d['rows'])==2000
    for key in ('format','crops_manifest_sha256','device','torch','numpy','threads'):
        assert data[20]['manifest'][key]==data[30]['manifest'][key]
    assert data[20]['manifest']['checkpoints']['12']==data[30]['manifest']['checkpoints']['12']
    assert [r for r in data[20]['rows'] if r['depth']==12]==[r for r in data[30]['rows'] if r['depth']==12]
    images=sorted({r['image'] for r in data[20]['rows']})
    assert len(images)==100 and set(images)=={r['image'] for r in data[30]['rows']}
    quality={}
    for epoch,d in data.items():
        for image in images:
            for depth in (2,4,6,12):
                rows=sorted([r for r in d['rows'] if r['image']==image and r['depth']==depth],key=lambda r:r['payload_bpp'])
                rates=np.array([r['payload_bpp'] for r in rows]);psnr=np.array([r['psnr_rgb'] for r in rows])
                assert len(rows)==5 and np.all(np.diff(rates)>0) and np.isfinite(psnr).all()
                quality[epoch,image,depth]=(float(np.interp(math.log(.2),np.log(rates),psnr))
                                           if rates[0]<=.2<=rates[-1] else None)
    common=[image for image in images if all(quality[epoch,image,depth] is not None
                                           for epoch in (20,30) for depth in (2,4,6,12))]
    assert len(common)==99
    rng=np.random.default_rng(20260927)
    draws=rng.integers(0,len(common),(5000,len(common)))
    comparisons=[]
    for epoch in (20,30):
        for ref,depth in ((2,4),(2,6),(4,6)):
            x=np.array([quality[epoch,image,depth]-quality[epoch,image,ref] for image in common])
            comparisons.append({'epoch':epoch,'reference_depth':ref,'depth':depth,'n':len(common),
                                'mean':float(x.mean()),'ci95':np.quantile(x[draws].mean(1),[.025,.975]).tolist(),
                                'per_image':dict(zip(common,x.tolist()))})
    changes=[]
    for depth in (2,4,6):
        x=np.array([quality[30,image,depth]-quality[20,image,depth] for image in common])
        changes.append({'depth':depth,'mean':float(x.mean()),
                        'ci95':np.quantile(x[draws].mean(1),[.025,.975]).tolist(),
                        'per_image':dict(zip(common,x.tolist()))})
    audit={'scope':'Fixed epoch20 and epoch30 milestones, same 99 images on common 0.2 payload-bpp support across both epochs and all four depths; no checkpoint selection.',
           'comparisons':comparisons,'epoch30_minus_epoch20':changes,'common_images':common,
           'statistics':'Linear interpolation in log payload bpp; 5000 paired image bootstrap draws, seed20260927. Fixed checkpoints and cohort; no training-seed uncertainty.',
           'source_sha256':{str(p.relative_to(ROOT)):hashlib.sha256(p.read_bytes()).hexdigest() for p in paths}}
    (DATA/'depth_milestone_audit.json').write_text(json.dumps(audit,indent=2,allow_nan=False)+'\n')
    table=[r'\begin{tabular}{llrr}',r'\toprule',r'Epoch & Contrast & Mean (dB) & 95\% interval \\',r'\midrule']
    for i,row in enumerate(comparisons):
        lo,hi=row['ci95'];label=str(row['epoch']) if i%3==0 else ''
        table.append(f"{label} & D{row['depth']}$-$D{row['reference_depth']} & {row['mean']:+.4f} & [{lo:+.4f}, {hi:+.4f}] \\\\")
        if i==2:table.append(r'\midrule')
    table += [r'\bottomrule',r'\end{tabular}']
    (DATA/'depth_milestone_table.tex').write_text('% Generated from complete fixed-milestone CPU bitstream results.\n'+'\n'.join(table)+'\n')
    return paths


def main():
    cap=json.loads((DATA/'delivered_frontier_audit.json').read_text())
    profile=json.loads((DATA/'exit_depth_profile.json').read_text())
    design=json.loads((DATA/'design_audit.json').read_text())
    rows={r['rule']:r for r in cap['summary'] if r['cap']==.1}
    assert len(rows)==4 and all(r['n']==265 for r in rows.values())
    macros=[]
    for rule,name in [('router','Router'),('dither','Dither'),('uniform','Uniform'),('oracle','Search')]:
        r=rows[rule]
        for suffix,val,fmt in [('Saving',r['mean'],'.2f'),('Loss',r['mean_loss'],'.4f'),('Fallback',r['fallbacks'],'d')]:
            macros.append(r'\newcommand{\Cap'+name+suffix+'}{'+format(val,fmt)+'}')
    for b,name in [(.1,'Main'),(.3,'Loose')]:
        r=next(r for r in cap['contrasts'] if r['cap']==b and r['contrast']=='router_minus_dither')
        for k,suffix in [('mean','Gain'),('lo','Low'),('hi','High')]:
            macros.append(r'\newcommand{\Cap'+name+suffix+'}{'+format(r[k],'.2f')+'}')
    for depth,name in [(6,'Six'),(8,'Eight'),(10,'Ten'),(12,'Twelve')]:
        r=next(r for r in profile['uniform_summary'] if r['depth']==depth and r['qp'] is None)
        macros.append(r'\newcommand{\UniformLoss'+name+'}{'+format(r['mean'],'.3f')+'}')
    header='% Generated by scripts/build_evidence_tables.py; sources are bundled JSON.\n'
    (DATA/'design_macros.tex').write_text(header+'\n'.join(macros)+'\n')
    table=[r'\begin{tabular}{lrrr}',r'\toprule',r'Policy & MAC saved & Mean loss & Fallbacks \\',r' & (\%) & (dB) & (/265) \\',r'\midrule']
    for rule,label in [('uniform','Uniform depth'),('dither','Bayer dither'),('router','Learned router'),('oracle','Source search')]:
        r=rows[rule];table.append(f"{label} & {r['mean']:.2f} & {r['mean_loss']:.4f} & {r['fallbacks']} \\\\")
    table += [r'\bottomrule',r'\end{tabular}']
    (DATA/'delivered_cap_table.tex').write_text(header+'\n'.join(table)+'\n')
    table=[r'\begin{tabular}{lrrl}',r'\toprule',r'Depth & Synthesis & Complete codec & Status \\',r'\midrule']
    for r in design['parameters']['rows']:
        status='Training' if r['depth']<8 else 'Planned' if r['depth']<12 else 'Released anchor'
        table.append(f"D{r['depth']} & {r['decoder_parameters']/1e6:.2f} & {r['total_parameters']/1e6:.2f} & {status} \\\\")
    table += [r'\bottomrule',r'\end{tabular}']
    (DATA/'depth_capacity_table.tex').write_text(header+'\n'.join(table)+'\n')
    replay=json.loads((ROOT/'data/research20260927/shared_crossfit_qp32/analysis.json').read_text())
    rows={(r['criterion'],r['policy']):r for r in replay['summaries']}
    table=[r'\begin{tabular}{llrrr}',r'\toprule',
           r'Control & Policy & MAC saved & RGB loss & $>0.1$ \\',
           r' & & (\%) & (dB) & (/53) \\',r'\midrule']
    for criterion,label in [('mean','Mean'),('q90','Q90')]:
        for i,(policy,name) in enumerate([('uniform','Uniform'),('dither','Dither'),('router','Router')]):
            r=rows[criterion,policy];m=r['metrics'];assert r['n']==53
            values=[label if i==0 else '',name,
                    f"{m['saving_points']['mean']:.2f}",
                    f"{m['cropped_rgb_loss_db']['mean']:.4f}",
                    str(r['above_nominal_target']['cropped_rgb_loss_db'])]
            table.append(' & '.join(values)+r' \\')
        if criterion=='mean':table.append(r'\midrule')
    table += [r'\bottomrule',r'\end{tabular}']
    (DATA/'fixed_control_table.tex').write_text(header+'\n'.join(table)+'\n')
    milestone_sources=milestone_table()
    sources=[DATA/'delivered_frontier_audit.json',DATA/'exit_depth_profile.json',
             DATA/'design_audit.json',ROOT/'data/research20260927/shared_crossfit_qp32/analysis.json']
    sources+=milestone_sources
    artifacts=['design_macros.tex','delivered_cap_table.tex','depth_capacity_table.tex','fixed_control_table.tex',
               'depth_milestone_table.tex','depth_milestone_audit.json']
    digest=lambda p:hashlib.sha256(p.read_bytes()).hexdigest()
    manifest={'script_sha256':digest(Path(__file__)),
              'source_sha256':{str(p.relative_to(ROOT)):digest(p) for p in sources},
              'artifacts':{name:digest(DATA/name) for name in artifacts},
              'scope':'Four numerical tables, 22 macros and a fixed-milestone paired audit generated only from bundled measured/architectural records.'}
    (DATA/'evidence_tables_manifest.json').write_text(json.dumps(manifest,indent=2)+'\n')
    # Update generated artifacts only; never rebaseline the measured source hashes.
    bundle=json.loads((DATA/'bundle_manifest.json').read_text())
    for name in artifacts+['evidence_tables_manifest.json']:
        bundle[name]=digest(DATA/name)
    (DATA/'bundle_manifest.json').write_text(json.dumps(bundle,indent=2)+'\n')
    print('Generated 22 evidence macros, four data-backed tables and one paired milestone audit; no forecast results.')


if __name__=='__main__':main()
