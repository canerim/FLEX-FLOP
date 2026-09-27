"""Predeclared cheap-source-feature association with interim depth benefit.

Exploratory association only: no trained router or held-out prediction claim.
"""
import argparse
import hashlib
import json
from pathlib import Path
import numpy as np
from scipy.stats import spearmanr

ROOT=Path('/data10/shareddata/can_karsal/dcvcuf_depth_20260927')
REPO=Path(__file__).resolve().parents[1]
OUT=REPO/'docs/research/2026-09-27-six-hour/source_features'
FEATURES=('rgb_std','rgb_gradient_energy','rgb_laplacian_energy')


def sha(path):return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def extract():
    manifest_path=ROOT/'research/div2k100_center512_rgb/manifest.json'
    manifest=json.loads(manifest_path.read_text());rows=[]
    for crop in manifest['images']:
        path=Path(crop['crop_path'])
        if sha(path)!=crop['crop_file_sha256']:raise ValueError('Source crop changed')
        x=np.load(path,allow_pickle=False).astype(np.float64)/255
        grad=.5*((np.diff(x,axis=0)**2).mean()+(np.diff(x,axis=1)**2).mean())
        lap=4*x[1:-1,1:-1]-x[:-2,1:-1]-x[2:,1:-1]-x[1:-1,:-2]-x[1:-1,2:]
        rows.append({'image':crop['image'],'crop_sha256':crop['crop_file_sha256'],
                     'rgb_std':float(np.sqrt(x.var(axis=(0,1)).mean())),
                     'rgb_gradient_energy':float(grad),'rgb_laplacian_energy':float((lap**2).mean())})
    result={'scope':'Three predeclared source-only RGB statistics; exploratory association, not a deployed feature runtime measurement',
            'definitions':{'rgb_std':'Square root of mean per-channel spatial variance, RGB in [0,1]',
                           'rgb_gradient_energy':'Mean squared first difference, averaged equally over horizontal/vertical axes and RGB channels',
                           'rgb_laplacian_energy':'Mean squared 4-neighbour Laplacian on interior RGB pixels'},
            'crop_manifest_sha256':sha(manifest_path),'script_sha256':sha(__file__),'rows':rows}
    OUT.mkdir(parents=True,exist_ok=True);path=OUT/'features.json'
    if path.exists() and json.loads(path.read_text())!=result:raise ValueError('Existing feature extraction differs')
    path.write_text(json.dumps(result,indent=2)+'\n')
    print(json.dumps({'features_extracted':len(rows),'out':str(path)}))


def analyze():
    fp=OUT/'features.json';features=json.loads(fp.read_text());fm={r['image']:r for r in features['rows']}
    source=REPO/'docs/research/2026-09-27-six-hour/div2k100_epoch020/analysis.json'
    quality=json.loads(source.read_text());rows=[]
    for record in quality['matched_rate']:
        if record['rate_field']!='payload_bpp' or record['interpolator']!='linear':continue
        for depth in (4,6):
            pair=next(p for p in record['pairs'] if (p['reference_depth'],p['depth'],p['cohort'])==(2,depth,'pairwise_support'))
            ids=pair['images'];deltas=pair['per_image_delta_db']
            for feature in FEATURES:
                xx=[fm[i][feature] for i in ids];yy=[deltas[i] for i in ids]
                rho=None
                if len(ids)>=5 and len(set(xx))>1 and len(set(yy))>1:rho=float(spearmanr(xx,yy).statistic)
                rows.append({'depth':depth,'reference_depth':2,'rate':record['target_bpp'],'feature':feature,
                             'n':len(ids),'spearman_rho':rho,'images':ids,'feature_values':xx,'depth_gain_db':yy})
    result={'scope':'Exploratory same-payload-rate associations; not held-out prediction, causal feature effects, or router success',
            'features_sha256':sha(fp),'quality_analysis_sha256':sha(source),'rows':rows,
            'limitations':'All 18 feature/depth/rate combinations reported. No significance selection. Rate support changes the included image cohort. Single epoch20 checkpoints; no training-seed replication.'}
    (OUT/'association.json').write_text(json.dumps(result,indent=2,allow_nan=False)+'\n')
    lines=['# Ucuz kaynak özelliği ile ek depth faydası arasındaki ilişki','',
           'Bu bir association teşhisidir; router eğitilmedi ve held-out seçim başarısı ölçülmedi. Bütün önceden belirlenmiş feature/depth/rate kombinasyonları gösteriliyor.',
           '', '| Model farkı | Gerçek payload bpp | Feature | n | Spearman rho |','|---|---:|---|---:|---:|']
    for r in rows:
        value='—' if r['spearman_rho'] is None else f"{r['spearman_rho']:+.3f}"
        lines.append(f"| D{r['depth']}−D2 | {r['rate']:.1f} | {r['feature']} | {r['n']} | {value} |")
    lines+=['','Yüksek korelasyon tek başına kullanışlı router kanıtı değildir; uzman seçiminin hatası, gerçek rate–quality ve feature/selector maliyeti ayrı ölçülmeli. Düşük korelasyon ise bu basit özelliğin tek başına yetersiz olabileceğine işaret eder; öğrenilmiş ucuz temsilin başarısını dışlamaz.',
            'Ortak rate desteği dışındaki görüntüler ekstrapole edilmedi. Farklı rate noktalarındaki kohortlar farklı olabilir. Sonuçlar epoch20 ara checkpoint\'lerine koşulludur.']
    (OUT/'REPORT_TR.md').write_text('\n'.join(lines)+'\n');print('Association analysis complete')


if __name__=='__main__':
    ap=argparse.ArgumentParser(description=__doc__);ap.add_argument('--features-only',action='store_true');args=ap.parse_args()
    if args.features_only:extract()
    else:analyze()
