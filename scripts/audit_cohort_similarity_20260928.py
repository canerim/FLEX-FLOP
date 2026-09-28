"""Outcome-blind first-frame similarity screening of the 53-sequence corpus.

This descriptive audit is not a proof of source independence. It reads no
quality losses, logits or routing outputs. Thresholds are fixed before
fingerprints are computed: 63-bit pHash distance<=6 AND centred64x64 luma
correlation>=0.98. Named resolution families are reported independently.
"""
import os
os.environ['CUDA_VISIBLE_DEVICES']=''
os.environ['OMP_NUM_THREADS']='2'
os.environ['OPENBLAS_NUM_THREADS']='1'
import datetime,hashlib,json,re,sys
from pathlib import Path
import numpy as np
from PIL import Image

REPO=Path(__file__).resolve().parents[1]
ROOT=Path('/data10/shareddata/can_karsal/dcvcuf_depth_20260927/research')
OUT=REPO/'docs/research/2026-09-27-six-hour/cohort_similarity'
SNAPSHOT=ROOT/'shared_metric_audit/source'
UPSTREAM=Path('/home/can_karsal/DCVC')


def sha(p):
    with Path(p).open('rb') as f:return hashlib.file_digest(f,'sha256').hexdigest()


def main():
    OUT.mkdir(exist_ok=True,parents=True)
    calibration_path=REPO/'docs/research/2026-09-27-six-hour/crossfit_control/analysis.json'
    calibration=json.loads(calibration_path.read_text());names=calibration['names'];folds=calibration['folds']
    # Only sequence identity and pre-existing fold assignment are accessed.
    cases=[json.loads(p.read_text()) for p in sorted((ROOT/'released_anchor_qp32/cases').glob('*.json'))]
    if len(cases)!=53 or {c['sequence'] for c in cases}!=set(names):raise ValueError('Expected complete53-frame corpus')
    protocol={'scope':__doc__,'declared_utc':datetime.datetime.now(datetime.timezone.utc).isoformat(),'thresholds':{'phash_hamming_max':6,'centred_luma_correlation_min':.98},'n_sequences':53,'all_pairs':1378,'names':names,'folds':folds,'calibration_manifest_sha256':sha(calibration_path),'script_sha256':sha(__file__),'pixel_processing':'Y plane of the exact archived first8bit420 frame; 32x32 Lanczos luma for63bit low-frequency DCT pHash excluding DC;64x64 Lanczos luma for centred Pearson correlation. No images removed or folds changed.'}
    (OUT/'protocol.json').write_text(json.dumps(protocol,indent=2)+'\n')
    sys.path.insert(0,str(SNAPSHOT));sys.path.insert(0,str(UPSTREAM));import ctc_intra as C
    sequences,_=C.discover([]);lookup={Path(s['path']).name:s for s in sequences}
    positions=np.arange(32);freq=np.arange(8)[:,None];basis=np.cos(np.pi*(2*positions[None,:]+1)*freq/(2*32))
    fingerprints=[];pixels=[];bits=[]
    for case in sorted(cases,key=lambda c:names.index(c['sequence'])):
        name=case['sequence'];h,w=case['height'],case['width'];path=Path(lookup[name]['path'])
        with path.open('rb') as f:raw=f.read(h*w*3//2)
        if len(raw)!=h*w*3//2 or hashlib.sha256(raw).hexdigest()!=case['frame_sha256']:raise ValueError('Archived source identity differs')
        luma=np.frombuffer(raw[:h*w],dtype=np.uint8).reshape(h,w)
        small32=np.asarray(Image.fromarray(luma).resize((32,32),Image.Resampling.LANCZOS),dtype=np.float64)
        small64=np.asarray(Image.fromarray(luma).resize((64,64),Image.Resampling.LANCZOS),dtype=np.float64).ravel()
        coeff=(basis@small32@basis.T).ravel()[1:];code=coeff>np.median(coeff)
        centred=small64-small64.mean();norm=float(np.linalg.norm(centred));vector=centred/norm if norm else np.zeros_like(centred)
        bits.append(code);pixels.append(vector)
        fingerprints.append({'sequence':name,'fold':folds[names.index(name)],'height':h,'width':w,'frame_sha256':case['frame_sha256'],'luma_sha256':hashlib.sha256(raw[:h*w]).hexdigest(),'phash63_bits':''.join('1' if x else '0' for x in code),'resized_luma_std':float(small64.std()),'named_family':re.split(r'_\d+x\d+',name)[0]})
    pairs=[]
    for i,a in enumerate(fingerprints):
        for j in range(i+1,len(fingerprints)):
            b=fingerprints[j];distance=int(np.count_nonzero(bits[i]!=bits[j]));correlation=float(np.dot(pixels[i],pixels[j])) if min(a['resized_luma_std'],b['resized_luma_std'])>0 else None
            pairs.append({'first':a['sequence'],'second':b['sequence'],'folds':[a['fold'],b['fold']],'crosses_folds':a['fold']!=b['fold'],'phash_hamming':distance,'luma_correlation':correlation,'similarity_candidate':distance<=6 and correlation is not None and correlation>=.98,'same_named_family':a['named_family']==b['named_family'],'same_raw_frame':a['frame_sha256']==b['frame_sha256']})
    assert len(pairs)==1378
    result={'protocol':protocol,'fingerprints':fingerprints,'pairs':pairs,'similarity_candidates':[p for p in pairs if p['similarity_candidate']],'named_family_pairs':[p for p in pairs if p['same_named_family']],'nearest_ten':sorted(pairs,key=lambda p:(p['phash_hamming'],-(p['luma_correlation'] or -1)))[:10],'interpretation':'A candidate is a screening flag, not proof of identical source content or temporal independence. Different frames/crops/mirrors can evade these fingerprints. Named resolution variants should be grouped conservatively in future calibration/test splits irrespective of this threshold. Existing sequence-fold results remain development-corpus evidence and are not reselected or recalibrated here.','completed_utc':datetime.datetime.now(datetime.timezone.utc).isoformat()}
    (OUT/'analysis.json').write_text(json.dumps(result,indent=2,allow_nan=False)+'\n')
    print(json.dumps({'candidates':result['similarity_candidates'],'named_families':result['named_family_pairs'],'nearest':result['nearest_ten'][:3]},indent=2))


if __name__=='__main__':main()
