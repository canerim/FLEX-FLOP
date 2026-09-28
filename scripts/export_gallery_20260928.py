"""Outcome-blind eight-source gallery from measured QP32 maps and raw source luma."""
import hashlib
import json
import os
import sys
from pathlib import Path
os.environ['CUDA_VISIBLE_DEVICES']=''
import numpy as np
REPO=Path(__file__).resolve().parents[1]
BASE=Path('/data10/shareddata/can_karsal/dcvcuf_depth_20260927/research')
OUT=REPO/'cvpr2027/data/gallery20260928'


def sha(path):
    with Path(path).open('rb') as stream:return hashlib.file_digest(stream,'sha256').hexdigest()


def main():
    raw_path=REPO/'flexplus/results/eval_rules_ctc_e15.json'
    raw=json.loads(raw_path.read_text())
    q32=[row for row in raw['rows'] if row['qp']==32]
    quotas=[('UVG',2),('MCL-JCV',2),('HEVC_B',1),('HEVC_C',1),('HEVC_D',1),('HEVC_E',1)]
    selected=[]
    for group,n in quotas:
        candidates=[r for r in q32 if r['cls']==group]
        selected.extend(sorted(candidates,key=lambda r:hashlib.sha256(('gallery-20260928:'+r['seq']).encode()).hexdigest())[:n])
    assert len(selected)==8 and len({r['seq'] for r in selected})==8
    sys.path.insert(0,str(BASE/'shared_metric_audit/source'))
    sys.path.insert(0,'/home/can_karsal/DCVC')
    import ctc_intra as C
    seqs,_=C.discover([])
    lookup={Path(s['path']).name:s for s in seqs}
    baselines={c['sequence']:c for c in [json.loads(p.read_text()) for p in (BASE/'shared_crossfit_qp32/cases').glob('*.json')]}
    images={};records=[]
    for index,row in enumerate(selected):
        # Legacy `frame` is the archive-wide sequence index, not a temporal offset.
        h,w=row['hw'];assert row['frame']==q32.index(row)
        path=Path(lookup[row['seq']]['path'])
        with path.open('rb') as f:frame=f.read(h*w*3//2)
        frame_sha=hashlib.sha256(frame).hexdigest()
        assert frame_sha==baselines[row['seq']]['first_frame_bytes_sha256']
        # Preserve original luma pixels; raster resampling happens only at plot rendering.
        images[str(index)]=np.frombuffer(frame[:h*w],dtype=np.uint8).reshape(h,w).copy()
        records.append({'gallery_index':index,**row,'first_frame_sha256':frame_sha})
    OUT.mkdir(parents=True,exist_ok=True)
    np.savez_compressed(OUT/'source_luma.npz',**images)
    record={'selection':'Before looking at outcomes: SHA256(gallery-20260928:sequence) ranking within fixed class quotas. UVG2, MCL-JCV2, HEVC B/C/D/E1 each. Retain missing/infeasible maps rather than reselecting images.',
            'scope':'Actual archived QP32 source-calibrated maps. Source luma is spatial context, not reconstructed output or a visual quality test.',
            'views':[{'policy':'source','budget':None},{'policy':'router','budget':.05},{'policy':'router','budget':.1},{'policy':'router','budget':.3},{'policy':'dither','budget':.1},{'policy':'oracle','budget':.1}],
            'source_sha256':{str(raw_path):sha(raw_path),str(Path(__file__)):sha(__file__)},'rows':records,
            'luma_archive_sha256':sha(OUT/'source_luma.npz')}
    (OUT/'gallery.json').write_text(json.dumps(record,indent=2,allow_nan=False)+'\n')
    (OUT/'manifest.json').write_text(json.dumps({name:sha(OUT/name) for name in ['gallery.json','source_luma.npz']},indent=2)+'\n')
    print(json.dumps({'sequences':[r['seq'] for r in records],'views':6}))


if __name__=='__main__':main()
