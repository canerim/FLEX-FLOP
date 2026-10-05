"""Freeze a third-cohort CLIC transfer before active-beta validation finishes.

The local 41-image CLIC collection is filtered only by geometry: images
large enough for the same centred 768x512 crop used on DIV2K. The two
smaller images are listed explicitly as exclusions. No codec outcome or
router decision enters the selection.
"""
from __future__ import annotations

import hashlib
import json
from pathlib import Path

from PIL import Image

ROOT = Path(__file__).resolve().parents[2]
HERE = Path(__file__).resolve().parent
SOURCE = ROOT/'data/clic'
OUT = HERE/'results/clic39_active_beta/manifest.json'
QPS = [0,16,32,48,63]


def sha(path:Path)->str:
    h=hashlib.sha256()
    with path.open('rb') as f:
        for block in iter(lambda:f.read(1024*1024),b''):
            h.update(block)
    return h.hexdigest()


def main()->None:
    paths=sorted(SOURCE.glob('*.png'))
    assert len(paths)==41
    paths.sort(key=lambda p:hashlib.sha256(p.name.encode()).hexdigest())
    included=[]
    excluded=[]
    for path in paths:
        with Image.open(path) as image:
            w,h=image.size
        row={'image':path.name,'source_sha256':sha(path),'source_size_wh':[w,h]}
        if w<768 or h<512:
            row['reason']='smaller than fixed 768x512 crop'
            excluded.append(row)
            continue
        row['crop_xywh']=[(w-768)//2,(h-512)//2,768,512]
        included.append(row)
    assert len(included)==39 and len(excluded)==2
    assert {r['image'] for r in excluded}=={'schicka-307.png','todd-quackenbush-222.png'}
    output={
        'schema':1,
        'scope':'Local CLIC41 collection; all 39 images at least 768x512, fixed centred 768x512 crop, five QPs; no quality or router outcome used to select images.',
        'selection_rule':'SHA256(filename) order; include every PNG >=768x512, exclude smaller sources by geometry only.',
        'source_dir':'data/clic', 'crop_size_wh':[768,512], 'qps':QPS,
        'expected_cases':len(included)*len(QPS),
        'included':included,'excluded':excluded,
        'script_sha256':sha(Path(__file__)),
        'status':'selection frozen before active beta validation results',
    }
    OUT.parent.mkdir(parents=True,exist_ok=True)
    if OUT.exists():
        assert json.loads(OUT.read_text())==output
    else:
        OUT.write_text(json.dumps(output,indent=2)+'\n')
    print(json.dumps({'images':len(included),'excluded':excluded,
                      'cases':output['expected_cases'],'manifest_sha256':sha(OUT)},indent=2))


if __name__=='__main__':main()
