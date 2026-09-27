"""Cache exact 512 centre RGB crops of all 100 DIV2K validation images."""
import argparse
import hashlib
import json
from pathlib import Path
import numpy as np
from PIL import Image
from model_io import ROOT,file_sha

HERE=Path(__file__).resolve().parent
REPO=HERE.parents[1]


def prepare(source,out):
    paths=sorted(source.glob('*.png'))
    expected=[f'{i:04d}.png' for i in range(801,901)]
    if [p.name for p in paths]!=expected:raise ValueError('Expected DIV2K 0801..0900 exactly')
    out.mkdir(parents=True,exist_ok=True)
    rows=[]
    for path in paths:
        with Image.open(path) as im:rgb=np.array(im.convert('RGB'))
        h,w=rgb.shape[:2]
        if min(h,w)<512:raise ValueError(f'Image smaller than crop: {path}')
        top,left=(h-512)//2,(w-512)//2
        crop=rgb[top:top+512,left:left+512].copy()
        target=out/(path.stem+'.npy')
        if target.exists():
            if not np.array_equal(np.load(target,allow_pickle=False),crop):raise ValueError('Existing crop differs')
        else:np.save(target,crop,allow_pickle=False)
        rows.append({'image':path.name,'source_path':str(path),'source_sha256':file_sha(path),
                     'original_height':h,'original_width':w,'top':top,'left':left,'height':512,'width':512,
                     'crop_path':str(target),'crop_file_sha256':file_sha(target),
                     'crop_pixel_sha256':hashlib.sha256(crop.tobytes()).hexdigest(),
                     'monitor_subset':path.name in expected[:4]})
    record={'dataset':'All 100 DIV2K validation images, centre 512x512 crops; no resizing',
            'images':rows,'used_for_checkpoint_selection':False,'prepare_script_sha256':file_sha(Path(__file__))}
    target=out/'manifest.json'
    if target.exists() and json.loads(target.read_text())!=record:
        raise ValueError('Existing crop manifest differs; use a new output folder')
    target.write_text(json.dumps(record,indent=2)+'\n')
    print(json.dumps({'images':len(rows),'out':str(out),'monitor_images':4,'additional_validation_images':96}))


if __name__=='__main__':
    ap=argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--source',type=Path,default=REPO/'data/DIV2K_valid_HR')
    ap.add_argument('--out',type=Path,default=ROOT/'research/div2k100_center512_rgb')
    args=ap.parse_args();prepare(args.source,args.out)
