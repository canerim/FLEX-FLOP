"""Derive an isolated CPU reference with the native16/latent4 padding geometry.

The existing running FUFREF1 reference remains byte-for-byte unchanged. FUFREF2
is a separate FP32 research format, not a native CUDA stream implementation.
"""
import hashlib
import json
from pathlib import Path
import shutil

HERE=Path(__file__).resolve().parent
BASE=HERE.parent/'dcvcuf_reference_20260927'
ROOT=Path('/data10/shareddata/can_karsal/dcvcuf_depth_20260927/research')
OUT=ROOT/'native_shape_reference/source'


def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()


def main():
    verification=json.loads((ROOT/'reference_verification_epoch020/verification.json').read_text())
    for name,digest in verification['code_sha256'].items():
        if sha(BASE/name)!=digest:raise ValueError('Frozen reference source changed: '+name)
    if OUT.exists():raise ValueError('Inspect existing derived source before retry')
    OUT.mkdir(parents=True)
    for name in ('reference_codec.py','model_io.py','decode_worker.py'):
        shutil.copyfile(BASE/name,OUT/name)
    p=OUT/'reference_codec.py';s=p.read_text()
    replacements={
        "MAGIC=b'FUFREF1\\x00'":"MAGIC=b'FUFREF2\\x00'",
        "xp=F.pad(x,(0,(-w)%64,0,(-h)%64),mode='replicate')":"xp=F.pad(x,(0,(-w)%16,0,(-h)%16),mode='replicate')",
        'z_hat=self.net.hyper_enc(y).round()':"yp=F.pad(y,(0,(-y.shape[-1])%4,0,(-y.shape[-2])%4),mode='replicate')\n        z_hat=self.net.hyper_enc(yp).round()",
        'hp,wp=((h+63)//64)*64,((w+63)//64)*64':'hp,wp=((h+15)//16)*16,((w+15)//16)*16',
        'yh,yw=hp//16,wp//16;zh,zw=hp//64,wp//64':'yh,yw=hp//16,wp//16;zh,zw=(yh+3)//4,(yw+3)//4',
    }
    for before,after in replacements.items():
        if s.count(before)!=1:raise ValueError('Expected unique source fragment: '+before)
        s=s.replace(before,after)
    s=s.replace('MAGIC=b\'FUFREF2',"# Native boundary geometry only: image-pad16, latent-pad4 before hyperanalysis.\nMAGIC=b'FUFREF2",1)
    p.write_text(s)
    upstream=Path('/data10/shareddata/can_karsal/dcvcuf_depth_20260927/upstream')
    sources=[upstream/'test_video.py',upstream/'src/layers/extensions/inference/dmci_proxy.cpp',
             upstream/'src/layers/extensions/inference/dmc_common.cpp']
    report={'format':'FUFREF2','scope':'CPU FP32 causal reference with native image-pad16 and replicate latent-pad4 geometry; not CUDA precision, skip-threshold, wire-format or latency parity',
        'base_verification_sha256':sha(ROOT/'reference_verification_epoch020/verification.json'),
        'base_source_sha256':{name:sha(BASE/name) for name in ('reference_codec.py','model_io.py','decode_worker.py')},
        'derived_source_sha256':{p.name:sha(p) for p in OUT.glob('*.py')},
        'native_source_sha256':{str(p):sha(p) for p in sources},'prepare_script_sha256':sha(__file__),
        'source_evidence':'test_video.py requests padding multiple16; DMCIProxy preallocates image dimensions to16 and pads latent dimensions to4; DMCCommon::pad_for_y uses replicate padding.',
        'remaining_checks':'Fresh-process decoding on non64 shapes; parity withFUFREF1 at64-aligned sizes; native GPU correctness remains pending.'}
    (OUT.parent/'preparation.json').write_text(json.dumps(report,indent=2)+'\n')
    (HERE/'preparation.json').write_text(json.dumps(report,indent=2)+'\n')
    print(json.dumps({'source':str(OUT),'format':'FUFREF2'}))


if __name__=='__main__':main()
