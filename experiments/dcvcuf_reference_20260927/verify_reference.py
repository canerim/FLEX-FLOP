"""Engineering preflight: actual bytes, fresh decoder process, exact forward parity.

Uses frozen epoch-20 snapshots and the released D12 checkpoint. Its output is
correctness evidence, never final trained-model quality or deployment timing.
"""
from __future__ import annotations
import argparse
import datetime
import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys
import time
os.environ['CUDA_VISIBLE_DEVICES']=''
os.environ.setdefault('OMP_NUM_THREADS','2')
import numpy as np
from PIL import Image
import torch
import torch.nn.functional as F
from model_io import ROOT,load_model,file_sha
from reference_codec import ReferenceCodec,HEADER,parse_container,symbol_array,scale_indexes,quarter_values,recover_quarter

HERE=Path(__file__).resolve().parent
REPO=HERE.parents[1]


def utc():return datetime.datetime.now(datetime.timezone.utc).isoformat()


def checkpoint(depth,epoch):
    return (ROOT/'reference_d12/released_cvpr2026_image.pth.tar' if depth==12 else
            ROOT/f'runs/d{depth}/weights_epoch{epoch:03d}.pt')


def must_reject(fn,label):
    try:fn()
    except (ValueError,TypeError):return label
    raise AssertionError(f'Accepted invalid input: {label}')


def guard_checks(codec,stream):
    rejected=[]
    for n in (0,HEADER.size-1,len(stream)-1):
        rejected.append(must_reject(lambda n=n:parse_container(stream[:n],codec.depth,codec.checkpoint_sha256),f'truncated_{n}'))
    rejected.append(must_reject(lambda:parse_container(stream+b'\0',codec.depth,codec.checkpoint_sha256),'trailing_byte'))
    bad=bytearray(stream);bad[-1]^=1
    rejected.append(must_reject(lambda:parse_container(bytes(bad),codec.depth,codec.checkpoint_sha256),'payload_corruption'))
    rejected.append(must_reject(lambda:parse_container(stream,12 if codec.depth!=12 else 2,codec.checkpoint_sha256),'wrong_depth'))
    rejected.append(must_reject(lambda:parse_container(stream,codec.depth,'00'*32),'wrong_checkpoint'))
    rejected.append(must_reject(lambda:symbol_array(torch.tensor([128.]),'test'),'positive_int8_overflow'))
    rejected.append(must_reject(lambda:symbol_array(torch.tensor([-129.]),'test'),'negative_int8_overflow'))
    rejected.append(must_reject(lambda:symbol_array(torch.tensor([float('nan')]),'test'),'nonfinite_symbol'))
    rejected.append(must_reject(lambda:scale_indexes(torch.tensor([float('inf')]),codec.net.gaussian_encoder),'nonfinite_scale'))
    for shape in ((1,256,4,4),(1,256,5,7)):
        masks=codec.net.get_mask_4x(*shape,torch.device('cpu'))
        pattern=torch.arange(int(np.prod(shape))).reshape(shape).float()%97
        for mask in masks:
            selected=pattern*mask
            assert torch.equal(selected,recover_quarter(quarter_values(selected),mask))
        assert torch.equal(sum(m.int() for m in masks),torch.ones(shape,dtype=torch.int32))
    # Exercise all signs and CDF extremes, especially the QP63 z-row offset.
    ec=codec.net.entropy_coder
    sym=np.array([-128,-9,-1,0,1,9,127],dtype=np.int8)
    idx=np.array([0,1,64,127,0,127,64],dtype=np.uint8)
    packed=((sym.astype(np.int16)<<8)+idx.astype(np.int16)).astype(np.int16)
    z=np.tile(np.array([-128,-1,0,1,127,0,0,0],dtype=np.int8),32)
    ec.encoder.reset();ec.encoder.encode_y(packed);ec.encoder.encode_z(z,63*128,128);ec.encoder.flush()
    payload=ec.encoder.get_encoded_stream()
    ec.decoder.set_stream(payload);ec.decoder.decode_z(len(z),63*128,128)
    first=ec.decoder.get_decoded_tensor();assert np.array_equal(first,z)
    ec.decoder.decode_y(idx);second=ec.decoder.get_decoded_tensor()
    assert np.array_equal(second,sym)
    assert np.array_equal(first,z),'Accessor aliases the mutable decoded buffer'
    return {'rejected_cases':rejected,'mask_recovery_shapes':[[4,4],[5,7]],'rans_extreme_symbol_and_owned_copy_check':True}


def run(args):
    torch.set_num_threads(2)
    args.out.mkdir(parents=True,exist_ok=True)
    image_path=REPO/'data/DIV2K_valid_HR/0801.png'
    rgb_full=np.array(Image.open(image_path).convert('RGB'))
    record={'started_utc':utc(),'state':'running','scope':'Engineering correctness preflight, not final RD or runtime evaluation',
            'epoch_for_shallow_models':args.epoch,'source_image':str(image_path),'source_image_sha256':file_sha(image_path),
            'numeric_backend':{'device':'cpu','precision':'float32','threads':2,'torch':torch.__version__,'numpy':np.__version__},
            'cases':[],'models':{},'all_passed':False}
    sources={p.name:file_sha(p) for p in HERE.glob('*.py')}
    record['code_sha256']=sources
    for depth in args.depths:
        net,source=load_model(checkpoint(depth,args.epoch),depth)
        codec=ReferenceCodec(net,source['sha256'],ROOT/'research/reference_entropy_v1')
        record['models'][str(depth)]=source
        from src.utils.transforms import rgb2ycbcr_np
        for h,w in [(64,64),(65,97),(512,512)]:
            rgb=rgb_full[200:200+h+h%2,200:200+w+w%2]
            # Upstream colour conversion requires even geometry; crop its
            # centred YCbCr output to exercise odd codec dimensions separately.
            yuv=(rgb2ycbcr_np(rgb.astype(np.float32)/255.)-.5)[:h,:w]
            assert yuv.shape[:2]==(h,w)
            x=torch.from_numpy(yuv).permute(2,0,1)[None].contiguous()
            for qp in (0,32,63):
                name=f'd{depth}_{h}x{w}_qp{qp}'
                encoded=codec.encode(x,qp)
                path=args.out/(name+'.fufref');path.write_bytes(encoded.stream)
                target=args.out/(name+'_decoded.npy')
                cmd=[sys.executable,str(HERE/'decode_file.py'),'--checkpoint',str(checkpoint(depth,args.epoch)),
                     '--depth',str(depth),'--stream',str(path),'--output',str(target)]
                completed=subprocess.run(cmd,text=True,capture_output=True,timeout=180)
                if completed.returncode:raise RuntimeError(name+'\n'+completed.stderr)
                decoded=np.load(target,allow_pickle=False)
                diag=json.loads(target.with_suffix('.json').read_text())
                if not np.array_equal(decoded,encoded.reconstruction.numpy()):
                    raise AssertionError(f'{name}: fresh decoder differs, maxabs={np.abs(decoded-encoded.reconstruction.numpy()).max()}')
                with torch.inference_mode():
                    padded=F.pad(x,(0,(-w)%64,0,(-h)%64),mode='replicate')
                    forward=net.forward_one_frame(padded,torch.tensor([qp],dtype=torch.int32),recon_only=True)[:,:,:h,:w]
                if not torch.equal(forward,encoded.reconstruction):raise AssertionError(name+': reference differs from model forward')
                for key in ('z_hat_sha256','y_hat_sha256'):
                    if diag[key]!=encoded.diagnostics[key]:raise AssertionError(name+': latent mismatch '+key)
                for a,b in zip(diag['stages'],encoded.diagnostics['stages']):
                    for key in ('symbols','symbols_sha256','indexes_sha256'):
                        if a[key]!=b[key]:raise AssertionError(name+': stage mismatch '+key)
                case={'name':name,'fresh_process':True,'exact_reconstruction_match':True,'exact_model_forward_match':True,
                      'exact_latent_and_coder_indexes':True,'stream_sha256':file_sha(path),**encoded.diagnostics,
                      'decoder_diagnostics':diag}
                if len(record['cases'])==0:record['guard_checks']=guard_checks(codec,encoded.stream)
                record['cases'].append(case)
                (args.out/'verification.json').write_text(json.dumps(record,indent=2)+'\n')
                print(json.dumps({'case':name,'passed':True,'payload_bytes':encoded.diagnostics['payload_bytes']}),flush=True)
        del codec,net
    if torch.cuda.is_initialized():raise AssertionError('CUDA unexpectedly initialized')
    if sources!={p.name:file_sha(p) for p in HERE.glob('*.py')}:
        raise AssertionError('Source code changed during verification')
    record.update(state='passed',all_passed=True,finished_utc=utc(),cuda_initialized=False)
    (args.out/'verification.json').write_text(json.dumps(record,indent=2)+'\n')
    print(json.dumps({'state':'passed','cases':len(record['cases']),'out':str(args.out),'cuda_initialized':False}),flush=True)


if __name__=='__main__':
    ap=argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--epoch',type=int,default=20)
    ap.add_argument('--depths',nargs='+',type=int,default=[2,4,6,12])
    ap.add_argument('--out',type=Path,default=ROOT/'research/reference_verification_epoch020')
    run(ap.parse_args())
