"""Causal FP32 CPU reference codec for independently trained DCVC-UF depths.

This is a research wire format, NOT the released CUDA bitstream format or
an optimized deployment implementation. It preserves the training-forward
quantization: all selected symbols are coded, with no threshold skipping or
silent int8 clipping. Gaussian CDFs and rANS kernels are pinned upstream;
scale indexes are computed in FP32. Encoder and decoder must use the same
model, numerical backend and stream version. Cross-device bit-exactness is
not assumed. The decoder consumes only bytes and its resident model.
"""
from __future__ import annotations
from dataclasses import dataclass
import hashlib
import importlib.util
import json
import math
from pathlib import Path
import struct
import sys
import time

import numpy as np
import torch
import torch.nn.functional as F

# Native boundary geometry only: image-pad16, latent-pad4 before hyperanalysis.
MAGIC=b'FUFREF2\x00'
# Original H,W; depth,QP,numeric-format,reserved; payload length; model and payload SHA256.
HEADER=struct.Struct('>8sIIBBBBI32s32s')
NUMERIC_FORMAT=1  # FP32, no skip, strict int8 symbols, one entropy partition.
DEPTHS=(2,4,6,8,10,12)
MAX_PIXELS=4096*4096
MAX_PAYLOAD=512*1024*1024


def sha(data):
    return hashlib.sha256(data).hexdigest()


def tensor_hash(value):
    return sha(value.detach().contiguous().numpy().tobytes())


def install_entropy_path(folder):
    """Expose only the explicitly hashed isolated build in this process."""
    folder=Path(folder).resolve()
    meta=json.loads((folder/'build_manifest.json').read_text())
    target=Path(meta['module_path'])
    target=(target if target.is_absolute() else folder/target).resolve()
    if target.parent!=folder or sha(target.read_bytes())!=meta['module_sha256']:
        raise ValueError('Entropy extension binary differs from its build manifest')
    present=sys.modules.get('MLCodec_extensions_cpp')
    if present is not None:
        if Path(present.__file__).resolve()!=target:
            raise RuntimeError('A different entropy extension is already imported')
        return meta
    spec=importlib.util.spec_from_file_location('MLCodec_extensions_cpp',target)
    module=importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    sys.modules[spec.name]=module
    return meta


def validate_dimensions(height,width):
    if not isinstance(height,int) or not isinstance(width,int):
        raise ValueError('Image dimensions must be integers')
    if height<1 or width<1 or height*width>MAX_PIXELS:
        raise ValueError('Invalid or excessive image dimensions')


def make_container(payload,height,width,depth,qp,model_digest):
    validate_dimensions(height,width)
    if depth not in DEPTHS or not isinstance(qp,int) or not 0<=qp<64:
        raise ValueError('Invalid depth or QP')
    if len(payload)<4 or len(payload)>MAX_PAYLOAD:
        raise ValueError('Invalid entropy payload length')
    model_id=bytes.fromhex(model_digest)
    if len(model_id)!=32:raise ValueError('Model identity must be a SHA256 digest')
    return HEADER.pack(MAGIC,height,width,depth,qp,NUMERIC_FORMAT,0,len(payload),
                       model_id,hashlib.sha256(payload).digest())+payload


def parse_container(data,expected_depth,expected_model_digest):
    if not isinstance(data,bytes) or len(data)<HEADER.size:
        raise ValueError('Truncated reference container')
    magic,h,w,depth,qp,numeric,reserved,length,model_id,payload_id=HEADER.unpack_from(data)
    if magic!=MAGIC or numeric!=NUMERIC_FORMAT or reserved!=0:
        raise ValueError('Unknown reference codec format')
    validate_dimensions(h,w)
    if depth!=expected_depth or depth not in DEPTHS or not 0<=qp<64:
        raise ValueError('Depth or QP does not match the decoder')
    if model_id.hex()!=expected_model_digest:
        raise ValueError('Stream was encoded with a different checkpoint')
    if length<4 or length>MAX_PAYLOAD or len(data)!=HEADER.size+length:
        raise ValueError('Payload length mismatch')
    payload=data[HEADER.size:]
    if hashlib.sha256(payload).digest()!=payload_id:
        raise ValueError('Entropy payload checksum mismatch')
    return {'height':h,'width':w,'depth':depth,'qp':qp,'payload':payload,
            'header_bytes':HEADER.size,'payload_bytes':length}


def quarter_values(masked):
    """One nonzero channel quarter per spatial site; preserve NHWC coder order."""
    chunks=masked.chunk(4,dim=1)
    quarter=((chunks[0]+chunks[1])+chunks[2])+chunks[3]
    return quarter.permute(0,2,3,1).contiguous().reshape(-1)


def recover_quarter(values,mask):
    b,c,h,w=mask.shape
    quarter=values.reshape(b,h,w,c//4).permute(0,3,1,2).contiguous()
    return torch.cat([quarter]*4,dim=1)*mask


def symbol_array(values,name):
    if not torch.isfinite(values).all():raise ValueError(f'Non-finite {name} symbols')
    if not torch.equal(values,values.round()):raise ValueError(f'Non-integral {name} symbols')
    if values.numel() and (values.min()<-128 or values.max()>127):
        raise ValueError(f'{name} symbols exceed int8; refusing silent reconstruction-changing clipping')
    return values.to(torch.int8).contiguous().numpy()


def scale_indexes(values,gaussian):
    if not torch.isfinite(values).all():raise ValueError('Non-finite Gaussian scales')
    lo,hi=gaussian.scale_min,gaussian.scale_max
    log_step=(math.log(hi)-math.log(lo))/(gaussian.scale_level-1)
    # Explicit FP32 reference convention; not half-precision CUDA index parity.
    idx=torch.floor((values.clamp(lo,hi).log()-math.log(lo))/log_step)
    return idx.clamp(0,gaussian.scale_level-1).to(torch.uint8).contiguous().numpy()


@dataclass
class Encoded:
    stream: bytes
    reconstruction: torch.Tensor | None
    diagnostics: dict


class ReferenceCodec:
    def __init__(self,net,checkpoint_sha256,extension_dir):
        if len(bytes.fromhex(checkpoint_sha256))!=32:raise ValueError('Invalid checkpoint digest')
        if net.training:raise ValueError('Codec must be in evaluation mode')
        if any(p.device.type!='cpu' or p.dtype!=torch.float32 for p in net.parameters()):
            raise ValueError('This reference implementation requires FP32 CPU parameters')
        self.net=net
        self.depth=len(net.dec.dec_1)-1
        if self.depth not in DEPTHS:raise ValueError('Unsupported synthesis depth')
        self.checkpoint_sha256=checkpoint_sha256
        self.extension_manifest=install_entropy_path(extension_dir)
        net.update(skip_thres=0)
        net.set_entropy_coder_parallel(1)

    def _qp(self,qp):
        return torch.tensor([qp],dtype=torch.int32)

    def _quant(self,qp):
        return [self.net.index_select_dim0(p,self._qp(qp)) for p in
                (self.net.q_scale_enc,self.net.q_scale_dec,self.net.q_scale_y_enc,self.net.q_scale_y_dec)]

    def _prior(self,z_hat,h,w):
        params=self.net.y_prior_fusion(self.net.hyper_dec(z_hat))[:,:,:h,:w]
        return params,self.net.y_spatial_prior_reduction(params)

    def _next_prior(self,stage,so_far,reduced):
        adapter=getattr(self.net,f'y_spatial_prior_adaptor_{stage}')
        return self.net.y_spatial_prior(adapter(torch.cat((so_far,reduced),1))).chunk(2,1)

    @torch.inference_mode()
    def encode(self,x,qp,*,audit=True,reconstruct=True):
        if not isinstance(qp,int) or not 0<=qp<64:raise ValueError('QP must be 0..63')
        if x.ndim!=4 or x.shape[:2]!=(1,3) or x.device.type!='cpu' or x.dtype!=torch.float32:
            raise ValueError('Expected one CPU FP32 NCHW centred YCbCr image')
        if not torch.isfinite(x).all() or x.min()<-.50001 or x.max()>.50001:
            raise ValueError('Expected finite centred YCbCr in [-.5,.5]')
        h,w=x.shape[-2:];validate_dimensions(h,w)
        xp=F.pad(x,(0,(-w)%16,0,(-h)%16),mode='replicate')
        started=time.perf_counter()
        q_enc,q_dec,yq_enc,yq_dec=self._quant(qp)
        y=self.net.enc(xp,q_enc)
        yp=F.pad(y,(0,(-y.shape[-1])%4,0,(-y.shape[-2])%4),mode='replicate')
        z_hat=self.net.hyper_enc(yp).round()
        z_flat=symbol_array(z_hat.permute(0,2,3,1).contiguous().reshape(-1),'z')
        yh,yw=y.shape[-2:]
        params,reduced=self._prior(z_hat,yh,yw)
        masks=self.net.get_mask_4x(*y.shape,y.device)
        scales,means=params.chunk(2,1)
        y=y*yq_enc
        packs=[];trace=[];so_far=None;symbols=[];all_scales=[]
        for stage,mask in enumerate(masks):
            if stage:scales,means=self._next_prior(stage,so_far,reduced)
            mu=means*mask
            y_q=((y-mu)*mask).round()
            flat=symbol_array(quarter_values(y_q),f'y{stage}')
            indexes=scale_indexes(quarter_values(scales*mask),self.net.gaussian_encoder)
            packed=((flat.astype(np.int16)<<8)+indexes.astype(np.int16)).astype(np.int16)
            packs.append(np.ascontiguousarray(packed))
            y_part=y_q+mu
            so_far=y_part if stage==0 else so_far+y_part
            if audit:
                symbols.append(y_q);all_scales.append(scales*mask)
            if audit:
                trace.append({'stage':stage,'symbols':flat.size,'symbols_sha256':sha(flat.tobytes()),
                              'indexes_sha256':sha(indexes.tobytes()),
                              'nonpositive_scales':int((quarter_values(scales*mask)<=0).sum())})
        y_hat=so_far*yq_dec if audit or reconstruct else None
        ec=self.net.entropy_coder
        ec.encoder.reset()
        # rANS is a stack: submit y3,y2,y1,y0,z so the decoder reads z,y0,y1,y2,y3.
        for packed in reversed(packs):ec.encoder.encode_y(packed)
        ec.encoder.encode_z(z_flat,qp*self.net.z_channel,self.net.z_channel)
        ec.encoder.flush()
        payload=ec.encoder.get_encoded_stream().tobytes()
        recon=self.net.dec(y_hat,q_dec)[:,:,:h,:w] if reconstruct else None
        # Diagnostic rate on exactly these quantized symbols, matching deterministic validation.
        if audit:
            sum_symbols=(symbols[0]+symbols[1])+(symbols[2]+symbols[3])
            sum_scales=(all_scales[0]+all_scales[1])+(all_scales[2]+all_scales[3])
            bits_y=float(self.net.get_y_bits(sum_symbols,sum_scales).sum())
            bits_z=float(self.net.get_z_bits(z_hat,self._qp(qp)).sum())
        else:
            bits_y=bits_z=None
        stream=make_container(payload,h,w,self.depth,qp,self.checkpoint_sha256)
        diagnostics={'depth':self.depth,'qp':qp,'height':h,'width':w,'header_bytes':HEADER.size,
                     'payload_bytes':len(payload),'container_bytes':len(stream),
                     'payload_bpp':len(payload)*8/(h*w),'container_bpp':len(stream)*8/(h*w),
                     'estimated_bpp':(bits_y+bits_z)/(h*w) if audit else None,
                     'estimated_bits_y':bits_y,'estimated_bits_z':bits_z,
                     'z_hat_sha256':tensor_hash(z_hat+0.0) if audit else None,
                     'y_hat_sha256':tensor_hash(y_hat) if audit else None,
                     'z_symbols':z_flat.size,'stages':trace,
                     'cpu_encode_and_diagnostic_seconds':time.perf_counter()-started,
                     'timing_scope':('CPU research implementation including diagnostic work; not deployment latency'
                                     if audit or reconstruct else
                                     'CPU research entropy encoder only; not deployment latency')}
        return Encoded(stream,recon,diagnostics)

    @torch.inference_mode()
    def decode(self,stream):
        """No image, encoder outputs, encoder-side scales or symbol indexes accepted."""
        y_hat, q_dec, meta = self.decode_latent(stream)
        h, w = meta['height'], meta['width']
        started = time.perf_counter()
        recon=self.net.dec(y_hat,q_dec)[:,:,:h,:w]
        meta['cpu_synthesis_seconds'] = time.perf_counter()-started
        meta['cpu_decode_seconds'] += meta['cpu_synthesis_seconds']
        return recon, meta

    @torch.inference_mode()
    def decode_latent(self,stream,*,audit=True):
        """Decode real rANS bytes through the shared analysis/entropy path.

        The returned latent and synthesis quantizer are CPU tensors.  Keeping
        this boundary explicit permits paired full-decoder timing without
        passing encoder-side tensors to either arm.
        """
        meta=parse_container(stream,self.depth,self.checkpoint_sha256)
        h,w,qp=meta['height'],meta['width'],meta['qp']
        hp,wp=((h+15)//16)*16,((w+15)//16)*16
        yh,yw=hp//16,wp//16;zh,zw=(yh+3)//4,(yw+3)//4
        started=time.perf_counter()
        ec=self.net.entropy_coder
        ec.decoder.set_stream(np.frombuffer(meta['payload'],dtype=np.uint8).copy())
        ec.decoder.decode_z(self.net.z_channel*zh*zw,qp*self.net.z_channel,self.net.z_channel)
        z_flat=ec.decoder.get_decoded_tensor()
        z_hat=torch.from_numpy(z_flat.astype(np.float32)).reshape(1,zh,zw,self.net.z_channel).permute(0,3,1,2).clone(memory_format=torch.contiguous_format)
        # For a 1x1 hyperlatent, contiguous() alone preserves ambiguous NHWC
        # strides and oneDNN may choose a different convolution path. The
        # explicit clone gives the same canonical NCHW layout as the encoder.
        params,reduced=self._prior(z_hat,yh,yw)
        scales,means=params.chunk(2,1)
        masks=self.net.get_mask_4x(1,256,yh,yw,torch.device('cpu'))
        so_far=None;trace=[]
        for stage,mask in enumerate(masks):
            if stage:scales,means=self._next_prior(stage,so_far,reduced)
            indexes=scale_indexes(quarter_values(scales*mask),self.net.gaussian_encoder)
            ec.decoder.decode_y(indexes)
            flat=ec.decoder.get_decoded_tensor()
            if flat.size!=indexes.size:raise RuntimeError('Decoded symbol count mismatch')
            symbols=torch.from_numpy(flat.astype(np.float32))
            y_q=recover_quarter(symbols,mask)
            y_part=y_q+means*mask
            so_far=y_part if stage==0 else so_far+y_part
            if audit:
                trace.append({'stage':stage,'symbols':flat.size,'symbols_sha256':sha(flat.tobytes()),
                              'indexes_sha256':sha(indexes.tobytes())})
        q=self._qp(qp)
        y_hat=so_far*self.net.index_select_dim0(self.net.q_scale_y_dec,q)
        q_dec=self.net.index_select_dim0(self.net.q_scale_dec,q)
        return y_hat,q_dec,{'height':h,'width':w,'qp':qp,
                      'z_hat_sha256':tensor_hash(z_hat+0.0) if audit else None,
                      'y_hat_sha256':tensor_hash(y_hat) if audit else None,'stages':trace,
                      'cpu_decode_seconds':time.perf_counter()-started,
                      'timing_scope':'CPU research implementation including diagnostic work; not deployment latency',
                      'header_bytes':meta['header_bytes'],'payload_bytes':meta['payload_bytes']}
