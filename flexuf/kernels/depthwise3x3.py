"""Inference-only Triton depthwise 3×3 for the early-exit decoder.

The CUDA implementation is benchmarked against PyTorch before opt-in use.
Supports the decoder's zero and replicate padding modes exactly in geometry.
"""
from __future__ import annotations

import torch
import torch.nn as nn

try:
    import triton
    import triton.language as tl
except ImportError:
    triton = None
    tl = None


if triton is not None:
    @triton.jit
    def _depthwise3x3(X,W,B,Y,
                      C:tl.constexpr,H:tl.constexpr,WIDTH:tl.constexpr,
                      TOTAL:tl.constexpr,REPLICATE:tl.constexpr,
                      BLOCK:tl.constexpr):
        idx=tl.program_id(0)*BLOCK+tl.arange(0,BLOCK)
        ok=idx<TOTAL
        xx=idx%WIDTH
        yy=idx//WIDTH%H
        ch=idx//(WIDTH*H)%C
        batch=idx//(WIDTH*H*C)
        channel_base=(batch*C+ch)*H*WIDTH
        acc=tl.load(B+ch,ok,0)
        for dy in range(3):
            y=yy+dy-1
            for dx in range(3):
                x=xx+dx-1
                if REPLICATE:
                    yload=tl.minimum(tl.maximum(y,0),H-1)
                    xload=tl.minimum(tl.maximum(x,0),WIDTH-1)
                    mask=ok
                else:
                    yload=y
                    xload=x
                    mask=ok&(y>=0)&(y<H)&(x>=0)&(x<WIDTH)
                value=tl.load(X+channel_base+yload*WIDTH+xload,mask,0)
                weight=tl.load(W+ch*9+dy*3+dx,ok,0)
                acc=tl.fma(value,weight,acc)
        tl.store(Y+idx,acc,ok)


def depthwise3x3(x:torch.Tensor,conv:nn.Conv2d) -> torch.Tensor:
    if triton is None:
        raise RuntimeError('Triton unavailable')
    if (not x.is_cuda or x.ndim!=4 or x.dtype!=torch.float32 or
            not x.is_contiguous() or conv.weight.device!=x.device or
            conv.bias is None or conv.bias.device!=x.device or
            conv.weight.dtype!=x.dtype or conv.bias.dtype!=x.dtype or
            not conv.weight.is_contiguous() or conv.kernel_size!=(3,3) or
            conv.stride!=(1,1) or conv.dilation!=(1,1) or
            conv.padding!=(1,1) or conv.groups!=x.shape[1] or
            conv.in_channels!=x.shape[1] or conv.out_channels!=x.shape[1] or
            conv.padding_mode not in ('zeros','replicate')):
        raise ValueError('unsupported depthwise convolution')
    n,c,h,w=x.shape
    y=torch.empty_like(x)
    total=n*c*h*w
    _depthwise3x3[(triton.cdiv(total,256),)](
        x,conv.weight,conv.bias,y,c,h,w,total,
        conv.padding_mode=='replicate',256,num_warps=4)
    return y
