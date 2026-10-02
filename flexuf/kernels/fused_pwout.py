"""Experimental FP32 pointwise projection with a fused residual addition.

Only an inference copy of the 384-channel trunk can be wrapped. The original
DepthConvBlock remains available for unsupported shapes and autograd.
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
    def _pointwise_epilogue(X, W, Bias, Residual, Y,
                       C: tl.constexpr, HW: tl.constexpr, M: tl.constexpr,
                       ADD: tl.constexpr, ACT: tl.constexpr,
                       BM: tl.constexpr, BN: tl.constexpr, BK: tl.constexpr):
        rows = tl.program_id(0) * BM + tl.arange(0, BM)
        cols = tl.program_id(1) * BN + tl.arange(0, BN)
        ks = tl.arange(0, BK)
        acc = tl.full((BM, BN), 0, tl.float32)
        for base in range(tl.cdiv(C, BK)):
            kk = base * BK + ks
            aa = tl.load(X + (rows[:,None]//HW)*C*HW +
                         kk[None,:]*HW + rows[:,None]%HW,
                         (rows[:,None]<M)&(kk[None,:]<C),0)
            bb = tl.load(W + cols[None,:]*C + kk[:,None],
                         (cols[None,:]<C)&(kk[:,None]<C),0)
            acc = tl.dot(aa,bb,acc,input_precision='tf32x3')
        offset = ((rows[:,None]//HW)*C*HW + cols[None,:]*HW +
                  rows[:,None]%HW)
        bias = tl.load(Bias + cols,cols<C,0)
        val = acc+bias[None,:]
        if ADD:
            res = tl.load(Residual + offset,(rows[:,None]<M)&(cols[None,:]<C),0)
            val = val + res
        if ACT:
            val = val * tl.sigmoid(4.0 * val)
        tl.store(Y + offset,val,
                 (rows[:,None]<M)&(cols[None,:]<C))


def pointwise_add(x: torch.Tensor, conv: nn.Conv2d,
                  residual: torch.Tensor) -> torch.Tensor:
    if triton is None:
        raise RuntimeError('Triton unavailable')
    if (not x.is_cuda or x.ndim != 4 or not x.is_contiguous() or
            not residual.is_contiguous() or residual.device != x.device or
            x.dtype != torch.float32 or residual.dtype != torch.float32 or
            x.shape != residual.shape or conv.in_channels != conv.out_channels or
            conv.in_channels != x.shape[1] or conv.bias is None or
            conv.weight.device != x.device or conv.bias.device != x.device or
            conv.weight.dtype != x.dtype or conv.bias.dtype != x.dtype or
            not conv.weight.is_contiguous() or conv.kernel_size != (1,1) or
            conv.groups != 1 or conv.stride != (1,1) or
            conv.dilation != (1,1) or conv.padding != (0,0) or
            torch.cuda.get_device_capability(x.device)[0] < 8):
        raise ValueError('unsupported pointwise-add shape/layout')
    n,c,h,w=x.shape
    y=torch.empty_like(x)
    m=n*h*w
    _pointwise_epilogue[(triton.cdiv(m,64),triton.cdiv(c,64))](
        x,conv.weight,conv.bias,residual,y,c,h*w,m,True,False,64,64,32,num_warps=4)
    return y


def pointwise_wsilu(x: torch.Tensor, conv: nn.Conv2d) -> torch.Tensor:
    if triton is None:
        raise RuntimeError('Triton unavailable')
    if (not x.is_cuda or x.ndim != 4 or not x.is_contiguous() or
            x.dtype != torch.float32 or
            conv.in_channels != conv.out_channels or conv.in_channels != x.shape[1] or
            conv.bias is None or conv.weight.device != x.device or
            conv.bias.device != x.device or not conv.weight.is_contiguous() or
            conv.weight.dtype != x.dtype or conv.bias.dtype != x.dtype or
            conv.kernel_size != (1,1) or conv.groups != 1 or
            conv.stride != (1,1) or conv.dilation != (1,1) or
            conv.padding != (0,0) or torch.cuda.get_device_capability(x.device)[0] < 8):
        raise ValueError('unsupported pointwise-activation shape/layout')
    n,c,h,w=x.shape
    y=torch.empty_like(x)
    m=n*h*w
    _pointwise_epilogue[(triton.cdiv(m,64),triton.cdiv(c,64))](
        x,conv.weight,conv.bias,x,y,c,h*w,m,False,True,64,64,32,num_warps=4)
    return y


class FusedTrunkBlock(nn.Module):
    def __init__(self, original: nn.Module):
        super().__init__()
        self.original=original
        self.train(original.training)

    def forward(self,x):
        block=self.original
        if (triton is None or self.training or torch.is_grad_enabled() or not x.is_cuda or
                not x.is_contiguous() or x.dtype != torch.float32 or
                torch.cuda.get_device_capability(x.device)[0] < 8):
            return block(x)
        adapted=block.adaptor(x) if block.adaptor is not None else x
        if not adapted.is_contiguous():
            return block(x)
        a=pointwise_wsilu(adapted,block.dc[0])
        from .depthwise3x3 import depthwise3x3
        dw=block.dc[2]
        if (dw.padding_mode not in ('zeros','replicate') or
                dw.padding!=(1,1) or dw.stride!=(1,1)):
            return block(x)
        b=depthwise3x3(a,dw)
        if not b.is_contiguous():
            return block(x)
        mid=pointwise_add(b,block.dc[3],adapted)
        hidden=block.ffn[0](mid)  # already fused expand + WSiLU + chunk-add
        if not hidden.is_contiguous():
            return block(x)
        out=pointwise_add(hidden,block.ffn[2],mid)
        return out+adapted if block.shortcut else out


def install_fused_trunk_blocks(decoder: nn.Module) -> int:
    """Wrap 384-channel suffix/stem blocks, after install_fused_ffn."""
    from .fused_ffn import FusedFirstPointwise
    if decoder.training:
        raise ValueError('inference-only; call eval()')
    count=0
    for group in decoder.groups:
        for idx,block in enumerate(group):
            if isinstance(block,FusedTrunkBlock):
                continue
            if (block.adaptor is None and not block.shortcut and
                    isinstance(block.ffn[0],FusedFirstPointwise) and
                    block.ffn[2].in_channels==block.ffn[2].out_channels):
                group[idx]=FusedTrunkBlock(block)
                count+=1
    return count


def install_fused_boundary_blocks(decoder: nn.Module) -> int:
    """Also fuse the shared upsample block and the RGB head block."""
    from .fused_ffn import FusedFirstPointwise
    if decoder.training:
        raise ValueError('inference-only; call eval()')
    count=0
    for parent,key in ((decoder.upsample,'conv'),(decoder,'head')):
        block=getattr(parent,key)
        if isinstance(block,FusedTrunkBlock):
            continue
        if not (isinstance(block.ffn[0],FusedFirstPointwise) and
                block.ffn[2].in_channels==block.ffn[2].out_channels):
            continue
        setattr(parent,key,FusedTrunkBlock(block))
        count+=1
    return count
