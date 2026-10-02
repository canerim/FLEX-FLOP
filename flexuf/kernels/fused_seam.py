"""Fuse the e15 full-frame grid seam repair into depthwise/pointwise passes."""
from __future__ import annotations

import torch
import torch.nn as nn

from flexuf.backbone.decoder import GridSeamRepair
from .depthwise3x3 import depthwise3x3

try:
    import triton
    import triton.language as tl
except ImportError:
    triton=None
    tl=None


if triton is not None:
    @triton.jit
    def _gated_projection(X,W,B,Gate,Residual,Y,
                          C:tl.constexpr,HW:tl.constexpr,WIDTH:tl.constexpr,
                          M:tl.constexpr,P:tl.constexpr,
                          BM:tl.constexpr,BN:tl.constexpr,BK:tl.constexpr):
        rows=tl.program_id(0)*BM+tl.arange(0,BM)
        cols=tl.program_id(1)*BN+tl.arange(0,BN)
        kk=tl.arange(0,BK)
        acc=tl.full((BM,BN),0,tl.float32)
        for base in range(tl.cdiv(C,BK)):
            k=base*BK+kk
            a=tl.load(X+(rows[:,None]//HW)*C*HW+
                      k[None,:]*HW+rows[:,None]%HW,
                      (rows[:,None]<M)&(k[None,:]<C),0)
            b=tl.load(W+cols[None,:]*C+k[:,None],
                      (cols[None,:]<C)&(k[:,None]<C),0)
            acc=tl.dot(a,b,acc,input_precision='tf32x3')
        offset=(rows[:,None]//HW)*C*HW+cols[None,:]*HW+rows[:,None]%HW
        bias=tl.load(B+cols,cols<C,0)
        pix=rows%HW
        gy=(pix//WIDTH)%P
        gx=(pix%WIDTH)%P
        gate=tl.load(Gate+gy*P+gx,rows<M,0)
        residual=tl.load(Residual+offset,(rows[:,None]<M)&(cols[None,:]<C),0)
        out=residual+gate[:,None]*(acc+bias[None,:])
        tl.store(Y+offset,out,(rows[:,None]<M)&(cols[None,:]<C))


def gated_projection(x:torch.Tensor,conv:nn.Conv2d,
                     gate:torch.Tensor,residual:torch.Tensor,patch:int) -> torch.Tensor:
    if triton is None:
        raise RuntimeError('Triton unavailable')
    if (not x.is_cuda or x.ndim!=4 or x.dtype!=torch.float32 or
            not x.is_contiguous() or not residual.is_contiguous() or
            residual.shape!=x.shape or residual.device!=x.device or
            residual.dtype!=x.dtype or gate.device!=x.device or
            gate.dtype!=x.dtype or not gate.is_contiguous() or
            gate.shape!=(1,1,patch,patch) or
            conv.in_channels!=x.shape[1] or conv.out_channels!=x.shape[1] or
            conv.groups!=1 or conv.kernel_size!=(1,1) or
            conv.padding!=(0,0) or conv.stride!=(1,1) or
            conv.dilation!=(1,1) or conv.bias is None or
            conv.weight.device!=x.device or conv.bias.device!=x.device or
            conv.weight.dtype!=x.dtype or conv.bias.dtype!=x.dtype or
            not conv.weight.is_contiguous() or
            torch.cuda.get_device_capability(x.device)[0]<8):
        raise ValueError('unsupported gated projection')
    n,c,h,w=x.shape
    y=torch.empty_like(x)
    m=n*h*w
    _gated_projection[(triton.cdiv(m,64),triton.cdiv(c,64))](
        x,conv.weight,conv.bias,gate,residual,y,c,h*w,w,m,patch,
        64,64,32,num_warps=4)
    return y


class FusedGridSeamRepair(nn.Module):
    def __init__(self,original:GridSeamRepair):
        super().__init__()
        self.original=original
        self.train(original.training)

    def forward(self,x):
        a=self.original
        if (triton is None or self.training or torch.is_grad_enabled() or
                not x.is_cuda or not x.is_contiguous() or x.dtype!=torch.float32 or
                torch.cuda.get_device_capability(x.device)[0]<8):
            return a(x)
        if a.dw.padding_mode not in ('zeros','replicate'):
            return a(x)
        h=depthwise3x3(x,a.dw,activate=True)
        return gated_projection(h,a.pw,a.gate,x,a.patch)


def install_fused_grid_seam(decoder:nn.Module) -> int:
    if decoder.training:
        raise ValueError('inference-only; call eval()')
    if isinstance(decoder.seam_repair,FusedGridSeamRepair):
        return 0
    if isinstance(decoder.seam_repair,GridSeamRepair):
        decoder.seam_repair=FusedGridSeamRepair(decoder.seam_repair)
        return 1
    return 0
