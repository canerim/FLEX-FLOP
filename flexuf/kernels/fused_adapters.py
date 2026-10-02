"""Inference-only fused early-exit adapters for the trained e15 ladder."""
from __future__ import annotations

import torch
import torch.nn as nn

from flexuf.backbone.decoder import Conv1x1Adapter, FFNAdapter
from .fused_ffn import fused_expand_activate_reduce, triton
from .fused_pwout import pointwise_add


class FusedAdapter(nn.Module):
    def __init__(self, original: nn.Module):
        super().__init__()
        self.original=original
        self.train(original.training)

    def forward(self,x):
        if (triton is None or self.training or torch.is_grad_enabled() or
                not x.is_cuda or not x.is_contiguous() or
                x.dtype != torch.float32 or
                torch.cuda.get_device_capability(x.device)[0] < 8):
            return self.original(x)
        a=self.original
        if isinstance(a,FFNAdapter):
            if (a.pw_in.out_channels != 4*a.pw_out.in_channels or
                    a.pw_out.in_channels != a.pw_out.out_channels or
                    a.pw_out.out_channels != x.shape[1]):
                return a(x)
            h=fused_expand_activate_reduce(x,a.pw_in)
            return pointwise_add(h,a.pw_out,x)
        if isinstance(a,Conv1x1Adapter):
            return pointwise_add(x,a.conv,x)
        return a(x)


def install_fused_adapters(decoder: nn.Module) -> int:
    """Replace only adapters in a frozen decoder instance; idempotent."""
    if decoder.training:
        raise ValueError('inference-only; call eval()')
    count=0
    for i,adapter in enumerate(decoder.adapters):
        if isinstance(adapter,FusedAdapter):
            continue
        supported_ffn=(isinstance(adapter,FFNAdapter) and
                       adapter.pw_in.out_channels==4*adapter.pw_out.in_channels and
                       adapter.pw_out.in_channels==adapter.pw_out.out_channels)
        if isinstance(adapter,Conv1x1Adapter) or supported_ffn:
            decoder.adapters[i]=FusedAdapter(adapter)
            count+=1
    return count
