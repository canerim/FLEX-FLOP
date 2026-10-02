"""Experimental Triton 1x1 FFN expansion fused with WSiLUChunkAdd.

This prototype is opt-in and inference-only. It computes the original first
pointwise convolution in FP32 with tf32x3 tensor-core precision, applies WSiLU,
reduces each group of four interleaved channels and writes only C channels.
The stock path materialises 4C channels before its activation. A paired
benchmark decides whether the fusion beats cuDNN on the deployment geometry.
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
    def _expand_activate_reduce(X, W, Bias, Y,
                                CI: tl.constexpr, CO: tl.constexpr,
                                HW: tl.constexpr, M: tl.constexpr,
                                BM: tl.constexpr, BN: tl.constexpr,
                                BK: tl.constexpr):
        rows = tl.program_id(0) * BM + tl.arange(0, BM)
        cols = tl.program_id(1) * BN + tl.arange(0, BN)
        ks = tl.arange(0, BK)
        acc = tl.full((BM, BN), 0, tl.float32)
        for base in range(tl.cdiv(CI, BK)):
            kk = base * BK + ks
            a = tl.load(X + (rows[:, None] // HW) * CI * HW +
                        kk[None, :] * HW + rows[:, None] % HW,
                        (rows[:, None] < M) & (kk[None, :] < CI), 0)
            b = tl.load(W + cols[None, :] * CI + kk[:, None],
                        (cols[None, :] < 4 * CO) & (kk[:, None] < CI), 0)
            acc = tl.dot(a, b, acc, input_precision='tf32x3')
        z = acc + tl.load(Bias + cols, cols < 4 * CO, 0)[None, :]
        activated = z * tl.sigmoid(4.0 * z)
        reduced = tl.sum(tl.reshape(activated, (BM, BN // 4, 4)), 2)
        out_cols = tl.program_id(1) * (BN // 4) + tl.arange(0, BN // 4)
        offset = ((rows[:, None] // HW) * CO * HW +
                  out_cols[None, :] * HW + rows[:, None] % HW)
        tl.store(Y + offset, reduced,
                 (rows[:, None] < M) & (out_cols[None, :] < CO))


def fused_expand_activate_reduce(x: torch.Tensor, conv: nn.Conv2d) -> torch.Tensor:
    if triton is None:
        raise RuntimeError('Triton unavailable')
    if (not x.is_cuda or not x.is_contiguous() or x.dtype != torch.float32 or
            x.ndim != 4 or not conv.weight.is_contiguous() or conv.bias is None or
            conv.weight.device != x.device or conv.bias.device != x.device or
            conv.weight.dtype != x.dtype or conv.bias.dtype != x.dtype or
            conv.in_channels != x.shape[1] or conv.out_channels % 4 or
            conv.kernel_size != (1,1) or conv.groups != 1 or
            conv.stride != (1,1) or conv.dilation != (1,1) or
            conv.padding != (0,0) or
            torch.cuda.get_device_capability(x.device)[0] < 8):
        raise ValueError('unsupported fused FFN shape/layout')
    n, ci, h, w = x.shape
    co = conv.out_channels // 4
    y = torch.empty((n, co, h, w), dtype=x.dtype, device=x.device)
    m = n*h*w
    _expand_activate_reduce[(triton.cdiv(m,64), triton.cdiv(co,16))](
        x, conv.weight, conv.bias, y, ci, co, h*w, m, 64, 64, 32,
        num_warps=4)
    return y


class FusedFirstPointwise(nn.Module):
    def __init__(self, conv: nn.Conv2d, activation: nn.Module):
        super().__init__()
        self.conv = conv
        self.activation = activation
        self.train(conv.training)

    def forward(self, x):
        if (triton is None or self.training or not x.is_cuda or not x.is_contiguous() or
                x.dtype != torch.float32 or torch.is_grad_enabled() or
                torch.cuda.get_device_capability(x.device)[0] < 8):
            return self.activation(self.conv(x))
        try:
            return fused_expand_activate_reduce(x, self.conv)
        except ValueError:
            return self.activation(self.conv(x))


def install_fused_ffn(decoder: nn.Module) -> int:
    """Fuse the first two FFN modules of each DepthConvBlock in an eval decoder."""
    if decoder.training:
        raise ValueError('inference-only; call eval()')
    from src.layers.layers import DepthConvBlock, WSiLUChunkAdd
    count = 0
    for block in list(decoder.modules()):
        if not isinstance(block, DepthConvBlock):
            continue
        conv, act = block.ffn[0], block.ffn[1]
        if (isinstance(conv, nn.Conv2d) and isinstance(act, WSiLUChunkAdd)
                and conv.out_channels == 4 * block.ffn[2].in_channels):
            block.ffn[0] = FusedFirstPointwise(conv, act)
            block.ffn[1] = nn.Identity()
            count += 1
    return count
