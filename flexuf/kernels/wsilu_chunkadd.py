"""Fused inference-only implementation of DCVC's WSiLUChunkAdd.

The stock module materialises a 4C-channel sigmoid product and launches three
adds after its pointwise convolution. This kernel reads the four interleaved
channels for each output element and writes their sum in a single launch.

No live training module is patched. Call ``install_fused_wsilu(decoder)`` only on
an already-loaded, eval-mode inference instance. The wrapper falls back to the
stock PyTorch implementation for CPU, non-contiguous tensors and autograd.
"""
from __future__ import annotations

import torch
import torch.nn as nn

try:
    import triton
    import triton.language as tl
except ImportError:  # CPU-only environments can still import the package.
    triton = None
    tl = None


if triton is not None:
    @triton.jit
    def _wsilu_kernel(X, Y, TOTAL: tl.constexpr, BLOCK: tl.constexpr):
        i = tl.program_id(0) * BLOCK + tl.arange(0, BLOCK)
        ok = i < TOTAL
        x = tl.load(X + i, ok, 0).to(tl.float32)
        tl.store(Y + i, x * tl.sigmoid(4.0 * x), ok)

    @triton.jit
    def _wsilu_chunkadd_kernel(X, Y, C: tl.constexpr, HW: tl.constexpr,
                               TOTAL: tl.constexpr, BLOCK: tl.constexpr):
        i = tl.program_id(0) * BLOCK + tl.arange(0, BLOCK)
        ok = i < TOTAL
        pix = i % HW
        ch = i // HW % C
        batch = i // (HW * C)
        base = batch * (4 * C * HW) + 4 * ch * HW + pix
        x0 = tl.load(X + base, ok, 0).to(tl.float32)
        x1 = tl.load(X + base + HW, ok, 0).to(tl.float32)
        x2 = tl.load(X + base + 2 * HW, ok, 0).to(tl.float32)
        x3 = tl.load(X + base + 3 * HW, ok, 0).to(tl.float32)
        a = x0 * tl.sigmoid(4.0 * x0)
        b = x1 * tl.sigmoid(4.0 * x1)
        c = x2 * tl.sigmoid(4.0 * x2)
        d = x3 * tl.sigmoid(4.0 * x3)
        tl.store(Y + i, ((a + b) + c) + d, ok)


def wsilu_chunkadd(x: torch.Tensor) -> torch.Tensor:
    """Return the stock module's [N, C, H, W] result for a [N, 4C, H, W] tensor."""
    if triton is None:
        raise RuntimeError("Triton is unavailable")
    if not (x.is_cuda and x.is_contiguous() and x.dtype == torch.float32 and
            x.ndim == 4 and x.shape[1] % 4 == 0):
        raise ValueError("expected contiguous float32 CUDA NCHW with channels divisible by four")
    n, four_c, h, w = x.shape
    c = four_c // 4
    y = torch.empty((n, c, h, w), device=x.device, dtype=x.dtype)
    total = n * c * h * w
    _wsilu_chunkadd_kernel[(triton.cdiv(total, 256),)](
        x, y, c, h * w, total, 256, num_warps=4)
    return y


def wsilu(x: torch.Tensor) -> torch.Tensor:
    """Fused sigmoid(4x)*x for the pointwise and seam-repair activations."""
    if triton is None:
        raise RuntimeError("Triton is unavailable")
    if not (x.is_cuda and x.is_contiguous() and x.dtype == torch.float32):
        raise ValueError("expected contiguous float32 CUDA tensor")
    y = torch.empty_like(x)
    _wsilu_kernel[(triton.cdiv(x.numel(), 512),)](
        x, y, x.numel(), 512, num_warps=4)
    return y


class FusedWSiLUChunkAdd(nn.Module):
    """A guarded drop-in for the parameter-free DCVC activation."""

    def __init__(self, original: nn.Module):
        super().__init__()
        self.original = original
        self.train(original.training)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if (triton is None or self.training or not x.is_cuda or not x.is_contiguous() or
                x.dtype != torch.float32 or
                torch.is_grad_enabled() and x.requires_grad):
            return self.original(x)
        return wsilu_chunkadd(x)


class FusedWSiLU(nn.Module):
    """Guarded inference-only replacement for the plain WSiLU activation."""

    def __init__(self, original: nn.Module):
        super().__init__()
        self.original = original
        self.train(original.training)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if (triton is None or self.training or not x.is_cuda or not x.is_contiguous() or
                x.dtype != torch.float32 or
                torch.is_grad_enabled() and x.requires_grad):
            return self.original(x)
        return wsilu(x)


def install_fused_wsilu(decoder: nn.Module) -> int:
    """Patch only this eval-mode decoder instance; return replaced module count."""
    if decoder.training:
        raise ValueError("install_fused_wsilu requires decoder.eval()")
    if triton is None:
        raise RuntimeError("Triton is unavailable")
    from src.layers.layers import WSiLUChunkAdd

    count = 0
    # Snapshot before mutation: traversing the newly installed wrapper's
    # `original` child would otherwise wrap it again without end.
    for parent in list(decoder.modules()):
        if isinstance(parent, FusedWSiLUChunkAdd):
            continue
        for name, child in list(parent.named_children()):
            if isinstance(child, WSiLUChunkAdd):
                parent._modules[name] = FusedWSiLUChunkAdd(child)
                count += 1
    return count


def install_fused_plain_wsilu(decoder: nn.Module) -> int:
    """Patch DCVC's plain WSiLU modules, leaving nested chunk-add fallbacks intact."""
    if decoder.training:
        raise ValueError("install_fused_plain_wsilu requires decoder.eval()")
    if triton is None:
        raise RuntimeError("Triton is unavailable")
    from src.layers.layers import WSiLU, WSiLUChunkAdd

    count = 0
    for parent in list(decoder.modules()):
        if isinstance(parent, (FusedWSiLU, FusedWSiLUChunkAdd, WSiLUChunkAdd)):
            continue
        for name, child in list(parent.named_children()):
            if isinstance(child, WSiLU):
                parent._modules[name] = FusedWSiLU(child)
                count += 1
    return count
