"""Fused GEMM+activation regression and inference-only guardrails."""
import copy
import sys
from pathlib import Path

import pytest
import torch

sys.path[:0] = [str(Path(__file__).resolve().parents[1]), str(Path.home() / 'DCVC')]

from src.layers.layers import DepthConvBlock
from flexuf.kernels.fused_ffn import install_fused_ffn


def _pair(device):
    torch.manual_seed(20261002)
    stock = DepthConvBlock(384, 384).to(device).eval()
    fused = copy.deepcopy(stock)
    assert install_fused_ffn(fused) == 1
    assert install_fused_ffn(fused) == 0
    return stock, fused


def test_install_requires_eval():
    with pytest.raises(ValueError, match='eval'):
        install_fused_ffn(DepthConvBlock(384,384))


def test_cpu_fallback_and_weight_gradients():
    stock, fused = _pair('cpu')
    x = torch.randn(1,384,4,4)
    # x itself needs no gradient: the fused first conv's weights still do.
    a = stock(x)
    b = fused(x)
    assert torch.equal(a,b)
    a.sum().backward()
    b.sum().backward()
    assert torch.equal(stock.ffn[0].weight.grad, fused.ffn[0].conv.weight.grad)


@pytest.mark.skipif(not torch.cuda.is_available(), reason='CUDA unavailable')
@pytest.mark.parametrize('shape', [(1,384,8,8),(4,384,16,16)])
def test_cuda_fused_full_block_matches_stock(shape):
    torch.backends.cudnn.allow_tf32 = False
    torch.backends.cuda.matmul.allow_tf32 = False
    stock, fused = _pair('cuda')
    x = torch.randn(shape,device='cuda')*.1
    with torch.inference_mode():
        a,b = stock(x),fused(x)
    assert torch.allclose(a,b,atol=2e-5,rtol=1e-5)


@pytest.mark.skipif(not torch.cuda.is_available(), reason='CUDA unavailable')
def test_cuda_grad_mode_uses_original_conv():
    stock, fused = _pair('cuda')
    x = torch.randn(1,384,4,4,device='cuda')
    a,b = stock(x),fused(x)
    assert torch.equal(a,b)
    a.sum().backward()
    b.sum().backward()
    assert torch.equal(stock.ffn[0].weight.grad, fused.ffn[0].conv.weight.grad)


@pytest.mark.skipif(not torch.cuda.is_available(), reason='CUDA unavailable')
def test_grouped_ffn_convolution_falls_back_without_error():
    torch.manual_seed(4)
    stock=DepthConvBlock(384,384).cuda().eval()
    stock.ffn[0]=torch.nn.Conv2d(384,1536,1,groups=2).cuda().eval()
    fused=copy.deepcopy(stock)
    assert install_fused_ffn(fused)==1
    x=torch.randn(1,384,4,4,device='cuda')*.1
    with torch.inference_mode():
        assert torch.equal(stock(x),fused(x))
