"""Inference-kernel correctness and guardrails; no checkpoint or training touched."""
import sys
from pathlib import Path

import pytest
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
sys.path.insert(0, str(Path.home() / 'DCVC'))

from src.layers.layers import WSiLU, WSiLUChunkAdd
from flexuf.kernels.wsilu_chunkadd import (
    FusedWSiLU, FusedWSiLUChunkAdd, install_fused_plain_wsilu,
    install_fused_wsilu, wsilu, wsilu_chunkadd,
)


def test_install_requires_eval_and_is_idempotent():
    model = torch.nn.Sequential(WSiLU(), WSiLUChunkAdd())
    with pytest.raises(ValueError, match='eval'):
        install_fused_wsilu(model)
    model.eval()
    assert install_fused_wsilu(model) == 1
    assert install_fused_wsilu(model) == 0
    assert install_fused_plain_wsilu(model) == 1
    assert install_fused_plain_wsilu(model) == 0


def test_cpu_and_grad_fallback():
    plain = FusedWSiLU(WSiLU())
    chunk = FusedWSiLUChunkAdd(WSiLUChunkAdd())
    x = torch.randn(2, 16, 3, 5, requires_grad=True)
    a = chunk(plain(x))
    b = WSiLUChunkAdd()(WSiLU()(x))
    assert torch.equal(a, b)
    a.sum().backward()
    assert torch.isfinite(x.grad).all()


@pytest.mark.skipif(not torch.cuda.is_available(), reason='CUDA unavailable')
@pytest.mark.parametrize('shape', [(1, 16, 3, 5), (3, 1536, 16, 16), (1, 1536, 32, 32)])
def test_triton_matches_stock_to_float32_tolerance(shape):
    torch.manual_seed(23)
    x = torch.randn(shape, device='cuda') * 4
    with torch.no_grad():
        ref_chunk = WSiLUChunkAdd().cuda()(x)
        got_chunk = wsilu_chunkadd(x)
        ref_plain = WSiLU().cuda()(x)
        got_plain = wsilu(x)
    assert torch.allclose(got_chunk, ref_chunk, atol=5e-6, rtol=2e-6)
    assert torch.allclose(got_plain, ref_plain, atol=5e-6, rtol=2e-6)


@pytest.mark.skipif(not torch.cuda.is_available(), reason='CUDA unavailable')
def test_cuda_autograd_and_half_fall_back_to_pytorch():
    op = FusedWSiLUChunkAdd(WSiLUChunkAdd())
    x = torch.randn(1, 16, 4, 4, device='cuda', requires_grad=True)
    assert torch.equal(op(x), WSiLUChunkAdd().cuda()(x))
    half = x.detach().half()
    assert torch.equal(op(half), WSiLUChunkAdd().cuda()(half))
