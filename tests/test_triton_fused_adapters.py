"""Both trained adapter families keep their inference output and grad fallback."""
import copy
import sys
from pathlib import Path

import pytest
import torch

sys.path[:0]=[str(Path(__file__).resolve().parents[1]),str(Path.home()/'DCVC')]

from flexuf.backbone.decoder import Conv1x1Adapter, FFNAdapter
from flexuf.kernels.fused_adapters import FusedAdapter, install_fused_adapters
from flexuf.backbone.decoder import MultiExitIntraDecoder
from flexuf.config import FlexUFConfig


@pytest.mark.parametrize('kind',[Conv1x1Adapter,FFNAdapter])
def test_cpu_fallback_and_weight_gradient(kind):
    torch.manual_seed(4)
    stock=kind(384).eval()
    fused=FusedAdapter(copy.deepcopy(stock))
    x=torch.randn(1,384,4,4)
    a,b=stock(x),fused(x)
    assert torch.equal(a,b)
    a.sum().backward();b.sum().backward()
    name='conv' if kind is Conv1x1Adapter else 'pw_in'
    assert torch.equal(getattr(stock,name).weight.grad,
                       getattr(fused.original,name).weight.grad)


@pytest.mark.skipif(not torch.cuda.is_available(),reason='CUDA unavailable')
@pytest.mark.parametrize('kind',[Conv1x1Adapter,FFNAdapter])
def test_cuda_fused_adapter_matches_stock(kind):
    torch.backends.cudnn.allow_tf32=False
    torch.backends.cuda.matmul.allow_tf32=False
    torch.manual_seed(4)
    stock=kind(384).cuda().eval()
    # Nonzero head so an incorrect output can no longer masquerade as identity.
    conv=stock.conv if kind is Conv1x1Adapter else stock.pw_out
    torch.nn.init.normal_(conv.weight,std=.002)
    torch.nn.init.normal_(conv.bias,std=.002)
    fused=FusedAdapter(copy.deepcopy(stock))
    x=torch.randn(3,384,8,8,device='cuda')*.1
    with torch.inference_mode():
        a,b=stock(x),fused(x)
    assert torch.allclose(a,b,atol=5e-6,rtol=2e-6)


def test_installer_is_idempotent():
    dec=MultiExitIntraDecoder(FlexUFConfig(adapter_kind='scaled')).eval()
    assert install_fused_adapters(dec)==dec.cfg.num_exits-1
    assert install_fused_adapters(dec)==0
