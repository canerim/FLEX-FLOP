"""Depthwise kernel preserves the decoder's zero/replicate boundary behavior."""
import sys
from pathlib import Path

import pytest
import torch
import torch.nn as nn

sys.path[:0]=[str(Path(__file__).resolve().parents[1]),str(Path.home()/'DCVC')]
from flexuf.kernels.depthwise3x3 import depthwise3x3


@pytest.mark.skipif(not torch.cuda.is_available(),reason='CUDA unavailable')
@pytest.mark.parametrize('padding_mode',['zeros','replicate'])
@pytest.mark.parametrize('shape',[(1,192,7,9),(3,384,16,16)])
def test_depthwise_matches_cudnn(shape,padding_mode):
    torch.manual_seed(77)
    x=torch.randn(*shape,device='cuda')*.1
    c=shape[1]
    conv=nn.Conv2d(c,c,3,padding=1,groups=c,padding_mode=padding_mode).cuda().eval()
    with torch.inference_mode():
        stock=conv(x)
        fused=depthwise3x3(x,conv)
    assert torch.allclose(stock,fused,atol=1e-6,rtol=1e-5)


def test_depthwise_rejects_cpu_tensor():
    conv=nn.Conv2d(4,4,3,padding=1,groups=4)
    with pytest.raises((ValueError,RuntimeError)):
        depthwise3x3(torch.randn(1,4,4,4),conv)
