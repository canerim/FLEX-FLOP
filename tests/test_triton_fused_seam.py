"""The trained grid seam repair can be evaluated in two fused passes."""
import copy
import sys
from pathlib import Path

import pytest
import torch

sys.path[:0]=[str(Path(__file__).resolve().parents[1]),str(Path.home()/'DCVC')]
from flexuf.backbone.decoder import GridSeamRepair,MultiExitIntraDecoder
from flexuf.config import FlexUFConfig
from flexuf.kernels.fused_seam import FusedGridSeamRepair,install_fused_grid_seam


def test_cpu_fallback_matches_stock():
    stock=GridSeamRepair(384,patch=16).eval()
    fused=FusedGridSeamRepair(copy.deepcopy(stock))
    x=torch.randn(1,384,16,16)
    with torch.inference_mode():
        assert torch.equal(stock(x),fused(x))
    assert not fused.training


@pytest.mark.skipif(not torch.cuda.is_available(),reason='CUDA unavailable')
def test_cuda_fused_grid_matches_stock():
    torch.backends.cudnn.allow_tf32=False
    torch.backends.cuda.matmul.allow_tf32=False
    torch.manual_seed(77)
    stock=GridSeamRepair(384,patch=16).cuda().eval()
    torch.nn.init.normal_(stock.pw.weight,std=.002)
    torch.nn.init.normal_(stock.pw.bias,std=.002)
    fused=FusedGridSeamRepair(copy.deepcopy(stock))
    x=torch.randn(1,384,32,32,device='cuda')*.1
    with torch.inference_mode():
        a,b=stock(x),fused(x)
    assert torch.allclose(a,b,atol=5e-6,rtol=2e-6)


def test_seam_installer_is_idempotent():
    dec=MultiExitIntraDecoder(FlexUFConfig(seam_repair='grid')).eval()
    assert install_fused_grid_seam(dec)==1
    assert install_fused_grid_seam(dec)==0
