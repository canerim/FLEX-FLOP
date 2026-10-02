"""Whole-block fusion is opt-in and retains the stock fallback and output."""
import copy
import sys
from pathlib import Path

import pytest
import torch

sys.path[:0] = [str(Path(__file__).resolve().parents[1]), str(Path.home() / 'DCVC')]

from src.layers.layers import DepthConvBlock
from flexuf.kernels.fused_ffn import install_fused_ffn
from flexuf.kernels.fused_pwout import FusedTrunkBlock
from flexuf.kernels.wsilu_chunkadd import install_fused_plain_wsilu
from flexuf.kernels import enable_fast_inference
from flexuf.backbone.decoder import MultiExitIntraDecoder
from flexuf.config import FlexUFConfig


def _pair(device):
    torch.manual_seed(31)
    stock=DepthConvBlock(384,384).to(device).eval()
    altered=copy.deepcopy(stock)
    assert install_fused_ffn(altered)==1
    install_fused_plain_wsilu(altered)
    wrapped=FusedTrunkBlock(altered)
    assert not wrapped.training
    assert not wrapped.original.ffn[0].training
    return stock,wrapped


def test_cpu_and_autograd_fallback():
    stock,fused=_pair('cpu')
    x=torch.randn(1,384,4,4)
    a,b=stock(x),fused(x)
    assert torch.equal(a,b)
    a.sum().backward();b.sum().backward()
    assert torch.equal(stock.dc[0].weight.grad,fused.original.dc[0].weight.grad)
    assert torch.equal(stock.ffn[2].weight.grad,fused.original.ffn[2].weight.grad)


@pytest.mark.skipif(not torch.cuda.is_available(), reason='CUDA unavailable')
def test_cuda_trunk_matches_stock():
    torch.backends.cudnn.allow_tf32=False
    torch.backends.cuda.matmul.allow_tf32=False
    stock,fused=_pair('cuda')
    x=torch.randn(4,384,16,16,device='cuda')*.1
    with torch.inference_mode():
        a,b=stock(x),fused(x)
    assert torch.allclose(a,b,atol=5e-6,rtol=2e-6)


@pytest.mark.skipif(not torch.cuda.is_available(), reason='CUDA unavailable')
def test_cuda_trunk_really_dispatches_fused_kernels(monkeypatch):
    import flexuf.kernels.fused_pwout as pw
    import flexuf.kernels.depthwise3x3 as dw
    _,fused=_pair('cuda')
    calls={'pointwise':0,'depthwise':0}
    real_pw=pw.pointwise_wsilu
    real_dw=dw.depthwise3x3

    def measured_pw(*args):
        calls['pointwise']+=1
        return real_pw(*args)

    def measured_dw(*args):
        calls['depthwise']+=1
        return real_dw(*args)

    monkeypatch.setattr(pw,'pointwise_wsilu',measured_pw)
    monkeypatch.setattr(dw,'depthwise3x3',measured_dw)
    with torch.inference_mode():
        fused(torch.randn(1,384,4,4,device='cuda'))
    assert calls=={'pointwise':1,'depthwise':1}


@pytest.mark.parametrize('in_ch,out_ch,shortcut',[(384,192,False),(384,384,True)])
def test_boundary_block_fallback_matches_stock_on_cpu(in_ch,out_ch,shortcut):
    torch.manual_seed(50)
    stock=DepthConvBlock(in_ch,out_ch,shortcut=shortcut).eval()
    fused=FusedTrunkBlock(copy.deepcopy(stock)).eval()
    x=torch.randn(1,in_ch,2,2)
    with torch.inference_mode():
        assert torch.equal(stock(x),fused(x))


@pytest.mark.skipif(not torch.cuda.is_available(), reason='CUDA unavailable')
def test_cuda_boundary_blocks_match_stock():
    torch.backends.cudnn.allow_tf32=False
    torch.backends.cuda.matmul.allow_tf32=False
    torch.manual_seed(51)
    for in_ch,out_ch,shortcut in ((384,192,False),(384,384,True)):
        stock=DepthConvBlock(in_ch,out_ch,shortcut=shortcut).cuda().eval()
        fused=copy.deepcopy(stock)
        assert install_fused_ffn(fused)==1
        fused=FusedTrunkBlock(fused).eval()
        x=torch.randn(2,in_ch,8,8,device='cuda')*.1
        with torch.inference_mode():
            a,b=stock(x),fused(x)
        assert torch.allclose(a,b,atol=5e-6,rtol=2e-6)


def test_public_api_is_opt_in_and_idempotent():
    dec=MultiExitIntraDecoder(FlexUFConfig()).eval()
    before={id(p):p.detach().clone() for p in dec.parameters()}
    counts=enable_fast_inference(dec)
    assert counts['fused_trunk_blocks']==12
    assert counts['fused_boundary_blocks']==2
    assert all(not module.training for module in dec.modules())
    assert counts['fused_ffn']>=12
    assert counts['fused_adapters']==dec.cfg.num_exits-1
    assert dec.cfg.sorted_tiles
    after={id(p):p for p in dec.parameters()}
    assert set(before)==set(after)
    assert all(torch.equal(value,after[key]) for key,value in before.items())
    assert enable_fast_inference(dec)=={'fused_ffn':0,'fused_plain_wsilu':0,
                                        'fused_trunk_blocks':0,'fused_boundary_blocks':0,
                                        'fused_adapters':0}


def test_public_api_rejects_live_training_model():
    dec=MultiExitIntraDecoder(FlexUFConfig())
    with pytest.raises(ValueError,match='eval'):
        enable_fast_inference(dec)


def test_public_api_rejects_untested_geometry():
    dec=MultiExitIntraDecoder(FlexUFConfig(tile_coupling=True)).eval()
    with pytest.raises(NotImplementedError,match='zero-halo'):
        enable_fast_inference(dec)
