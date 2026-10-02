"""CPU mode-map planning must reproduce the existing sorted inference path."""
import sys
from pathlib import Path

import pytest
import torch

sys.path[:0] = [str(Path(__file__).resolve().parents[1]), str(Path.home() / 'DCVC')]

from flexuf.backbone.decoder import MultiExitIntraDecoder
from flexuf.config import FlexUFConfig
from flexuf.kernels.planned_decoder import forward_with_cpu_map, make_tile_plan


def test_plan_is_stable_and_clamps_like_decoder():
    plan = make_tile_plan([5,2,5,9,0,3],n_tiles=6,split_depth=2,
                          num_exits=6,device=torch.device('cpu'))
    assert plan.order.tolist() == [0,2,3,5,1,4]
    assert plan.inverse[plan.order].tolist() == list(range(6))
    assert plan.bounds == (4,3,3,0)


def test_plan_rejects_wrong_shape():
    with pytest.raises(ValueError,match='expected'):
        make_tile_plan([2,3],n_tiles=3,split_depth=2,num_exits=6,
                       device=torch.device('cpu'))


@pytest.mark.skipif(not torch.cuda.is_available(), reason='CUDA unavailable')
@pytest.mark.parametrize('repair',['none','grid'])
def test_planned_matches_sorted_on_mixed_map(repair):
    torch.backends.cudnn.allow_tf32=False
    torch.backends.cuda.matmul.allow_tf32=False
    torch.manual_seed(13)
    cfg=FlexUFConfig(latent_patch=4,tile_pad_mode='replicate',
                     seam_repair=repair,sorted_tiles=True)
    dec=MultiExitIntraDecoder(cfg).cuda().eval()
    y=torch.randn(1,256,16,16,device='cuda')*.1
    q=torch.ones(1,384,1,1,device='cuda')
    em_cpu=torch.tensor([2,3,4,5]*4)
    with torch.inference_mode():
        ref=dec(y,q,exit_map=em_cpu.cuda())
        out=forward_with_cpu_map(dec,y,q,em_cpu)
    assert torch.equal(ref,out)


@pytest.mark.skipif(not torch.cuda.is_available(), reason='CUDA unavailable')
def test_gpu_mode_map_is_rejected():
    with pytest.raises(ValueError,match='CPU mode map'):
        make_tile_plan(torch.tensor([2,3],device='cuda'),n_tiles=2,
                       split_depth=2,num_exits=6,device=torch.device('cuda'))
