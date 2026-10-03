"""The matched released-D12 inference control preserves the stock fallback."""
import copy
import sys
from pathlib import Path

import torch

sys.path[:0] = [str(Path(__file__).resolve().parents[1]), str(Path.home() / "DCVC")]

from src.models.image_model import IntraDecoder
from flexuf.kernels.released_decoder import enable_fast_released_inference


def test_released_kernel_installation_and_cpu_fallback():
    torch.set_num_threads(2)
    torch.manual_seed(20261003)
    stock = IntraDecoder().eval()
    fast = copy.deepcopy(stock)
    counts = enable_fast_released_inference(fast)
    assert counts == {"fused_ffn": 14, "fused_plain_wsilu": 14, "fused_blocks": 14}
    latent = torch.randn(1, 256, 2, 2)
    quant_step = torch.ones(1, 384, 1, 1)
    with torch.inference_mode():
        assert torch.equal(stock(latent, quant_step), fast(latent, quant_step))
