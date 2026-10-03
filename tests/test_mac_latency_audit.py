"""The analytic proof MAC count agrees with conv hooks on the actual model."""
import sys
from pathlib import Path

import torch
from torch import nn

sys.path[:0] = [str(Path(__file__).resolve().parents[1]), str(Path.home() / "DCVC")]

from src.models.image_model import IntraDecoder
from proof.early_exit_vs_released.mac_latency_audit import (
    CHANNELS, HEAD_CHANNELS, block_macs_per_feature_pixel,
)


def test_released_decoder_conv_mac_formula_matches_hooks():
    torch.set_num_threads(2)
    decoder = IntraDecoder().eval()
    counts = []
    hooks = []

    def count(module, _inputs, output):
        positions = output.shape[0] * output.shape[2] * output.shape[3]
        counts.append(positions * output.shape[1] *
                      module.kernel_size[0] * module.kernel_size[1] *
                      (module.in_channels // module.groups))

    for module in decoder.modules():
        if isinstance(module, nn.Conv2d):
            hooks.append(module.register_forward_hook(count))
    try:
        with torch.inference_mode():
            decoder(torch.randn(1, 256, 2, 2), torch.ones(1, CHANNELS, 1, 1))
    finally:
        for hook in hooks:
            hook.remove()
    feature_pixels = 4 * 4
    stem = 256 * (4 * CHANNELS) // 4 + block_macs_per_feature_pixel(CHANNELS)
    head = CHANNELS * HEAD_CHANNELS + block_macs_per_feature_pixel(HEAD_CHANNELS)
    predicted = feature_pixels * (
        stem + 12 * block_macs_per_feature_pixel(CHANNELS) + head)
    assert sum(counts) == predicted
