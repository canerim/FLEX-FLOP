"""Opt-in inference kernels for the *released* DCVC-UF synthesis decoder.

This is the matched implementation control for the e15 fast path. It changes
neither released weights nor the training graph. Unsupported shapes fall back
to the original PyTorch block through ``FusedTrunkBlock``.
"""
from __future__ import annotations

import torch.nn as nn


def enable_fast_released_inference(decoder: nn.Module) -> dict[str, int]:
    """Patch an eval-mode Microsoft ``IntraDecoder`` instance in place."""
    if decoder.training:
        raise ValueError("released inference kernels require decoder.eval()")
    if not hasattr(decoder, "dec_1") or not hasattr(decoder, "dec_2"):
        raise TypeError("expected Microsoft's IntraDecoder")

    from src.layers.layers import DepthConvBlock
    from .fused_ffn import FusedFirstPointwise, install_fused_ffn
    from .fused_pwout import FusedTrunkBlock
    from .wsilu_chunkadd import install_fused_plain_wsilu

    counts = {
        "fused_ffn": install_fused_ffn(decoder),
        "fused_plain_wsilu": install_fused_plain_wsilu(decoder),
    }
    wrapped = 0
    for parent in (decoder.dec_1, decoder):
        for name, block in list(parent.named_children()):
            if not isinstance(block, DepthConvBlock):
                continue
            if not (isinstance(block.ffn[0], FusedFirstPointwise)
                    and block.ffn[2].in_channels == block.ffn[2].out_channels):
                continue
            parent._modules[name] = FusedTrunkBlock(block)
            wrapped += 1
    # The upsampled boundary block is a child of ResidualBlockUpsample.
    upsample = decoder.dec_1[0]
    block = upsample.conv
    if (isinstance(block, DepthConvBlock)
            and isinstance(block.ffn[0], FusedFirstPointwise)
            and block.ffn[2].in_channels == block.ffn[2].out_channels):
        upsample.conv = FusedTrunkBlock(block)
        wrapped += 1
    counts["fused_blocks"] = wrapped
    if counts["fused_ffn"] != 14 or wrapped != 14:
        raise RuntimeError(f"unexpected released D12 structure: {counts}")
    return counts
