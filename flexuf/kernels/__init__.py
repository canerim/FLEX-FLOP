"""Optional inference kernels. Live training imports no code from this package.

Usage after loading a frozen checkpoint::

    net.eval()
    from flexuf.kernels import enable_fast_inference
    enable_fast_inference(net.dec)
    with torch.inference_mode():
        reconstruction = net.dec(y_hat, q_dec, exit_map=mode_map_cuda)

The sorted path is bit-exact against the original map execution. The fused FFN
uses tf32x3 tensor cores and may differ by a few FP32 ULPs; see the benchmark
and quality audit before using it for a reported RD result.
"""
from dataclasses import replace


def enable_fast_inference(decoder, *, sort_tiles: bool = True) -> dict[str, int]:
    """Opt in on one eval decoder instance, after its checkpoint has been loaded."""
    if decoder.training:
        raise ValueError('enable_fast_inference requires decoder.eval()')
    cfg = decoder.cfg
    if (cfg.tile_coupling or cfg.trunk_halo or not cfg.full_frame_head or
            cfg.tile_pad_mode not in ('zeros', 'replicate')):
        raise NotImplementedError('fast kernels are audited for zero-halo, uncoupled, full-head e15 geometry')
    from .fused_ffn import install_fused_ffn
    from .fused_pwout import install_fused_trunk_blocks, install_fused_boundary_blocks
    from .fused_adapters import install_fused_adapters
    from .wsilu_chunkadd import install_fused_plain_wsilu

    counts = {'fused_ffn': install_fused_ffn(decoder),
              'fused_plain_wsilu': install_fused_plain_wsilu(decoder)}
    counts['fused_trunk_blocks'] = install_fused_trunk_blocks(decoder)
    counts['fused_boundary_blocks'] = install_fused_boundary_blocks(decoder)
    counts['fused_adapters'] = install_fused_adapters(decoder)
    if sort_tiles:
        decoder.cfg = replace(decoder.cfg, sorted_tiles=True)
    return counts
