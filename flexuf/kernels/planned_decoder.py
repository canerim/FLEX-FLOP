"""Inference-only planned early-exit execution from a CPU mode map.

In a real bitstream decoder the mode map is parsed on the host. Sorting it on
the GPU and reading cumulative counts back to the host creates a synchronization
that can exceed the small-tile convolutions. This path computes the stable order
and group boundaries on the host before launching the GPU suffix. It does not
modify the live training decoder and supports the frozen e15 deployment geometry
(zero trunk halo, no canvas coupling, full-frame head, no tile gate).
"""
from __future__ import annotations

from dataclasses import dataclass

import torch

from flexuf.backbone.decoder import patchify, unpatchify


@dataclass(frozen=True)
class TilePlan:
    order: torch.Tensor
    inverse: torch.Tensor
    bounds: tuple[int, ...]
    n_tiles: int


def make_tile_plan(exit_map_cpu, *, n_tiles: int, split_depth: int,
                   num_exits: int, device: torch.device) -> TilePlan:
    """Stable descending order and cumulative survivor counts from host metadata."""
    if isinstance(exit_map_cpu, torch.Tensor) and exit_map_cpu.is_cuda:
        raise ValueError('CPU mode map required; GPU-to-host copying would synchronize')
    em = torch.as_tensor(exit_map_cpu, dtype=torch.long, device='cpu')
    if em.shape != (n_tiles,):
        raise ValueError(f'exit_map shape {tuple(em.shape)}, expected ({n_tiles},)')
    if n_tiles <= 64:
        # Maps for the predeclared CTC cohort contain 8--40 tiles. A stable
        # bucket pass avoids several small CPU tensor kernels and dispatches;
        # the histogram gives all suffix survivor bounds in the same pass.
        buckets = [[] for _ in range(num_exits)]
        for tile, value in enumerate(em.tolist()):
            buckets[min(num_exits - 1, max(split_depth, value))].append(tile)
        order = [tile for mode in range(num_exits - 1, split_depth - 1, -1)
                 for tile in buckets[mode]]
        inverse = [0] * n_tiles
        for rank, tile in enumerate(order):
            inverse[tile] = rank
        survivors = n_tiles
        bound_list = []
        for mode in range(split_depth, num_exits):
            survivors -= len(buckets[mode])
            bound_list.append(survivors)
        order_cpu = torch.tensor(order, dtype=torch.long)
        inverse_cpu = torch.tensor(inverse, dtype=torch.long)
        bounds = tuple(bound_list)
    else:
        em = em.clamp(min=split_depth, max=num_exits-1)
        order_cpu = torch.argsort(em, descending=True, stable=True)
        sorted_em = em[order_cpu]
        inverse_cpu = torch.empty_like(order_cpu)
        inverse_cpu[order_cpu] = torch.arange(n_tiles)
        bounds = tuple(int((sorted_em > g).sum())
                       for g in range(split_depth, num_exits))
    return TilePlan(order_cpu.to(device, non_blocking=True),
                    inverse_cpu.to(device, non_blocking=True), bounds, n_tiles)


def forward_with_cpu_map(decoder, y_hat: torch.Tensor, quant_step: torch.Tensor,
                         exit_map_cpu) -> torch.Tensor:
    """Decode one frame with host-planned suffix; exact ordering of sorted path."""
    cfg = decoder.cfg
    if decoder.training or torch.is_grad_enabled():
        raise ValueError('planned decoder is inference-only; use eval() and inference_mode()')
    if not y_hat.is_cuda or y_hat.shape[0] != 1:
        raise ValueError('expected one CUDA latent frame')
    if cfg.tile_coupling or cfg.trunk_halo != 0 or not cfg.full_frame_head:
        raise NotImplementedError('requires no coupling/halo and a full-frame head')
    K, j, Fp = cfg.num_exits, cfg.split_depth, cfg.feature_patch
    feat = decoder.upsample(y_hat)
    for g in range(j):
        feat = decoder.groups[g](feat)
    return forward_from_stem_with_cpu_map(decoder, feat, quant_step, exit_map_cpu)


def forward_from_stem_with_cpu_map(decoder, feat: torch.Tensor,
                                   quant_step: torch.Tensor, exit_map_cpu) -> torch.Tensor:
    """Route from a stem already computed by the decoder-side router.

    The router reads this exact tensor, so recomputing the upsample and shared
    groups inside synthesis would duplicate several expensive blocks. The
    helper also makes the reuse explicit in bitstream-to-image timing.
    """
    cfg = decoder.cfg
    if decoder.training or torch.is_grad_enabled():
        raise ValueError('planned decoder is inference-only; use eval() and inference_mode()')
    if feat.shape[0] != 1 or cfg.tile_coupling or cfg.trunk_halo != 0 or not cfg.full_frame_head:
        raise ValueError('unsupported shared-stem routing geometry')
    K, j, Fp = cfg.num_exits, cfg.split_depth, cfg.feature_patch
    if j >= K:
        return decoder._apply_head(feat, quant_step)

    tiles, nh, nw = patchify(feat, Fp)
    plan = make_tile_plan(exit_map_cpu, n_tiles=tiles.shape[0],
                          split_depth=j, num_exits=K, device=tiles.device)
    undo_pad = None
    if cfg.tile_pad_mode != 'zeros':
        undo_pad = decoder._set_tile_padding(cfg.tile_pad_mode, j)
    try:
        work = tiles[plan.order]
        out = torch.empty_like(work)
        lo = plan.n_tiles
        for idx, g in enumerate(range(j, K)):
            work = decoder.groups[g](work)
            hi = plan.bounds[idx]
            if hi < lo:
                out[hi:lo] = decoder._at_exit(work[hi:lo], g)
                work = work[:hi]
                lo = hi
            if hi == 0:
                break
        stitched = unpatchify(out[plan.inverse], nh, nw, batch=1)
    finally:
        if undo_pad is not None:
            undo_pad()
        elif cfg.tile_pad_mode != 'zeros':
            decoder._set_tile_padding('zeros', j)
    if decoder.seam_repair is not None:
        stitched = decoder.seam_repair(stitched)
    return decoder._apply_head(stitched, quant_step)
