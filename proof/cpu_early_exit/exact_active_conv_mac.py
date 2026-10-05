"""Exact padded-frame convolution MAC for the active-canvas early exit.

The coupled 3x3 depthwise reads a (P+2)x(P+2) input with *valid* convolution,
so it emits P x P cells, exactly as the isolated 3x3 with one-cell padding.
Halo assembly changes memory/host work, not convolution output count. The
repair-free active decoder therefore has the routed convolution count of the
deployed decoder minus its full-frame GridSeamRepair convolution count.
"""
from __future__ import annotations

import sys
from pathlib import Path

ROOT=Path(__file__).resolve().parents[2]
sys.path.insert(0,str(ROOT/'proof/early_exit_vs_released'))
from mac_latency_audit import CHANNELS,FEATURE_STRIDE,case_macs


def exact_active_no_repair_saving_pct(exit_map:list[int],
                                      shape_hw:tuple[int,int]=(512,768))->float:
    height,width=shape_hw
    if len(exit_map)!=(height//256)*(width//256):
        raise ValueError('Exit map does not match equal 256-pixel tile grid')
    if any(k not in (2,3,4,5) for k in exit_map):
        raise ValueError('Only the frozen e15 four-exit decoder is supported')
    counts=[exit_map.count(k) for k in range(6)]
    measured=case_macs({'padded_shape':[height,width],'tile_counts':counts})
    feature_pixels=height*width//FEATURE_STRIDE**2
    repair=feature_pixels*(CHANNELS**2+9*CHANNELS)
    active=measured['e15_routed_conv_macs']-repair
    assert active>0
    return 100*(1-active/measured['released_conv_macs'])


def exact_deployed_saving_pct(exit_map:list[int],
                              shape_hw:tuple[int,int]=(512,768))->float:
    height,width=shape_hw
    if len(exit_map)!=(height//256)*(width//256):
        raise ValueError('Exit map does not match equal 256-pixel tile grid')
    if any(k not in (2,3,4,5) for k in exit_map):
        raise ValueError('Only the frozen e15 four-exit decoder is supported')
    counts=[exit_map.count(k) for k in range(6)]
    measured=case_macs({'padded_shape':[height,width],'tile_counts':counts})
    return 100*measured['e15_routed_conv_mac_saving_fraction']
