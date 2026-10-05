"""Inference-only sparse pointwise path for a frozen GridSeamRepair module.

The depthwise convolution still covers the full frame. The expensive 1x1
channel mix is evaluated only where the learned gate exceeds a fixed
threshold. A plan is created on CPU so its `nonzero` step does not force a
device-side synchronisation during decode. No speed claim is implied.
"""
from __future__ import annotations

from dataclasses import dataclass
import math

import torch
import torch.nn.functional as F


@dataclass(frozen=True)
class SparseRepairPlan:
    batch: int
    height: int
    width: int
    channels: int
    threshold: float
    gate_indices: torch.Tensor
    gate_values: torch.Tensor
    gate_sha256: str
    gate_parameter_id: int
    gate_version: int

    @property
    def active_fraction(self) -> float:
        return self.gate_indices.numel() / (self.batch * self.height * self.width)


def make_plan(module, *, batch: int, height: int, width: int,
              threshold: float, device: torch.device, dtype: torch.dtype) -> SparseRepairPlan:
    """Precompute active spatial indices for one frozen weight/shape setting."""
    import hashlib

    assert not module.training and batch > 0 and height > 0 and width > 0
    assert 0 <= threshold <= 1 and module.pw is not None
    if dtype != torch.float32:
        raise ValueError("The frozen e15 sparse repair path is validated only in FP32")
    gate = module.gate.detach().to(device="cpu", dtype=torch.float32).contiguous()
    patch = int(module.patch)
    assert tuple(gate.shape) == (1, 1, patch, patch)
    tiled = gate.repeat(1, 1, math.ceil(height / patch), math.ceil(width / patch))
    tiled = tiled[:, :, :height, :width].expand(batch, 1, height, width).contiguous()
    flat = tiled.reshape(-1)
    indices = torch.nonzero(flat >= threshold, as_tuple=False).flatten()
    values = flat.index_select(0, indices)
    return SparseRepairPlan(
        batch=batch, height=height, width=width, channels=int(module.dw.in_channels),
        threshold=float(threshold),
        gate_indices=indices.to(device=device, dtype=torch.long),
        gate_values=values.to(device=device, dtype=dtype),
        gate_sha256=hashlib.sha256(gate.numpy().tobytes()).hexdigest(),
        gate_parameter_id=id(module.gate), gate_version=module.gate._version,
    )


def apply_sparse_repair(x: torch.Tensor, module, plan: SparseRepairPlan) -> torch.Tensor:
    """Return `x + gate*PW(act(DW(x)))` at planned active positions."""
    if torch.is_grad_enabled():
        raise RuntimeError("Sparse repair is inference-only; disable autograd")
    n, channels, height, width = x.shape
    if x.dtype != torch.float32:
        raise ValueError("The frozen e15 sparse repair path is validated only in FP32")
    assert (n, height, width, channels) == (
        plan.batch, plan.height, plan.width, plan.channels)
    assert plan.gate_indices.device == x.device and plan.gate_values.dtype == x.dtype
    assert not module.training
    if id(module.gate) != plan.gate_parameter_id or module.gate._version != plan.gate_version:
        raise RuntimeError("Gate changed after sparse plan construction")
    if plan.gate_indices.numel() == 0:
        return x

    # Depthwise context is needed even at the ring, where 3x3 neighborhoods
    # touch pixels that are not themselves active pointwise destinations.
    activated = module.act(module.dw(x))
    flat_activated = activated.permute(0, 2, 3, 1).reshape(-1, channels)
    chosen = flat_activated.index_select(0, plan.gate_indices)
    weight = module.pw.weight.reshape(channels, channels)
    correction = F.linear(chosen, weight, module.pw.bias)
    contribution = plan.gate_values[:, None] * correction
    flat_output = x.permute(0, 2, 3, 1).contiguous().reshape(-1, channels)
    flat_output.index_add_(0, plan.gate_indices, contribution)
    return flat_output.reshape(n, height, width, channels).permute(0, 3, 1, 2).contiguous()
