"""Optimizer construction and gradient post-processing helpers.

This module is intentionally dependency-light (torch only, no Triton / CUDA
model imports) so its pure functions can be unit-tested on a CPU-only machine.
"""

from typing import List

import torch
import torch.distributed as dist
import torch.nn as nn
from torch.distributed._tensor import DTensor


def build_param_groups(model: nn.Module, weight_decay: float) -> List[dict]:
    """
    Split parameters into weight-decay and no-weight-decay groups.

    Matrix-like parameters (``dim >= 2``: embeddings, projections, expert
    weights, ``lm_head``) receive ``weight_decay``; 1-D parameters (LayerNorm /
    RMSNorm weights, biases) are excluded so normalization scales are not
    shrunk toward zero. Parameters with ``requires_grad=False`` are dropped.

    Returns a list of param-group dicts suitable for ``torch.optim.*``. Empty
    groups are omitted so the optimizer never sees a zero-length group.
    """
    decay: List[nn.Parameter] = []
    no_decay: List[nn.Parameter] = []
    for param in model.parameters():
        if not param.requires_grad:
            continue
        if param.dim() < 2:
            no_decay.append(param)
        else:
            decay.append(param)
    groups: List[dict] = []
    if decay:
        groups.append({"params": decay, "weight_decay": weight_decay})
    if no_decay:
        groups.append({"params": no_decay, "weight_decay": 0.0})
    return groups


@torch.no_grad()
def scale_and_clip_grad_norm_(
    model: nn.Module,
    scale: float = 1.0,
    max_norm: float = 1.0,
    norm_type: float = 2.0,
) -> torch.Tensor:
    """
    Fold gradient-accumulation rescaling and global-norm clipping into a single
    pass over the gradients.

    The pipeline sums gradients over ``accumulate_steps`` micro-batches, so the
    effective mean-reduced gradient is ``scale * g`` with ``scale =
    1/accumulate_steps``. Clipping that gradient to ``max_norm`` multiplies it by
    ``clip_coef = min(1, max_norm / ||scale * g||)``. Rather than sweeping every
    (possibly CPU-offloaded) gradient twice -- once to apply ``scale`` and once
    to apply ``clip_coef`` -- we compute the combined multiplier ``scale *
    clip_coef`` and apply it in one sweep.

    The returned norm is the true pre-clip norm of the mean-reduced gradient
    (i.e. ``scale * ||g||``), matching what the optimizer would have seen.

    Works under FSDP (local-shard ``DTensor`` grads) + pipeline parallelism:
    the squared local norm is all-reduced across every rank when a process
    group is initialized.
    """
    grads: List[torch.Tensor] = []
    for param in model.parameters():
        if param.grad is None:
            continue
        grad = param.grad
        if isinstance(grad, DTensor):
            grad = grad.to_local()
        grads.append(grad)

    if not grads:
        first = next(model.parameters(), None)
        device = first.device if first is not None else torch.device("cpu")
        return torch.tensor(0.0, device=device)

    local_norm = torch.nn.utils.get_total_norm(grads, norm_type=norm_type)
    # Global L2 norm: all-reduce the sum of squared local norms (FSDP + pipeline).
    local_norm_pow = local_norm**norm_type
    if dist.is_available() and dist.is_initialized():
        dist.all_reduce(local_norm_pow, op=dist.ReduceOp.SUM)
    unscaled_norm = local_norm_pow ** (1.0 / norm_type)

    # Norm of the mean-reduced gradient is scale * ||g||; this is what we report
    # and what clipping is computed against.
    total_norm = (unscaled_norm * scale).clamp(min=1e-6)
    clip_coef = (max_norm / total_norm).clamp(max=1.0)

    # Single sweep with the combined multiplier. Skip entirely only when it is a
    # provable no-op (no accumulation rescaling AND no clipping needed); the
    # ``clip_coef < 1.0`` test costs one device->host sync either way.
    apply_clip = bool(clip_coef < 1.0)
    if scale != 1.0 or apply_clip:
        multiplier = clip_coef * scale
        for param in model.parameters():
            if param.grad is not None:
                # Expert grads may live on CPU under FSDP CPUOffloadPolicy while
                # the multiplier is on GPU; match each grad's device.
                param.grad.mul_(multiplier.to(param.grad.device))
    return total_norm
