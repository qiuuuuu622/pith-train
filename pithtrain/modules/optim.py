"""Optimizer construction and gradient post-processing helpers.

This module is intentionally dependency-light (torch only, no Triton / CUDA
model imports) so its pure functions can be unit-tested on a CPU-only machine.
"""

from typing import List

import torch
import torch.distributed as dist
import torch.nn as nn
from torch.distributed._tensor import DTensor


class ForeachOffloadAdamW(torch.optim.Optimizer):
    """AdamW whose update runs ``torch._foreach_*`` on local DTensor shards.

    ``torch.optim.AdamW``'s ``foreach`` and ``fused`` paths reject FSDP2
    CPU-offloaded ``DTensor`` parameters (the ``_foreach`` kernels have no
    DTensor dispatch), so it silently falls back to the single-tensor path.
    That path sweeps the optimizer state roughly once per arithmetic sub-op per
    tensor -- for the 35B expert state (~4B fp32 params/rank, ~48GB) that is a
    dozen full passes over host memory and measured at ~40s/step, which fully
    exposes the GPU (it sits idle the whole time).

    This optimizer unwraps each parameter / gradient to its *local* tensor with
    ``DTensor.to_local()`` (a view onto the same storage, so in-place updates
    propagate back to the sharded parameter) and runs the AdamW update with the
    batched ``torch._foreach_*`` ops. The whole shard updates in ~2 memory
    passes. The math is identical to ``torch.optim.AdamW`` (decoupled weight
    decay, no amsgrad); see ``tests/test_foreach_offload_adamw.py`` for the
    numerical-equivalence check against the reference.

    All parameters in a group are stepped together every ``step()`` call, so a
    single Python step counter per group drives the (scalar) bias correction --
    no per-tensor step tensors are needed.
    """

    def __init__(self, params, lr, betas=(0.9, 0.95), eps=1e-8, weight_decay=0.0, moment_dtype=None):
        if lr < 0.0:
            raise ValueError(f"Invalid learning rate: {lr}")
        if not 0.0 <= betas[0] < 1.0 or not 0.0 <= betas[1] < 1.0:
            raise ValueError(f"Invalid betas: {betas}")
        # moment_dtype=torch.bfloat16 stores exp_avg/exp_avg_sq in bf16, cutting
        # the optimizer state from 12 to 8 bytes/param. The fp32 master (the
        # sharded param) is untouched and the AdamW math is done in fp32 by
        # upcasting per tensor, so this is a precision-aware optimizer: the only
        # loss is bf16 rounding of the stored moments. It lets the ~48GB->~32GB
        # expert state stay in GPU HBM (no host offload), turning the ~37s
        # bandwidth-bound CPU step into a sub-second HBM step.
        self.moment_dtype = moment_dtype
        defaults = dict(lr=lr, betas=betas, eps=eps, weight_decay=weight_decay, _step=0)
        super().__init__(params, defaults)

    @staticmethod
    def _local(t):
        return t.to_local() if isinstance(t, DTensor) else t

    @torch.no_grad()
    def step(self, closure=None):
        loss = closure() if closure is not None else None
        for group in self.param_groups:
            beta1, beta2 = group["betas"]
            lr, eps, wd = group["lr"], group["eps"], group["weight_decay"]

            params, grads, exp_avgs, exp_avg_sqs = [], [], [], []
            for p in group["params"]:
                if p.grad is None:
                    continue
                p_local = self._local(p)
                state = self.state[p]
                if not state:
                    mdt = self.moment_dtype or p_local.dtype
                    state["exp_avg"] = torch.zeros_like(p_local, dtype=mdt)
                    state["exp_avg_sq"] = torch.zeros_like(p_local, dtype=mdt)
                params.append(p_local)
                grads.append(self._local(p.grad))
                exp_avgs.append(state["exp_avg"])
                exp_avg_sqs.append(state["exp_avg_sq"])
            if not params:
                continue

            group["_step"] += 1
            step = group["_step"]
            bias_correction1 = 1.0 - beta1**step
            bias_correction2 = 1.0 - beta2**step

            # Precision-aware path: moments stored in a reduced dtype. Upcast per
            # tensor so the AdamW math runs in fp32 with only one tensor's worth
            # of transient fp32 memory (keeping the state-size win during step).
            if self.moment_dtype is not None:
                bc2_sqrt = bias_correction2**0.5
                step_size = lr / bias_correction1
                for p_l, g_l, m, v in zip(params, grads, exp_avgs, exp_avg_sqs):
                    mf = m.float()
                    vf = v.float()
                    gf = g_l.float()
                    if wd != 0.0:
                        p_l.mul_(1.0 - lr * wd)
                    mf.lerp_(gf, 1.0 - beta1)
                    vf.mul_(beta2).addcmul_(gf, gf, value=1.0 - beta2)
                    denom = vf.sqrt().div_(bc2_sqrt).add_(eps)
                    p_l.addcdiv_(mf, denom, value=-step_size)
                    m.copy_(mf)
                    v.copy_(vf)
                continue

            # Decoupled weight decay: p *= 1 - lr*wd.
            if wd != 0.0:
                torch._foreach_mul_(params, 1.0 - lr * wd)
            # m = beta1*m + (1-beta1)*g  (lerp toward g by 1-beta1).
            torch._foreach_lerp_(exp_avgs, grads, 1.0 - beta1)
            # v = beta2*v + (1-beta2)*g^2.
            torch._foreach_mul_(exp_avg_sqs, beta2)
            torch._foreach_addcmul_(exp_avg_sqs, grads, grads, 1.0 - beta2)
            # denom = sqrt(v)/sqrt(bias_correction2) + eps.
            denom = torch._foreach_sqrt(exp_avg_sqs)
            torch._foreach_div_(denom, bias_correction2**0.5)
            torch._foreach_add_(denom, eps)
            # p += -(lr/bias_correction1) * m / denom.
            torch._foreach_addcdiv_(params, exp_avgs, denom, -lr / bias_correction1)
        return loss


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
