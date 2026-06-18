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

    def __init__(
        self,
        params,
        lr,
        betas=(0.9, 0.95),
        eps=1e-8,
        weight_decay=0.0,
        moment_dtype=None,
        stochastic_rounding=False,
        compute_device=None,
    ):
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
        #
        # stochastic_rounding=True applies unbiased stochastic rounding when
        # writing fp32 results back to a bf16 destination. This is what makes a
        # bf16 *master* viable: with round-to-nearest, any per-step update smaller
        # than the bf16 ULP (~2^-8 of the weight magnitude -- common once the LR
        # has decayed) is silently dropped and training stalls; stochastic
        # rounding preserves those updates in expectation. It frees a further
        # ~16GB (no fp32 master) for a larger micro-batch.
        # compute_device routes the AdamW math onto a fast device while the state
        # stays put. For CPU-offloaded experts (state in host RAM) the host AdamW
        # is bandwidth-bound at ~37s/step (8 ranks share the host) and fully
        # exposes the GPU. Setting compute_device="cuda" streams each tensor's
        # state+grad to HBM, does the (sub-second) math there, and writes the
        # result back to host -- turning a CPU-compute-bound step into a
        # PCIe-transfer-bound one (~48GB x2, a few seconds, and overlap-able).
        # None keeps the math on each tensor's own device (no transfer).
        self.moment_dtype = moment_dtype
        self.stochastic_rounding = stochastic_rounding
        self.compute_device = None if compute_device is None else torch.device(compute_device)
        defaults = dict(lr=lr, betas=betas, eps=eps, weight_decay=weight_decay, _step=0)
        super().__init__(params, defaults)

    @staticmethod
    def _local(t):
        return t.to_local() if isinstance(t, DTensor) else t

    @staticmethod
    def _sr_to_bf16(x: torch.Tensor) -> torch.Tensor:
        """Stochastically round fp32 ``x`` to bf16 via the bit trick: add uniform
        noise across the 16 truncated mantissa bits, then truncate. A value k/2^16
        of the way to the next bf16 rounds up with probability k/2^16, so rounding
        is unbiased. Exactly-representable values (low 16 bits zero) stay exact."""
        xi = x.contiguous().view(torch.int32)
        noise = torch.randint(0, 1 << 16, x.shape, dtype=torch.int32, device=x.device)
        return ((xi + noise) & -65536).view(torch.float32).to(torch.bfloat16)

    @classmethod
    def _store(cls, dst: torch.Tensor, src_fp32: torch.Tensor, stochastic: bool) -> None:
        """Write fp32 ``src_fp32`` into ``dst`` (possibly low precision)."""
        if stochastic and dst.dtype == torch.bfloat16:
            dst.copy_(cls._sr_to_bf16(src_fp32))
        else:
            dst.copy_(src_fp32)

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

            # Precision-aware path: moments (and optionally the master) stored in a
            # reduced dtype. Upcast per tensor so the AdamW math runs in fp32 with
            # only one tensor's worth of transient fp32 memory (keeping the
            # state-size win during step), then write back, stochastically rounding
            # bf16 destinations when enabled. For an fp32 master with
            # stochastic_rounding=False this is bitwise-identical to the foreach
            # path below (``_store`` is a plain copy when dtypes match).
            if (
                self.moment_dtype is not None
                or self.stochastic_rounding
                or self.compute_device is not None
            ):
                sr = self.stochastic_rounding
                cd = self.compute_device
                bc2_sqrt = bias_correction2**0.5
                step_size = lr / bias_correction1

                def _fp32(t):
                    # Upcast (and move to the compute device, streaming offloaded
                    # state to fast memory when cd is set). cd=None keeps it home.
                    return t.to(cd, torch.float32) if cd is not None else t.float()

                for p_l, g_l, m, v in zip(params, grads, exp_avgs, exp_avg_sqs):
                    pf = _fp32(p_l)
                    mf = _fp32(m)
                    vf = _fp32(v)
                    gf = _fp32(g_l)
                    if wd != 0.0:
                        pf.mul_(1.0 - lr * wd)
                    mf.lerp_(gf, 1.0 - beta1)
                    vf.mul_(beta2).addcmul_(gf, gf, value=1.0 - beta2)
                    denom = vf.sqrt().div_(bc2_sqrt).add_(eps)
                    pf.addcdiv_(mf, denom, value=-step_size)
                    # _store copies back to the home tensor (D2H when cd is a GPU).
                    self._store(p_l, pf, sr)
                    self._store(m, mf, sr)
                    self._store(v, vf, sr)
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

    # Grads may span devices: offloaded shards (e.g. offload_head) live on CPU
    # while expert/other shards are on GPU. Reduce the per-tensor norms onto the
    # CUDA device so the (NCCL) all-reduce never sees a CPU tensor.
    reduce_device = (
        torch.device(torch.cuda.current_device())
        if torch.cuda.is_available()
        else grads[0].device
    )
    per_tensor_pow = [
        torch.linalg.vector_norm(g, norm_type).to(reduce_device) ** norm_type for g in grads
    ]
    # Global L2 norm: all-reduce the sum of squared local norms (FSDP + pipeline).
    local_norm_pow = torch.stack(per_tensor_pow).sum()
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
