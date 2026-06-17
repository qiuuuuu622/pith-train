"""Fused linear + cross-entropy that never materializes the full logit matrix.

The language-model head projects the hidden state ``[N, H]`` to vocab logits
``[N, V]`` and then takes a cross-entropy loss. For a large vocabulary (Qwen3.5
has ``V = 248320``) the ``[N, V]`` logit tensor dominates activation memory on
the pipeline ranks that own the head, and -- because it is saved for backward --
it stays resident through the backward pass, once per in-flight micro-batch.

This op fuses the projection and the loss and tiles the work over the token
dimension ``N``:

* **forward** tiles only to accumulate the scalar loss, then saves just the
  inputs (``hidden`` ``[N, H]``, ``weight`` ``[V, H]``, ``target`` ``[N]``).
  No ``[N, V]`` tensor survives the call.
* **backward** recomputes each ``[chunk, V]`` logit tile, derives the gradient
  contributions, and accumulates them into ``grad_hidden`` ``[N, H]`` and a
  single ``grad_weight`` ``[V, H]`` buffer.

Peak extra memory is therefore ``O(chunk * V)`` (one tile) plus the unavoidable
``grad_weight`` ``[V, H]``, instead of ``O(N * V)`` held across the whole
backward per in-flight micro-batch. The matmuls run in the weight dtype (bf16);
only the per-tile ``[chunk, V]`` logits are promoted to fp32 for a numerically
stable softmax, so the full weight is never cast to fp32.

NOTE: the head weight is captured by ``save_for_backward`` and re-read in
backward, so the owning FSDP module must keep it unsharded across the backward
(``reshard_after_forward=False``); otherwise the saved tensor is stale.

The math matches mean-reduced cross-entropy with ``ignore_index`` exactly, so it
is a drop-in for ``lm_head`` followed by
:func:`pithtrain.operators.cross_entropy.cross_entropy`.
"""

import torch


def _ceil_div(a: int, b: int) -> int:
    return (a + b - 1) // b


class FusedLinearCrossEntropy(torch.autograd.Function):
    @staticmethod
    def forward(ctx, hidden, weight, target, ignore_index=-100, num_chunks=8):
        N, _ = hidden.shape
        target = target.reshape(-1)
        assert target.shape[0] == N, (target.shape, N)
        n_nonignore = (target != ignore_index).sum().clamp(min=1).to(torch.float32)

        loss = torch.zeros((), dtype=torch.float32, device=hidden.device)
        chunk = _ceil_div(N, num_chunks)
        for s in range(0, N, chunk):
            e = min(s + chunk, N)
            t = target[s:e]
            valid = t != ignore_index
            # bf16 matmul, fp32 only for the [chunk, V] logits (stable softmax).
            logits = (hidden[s:e] @ weight.t()).float()
            m = logits.max(dim=-1, keepdim=True).values
            denom = torch.exp(logits - m).sum(dim=-1, keepdim=True)
            true_logit = logits.gather(-1, t.clamp(min=0)[:, None])
            nll = (m + denom.log()) - true_logit
            loss += torch.where(valid[:, None], nll, torch.zeros_like(nll)).sum()
        loss = loss / n_nonignore

        ctx.save_for_backward(hidden, weight, target)
        ctx.ignore_index = ignore_index
        ctx.num_chunks = num_chunks
        ctx.n_nonignore = n_nonignore
        return loss

    @staticmethod
    def backward(ctx, grad_output):
        hidden, weight, target = ctx.saved_tensors
        n_nonignore = ctx.n_nonignore
        N, _ = hidden.shape

        grad_hidden = torch.empty_like(hidden)
        need_w = ctx.needs_input_grad[1]
        # fp32 accumulator for the weight grad (exists only during backward).
        grad_weight = torch.zeros_like(weight, dtype=torch.float32) if need_w else None
        scale = grad_output / n_nonignore  # fold the (scalar) cotangent + mean here

        chunk = _ceil_div(N, ctx.num_chunks)
        for s in range(0, N, chunk):
            e = min(s + chunk, N)
            t = target[s:e]
            valid = t != ctx.ignore_index
            t_safe = t.clamp(min=0)

            logits = (hidden[s:e] @ weight.t()).float()  # [c, V]
            m = logits.max(dim=-1, keepdim=True).values
            prob = torch.exp(logits - m)
            prob = prob / prob.sum(dim=-1, keepdim=True)  # softmax, [c, V]
            prob = torch.where(valid[:, None], prob, torch.zeros_like(prob))
            prob.scatter_add_(
                -1, t_safe[:, None], torch.where(valid, -1.0, 0.0)[:, None]
            )
            grad_logits = (prob * scale).to(weight.dtype)  # [c, V]

            grad_hidden[s:e] = grad_logits @ weight  # [c, H]
            if need_w:
                # In-place fp32 accumulate into grad_weight. addmm_ writes the
                # [V, H] product straight into the accumulator -- no per-chunk
                # [V, H] temporary -- and the matmul runs in fp32 (more accurate
                # than the previous bf16 matmul + .float()).
                grad_weight.addmm_(grad_logits.float().t(), hidden[s:e].float())

        gw = grad_weight.to(weight.dtype) if need_w else None
        return grad_hidden, gw, None, None, None


def fused_linear_cross_entropy(
    hidden: torch.Tensor,
    weight: torch.Tensor,
    target: torch.Tensor,
    ignore_index: int = -100,
    num_chunks: int = 8,
) -> torch.Tensor:
    """Mean-reduced cross-entropy of ``hidden @ weight.T`` against ``target``.

    Equivalent to ``F.linear(hidden, weight)`` followed by mean cross-entropy
    with ``ignore_index``, but never materializes the full ``[N, V]`` logits.

    Parameters
    ----------
    hidden : ``[N, H]`` head input (typically the post-norm hidden state).
    weight : ``[V, H]`` head weight (no bias).
    target : ``[N]`` class indices; rows equal to ``ignore_index`` are skipped.
    num_chunks : number of token-dimension tiles; higher -> less peak memory.
    """
    return FusedLinearCrossEntropy.apply(hidden, weight, target, ignore_index, num_chunks)
