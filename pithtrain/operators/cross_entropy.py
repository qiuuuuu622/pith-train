"""In-place fused cross-entropy (mean reduction).

Computes the per-token cross-entropy of ``[N, V]`` logits against ``[N]``
targets and -- to avoid a second ``[N, V]`` activation -- overwrites the logit
tensor in place with the loss gradient during the forward pass. Backward only
scales that stored gradient by the incoming cotangent.

History: a previous Triton implementation produced a numerically wrong gradient
once enough rows ran concurrently (the loss was correct, but the gradient was
nearly orthogonal to the true gradient -- a silent training-correctness bug with
no regression test). This pure-PyTorch version computes the standard
``(softmax - onehot) / n_nonignore`` gradient in fp32, tiled over the token
dimension so peak extra memory stays ``O(chunk * V)``. It matches
``F.cross_entropy(..., reduction="mean")`` exactly and is covered by
``tests/test_cross_entropy.py``.
"""

import torch


def _ceil_div(a: int, b: int) -> int:
    return (a + b - 1) // b


class CrossEntropy(torch.autograd.Function):
    """Mean cross-entropy that overwrites the logits with their gradient."""

    @staticmethod
    def forward(ctx, inp, target, ignore_index, num_chunks=8):
        N, _ = inp.shape
        target = target.reshape(-1)
        assert target.shape[0] == N, (target.shape, N)
        n_non_ignore = (target != ignore_index).sum().clamp(min=1).to(torch.float32)

        loss = torch.zeros((), dtype=torch.float32, device=inp.device)
        chunk = _ceil_div(N, num_chunks)
        for s in range(0, N, chunk):
            e = min(s + chunk, N)
            t = target[s:e]
            valid = t != ignore_index
            t_safe = t.clamp(min=0)

            logits = inp[s:e].float()  # [c, V]
            m = logits.max(dim=-1, keepdim=True).values
            exp = torch.exp(logits - m)
            denom = exp.sum(dim=-1, keepdim=True)  # [c, 1]
            true_logit = logits.gather(-1, t_safe[:, None])  # [c, 1]
            nll = (m + denom.log()) - true_logit
            loss += torch.where(valid[:, None], nll, torch.zeros_like(nll)).sum()

            grad = exp / denom  # softmax, [c, V]
            grad = torch.where(valid[:, None], grad, torch.zeros_like(grad))
            grad.scatter_add_(-1, t_safe[:, None], torch.where(valid, -1.0, 0.0)[:, None])
            # Store (softmax - onehot) / n_nonignore back into the logits in place.
            inp[s:e] = (grad / n_non_ignore).to(inp.dtype)

        loss = loss / n_non_ignore
        ctx.save_for_backward(inp.detach())
        return loss

    @staticmethod
    def backward(ctx, grad_output):
        (inp,) = ctx.saved_tensors
        inp.mul_(grad_output.to(inp.dtype))
        return inp, None, None, None


def cross_entropy(
    inp: torch.Tensor,
    target: torch.Tensor,
    ignore_index: int = -100,
    num_chunks: int = 8,
) -> torch.Tensor:
    """
    In-place fused cross-entropy loss with mean reduction.

    Overwrites ``inp`` with pre-computed gradients during the forward pass,
    eliminating the need for a separate activation tensor. All arithmetic is in
    FP32; gradients are stored in the original dtype of ``inp``. Equivalent to
    ``F.cross_entropy(inp.float(), target, ignore_index=ignore_index,
    reduction="mean")``.

    Parameters
    ----------
    inp : torch.Tensor
        Logits of shape ``(N, V)``. Modified in place.
    target : torch.Tensor
        Target indices of shape ``(N,)``; rows equal to ``ignore_index`` are
        ignored in both loss and gradient.
    ignore_index : int
        Target value to ignore.
    num_chunks : int
        Token-dimension tiles; higher -> less peak memory.
    """
    return CrossEntropy.apply(inp, target, ignore_index, num_chunks)
