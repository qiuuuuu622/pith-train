"""Context-parallel full attention for the contiguous shard layout (Increment 3).

The 10 full_attention layers in Qwen3.5 MoE need cross-rank attention when
cp_size > 1.  Under the contiguous layout (rank r owns tokens
[r*S/cp : (r+1)*S/cp]) the correct approach for a causal mask is:

  * All-gather K and V across all CP ranks so every rank has K/V for tokens
    [0 : (r+1)*S/cp].
  * Run flash_attn_func(causal=True) with the local Q (the sequence suffix)
    against the gathered K/V prefix.  Flash attention's bottom-right causal
    alignment gives the correct mask: Q token i (global position r*Sl + i)
    only attends to K/V tokens 0..(r*Sl+i).

Crucially rank r needs only the K/V prefix [0..(r+1)*Sl], NOT the full
gathered K/V [0..cp_size*Sl].  We implement this via an all-gather followed
by a trim: each rank gathers all K/V shards and then keeps only the prefix it
needs.  The backward reduces gradients back to each rank's own shard.

RoPE: callers must pass ``position_embeddings`` computed for the *global*
positions [r*S/cp : (r+1)*S/cp], not local positions [0 : S/cp].  The model-
level wiring (Increment 3b counterpart) is responsible for providing the
correct cos/sin slices; this file only handles the attention communication.

When cp_group is None or size 1 the function degenerates to a plain causal
flash_attn_func call so the single-rank path is unchanged.
"""

from typing import Optional

import torch
import torch.distributed as dist

from pithtrain.operators.flash_attn_v4 import flash_attn_func


class _PrefixGatherKV(torch.autograd.Function):
    """All-gather K (or V) across CP ranks and trim to this rank's causal prefix.

    Forward:
        1. all_gather x across cp_group -> gathered [B, cp_size*Sl, HK, D]
        2. Trim to prefix [0..(cp_rank+1)*Sl] for causal correctness.
           Rank r's Q (global positions [r*Sl..(r+1)*Sl)) can attend to
           K/V [0..(r+1)*Sl).

    Backward:
        The gradient for this rank's own K shard is grad_out[:, cp_rank*Sl:(cp_rank+1)*Sl].
        Gradients for K shards owned by other ranks are summed back via
        reduce-scatter: each rank sends d_k for [0..r*Sl] back to the owners.
    """

    @staticmethod
    def forward(ctx, x: torch.Tensor, cp_group: dist.ProcessGroup) -> torch.Tensor:
        cp_rank = dist.get_rank(cp_group)
        cp_size = dist.get_world_size(cp_group)
        ctx.cp_rank = cp_rank
        ctx.cp_size = cp_size
        ctx.cp_group = cp_group
        ctx.shard_len = x.shape[1]

        # All-gather all shards.
        gathered = [torch.empty_like(x) for _ in range(cp_size)]
        dist.all_gather(gathered, x.contiguous(), group=cp_group)
        # Each rank only needs the prefix up to its own shard (inclusive).
        prefix = torch.cat(gathered[: cp_rank + 1], dim=1)
        return prefix

    @staticmethod
    def backward(ctx, grad_prefix: torch.Tensor) -> tuple:
        cp_rank = ctx.cp_rank
        cp_size = ctx.cp_size
        cp_group = ctx.cp_group
        shard = ctx.shard_len
        B, _, HK, D = grad_prefix.shape

        # grad_prefix has shape [B, (cp_rank+1)*Sl, HK, D].
        # We need to distribute gradients back to ranks 0..cp_rank.
        # Strategy: pad grad_prefix to full length [B, cp_size*Sl, HK, D],
        # all-reduce across ALL ranks (symmetric call — no deadlock), then
        # each rank picks its own shard from the result.
        grad_full = torch.zeros(
            B, cp_size * shard, HK, D,
            dtype=grad_prefix.dtype, device=grad_prefix.device,
        )
        grad_full[:, : (cp_rank + 1) * shard] = grad_prefix
        dist.all_reduce(grad_full, group=cp_group)
        grad_local = grad_full[:, cp_rank * shard : (cp_rank + 1) * shard].contiguous()
        return grad_local, None


def full_attn_cp(
    query: torch.Tensor,
    key: torch.Tensor,
    value: torch.Tensor,
    softmax_scale: float,
    cp_group: Optional[dist.ProcessGroup],
) -> torch.Tensor:
    """Causal full attention over a contiguous CP shard.

    Args:
        query: [B, S_local, HQ, D] — this rank's Q shard (global pos r*Sl..)
        key:   [B, S_local, HK, D] — this rank's K shard
        value: [B, S_local, HK, D] — this rank's V shard
        softmax_scale: 1/sqrt(D)
        cp_group: CP process group (None / size-1 -> plain causal attention)

    Returns:
        [B, S_local, HQ, D] attention output for this rank's tokens.
    """
    if cp_group is None or dist.get_world_size(cp_group) == 1:
        return flash_attn_func(query, key, value, softmax_scale, causal=True)

    # Each rank gathers K/V up to its own prefix [0..(r+1)*Sl].
    key_prefix = _PrefixGatherKV.apply(key, cp_group)
    value_prefix = _PrefixGatherKV.apply(value, cp_group)

    # Flash attention with causal=True on the prefix K/V.
    # FA4's bottom-right causal alignment: Q_i (local index i, global r*Sl+i)
    # attends to K[0..len_K-len_Q+i] = K[0..r*Sl+i].  Correct.
    return flash_attn_func(query, key_prefix, value_prefix, softmax_scale, causal=True)
