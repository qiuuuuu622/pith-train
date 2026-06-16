"""Context-parallel (CP) path for the Gated DeltaNet token mixer.

Qwen3.5's linear-attention layers are a *recurrent* (chunk-wise gated delta rule)
scan, not softmax attention, so they cannot reuse the zigzag ring-attention kernel.
Their sequence dependency is, however, simpler: only two things cross a CP rank
boundary when the global sequence is split *contiguously* across ``cp_size`` ranks
(rank ``r`` owns tokens ``[r*S/cp : (r+1)*S/cp]``):

1. the depthwise **causal conv1d** (kernel ``K``) needs the last ``K-1`` tokens of
   the left neighbour -- a small *halo* exchange;
2. the **recurrent state** of the delta rule (shape ``[B, H, Dk, Dv]``) must flow
   left-to-right: rank ``r`` starts its scan from the final state produced by rank
   ``r-1``.

This module implements both as differentiable ops. v1 threads the state
*sequentially* (rank ``r`` waits on ``r-1``); the heavy per-segment matmuls are
already done by the FLA / torch chunk kernel, and only the tiny state matrix is
communicated. A parallel prefix scan (O(log cp)) is a later optimisation.

Backward mirrors forward with the direction reversed:
* the conv halo sends the gradient of the borrowed columns back to the left;
* the state scan flows ``dstate`` right-to-left.

For ``cp_size == 1`` every helper is a no-op and reduces to the single-rank path.
"""

from typing import Callable, Optional

import torch
import torch.distributed as dist

# P2P primitives must not be traced by the FLA self-compile / inductor.
_send = torch.compiler.disable(dist.send)
_recv = torch.compiler.disable(dist.recv)


def _neighbor_global_ranks(cp_group: dist.ProcessGroup) -> tuple[int, int, int, int]:
    """Return ``(cp_rank, cp_size, left_global, right_global)``.

    ``left``/``right`` are ``-1`` at the contiguous-sequence boundaries (rank 0 has
    no left neighbour, the last rank has no right neighbour) -- the scan does not
    wrap around the way ring attention does.
    """
    cp_rank = dist.get_rank(cp_group)
    cp_size = dist.get_world_size(cp_group)
    left = dist.get_global_rank(cp_group, cp_rank - 1) if cp_rank > 0 else -1
    right = dist.get_global_rank(cp_group, cp_rank + 1) if cp_rank < cp_size - 1 else -1
    return cp_rank, cp_size, left, right


# ---------------------------------------------------------------------------
# Causal conv1d halo exchange
# ---------------------------------------------------------------------------


class _LeftHalo(torch.autograd.Function):
    """Prepend the left neighbour's last ``halo`` columns to ``x``.

    ``x`` is the conv input laid out as ``[B, C, S_local]`` (channels = conv_dim).
    Forward sends our last ``halo`` columns to the right neighbour (they become its
    halo) and receives ``halo`` columns from the left neighbour (zeros at rank 0),
    returning ``cat([recv, x], dim=-1)`` of width ``S_local + halo``.

    Backward splits the incoming gradient: the first ``halo`` columns belong to the
    left neighbour's tail and are shipped left; the remainder is the local gradient,
    to which we add the gradient of the columns we lent to the right.
    """

    @staticmethod
    def forward(ctx, x: torch.Tensor, halo: int, cp_group: dist.ProcessGroup):
        cp_rank, cp_size, left, right = _neighbor_global_ranks(cp_group)
        ctx.halo = halo
        ctx.cp_group = cp_group
        ctx.left = left
        ctx.right = right

        send_cols = x[..., -halo:].contiguous()  # our tail -> right neighbour's halo
        recv_cols = torch.zeros_like(send_cols)  # left halo; stays zero at rank 0

        # Ordered to avoid deadlock: even ranks send-then-recv, odd recv-then-send.
        if cp_rank % 2 == 0:
            if right >= 0:
                _send(send_cols, right)
            if left >= 0:
                _recv(recv_cols, left)
        else:
            if left >= 0:
                _recv(recv_cols, left)
            if right >= 0:
                _send(send_cols, right)

        return torch.cat([recv_cols, x], dim=-1)

    @staticmethod
    def backward(ctx, grad_out: torch.Tensor):
        halo = ctx.halo
        left, right = ctx.left, ctx.right
        cp_rank = dist.get_rank(ctx.cp_group)

        grad_halo = grad_out[..., :halo].contiguous()  # -> left neighbour's tail
        grad_local = grad_out[..., halo:].contiguous()  # our own columns
        grad_tail = torch.zeros_like(grad_halo)  # grad for cols we lent right

        if cp_rank % 2 == 0:
            if left >= 0:
                _send(grad_halo, left)
            if right >= 0:
                _recv(grad_tail, right)
        else:
            if right >= 0:
                _recv(grad_tail, right)
            if left >= 0:
                _send(grad_halo, left)

        # Add the gradient flowing back for the tail columns we sent right.
        grad_local[..., -halo:] += grad_tail
        return grad_local, None, None


def causal_conv_with_halo(
    conv1d: torch.nn.Conv1d,
    x: torch.Tensor,
    cp_group: Optional[dist.ProcessGroup],
) -> torch.Tensor:
    """Depthwise causal conv over the sequence with cross-rank left context.

    ``x`` is ``[B, C, S_local]``. Returns ``[B, C, S_local]`` matching the
    single-rank causal conv (left-padded by zeros at the global sequence start).
    """
    k_minus_1 = conv1d.kernel_size[0] - 1
    if cp_group is None or dist.get_world_size(cp_group) == 1 or k_minus_1 == 0:
        # Single rank: replicate the original (pad both sides, keep first S).
        return conv1d(x)[..., : x.shape[-1]]
    # Provide real left context, then run the conv with no extra left padding.
    x_haloed = _LeftHalo.apply(x, k_minus_1, cp_group)
    # conv1d keeps padding=k_minus_1; the haloed input already carries the left
    # context, so trim the leading halo and the right padding to S_local.
    out = conv1d(x_haloed)[..., k_minus_1 : k_minus_1 + x.shape[-1]]
    return out


# ---------------------------------------------------------------------------
# Recurrent-state scan across CP ranks
# ---------------------------------------------------------------------------


class _SendFinalState(torch.autograd.Function):
    """Ship this segment's final recurrent state to the right neighbour.

    Forward sends ``state`` (no local consumer) and returns a zero scalar that the
    caller adds to the output so this node stays in the autograd graph. Backward
    receives ``dstate`` from the right neighbour (zeros at the last rank) and returns
    it as the gradient of ``state``, routing it into the chunk kernel's backward.
    """

    @staticmethod
    def forward(ctx, state: torch.Tensor, cp_group: dist.ProcessGroup):
        _, _, _, right = _neighbor_global_ranks(cp_group)
        ctx.cp_group = cp_group
        ctx.right = right
        ctx.state_shape = state.shape
        ctx.state_dtype = state.dtype
        ctx.state_device = state.device
        if right >= 0:
            _send(state.contiguous(), right)
        return state.new_zeros(())

    @staticmethod
    def backward(ctx, _grad_zero: torch.Tensor):
        dstate = torch.zeros(
            ctx.state_shape, dtype=ctx.state_dtype, device=ctx.state_device
        )
        if ctx.right >= 0:
            _recv(dstate, ctx.right)
        return dstate, None


def chunk_gated_delta_rule_cp(
    kernel: Callable,
    query: torch.Tensor,
    key: torch.Tensor,
    value: torch.Tensor,
    g: torch.Tensor,
    beta: torch.Tensor,
    cp_group: Optional[dist.ProcessGroup],
    *,
    state_shape: tuple[int, int, int, int],
    state_dtype: torch.dtype = torch.float32,
    use_qk_l2norm_in_kernel: bool = True,
) -> torch.Tensor:
    """Run the chunk gated delta rule on a contiguous sequence shard.

    ``kernel`` is the chunk delta-rule callable (FLA or the torch fallback) and must
    accept ``initial_state`` / ``output_final_state`` and return
    ``(core_attn_out, final_state)``. ``state_shape`` is ``[B, H, Dk, Dv]`` and
    ``state_dtype`` must match what the kernel produces for ``final_state`` (the
    torch fallback computes in float32; pass that here for cross-rank consistency).

    The incoming state is received from the left neighbour (zeros at rank 0). Its
    gradient is shipped back left by a tensor hook. The outgoing final state is sent
    right by ``_SendFinalState``, whose backward pulls ``dstate`` from the right.
    """
    single = cp_group is None or dist.get_world_size(cp_group) == 1
    if single:
        out, _ = kernel(
            query, key, value, g=g, beta=beta,
            initial_state=None, output_final_state=False,
            use_qk_l2norm_in_kernel=use_qk_l2norm_in_kernel,
        )
        return out

    cp_rank, _, left, _ = _neighbor_global_ranks(cp_group)

    if cp_rank == 0:
        initial_state = None
    else:
        s_in = torch.empty(state_shape, dtype=state_dtype, device=query.device)
        _recv(s_in, left)
        s_in.requires_grad_(True)
        # Backward: ship the gradient of the borrowed initial state back left.
        s_in.register_hook(lambda grad: (_send(grad.contiguous(), left), grad)[1])
        initial_state = s_in

    out, final_state = kernel(
        query, key, value, g=g, beta=beta,
        initial_state=initial_state, output_final_state=True,
        use_qk_l2norm_in_kernel=use_qk_l2norm_in_kernel,
    )

    # Keep _SendFinalState in the graph so its backward (recv dstate) runs.
    coupling = _SendFinalState.apply(final_state.to(state_dtype), cp_group)
    return out + coupling
