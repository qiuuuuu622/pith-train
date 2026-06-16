"""Context-parallel (CP) path for the Gated DeltaNet token mixer.

This follows the design used by Megatron-LM for Mamba/SSM-family layers
(``megatron/core/ssm/mamba_context_parallel.py``): **all-to-all head
parallelism**, *not* a contiguous-shard sequential state scan.

Under CP the global sequence is sharded across ``cp_size`` ranks in the
*zigzag* (load-balanced) layout that the attention ring uses -- rank ``r`` owns
chunk ``r`` and chunk ``2*cp_size-r-1`` of ``2*cp_size`` equal chunks.  The
linear-attention layer cannot consume that layout directly: its causal conv and
recurrent scan are order-sensitive.  So per layer we:

1. ``all_to_all_cp2hp``: turn the *sequence-sharded, all-heads* activations into
   *full-sequence, head-sharded* activations (each rank keeps ``1/cp`` of the
   heads but the whole sequence);
2. ``undo_zigzag``: restore natural token order on the now-full sequence;
3. run the conv + chunk delta-rule scan locally on the full sequence for this
   rank's head slice -- **no halo, no cross-rank state passing**;
4. ``redo_zigzag`` + ``all_to_all_hp2cp``: go back to the sequence-sharded,
   all-heads layout for the residual / out-projection.

Versus the sequential state-scan alternative, this trades a little extra
bandwidth (two all-to-alls of the activations) for *no* serialization across CP
ranks: every rank runs an independent full-sequence scan, perfectly balanced.

For ``cp_size == 1`` every helper is a no-op and the path reduces to the plain
single-rank forward.
"""

from typing import Callable, Optional

import torch
import torch.distributed as dist
import torch.nn.functional as F

# All-to-all must not be traced by the FLA self-compile / inductor.
_all_to_all_single = torch.compiler.disable(dist.all_to_all_single)


# ---------------------------------------------------------------------------
# Zigzag <-> natural sequence reordering (along dim=1)
# ---------------------------------------------------------------------------
# After ``all_to_all_cp2hp`` the full sequence is the concatenation of every
# rank's zigzag-local shard, in rank order:
#     [r0.front, r0.back, r1.front, r1.back, ...]
#   = [chunk0,   chunk2c-1, chunk1,  chunk2c-2, ...]
# where ``front`` of rank r is global chunk ``r`` and ``back`` is global chunk
# ``2*cp-1-r``.  ``undo_zigzag`` permutes those ``2*cp`` chunks back to natural
# order ``[chunk0, chunk1, ..., chunk2c-1]``; ``redo_zigzag`` is the inverse.


def undo_zigzag(x: torch.Tensor, cp_size: int, dim: int = 1) -> torch.Tensor:
    """Gathered-zigzag chunk order -> natural token order."""
    if cp_size == 1:
        return x
    gathered = list(torch.chunk(x, 2 * cp_size, dim=dim))
    natural = [None] * (2 * cp_size)
    for g in range(2 * cp_size):
        r = g // 2
        c = r if g % 2 == 0 else (2 * cp_size - 1 - r)
        natural[c] = gathered[g]
    return torch.cat(natural, dim=dim)


def redo_zigzag(x: torch.Tensor, cp_size: int, dim: int = 1) -> torch.Tensor:
    """Natural token order -> gathered-zigzag chunk order (inverse of undo)."""
    if cp_size == 1:
        return x
    natural = list(torch.chunk(x, 2 * cp_size, dim=dim))
    gathered = []
    for g in range(2 * cp_size):
        r = g // 2
        c = r if g % 2 == 0 else (2 * cp_size - 1 - r)
        gathered.append(natural[c])
    return torch.cat(gathered, dim=dim)


# ---------------------------------------------------------------------------
# All-to-all: sequence-parallel <-> head-parallel
# ---------------------------------------------------------------------------
# Input layout is ``[B, S, C]`` (sequence dim 1, channel/head dim 2).  cp2hp
# turns ``[B, S/cp, C]`` (this rank's seq shard, all channels) into
# ``[B, S, C/cp]`` (full sequence, this rank's channel slice).  hp2cp is the
# inverse.  ``C`` must be divisible by ``cp_size`` and laid out head-major so a
# contiguous ``C/cp`` slice corresponds to a contiguous group of heads.


def _raw_cp2hp(x: torch.Tensor, group: dist.ProcessGroup, cp: int) -> torch.Tensor:
    B, Sl, C = x.shape
    Cc = C // cp
    # split channels into cp groups; group h is destined for rank h
    x = x.view(B, Sl, cp, Cc).permute(2, 0, 1, 3).contiguous()  # [cp(group), B, Sl, Cc]
    out = torch.empty_like(x)
    _all_to_all_single(out, x, group=group)  # out[s] = source s's seq shard for our group
    return out.permute(1, 0, 2, 3).reshape(B, cp * Sl, Cc)  # [B, S(source-major), Cc]


def _raw_hp2cp(x: torch.Tensor, group: dist.ProcessGroup, cp: int) -> torch.Tensor:
    B, S, Cc = x.shape
    Sl = S // cp
    x = x.view(B, cp, Sl, Cc).permute(1, 0, 2, 3).contiguous()  # [cp(source), B, Sl, Cc]
    out = torch.empty_like(x)
    _all_to_all_single(out, x, group=group)  # out[h] = channel group h gathered over seq
    return out.permute(1, 2, 0, 3).reshape(B, Sl, cp * Cc)  # [B, Sl, C]


class _AllToAllCP2HP(torch.autograd.Function):
    @staticmethod
    def forward(ctx, x, group, cp):
        ctx.group, ctx.cp = group, cp
        return _raw_cp2hp(x, group, cp)

    @staticmethod
    def backward(ctx, g):
        return _raw_hp2cp(g.contiguous(), ctx.group, ctx.cp), None, None


class _AllToAllHP2CP(torch.autograd.Function):
    @staticmethod
    def forward(ctx, x, group, cp):
        ctx.group, ctx.cp = group, cp
        return _raw_hp2cp(x, group, cp)

    @staticmethod
    def backward(ctx, g):
        return _raw_cp2hp(g.contiguous(), ctx.group, ctx.cp), None, None


def all_to_all_cp2hp(x: torch.Tensor, cp_group: dist.ProcessGroup) -> torch.Tensor:
    """[B, S/cp, C] (seq-sharded) -> [B, S, C/cp] (head-sharded). Differentiable."""
    cp = dist.get_world_size(cp_group)
    if cp == 1:
        return x
    return _AllToAllCP2HP.apply(x.contiguous(), cp_group, cp)


def all_to_all_hp2cp(x: torch.Tensor, cp_group: dist.ProcessGroup) -> torch.Tensor:
    """[B, S, C/cp] (head-sharded) -> [B, S/cp, C] (seq-sharded). Differentiable."""
    cp = dist.get_world_size(cp_group)
    if cp == 1:
        return x
    return _AllToAllHP2CP.apply(x.contiguous(), cp_group, cp)


# ---------------------------------------------------------------------------
# Per-rank parameter slices (head-parallel)
# ---------------------------------------------------------------------------


def _conv_channel_index(
    cp_rank: int, cp_size: int, key_dim: int, value_dim: int, device: torch.device
) -> torch.Tensor:
    """Channel indices of the depthwise conv weight owned by this CP rank.

    The conv operates on ``mixed_qkv = [q (key_dim) | k (key_dim) | v (value_dim)]``.
    cp2hp keeps channel group ``cp_rank`` of each component, i.e. a contiguous
    ``dim/cp`` slice within each of the q/k/v blocks.
    """
    kc, vc = key_dim // cp_size, value_dim // cp_size
    q = torch.arange(cp_rank * kc, (cp_rank + 1) * kc, device=device)
    k = key_dim + q
    v = 2 * key_dim + torch.arange(cp_rank * vc, (cp_rank + 1) * vc, device=device)
    return torch.cat([q, k, v])


# ---------------------------------------------------------------------------
# Head-parallel Gated DeltaNet forward
# ---------------------------------------------------------------------------


def head_parallel_gated_delta_net(
    mixed_qkv: torch.Tensor,  # [B, S_local, conv_dim]  (post in_proj, pre-conv)
    z: torch.Tensor,  # [B, S_local, value_dim]
    b: torch.Tensor,  # [B, S_local, num_v_heads]
    a: torch.Tensor,  # [B, S_local, num_v_heads]
    *,
    conv_weight: torch.Tensor,  # [conv_dim, 1, K]  (depthwise, no bias)
    A_log: torch.Tensor,  # [num_v_heads]
    dt_bias: torch.Tensor,  # [num_v_heads]
    norm: Callable,  # Qwen3_5MoeRMSNormGated
    kernel: Callable,  # chunk gated delta-rule (FLA or torch fallback)
    key_dim: int,
    value_dim: int,
    num_k_heads: int,
    num_v_heads: int,
    head_k_dim: int,
    head_v_dim: int,
    cp_group: Optional[dist.ProcessGroup],
    use_qk_l2norm_in_kernel: bool = True,
) -> torch.Tensor:
    """Run conv + chunk delta-rule under all-to-all head parallelism.

    Returns ``core_attn_out`` shaped ``[B, S_local, value_dim]`` (sequence-sharded,
    zigzag layout) ready for the residual out-projection.  For ``cp_group`` None or
    size 1 this is the plain single-rank forward.
    """
    cp_size = dist.get_world_size(cp_group) if cp_group is not None else 1
    cp_rank = dist.get_rank(cp_group) if cp_group is not None else 0
    batch_size = mixed_qkv.shape[0]

    if cp_size > 1:
        # 1) seq-sharded -> head-sharded full sequence (split q/k/v separately so
        #    head boundaries stay intact), then 2) restore natural token order.
        q, k, v = torch.split(mixed_qkv, [key_dim, key_dim, value_dim], dim=-1)
        q = undo_zigzag(all_to_all_cp2hp(q, cp_group), cp_size)
        k = undo_zigzag(all_to_all_cp2hp(k, cp_group), cp_size)
        v = undo_zigzag(all_to_all_cp2hp(v, cp_group), cp_size)
        mixed_qkv = torch.cat([q, k, v], dim=-1)
        z = undo_zigzag(all_to_all_cp2hp(z, cp_group), cp_size)
        b = undo_zigzag(all_to_all_cp2hp(b, cp_group), cp_size)
        a = undo_zigzag(all_to_all_cp2hp(a, cp_group), cp_size)

        idx = _conv_channel_index(cp_rank, cp_size, key_dim, value_dim, mixed_qkv.device)
        conv_weight = conv_weight.index_select(0, idx)
        A_log = A_log[cp_rank * (num_v_heads // cp_size) : (cp_rank + 1) * (num_v_heads // cp_size)]
        dt_bias = dt_bias[
            cp_rank * (num_v_heads // cp_size) : (cp_rank + 1) * (num_v_heads // cp_size)
        ]
        nk = num_k_heads // cp_size
        nv = num_v_heads // cp_size
        kdim = key_dim // cp_size
        vdim = value_dim // cp_size
    else:
        nk, nv, kdim, vdim = num_k_heads, num_v_heads, key_dim, value_dim

    seq_len = mixed_qkv.shape[1]
    k_minus_1 = conv_weight.shape[-1] - 1
    conv_dim_local = mixed_qkv.shape[-1]

    # Depthwise causal conv over the (now full, natural-order) sequence.
    x = mixed_qkv.transpose(1, 2)  # [B, conv_dim_local, S]
    x = F.conv1d(x, conv_weight, bias=None, padding=k_minus_1, groups=conv_dim_local)
    x = x[..., :seq_len]
    x = F.silu(x).transpose(1, 2)  # [B, S, conv_dim_local]

    query, key, value = torch.split(x, [kdim, kdim, vdim], dim=-1)
    query = query.reshape(batch_size, seq_len, nk, head_k_dim)
    key = key.reshape(batch_size, seq_len, nk, head_k_dim)
    value = value.reshape(batch_size, seq_len, nv, head_v_dim)

    beta = b.sigmoid()
    # float32: A_log.exp() can overflow in bf16/fp16.
    g = -A_log.float().exp() * F.softplus(a.float() + dt_bias)
    if nv // nk > 1:
        query = query.repeat_interleave(nv // nk, dim=2)
        key = key.repeat_interleave(nv // nk, dim=2)

    core_attn_out, _ = kernel(
        query,
        key,
        value,
        g=g,
        beta=beta,
        initial_state=None,
        output_final_state=False,
        use_qk_l2norm_in_kernel=use_qk_l2norm_in_kernel,
    )

    core_attn_out = core_attn_out.reshape(-1, head_v_dim)
    z_flat = z.reshape(-1, head_v_dim)
    core_attn_out = norm(core_attn_out, z_flat)
    core_attn_out = core_attn_out.reshape(batch_size, seq_len, nv * head_v_dim)

    if cp_size > 1:
        # head-sharded full sequence -> seq-sharded all-heads (zigzag) for the residual.
        core_attn_out = all_to_all_hp2cp(redo_zigzag(core_attn_out, cp_size), cp_group)

    return core_attn_out
