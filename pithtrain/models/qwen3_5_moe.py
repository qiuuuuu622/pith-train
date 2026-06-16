"""Qwen/Qwen3.5-35B-A3B (text model only).

Hybrid token mixer: 3 of every 4 layers use Gated DeltaNet linear attention,
every 4th layer uses gated full attention (see config ``layer_types``).
The vision tower / multimodal wrapper of the HF release is out of scope —
this mirrors HF's ``Qwen3_5MoeTextModel``.
"""

from dataclasses import fields
from typing import List, Optional, Tuple

import torch
import torch.distributed as dist
import torch.nn.functional as F
from torch import nn

from pithtrain.dualpipe.execution import EpilogArgs, IntermediateTensors, PrologArgs, PrologOuts
from pithtrain.dualpipe.layer_partition import layer_partition
from pithtrain.dualpipe.modeling import decoder_layer_backward, decoder_layer_forward
from pithtrain.dualpipe.utils import FP8WeightCacheControl, run_backward
from pithtrain.layers.deepgemm_fp8_linear import FP8GroupLinearFunc
from pithtrain.layers.factory import ModelImplMode, get_linear_cls
from pithtrain.layers.group_linear import GroupLinearFunc
from pithtrain.models.interface import ForwardAttnOutput
from pithtrain.modules.load_balance import MoELoadBalanceLossInjector, MoELoadBalanceLossTracker
from pithtrain.operators.deepgemm_fp8_quantize import fused_blockwise_transpose_cast_to_fp8_batched
from pithtrain.operators.ep_dispatch import moe_ep_prepare_dispatch
from pithtrain.operators.flash_attn_v4 import flash_attn_func
from pithtrain.operators.gated_delta_rule_cp import head_parallel_gated_delta_net
from pithtrain.operators.ring_attention import ring_attention_func
from pithtrain.operators.silu_mul import silu_mul
from pithtrain.operators.token_scatter import (
    padded_index_gather,
    precompute_group_indices,
    scatter_for_grouped_gemm,
)

torch._dynamo.allow_in_graph(MoELoadBalanceLossInjector)

try:
    from fla.ops.gated_delta_rule import chunk_gated_delta_rule
except ImportError:
    chunk_gated_delta_rule = None


# ---------------------------------------------------------------------------
# Norms
# ---------------------------------------------------------------------------


class Qwen3_5MoeRMSNorm(nn.Module):
    """Zero-centered RMSNorm: ``x_norm * (1 + weight)`` with weight init at 0."""

    def __init__(self, dim: int, eps: float = 1e-6):
        super().__init__()
        self.eps = eps
        self.weight = nn.Parameter(torch.zeros(dim))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        output = x.float() * torch.rsqrt(x.float().pow(2).mean(-1, keepdim=True) + self.eps)
        output = output * (1.0 + self.weight.float())
        return output.type_as(x)


class Qwen3_5MoeRMSNormGated(nn.Module):
    """RMSNorm followed by a SiLU gate (norm before gate). Weight init at 1."""

    def __init__(self, hidden_size: int, eps: float = 1e-6):
        super().__init__()
        self.weight = nn.Parameter(torch.ones(hidden_size))
        self.variance_epsilon = eps

    def forward(self, hidden_states: torch.Tensor, gate: torch.Tensor) -> torch.Tensor:
        input_dtype = hidden_states.dtype
        hidden_states = hidden_states.to(torch.float32)
        variance = hidden_states.pow(2).mean(-1, keepdim=True)
        hidden_states = hidden_states * torch.rsqrt(variance + self.variance_epsilon)
        hidden_states = self.weight * hidden_states.to(input_dtype)
        hidden_states = hidden_states * F.silu(gate.to(torch.float32))
        return hidden_states.to(input_dtype)


# ---------------------------------------------------------------------------
# Rotary embedding (partial RoPE; text-only degenerate interleaved mRoPE)
# ---------------------------------------------------------------------------


class Qwen3_5MoeTextRotaryEmbedding(nn.Module):
    """Partial-rotary RoPE.

    HF applies interleaved mRoPE over (T, H, W) position grids; with text-only
    inputs all three grids carry identical position ids, so the interleave is
    the identity and this reduces to standard RoPE over the rotary fraction of
    the head dim.
    """

    def __init__(
        self,
        rotary_dim: int,
        max_position_embeddings: int = 262144,
        base: float = 10000000.0,
        device: Optional[torch.device] = None,
    ):
        super().__init__()
        self.dim = rotary_dim
        self.max_position_embeddings = max_position_embeddings
        self.base = base

        inv_freq = 1.0 / (
            self.base
            ** (torch.arange(0, self.dim, 2, dtype=torch.float32, device=device) / self.dim)
        )
        self.register_buffer("inv_freq", inv_freq, persistent=False)
        self._set_cos_sin_cache(
            seq_len=max_position_embeddings,
            device=device,
            dtype=torch.get_default_dtype(),
        )

    def _set_cos_sin_cache(self, seq_len: int, device: Optional[torch.device], dtype: torch.dtype):
        self.max_seq_len_cached = seq_len
        t = torch.arange(seq_len, device=device, dtype=torch.float32)
        freqs = torch.outer(t, self.inv_freq)
        emb = torch.cat((freqs, freqs), dim=-1)
        self.register_buffer("cos_cached", emb.cos().to(dtype), persistent=False)
        self.register_buffer("sin_cached", emb.sin().to(dtype), persistent=False)

    def forward(self, x: torch.Tensor, seq_len: int) -> Tuple[torch.Tensor, torch.Tensor]:
        if seq_len > self.max_seq_len_cached:
            self._set_cos_sin_cache(seq_len, x.device, x.dtype)
        return (
            self.cos_cached[:seq_len].to(dtype=x.dtype),
            self.sin_cached[:seq_len].to(dtype=x.dtype),
        )


def rotate_half(x: torch.Tensor) -> torch.Tensor:
    """Rotates half the hidden dims of the input."""
    x1 = x[..., : x.shape[-1] // 2]
    x2 = x[..., x.shape[-1] // 2 :]
    return torch.cat((-x2, x1), dim=-1)


def apply_rotary_pos_emb(
    q: torch.Tensor,
    k: torch.Tensor,
    cos: torch.Tensor,
    sin: torch.Tensor,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """Partial rotary: rotate the first ``cos.shape[-1]`` dims, pass the rest."""
    cos = cos.unsqueeze(2)
    sin = sin.unsqueeze(2)

    rotary_dim = cos.shape[-1]
    q_rot, q_pass = q[..., :rotary_dim], q[..., rotary_dim:]
    k_rot, k_pass = k[..., :rotary_dim], k[..., rotary_dim:]

    q_embed = (q_rot * cos) + (rotate_half(q_rot) * sin)
    k_embed = (k_rot * cos) + (rotate_half(k_rot) * sin)

    q_embed = torch.cat([q_embed, q_pass], dim=-1)
    k_embed = torch.cat([k_embed, k_pass], dim=-1)
    return q_embed, k_embed


# ---------------------------------------------------------------------------
# Gated DeltaNet (linear attention token mixer)
# ---------------------------------------------------------------------------


def l2norm(x: torch.Tensor, dim: int = -1, eps: float = 1e-6) -> torch.Tensor:
    """Aligned with the l2norm implementation in the FLA library."""
    inv_norm = torch.rsqrt((x * x).sum(dim=dim, keepdim=True) + eps)
    return x * inv_norm


def torch_chunk_gated_delta_rule(
    query: torch.Tensor,
    key: torch.Tensor,
    value: torch.Tensor,
    g: torch.Tensor,
    beta: torch.Tensor,
    chunk_size: int = 64,
    use_qk_l2norm_in_kernel: bool = False,
    initial_state: Optional[torch.Tensor] = None,
    output_final_state: bool = False,
) -> Tuple[torch.Tensor, Optional[torch.Tensor]]:
    """Chunk-wise gated delta rule, mirroring HF's torch fallback (no cache).

    Returns ``(core_attn_out, final_state)``. ``final_state`` is the recurrent state
    after the last chunk (float32, shape ``[B, H, Dk, Dv]``) when
    ``output_final_state`` is set, else ``None``. ``initial_state`` seeds the scan;
    both are the cross-rank carry used by the context-parallel path.
    """
    initial_dtype = query.dtype
    if use_qk_l2norm_in_kernel:
        query = l2norm(query, dim=-1, eps=1e-6)
        key = l2norm(key, dim=-1, eps=1e-6)
    query, key, value, beta, g = [
        x.transpose(1, 2).contiguous().to(torch.float32) for x in (query, key, value, beta, g)
    ]

    batch_size, num_heads, sequence_length, k_head_dim = key.shape
    v_head_dim = value.shape[-1]
    pad_size = (chunk_size - sequence_length % chunk_size) % chunk_size
    query = F.pad(query, (0, 0, 0, pad_size))
    key = F.pad(key, (0, 0, 0, pad_size))
    value = F.pad(value, (0, 0, 0, pad_size))
    beta = F.pad(beta, (0, pad_size))
    g = F.pad(g, (0, pad_size))
    total_sequence_length = sequence_length + pad_size
    scale = 1 / (query.shape[-1] ** 0.5)
    query = query * scale

    v_beta = value * beta.unsqueeze(-1)
    k_beta = key * beta.unsqueeze(-1)
    query, key, value, k_beta, v_beta = [
        x.reshape(x.shape[0], x.shape[1], -1, chunk_size, x.shape[-1])
        for x in (query, key, value, k_beta, v_beta)
    ]
    g = g.reshape(g.shape[0], g.shape[1], -1, chunk_size)
    mask = torch.triu(
        torch.ones(chunk_size, chunk_size, dtype=torch.bool, device=query.device), diagonal=0
    )

    g = g.cumsum(dim=-1)
    decay_mask = ((g.unsqueeze(-1) - g.unsqueeze(-2)).tril().exp().float()).tril()
    attn = -((k_beta @ key.transpose(-1, -2)) * decay_mask).masked_fill(mask, 0)
    for i in range(1, chunk_size):
        row = attn[..., i, :i].clone()
        sub = attn[..., :i, :i].clone()
        attn[..., i, :i] = row + (row.unsqueeze(-1) * sub).sum(-2)
    attn = attn + torch.eye(chunk_size, dtype=attn.dtype, device=attn.device)
    value = attn @ v_beta
    k_cumdecay = attn @ (k_beta * g.exp().unsqueeze(-1))
    if initial_state is None:
        last_recurrent_state = torch.zeros(
            batch_size, num_heads, k_head_dim, v_head_dim, dtype=value.dtype, device=value.device
        )
    else:
        last_recurrent_state = initial_state.to(value.dtype)
    core_attn_out = torch.zeros_like(value)
    mask = torch.triu(
        torch.ones(chunk_size, chunk_size, dtype=torch.bool, device=query.device), diagonal=1
    )

    for i in range(0, total_sequence_length // chunk_size):
        q_i, k_i, v_i = query[:, :, i], key[:, :, i], value[:, :, i]
        attn = q_i @ k_i.transpose(-1, -2) * decay_mask[:, :, i]
        v_prime = (k_cumdecay[:, :, i]) @ last_recurrent_state
        v_new = v_i - v_prime
        attn_inter = (q_i * g[:, :, i, :, None].exp()) @ last_recurrent_state
        core_attn_out[:, :, i] = attn_inter + attn @ v_new
        last_recurrent_state = (
            last_recurrent_state * g[:, :, i, -1, None, None].exp()
            + (k_i * (g[:, :, i, -1, None] - g[:, :, i]).exp()[..., None]).transpose(-1, -2) @ v_new
        )

    core_attn_out = core_attn_out.reshape(
        core_attn_out.shape[0], core_attn_out.shape[1], -1, core_attn_out.shape[-1]
    )
    core_attn_out = core_attn_out[:, :, :sequence_length]
    core_attn_out = core_attn_out.transpose(1, 2).contiguous().to(initial_dtype)
    final_state = last_recurrent_state if output_final_state else None
    return core_attn_out, final_state


class Qwen3_5MoeGatedDeltaNet(nn.Module):
    """Gated DeltaNet token mixer (training path: chunked rule, no cache)."""

    def __init__(
        self,
        hidden_size: int,
        linear_num_value_heads: int,
        linear_num_key_heads: int,
        linear_key_head_dim: int,
        linear_value_head_dim: int,
        linear_conv_kernel_dim: int,
        rms_norm_eps: float = 1e-6,
        cp_group: Optional["dist.ProcessGroup"] = None,
    ):
        super().__init__()
        self.hidden_size = hidden_size
        self.num_v_heads = linear_num_value_heads
        self.num_k_heads = linear_num_key_heads
        self.head_k_dim = linear_key_head_dim
        self.head_v_dim = linear_value_head_dim
        self.key_dim = self.head_k_dim * self.num_k_heads
        self.value_dim = self.head_v_dim * self.num_v_heads
        self.conv_kernel_size = linear_conv_kernel_dim
        self.use_fla = chunk_gated_delta_rule is not None
        # Context-parallel process group (None / size-1 -> single-rank path).
        self.cp_group = cp_group

        self.conv_dim = self.key_dim * 2 + self.value_dim
        self.conv1d = nn.Conv1d(
            in_channels=self.conv_dim,
            out_channels=self.conv_dim,
            bias=False,
            kernel_size=self.conv_kernel_size,
            groups=self.conv_dim,
            padding=self.conv_kernel_size - 1,
        )

        self.dt_bias = nn.Parameter(torch.ones(self.num_v_heads))
        A = torch.empty(self.num_v_heads).uniform_(0, 16)
        self.A_log = nn.Parameter(torch.log(A))

        self.norm = Qwen3_5MoeRMSNormGated(self.head_v_dim, eps=rms_norm_eps)

        LinearCls = get_linear_cls()
        self.in_proj_qkv = LinearCls(
            self.hidden_size, self.key_dim * 2 + self.value_dim, bias=False
        )
        self.in_proj_z = LinearCls(self.hidden_size, self.value_dim, bias=False)
        # b/a project to num_v_heads (e.g. 32), an N too small for the DeepGEMM FP8
        # GEMM's TMA tiling (degenerate descriptor -> CUDA_ERROR_INVALID_VALUE). They
        # are tiny, so keep them in BF16 -- no FP8 throughput is lost.
        self.in_proj_b = nn.Linear(self.hidden_size, self.num_v_heads, bias=False)
        self.in_proj_a = nn.Linear(self.hidden_size, self.num_v_heads, bias=False)
        self.out_proj = LinearCls(self.value_dim, self.hidden_size, bias=False)

    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        # Projections are sequence-sharded (zigzag) under CP; the head-parallel
        # helper handles the all-to-all / reorder so the conv + chunk scan run on
        # the full natural-order sequence for this rank's head slice.
        mixed_qkv = self.in_proj_qkv(hidden_states)
        z = self.in_proj_z(hidden_states)
        b = self.in_proj_b(hidden_states)
        a = self.in_proj_a(hidden_states)

        kernel = chunk_gated_delta_rule if self.use_fla else torch_chunk_gated_delta_rule
        core_attn_out = head_parallel_gated_delta_net(
            mixed_qkv,
            z,
            b,
            a,
            conv_weight=self.conv1d.weight,
            A_log=self.A_log,
            dt_bias=self.dt_bias,
            norm=self.norm,
            kernel=kernel,
            key_dim=self.key_dim,
            value_dim=self.value_dim,
            num_k_heads=self.num_k_heads,
            num_v_heads=self.num_v_heads,
            head_k_dim=self.head_k_dim,
            head_v_dim=self.head_v_dim,
            cp_group=self.cp_group,
            use_qk_l2norm_in_kernel=True,
        )
        return self.out_proj(core_attn_out)


# ---------------------------------------------------------------------------
# Full attention (gated output, partial rotary)
# ---------------------------------------------------------------------------


class Qwen3_5MoeAttention(nn.Module):
    """GQA with per-head output gate: q_proj emits [query | gate] per head."""

    def __init__(
        self,
        hidden_size: int,
        num_attention_heads: int,
        num_key_value_heads: int,
        head_dim: int,
        rms_norm_eps: float = 1e-6,
        attention_bias: bool = False,
        attn_output_gate: bool = True,
        cp_group: Optional["dist.ProcessGroup"] = None,
    ):
        super().__init__()
        self.cp_group = cp_group
        self.use_ring_attn = cp_group is not None and cp_group.size() > 1
        self.hidden_size = hidden_size
        self.num_heads = num_attention_heads
        self.num_kv_heads = num_key_value_heads
        self.head_dim = head_dim
        self.num_key_value_groups = num_attention_heads // num_key_value_heads
        self.scaling = head_dim**-0.5
        self.attn_output_gate = attn_output_gate

        q_out = num_attention_heads * head_dim * (2 if attn_output_gate else 1)
        LinearCls = get_linear_cls()
        self.q_proj = LinearCls(hidden_size, q_out, bias=attention_bias)
        self.k_proj = LinearCls(hidden_size, num_key_value_heads * head_dim, bias=attention_bias)
        self.v_proj = LinearCls(hidden_size, num_key_value_heads * head_dim, bias=attention_bias)
        self.o_proj = LinearCls(num_attention_heads * head_dim, hidden_size, bias=attention_bias)

        self.q_norm = Qwen3_5MoeRMSNorm(head_dim, eps=rms_norm_eps)
        self.k_norm = Qwen3_5MoeRMSNorm(head_dim, eps=rms_norm_eps)

    def forward(
        self,
        hidden_states: torch.Tensor,
        position_embeddings: Tuple[torch.Tensor, torch.Tensor],
    ) -> torch.Tensor:
        bsz, seq_len, _ = hidden_states.size()

        if self.attn_output_gate:
            query_states, gate = torch.chunk(
                self.q_proj(hidden_states).view(bsz, seq_len, -1, self.head_dim * 2), 2, dim=-1
            )
            gate = gate.reshape(bsz, seq_len, -1)
        else:
            query_states = self.q_proj(hidden_states).view(bsz, seq_len, -1, self.head_dim)
            gate = None

        query_states = self.q_norm(
            query_states.reshape(bsz, seq_len, self.num_heads, self.head_dim)
        )
        key_states = self.k_norm(
            self.k_proj(hidden_states).view(bsz, seq_len, self.num_kv_heads, self.head_dim)
        )
        value_states = self.v_proj(hidden_states).view(
            bsz, seq_len, self.num_kv_heads, self.head_dim
        )

        cos, sin = position_embeddings
        query_states, key_states = apply_rotary_pos_emb(query_states, key_states, cos, sin)

        if self.use_ring_attn:
            # Zigzag ring attention across CP ranks (load-balanced causal).
            attn_output = ring_attention_func(
                query_states,
                key_states,
                value_states,
                sm_scale=self.scaling,
                cp_group=self.cp_group,
            )
        else:
            attn_output = flash_attn_func(
                query_states,
                key_states,
                value_states,
                softmax_scale=self.scaling,
                causal=True,
            )

        attn_output = attn_output.reshape(bsz, seq_len, self.num_heads * self.head_dim)
        if gate is not None:
            attn_output = attn_output * torch.sigmoid(gate)
        return self.o_proj(attn_output)


# ---------------------------------------------------------------------------
# MoE: shared expert MLP, routed experts, router
# ---------------------------------------------------------------------------


class Qwen3_5MoeMLP(nn.Module):
    """Dense SwiGLU MLP (used as the shared expert)."""

    def __init__(self, hidden_size: int, intermediate_size: int):
        super().__init__()
        self.hidden_size = hidden_size
        self.intermediate_size = intermediate_size

        LinearCls = get_linear_cls()
        self.gate_proj = LinearCls(hidden_size, intermediate_size, bias=False)
        self.up_proj = LinearCls(hidden_size, intermediate_size, bias=False)
        self.down_proj = LinearCls(intermediate_size, hidden_size, bias=False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.down_proj(silu_mul(self.gate_proj(x), self.up_proj(x)))


class Qwen3_5MoeExperts(nn.Module):
    """Routed experts with fused gate_up stored as raw nn.Parameter [E, 2I, H].

    Matches HF's tensor structure (fused gate_up, chunk-split into gate / up
    halves along the output dim). FP8 path mirrors GptOssExperts: raw
    Parameters cannot use the FP8GroupLinear module wrapper, so we dispatch
    FP8GroupLinearFunc directly and host the quantized-weight cache here.
    """

    def __init__(self, num_experts: int, hidden_size: int, intermediate_size: int):
        super().__init__()
        self.num_experts = num_experts
        self.hidden_size = hidden_size
        self.intermediate_size = intermediate_size
        self.gate_up_proj = nn.Parameter(
            torch.empty(num_experts, 2 * intermediate_size, hidden_size)
        )
        self.down_proj = nn.Parameter(torch.empty(num_experts, hidden_size, intermediate_size))

        self._fp8 = ModelImplMode.fp8_training == "deep-gemm"
        self._wq_cache: dict[str, tuple] | None = None
        self._wq_version: int = -1

    def _quantized_weight(self, name: str, weight: torch.Tensor) -> tuple:
        if torch.compiler.is_compiling():
            return fused_blockwise_transpose_cast_to_fp8_batched(weight)
        ver = FP8WeightCacheControl._version
        cache = self._wq_cache
        if FP8WeightCacheControl.enabled and self._wq_version == ver and cache is not None:
            hit = cache.get(name)
            if hit is not None:
                return hit
        result = fused_blockwise_transpose_cast_to_fp8_batched(weight)
        if FP8WeightCacheControl.enabled:
            if self._wq_version != ver or cache is None:
                self._wq_cache = {name: result}
                self._wq_version = ver
            else:
                cache[name] = result
        return result

    def _group_linear(
        self,
        x: torch.Tensor,
        weight: nn.Parameter,
        name: str,
        offs: torch.Tensor,
        ks: list | None,
        ks_tensor: torch.Tensor | None,
        group_indices: torch.Tensor | None,
    ) -> torch.Tensor:
        if x.shape[0] == 0:
            return x @ weight[0].transpose(-2, -1)
        if self._fp8:
            return FP8GroupLinearFunc.apply(
                x, weight, offs, ks, ks_tensor, self._quantized_weight(name, weight), group_indices
            )
        return GroupLinearFunc.apply(x, weight, offs)

    def forward(
        self,
        x: torch.Tensor,
        grouped_mm_offs: torch.Tensor,
        ks: list | None = None,
        ks_tensor: torch.Tensor | None = None,
    ) -> torch.Tensor:
        gi = precompute_group_indices(grouped_mm_offs, x.shape[0]) if self._fp8 else None

        gate_up = self._group_linear(
            x, self.gate_up_proj, "gate_up_proj", grouped_mm_offs, ks, ks_tensor, gi
        )
        gate, up = gate_up.chunk(2, dim=-1)
        activated = silu_mul(gate.contiguous(), up.contiguous())

        return self._group_linear(
            activated, self.down_proj, "down_proj", grouped_mm_offs, ks, ks_tensor, gi
        )


class Qwen3_5MoeTopKRouter(nn.Module):
    """Softmax-then-top-k router with renormalized weights."""

    def __init__(self, hidden_size: int, num_experts: int, num_experts_per_tok: int):
        super().__init__()
        self.num_experts = num_experts
        self.num_experts_per_tok = num_experts_per_tok
        self.load_balance_loss_fn = None
        self.weight = nn.Parameter(torch.zeros((num_experts, hidden_size)), requires_grad=True)

    @torch.compile(fullgraph=True)
    def compute(
        self, hidden_states: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor, Optional[torch.Tensor]]:
        batch_size, seq_len, hidden_size = hidden_states.shape
        hidden_states = hidden_states.view(-1, hidden_size)

        logits = F.linear(hidden_states, self.weight, None)
        scores = logits.softmax(dim=-1, dtype=torch.float32)
        topk_weight, topk_idx = torch.topk(scores, k=self.num_experts_per_tok, dim=-1, sorted=False)
        topk_weight = topk_weight / topk_weight.sum(dim=-1, keepdim=True)
        topk_weight = topk_weight.to(logits.dtype)

        if self.training and self.load_balance_loss_fn is not None:
            lb_loss = self.load_balance_loss_fn(
                scores, topk_idx, self.num_experts, self.num_experts_per_tok
            )
            topk_weight = MoELoadBalanceLossInjector.apply(topk_weight, lb_loss)
        else:
            lb_loss = None

        return topk_idx, topk_weight, lb_loss

    def forward(self, hidden_states: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        topk_idx, topk_weight, lb_loss = self.compute(hidden_states)
        if lb_loss is not None:
            MoELoadBalanceLossTracker.add(lb_loss)
        return topk_idx, topk_weight


class Qwen3_5MoeSparseMoeBlock(nn.Module):
    """Routed experts + always-on shared expert with a sigmoid gate.

    The shared expert is *not* applied in this module's pipelined path — it is
    folded into the residual inside ``_forward_attn_compute`` so its compute
    overlaps the stage-2 all-to-all (Hard Rule 3). ``forward`` (the reference
    path) applies it inline, mirroring HF.
    """

    def __init__(
        self,
        hidden_size: int,
        num_experts: int,
        num_experts_per_tok: int,
        moe_intermediate_size: int,
        shared_expert_intermediate_size: int,
        ep_size: int = 1,
        ep_group: Optional[dist.ProcessGroup] = None,
    ):
        super().__init__()
        self.hidden_size = hidden_size
        self.num_experts = num_experts
        self.num_experts_per_tok = num_experts_per_tok
        self.moe_intermediate_size = moe_intermediate_size

        self.ep_size = ep_size
        self.ep_group = ep_group
        self.ep_rank = ep_group.rank() if ep_group is not None else 0
        self.experts_per_rank = num_experts // ep_size

        self.gate = Qwen3_5MoeTopKRouter(hidden_size, num_experts, num_experts_per_tok)
        self.experts = Qwen3_5MoeExperts(self.experts_per_rank, hidden_size, moe_intermediate_size)
        self.shared_expert = Qwen3_5MoeMLP(hidden_size, shared_expert_intermediate_size)
        # N=1 scalar gate: too small for the DeepGEMM FP8 GEMM's TMA tiling
        # (degenerate descriptor -> CUDA_ERROR_INVALID_VALUE). Keep it BF16.
        self.shared_expert_gate = nn.Linear(hidden_size, 1, bias=False)

    def shared_expert_output(self, hidden_states: torch.Tensor) -> torch.Tensor:
        return torch.sigmoid(self.shared_expert_gate(hidden_states)) * self.shared_expert(
            hidden_states
        )

    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        orig_shape = hidden_states.shape
        shared_out = self.shared_expert_output(hidden_states)
        topk_idx, topk_weight = self.gate(hidden_states)
        hidden_states = hidden_states.view(-1, hidden_states.shape[-1])
        y = self.moe_infer(hidden_states, topk_idx, topk_weight).view(*orig_shape)
        return y + shared_out

    def moe_infer(
        self,
        x: torch.Tensor,
        topk_ids: torch.Tensor,
        topk_weight: torch.Tensor,
    ) -> torch.Tensor:
        assert self.ep_size == 1, "Reference implementation only supports ep_size=1"
        expert_idxs = topk_ids.view(-1)
        sorted_tokens = (
            x.unsqueeze(1).expand(-1, self.num_experts_per_tok, -1).reshape(-1, x.shape[-1])
        )
        output_tokens, reverse_shuffle_idxs, grouped_mm_offs, ks, ks_tensor = (
            scatter_for_grouped_gemm(sorted_tokens, expert_idxs, self.experts_per_rank)
        )
        outs = self.experts(output_tokens, grouped_mm_offs, ks=ks, ks_tensor=ks_tensor)
        outs = outs[reverse_shuffle_idxs]

        final_out = (
            (outs.view(*topk_ids.shape, -1) * topk_weight.unsqueeze(dim=-1))
            .sum(dim=1)
            .to(outs.dtype)
        )
        return final_out


# ---------------------------------------------------------------------------
# Decoder layer (5-stage protocol)
# ---------------------------------------------------------------------------


class Qwen3_5MoeDecoderLayer(nn.Module):
    """Hybrid decoder layer: Gated DeltaNet or gated full attention + MoE."""

    def __init__(
        self,
        layer_type: str,
        hidden_size: int,
        num_attention_heads: int,
        num_key_value_heads: int,
        head_dim: int,
        linear_num_value_heads: int,
        linear_num_key_heads: int,
        linear_key_head_dim: int,
        linear_value_head_dim: int,
        linear_conv_kernel_dim: int,
        num_experts: int,
        num_experts_per_tok: int,
        moe_intermediate_size: int,
        shared_expert_intermediate_size: int,
        rms_norm_eps: float,
        attention_bias: bool,
        attn_output_gate: bool,
        layer_idx: int,
        ep_size: int = 1,
        ep_group: Optional[dist.ProcessGroup] = None,
        cp_group: Optional[dist.ProcessGroup] = None,
    ):
        super().__init__()
        self.idx = layer_idx
        self.hidden_size = hidden_size
        self.layer_type = layer_type

        if layer_type == "linear_attention":
            self.linear_attn = Qwen3_5MoeGatedDeltaNet(
                hidden_size=hidden_size,
                linear_num_value_heads=linear_num_value_heads,
                linear_num_key_heads=linear_num_key_heads,
                linear_key_head_dim=linear_key_head_dim,
                linear_value_head_dim=linear_value_head_dim,
                linear_conv_kernel_dim=linear_conv_kernel_dim,
                rms_norm_eps=rms_norm_eps,
                cp_group=cp_group,
            )
        elif layer_type == "full_attention":
            self.self_attn = Qwen3_5MoeAttention(
                hidden_size=hidden_size,
                num_attention_heads=num_attention_heads,
                num_key_value_heads=num_key_value_heads,
                head_dim=head_dim,
                rms_norm_eps=rms_norm_eps,
                attention_bias=attention_bias,
                attn_output_gate=attn_output_gate,
                cp_group=cp_group,
            )
        else:
            raise ValueError(f"Unknown layer_type: {layer_type}")

        self.mlp = Qwen3_5MoeSparseMoeBlock(
            hidden_size=hidden_size,
            num_experts=num_experts,
            num_experts_per_tok=num_experts_per_tok,
            moe_intermediate_size=moe_intermediate_size,
            shared_expert_intermediate_size=shared_expert_intermediate_size,
            ep_size=ep_size,
            ep_group=ep_group,
        )

        self.input_layernorm = Qwen3_5MoeRMSNorm(hidden_size, eps=rms_norm_eps)
        self.post_attention_layernorm = Qwen3_5MoeRMSNorm(hidden_size, eps=rms_norm_eps)

        # The FLA chunked delta-rule kernel self-compiles; nested compile is
        # not supported, so unwrap the compiled region only when it is active.
        cp_active = cp_group is not None and cp_group.size() > 1
        unwrap_compile = layer_type == "linear_attention" and (
            self.linear_attn.use_fla or cp_active
        )
        if layer_type == "full_attention" and cp_active:
            # Ring attention uses async P2P + a custom autograd Function that
            # torch.compile(fullgraph=True) cannot trace; run it eagerly.
            unwrap_compile = True
        if unwrap_compile:
            self._forward_attn_compute = self._forward_attn_compute.__wrapped__.__get__(
                self, type(self)
            )

    def _token_mixer(self, hidden_states: torch.Tensor) -> torch.Tensor:
        if self.layer_type == "linear_attention":
            return self.linear_attn(hidden_states)
        position_embeddings = getattr(self, "_position_embeddings", None)
        if position_embeddings is None:
            raise RuntimeError("Position embeddings must be set before calling forward_attn")
        return self.self_attn(hidden_states, position_embeddings=position_embeddings)

    @torch.compile(fullgraph=True)
    def _forward_attn_compute(self, hidden_states: torch.Tensor):
        residual = hidden_states
        hidden_states = self.input_layernorm(hidden_states)
        hidden_states = self._token_mixer(hidden_states)
        hidden_states = residual + hidden_states

        residual = hidden_states
        hidden_states = self.post_attention_layernorm(hidden_states)

        # Shared expert folds into the residual here so its compute overlaps
        # the stage-2 all-to-all dispatch of the routed tokens.
        residual = residual + self.mlp.shared_expert_output(hidden_states)

        return hidden_states, residual

    def forward_attn(self, hidden_states: torch.Tensor) -> ForwardAttnOutput:
        """LN + token mixer + LN + shared expert + expert routing."""
        hidden_states, residual = self._forward_attn_compute(hidden_states)

        topk_ids, topk_weight = self.mlp.gate(hidden_states)
        (
            sorted_tokens,
            idxs,
            expert_idxs,
            expand_idx,
            dedup_input_splits,
            dedup_output_splits,
            input_splits,
            output_splits,
        ) = moe_ep_prepare_dispatch(
            hidden_states,
            topk_ids,
            self.mlp.num_experts,
            self.mlp.ep_size,
            self.mlp.experts_per_rank,
            self.mlp.ep_group,
        )

        return ForwardAttnOutput(
            sorted_tokens,
            idxs,
            topk_weight,
            output_splits,
            input_splits,
            expert_idxs,
            residual,
            expand_idx,
            dedup_input_splits,
            dedup_output_splits,
        )

    def forward_mlp(
        self,
        gathered_tokens: torch.Tensor,
        expert_idxs: Optional[torch.Tensor] = None,
        expand_idx: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """Routed expert forward (grouped GEMM)."""
        assert expert_idxs is not None
        if expand_idx is not None:
            gathered_tokens = padded_index_gather(gathered_tokens, expand_idx)
        output_tokens, reverse_shuffle_idxs, grouped_mm_offs, ks, ks_tensor = (
            scatter_for_grouped_gemm(gathered_tokens, expert_idxs, self.mlp.experts_per_rank)
        )
        del gathered_tokens  # free expanded tokens; no longer needed after scatter
        outs = self.mlp.experts(output_tokens, grouped_mm_offs, ks=ks, ks_tensor=ks_tensor)
        outs = padded_index_gather(outs, reverse_shuffle_idxs)
        return outs

    @torch.compile(fullgraph=True)
    def forward_aggregate(
        self,
        moe_outs: torch.Tensor,
        moe_local_idxs: Optional[torch.Tensor],
        topk_weight: Optional[torch.Tensor],
        residual: torch.Tensor,
    ) -> torch.Tensor:
        """Weighted expert output + residual (shared expert already in residual)."""
        if self.mlp.ep_size > 1:
            assert moe_local_idxs is not None
            seq_len, topk = topk_weight.shape
            permuted_probs = topk_weight.view(-1)[moe_local_idxs]
            token_indices = moe_local_idxs // topk
            weighted = (moe_outs.float() * permuted_probs.unsqueeze(-1)).to(moe_outs.dtype)
            hidden_states = moe_outs.new_zeros(seq_len, moe_outs.shape[-1])
            hidden_states.scatter_add_(0, token_indices[:, None].expand_as(weighted), weighted)
            hidden_states = hidden_states.view(*residual.shape)
        else:
            assert moe_local_idxs is None
            final_out = moe_outs.view(*topk_weight.shape, -1) * topk_weight.unsqueeze(dim=-1)
            hidden_states = final_out.sum(dim=1).to(moe_outs.dtype).view(*residual.shape)

        return residual + hidden_states

    def reference_forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        """Non-pipelined eager forward for correctness validation."""
        residual = hidden_states
        hidden_states = self.input_layernorm(hidden_states)
        hidden_states = self._token_mixer(hidden_states)
        hidden_states = residual + hidden_states

        residual = hidden_states
        hidden_states = self.post_attention_layernorm(hidden_states)
        hidden_states = self.mlp(hidden_states)
        return residual + hidden_states


# ---------------------------------------------------------------------------
# Model
# ---------------------------------------------------------------------------


class Qwen3_5MoeModel(nn.Module):
    """Qwen3.5 MoE text model for DualPipeV pipeline parallelism."""

    def __init__(
        self,
        config,
        num_stages: int,
        stage_id: int,
        cp_group: Optional[dist.ProcessGroup] = None,
        ep_group: Optional[dist.ProcessGroup] = None,
    ):
        super().__init__()
        self.cp_group = cp_group
        self.cp_rank = cp_group.rank() if cp_group is not None else 0
        self.cp_size = cp_group.size() if cp_group is not None else 1

        self.config = config
        self.stage_id = stage_id
        self.num_stages = num_stages

        hidden_size = config.hidden_size
        rms_norm_eps = config.rms_norm_eps
        vocab_size = config.vocab_size
        layer_types = config.layer_types
        rope_parameters = getattr(config, "rope_parameters", None) or {}
        rope_theta = rope_parameters.get("rope_theta", 10000000.0)
        partial_rotary_factor = rope_parameters.get("partial_rotary_factor", 0.25)
        head_dim = getattr(config, "head_dim", hidden_size // config.num_attention_heads)

        ep_size = getattr(config, "ep_size", 1)

        self.embed_tokens = nn.Embedding(vocab_size, hidden_size) if stage_id == 0 else None

        num_local_layers = layer_partition(config.num_hidden_layers, num_stages)
        layer_id_begin = sum(num_local_layers[:stage_id])
        layer_id_end = layer_id_begin + num_local_layers[stage_id]

        self.layers = nn.ModuleDict(
            {
                str(i): Qwen3_5MoeDecoderLayer(
                    layer_type=layer_types[i],
                    hidden_size=hidden_size,
                    num_attention_heads=config.num_attention_heads,
                    num_key_value_heads=config.num_key_value_heads,
                    head_dim=head_dim,
                    linear_num_value_heads=config.linear_num_value_heads,
                    linear_num_key_heads=config.linear_num_key_heads,
                    linear_key_head_dim=config.linear_key_head_dim,
                    linear_value_head_dim=config.linear_value_head_dim,
                    linear_conv_kernel_dim=config.linear_conv_kernel_dim,
                    num_experts=config.num_experts,
                    num_experts_per_tok=config.num_experts_per_tok,
                    moe_intermediate_size=config.moe_intermediate_size,
                    shared_expert_intermediate_size=config.shared_expert_intermediate_size,
                    rms_norm_eps=rms_norm_eps,
                    attention_bias=getattr(config, "attention_bias", False),
                    attn_output_gate=getattr(config, "attn_output_gate", True),
                    layer_idx=i,
                    ep_size=ep_size,
                    ep_group=ep_group,
                    cp_group=cp_group,
                )
                for i in range(layer_id_begin, layer_id_end)
            }
        )

        if stage_id == num_stages - 1:
            self.norm = Qwen3_5MoeRMSNorm(hidden_size, eps=rms_norm_eps)
            self.lm_head = nn.Linear(hidden_size, vocab_size, bias=False)
        else:
            self.norm = None
            self.lm_head = None

        self.rotary_emb = Qwen3_5MoeTextRotaryEmbedding(
            int(head_dim * partial_rotary_factor),
            max_position_embeddings=config.max_position_embeddings,
            base=rope_theta,
        )

    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        intermediate_tensors: Optional[IntermediateTensors] = getattr(
            self, "_intermediate_tensors", None
        )

        if self.embed_tokens is not None:
            input_ids = hidden_states
            hidden_states = self.embed_tokens(input_ids)

        bsz, seq_len, _ = hidden_states.shape

        if self.cp_size == 1:
            cos, sin = self.rotary_emb(hidden_states, seq_len=seq_len)
            position_embeddings = (cos.unsqueeze(0), sin.unsqueeze(0))
        else:
            # Zigzag CP: the local seq_len tokens come from two non-contiguous
            # global chunks. Build global position ids (front + mirror back) and
            # gather cos/sin so RoPE uses true global positions.
            block = seq_len // 2
            global_seq_len = seq_len * self.cp_size
            front_start = self.cp_rank * block
            back_start = (2 * self.cp_size - self.cp_rank - 1) * block
            position_ids = torch.cat(
                [
                    torch.arange(front_start, front_start + block, device=hidden_states.device),
                    torch.arange(back_start, back_start + block, device=hidden_states.device),
                ]
            )
            cos, sin = self.rotary_emb(hidden_states, seq_len=global_seq_len)
            position_embeddings = (cos[position_ids].unsqueeze(0), sin[position_ids].unsqueeze(0))

        for _, layer in self.layers.items():
            layer._position_embeddings = position_embeddings

        if intermediate_tensors is None:
            for _, layer in self.layers.items():
                ret = decoder_layer_forward(layer, hidden_states)
                hidden_states = ret[0] if isinstance(ret, tuple) else ret
            if self.norm is not None:
                hidden_states = self.norm(hidden_states)
                hidden_states = self.lm_head(hidden_states)
            return hidden_states

        layer_idx = 0
        if self.embed_tokens is not None:
            intermediate_tensors.prolog.args = PrologArgs()
            intermediate_tensors.prolog.outs = PrologOuts(hidden_states)

        for _, layer in self.layers.items():
            ret = decoder_layer_forward(layer, hidden_states)
            if len(ret) == 2:
                hidden_states, layer_record = ret
                dst = intermediate_tensors.layers[layer_idx]
                for field in fields(layer_record):
                    src_rec = getattr(layer_record, field.name)
                    dst_rec = getattr(dst, field.name)
                    for rf in fields(src_rec):
                        setattr(dst_rec, rf.name, getattr(src_rec, rf.name))
            else:
                hidden_states = ret[0]
                dst = intermediate_tensors.layers[layer_idx]
                for field in fields(dst):
                    record = getattr(dst, field.name)
                    for rf in fields(record):
                        setattr(record, rf.name, None)
            layer_idx += 1

        if self.norm is not None:
            assert self.lm_head is not None
            if not ModelImplMode.use_reference_fwd:
                hidden_states = hidden_states.detach().requires_grad_()
            intermediate_tensors.epilog.args = EpilogArgs(hidden_states)
            hidden_states = self.norm(hidden_states)
            hidden_states = self.lm_head(hidden_states)

        return hidden_states

    @staticmethod
    def backward(
        module: "Qwen3_5MoeModel",
        dy: Optional[List[torch.Tensor]],
        loss: Optional[torch.Tensor],
        intermediate_tensors: IntermediateTensors,
    ):
        """Backward pass for the model."""
        assert (dy is None) != (loss is None), "Either dy or loss should be provided"

        if loss is not None:
            assert module.norm is not None
            assert module.lm_head is not None
            loss.backward()
            loss.detach_()
            dy = (intermediate_tensors.epilog.args.hidden_states.grad,)
            intermediate_tensors.epilog.args = None
            loss = None
        else:
            assert module.norm is None
            assert module.lm_head is None

        dx = dy
        layers_list = [layer for _, layer in module.layers.items()]
        for layer, intermediate_tensors_layer in zip(
            reversed(layers_list), reversed(intermediate_tensors.layers)
        ):
            dx = (decoder_layer_backward(layer, dx, loss, intermediate_tensors_layer),)

        final_grads = dx
        if module.embed_tokens is not None:
            record = intermediate_tensors.prolog
            run_backward(record.outs, dx)
            for rf in fields(record):
                setattr(record, rf.name, None)
            final_grads = (None,)

        return final_grads
