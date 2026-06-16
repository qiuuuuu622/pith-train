"""Megatron-style all-to-all head-parallel CP for the Gated DeltaNet.

Validates ``head_parallel_gated_delta_net`` against a single-rank, full-sequence
reference: same weights and inputs, the CP path runs each rank on its zigzag
shard and must reproduce the reference forward output and input gradients.

Also round-trips the all-to-all (cp2hp -> hp2cp == identity).  Uses float32 + the
deterministic torch fallback kernel for a tight tolerance.
"""

from dataclasses import dataclass

import pytest
import torch

from pithtrain.modules.distributed import DistributedCfg, DistributedCtx
from pithtrain.models.qwen3_5_moe import Qwen3_5MoeRMSNormGated, torch_chunk_gated_delta_rule
from pithtrain.operators.gated_delta_rule_cp import (
    all_to_all_cp2hp,
    all_to_all_hp2cp,
    head_parallel_gated_delta_net,
)
from tests.utilities import cosine_error, launch


def extract_zigzag(x: torch.Tensor, cp_rank: int, cp_size: int) -> torch.Tensor:
    """This rank's zigzag-local slice along the sequence dim (dim=1)."""
    chunks = x.chunk(2 * cp_size, dim=1)
    return torch.cat([chunks[cp_rank], chunks[2 * cp_size - cp_rank - 1]], dim=1).contiguous()


@dataclass
class Request:
    B: int = 1
    S: int = 256
    num_v_heads: int = 8
    num_k_heads: int = 4
    head_k_dim: int = 32
    head_v_dim: int = 32
    conv_kernel: int = 4
    atol: float = 1e-5


def _run_a2a_roundtrip(ctx: DistributedCtx, req: Request) -> None:
    cp_group = ctx.device_mesh.get_group("cp")
    cp_size = cp_group.size()
    device = torch.cuda.current_device()
    C = 64
    torch.manual_seed(7 + cp_group.rank())
    x = torch.randn(req.B, req.S // cp_size, C, device=device, dtype=torch.float32)
    y = all_to_all_hp2cp(all_to_all_cp2hp(x, cp_group), cp_group)
    err = cosine_error(x, y)
    if err >= 1e-6:
        raise AssertionError(f"a2a roundtrip mismatch {err=:.2e}")


def _run_layer(ctx: DistributedCtx, req: Request) -> None:
    cp_group = ctx.device_mesh.get_group("cp")
    cp_rank, cp_size = cp_group.rank(), cp_group.size()
    device = torch.cuda.current_device()

    key_dim = req.num_k_heads * req.head_k_dim
    value_dim = req.num_v_heads * req.head_v_dim
    conv_dim = key_dim * 2 + value_dim

    # Identical weights + inputs on every rank.
    torch.manual_seed(1234)
    conv_weight = torch.randn(conv_dim, 1, req.conv_kernel, device=device, dtype=torch.float32)
    A_log = torch.rand(req.num_v_heads, device=device, dtype=torch.float32)
    dt_bias = torch.rand(req.num_v_heads, device=device, dtype=torch.float32)
    norm = Qwen3_5MoeRMSNormGated(req.head_v_dim).to(device=device, dtype=torch.float32)

    mixed_full = torch.randn(req.B, req.S, conv_dim, device=device, dtype=torch.float32)
    z_full = torch.randn(req.B, req.S, value_dim, device=device, dtype=torch.float32)
    b_full = torch.randn(req.B, req.S, req.num_v_heads, device=device, dtype=torch.float32)
    a_full = torch.randn(req.B, req.S, req.num_v_heads, device=device, dtype=torch.float32)

    common = dict(
        conv_weight=conv_weight, A_log=A_log, dt_bias=dt_bias, norm=norm,
        kernel=torch_chunk_gated_delta_rule, key_dim=key_dim, value_dim=value_dim,
        num_k_heads=req.num_k_heads, num_v_heads=req.num_v_heads,
        head_k_dim=req.head_k_dim, head_v_dim=req.head_v_dim,
    )

    # ---- Reference: single-rank, full sequence ----
    ins_ref = [t.clone().requires_grad_(True) for t in (mixed_full, z_full, b_full, a_full)]
    out_ref = head_parallel_gated_delta_net(*ins_ref, cp_group=None, **common)
    out_ref.sum().backward()

    # ---- CP path: this rank's zigzag shard ----
    ins_cp = [
        extract_zigzag(t, cp_rank, cp_size).detach().requires_grad_(True)
        for t in (mixed_full, z_full, b_full, a_full)
    ]
    out_cp = head_parallel_gated_delta_net(*ins_cp, cp_group=cp_group, **common)
    out_cp.sum().backward()

    # ---- Compare forward + input grads on this rank's zigzag slice ----
    err_out = cosine_error(extract_zigzag(out_ref.detach(), cp_rank, cp_size), out_cp.detach())
    if err_out >= req.atol:
        raise AssertionError(f"[rank {cp_rank}] fwd mismatch {err_out=:.2e}")
    names = ["mixed_qkv", "z", "b", "a"]
    for name, ref, cp in zip(names, ins_ref, ins_cp):
        err = cosine_error(extract_zigzag(ref.grad, cp_rank, cp_size), cp.grad)
        if err >= req.atol:
            raise AssertionError(f"[rank {cp_rank}] d{name} mismatch {err=:.2e}")


@pytest.mark.parametrize("cp_size", [2, 4])
def test_a2a_roundtrip(cp_size: int) -> None:
    cfg = DistributedCfg()
    cfg.context_parallel_size = cp_size
    launch(cfg, _run_a2a_roundtrip, Request())


@pytest.mark.parametrize("cp_size", [2, 4])
def test_head_parallel_matches_full_sequence(cp_size: int) -> None:
    cfg = DistributedCfg()
    cfg.context_parallel_size = cp_size
    launch(cfg, _run_layer, Request())
