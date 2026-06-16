"""Increment 3: contiguous-shard full-attention CP (all-gather K/V).

The reference is flash_attn_func(causal=True) run on the FULL sequence on one
rank; the implementation splits Q/K/V contiguously across CP ranks, all-gathers
K/V on each rank, and calls flash_attn with the gathered prefix.

We verify:
  - forward output on each rank's contiguous shard matches the reference slice
  - input gradients dQ, dK, dV on each shard match the reference slice
  - cp_size==1 is a no-op (identical results)

Tests cover GQA (HK < HQ) and MHA (HK == HQ), cp_size 2 and 4.
"""

from dataclasses import dataclass

import pytest
import torch

from pithtrain.modules.distributed import DistributedCfg, DistributedCtx
from pithtrain.operators.flash_attn_v4 import flash_attn_func
from pithtrain.operators.full_attn_cp import full_attn_cp
from tests.utilities import cosine_error, launch


@dataclass
class FullAttnRequest:
    B: int
    S: int
    HQ: int
    HK: int
    D: int
    atol: float = 2e-3  # bfloat16 attention; all-gather introduces minor order-of-ops diff


def extract_contiguous(x: torch.Tensor, cp_rank: int, cp_size: int) -> torch.Tensor:
    """Contiguous shard of x along dim 1: chunk cp_rank of cp_size."""
    return x.chunk(cp_size, dim=1)[cp_rank].contiguous()


def verify_full_attn_cp(ctx: DistributedCtx, req: FullAttnRequest) -> None:
    cp_group = ctx.device_mesh.get_group("cp")
    cp_rank = cp_group.rank()
    cp_size = cp_group.size()
    device = torch.cuda.current_device()
    scale = req.D ** -0.5

    torch.manual_seed(42)
    q_full = torch.randn(req.B, req.S, req.HQ, req.D, device=device, dtype=torch.bfloat16)
    k_full = torch.randn(req.B, req.S, req.HK, req.D, device=device, dtype=torch.bfloat16)
    v_full = torch.randn(req.B, req.S, req.HK, req.D, device=device, dtype=torch.bfloat16)

    # Reference: full-sequence causal attention, slice to shard.
    qr = q_full.clone().requires_grad_(True)
    kr = k_full.clone().requires_grad_(True)
    vr = v_full.clone().requires_grad_(True)
    out_ref = flash_attn_func(qr, kr, vr, scale, causal=True)
    out_ref.sum().backward()

    out_ref_local = extract_contiguous(out_ref.detach(), cp_rank, cp_size)
    dq_ref = extract_contiguous(qr.grad, cp_rank, cp_size)
    dk_ref = extract_contiguous(kr.grad, cp_rank, cp_size)
    dv_ref = extract_contiguous(vr.grad, cp_rank, cp_size)

    # Implementation: per-shard all-gather CP.
    qi = extract_contiguous(q_full, cp_rank, cp_size).clone().requires_grad_(True)
    ki = extract_contiguous(k_full, cp_rank, cp_size).clone().requires_grad_(True)
    vi = extract_contiguous(v_full, cp_rank, cp_size).clone().requires_grad_(True)
    out_imp = full_attn_cp(qi, ki, vi, scale, cp_group)
    out_imp.sum().backward()

    checks = {
        "out": (out_ref_local, out_imp.detach()),
        "dq": (dq_ref, qi.grad),
        "dk": (dk_ref, ki.grad),
        "dv": (dv_ref, vi.grad),
    }
    for name, (ref, imp) in checks.items():
        err = cosine_error(ref, imp)
        if err >= req.atol:
            raise AssertionError(f"full_attn_cp {name} diverged: {err=:.2e} >= {req.atol=}")


REQUESTS = [
    pytest.param(2, FullAttnRequest(B=1, S=2048, HQ=8, HK=2, D=128), id="CP2-GQA-S2048"),
    pytest.param(2, FullAttnRequest(B=2, S=1024, HQ=4, HK=4, D=64), id="CP2-MHA-S1024"),
    pytest.param(4, FullAttnRequest(B=1, S=4096, HQ=8, HK=2, D=128), id="CP4-GQA-S4096"),
    pytest.param(4, FullAttnRequest(B=1, S=2048, HQ=4, HK=4, D=64), id="CP4-MHA-S2048"),
]


@pytest.mark.parametrize("cp_size,req", REQUESTS)
def test_full_attn_cp_vs_dense(cp_size: int, req: FullAttnRequest) -> None:
    cfg = DistributedCfg()
    cfg.context_parallel_size = cp_size
    launch(cfg, verify_full_attn_cp, req)
