"""Increment 2: selective activation recompute for Qwen3_5MoeGatedDeltaNet.

Correctness test: run the layer with use_selective_recompute=True and compare
forward output + all input gradients against use_selective_recompute=False.

Key property being validated: during recompute the CP halo and initial_state
are NOT re-communicated — they are saved checkpoint inputs.  If re-communication
happened it would either deadlock (P2P ops run out of order) or produce wrong
gradients.  Matching gradients between the two paths proves correctness.

Tests cover:
  - cp_size 1 (no CP, verifies recompute doesn't break single-rank path)
  - cp_size 2 and 4 (verifies comm/compute split is correct end-to-end)
"""

from dataclasses import dataclass
from types import SimpleNamespace

import pytest
import torch
import torch.nn as nn

from pithtrain.modules.distributed import DistributedCfg, DistributedCtx
from pithtrain.models.qwen3_5_moe import Qwen3_5MoeGatedDeltaNet
from tests.utilities import cosine_error, launch


@dataclass
class RecomputeRequest:
    B: int
    S: int  # must be divisible by cp_size and shard must be multiple of chunk_size (64)
    hidden_size: int = 256
    num_v_heads: int = 4
    num_k_heads: int = 4
    head_k_dim: int = 32
    head_v_dim: int = 32
    conv_kernel: int = 4
    atol: float = 1e-5  # float32, so tight tolerance


def _make_layer(req: RecomputeRequest, cp_group=None) -> Qwen3_5MoeGatedDeltaNet:
    layer = Qwen3_5MoeGatedDeltaNet(
        hidden_size=req.hidden_size,
        linear_num_value_heads=req.num_v_heads,
        linear_num_key_heads=req.num_k_heads,
        linear_key_head_dim=req.head_k_dim,
        linear_value_head_dim=req.head_v_dim,
        linear_conv_kernel_dim=req.conv_kernel,
        rms_norm_eps=1e-6,
        cp_group=cp_group,
    ).to(device=torch.cuda.current_device(), dtype=torch.float32)
    # FLA chunk kernels are bf16/fp16 only (TileLang fails to compile the bwd in
    # float32); force the deterministic torch fallback for the float32 reference.
    layer.use_fla = False
    return layer


def verify_recompute(ctx: DistributedCtx, req: RecomputeRequest) -> None:
    cp_group = ctx.device_mesh.get_group("cp")
    cp_rank = cp_group.rank()
    cp_size = cp_group.size()
    device = torch.cuda.current_device()

    # Use a contiguous shard: each rank owns hidden_states[:, r*Sl:(r+1)*Sl]
    Sl = req.S // cp_size

    torch.manual_seed(42)
    hidden_full = torch.randn(req.B, req.S, req.hidden_size, device=device, dtype=torch.float32)
    hidden_local = hidden_full[:, cp_rank * Sl : (cp_rank + 1) * Sl].clone()

    # ---- Standard path ----
    torch.manual_seed(99)
    layer_std = _make_layer(req, cp_group=cp_group)
    layer_std.use_selective_recompute = False
    x_std = hidden_local.clone().requires_grad_(True)
    out_std = layer_std(x_std)
    out_std.sum().backward()

    # ---- Recompute path (same weights, same input) ----
    torch.manual_seed(99)
    layer_rc = _make_layer(req, cp_group=cp_group)
    layer_rc.use_selective_recompute = True
    # Copy weights from standard layer so outputs are comparable.
    for (n1, p1), (n2, p2) in zip(layer_std.named_parameters(), layer_rc.named_parameters()):
        p2.data.copy_(p1.data)
    x_rc = hidden_local.clone().requires_grad_(True)
    out_rc = layer_rc(x_rc)
    out_rc.sum().backward()

    # ---- Compare ----
    err_out = cosine_error(out_std.detach(), out_rc.detach())
    err_dx = cosine_error(x_std.grad, x_rc.grad)
    if err_out >= req.atol:
        raise AssertionError(
            f"[rank {cp_rank}] output mismatch: {err_out=:.2e} >= {req.atol=}"
        )
    if err_dx >= req.atol:
        raise AssertionError(
            f"[rank {cp_rank}] dx mismatch: {err_dx=:.2e} >= {req.atol=}"
        )


REQUESTS = [
    pytest.param(1, RecomputeRequest(B=1, S=128, hidden_size=128, num_v_heads=2, num_k_heads=2,
                                      head_k_dim=32, head_v_dim=32), id="CP1-noop"),
    pytest.param(2, RecomputeRequest(B=1, S=256, hidden_size=128, num_v_heads=2, num_k_heads=2,
                                      head_k_dim=32, head_v_dim=32), id="CP2-S256"),
    pytest.param(4, RecomputeRequest(B=1, S=512, hidden_size=256, num_v_heads=4, num_k_heads=4,
                                      head_k_dim=32, head_v_dim=32), id="CP4-S512"),
]


@pytest.mark.parametrize("cp_size,req", REQUESTS)
def test_selective_recompute_matches_standard(cp_size: int, req: RecomputeRequest) -> None:
    cfg = DistributedCfg()
    cfg.context_parallel_size = cp_size
    launch(cfg, verify_recompute, req)
