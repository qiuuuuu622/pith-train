"""Correctness of the Gated DeltaNet context-parallel path (contiguous shard).

Two cross-rank ops are exercised against a single-rank, full-sequence reference:

1. ``causal_conv_with_halo`` -- depthwise causal conv1d with a (K-1)-token left
   halo borrowed from the previous CP rank.
2. ``chunk_gated_delta_rule_cp`` -- the recurrent chunk delta-rule run on a
   contiguous sequence shard with the recurrent state threaded left-to-right.

For each, the reference is the op run on the FULL sequence on one logical rank;
the implementation splits the sequence contiguously across ``cp_size`` ranks and
runs the CP op on each shard. We compare the forward output and every input
gradient on this rank's contiguous slice. Both the forward (state/halo) and the
backward (reverse state/halo) communication are therefore covered.

The torch fallback kernel is used so the tests need no FLA install; everything runs
in float32 for a tight tolerance. ``launch`` skips when too few GPUs are available.
"""

from dataclasses import dataclass

import pytest
import torch
import torch.nn as nn

from pithtrain.modules.distributed import DistributedCfg, DistributedCtx
from pithtrain.models.qwen3_5_moe import torch_chunk_gated_delta_rule
from pithtrain.operators.gated_delta_rule_cp import (
    causal_conv_with_halo,
    chunk_gated_delta_rule_cp,
)
from tests.utilities import cosine_error, launch


def extract_contiguous(x: torch.Tensor, cp_rank: int, cp_size: int, dim: int) -> torch.Tensor:
    """This rank's contiguous shard along ``dim``: chunk ``cp_rank`` of ``cp_size``."""
    return x.chunk(cp_size, dim=dim)[cp_rank].contiguous()


# ---------------------------------------------------------------------------
# 1. Causal conv1d with halo exchange
# ---------------------------------------------------------------------------


@dataclass
class ConvRequest:
    B: int
    C: int
    S: int
    K: int
    atol: float = 1e-5


def verify_conv(ctx: DistributedCtx, req: ConvRequest) -> None:
    cp_group = ctx.device_mesh.get_group("cp")
    cp_rank, cp_size = cp_group.rank(), cp_group.size()
    device = torch.cuda.current_device()

    # Identical conv weights and inputs on every rank (same seed, same shapes).
    torch.manual_seed(42)
    conv = nn.Conv1d(req.C, req.C, kernel_size=req.K, groups=req.C, padding=req.K - 1, bias=False)
    conv = conv.to(device=device, dtype=torch.float32)
    x_full = torch.randn(req.B, req.C, req.S, device=device, dtype=torch.float32)

    # Reference: full-sequence causal conv on one rank, sliced to this shard.
    x_ref = x_full.clone().requires_grad_(True)
    out_ref = conv(x_ref)[..., : req.S]
    out_ref.sum().backward()
    out_ref_local = extract_contiguous(out_ref, cp_rank, cp_size, dim=2)
    dx_ref_local = extract_contiguous(x_ref.grad, cp_rank, cp_size, dim=2)

    # Implementation: per-shard halo conv.
    x_imp = extract_contiguous(x_full, cp_rank, cp_size, dim=2).clone().requires_grad_(True)
    out_imp = causal_conv_with_halo(conv, x_imp, cp_group)
    out_imp.sum().backward()

    for name, ref, imp in [("out", out_ref_local, out_imp), ("dx", dx_ref_local, x_imp.grad)]:
        err = cosine_error(ref, imp)
        if err >= req.atol:
            raise AssertionError(f"conv {name} diverged: {err=:.2e} >= {req.atol=}")


CONV_REQUESTS = [
    pytest.param(2, ConvRequest(B=1, C=64, S=256, K=4), id="CP2-K4-S256"),
    pytest.param(2, ConvRequest(B=2, C=128, S=512, K=4), id="CP2-K4-S512"),
    pytest.param(4, ConvRequest(B=1, C=64, S=512, K=4), id="CP4-K4-S512"),
    pytest.param(2, ConvRequest(B=1, C=32, S=384, K=3), id="CP2-K3-S384"),
]


@pytest.mark.parametrize("cp_size,req", CONV_REQUESTS)
def test_causal_conv_with_halo(cp_size: int, req: ConvRequest) -> None:
    cfg = DistributedCfg()
    cfg.context_parallel_size = cp_size
    launch(cfg, verify_conv, req)


# ---------------------------------------------------------------------------
# 2. Chunk gated delta rule recurrent-state scan
# ---------------------------------------------------------------------------


@dataclass
class ScanRequest:
    B: int
    S: int  # must be divisible by cp_size; shard should be a multiple of chunk_size (64)
    H: int  # num value heads (== num key heads here, no GQA repeat)
    Dk: int
    Dv: int
    atol: float = 2e-4


def _delta_rule_inputs(req: ScanRequest, device):
    """Reproducible q,k,v,g,beta in the kernel's [B, S, H, D] / [B, S, H] layout."""
    torch.manual_seed(7)
    q = torch.randn(req.B, req.S, req.H, req.Dk, device=device, dtype=torch.float32)
    k = torch.randn(req.B, req.S, req.H, req.Dk, device=device, dtype=torch.float32)
    v = torch.randn(req.B, req.S, req.H, req.Dv, device=device, dtype=torch.float32)
    # g is a (negative) log-decay; keep it small so exp() is well behaved.
    g = -0.1 * torch.rand(req.B, req.S, req.H, device=device, dtype=torch.float32)
    beta = torch.rand(req.B, req.S, req.H, device=device, dtype=torch.float32)
    return q, k, v, g, beta


def verify_scan(ctx: DistributedCtx, req: ScanRequest) -> None:
    cp_group = ctx.device_mesh.get_group("cp")
    cp_rank, cp_size = cp_group.rank(), cp_group.size()
    device = torch.cuda.current_device()

    q, k, v, g, beta = _delta_rule_inputs(req, device)

    # Reference: single-rank full-sequence delta rule.
    qr, kr, vr, gr, br = (t.clone().requires_grad_(True) for t in (q, k, v, g, beta))
    out_ref, _ = torch_chunk_gated_delta_rule(
        qr, kr, vr, g=gr, beta=br, use_qk_l2norm_in_kernel=True
    )
    out_ref.sum().backward()
    ref = {
        "out": extract_contiguous(out_ref, cp_rank, cp_size, dim=1),
        "dq": extract_contiguous(qr.grad, cp_rank, cp_size, dim=1),
        "dk": extract_contiguous(kr.grad, cp_rank, cp_size, dim=1),
        "dv": extract_contiguous(vr.grad, cp_rank, cp_size, dim=1),
        "dg": extract_contiguous(gr.grad, cp_rank, cp_size, dim=1),
        "dbeta": extract_contiguous(br.grad, cp_rank, cp_size, dim=1),
    }

    # Implementation: contiguous shard + cross-rank state scan.
    qi, ki, vi, gi, bi = (
        extract_contiguous(t, cp_rank, cp_size, dim=1).clone().requires_grad_(True)
        for t in (q, k, v, g, beta)
    )
    out_imp = chunk_gated_delta_rule_cp(
        torch_chunk_gated_delta_rule,
        qi, ki, vi, gi, bi,
        cp_group,
        state_shape=(req.B, req.H, req.Dk, req.Dv),
        state_dtype=torch.float32,
        use_qk_l2norm_in_kernel=True,
    )
    out_imp.sum().backward()
    imp = {
        "out": out_imp, "dq": qi.grad, "dk": ki.grad,
        "dv": vi.grad, "dg": gi.grad, "dbeta": bi.grad,
    }

    for name in ref:
        err = cosine_error(ref[name], imp[name])
        if err >= req.atol:
            raise AssertionError(f"scan {name} diverged: {err=:.2e} >= {req.atol=}")


SCAN_REQUESTS = [
    pytest.param(2, ScanRequest(B=1, S=256, H=4, Dk=64, Dv=64), id="CP2-S256-H4"),
    pytest.param(2, ScanRequest(B=2, S=512, H=8, Dk=128, Dv=128), id="CP2-S512-H8"),
    pytest.param(4, ScanRequest(B=1, S=512, H=4, Dk=128, Dv=128), id="CP4-S512-H4"),
]


@pytest.mark.parametrize("cp_size,req", SCAN_REQUESTS)
def test_chunk_gated_delta_rule_cp(cp_size: int, req: ScanRequest) -> None:
    cfg = DistributedCfg()
    cfg.context_parallel_size = cp_size
    launch(cfg, verify_scan, req)
