"""Increment 3b: contiguous sequence split + CP loss gather.

Verifies two things:
1. ``split_sequence_for_cp`` correctly shards a [B, S] token tensor along S so
   each rank holds tokens [r*S/cp : (r+1)*S/cp] and a matching label shard.
2. ``reduce_loss_across_cp`` produces a scalar loss that matches the loss
   computed on the full sequence in one shot (token-weighted mean).

Both helpers live in ``pithtrain.tasks.pretrain_lm``.  For ``cp_size == 1``
everything must be a no-op so the existing single-rank path is unchanged.
"""

from dataclasses import dataclass

import pytest
import torch
import torch.distributed as dist

from pithtrain.modules.distributed import DistributedCfg, DistributedCtx
from pithtrain.tasks.pretrain_lm import reduce_loss_across_cp, split_sequence_for_cp
from tests.utilities import cosine_error, launch


@dataclass
class DataPathRequest:
    B: int
    S: int
    V: int  # vocab size, determines logit width for loss test
    atol: float = 1e-5


# ---------------------------------------------------------------------------
# 1. Sequence split correctness
# ---------------------------------------------------------------------------


def verify_split(ctx: DistributedCtx, req: DataPathRequest) -> None:
    cp_group = ctx.device_mesh.get_group("cp")
    cp_rank = cp_group.rank()
    cp_size = cp_group.size()
    device = torch.cuda.current_device()

    torch.manual_seed(0)
    tokens_full = torch.randint(0, req.V, (req.B, req.S), device=device)
    labels_full = torch.randint(0, req.V, (req.B, req.S), device=device)

    tokens_local, labels_local = split_sequence_for_cp(tokens_full, labels_full, cp_group)

    shard_len = req.S // cp_size
    expected_tokens = tokens_full[:, cp_rank * shard_len : (cp_rank + 1) * shard_len]
    expected_labels = labels_full[:, cp_rank * shard_len : (cp_rank + 1) * shard_len]

    assert tokens_local.shape == (req.B, shard_len), (
        f"tokens shape {tokens_local.shape}, expected ({req.B}, {shard_len})"
    )
    assert torch.equal(tokens_local, expected_tokens), "tokens shard mismatch"
    assert torch.equal(labels_local, expected_labels), "labels shard mismatch"


# ---------------------------------------------------------------------------
# 2. Loss reduce correctness
# ---------------------------------------------------------------------------


def verify_loss_reduce(ctx: DistributedCtx, req: DataPathRequest) -> None:
    cp_group = ctx.device_mesh.get_group("cp")
    cp_rank = cp_group.rank()
    cp_size = cp_group.size()
    device = torch.cuda.current_device()

    shard_len = req.S // cp_size
    n_tokens_per_shard = req.B * shard_len

    # Each rank holds a different slice of logits/labels.
    torch.manual_seed(cp_rank + 1)
    logits_local = torch.randn(n_tokens_per_shard, req.V, device=device, dtype=torch.float32)
    labels_local = torch.randint(0, req.V, (n_tokens_per_shard,), device=device)

    # Reference: gather all logits and labels across CP ranks and compute loss globally.
    # We all-gather so every rank has the full tensor for reference.
    logits_all = [torch.zeros_like(logits_local) for _ in range(cp_size)]
    labels_all = [torch.zeros_like(labels_local) for _ in range(cp_size)]
    dist.all_gather(logits_all, logits_local.detach(), group=cp_group)
    dist.all_gather(labels_all, labels_local, group=cp_group)
    logits_full = torch.cat(logits_all, dim=0)  # [B*S, V]
    labels_full = torch.cat(labels_all, dim=0)  # [B*S]

    # Reference loss: standard cross-entropy on full sequence.
    loss_ref = torch.nn.functional.cross_entropy(logits_full, labels_full)

    # Implementation: per-shard loss reduced across CP.
    loss_shard = torch.nn.functional.cross_entropy(logits_local, labels_local)
    # Pass n_tokens_per_shard so the helper can weight by local token count.
    loss_global = reduce_loss_across_cp(loss_shard, n_tokens_per_shard, cp_group)

    err = abs(loss_global.item() - loss_ref.item())
    assert err < req.atol, f"loss mismatch: {loss_global.item()=:.6f} vs {loss_ref.item()=:.6f}, {err=:.2e}"


# ---------------------------------------------------------------------------
# 3. cp_size==1 no-op
# ---------------------------------------------------------------------------


def verify_noop(ctx: DistributedCtx, req: DataPathRequest) -> None:
    cp_group = ctx.device_mesh.get_group("cp")
    device = torch.cuda.current_device()

    torch.manual_seed(42)
    tokens = torch.randint(0, req.V, (req.B, req.S), device=device)
    labels = torch.randint(0, req.V, (req.B, req.S), device=device)

    t_out, l_out = split_sequence_for_cp(tokens, labels, cp_group)
    assert torch.equal(t_out, tokens), "no-op: tokens should be unchanged"
    assert torch.equal(l_out, labels), "no-op: labels should be unchanged"

    logits = torch.randn(req.B * req.S, req.V, device=device)
    tgt = labels.view(-1)
    loss = torch.nn.functional.cross_entropy(logits, tgt)
    loss_out = reduce_loss_across_cp(loss, req.B * req.S, cp_group)
    err = abs(loss_out.item() - loss.item())
    assert err < 1e-6, f"no-op loss changed: {err=:.2e}"


SPLIT_REQUESTS = [
    pytest.param(2, DataPathRequest(B=2, S=512, V=256), id="CP2-B2-S512"),
    pytest.param(4, DataPathRequest(B=1, S=1024, V=512), id="CP4-B1-S1024"),
]

REDUCE_REQUESTS = [
    pytest.param(2, DataPathRequest(B=2, S=512, V=256), id="CP2-B2-S512"),
    pytest.param(4, DataPathRequest(B=1, S=1024, V=512), id="CP4-B1-S1024"),
]

NOOP_REQUESTS = [
    pytest.param(1, DataPathRequest(B=2, S=128, V=64), id="CP1-noop"),
]


@pytest.mark.parametrize("cp_size,req", SPLIT_REQUESTS)
def test_split_sequence_for_cp(cp_size: int, req: DataPathRequest) -> None:
    cfg = DistributedCfg()
    cfg.context_parallel_size = cp_size
    launch(cfg, verify_split, req)


@pytest.mark.parametrize("cp_size,req", REDUCE_REQUESTS)
def test_reduce_loss_across_cp(cp_size: int, req: DataPathRequest) -> None:
    cfg = DistributedCfg()
    cfg.context_parallel_size = cp_size
    launch(cfg, verify_loss_reduce, req)


@pytest.mark.parametrize("cp_size,req", NOOP_REQUESTS)
def test_noop_cp1(cp_size: int, req: DataPathRequest) -> None:
    cfg = DistributedCfg()
    cfg.context_parallel_size = cp_size
    launch(cfg, verify_noop, req)
