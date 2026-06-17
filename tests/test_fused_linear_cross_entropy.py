"""CPU-only numerical equivalence tests for fused linear cross-entropy.

Compares :func:`fused_linear_cross_entropy` against the reference
``F.linear`` + ``F.cross_entropy`` (mean reduction) on both loss and the
hidden/weight gradients, including ignored rows and chunk-count invariance.
"""

import pytest
import torch
import torch.nn.functional as F

from pithtrain.operators.fused_linear_cross_entropy import fused_linear_cross_entropy


def _reference(hidden, weight, target, ignore_index):
    logits = F.linear(hidden, weight)
    loss = F.cross_entropy(
        logits.float(), target, ignore_index=ignore_index, reduction="mean"
    )
    return loss


def _make(N=24, H=16, V=50, seed=0, ignore_rows=()):
    torch.manual_seed(seed)
    hidden = torch.randn(N, H, dtype=torch.float32)
    weight = torch.randn(V, H, dtype=torch.float32) * 0.1
    target = torch.randint(0, V, (N,))
    for r in ignore_rows:
        target[r] = -100
    return hidden, weight, target


@pytest.mark.parametrize("num_chunks", [1, 3, 8])
def test_loss_matches_reference(num_chunks):
    hidden, weight, target = _make()
    ref = _reference(hidden, weight, target, -100)
    got = fused_linear_cross_entropy(hidden, weight, target, -100, num_chunks)
    assert torch.allclose(ref, got, rtol=1e-5, atol=1e-5), (ref.item(), got.item())


@pytest.mark.parametrize("num_chunks", [1, 4])
def test_grads_match_reference(num_chunks):
    hidden, weight, target = _make(seed=1)

    h_ref = hidden.clone().requires_grad_(True)
    w_ref = weight.clone().requires_grad_(True)
    _reference(h_ref, w_ref, target, -100).backward()

    h_f = hidden.clone().requires_grad_(True)
    w_f = weight.clone().requires_grad_(True)
    fused_linear_cross_entropy(h_f, w_f, target, -100, num_chunks).backward()

    assert torch.allclose(h_ref.grad, h_f.grad, rtol=1e-4, atol=1e-5)
    assert torch.allclose(w_ref.grad, w_f.grad, rtol=1e-4, atol=1e-5)


def test_ignore_index_rows():
    # Several ignored rows, including a whole-tile case to exercise the masking.
    hidden, weight, target = _make(N=20, seed=2, ignore_rows=(0, 1, 7, 19))
    ref = _reference(hidden, weight, target, -100)
    got = fused_linear_cross_entropy(hidden, weight, target, -100, num_chunks=4)
    assert torch.allclose(ref, got, rtol=1e-5, atol=1e-5)

    h_ref = hidden.clone().requires_grad_(True)
    w_ref = weight.clone().requires_grad_(True)
    _reference(h_ref, w_ref, target, -100).backward()
    h_f = hidden.clone().requires_grad_(True)
    w_f = weight.clone().requires_grad_(True)
    fused_linear_cross_entropy(h_f, w_f, target, -100, num_chunks=4).backward()
    assert torch.allclose(h_ref.grad, h_f.grad, rtol=1e-4, atol=1e-5)
    # Ignored rows contribute zero gradient to the hidden states.
    for r in (0, 1, 7, 19):
        assert torch.allclose(h_f.grad[r], torch.zeros_like(h_f.grad[r]))


def test_chunk_count_invariance():
    hidden, weight, target = _make(N=30, seed=3)
    losses = [
        fused_linear_cross_entropy(hidden, weight, target, -100, nc).item()
        for nc in (1, 2, 5, 30)
    ]
    assert max(losses) - min(losses) < 1e-5


def test_no_weight_grad_path():
    hidden, weight, target = _make(seed=4)
    h = hidden.clone().requires_grad_(True)
    w = weight.clone()  # requires_grad=False -> grad_weight branch skipped
    loss = fused_linear_cross_entropy(h, w, target, -100, num_chunks=4)
    loss.backward()
    assert h.grad is not None
    assert w.grad is None


def test_all_ignored_is_safe():
    # Every row ignored -> loss 0, finite grads (n_nonignore clamped to 1).
    hidden, weight, target = _make(N=8, seed=5)
    target[:] = -100
    loss = fused_linear_cross_entropy(hidden, weight, target, -100, num_chunks=2)
    assert torch.isfinite(loss) and loss.item() == 0.0
