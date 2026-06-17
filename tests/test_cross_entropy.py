"""CPU-only correctness tests for the in-place fused cross-entropy.

Guards against the silent gradient bug the previous Triton kernel had: the loss
was correct but the gradient was wrong once enough rows ran concurrently. These
assert both loss and the (in-place) gradient against ``F.cross_entropy``.
"""

import pytest
import torch
import torch.nn.functional as F

from pithtrain.operators.cross_entropy import cross_entropy


def _ref_loss_and_grad(logits, target, ignore_index=-100):
    x = logits.clone().requires_grad_(True)
    loss = F.cross_entropy(x.float(), target, ignore_index=ignore_index, reduction="mean")
    loss.backward()
    return loss.detach(), x.grad.detach()


@pytest.mark.parametrize("N,V", [(2, 8), (8, 64), (64, 64), (512, 64), (512, 4096)])
def test_loss_and_grad_match_reference(N, V):
    torch.manual_seed(N * V)
    logits = torch.randn(N, V, dtype=torch.float32)
    target = torch.randint(0, V, (N,))
    ref_loss, ref_grad = _ref_loss_and_grad(logits, target)

    inp = logits.clone()  # overwritten in place with the gradient
    loss = cross_entropy(inp, target, ignore_index=-100)

    assert torch.allclose(loss, ref_loss, rtol=1e-5, atol=1e-5)
    # inp now holds d loss / d logits.
    cos = F.cosine_similarity(inp.flatten(), ref_grad.flatten(), dim=0)
    assert cos > 0.9999, f"grad direction off: cos={cos.item()}"
    assert torch.allclose(inp, ref_grad, rtol=1e-3, atol=1e-6)


def test_backward_scales_stored_grad():
    torch.manual_seed(0)
    logits = torch.randn(32, 100, dtype=torch.float32)
    target = torch.randint(0, 100, (32,))
    ref_loss, ref_grad = _ref_loss_and_grad(logits, target)

    x = logits.clone().requires_grad_(True)
    cross_entropy(x, target, ignore_index=-100).backward()
    assert torch.allclose(x.grad, ref_grad, rtol=1e-3, atol=1e-6)


def test_ignore_index():
    torch.manual_seed(1)
    N, V = 40, 128
    logits = torch.randn(N, V, dtype=torch.float32)
    target = torch.randint(0, V, (N,))
    target[::5] = -100  # ignore every 5th row
    ref_loss, ref_grad = _ref_loss_and_grad(logits, target)

    inp = logits.clone()
    loss = cross_entropy(inp, target, ignore_index=-100)
    assert torch.allclose(loss, ref_loss, rtol=1e-5, atol=1e-5)
    assert torch.allclose(inp, ref_grad, rtol=1e-3, atol=1e-6)
    # Ignored rows get zero gradient.
    assert torch.allclose(inp[::5], torch.zeros_like(inp[::5]))


def test_chunk_count_invariance():
    torch.manual_seed(2)
    logits = torch.randn(50, 200, dtype=torch.float32)
    target = torch.randint(0, 200, (50,))
    outs = []
    for nc in (1, 3, 8, 50):
        inp = logits.clone()
        cross_entropy(inp, target, ignore_index=-100, num_chunks=nc)
        outs.append(inp)
    for o in outs[1:]:
        assert torch.allclose(outs[0], o, atol=1e-6)
