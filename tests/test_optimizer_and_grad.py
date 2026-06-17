"""CPU-only unit tests for optimizer construction and gradient post-processing.

These cover :mod:`pithtrain.modules.optim`, which is deliberately torch-only so
the tests run without CUDA / Triton / a process group (e.g. on a laptop).
"""

import math

import torch
import torch.nn as nn

from pithtrain.modules.optim import build_param_groups, scale_and_clip_grad_norm_


def _global_grad_norm(model: nn.Module, norm_type: float = 2.0) -> float:
    total = 0.0
    for p in model.parameters():
        if p.grad is not None:
            total += p.grad.detach().abs().pow(norm_type).sum().item()
    return total ** (1.0 / norm_type)


class TinyModel(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.linear = nn.Linear(4, 6)  # weight: 2-D (decay), bias: 1-D (no-decay)
        self.norm = nn.LayerNorm(6)  # weight + bias both 1-D (no-decay)


# --------------------------------------------------------------------------
# build_param_groups
# --------------------------------------------------------------------------


def test_param_groups_split_by_dim():
    model = TinyModel()
    groups = build_param_groups(model, weight_decay=0.1)
    assert len(groups) == 2

    decay = next(g for g in groups if g["weight_decay"] == 0.1)
    no_decay = next(g for g in groups if g["weight_decay"] == 0.0)

    # Only the Linear weight (2-D) is decayed.
    assert decay["params"] == [model.linear.weight]
    # Linear bias + both LayerNorm params (all 1-D) are excluded.
    no_decay_ids = {id(p) for p in no_decay["params"]}
    assert no_decay_ids == {id(model.linear.bias), id(model.norm.weight), id(model.norm.bias)}


def test_param_groups_respect_weight_decay_value():
    model = TinyModel()
    groups = build_param_groups(model, weight_decay=0.37)
    decay = next(g for g in groups if g["params"] == [model.linear.weight])
    assert decay["weight_decay"] == 0.37


def test_param_groups_skip_frozen_params():
    model = TinyModel()
    model.linear.weight.requires_grad_(False)
    groups = build_param_groups(model, weight_decay=0.1)
    # No 2-D trainable params remain -> the decay group is omitted entirely.
    assert all(g["weight_decay"] == 0.0 for g in groups)
    all_param_ids = {id(p) for g in groups for p in g["params"]}
    assert id(model.linear.weight) not in all_param_ids


def test_param_groups_omit_empty_groups():
    # A model with only matrix params -> no "no_decay" group should be created.
    model = nn.Sequential(nn.Linear(3, 3, bias=False))
    groups = build_param_groups(model, weight_decay=0.1)
    assert len(groups) == 1
    assert groups[0]["weight_decay"] == 0.1


def test_optimizer_accepts_param_groups():
    """The groups must be directly consumable by a real optimizer."""
    model = TinyModel()
    groups = build_param_groups(model, weight_decay=0.1)
    opt = torch.optim.AdamW(groups, lr=1e-3, betas=(0.9, 0.95), eps=1e-8)
    assert len(opt.param_groups) == 2
    assert {g["weight_decay"] for g in opt.param_groups} == {0.1, 0.0}


# --------------------------------------------------------------------------
# scale_and_clip_grad_norm_
# --------------------------------------------------------------------------


def _seed_grads(model: nn.Module, scale_factor: float = 1.0) -> None:
    torch.manual_seed(0)
    for p in model.parameters():
        p.grad = torch.randn_like(p) * scale_factor


def test_returns_zero_when_no_grads():
    model = TinyModel()
    out = scale_and_clip_grad_norm_(model, scale=1.0, max_norm=1.0)
    assert out.item() == 0.0


def test_no_scale_no_clip_leaves_grads_unchanged():
    model = TinyModel()
    _seed_grads(model, scale_factor=1e-3)  # tiny grads -> norm well below max_norm
    before = _global_grad_norm(model)
    assert before < 1.0  # precondition: no clipping expected

    returned = scale_and_clip_grad_norm_(model, scale=1.0, max_norm=1.0)
    after = _global_grad_norm(model)

    assert math.isclose(returned.item(), before, rel_tol=1e-5)
    assert math.isclose(after, before, rel_tol=1e-6)  # grads untouched


def test_scale_only_rescales_grads_and_reports_scaled_norm():
    model = TinyModel()
    _seed_grads(model, scale_factor=1e-3)
    unscaled = _global_grad_norm(model)
    scale = 0.25

    returned = scale_and_clip_grad_norm_(model, scale=scale, max_norm=1.0)
    after = _global_grad_norm(model)

    # No clipping (scaled norm still < max_norm): grads multiplied by exactly scale.
    assert math.isclose(after, unscaled * scale, rel_tol=1e-5)
    # Reported norm is the mean-reduced (scaled) norm.
    assert math.isclose(returned.item(), unscaled * scale, rel_tol=1e-5)


def test_clip_only_brings_norm_to_max_and_reports_preclip_norm():
    model = TinyModel()
    _seed_grads(model, scale_factor=1.0)  # large grads -> norm >> max_norm
    unscaled = _global_grad_norm(model)
    max_norm = 0.5
    assert unscaled > max_norm  # precondition: clipping expected

    returned = scale_and_clip_grad_norm_(model, scale=1.0, max_norm=max_norm)
    after = _global_grad_norm(model)

    # After clipping the global norm equals max_norm.
    assert math.isclose(after, max_norm, rel_tol=1e-5)
    # Returned value is the pre-clip norm (scale=1 -> equals unscaled).
    assert math.isclose(returned.item(), unscaled, rel_tol=1e-5)


def test_combined_scale_and_clip_single_pass():
    model = TinyModel()
    _seed_grads(model, scale_factor=1.0)
    unscaled = _global_grad_norm(model)
    scale = 0.5
    max_norm = 0.5
    # Precondition: even after scaling the norm still exceeds max_norm.
    assert unscaled * scale > max_norm

    returned = scale_and_clip_grad_norm_(model, scale=scale, max_norm=max_norm)
    after = _global_grad_norm(model)

    # The combined multiplier (scale * clip_coef) lands the norm exactly at max_norm.
    assert math.isclose(after, max_norm, rel_tol=1e-5)
    # Reported norm is the scaled, pre-clip norm.
    assert math.isclose(returned.item(), unscaled * scale, rel_tol=1e-5)


def test_equivalent_to_naive_two_pass():
    """The fused path must match the original scale-then-clip two-pass logic."""
    scale, max_norm = 0.5, 0.5

    def naive(model):
        for p in model.parameters():
            if p.grad is not None:
                p.grad.mul_(scale)
        total = torch.tensor(_global_grad_norm(model)).clamp(min=1e-6)
        coef = (max_norm / total).clamp(max=1.0)
        if coef < 1.0:
            for p in model.parameters():
                if p.grad is not None:
                    p.grad.mul_(coef)
        return total

    ref = TinyModel()
    _seed_grads(ref, scale_factor=1.0)
    ref_norm = naive(ref)

    fused = TinyModel()
    _seed_grads(fused, scale_factor=1.0)  # same seed -> identical grads
    fused_norm = scale_and_clip_grad_norm_(fused, scale=scale, max_norm=max_norm)

    assert math.isclose(ref_norm.item(), fused_norm.item(), rel_tol=1e-5)
    for pr, pf in zip(ref.parameters(), fused.parameters()):
        assert torch.allclose(pr.grad, pf.grad, atol=1e-7)
