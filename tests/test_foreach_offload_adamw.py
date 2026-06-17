"""Numerical-equivalence test for ForeachOffloadAdamW vs torch.optim.AdamW.

Runs on CPU (no DTensor / GPU needed): the foreach-on-local update must match
the reference AdamW to within fp32 round-off over many steps and param groups.
"""

import copy

import torch

from pithtrain.modules.optim import ForeachOffloadAdamW


def _make_params(seed):
    torch.manual_seed(seed)
    # Mixed shapes incl. a stacked-expert-like 3D tensor and 1D norms.
    shapes = [(256, 512), (8, 384, 192), (512,), (1024, 256), (32,)]
    return [torch.randn(s, requires_grad=True) for s in shapes]


def test_matches_torch_adamw():
    ref_params = _make_params(0)
    ours_params = [p.detach().clone().requires_grad_(True) for p in ref_params]

    cfg = dict(lr=3e-4, betas=(0.9, 0.95), eps=1e-8, weight_decay=0.1)
    ref = torch.optim.AdamW(ref_params, **cfg)
    ours = ForeachOffloadAdamW(ours_params, **cfg)

    torch.manual_seed(123)
    for _ in range(25):
        grads = [torch.randn_like(p) for p in ref_params]
        for p, g in zip(ref_params, grads):
            p.grad = g.clone()
        for p, g in zip(ours_params, grads):
            p.grad = g.clone()
        ref.step()
        ours.step()
        for rp, op in zip(ref_params, ours_params):
            torch.testing.assert_close(op, rp, rtol=1e-5, atol=1e-6)


def test_bf16_moments_close_to_fp32():
    # Precision-aware path (bf16 moments, fp32 master + math) must track the
    # fp32 reference closely -- only bf16 rounding of the stored moments differs.
    ref_params = _make_params(3)
    ours_params = [p.detach().clone().requires_grad_(True) for p in ref_params]
    cfg = dict(lr=3e-4, betas=(0.9, 0.95), eps=1e-8, weight_decay=0.1)
    ref = torch.optim.AdamW(ref_params, **cfg)
    ours = ForeachOffloadAdamW(ours_params, moment_dtype=torch.bfloat16, **cfg)
    torch.manual_seed(99)
    for _ in range(25):
        for rp, op in zip(ref_params, ours_params):
            g = torch.randn_like(rp)
            rp.grad = g.clone()
            op.grad = g.clone()
        ref.step()
        ours.step()
    # Master stays fp32; bf16 moment rounding only perturbs the update slightly.
    for rp, op in zip(ref_params, ours_params):
        assert op.dtype == torch.float32
        torch.testing.assert_close(op, rp, rtol=3e-2, atol=2e-3)


def test_param_groups_distinct_wd():
    # decay group + no-decay group, like build_param_groups.
    ps = _make_params(7)
    ref_groups = [
        {"params": [ps[0].detach().clone().requires_grad_(True),
                    ps[1].detach().clone().requires_grad_(True)], "weight_decay": 0.1},
        {"params": [ps[2].detach().clone().requires_grad_(True)], "weight_decay": 0.0},
    ]
    ours_groups = copy.deepcopy(ref_groups)
    common = dict(lr=1e-3, betas=(0.9, 0.95), eps=1e-8)
    ref = torch.optim.AdamW(ref_groups, **common)
    ours = ForeachOffloadAdamW(ours_groups, **common)
    torch.manual_seed(5)
    for _ in range(15):
        for grp_r, grp_o in zip(ref_groups, ours_groups):
            for pr, po in zip(grp_r["params"], grp_o["params"]):
                g = torch.randn_like(pr)
                pr.grad = g.clone()
                po.grad = g.clone()
        ref.step()
        ours.step()
        for grp_r, grp_o in zip(ref_groups, ours_groups):
            for pr, po in zip(grp_r["params"], grp_o["params"]):
                torch.testing.assert_close(po, pr, rtol=1e-5, atol=1e-6)
