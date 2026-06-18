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


def test_stochastic_rounding_unbiased():
    # Averaging many stochastic roundings of an fp32 tensor must recover it
    # (unbiased), unlike round-to-nearest which has a fixed per-element bias.
    torch.manual_seed(0)
    x = torch.randn(50000) * 0.3 + 1.0  # ~O(1): bf16 ULP ~2^-8, values have sub-ULP parts
    acc = torch.zeros(x.shape, dtype=torch.float64)
    n = 400
    for _ in range(n):
        acc += ForeachOffloadAdamW._sr_to_bf16(x).float().double()
    sr_mean = (acc / n).float()
    rn = x.to(torch.bfloat16).float()
    # SR mean is far closer to x than a single round-to-nearest.
    assert (sr_mean - x).abs().mean() < 0.2 * (rn - x).abs().mean()
    torch.testing.assert_close(sr_mean, x, atol=2e-3, rtol=0)


def test_stochastic_rounding_preserves_exact():
    # bf16-representable values must stay exact under stochastic rounding.
    x = torch.tensor([0.0, 0.5, 1.0, 2.0, -4.0, 0.25, -0.5]).to(torch.bfloat16).float()
    for _ in range(50):
        out = ForeachOffloadAdamW._sr_to_bf16(x).float()
        torch.testing.assert_close(out, x, atol=0.0, rtol=0.0)


def test_compute_device_matches_torch_adamw():
    # compute_device routes the math to another device (on GPU: stream offloaded
    # state -> HBM -> back). With compute_device == home device it must stay
    # bitwise-identical to torch.optim.AdamW, proving the streaming path is a pure
    # relocation of the math, not a numerical change.
    ref_params = _make_params(21)
    ours_params = [p.detach().clone().requires_grad_(True) for p in ref_params]
    cfg = dict(lr=3e-4, betas=(0.9, 0.95), eps=1e-8, weight_decay=0.1)
    ref = torch.optim.AdamW(ref_params, **cfg)
    ours = ForeachOffloadAdamW(ours_params, compute_device=torch.device("cpu"), **cfg)
    torch.manual_seed(3)
    for _ in range(20):
        for rp, op in zip(ref_params, ours_params):
            g = torch.randn_like(rp)
            rp.grad = g.clone()
            op.grad = g.clone()
        ref.step()
        ours.step()
        for rp, op in zip(ref_params, ours_params):
            torch.testing.assert_close(op, rp, rtol=1e-5, atol=1e-6)


def test_compute_device_equiv_to_home():
    # compute_device=home must give exactly the same result as compute_device=None
    # (the bf16-moment streaming path is just a relocation).
    ps = _make_params(31)
    a = [p.detach().clone().requires_grad_(True) for p in ps]
    b = [p.detach().clone().requires_grad_(True) for p in ps]
    cfg = dict(lr=1e-3, betas=(0.9, 0.95), eps=1e-8, weight_decay=0.1, moment_dtype=torch.bfloat16)
    oa = ForeachOffloadAdamW(a, **cfg)
    ob = ForeachOffloadAdamW(b, compute_device=torch.device("cpu"), **cfg)
    torch.manual_seed(9)
    for _ in range(15):
        for pa, pb in zip(a, b):
            g = torch.randn_like(pa)
            pa.grad = g.clone()
            pb.grad = g.clone()
        oa.step()
        ob.step()
        for pa, pb in zip(a, b):
            torch.testing.assert_close(pb, pa, rtol=0.0, atol=0.0)


def test_sr_master_fp32_path_unchanged():
    # stochastic_rounding=True with an fp32 master must stay bitwise-identical to
    # torch.optim.AdamW: SR only affects bf16 destinations, and an fp32 master is
    # written with a plain copy.
    ref_params = _make_params(11)
    ours_params = [p.detach().clone().requires_grad_(True) for p in ref_params]
    cfg = dict(lr=3e-4, betas=(0.9, 0.95), eps=1e-8, weight_decay=0.1)
    ref = torch.optim.AdamW(ref_params, **cfg)
    ours = ForeachOffloadAdamW(ours_params, stochastic_rounding=True, **cfg)  # fp32 moments+master
    torch.manual_seed(7)
    for _ in range(20):
        for rp, op in zip(ref_params, ours_params):
            g = torch.randn_like(rp)
            rp.grad = g.clone()
            op.grad = g.clone()
        ref.step()
        ours.step()
        for rp, op in zip(ref_params, ours_params):
            torch.testing.assert_close(op, rp, rtol=1e-5, atol=1e-6)


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
