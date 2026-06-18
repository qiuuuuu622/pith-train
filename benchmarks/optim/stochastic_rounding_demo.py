"""Minimal demo: stochastic rounding lets a bf16 master accumulate sub-ULP updates.

A bf16 master halves the optimizer's master-weight memory (a further ~16GB for the
35B expert state, freeing room for a larger micro-batch). The risk: once the LR has
decayed, each step's update can be smaller than the bf16 ULP (~2^-8 of the weight).
With round-to-nearest writeback those updates round back to the current value and the
weight freezes short of where it should be; stochastic rounding keeps moving it in
expectation.

This isolates exactly that effect (no model, no data noise): drive a bf16 parameter
from 0 toward a constant target with updates deliberately below the bf16 ULP. The
parameter *is* the bf16 master, exercising ForeachOffloadAdamW's real writeback.

  - fp32 ref     : fp32 param + torch.optim.AdamW   -> reaches the target
  - bf16 RN      : ForeachOffloadAdamW, round-to-nearest -> freezes ~1 ULP short
  - bf16 SR      : ForeachOffloadAdamW, stochastic_rounding=True -> reaches the target

(Note: stochastic rounding is not free -- on a task with a clean optimum its injected
noise can raise the loss floor. It pays off precisely when the model must keep making
small persistent progress, i.e. real LM training, which is the bf16-master use case.)

Run: PYTHONPATH=. python -m benchmarks.optim.stochastic_rounding_demo
"""

import torch

from pithtrain.modules.optim import ForeachOffloadAdamW

D, TARGET, LR, STEPS = 512, 3.0, 3e-4, 4000


def _run(bf16, make_opt):
    torch.manual_seed(0)
    dtype = torch.bfloat16 if bf16 else torch.float32
    w = torch.zeros(D, dtype=dtype, requires_grad=True)
    opt = make_opt([w])
    for _ in range(STEPS):
        # loss = 0.5 * sum((w - TARGET)^2); grad = (w - TARGET).
        w.grad = (w.detach().float() - TARGET).to(dtype)
        opt.step()
    return w.detach().float()


cfg = dict(lr=LR, betas=(0.9, 0.95), eps=1e-8, weight_decay=0.0)
runs = {
    "fp32 ref": _run(False, lambda p: torch.optim.AdamW(p, **cfg)),
    "bf16 RN": _run(True, lambda p: ForeachOffloadAdamW(p, moment_dtype=torch.bfloat16, **cfg)),
    "bf16 SR": _run(
        True,
        lambda p: ForeachOffloadAdamW(
            p, moment_dtype=torch.bfloat16, stochastic_rounding=True, **cfg
        ),
    ),
}

print(f"target = {TARGET}, {STEPS} steps, lr {LR} (update/step << bf16 ULP near target)\n")
print(f"{'optimizer':>10} | {'mean w':>9} | {'gap to target':>14}")
for name, w in runs.items():
    gap = (TARGET - w).mean().item()
    print(f"{name:>10} | {w.mean().item():>9.4f} | {gap:>14.5f}")
print("\nbf16 RN should freeze ~1 bf16 ULP short; bf16 SR should reach the target.")
