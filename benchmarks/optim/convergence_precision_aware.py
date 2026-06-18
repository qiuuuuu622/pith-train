"""Convergence validation for ForeachOffloadAdamW + bf16 moments (CPU, no GPU).

Trains a tiny transformer LM on a synthetic copy task with THREE optimizers from
identical init and compares loss trajectories:

  1. torch.optim.AdamW (fp32)            -- reference
  2. ForeachOffloadAdamW (fp32 moments)  -- must match (1) ~exactly (validates the
     foreach-on-local-shard step is numerically equivalent in real training, not
     just the static unit test)
  3. ForeachOffloadAdamW (bf16 moments)  -- the precision-aware path used to keep
     the 35B expert optimizer state in HBM. Checks that bf16 exp_avg_sq does not
     underflow into divergence / plateau.

Run:
    PYTHONPATH=. python -m benchmarks.optim.convergence_precision_aware

Recorded result (600 steps, this proxy): Foreach-fp32 final-50 avg matches the
reference to +0.0000; Foreach-bf16mom tracks within -0.0012 (~0.06%) with no
divergence. This is strong (not absolute) evidence -- bf16 moment risk is most
likely to surface at large scale / tens of thousands of steps / late LR decay,
which needs a long GPU run to fully confirm.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F

from pithtrain.modules.optim import ForeachOffloadAdamW

V, D, L, H, SEQ, B, STEPS = 64, 128, 3, 4, 24, 32, 600


class TinyGPT(nn.Module):
    def __init__(self):
        super().__init__()
        self.emb = nn.Embedding(V, D)
        self.pos = nn.Embedding(SEQ, D)
        self.blocks = nn.ModuleList(
            [
                nn.TransformerEncoderLayer(D, H, 4 * D, batch_first=True, dropout=0.0)
                for _ in range(L)
            ]
        )
        self.norm = nn.LayerNorm(D)
        self.head = nn.Linear(D, V)

    def forward(self, x):
        h = self.emb(x) + self.pos(torch.arange(x.shape[1]))
        mask = nn.Transformer.generate_square_subsequent_mask(x.shape[1])
        for b in self.blocks:
            h = b(h, src_mask=mask)
        return self.head(self.norm(h))


def _batch():
    # Copy task: a random pattern repeated twice; predict the next token.
    pat = torch.randint(0, V, (B, SEQ // 2))
    seq = torch.cat([pat, pat], dim=1)
    return seq[:, :-1], seq[:, 1:]


def _train(make_opt, steps=STEPS):
    torch.manual_seed(0)  # identical init across optimizers
    model = TinyGPT()
    opt = make_opt(model)
    losses = []
    for s in range(steps):
        torch.manual_seed(1000 + s)  # identical data stream across optimizers
        x, y = _batch()
        loss = F.cross_entropy(model(x).reshape(-1, V), y.reshape(-1))
        opt.zero_grad()
        loss.backward()
        opt.step()
        losses.append(loss.item())
    return losses


def main():
    cfg = dict(lr=3e-3, betas=(0.9, 0.95), eps=1e-8, weight_decay=0.1)
    runs = {
        "AdamW-fp32(ref)": lambda m: torch.optim.AdamW(m.parameters(), **cfg),
        "Foreach-fp32": lambda m: ForeachOffloadAdamW(list(m.parameters()), **cfg),
        "Foreach-bf16mom": lambda m: ForeachOffloadAdamW(
            list(m.parameters()), moment_dtype=torch.bfloat16, **cfg
        ),
    }
    res = {name: _train(mk) for name, mk in runs.items()}

    print(f"{'step':>5} | " + " | ".join(f"{n:>16}" for n in runs))
    for s in [0, 50, 100, 200, 300, 400, 500, STEPS - 1]:
        print(f"{s:>5} | " + " | ".join(f"{res[n][s]:>16.4f}" for n in runs))

    ref_tail = sum(res["AdamW-fp32(ref)"][-50:]) / 50
    for n in ["Foreach-fp32", "Foreach-bf16mom"]:
        tail = sum(res[n][-50:]) / 50
        print(f"\n{n}: final-50-avg {tail:.4f} vs ref {ref_tail:.4f}  (delta {tail - ref_tail:+.4f})")


if __name__ == "__main__":
    main()
