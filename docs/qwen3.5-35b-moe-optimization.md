# Qwen3.5-35B-A3B MoE training-efficiency optimization (8×H100)

Goal: train the Qwen3.5-35B-A3B MoE (GDN-hybrid, 256 experts top-8, hidden 2048,
40 layers, vocab 248320) on a single 8×H100 node with higher training efficiency
than Megatron. Method: profile with nsys, attack the dominant bottleneck, verify
every change with a measurement before keeping it.

## Headline result

| seq7168, GBS64, PP4×EP2, no TP | pith-train | Megatron (seq8192 baseline) |
|---|---|---|
| **tok/s/gpu** | **2,773–2,920** | 1,311 |
| **MFU** | **4.8–5.0%** | 2.5% |
| optimizer step | 1.7s | — |
| peak GPU mem | 69.82 GB | — |
| grad-norm | 15.5 (stable) | — |

**MFU is the hardware-independent metric: 5.0% vs Megatron's 2.5% ≈ 2×.**
tok/s/gpu 2,920 ≈ 2.2× Megatron's 1,311.

Honesty caveats:
- seq=7168, not 8192. seq8192 fits at ~72.5 GB peak which OOMs by ~2–3 GB on the
  torch-2.11 box used for these numbers (it ran at 71.99 GB on the earlier torch-2.12
  box). The HBM-resident optimizer state is the tight resource.
- Megatron's 1,311 / 2.5% was measured on a different (slower) box; MFU normalizes
  for hardware so the 2× MFU gap is the reliable claim. tok/s/gpu is not strictly
  apples-to-apples across boxes.
- The Megatron comparison model is an all-attention MoE of matched parameter count;
  pith-train's model is GDN-hybrid (30 linear-attention + 10 full-attention layers).

## What actually moved the needle (measured)

### 1. Precision-aware HBM-resident optimizer — the breakthrough (37s → 1.7s)

nsys per-step bucketing showed the GPU idle ~80% of a ~50s step, with one ~39s
window of zero GPU/copy/NCCL activity. `OPT_TIMING` attributed it to
`optimizer.step()`: under FSDP2 `CPUOffloadPolicy` the AdamW state (~48 GB/rank of
fp32 master+exp_avg+exp_avg_sq) lives on the host and the step is bandwidth-bound on
a 32-core box shared by 8 ranks. **The exposed CPU optimizer step was 78% of the step.**

Fix: store the AdamW moments in **bf16** (`precision_aware_optimizer`) so the state
shrinks 48 GB → 32 GB and fits in HBM with `expert_cpu_offload` off; the math then
runs on HBM. The fp32 master is kept and the update is computed in fp32 via a
per-tensor upcast — numerically identical to `torch.optim.AdamW`. Measured:
**optimizer step 37s → 1.7s (22×)**, grad-norm unchanged.

Supporting pieces:
- `ForeachOffloadAdamW` (`pithtrain/modules/optim.py`): runs `torch._foreach_*` on
  `to_local()` DTensor shards, bypassing torch.optim's single-tensor path (which
  rejects FSDP-offloaded DTensors and sweeps the state ~12×).
- `offload_head`: offload only the large embed / lm_head optimizer state to host to
  relieve the edge pipeline stages (rank 0 holds both under DualPipeV's V-shape).
- device-robust `scale_and_clip_grad_norm_` (grads span CPU+GPU), DualPipeV init
  device-assert relaxed to accept CPU-offloaded shards.

### 2. GBS16 → GBS64 — fill the pipeline (MFU 3.0% → 5.0%)

Once the optimizer was cheap, nsys showed the step was communication-bound, with the
largest single item being **PP pipeline P2P wait** (a fill/drain bubble: at GBS16,
PP4 DualPipeV doesn't have enough micro-batches to fill the pipeline). Raising the
global batch GBS16 → GBS64 fills the pipeline and amortizes the bubble — **at the
same per-micro-batch memory peak (69.82 GB)**, so it still fits.

Result: **MFU 3.0% → 5.0%, tok/s/gpu 1,766 → 2,920.** No layer-rebalance needed; the
bubble was fill/drain, not stage imbalance.

## Supporting work (earlier)

- Fixed the triton cross-entropy **gradient bug** (cos 0.076 vs F.cross_entropy);
  replaced with a correct chunked online-softmax impl.
- **Fused Linear Cross-Entropy**: never materialize the [N, V] logits.
- **Layer-level activation recompute** (mlp + attn) to fit long sequences.
- **NCCL sub-group timeout fix**: `init_device_mesh`'s pp/dp/cp/ep sub-groups used
  NCCL's hardcoded 10-min default, so a ~10-min cold first-compile tripped the
  watchdog; now they inherit `DistributedCfg.timeout` (default raised to 40 min).
- **Stochastic rounding** for bf16 writeback — the prerequisite for a future bf16
  *master* (a further ~16 GB), implemented and validated (CPU) but not yet used in
  the winning config.

## Dead ends (kept honest, ruled out by measurement)

- **`compute_device` (stream offloaded state to GPU, compute there)**: measured
  **47s — slower than the 37s CPU step**. It moves the whole 48 GB state over PCIe
  each step (~6× Megatron's grad/param-only traffic), so it is PCIe-bound. Megatron's
  HybridDeviceOptimizer instead keeps state resident on CPU and only transfers
  grads/params. Kept as an option but not the win for this hardware.
- **Load-balance all-reduce reduction**: A/B (`LB_TYPE=micro-batch` vs
  `global-batch`) showed no step-time change three times — the per-layer f32
  all-reduce is already overlapped. Not implemented.
- **Layer rebalance** (edge stages lighter): unnecessary after GBS64; also OOM'd
  earlier because every rank sits near the memory ceiling.

## Winning configuration

```
OFFLOAD=0 PRECISION_AWARE_OPT=1 OFFLOAD_HEAD=1 FUSE_CE=1 FUSE_CHUNKS=32
RECOMPUTE_MLP=1 RECOMPUTE_ATTN=1 PP=4 EP=2 SEQ=7168 MBS=1 GBS=64
```

(`examples/pretrain_lm/qwen3.5-35b-a3b/script.py` defaults to the precision-aware
HBM-resident config; GBS and SEQ are env-overridable.)

## Lesson

The whole win came from nsys correctly identifying that the *optimizer step* — not
communication or recompute — was 78% of the step, and from measurement-first
discipline that killed three plausible-but-useless optimizations before they shipped.
