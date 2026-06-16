# Context Parallelism for Qwen3.5 MoE (long-context support)

PithTrain optimises **short-sequence** MoE pretraining (FP8 + DualPipeV overlap) and
historically rejected context parallelism (CP) for Qwen3.5 because its **Gated
DeltaNet** linear-attention layers (30 of 40 layers) had no sequence-parallel path.
Agent / long-context workloads (8k+ tokens) make CP a requirement, so this is the
plan to add it. The work is split into three increments; **only increment 1 is
implemented so far**.

## Sequence layout decision: contiguous shard

The global sequence is split **contiguously** across `cp_size` ranks: rank `r` owns
tokens `[r*S/cp : (r+1)*S/cp]`. This is the natural layout for the Gated DeltaNet
recurrence (state flows strictly left-to-right) at the cost of a load-imbalanced
causal pattern for the full-attention layers (addressed in increment 3). The
alternative — zigzag (reused by the existing `ring_attention.py`) — was rejected
because the recurrent state scan over non-contiguous chunks is far more complex.

Only two operations cross a rank boundary; everything else (projections, norms,
gating, MoE) is token-local and needs no communication.

---

## Increment 1 — Gated DeltaNet CP (DONE)

Files: `pithtrain/operators/gated_delta_rule_cp.py` (new),
`pithtrain/models/qwen3_5_moe.py` (wiring),
`tests/operators/test_gated_delta_rule_cp.py` (tests).

### 1a. Causal conv1d halo exchange — `causal_conv_with_halo`
The depthwise causal conv (kernel `K`) at the first `K-1` positions of a shard needs
the previous rank's last `K-1` tokens. `_LeftHalo` (autograd Function) sends our tail
`K-1` columns to the right neighbour and receives the left halo (zeros at rank 0),
prepends it, runs the conv, and trims. Backward ships the halo-column gradient back
left and adds the borrowed-tail gradient returned from the right. Halo exchange is
fully parallel (it moves conv *inputs*, which are computed locally).

### 1b. Recurrent-state scan — `chunk_gated_delta_rule_cp`
The chunk gated delta rule carries a recurrent state `[B, H, Dk, Dv]`. v1 threads it
**sequentially**: rank `r` receives `initial_state` from `r-1`, runs the kernel with
`output_final_state=True`, and sends the final state to `r+1`.

- Receive side: a plain `recv` + a **tensor hook** that, during backward, ships the
  gradient of the borrowed `initial_state` back to the left neighbour.
- Send side: `_SendFinalState` (autograd Function) sends the final state forward; its
  backward pulls `dstate` from the right and routes it into the kernel's backward.
- The FLA kernel and the torch fallback (`torch_chunk_gated_delta_rule`, extended here
  to accept `initial_state` / `output_final_state` and return `(out, final_state)`)
  share one contract, so tests run without FLA.

Only the small state matrix is communicated (not KV blocks) — the key reason linear
attention CP is cheaper than softmax ring attention.

### Correctness model
The reference is the op on the **full** sequence on one rank; the implementation runs
on each contiguous shard. Outputs and all input gradients (sliced to the shard) must
match. Tests cover `cp_size` 2 and 4 in float32. **Shard length must be a multiple of
the kernel chunk size (64)** so chunk boundaries align with the full-sequence run.

### Safety
Every helper is a no-op when `cp_group` is `None` or size 1, so the existing single-
rank / `cp=1` training path is unchanged. The model still raises a clear
`NotImplementedError` when `cp_size > 1` **and** any `full_attention` layer is present,
until increment 3 lands.

---

## Increment 2 — Selective activation recompute (TODO)

Even with CP, long sequences make activations the dominant memory term, and PithTrain
does **no** activation recompute by default (it would break DualPipeV's
forward/backward overlap if applied naively).

Plan:
- Apply `torch.utils.checkpoint` **selectively** to the compute-heavy, non-overlapped
  regions only: the attention / Gated DeltaNet core and the MoE expert GEMMs.
- Keep the DualPipeV dispatch/combine (stage 2/4 all-to-all) and the overlapped
  forward/backward boundaries **outside** any checkpoint, so the pipeline schedule is
  untouched.
- Gate it behind a config flag (e.g. `training.recompute = "selective" | "none"`) so it
  is opt-in and short-sequence runs keep full overlap.
- Interaction to verify: recompute re-runs the forward in backward, which would re-
  trigger the CP halo/state communication. The recomputed region must **reuse the
  saved cross-rank tensors** (halo, initial_state) rather than re-communicating, or be
  excluded from the checkpoint. This is the main correctness risk and must be tested.

Expected effect: activation memory roughly traded for ~30% extra compute, matching
Megatron's `recompute_granularity: full` lever, but applied only where it does not
disturb overlap.

## Increment 3 — Full-attention ring (TODO)

The 10 `full_attention` layers currently block end-to-end CP. Under the contiguous
layout they need cross-rank attention.

Plan (v1, simplest correct):
- **All-gather K/V** across CP ranks; each rank runs `flash_attn_func(causal=True)` of
  its local Q (the sequence suffix) against the gathered K/V `[0 : (r+1)*S/cp]`. Flash
  attention's bottom-right causal alignment gives the correct mask for a query suffix.
- K/V is tiny (`S x kv_heads x head_dim`), so the gather is cheap; only 10 layers use
  it. Make the gather **differentiable** (gradient is a reduce-scatter), or wrap an
  existing differentiable all-gather.
- **RoPE global positions**: each shard's tokens have global positions
  `[r*S/cp : (r+1)*S/cp]`, so `position_embeddings` must be computed for the global
  offset, not the local one. This is a model-level wiring change.

Plan (v2, optimal): replace the all-gather with a **contiguous causal ring** (KV blocks
rotate, online softmax), reusing the machinery in `ring_attention.py` but with a
contiguous (non-zigzag) chunk assignment and causal masking by global position.

## Increment 3b — Data path: sequence split + loss gather (TODO)

- Split the input token batch `[B, S]` along `S` into `cp_size` contiguous shards
  **before** the embedding (in `pithtrain/tasks/pretrain_lm.py`), each rank keeping its
  shard.
- Compute the per-shard cross-entropy on the local logits and **reduce** across the CP
  group for the global loss (token-weighted), so the loss matches the full-sequence
  value.
- All of this must be guarded by `cp_size > 1` to keep the current training loop
  byte-for-byte unchanged when CP is off.

---

## Optimisation backlog (after correctness)

- **Parallel prefix scan** for the recurrent state: replace the sequential
  rank-by-rank state pass (O(cp) serial) with an associative scan (O(log cp)),
  overlapping each shard's local compute with the cross-rank state combine.
- **Comm/compute overlap** for the halo and state P2P (prefetch, dedicated stream).
- **Zigzag load balancing** for the full-attention ring once correctness is in.
