from typing import Literal

import torch.nn as nn


class ModelImplMode:
    """
    Model Implementation Mode. Turn on the reference implementation to validate
    the correctness of the optimized, possibly distributed implementation.
    """

    use_reference_fwd = False
    fp8_training: Literal["deep-gemm", "disabled"] = "disabled"
    # Fuse the lm_head projection with the cross-entropy loss so the [N, V]
    # logits are never materialized (see operators/fused_linear_cross_entropy).
    fuse_lmhead_ce: bool = False
    fuse_lmhead_ce_chunks: int = 8
    # Activation recompute: don't save the expert-MLP forward activations; recompute
    # them in backward (torch.utils.checkpoint). Trades ~1 extra MLP forward for the
    # largest single activation tensor on each rank. Compatible with the DualPipeV
    # deferred-wgrad path (the recomputed activations feed the deferred wgrad closure).
    recompute_mlp: bool = False
    # Recompute the attention/GatedDeltaNet block (`_forward_attn_compute`). This is
    # the dominant activation at long sequence length (the GDN scan + full attention).
    # Only the compute is checkpointed -- the router stays outside so the load-balance
    # aux loss is not double-counted on recompute. Needed to fit long sequences.
    recompute_attn: bool = False


def get_linear_cls():
    """Return the appropriate Linear class based on ModelImplMode.fp8_training."""
    if ModelImplMode.fp8_training == "deep-gemm":
        from pithtrain.layers.deepgemm_fp8_linear import FP8Linear

        return FP8Linear
    return nn.Linear


def get_group_linear_cls():
    """Return the appropriate GroupLinear class based on ModelImplMode.fp8_training."""
    if ModelImplMode.fp8_training == "deep-gemm":
        from pithtrain.layers.deepgemm_fp8_linear import FP8GroupLinear

        return FP8GroupLinear
    from pithtrain.layers.group_linear import GroupLinear

    return GroupLinear
