"""Pretrain Qwen3.5-35B-A3B (text model) on a single 8-GPU H100/H200/B200 node.

Architecture config is the authoritative text_config extracted from
/mnt/model_35b. Key throughput knobs are env-overridable so configs can be
swept without editing the file:

  PP, EP, CP, SEQ, MBS, GBS, OFFLOAD (0/1), STEPS, WARMUP, FP8 (disabled/deep-gemm)

On 8x H100 80GB the per-GPU expert optimizer state (~48GB fp32 moments) only
fits with OFFLOAD=1 at EP<=4; OFFLOAD=0 trades that for a large step-time win
but needs the headroom to come from elsewhere (higher EP / recompute).
"""

import os
from pathlib import Path

from pithtrain.modules.logging import LoggingWandbCfg  # noqa: F401
from pithtrain.tasks.pretrain_lm import PretrainLMCfg, launch


def _i(name: str, default: int) -> int:
    return int(os.environ.get(name, default))


def _i_opt(name: str):
    v = os.environ.get(name)
    return int(v) if v is not None else None


cfg = PretrainLMCfg()

distributed = cfg.distributed
distributed.context_parallel_size = _i("CP", 1)
distributed.pipeline_parallel_size = _i("PP", 4)
distributed.expert_parallel_size = _i("EP", 2)

training = cfg.training
training.model = Path("examples/pretrain_lm/qwen3.5-35b-a3b/config.json")
training.optimizer = "AdamW"
training.weight_decay = 0.1
training.adam_beta1 = 0.9
training.adam_beta2 = 0.95
training.scheduler = "CosineAnnealing"
training.max_lr = 3.0e-4
training.min_lr = 1.0e-5
training.expert_cpu_offload = bool(_i("OFFLOAD", 1))
training.fuse_lmhead_ce = bool(_i("FUSE_CE", 0))
training.fuse_lmhead_ce_chunks = _i("FUSE_CHUNKS", 8)
training.recompute_mlp = bool(_i("RECOMPUTE_MLP", 0))
training.recompute_attn = bool(_i("RECOMPUTE_ATTN", 0))
training.warmup_steps = _i("WARMUP", 2)
training.max_steps = _i("STEPS", 10)
training.micro_batch_size = _i("MBS", 1)
training.global_batch_size = _i("GBS", 128)
training.sequence_length = _i("SEQ", 2048)
training.dataset = Path("/root/pith-train/dataset/synthetic/qwen3.5")
training.moe_load_balance_type = "global-batch"
training.moe_load_balance_coef = 1e-3
training.fp8_training = os.environ.get("FP8", "disabled")
training.save_interval = None
training.save_location = None
# Nsys: profile a single steady-state step (set NSYS_START/NSYS_STOP).
training.nsys_start = _i_opt("NSYS_START")
training.nsys_stop = _i_opt("NSYS_STOP")
# CUDA memory snapshot: record at MEMPROF_START, dump at MEMPROF_STOP.
training.memory_profile_start = _i_opt("MEMPROF_START")
training.memory_profile_stop = _i_opt("MEMPROF_STOP")
if os.environ.get("MEMPROF_OUT"):
    training.memory_profile_output = Path(os.environ["MEMPROF_OUT"])
# Reduce per-step CPU stall during throughput measurement.
training.gc_collect_interval = _i("GC", 5)

logging = cfg.logging  # noqa: F841  (wandb off for benchmarking)

if __name__ == "__main__":
    launch(cfg)
