"""Pretrain Qwen3.5-35B-A3B (text model) on a single 8-GPU H200/B200 node."""

from pathlib import Path

from pithtrain.modules.logging import LoggingWandbCfg  # noqa: F401
from pithtrain.tasks.pretrain_lm import PretrainLMCfg, launch

cfg = PretrainLMCfg()

distributed = cfg.distributed
distributed.context_parallel_size = 1  # Gated DeltaNet layers have no CP path
# pp=4 keeps only ~10 layers/rank (40 layers / 8 V-chunks), which is what brings the
# activation + deferred-wgrad peak under the 8x H200 budget. world = pp * ep = 8.
distributed.pipeline_parallel_size = 4
distributed.expert_parallel_size = 2

training = cfg.training
training.model = Path("examples/pretrain_lm/qwen3.5-35b-a3b/config.json")
training.optimizer = "AdamW"
training.weight_decay = 0.1
training.adam_beta1 = 0.9
training.adam_beta2 = 0.95
training.scheduler = "CosineAnnealing"
training.max_lr = 3.0e-4
training.min_lr = 1.0e-5
training.warmup_steps = 32
training.max_steps = 256
training.micro_batch_size = 1
# Smoke-test value. Raise to 128+ before measuring throughput so the pipeline
# warmup/drain bubble is amortized and tokens/sec is not understated.
training.global_batch_size = 16
training.sequence_length = 1024  
training.dataset = Path("/root/pith-train/dataset/dclm-baseline/toktxt/qwen3.5")
training.moe_load_balance_type = "global-batch"
training.moe_load_balance_coef = 1e-3
# "disabled" = BF16 (no DeepGEMM). Switch to "deep-gemm" for the FP8 fast path.
training.fp8_training = "disabled"
training.save_interval = 256
training.save_location = Path("/root/pith-train/workspace/checkpoints/qwen3.5-35b-a3b")

# Wandb logging configuration. Comment out to disable.
logging = cfg.logging
logging.wandb = LoggingWandbCfg()
logging.wandb.entity = None  # None -> use the default entity of the logged-in API key
logging.wandb.project = "qwen3.5-pretrain"
logging.wandb.name = "qwen3.5-35b-a3b"

if __name__ == "__main__":
    launch(cfg)
