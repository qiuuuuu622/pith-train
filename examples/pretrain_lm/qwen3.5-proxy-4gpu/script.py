"""Smoke-test pretrain: Qwen3.5-MoE *proxy* (same architecture, fewer layers/experts)
on a single 4-GPU node. Validates the full training path (incl. the Gated DeltaNet
and MoE/PP/EP machinery) end to end. CP stays at 1 because the full_attention
layers still raise NotImplementedError for cp_size > 1.
"""

from pathlib import Path

from pithtrain.tasks.pretrain_lm import PretrainLMCfg, launch

cfg = PretrainLMCfg()

distributed = cfg.distributed
distributed.context_parallel_size = 1
distributed.pipeline_parallel_size = 2
distributed.expert_parallel_size = 2
# world = pp * ep * cp * dp = 4 -> dp = 1

training = cfg.training
training.model = Path("examples/pretrain_lm/qwen3.5-proxy-4gpu/config.json")
training.optimizer = "Adam"
training.scheduler = "CosineAnnealing"
training.max_lr = 3.0e-4
training.min_lr = 1.0e-5
training.warmup_steps = 2
training.max_steps = 5
training.micro_batch_size = 1
training.global_batch_size = 8
training.sequence_length = 512
training.dataset = Path("/root/pith-train/dataset/synthetic/qwen3.5")
training.moe_load_balance_type = "global-batch"
training.moe_load_balance_coef = 1e-3
training.fp8_training = "disabled"
training.save_interval = None
training.save_location = None

if __name__ == "__main__":
    launch(cfg)
