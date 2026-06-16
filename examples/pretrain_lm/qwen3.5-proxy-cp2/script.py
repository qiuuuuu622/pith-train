"""Proxy Qwen3.5-MoE end-to-end with context parallelism (cp=2) on a long sequence.
Exercises the all-to-all head-parallel Gated DeltaNet CP + zigzag ring full-attn.
4 GPUs: cp=2 * ep=2 (pp=1, dp=1).
"""
from pathlib import Path
from pithtrain.tasks.pretrain_lm import PretrainLMCfg, launch

cfg = PretrainLMCfg()
d = cfg.distributed
d.context_parallel_size = 2
d.expert_parallel_size = 2
d.pipeline_parallel_size = 1

t = cfg.training
t.model = Path("examples/pretrain_lm/qwen3.5-proxy-4gpu/config.json")
t.optimizer = "Adam"; t.scheduler = "CosineAnnealing"
t.max_lr = 3e-4; t.min_lr = 1e-5; 
t.warmup_steps = 2; 
t.max_steps = 5
t.micro_batch_size = 1; 
t.global_batch_size = 8
t.sequence_length = 4096
t.dataset = Path("/root/pith-train/dataset/synthetic/qwen3.5")
t.moe_load_balance_type = "global-batch";
t.moe_load_balance_coef = 1e-3
t.fp8_training = "disabled";
t.save_interval = None; 
t.save_location = None

if __name__ == "__main__":
    launch(cfg)

'''
/goal [Pasted text #1 +23 lines] 公平对比建议固定同一种（如 cp=1, pp=1, ep=2, dp=2 或 cp=2,
  ep=2），并附一组各自最优的参考。效率指标：以 tokens/sec/GPU 为主 但是也要有MFU
  写到wandb里面，建议两边都关 offload、BF16，纯比算力效率。
'''