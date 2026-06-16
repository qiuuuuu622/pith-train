"""pith-train throughput benchmark: Qwen3.5 proxy, fixed fair config.
cp=1, pp=1, ep=2, dp=2 ; BF16 ; expert offload OFF ; steady-state throughput + MFU.
"""
from pathlib import Path
from pithtrain.modules.logging import LoggingWandbCfg
from pithtrain.tasks.pretrain_lm import PretrainLMCfg, launch

cfg = PretrainLMCfg()
d = cfg.distributed
d.context_parallel_size = 1
d.pipeline_parallel_size = 1
d.expert_parallel_size = 2     # world=4 -> dp=2

t = cfg.training
t.model = Path("examples/pretrain_lm/qwen3.5-proxy-4gpu/config.json")
t.optimizer = "Adam"; t.scheduler = "CosineAnnealing"
t.max_lr = 3e-4; t.min_lr = 3e-4; t.warmup_steps = 5; t.max_steps = 40
t.micro_batch_size = 1; t.global_batch_size = 32   # accumulate = 32/(1*2*2)=8
t.sequence_length = 2048
t.dataset = Path("/root/pith-train/dataset/synthetic/qwen3.5")
t.moe_load_balance_type = "global-batch"; t.moe_load_balance_coef = 1e-3
t.fp8_training = "disabled"; t.save_interval = None; t.save_location = None
t.expert_cpu_offload = False   # model fits in HBM -> disable offload for throughput

lg = cfg.logging
lg.wandb = LoggingWandbCfg()
lg.wandb.entity = None
lg.wandb.project = "qwen3.5-proxy-fwk-bench"
lg.wandb.name = "pithtrain-cp1-ep2-dp2-s2048-bf16"

if __name__ == "__main__":
    launch(cfg)
