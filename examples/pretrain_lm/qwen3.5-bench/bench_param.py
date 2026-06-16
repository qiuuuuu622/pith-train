"""Parameterized pith-train throughput bench. Knobs via env: SEQ, GBS, MBS, EP, PP, CP, OFFLOAD, RUNNAME."""
import os
from pathlib import Path
from pithtrain.modules.logging import LoggingWandbCfg
from pithtrain.tasks.pretrain_lm import PretrainLMCfg, launch

E = os.environ
cfg = PretrainLMCfg()
d = cfg.distributed
d.context_parallel_size = int(E.get("CP", 1))
d.pipeline_parallel_size = int(E.get("PP", 1))
d.expert_parallel_size = int(E.get("EP", 2))

t = cfg.training
t.model = Path("examples/pretrain_lm/qwen3.5-proxy-4gpu/config.json")
t.optimizer = "Adam"; t.scheduler = "CosineAnnealing"
t.max_lr = 3e-4; t.min_lr = 3e-4
t.warmup_steps = int(E.get("WARMUP", 5)); t.max_steps = int(E.get("STEPS", 30))
t.micro_batch_size = int(E.get("MBS", 1)); t.global_batch_size = int(E.get("GBS", 64))
t.sequence_length = int(E.get("SEQ", 4096))
t.dataset = Path("/root/pith-train/dataset/synthetic/qwen3.5")
t.moe_load_balance_type = "global-batch"; t.moe_load_balance_coef = 1e-3
t.fp8_training = "disabled"; t.save_interval = None; t.save_location = None
t.expert_cpu_offload = E.get("OFFLOAD", "0") == "1"

if E.get("WANDB", "1") == "1":
    lg = cfg.logging
    lg.wandb = LoggingWandbCfg(); lg.wandb.entity = None
    lg.wandb.project = "qwen3.5-proxy-fwk-bench"
    lg.wandb.name = E.get("RUNNAME", "pithtrain-run")
else:
    cfg.logging.wandb = None

if __name__ == "__main__":
    launch(cfg)
