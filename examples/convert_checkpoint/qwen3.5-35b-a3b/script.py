"""Convert Qwen3.5-35B-A3B checkpoint between HuggingFace and DCP formats."""

from pathlib import Path

from pithtrain.tasks.convert_checkpoint import ConvertCheckpointCfg, launch

# hf2dcp: convert downloaded HF weights to pith-train DCP format for training
cfg = ConvertCheckpointCfg()
cfg.operation = "hf2dcp"
cfg.load_path = Path("/root/pith-train/model/Qwen3.5-35B-A3B")
cfg.save_path = Path("/root/pith-train/workspace/checkpoints/qwen3.5-35b-a3b/torch-dcp/step-00000000")

if __name__ == "__main__":
    launch(cfg)
