"""Convert Qwen3.5-35B-A3B checkpoint between HuggingFace and DCP formats.

Qwen3.5-35B-A3B is a VL model in HF format: weights are under language_model.*,
visual.*, and mtp.*. pith-train only trains the text model, so we:
  - strip the "language_model." prefix from text weights
  - drop visual.* and mtp.* weights
"""

import json
from pathlib import Path

import torch
import torch.distributed.checkpoint as dcp
from safetensors import safe_open

if __name__ == "__main__":
    load_path = Path("/root/pith-train/model/Qwen3.5-35B-A3B")
    save_path = Path("/root/pith-train/workspace/checkpoints/qwen3.5-35b-a3b/torch-dcp/step-00000000")

    with open(load_path / "model.safetensors.index.json") as f:
        weight_map = json.load(f)["weight_map"]

    shard_files = set(weight_map.values())
    print(f"Converting {len(shard_files)} shards from {load_path}")

    model_state_dict = {}
    for i, shard_file in enumerate(sorted(shard_files), 1):
        print(f"Reading shard {i}/{len(shard_files)}: {shard_file}")
        with safe_open(str(load_path / shard_file), framework="pt", device="cpu") as f:
            for key in f.keys():
                # strip "model." prefix
                k = key.removeprefix("model.")
                # only keep language_model.* weights, strip that prefix too
                if k.startswith("language_model."):
                    k = k.removeprefix("language_model.")
                    model_state_dict[k] = f.get_tensor(key)
                elif k.startswith("lm_head."):
                    model_state_dict[k] = f.get_tensor(key)
                # skip visual.*, mtp.* etc.

    save_path.mkdir(parents=True, exist_ok=True)
    print(f"Saving {len(model_state_dict)} weights to {save_path}")
    dcp.save({"app": {"model": model_state_dict}}, checkpoint_id=save_path, no_dist=True)
    print("Done.")
