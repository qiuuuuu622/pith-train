"""Convert Qwen3.5-35B-A3B checkpoint between HuggingFace and DCP formats.

Qwen3.5-35B-A3B is a VL model in HF format. We:
  - strip the "language_model." prefix from text weights
  - drop visual.* and mtp.* weights
  - split stacked expert tensors (shape [E, ...]) into per-expert keys
    e.g. experts.gate_up_proj -> experts.0.gate_up_proj, experts.1.gate_up_proj, ...
"""

import json
from pathlib import Path

import torch
import torch.distributed.checkpoint as dcp
from safetensors import safe_open

EXPERT_STACKED_KEYS = {"gate_up_proj", "down_proj"}


def convert(load_path: Path, save_path: Path) -> None:
    with open(load_path / "model.safetensors.index.json") as f:
        weight_map = json.load(f)["weight_map"]

    shard_files = set(weight_map.values())
    print(f"Converting {len(shard_files)} shards from {load_path}")

    model_state_dict: dict[str, torch.Tensor] = {}
    for i, shard_file in enumerate(sorted(shard_files), 1):
        print(f"Reading shard {i}/{len(shard_files)}: {shard_file}")
        with safe_open(str(load_path / shard_file), framework="pt", device="cpu") as f:
            for key in f.keys():
                k = key.removeprefix("model.")
                if k.startswith("language_model."):
                    k = k.removeprefix("language_model.")
                elif not k.startswith("lm_head."):
                    continue  # skip visual.*, mtp.*, etc.

                # split stacked expert tensors into per-expert keys
                # e.g. layers.0.mlp.experts.gate_up_proj -> layers.0.mlp.experts.0.gate_up_proj
                parts = k.split(".experts.")
                if len(parts) == 2 and parts[1] in EXPERT_STACKED_KEYS:
                    tensor = f.get_tensor(key)  # [num_experts, ...]
                    for idx in range(tensor.shape[0]):
                        new_key = f"{parts[0]}.experts.{idx}.{parts[1]}"
                        model_state_dict[new_key] = tensor[idx]
                else:
                    model_state_dict[k] = f.get_tensor(key)

    save_path.mkdir(parents=True, exist_ok=True)
    print(f"Saving {len(model_state_dict)} weights to {save_path}")
    dcp.save({"app": {"model": model_state_dict}}, checkpoint_id=save_path, no_dist=True)
    print("Done.")


if __name__ == "__main__":
    convert(
        load_path=Path("/root/pith-train/model/Qwen3.5-35B-A3B"),
        save_path=Path("/root/pith-train/workspace/checkpoints/qwen3.5-35b-a3b/torch-dcp/step-00000000"),
    )
