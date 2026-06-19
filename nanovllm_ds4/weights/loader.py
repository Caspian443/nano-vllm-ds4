"""Load safetensors weights into an already constructed model."""

import json
from pathlib import Path

import torch
from safetensors import safe_open


def load_model(model, path):
    """Copy checkpoint tensors into an already constructed model."""
    path = Path(path)

    # weight_map describes where every checkpoint tensor is stored:
    # {
    #     "embed.weight": "model-00001-of-00002.safetensors",
    #     "head.weight": "model-00002-of-00002.safetensors",
    # }
    with (path / "model.safetensors.index.json").open(encoding="utf-8") as file:
        weight_map = json.load(file)["weight_map"]

    # weights_by_file groups tensor names by shard so each file is opened once:
    # {
    #     "model-00001-of-00002.safetensors": [
    #         "embed.weight",
    #         "layers.0.attn.wq_a.weight",
    #     ],
    # }
    weights_by_file = {}
    for name, filename in weight_map.items():
        weights_by_file.setdefault(filename, []).append(name)

    # target_tensors maps names to the actual Parameters and buffers in model:
    # {
    #     "embed.weight": model.embed.weight,
    #     "layers.0.attn.wq_a.weight": model.layers[0].attn.wq_a.weight,
    # }
    target_tensors = model.state_dict(keep_vars=True)

    with torch.no_grad():
        for filename, names in weights_by_file.items():
            with safe_open(str(path / filename), framework="pt", device="cpu") as file:
                for name in names:
                    # source is read from disk; target is owned by the model.
                    source = file.get_tensor(name)
                    target = target_tensors[name]

                    if source.shape != target.shape:
                        raise ValueError(f"Shape mismatch: {name}")

                    # copy_ changes the existing model tensor in place.
                    source = source.to(device=target.device, dtype=target.dtype)
                    target.copy_(source)
