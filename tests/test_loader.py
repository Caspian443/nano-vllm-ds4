import json

import torch
from safetensors.torch import save_file
from torch import nn

from nanovllm_ds4.weights.loader import load_model


class TinyModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.weight = nn.Parameter(torch.zeros(2, 2))


def test_load_model(tmp_path):
    filename = "model-00001-of-00001.safetensors"
    expected = torch.tensor([[1.0, 2.0], [3.0, 4.0]])

    save_file({"weight": expected}, str(tmp_path / filename))
    index = {"weight_map": {"weight": filename}}
    (tmp_path / "model.safetensors.index.json").write_text(
        json.dumps(index),
        encoding="utf-8",
    )

    model = TinyModel()
    load_model(model, tmp_path)

    assert torch.equal(model.weight, expected)
