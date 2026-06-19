import json
import os
from pathlib import Path

import pytest
import torch
from safetensors import safe_open

from nanovllm_ds4.models import DeepseekV4Config, DeepseekV4ForCausalLM
from nanovllm_ds4.weights.loader import load_model


def test_load_real_deepseek_model():
    checkpoint = os.environ.get("DEEPSEEK_V4_CHECKPOINT")
    if checkpoint is None:
        pytest.skip("DEEPSEEK_V4_CHECKPOINT is not set")

    checkpoint_path = Path(checkpoint)
    config = DeepseekV4Config.from_pretrained(checkpoint_path)
    model = DeepseekV4ForCausalLM(config)
    load_model(model, checkpoint_path)

    name = "layers.0.attn.wq_a.weight"
    with (checkpoint_path / "model.safetensors.index.json").open(
        encoding="utf-8"
    ) as file:
        weight_map = json.load(file)["weight_map"]

    with safe_open(
        str(checkpoint_path / weight_map[name]),
        framework="pt",
        device="cpu",
    ) as file:
        expected = file.get_tensor(name)

    actual = model.state_dict()[name]
    assert torch.equal(actual, expected.to(dtype=actual.dtype))
