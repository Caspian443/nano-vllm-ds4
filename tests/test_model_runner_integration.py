import importlib
import os
import sys
from pathlib import Path

import pytest
import torch
from transformers import AutoModelForCausalLM

from nanovllm_ds4.engine import create_model, generate


def test_model_runner_matches_transformers():
    checkpoint = os.environ.get("DEEPSEEK_V4_CHECKPOINT")
    if checkpoint is None:
        pytest.skip("DEEPSEEK_V4_CHECKPOINT is not set")
    if not torch.cuda.is_available():
        pytest.skip("CUDA is required for the long-request integration test")

    checkpoint_path = Path(checkpoint)
    device = "cuda"

    sys.path.insert(0, str(checkpoint_path / "code"))
    importlib.import_module("deepseek_v4")

    model = create_model(checkpoint_path, device=device)
    input_ids = (
        torch.arange(101, device=device) % (model.config.vocab_size - 2) + 2
    ).unsqueeze(0)
    assert model.embed.weight.dtype == torch.bfloat16
    assert model.layers[2].ffn.gate.bias.dtype == torch.float32
    assert model.layers[0].ffn.gate.tid2eid.dtype == torch.int64

    output_ids = generate(model, input_ids, max_new_tokens=2).cpu()
    del model
    if device == "cuda":
        torch.cuda.empty_cache()

    reference_model = AutoModelForCausalLM.from_pretrained(
        checkpoint_path,
        dtype=torch.bfloat16,
    ).to(device)
    reference_model.eval()

    reference_ids = input_ids
    with torch.inference_mode():
        for _ in range(2):
            logits = reference_model(reference_ids).logits
            next_token_ids = logits[:, -1, :].argmax(dim=-1)
            reference_ids = torch.cat(
                [reference_ids, next_token_ids[:, None]],
                dim=-1,
            )

    assert torch.equal(output_ids, reference_ids.cpu())
