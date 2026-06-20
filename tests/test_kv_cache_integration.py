import os
from pathlib import Path

import pytest
import torch

from nanovllm_ds4.engine import create_cache_manager, create_model


def test_two_long_requests_can_decode_interleaved():
    checkpoint = os.environ.get("DEEPSEEK_V4_CHECKPOINT")
    if checkpoint is None:
        pytest.skip("DEEPSEEK_V4_CHECKPOINT is not set")
    if not torch.cuda.is_available():
        pytest.skip("CUDA is required for the long-request integration test")

    model = create_model(Path(checkpoint), device="cuda")
    vocab_size = model.config.vocab_size
    prompts = [
        (torch.arange(101, device="cuda") % (vocab_size - 2) + 2).unsqueeze(0),
        (
            (torch.arange(191, device="cuda") * 17 + 11)
            % (vocab_size - 2)
            + 2
        ).unsqueeze(0),
    ]
    max_new_tokens = 2
    cache_manager = create_cache_manager(
        model,
        max_requests=2,
        max_seq_len=193,
    )
    request_indices = [
        cache_manager.allocate_request(),
        cache_manager.allocate_request(),
    ]
    cached_ids = [prompt.clone() for prompt in prompts]
    logits = []

    with torch.inference_mode():
        for request_index, prompt in zip(request_indices, prompts):
            cache_manager.allocate_tokens(request_index, prompt.shape[1])
            logits.append(
                model.forward_inference(prompt, cache_manager, request_index)
            )

        for step in range(max_new_tokens):
            for index, request_index in enumerate(request_indices):
                next_token = logits[index][:, -1, :].argmax(dim=-1)
                cached_ids[index] = torch.cat(
                    [cached_ids[index], next_token[:, None]],
                    dim=-1,
                )
                if step + 1 < max_new_tokens:
                    cache_manager.allocate_tokens(request_index, 1)
                    logits[index] = model.forward_inference(
                        next_token[:, None],
                        cache_manager,
                        request_index,
                    )

        reference_ids = []
        for prompt in prompts:
            output_ids = prompt.clone()
            for _ in range(max_new_tokens):
                next_token = model(output_ids).logits[:, -1, :].argmax(dim=-1)
                output_ids = torch.cat(
                    [output_ids, next_token[:, None]],
                    dim=-1,
                )
            reference_ids.append(output_ids)

    for cached, reference in zip(cached_ids, reference_ids):
        assert torch.equal(cached, reference)

    long_request = request_indices[1]
    assert cache_manager.request_pool.request_lengths[long_request] == 192
    second_c96_full_slot = cache_manager.request_pool.req_to_token[
        long_request,
        191,
    ]
    second_c96_slot = second_c96_full_slot // 96
    assert torch.count_nonzero(
        cache_manager.kv_pool.compressed_cache[3][second_c96_slot]
    ).item() > 0
