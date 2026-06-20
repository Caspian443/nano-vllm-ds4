import os
from pathlib import Path

import pytest
import torch

from nanovllm_ds4.engine import Scheduler, create_model


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
    scheduler = Scheduler(
        model,
        max_requests=2,
        max_seq_len=193,
    )
    for prompt in prompts:
        scheduler.add_request(prompt, max_new_tokens)

    with torch.inference_mode():
        cached_ids = scheduler.run()

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

    cache_manager = scheduler.cache_manager
    assert cache_manager.request_pool.request_lengths == [-1, -1]
    assert len(cache_manager.slot_allocator.free_pages) == 6
    assert torch.count_nonzero(cache_manager.kv_pool.swa_cache).item() == 0
    assert torch.count_nonzero(
        cache_manager.kv_pool.compressed_cache[3][3]
    ).item() > 0
