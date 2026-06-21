import os
from pathlib import Path

import pytest
import torch
from transformers import AutoTokenizer

from nanovllm_ds4.engine import Scheduler, create_model


TEXT_PROMPTS = [
    "请用三句话解释为什么推理引擎需要 KV Cache，并给出一个生活中的类比。",
    "Explain in three sentences why continuous batching improves GPU utilization.",
    "Write a small Python function that returns the first n Fibonacci numbers.",
    (
        "下面是一段项目背景：我们正在实现一个教学型大模型推理引擎，已经完成权重加载、"
        "分页 KV Cache、连续批处理、等长批量预填充、变长 packed prefill 和 chunked "
        "prefill。项目希望继续研究 CPU/GPU offload、Shadow Radix Tree，以及针对 "
        "SM120 架构的算子优化。请总结当前成果，并给出下一阶段最重要的两个技术问题。"
    ),
]


def tokenize_prompt(tokenizer, text, device):
    return tokenizer.apply_chat_template(
        [{"role": "user", "content": text}],
        tokenize=True,
        add_generation_prompt=True,
        return_tensors="pt",
    ).to(device)


@torch.inference_mode()
def generate_reference(model, input_ids, max_new_tokens):
    output_ids = input_ids.clone()
    for _ in range(max_new_tokens):
        next_token = model(output_ids).logits[:, -1].argmax(dim=-1)
        output_ids = torch.cat([output_ids, next_token[:, None]], dim=1)
    return output_ids


def test_real_text_exercises_all_scheduler_paths():
    checkpoint = os.environ.get("DEEPSEEK_V4_CHECKPOINT")
    if checkpoint is None:
        pytest.skip("DEEPSEEK_V4_CHECKPOINT is not set")
    if not torch.cuda.is_available():
        pytest.skip("CUDA is required for the real-text integration test")

    checkpoint = Path(checkpoint)
    tokenizer = AutoTokenizer.from_pretrained(checkpoint)
    model = create_model(checkpoint, device="cuda")
    prompts = [
        tokenize_prompt(tokenizer, text, "cuda")
        for text in TEXT_PROMPTS
    ]

    equal_length = min(prompts[0].shape[1], prompts[1].shape[1])
    prompts[0] = prompts[0][:, :equal_length]
    prompts[1] = prompts[1][:, :equal_length]
    assert prompts[3].shape[1] > prompts[2].shape[1]

    max_new_tokens = 2
    joined_prefill_budget = prompts[2].shape[1] + prompts[3].shape[1] - 1
    scheduler = Scheduler(
        model,
        max_requests=len(prompts),
        max_seq_len=max(prompt.shape[1] for prompt in prompts) + max_new_tokens,
        max_prefill_tokens=joined_prefill_budget,
    )

    packed_layouts = []
    prefill_shapes = []
    decode_batch_sizes = []
    forward_packed_prefill = model.forward_packed_prefill
    forward_prefill = model.forward_prefill
    forward_decode = model.forward_decode

    def record_packed(batch, cache_manager):
        packed_layouts.append(
            (batch.seq_lens.tolist(), batch.start_positions.tolist())
        )
        return forward_packed_prefill(batch, cache_manager)

    def record_prefill(input_ids, cache_manager, request_indices, full_slots):
        prefill_shapes.append(tuple(input_ids.shape))
        return forward_prefill(
            input_ids,
            cache_manager,
            request_indices,
            full_slots,
        )

    def record_decode(input_ids, cache_manager, request_indices):
        decode_batch_sizes.append(input_ids.shape[0])
        return forward_decode(input_ids, cache_manager, request_indices)

    model.forward_packed_prefill = record_packed
    model.forward_prefill = record_prefill
    model.forward_decode = record_decode

    scheduler.add_request(prompts[0], max_new_tokens)
    scheduler.add_request(prompts[1], max_new_tokens)
    scheduler.step()
    scheduler.add_request(prompts[2], max_new_tokens)
    scheduler.add_request(prompts[3], max_new_tokens)
    outputs = scheduler.run()

    references = [
        generate_reference(model, prompt, max_new_tokens)
        for prompt in prompts
    ]
    for output, reference in zip(outputs, references):
        assert torch.equal(output, reference)

    assert prefill_shapes[0] == (2, equal_length)
    assert packed_layouts == [
        ([equal_length, equal_length], [0, 0]),
        ([prompts[2].shape[1], prompts[3].shape[1] - 1], [0, 0]),
        ([1], [prompts[3].shape[1] - 1]),
    ]
    assert decode_batch_sizes == [2, 1, 1]
    assert not scheduler.has_requests
