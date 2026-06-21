import torch

from nanovllm_ds4.engine import Scheduler
from nanovllm_ds4.models import DeepseekV4Config, DeepseekV4ForCausalLM


def create_tiny_model():
    config = DeepseekV4Config(
        vocab_size=32,
        hidden_size=16,
        num_hidden_layers=3,
        num_attention_heads=2,
        head_dim=8,
        qk_rope_head_dim=4,
        q_lora_rank=8,
        o_lora_rank=8,
        o_groups=1,
        moe_intermediate_size=16,
        n_routed_experts=2,
        n_shared_experts=1,
        num_experts_per_tok=1,
        num_hash_layers=0,
        index_n_heads=2,
        index_head_dim=4,
        index_topk=4,
        compress_ratios=[0, 4, 16],
        sliding_window=8,
        hc_mult=2,
        hc_sinkhorn_iters=2,
        num_nextn_predict_layers=0,
    )
    torch.manual_seed(0)
    return DeepseekV4ForCausalLM(config).eval()


@torch.inference_mode()
def generate_reference(model, prompt, max_new_tokens):
    output_ids = prompt.clone()
    for _ in range(max_new_tokens):
        next_token = model(output_ids).logits[:, -1].argmax(dim=-1)
        output_ids = torch.cat([output_ids, next_token[:, None]], dim=1)
    return output_ids


def test_equal_packed_and_chunked_prefill_match_full_forward():
    model = create_tiny_model()

    equal_prompts = [
        torch.tensor([[2, 3, 4, 5, 6]]),
        torch.tensor([[7, 8, 9, 10, 11]]),
    ]
    prefill_shapes = []
    forward_prefill = model.forward_prefill

    def record_prefill(input_ids, cache_manager, request_indices, full_slots):
        prefill_shapes.append(tuple(input_ids.shape))
        return forward_prefill(
            input_ids,
            cache_manager,
            request_indices,
            full_slots,
        )

    model.forward_prefill = record_prefill
    scheduler = Scheduler(model, max_requests=2, max_seq_len=7)
    for prompt in equal_prompts:
        scheduler.add_request(prompt, max_new_tokens=2)
    equal_outputs = scheduler.run()
    model.forward_prefill = forward_prefill

    assert prefill_shapes == [(2, 5)]
    for output, prompt in zip(equal_outputs, equal_prompts):
        assert torch.equal(output, generate_reference(model, prompt, 2))

    variable_prompts = [
        torch.tensor([[2, 3, 4]]),
        torch.tensor([[5, 6, 7, 8, 9, 10]]),
    ]
    packed_layouts = []
    forward_packed_prefill = model.forward_packed_prefill

    def record_packed(batch, cache_manager):
        packed_layouts.append(
            (
                batch.seq_lens.tolist(),
                batch.offsets.tolist(),
                batch.start_positions.tolist(),
            )
        )
        return forward_packed_prefill(batch, cache_manager)

    model.forward_packed_prefill = record_packed
    scheduler = Scheduler(model, max_requests=2, max_seq_len=8)
    for prompt in variable_prompts:
        scheduler.add_request(prompt, max_new_tokens=2)
    variable_outputs = scheduler.run()

    assert packed_layouts == [([3, 6], [0, 3, 9], [0, 0])]
    for output, prompt in zip(variable_outputs, variable_prompts):
        assert torch.equal(output, generate_reference(model, prompt, 2))

    chunk_prompt = torch.tensor([[2, 3, 4, 5, 6, 7]])
    scheduler = Scheduler(
        model,
        max_requests=1,
        max_seq_len=8,
        max_prefill_tokens=4,
    )
    scheduler.add_request(chunk_prompt, max_new_tokens=2)
    chunk_output = scheduler.run()[0]

    assert packed_layouts[-2:] == [
        ([4], [0, 4], [0]),
        ([2], [0, 2], [4]),
    ]
    assert torch.equal(chunk_output, generate_reference(model, chunk_prompt, 2))
