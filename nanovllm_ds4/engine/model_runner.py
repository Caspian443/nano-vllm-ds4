import torch

from nanovllm_ds4.engine.kv_cache import KVCacheManager
from nanovllm_ds4.models import DeepseekV4Config, DeepseekV4ForCausalLM
from nanovllm_ds4.weights import load_model


def create_model(path, device="cpu", dtype=torch.bfloat16):
    config = DeepseekV4Config.from_pretrained(path)

    original_dtype = torch.get_default_dtype()
    torch.set_default_dtype(dtype)
    try:
        with torch.device(device):
            model = DeepseekV4ForCausalLM(config)
    finally:
        torch.set_default_dtype(original_dtype)

    load_model(model, path)
    model.eval()
    return model


def generate_next_token(model, input_ids):
    # input_ids: [batch_size, sequence_length]
    with torch.inference_mode():
        # logits: [batch_size, sequence_length, vocab_size]
        logits = model(input_ids).logits

        # next_token_logits: [batch_size, vocab_size]
        next_token_logits = logits[:, -1, :]

        # next_token_ids: [batch_size]
        next_token_ids = next_token_logits.argmax(dim=-1)

    return next_token_ids


def create_cache_manager(model, max_requests, max_seq_len, page_size=96):
    num_pages_per_request = (max_seq_len + page_size - 1) // page_size
    device = next(model.parameters()).device
    dtype = next(model.parameters()).dtype
    model.setup_caches(max_requests, max_seq_len)
    return KVCacheManager(
        model.config,
        max_requests=max_requests,
        max_seq_len=max_seq_len,
        num_pages=max_requests * num_pages_per_request,
        page_size=page_size,
        dtype=dtype,
        device=device,
    )


def generate(model, input_ids, max_new_tokens):
    # input_ids: [batch_size, sequence_length]
    if max_new_tokens == 0:
        return input_ids
    if input_ids.shape[0] != 1:
        raise ValueError("generate currently supports batch_size=1")

    cache_manager = create_cache_manager(
        model,
        max_requests=1,
        max_seq_len=input_ids.shape[1] + max_new_tokens,
    )
    request_index = cache_manager.allocate_request()
    cache_manager.allocate_tokens(request_index, input_ids.shape[1])

    with torch.inference_mode():
        # Prefill reads the whole prompt and stores its KV states.
        logits = model.forward_inference(
            input_ids,
            cache_manager,
            request_index,
        )

        for step in range(max_new_tokens):
            # logits: [batch_size, current_input_length, vocab_size]
            next_token_ids = logits[:, -1, :].argmax(dim=-1)

            # [batch_size, current_sequence_length + 1]
            input_ids = torch.cat([input_ids, next_token_ids[:, None]], dim=-1)

            if step + 1 < max_new_tokens:
                # Decode reads only the newest token; previous KV states stay cached.
                cache_manager.allocate_tokens(request_index, 1)
                logits = model.forward_inference(
                    next_token_ids[:, None],
                    cache_manager,
                    request_index,
                )

    # input_ids: [batch_size, sequence_length + max_new_tokens]
    return input_ids
