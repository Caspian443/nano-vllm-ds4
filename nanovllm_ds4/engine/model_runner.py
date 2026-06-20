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


def generate(model, input_ids, max_new_tokens):
    # input_ids: [batch_size, sequence_length]
    if max_new_tokens == 0:
        return input_ids

    cache = KVCacheManager(
        model,
        max_batch_size=input_ids.shape[0],
        max_seq_len=input_ids.shape[1] + max_new_tokens,
    )

    with torch.inference_mode():
        # Prefill reads the whole prompt and stores its KV states.
        logits = cache.prefill(input_ids)

        for step in range(max_new_tokens):
            # logits: [batch_size, current_input_length, vocab_size]
            next_token_ids = logits[:, -1, :].argmax(dim=-1)

            # [batch_size, current_sequence_length + 1]
            input_ids = torch.cat([input_ids, next_token_ids[:, None]], dim=-1)

            if step + 1 < max_new_tokens:
                # Decode reads only the newest token; previous KV states stay cached.
                logits = cache.decode(next_token_ids[:, None])

    # input_ids: [batch_size, sequence_length + max_new_tokens]
    return input_ids
