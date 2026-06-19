import torch

from nanovllm_ds4.models import DeepseekV4Config, DeepseekV4ForCausalLM
from nanovllm_ds4.weights import load_model


def create_model(path, device="cpu"):
    config = DeepseekV4Config.from_pretrained(path)
    model = DeepseekV4ForCausalLM(config)
    model.to(device)
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
    for _ in range(max_new_tokens):
        # next_token_ids: [batch_size]
        next_token_ids = generate_next_token(model, input_ids)

        # [batch_size, sequence_length] + [batch_size, 1]
        input_ids = torch.cat([input_ids, next_token_ids[:, None]], dim=-1)

    # input_ids: [batch_size, sequence_length + max_new_tokens]
    return input_ids
