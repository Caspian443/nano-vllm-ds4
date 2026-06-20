import torch

from nanovllm_ds4.engine.scheduler import Scheduler
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
    scheduler = Scheduler(
        model,
        max_requests=1,
        max_seq_len=input_ids.shape[1] + max_new_tokens,
    )
    scheduler.add_request(input_ids, max_new_tokens)
    return scheduler.run()[0]
