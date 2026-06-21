import argparse
import os
from pathlib import Path

import torch
from transformers import PreTrainedTokenizerFast

from benchmarks.benchmark_text_generation import PROMPTS, synchronize, tokenize
from nanovllm_ds4.engine import Scheduler, create_model


@torch.inference_mode()
def full_single_logits(model, prompts):
    return torch.stack(
        [model(prompt).logits[0, -1].float() for prompt in prompts]
    )


@torch.inference_mode()
def full_batch_logits(model, prompts):
    input_ids = torch.cat(prompts, dim=0)
    return model(input_ids).logits[:, -1].float()


@torch.inference_mode()
def cached_logits(model, prompts):
    captured = []
    forward_packed_prefill = model.forward_packed_prefill

    def capture(batch, cache_manager):
        logits = forward_packed_prefill(batch, cache_manager)
        captured.append(logits[:, -1].float().clone())
        return logits

    model.forward_packed_prefill = capture
    try:
        prompt_length = prompts[0].shape[1]
        scheduler = Scheduler(
            model,
            max_requests=len(prompts),
            max_seq_len=prompt_length + 1,
            max_prefill_tokens=len(prompts) * prompt_length,
        )
        for prompt in prompts:
            scheduler.add_request(prompt, max_new_tokens=1)
        scheduler.step()
    finally:
        model.forward_packed_prefill = forward_packed_prefill

    if len(captured) != 1:
        raise RuntimeError(f"expected one packed prefill call, got {len(captured)}")
    return captured[0]


def top1_margin(logits):
    top2 = logits.topk(2, dim=-1).values
    return top2[:, 0] - top2[:, 1]


def token_text(tokenizer, token_ids):
    return [
        tokenizer.decode([token_id], skip_special_tokens=False)
        for token_id in token_ids.tolist()
    ]


def print_logits(name, tokenizer, logits):
    token_ids = logits.argmax(dim=-1)
    print(f"\n{name}")
    print(f"argmax ids:    {token_ids.tolist()}")
    print(f"argmax tokens: {token_text(tokenizer, token_ids)}")
    print(f"top1 margins:  {top1_margin(logits).tolist()}")


def compare(name, left, right):
    difference = (left - right).abs()
    left_ids = left.argmax(dim=-1)
    right_ids = right.argmax(dim=-1)
    print(f"\n{name}")
    print(f"argmax equal:  {(left_ids == right_ids).tolist()}")
    print(f"max abs diff:  {difference.max().item():.8f}")
    print(f"mean abs diff: {difference.mean().item():.8f}")


def diagnose(full_single, full_batch, cached_single, cached_batch):
    full_batch_equal = torch.equal(
        full_single.argmax(dim=-1),
        full_batch.argmax(dim=-1),
    )
    cached_single_equal = torch.equal(
        full_single.argmax(dim=-1),
        cached_single.argmax(dim=-1),
    )
    cached_batch_equal = torch.equal(
        full_batch.argmax(dim=-1),
        cached_batch.argmax(dim=-1),
    )

    print("\nDiagnosis")
    if not full_batch_equal:
        print("full B1 and full B2 already disagree: inspect BF16 batch sensitivity first")
    elif not cached_single_equal:
        print("cached B1 differs from full B1: inspect short-sequence cache prefill")
    elif not cached_batch_equal:
        print("cached B2 differs from full B2: inspect batched cache state and writes")
    else:
        print("all four paths agree on argmax")


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--checkpoint",
        default=os.environ.get("DEEPSEEK_V4_CHECKPOINT"),
    )
    parser.add_argument("--device", default="cuda")
    return parser.parse_args()


def main():
    args = parse_args()
    if args.checkpoint is None:
        raise SystemExit("set DEEPSEEK_V4_CHECKPOINT or pass --checkpoint")

    checkpoint = Path(args.checkpoint)
    device = torch.device(args.device)
    tokenizer = PreTrainedTokenizerFast.from_pretrained(checkpoint)
    model = create_model(checkpoint, device=device)
    prompts = [
        tokenize(tokenizer, text, device)
        for text in PROMPTS[:2]
    ]
    equal_length = min(prompt.shape[1] for prompt in prompts)
    prompts = [prompt[:, :equal_length] for prompt in prompts]

    synchronize(device)
    full_single = full_single_logits(model, prompts)
    synchronize(device)
    full_batch = full_batch_logits(model, prompts)
    synchronize(device)
    cached_single = torch.cat(
        [cached_logits(model, [prompt]) for prompt in prompts],
        dim=0,
    )
    synchronize(device)
    cached_batch = cached_logits(model, prompts)
    synchronize(device)

    print(f"prompt length: {equal_length}")
    print_logits("full B1", tokenizer, full_single)
    print_logits("full B2", tokenizer, full_batch)
    print_logits("cached B1", tokenizer, cached_single)
    print_logits("cached B2", tokenizer, cached_batch)
    compare("full B1 vs full B2", full_single, full_batch)
    compare("full B1 vs cached B1", full_single, cached_single)
    compare("full B2 vs cached B2", full_batch, cached_batch)
    diagnose(full_single, full_batch, cached_single, cached_batch)


if __name__ == "__main__":
    main()
