import argparse
import gc
import json
import os
import time
from pathlib import Path

import torch
from transformers import PreTrainedTokenizerFast

from nanovllm_ds4.engine import Scheduler, create_model


PROMPTS = [
    "请用三句话解释为什么推理引擎需要 KV Cache，并给出一个生活中的类比。",
    "Explain in three sentences why continuous batching improves GPU utilization.",
    "Write a small Python function that returns the first n Fibonacci numbers.",
    (
        "下面是一段项目背景：我们正在实现一个教学型大模型推理引擎，已经完成权重加载、"
        "分页 KV Cache、连续批处理、等长批量预填充、变长 packed prefill 和 chunked "
        "prefill。项目希望继续研究 CPU/GPU offload、Shadow Radix Tree，以及针对 "
        "SM120 架构的算子优化。请总结当前成果，并给出下一阶段最重要的两个技术问题。"
    ),
    "A shop sold 18 notebooks on Monday and twice as many on Tuesday. Explain the total.",
]


def percentile(values, fraction):
    if not values:
        return 0.0
    values = sorted(values)
    return values[round((len(values) - 1) * fraction)]


def synchronize(device):
    if device.type == "cuda":
        torch.cuda.synchronize(device)


def tokenize(tokenizer, text, device):
    encoded = tokenizer.apply_chat_template(
        [{"role": "user", "content": text}],
        tokenize=True,
        add_generation_prompt=True,
        return_tensors="pt",
    )
    if not isinstance(encoded, torch.Tensor):
        encoded = encoded["input_ids"]
    return encoded.to(device)


@torch.inference_mode()
def generate_reference(model, input_ids, max_new_tokens):
    output_ids = input_ids.clone()
    for _ in range(max_new_tokens):
        next_token = model(output_ids).logits[:, -1].argmax(dim=-1)
        output_ids = torch.cat([output_ids, next_token[:, None]], dim=1)
    return output_ids


def timed_model_call(device, records, function, *args):
    if device.type == "cuda":
        start = torch.cuda.Event(enable_timing=True)
        end = torch.cuda.Event(enable_timing=True)
        start.record()
        result = function(*args)
        end.record()
        records.append((start, end))
        return result

    start = time.perf_counter()
    result = function(*args)
    records.append((time.perf_counter() - start) * 1000)
    return result


def elapsed_ms(device, records):
    if device.type == "cuda":
        return sum(start.elapsed_time(end) for start, end in records)
    return sum(records)


def run_scenario(
    name,
    model,
    tokenizer,
    inputs,
    max_new_tokens,
    max_requests,
    max_prefill_tokens,
    initial_requests,
    validation_tokens,
):
    device = next(model.parameters()).device
    max_seq_len = max(ids.shape[1] for ids in inputs) + max_new_tokens
    scheduler = Scheduler(
        model,
        max_requests=max_requests,
        max_seq_len=max_seq_len,
        max_prefill_tokens=max_prefill_tokens,
    )

    prefill_layouts = []
    equal_prefill_shapes = []
    decode_batch_sizes = []
    prefill_events = []
    decode_events = []
    forward_packed_prefill = model.forward_packed_prefill
    forward_prefill = model.forward_prefill
    forward_decode = model.forward_decode

    def record_packed(batch, cache_manager):
        prefill_layouts.append(
            {
                "seq_lens": batch.seq_lens.tolist(),
                "start_positions": batch.start_positions.tolist(),
                "total_tokens": batch.packed_tokens.numel(),
            }
        )
        return timed_model_call(
            device,
            prefill_events,
            forward_packed_prefill,
            batch,
            cache_manager,
        )

    def record_prefill(input_ids, cache_manager, request_indices, full_slots):
        equal_prefill_shapes.append(list(input_ids.shape))
        return forward_prefill(
            input_ids,
            cache_manager,
            request_indices,
            full_slots,
        )

    def record_decode(input_ids, cache_manager, request_indices):
        decode_batch_sizes.append(input_ids.shape[0])
        return timed_model_call(
            device,
            decode_events,
            forward_decode,
            input_ids,
            cache_manager,
            request_indices,
        )

    model.forward_packed_prefill = record_packed
    model.forward_prefill = record_prefill
    model.forward_decode = record_decode

    requests = []
    submitted_at = {}
    token_times = {}
    previous_generated = {}
    finished_at = {}
    step_times_ms = []
    max_used_pages = 0

    def submit(input_ids):
        request = scheduler.add_request(input_ids, max_new_tokens)
        requests.append(request)
        submitted_at[id(request)] = time.perf_counter()
        token_times[id(request)] = []
        previous_generated[id(request)] = 0

    def run_step():
        nonlocal max_used_pages
        synchronize(device)
        start = time.perf_counter()
        finished = scheduler.step()
        synchronize(device)
        end = time.perf_counter()
        step_times_ms.append((end - start) * 1000)

        for request in requests:
            previous = previous_generated[id(request)]
            if request.generated_tokens > previous:
                token_times[id(request)].extend(
                    [end] * (request.generated_tokens - previous)
                )
                previous_generated[id(request)] = request.generated_tokens
        for request in finished:
            finished_at[id(request)] = end

        allocator = scheduler.cache_manager.slot_allocator
        used_pages = allocator.num_pages - len(allocator.free_pages)
        max_used_pages = max(max_used_pages, used_pages)

    gc.collect()
    if device.type == "cuda":
        torch.cuda.empty_cache()
        torch.cuda.reset_peak_memory_stats(device)

    synchronize(device)
    scenario_start = time.perf_counter()
    for input_ids in inputs[:initial_requests]:
        submit(input_ids)
    run_step()
    for input_ids in inputs[initial_requests:]:
        submit(input_ids)
    while scheduler.has_requests:
        run_step()
    synchronize(device)
    scenario_end = time.perf_counter()
    peak_memory = (
        torch.cuda.max_memory_allocated(device) / (1024 ** 3)
        if device.type == "cuda"
        else 0.0
    )

    model.forward_packed_prefill = forward_packed_prefill
    model.forward_prefill = forward_prefill
    model.forward_decode = forward_decode

    prefill_time_ms = elapsed_ms(device, prefill_events)
    decode_time_ms = elapsed_ms(device, decode_events)
    request_metrics = []
    all_inter_token_ms = []
    for request, input_ids in zip(requests, inputs):
        times = token_times[id(request)]
        ttft_ms = (times[0] - submitted_at[id(request)]) * 1000
        e2e_ms = (finished_at[id(request)] - submitted_at[id(request)]) * 1000
        intervals = [
            (right - left) * 1000
            for left, right in zip(times, times[1:])
        ]
        all_inter_token_ms.extend(intervals)
        generated = request.output_ids[0, input_ids.shape[1]:]
        request_metrics.append(
            {
                "request": len(request_metrics),
                "request_slot": request.request_index,
                "prompt_tokens": input_ids.shape[1],
                "output_tokens": generated.numel(),
                "ttft_ms": ttft_ms,
                "tpot_ms": sum(intervals) / len(intervals) if intervals else 0.0,
                "e2e_ms": e2e_ms,
                "output_text": tokenizer.decode(
                    generated,
                    skip_special_tokens=True,
                ),
            }
        )

    validation_passed = True
    if validation_tokens:
        validation_tokens = min(validation_tokens, max_new_tokens)
        for request, input_ids in zip(requests, inputs):
            reference = generate_reference(model, input_ids, validation_tokens)
            expected = reference[:, input_ids.shape[1]:]
            actual = request.output_ids[
                :,
                input_ids.shape[1]:input_ids.shape[1] + validation_tokens,
            ]
            validation_passed = validation_passed and torch.equal(actual, expected)

    wall_seconds = scenario_end - scenario_start
    input_tokens = sum(ids.shape[1] for ids in inputs)
    output_tokens = len(inputs) * max_new_tokens
    ttfts = [item["ttft_ms"] for item in request_metrics]
    tpots = [item["tpot_ms"] for item in request_metrics]
    total_prefill_tokens = sum(
        layout["total_tokens"] for layout in prefill_layouts
    )
    total_decode_tokens = sum(decode_batch_sizes)

    return {
        "scenario": name,
        "validation_passed": validation_passed,
        "requests": len(inputs),
        "input_tokens": input_tokens,
        "output_tokens": output_tokens,
        "wall_ms": wall_seconds * 1000,
        "ttft_p50_ms": percentile(ttfts, 0.50),
        "ttft_p95_ms": percentile(ttfts, 0.95),
        "tpot_mean_ms": sum(tpots) / len(tpots),
        "itl_p50_ms": percentile(all_inter_token_ms, 0.50),
        "itl_p95_ms": percentile(all_inter_token_ms, 0.95),
        "output_tokens_per_second": output_tokens / wall_seconds,
        "total_tokens_per_second": (input_tokens + output_tokens) / wall_seconds,
        "prefill_tokens_per_second": (
            total_prefill_tokens / (prefill_time_ms / 1000)
            if prefill_time_ms > 0
            else 0.0
        ),
        "decode_tokens_per_second": (
            total_decode_tokens / (decode_time_ms / 1000)
            if decode_time_ms > 0
            else 0.0
        ),
        "peak_gpu_memory_gb": peak_memory,
        "max_used_kv_pages": max_used_pages,
        "request_slot_reuses": len(requests)
        - len({request.request_index for request in requests}),
        "step_times_ms": step_times_ms,
        "prefill_layouts": prefill_layouts,
        "equal_prefill_shapes": equal_prefill_shapes,
        "decode_batch_sizes": decode_batch_sizes,
        "request_metrics": request_metrics,
    }


def print_results(results):
    print("\nScenario summary")
    print(
        "scenario                 req  input  output  TTFT p50/p95 ms  "
        "TPOT ms  output tok/s  peak GB  KV pages  valid"
    )
    for result in results:
        print(
            f"{result['scenario']:<24} "
            f"{result['requests']:>3} "
            f"{result['input_tokens']:>6} "
            f"{result['output_tokens']:>7} "
            f"{result['ttft_p50_ms']:>8.2f}/{result['ttft_p95_ms']:<8.2f} "
            f"{result['tpot_mean_ms']:>7.2f} "
            f"{result['output_tokens_per_second']:>12.2f} "
            f"{result['peak_gpu_memory_gb']:>7.2f} "
            f"{result['max_used_kv_pages']:>8} "
            f"{str(result['validation_passed']):>5}"
        )

    for result in results:
        print(f"\n{result['scenario']} requests")
        print("id  prompt  output  TTFT ms  TPOT ms  E2E ms  slot  text")
        for item in result["request_metrics"]:
            text = item["output_text"].replace("\n", " ")[:80]
            print(
                f"{item['request']:>2} "
                f"{item['prompt_tokens']:>7} "
                f"{item['output_tokens']:>7} "
                f"{item['ttft_ms']:>8.2f} "
                f"{item['tpot_ms']:>8.2f} "
                f"{item['e2e_ms']:>7.2f} "
                f"{item['request_slot']:>5}  {text}"
            )
        print(f"prefill layouts: {result['prefill_layouts']}")
        print(f"decode batch sizes: {result['decode_batch_sizes']}")
        print(
            "model throughput: "
            f"prefill={result['prefill_tokens_per_second']:.2f} tok/s, "
            f"decode={result['decode_tokens_per_second']:.2f} tok/s"
        )


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--checkpoint",
        default=os.environ.get("DEEPSEEK_V4_CHECKPOINT"),
    )
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--max-new-tokens", type=int, default=16)
    parser.add_argument("--max-prefill-tokens", type=int, default=128)
    parser.add_argument("--validation-tokens", type=int, default=2)
    parser.add_argument("--json-output", type=Path)
    return parser.parse_args()


def main():
    args = parse_args()
    if args.checkpoint is None:
        raise SystemExit("set DEEPSEEK_V4_CHECKPOINT or pass --checkpoint")

    checkpoint = Path(args.checkpoint)
    device = torch.device(args.device)
    tokenizer = PreTrainedTokenizerFast.from_pretrained(checkpoint)
    model = create_model(checkpoint, device=device)
    inputs = [tokenize(tokenizer, prompt, device) for prompt in PROMPTS]

    equal_length = min(inputs[0].shape[1], inputs[1].shape[1])
    equal_inputs = [
        inputs[0][:, :equal_length],
        inputs[1][:, :equal_length],
    ]

    warmup = Scheduler(
        model,
        max_requests=1,
        max_seq_len=equal_length + 2,
    )
    warmup.add_request(equal_inputs[0], max_new_tokens=2)
    warmup.run()
    synchronize(device)
    del warmup
    gc.collect()
    if device.type == "cuda":
        torch.cuda.empty_cache()

    results = [
        run_scenario(
            "equal_batched_prefill",
            model,
            tokenizer,
            equal_inputs,
            args.max_new_tokens,
            max_requests=2,
            max_prefill_tokens=2 * equal_length,
            initial_requests=2,
            validation_tokens=args.validation_tokens,
        ),
        run_scenario(
            "mixed_continuous_chunked",
            model,
            tokenizer,
            inputs,
            args.max_new_tokens,
            max_requests=4,
            max_prefill_tokens=args.max_prefill_tokens,
            initial_requests=2,
            validation_tokens=args.validation_tokens,
        ),
    ]
    print_results(results)

    if args.json_output:
        args.json_output.write_text(
            json.dumps(results, indent=2, ensure_ascii=False),
            encoding="utf-8",
        )
        print(f"\nJSON written to {args.json_output}")

    if not all(result["validation_passed"] for result in results):
        raise SystemExit("benchmark output did not match full-forward validation")


if __name__ == "__main__":
    main()
