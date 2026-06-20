from dataclasses import dataclass

import torch

from nanovllm_ds4.engine.kv_cache import KVCacheManager


@dataclass
class Request:
    output_ids: torch.Tensor
    max_new_tokens: int
    request_index: int = -1
    generated_tokens: int = 0
    logits: torch.Tensor | None = None


class Scheduler:
    """Prefill pending requests, then decode them in round-robin order."""

    def __init__(self, model, max_requests, max_seq_len, page_size=96):
        self.model = model
        self.requests = []

        num_pages_per_request = (max_seq_len + page_size - 1) // page_size
        device = next(model.parameters()).device
        dtype = next(model.parameters()).dtype
        model.setup_caches(max_requests, max_seq_len)
        self.cache_manager = KVCacheManager(
            model.config,
            max_requests=max_requests,
            max_seq_len=max_seq_len,
            num_pages=max_requests * num_pages_per_request,
            page_size=page_size,
            dtype=dtype,
            device=device,
        )

    def add_request(self, input_ids, max_new_tokens):
        if input_ids.ndim != 2 or input_ids.shape[0] != 1:
            raise ValueError("each request must have input_ids shaped [1, sequence]")
        if input_ids.shape[1] == 0:
            raise ValueError("input_ids must contain at least one token")
        if max_new_tokens < 0:
            raise ValueError("max_new_tokens must be non-negative")
        if len(self.requests) >= self.cache_manager.request_pool.req_to_token.shape[0]:
            raise RuntimeError("scheduler request capacity exceeded")
        if input_ids.shape[1] + max_new_tokens > (
            self.cache_manager.request_pool.req_to_token.shape[1]
        ):
            raise RuntimeError("request exceeds max_seq_len")

        self.requests.append(Request(input_ids, max_new_tokens))

    def run(self):
        requests = self.requests
        self.requests = []
        active = []

        with torch.inference_mode():
            for request in requests:
                if request.max_new_tokens == 0:
                    continue
                request.request_index = self.cache_manager.allocate_request()
                self.cache_manager.allocate_tokens(
                    request.request_index,
                    request.output_ids.shape[1],
                )
                request.logits = self.model.forward_inference(
                    request.output_ids,
                    self.cache_manager,
                    request.request_index,
                )
                active.append(request)

            while active:
                next_active = []
                for request in active:
                    next_token = request.logits[:, -1, :].argmax(dim=-1)
                    request.output_ids = torch.cat(
                        [request.output_ids, next_token[:, None]],
                        dim=-1,
                    )
                    request.generated_tokens += 1

                    if request.generated_tokens < request.max_new_tokens:
                        self.cache_manager.allocate_tokens(request.request_index, 1)
                        request.logits = self.model.forward_inference(
                            next_token[:, None],
                            self.cache_manager,
                            request.request_index,
                        )
                        next_active.append(request)
                    else:
                        self.cache_manager.free_request(request.request_index)
                        request.logits = None

                active = next_active

        return [request.output_ids for request in requests]
