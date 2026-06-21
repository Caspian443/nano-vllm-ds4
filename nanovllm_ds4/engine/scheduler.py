from dataclasses import dataclass

import torch

from nanovllm_ds4.engine.kv_cache import KVCacheManager


@dataclass
class Request:
    output_ids: torch.Tensor
    max_new_tokens: int
    prompt_length: int
    arrival_index: int
    request_index: int = -1
    prefilled_tokens: int = 0
    generated_tokens: int = 0
    logits: torch.Tensor | None = None


@dataclass
class PrefillBatch:
    # Packed token layout consumed by the reference backend and future kernels.
    packed_tokens: torch.Tensor       # [total_tokens]
    seq_lens: torch.Tensor            # [batch_size]
    offsets: torch.Tensor             # [batch_size + 1]
    request_indices: torch.Tensor      # [batch_size]
    start_positions: torch.Tensor      # [batch_size]
    full_slots: torch.Tensor           # [total_tokens]


class Scheduler:
    """Continuously admit pending requests and batch active decode tokens."""

    def __init__(self, model, max_requests, max_seq_len, page_size=96,
                 max_prefill_tokens=None):
        self.model = model
        self.pending_requests = []
        self.prefill_requests = []
        self.running_requests = []
        self.next_arrival_index = 0
        if max_prefill_tokens is None:
            max_prefill_tokens = max_requests * max_seq_len
        if max_prefill_tokens <= 0:
            raise ValueError("max_prefill_tokens must be positive")
        self.max_prefill_tokens = max_prefill_tokens

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
        if input_ids.shape[1] + max_new_tokens > (
            self.cache_manager.request_pool.req_to_token.shape[1]
        ):
            raise RuntimeError("request exceeds max_seq_len")

        request = Request(
            input_ids,
            max_new_tokens,
            prompt_length=input_ids.shape[1],
            arrival_index=self.next_arrival_index,
        )
        self.next_arrival_index += 1
        self.pending_requests.append(request)
        return request

    @property
    def has_requests(self):
        return bool(
            self.pending_requests
            or self.prefill_requests
            or self.running_requests
        )

    def _prefill(self):
        budget = self.max_prefill_tokens
        scheduled = []
        waiting = []
        partial = []

        for index, request in enumerate(self.prefill_requests):
            if budget == 0:
                waiting.append(request)
                continue

            requests_left = len(self.prefill_requests) - index
            chunk_size = min(
                request.prompt_length - request.prefilled_tokens,
                max(1, budget // requests_left),
            )
            start = request.prefilled_tokens
            end = start + chunk_size
            full_slots = self.cache_manager.allocate_tokens(
                request.request_index,
                chunk_size,
            )
            scheduled.append(
                (
                    request,
                    request.output_ids[0, start:end],
                    full_slots,
                    start,
                )
            )
            request.prefilled_tokens = end
            budget -= chunk_size

            if end < request.prompt_length:
                partial.append(request)

        self.prefill_requests = waiting + partial
        if not scheduled:
            return

        device = scheduled[0][1].device
        seq_lens = torch.tensor(
            [tokens.numel() for _, tokens, _, _ in scheduled],
            dtype=torch.long,
            device=device,
        )
        offsets = torch.cat(
            [
                torch.zeros(1, dtype=torch.long, device=device),
                seq_lens.cumsum(dim=0),
            ]
        )
        batch = PrefillBatch(
            packed_tokens=torch.cat(
                [tokens for _, tokens, _, _ in scheduled]
            ),
            seq_lens=seq_lens,
            offsets=offsets,
            request_indices=torch.tensor(
                [request.request_index for request, _, _, _ in scheduled],
                dtype=torch.long,
                device=device,
            ),
            start_positions=torch.tensor(
                [start for _, _, _, start in scheduled],
                dtype=torch.long,
                device=device,
            ),
            full_slots=torch.cat(
                [full_slots for _, _, full_slots, _ in scheduled]
            ),
        )
        logits = self.model.forward_packed_prefill(batch, self.cache_manager)
        for batch_index, (request, _, _, _) in enumerate(scheduled):
            request.logits = logits[batch_index:batch_index + 1]
            if request.prefilled_tokens == request.prompt_length:
                self.running_requests.append(request)

    def step(self):
        """Run one scheduling step and return requests finished in this step."""
        finished_requests = []
        with torch.inference_mode():
            while self.pending_requests:
                request = self.pending_requests[0]
                if request.max_new_tokens == 0:
                    self.pending_requests.pop(0)
                    finished_requests.append(request)
                    continue
                if not self.cache_manager.request_pool.free_request_slots:
                    break

                self.pending_requests.pop(0)
                request.request_index = self.cache_manager.allocate_request()
                self.prefill_requests.append(request)

            self._prefill()

            next_running = []
            decode_tokens = []
            for request in self.running_requests:
                next_token = request.logits[:, -1, :].argmax(dim=-1)
                request.output_ids = torch.cat(
                    [request.output_ids, next_token[:, None]],
                    dim=-1,
                )
                request.generated_tokens += 1

                if request.generated_tokens < request.max_new_tokens:
                    self.cache_manager.allocate_tokens(request.request_index, 1)
                    next_running.append(request)
                    decode_tokens.append(next_token[:, None])
                else:
                    self.cache_manager.free_request(request.request_index)
                    request.logits = None
                    finished_requests.append(request)

            if next_running:
                # input_ids: [B, 1], request_indices: [B]
                input_ids = torch.cat(decode_tokens, dim=0)
                request_indices = torch.tensor(
                    [request.request_index for request in next_running],
                    dtype=torch.long,
                    device=input_ids.device,
                )
                logits = self.model.forward_decode(
                    input_ids,
                    self.cache_manager,
                    request_indices,
                )
                for batch_index, request in enumerate(next_running):
                    request.logits = logits[batch_index:batch_index + 1]

            self.running_requests = next_running

        return finished_requests

    def run(self):
        requests = sorted(
            [
                *self.running_requests,
                *self.prefill_requests,
                *self.pending_requests,
            ],
            key=lambda request: request.arrival_index,
        )
        while self.has_requests:
            self.step()

        return [request.output_ids for request in requests]
