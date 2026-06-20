import torch


class RequestTokenPool:
    """Map each request token position to a full KV-cache slot."""

    def __init__(self, max_requests, max_seq_len, device="cpu"):
        # req_to_token[request_index, token_position] = full_slot
        self.req_to_token = torch.full(
            (max_requests, max_seq_len),
            -1,
            dtype=torch.long,
            device=device,
        )
        # -1 means the request row is free; otherwise this is its token count.
        self.request_lengths = [-1] * max_requests
        self.free_request_slots = list(range(max_requests - 1, -1, -1))

    def allocate_request(self):
        if not self.free_request_slots:
            raise RuntimeError("no free request slots")

        request_index = self.free_request_slots.pop()
        self.request_lengths[request_index] = 0
        return request_index

    def append(self, request_index, full_slots):
        start = self.request_lengths[request_index]
        if start < 0:
            raise RuntimeError("request slot is not allocated")

        full_slots = torch.as_tensor(
            full_slots,
            dtype=torch.long,
            device=self.req_to_token.device,
        ).reshape(-1)
        end = start + full_slots.numel()
        if end > self.req_to_token.shape[1]:
            raise RuntimeError("request exceeds max_seq_len")

        self.req_to_token[request_index, start:end] = full_slots
        self.request_lengths[request_index] = end

    def free_request(self, request_index):
        length = self.request_lengths[request_index]
        if length < 0:
            raise RuntimeError("request slot is not allocated")

        full_slots = self.req_to_token[request_index, :length].clone()
        self.req_to_token[request_index].fill_(-1)
        self.request_lengths[request_index] = -1
        self.free_request_slots.append(request_index)
        return full_slots


class PagedSlotAllocator:
    """Allocate full KV-cache slots one page at a time."""

    def __init__(self, num_pages, page_size=96):
        self.page_size = page_size
        self.num_pages = num_pages
        self.free_pages = list(range(num_pages - 1, -1, -1))

    def allocate_page(self):
        if not self.free_pages:
            raise RuntimeError("no free KV-cache pages")
        return self.free_pages.pop()

    def free_page(self, page_id):
        if page_id < 0 or page_id >= self.num_pages:
            raise ValueError("invalid page id")
        if page_id in self.free_pages:
            raise RuntimeError("KV-cache page is already free")
        self.free_pages.append(page_id)

    def page_slots(self, page_id):
        if page_id < 0 or page_id >= self.num_pages:
            raise ValueError("invalid page id")
        start = page_id * self.page_size
        return range(start, start + self.page_size)


class DeepseekV4KVPool:
    """Store DeepSeek V4 sliding-window and compressed KV tensors."""

    def __init__(
        self,
        config,
        max_requests,
        num_pages,
        page_size=96,
        dtype=torch.bfloat16,
        device="cpu",
    ):
        self.compress_ratios = tuple(config.compress_ratios)
        total_full_slots = num_pages * page_size

        # swa_cache[layer, request, token_position % sliding_window] = kv
        self.swa_cache = torch.zeros(
            (
                len(self.compress_ratios),
                max_requests,
                config.sliding_window,
                config.head_dim,
            ),
            dtype=dtype,
            device=device,
        )

        # Completed C4/C96 entries used directly by attention.
        self.compressed_cache = []
        self.indexer_cache = []

        # Inflight entries have not collected enough tokens to be compressed.
        # They share the active request lifetime with swa_cache, but stay in
        # separate FP32 tensors because compression also needs score values.
        self.kv_state = []
        self.score_state = []
        self.indexer_kv_state = []
        self.indexer_score_state = []

        for ratio in self.compress_ratios:
            if ratio == 0:
                self.compressed_cache.append(None)
                self.indexer_cache.append(None)
                self.kv_state.append(None)
                self.score_state.append(None)
                self.indexer_kv_state.append(None)
                self.indexer_score_state.append(None)
                continue

            if page_size % ratio != 0:
                raise ValueError("page size must be divisible by compression ratio")

            num_compressed_slots = total_full_slots // ratio
            self.compressed_cache.append(
                torch.zeros(
                    num_compressed_slots,
                    config.head_dim,
                    dtype=dtype,
                    device=device,
                )
            )

            overlap = ratio < 16
            coefficient = 2 if overlap else 1
            state_shape = (
                max_requests,
                coefficient * ratio,
                coefficient * config.head_dim,
            )
            self.kv_state.append(
                torch.zeros(state_shape, dtype=torch.float32, device=device)
            )
            self.score_state.append(
                torch.full(
                    state_shape,
                    float("-inf"),
                    dtype=torch.float32,
                    device=device,
                )
            )

            if overlap:
                self.indexer_cache.append(
                    torch.zeros(
                        num_compressed_slots,
                        config.index_head_dim,
                        dtype=dtype,
                        device=device,
                    )
                )
                indexer_state_shape = (
                    max_requests,
                    2 * ratio,
                    2 * config.index_head_dim,
                )
                self.indexer_kv_state.append(
                    torch.zeros(
                        indexer_state_shape,
                        dtype=torch.float32,
                        device=device,
                    )
                )
                self.indexer_score_state.append(
                    torch.full(
                        indexer_state_shape,
                        float("-inf"),
                        dtype=torch.float32,
                        device=device,
                    )
                )
            else:
                self.indexer_cache.append(None)
                self.indexer_kv_state.append(None)
                self.indexer_score_state.append(None)

    def write_swa(self, layer_index, request_index, token_position, kv):
        swa_slot = token_position % self.swa_cache.shape[2]
        self.swa_cache[layer_index, request_index, swa_slot].copy_(kv)

    def write_compressed(self, layer_index, full_slot, kv):
        cache = self.compressed_cache[layer_index]
        if cache is None:
            raise RuntimeError("layer does not use compressed KV cache")
        full_slots = torch.as_tensor(
            full_slot,
            dtype=torch.long,
            device=cache.device,
        ).reshape(-1)
        compressed_slots = full_slots // self.compress_ratios[layer_index]
        values = kv.reshape(-1, cache.shape[-1]).to(cache)
        cache.index_copy_(0, compressed_slots, values)

    def write_indexer(self, layer_index, full_slot, key):
        cache = self.indexer_cache[layer_index]
        if cache is None:
            raise RuntimeError("layer does not use indexer cache")
        full_slots = torch.as_tensor(
            full_slot,
            dtype=torch.long,
            device=cache.device,
        ).reshape(-1)
        compressed_slots = full_slots // self.compress_ratios[layer_index]
        values = key.reshape(-1, cache.shape[-1]).to(cache)
        cache.index_copy_(0, compressed_slots, values)

    def reset_request(self, request_index):
        self.swa_cache[:, request_index].zero_()
        for layer_index in range(len(self.compress_ratios)):
            if self.kv_state[layer_index] is not None:
                self.kv_state[layer_index][request_index].zero_()
                self.score_state[layer_index][request_index].fill_(float("-inf"))
            if self.indexer_kv_state[layer_index] is not None:
                self.indexer_kv_state[layer_index][request_index].zero_()
                self.indexer_score_state[layer_index][request_index].fill_(
                    float("-inf")
                )


class KVCacheManager:
    """Connect request mappings, page allocation, and physical KV tensors."""

    def __init__(
        self,
        config,
        max_requests,
        max_seq_len,
        num_pages,
        page_size=96,
        dtype=torch.bfloat16,
        device="cpu",
    ):
        self.request_pool = RequestTokenPool(
            max_requests,
            max_seq_len,
            device=device,
        )
        self.slot_allocator = PagedSlotAllocator(num_pages, page_size)
        self.kv_pool = DeepseekV4KVPool(
            config,
            max_requests,
            num_pages,
            page_size=page_size,
            dtype=dtype,
            device=device,
        )

    def allocate_request(self):
        return self.request_pool.allocate_request()

    def allocate_tokens(self, request_index, num_tokens):
        if num_tokens < 0:
            raise ValueError("num_tokens must be non-negative")

        start = self.request_pool.request_lengths[request_index]
        if start < 0:
            raise RuntimeError("request slot is not allocated")
        if start + num_tokens > self.request_pool.req_to_token.shape[1]:
            raise RuntimeError("request exceeds max_seq_len")

        page_size = self.slot_allocator.page_size
        offset = start % page_size
        current_page_space = page_size - offset if start > 0 and offset else 0
        remaining = max(0, num_tokens - current_page_space)
        pages_needed = (remaining + page_size - 1) // page_size
        if pages_needed > len(self.slot_allocator.free_pages):
            raise RuntimeError("no free KV-cache pages")

        full_slots = []
        position = start
        while len(full_slots) < num_tokens:
            offset = position % page_size
            if offset == 0:
                page_id = self.slot_allocator.allocate_page()
                first_slot = page_id * page_size
            else:
                previous_slot = self.request_pool.req_to_token[
                    request_index,
                    position - 1,
                ].item()
                first_slot = previous_slot + 1

            count = min(num_tokens - len(full_slots), page_size - offset)
            full_slots.extend(range(first_slot, first_slot + count))
            position += count

        full_slots = torch.tensor(
            full_slots,
            dtype=torch.long,
            device=self.request_pool.req_to_token.device,
        )
        self.request_pool.append(request_index, full_slots)
        return full_slots

    def free_request(self, request_index):
        full_slots = self.request_pool.free_request(request_index)
        if full_slots.numel() > 0:
            page_ids = torch.unique(
                full_slots // self.slot_allocator.page_size
            ).tolist()
            for page_id in page_ids:
                self.slot_allocator.free_page(page_id)
        self.kv_pool.reset_request(request_index)
        return full_slots
