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

        self.compressed_cache = []
        self.indexer_cache = []
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
        compressed_slot = full_slot // self.compress_ratios[layer_index]
        cache[compressed_slot].copy_(kv)

    def write_indexer(self, layer_index, full_slot, key):
        cache = self.indexer_cache[layer_index]
        if cache is None:
            raise RuntimeError("layer does not use indexer cache")
        compressed_slot = full_slot // self.compress_ratios[layer_index]
        cache[compressed_slot].copy_(key)

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
    """Own the KV-cache lifecycle for one active generation batch.

    The tensors remain in each attention layer:
      layer.attn.kv_cache: [batch, sliding_window + compressed_blocks, head_dim]
      compressor state:    [batch, partial_block_tokens, projected_head_dim]

    ``compressed_blocks`` is derived from the layer's config ratio, so the
    mini checkpoint naturally uses its C4 and C96 layouts.
    """

    def __init__(self, model, max_batch_size, max_seq_len):
        self.model = model
        self.model.setup_caches(max_batch_size, max_seq_len)

    def reset(self):
        self.model.reset_caches()

    def prefill(self, input_ids):
        # input_ids: [batch_size, prompt_length]
        self.reset()
        return self.model.forward_inference(input_ids, start_pos=0)

    def decode(self, input_ids):
        # input_ids: [batch_size, 1]
        return self.model.forward_inference(input_ids)
