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
