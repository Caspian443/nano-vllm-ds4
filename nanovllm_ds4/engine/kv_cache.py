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
