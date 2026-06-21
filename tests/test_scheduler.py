from types import SimpleNamespace

import torch

from nanovllm_ds4.engine import Scheduler


class FakeModel(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.weight = torch.nn.Parameter(torch.zeros(1))
        self.config = SimpleNamespace(
            compress_ratios=[0],
            sliding_window=4,
            head_dim=2,
            index_head_dim=2,
        )
        self.prefill_shapes = []
        self.decode_shapes = []

    def setup_caches(self, max_batch_size, max_seq_len):
        pass

    def _logits(self, input_ids):
        logits = torch.zeros(*input_ids.shape, 8)
        next_ids = (input_ids + 1) % 8
        return logits.scatter_(-1, next_ids.unsqueeze(-1), 1)

    def forward_inference(self, input_ids, cache_manager, request_index):
        self.prefill_shapes.append(tuple(input_ids.shape))
        return self._logits(input_ids)

    def forward_decode(self, input_ids, cache_manager, request_indices):
        self.decode_shapes.append(tuple(input_ids.shape))
        return self._logits(input_ids)


def test_scheduler_batches_decode_and_shrinks_the_batch():
    model = FakeModel()
    scheduler = Scheduler(model, max_requests=2, max_seq_len=5)
    scheduler.add_request(torch.tensor([[1, 2]]), max_new_tokens=3)
    scheduler.add_request(torch.tensor([[4]]), max_new_tokens=2)

    outputs = scheduler.run()

    assert torch.equal(outputs[0], torch.tensor([[1, 2, 3, 4, 5]]))
    assert torch.equal(outputs[1], torch.tensor([[4, 5, 6]]))
    assert model.prefill_shapes == [(1, 2), (1, 1)]
    assert model.decode_shapes == [(2, 1), (1, 1)]
