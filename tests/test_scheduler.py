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
        self.prefill_batches = []
        self.prefill_cache_slots = []
        self.decode_shapes = []

    def setup_caches(self, max_batch_size, max_seq_len):
        pass

    def _logits(self, input_ids):
        logits = torch.zeros(*input_ids.shape, 8)
        next_ids = (input_ids + 1) % 8
        return logits.scatter_(-1, next_ids.unsqueeze(-1), 1)

    def forward_packed_prefill(self, batch, cache_manager):
        seq_lens = batch.seq_lens.tolist()
        offsets = batch.offsets.tolist()
        self.prefill_batches.append(
            (seq_lens, offsets, batch.start_positions.tolist())
        )
        last_tokens = []
        for batch_index in range(len(seq_lens)):
            start = offsets[batch_index]
            end = offsets[batch_index + 1]
            self.prefill_cache_slots.append(
                (
                    batch.request_indices[batch_index].item(),
                    batch.full_slots[start].item(),
                )
            )
            last_tokens.append(batch.packed_tokens[end - 1])
        return self._logits(torch.stack(last_tokens).unsqueeze(1))

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
    assert model.prefill_batches == [([2, 1], [0, 2, 3], [0, 0])]
    assert model.decode_shapes == [(2, 1), (1, 1)]


def test_scheduler_admits_a_new_request_after_cache_is_freed():
    model = FakeModel()
    scheduler = Scheduler(model, max_requests=1, max_seq_len=3)
    first = scheduler.add_request(torch.tensor([[1]]), max_new_tokens=2)

    assert scheduler.step() == []
    second = scheduler.add_request(torch.tensor([[5]]), max_new_tokens=1)

    assert scheduler.step() == [first]
    assert scheduler.pending_requests == [second]
    assert scheduler.step() == [second]

    assert torch.equal(first.output_ids, torch.tensor([[1, 2, 3]]))
    assert torch.equal(second.output_ids, torch.tensor([[5, 6]]))
    assert model.prefill_cache_slots == [(0, 0), (0, 0)]
    assert not scheduler.has_requests


def test_chunked_prefill_does_not_block_a_short_request():
    model = FakeModel()
    scheduler = Scheduler(
        model,
        max_requests=2,
        max_seq_len=7,
        max_prefill_tokens=4,
    )
    long_request = scheduler.add_request(
        torch.tensor([[1, 2, 3, 4, 5, 6]]),
        max_new_tokens=1,
    )
    short_request = scheduler.add_request(
        torch.tensor([[4, 5]]),
        max_new_tokens=1,
    )

    assert scheduler.step() == [short_request]
    assert long_request.prefilled_tokens == 2
    assert scheduler.step() == [long_request]

    assert model.prefill_batches == [
        ([2, 2], [0, 2, 4], [0, 0]),
        ([4], [0, 4], [2]),
    ]
    assert torch.equal(
        long_request.output_ids,
        torch.tensor([[1, 2, 3, 4, 5, 6, 7]]),
    )
    assert torch.equal(short_request.output_ids, torch.tensor([[4, 5, 6]]))
