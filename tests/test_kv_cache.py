import pytest
import torch

from nanovllm_ds4.engine.kv_cache import PagedSlotAllocator, RequestTokenPool


def test_request_token_pool_allocates_maps_and_reuses_request_slots():
    pool = RequestTokenPool(max_requests=2, max_seq_len=4)

    request_index = pool.allocate_request()
    pool.append(request_index, [96, 97, 98])

    assert request_index == 0
    assert pool.request_lengths == [3, -1]
    assert torch.equal(
        pool.req_to_token[request_index],
        torch.tensor([96, 97, 98, -1]),
    )

    with pytest.raises(RuntimeError, match="request exceeds max_seq_len"):
        pool.append(request_index, [99, 100])

    pool.allocate_request()
    with pytest.raises(RuntimeError, match="no free request slots"):
        pool.allocate_request()

    full_slots = pool.free_request(request_index)

    assert torch.equal(full_slots, torch.tensor([96, 97, 98]))
    assert torch.equal(pool.req_to_token[request_index], torch.full((4,), -1))
    assert pool.allocate_request() == request_index


def test_paged_slot_allocator_allocates_and_reuses_pages():
    allocator = PagedSlotAllocator(num_pages=2, page_size=96)

    first_page = allocator.allocate_page()
    second_page = allocator.allocate_page()

    assert first_page == 0
    assert list(allocator.page_slots(second_page)) == list(range(96, 192))

    with pytest.raises(RuntimeError, match="no free KV-cache pages"):
        allocator.allocate_page()

    allocator.free_page(first_page)

    assert allocator.allocate_page() == first_page
