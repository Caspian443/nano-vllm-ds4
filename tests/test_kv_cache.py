from types import SimpleNamespace

import pytest
import torch

from nanovllm_ds4.engine.kv_cache import (
    DeepseekV4KVPool,
    KVCacheManager,
    PagedSlotAllocator,
    RequestTokenPool,
)


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


@pytest.mark.parametrize(
    "device",
    [
        "cpu",
        pytest.param(
            "cuda",
            marks=pytest.mark.skipif(
                not torch.cuda.is_available(),
                reason="CUDA is not available",
            ),
        ),
    ],
)
def test_deepseek_v4_kv_pool_tensor_layout_and_writes(device):
    config = SimpleNamespace(
        compress_ratios=[0, 4, 96],
        sliding_window=64,
        head_dim=64,
        index_head_dim=64,
    )
    pool = DeepseekV4KVPool(
        config,
        max_requests=2,
        num_pages=2,
        device=device,
    )

    assert pool.swa_cache.shape == (3, 2, 64, 64)
    assert pool.swa_cache.dtype == torch.bfloat16
    assert pool.swa_cache.device.type == device

    assert pool.compressed_cache[0] is None
    assert pool.compressed_cache[1].shape == (48, 64)
    assert pool.compressed_cache[2].shape == (2, 64)
    assert pool.indexer_cache[1].shape == (48, 64)
    assert pool.indexer_cache[2] is None

    assert pool.kv_state[1].shape == (2, 8, 128)
    assert pool.kv_state[2].shape == (2, 96, 64)
    assert pool.kv_state[1].dtype == torch.float32
    assert pool.score_state[1].dtype == torch.float32

    kv = torch.arange(64, device=device).to(torch.bfloat16)
    pool.write_swa(0, request_index=1, token_position=65, kv=kv)
    pool.write_compressed(1, full_slot=99, kv=kv)
    pool.write_compressed(2, full_slot=191, kv=kv)
    pool.write_indexer(1, full_slot=99, key=kv)

    assert torch.equal(pool.swa_cache[0, 1, 1], kv)
    assert torch.equal(pool.compressed_cache[1][24], kv)
    assert torch.equal(pool.compressed_cache[2][1], kv)
    assert torch.equal(pool.indexer_cache[1][24], kv)

    pool.kv_state[1][1].fill_(1)
    pool.score_state[1][1].fill_(0)
    pool.indexer_kv_state[1][1].fill_(1)
    pool.indexer_score_state[1][1].fill_(0)
    pool.reset_request(1)

    assert torch.count_nonzero(pool.swa_cache[:, 1]).item() == 0
    assert torch.count_nonzero(pool.kv_state[1][1]).item() == 0
    assert torch.isneginf(pool.score_state[1][1]).all().item()
    assert torch.count_nonzero(pool.indexer_kv_state[1][1]).item() == 0
    assert torch.isneginf(pool.indexer_score_state[1][1]).all().item()
    assert torch.equal(pool.compressed_cache[1][24], kv)


@pytest.mark.parametrize(
    "device",
    [
        "cpu",
        pytest.param(
            "cuda",
            marks=pytest.mark.skipif(
                not torch.cuda.is_available(),
                reason="CUDA is not available",
            ),
        ),
    ],
)
def test_kv_cache_manager_allocates_frees_and_reuses_pages(device):
    config = SimpleNamespace(
        compress_ratios=[0, 4, 96],
        sliding_window=64,
        head_dim=64,
        index_head_dim=64,
    )
    manager = KVCacheManager(
        config,
        max_requests=2,
        max_seq_len=256,
        num_pages=3,
        device=device,
    )

    first_request = manager.allocate_request()
    first_slots = manager.allocate_tokens(first_request, 100)
    assert torch.equal(first_slots, torch.arange(100, device=device))

    second_request = manager.allocate_request()
    second_slots = manager.allocate_tokens(second_request, 2)
    assert torch.equal(second_slots, torch.tensor([192, 193], device=device))

    continued_slots = manager.allocate_tokens(first_request, 2)
    assert torch.equal(continued_slots, torch.tensor([100, 101], device=device))

    with pytest.raises(RuntimeError, match="no free KV-cache pages"):
        manager.allocate_tokens(first_request, 95)
    assert manager.request_pool.request_lengths[first_request] == 102

    manager.kv_pool.swa_cache[:, first_request].fill_(1)
    released_slots = manager.free_request(first_request)

    assert released_slots.numel() == 102
    assert manager.request_pool.request_lengths[first_request] == -1
    assert torch.count_nonzero(
        manager.request_pool.req_to_token[first_request] + 1
    ).item() == 0
    assert torch.count_nonzero(
        manager.kv_pool.swa_cache[:, first_request]
    ).item() == 0

    reused_request = manager.allocate_request()
    reused_slots = manager.allocate_tokens(reused_request, 97)
    assert reused_request == first_request
    assert torch.equal(reused_slots[:96], torch.arange(96, 192, device=device))
    assert reused_slots[-1].item() == 0
