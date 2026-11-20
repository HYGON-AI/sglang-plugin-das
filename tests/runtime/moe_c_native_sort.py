import pytest
import torch

from hcu_sglang.layers.moe.moe_c_native_sort import moe_c_native_sort


@pytest.mark.skipif(not torch.cuda.is_available() or torch.version.hip is None, reason="DTK required")
@pytest.mark.parametrize("tokens", [1, 4, 17, 129, 1025])
@pytest.mark.parametrize("block", [16, 32, 64, 128])
@pytest.mark.parametrize("experts", [129, 257])
def test_all_routes_including_repeated_remote_routes_are_preserved(tokens, block, experts):
    torch.manual_seed(tokens)
    drop_id = experts - 1
    ids = torch.stack([torch.randperm(drop_id, device="cuda")[:10] for _ in range(tokens)]).to(torch.int32)
    ids[:, 3:] = drop_id
    expert_map = torch.arange(experts, dtype=torch.int32, device="cuda")
    expert_map[-1] = -1
    flat, sorted_experts, count = moe_c_native_sort(ids, block, experts, expert_map)
    n = count.item()
    routes = flat[:n].cpu()
    mask = routes < tokens * 10
    assert torch.equal(routes[mask].sort().values, torch.arange(tokens * 10))
    expected = ids.flatten().cpu()[routes[mask]].long()
    expected[expected == drop_id] = -1
    assert torch.equal(sorted_experts.repeat_interleave(block)[:n].cpu()[mask], expected)
    assert torch.all(flat[n:] == tokens * 10)
