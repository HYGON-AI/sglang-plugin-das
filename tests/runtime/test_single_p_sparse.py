import ast
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

import pytest
import torch

from sglang.kernels.ops.attention.dsv4.kv_layout import KVLayout
from hcu_sglang.models.deepseek.current_v4 import flat_kv_dequant as sparse


def isolated_backend_method(name, namespace):
    # Load the exact plugin method while the full backend has no initialized worker.
    backend = Path(sparse.__file__).parent / "hcu_backend/backend.py"
    func = next(n for n in ast.walk(ast.parse(backend.read_text()))
                if isinstance(n, ast.FunctionDef) and n.name == name)
    exec(compile(ast.Module(body=[func], type_ignores=[]), str(backend), "exec"), namespace)
    return namespace[name]


def test_cross_request_budget_keeps_geometry():
    calls = []
    def logits(q, kv, weights, ks, ke, width, *, candidate_query_lens, candidate_blocks, **kwargs):
        calls.append(candidate_query_lens)
        for rows, block_ids in zip(candidate_query_lens, candidate_blocks):
            assert rows == block_ids.shape[0]
        return torch.arange(width, dtype=torch.float32).expand(q[0].shape[0], -1)

    def topk(logits, lengths, out_offsets, out_indices):
        out_indices.copy_(out_offsets[:, None])

    def postprocess(logits, selected, ks, slots, pages, raw, masked):
        pages.copy_(slots[selected.long()])
        raw.copy_(selected - ks[:, None])

    namespace = dict(torch=torch, _is_hcu=False, ceil_align=lambda n, a: (n + a - 1) // a * a,
                     published_masks=lambda meta: None, _hcu_dense_fp4_mqa_logits=logits,
                     topk_transform_ragged_v2=topk)
    execute = isolated_backend_method("_low_ratio_index_topk_chunked", namespace)
    owner = SimpleNamespace(_dense_mqa_workspace=None, forward_metadata=SimpleNamespace(candidate_metadata=None))
    indexer = SimpleNamespace(index_topk=1, is_candidate_source=False, uses_candidates=True)
    lens, rows = [4, 3, 6], [3, 0, 4]
    qs = torch.zeros(10, 1)
    ks = torch.tensor([0, 0, 0, 7, 7, 7, 7], dtype=torch.int32)
    slots = torch.arange(16, dtype=torch.int32) + 100
    blocks = [torch.zeros(n, 1, dtype=torch.int32) for n in rows]
    pages = torch.full((10, 1), -1, dtype=torch.int32)
    raw = torch.full_like(pages, -1)
    with patch("lightop.attention.dsv41_ragged_topk_postprocess", side_effect=postprocess):
        execute(owner, indexer, qs, None, qs, ks, ks + 1, ks * 0 + 1,
                             slots, lens, rows, pages, raw, blocks, 4)
    assert calls == [[3, 1], [3]]
    assert pages.flatten().tolist() == [100] * 3 + [107] * 4 + [-1] * 3
    assert raw.flatten().tolist() == [0] * 7 + [-1] * 3


@pytest.mark.skipif(not torch.version.hip or not torch.cuda.is_available(), reason="HCU required")
def test_compact_and_selected_dequant_match_full_gather():
    from hcu_sglang.models.deepseek.current_v4.hcu_backend.dequant_k_cache import (
        PADDED_SCALE_PER_TOKEN, NOPE_ROPE_BYTES, dequantize_k_cache_paged,
    )
    from hcu_sglang.models.deepseek.current_v4.hcu_backend.sparse_prefill_utils import SparsePrefillChunkCache

    device = torch.device("cuda")
    page_size = 64
    width = page_size * (NOPE_ROPE_BYTES + PADDED_SCALE_PER_TOKEN)
    # Finite E4M3 activation bytes and exponent=127 yield an exact BF16 oracle.
    cache = torch.full((1, width), 0x38, dtype=torch.uint8, device=device)
    packed_bytes = cache[:, :page_size * NOPE_ROPE_BYTES].reshape(page_size, NOPE_ROPE_BYTES)
    packed_bytes.copy_((torch.arange(page_size, device=device) % 8 + 0x38).to(torch.uint8)[:, None])
    cache[:, page_size * NOPE_ROPE_BYTES:] = 127
    ids = torch.arange(8, dtype=torch.int32, device=device)
    owner = SimpleNamespace(swa_token_ids=ids[:2], compact_seq_lens_cpu=(8,))
    owner.pack_compressed_topk = lambda *args: SparsePrefillChunkCache.pack_compressed_topk(owner, *args)
    for rows in (1, 4):
        indices = torch.tensor([[3, 3, 8, -1]], dtype=torch.int32, device=device).repeat(rows, 1)
        if rows > 1:
            indices[1, :2] = torch.tensor([6, 2], dtype=torch.int32, device=device)
        lengths = torch.full((rows,), 3, dtype=torch.int32, device=device)
        packed, mapped, count, selected = sparse.plan_topk_rows(owner, ids, indices, lengths, 2, KVLayout.V4)
        dest = torch.full((len(packed), 1, 512), -17, dtype=torch.bfloat16, device=device)
        if selected:
            sparse.dequantize_selected_rows(cache, packed, mapped, lengths, dest, 2, page_size)
            assert torch.all(dest[0] == -17)
        else:
            assert count.item() == 1
            sparse.dequantize_sparse_prefill_rows(cache, packed, dest, page_size, KVLayout.V4,
                                                  lightop=True, fallback=None, valid_count=count)
            assert torch.all(dest[1:] == -17)
        oracle = dequantize_k_cache_paged(cache, ids, page_size, layout=KVLayout.V4)
        torch.testing.assert_close(dest[mapped[:, :2].long()], oracle[indices[:, :2].long()], rtol=0, atol=0)
        assert torch.all(mapped[:, 2] == len(packed))
        assert torch.all(mapped[:, 3] == -1)


def test_window_and_tail_publication_geometry():
    from hcu_sglang.models.deepseek.current_v4.hcu_backend.hcu_prefill_candidate import (
        cand_window_covers_all, get_late_tail_publish_target,
    )
    indexer = SimpleNamespace(candidate_topk_blocks=2048, candidate_block_size=8)
    assert cand_window_covers_all(indexer, [4096, 16384])
    assert not cand_window_covers_all(indexer, [4096, 16385])
    tail = SimpleNamespace(cp_metadata=object(), local_lens_cpu=[0, 5], extend_seq_lens_cpu=[0, 40])
    meta = SimpleNamespace(late_layer_tail=tail)
    owner = SimpleNamespace(forward_metadata=object(), tail_forward_metadata=meta)
    assert get_late_tail_publish_target(owner, [3, 8]) == (meta, [0, 5])
    assert get_late_tail_publish_target(owner, [3, 4]) is None

@pytest.mark.skipif(not torch.version.hip or not torch.cuda.is_available(), reason="HCU required")
@pytest.mark.parametrize("ratio,expected_ids,expected_bases", [
    (1, [8, 9, 10, 11, 16, 0, 0, 24, 25, 26, 27], [0, 5, 7]),
    (2, [8, 9, 0, 24, 25], [0, 2, 3]),
])
def test_compact_page_pack_preserves_request_offsets(ratio, expected_ids, expected_bases):
    from hcu_sglang.models.deepseek.current_v4.hcu_backend.sparse_prefill_utils import SparsePrefillChunkCache

    device = torch.device("cuda")
    owner = SimpleNamespace(
        compact_seq_lens_cpu=(5, 2, 4),
        seq_lens=torch.tensor([5, 2, 4], device=device, dtype=torch.int32),
        swa_offsets=torch.tensor([0, 5, 7, 11], device=device, dtype=torch.int32),
        swa_token_ids=torch.arange(11, device=device, dtype=torch.int32),
        num_reqs=3,
    )
    pages = torch.tensor([[2, 4], [-1, -1], [6, 0]], device=device, dtype=torch.int32)
    ids, base, swa = SparsePrefillChunkCache._pack_compact_compressed_ids(owner, pages, ratio, 4)
    assert ids.tolist() == expected_ids
    assert base.tolist() == expected_bases
    assert swa.tolist() == [len(expected_ids) + i for i in [0, 5, 7]]

@pytest.mark.skipif(not torch.version.hip or not torch.cuda.is_available(), reason="HCU required")
def test_candidate_tail_equals_full_publish(monkeypatch):
    import os
    from sglang.srt.environ import envs
    from hcu_sglang.environ import register_envs
    from hcu_sglang.models.deepseek.current_v4.hcu_backend.candidate_indexer import (
        CandidateMasks, select_candidate_blocks,
    )

    register_envs()
    monkeypatch.setenv("SGLANG_USE_LIGHTOP_DSV41_CANDIDATE_BLOCKS", "1")
    monkeypatch.setenv("SGLANG_HCU_OPT_DSV41_CANDIDATE_LOGITS", "1")
    kernel = isolated_backend_method("_publish_or_consume_candidates", dict(
        torch=torch, os=os, _is_hcu=True, _hcu_candidate_logits_available=lambda: True,
        CandidateMasks=CandidateMasks, envs=envs, select_candidate_blocks=select_candidate_blocks,
        _TORCH_INDEXER_SCORE_BUDGET_BYTES=1 << 20,
    ))
    counts = [3, 4]
    tail_lengths = [1, 2]
    tail = SimpleNamespace(cp_metadata=object(), local_lens_cpu=tail_lengths)
    owner = SimpleNamespace(forward_metadata=SimpleNamespace(candidate_metadata=None),
                            tail_forward_metadata=SimpleNamespace(late_layer_tail=tail))
    indexer = SimpleNamespace(is_candidate_source=True, candidate_block_size=8,
                              n_local_heads=32, candidate_topk_blocks=2)
    lens = torch.tensor([3, 12, 17, 19, 20, 21, 27], dtype=torch.int32, device="cuda")
    logits = torch.arange(7 * 64, dtype=torch.float, device="cuda").reshape(7, 64)
    empty = torch.empty((0, 0), dtype=torch.bool, device="cuda")
    monkeypatch.setenv("SGLANG_HCU_OPT_DSV41_PREFILL_CONSUMER_40352", "0")
    kernel(owner, indexer, logits.clone(), lens, [31, 40], counts, empty)
    full = owner.forward_metadata.candidate_metadata.request_block_ids
    monkeypatch.setenv("SGLANG_HCU_OPT_DSV41_PREFILL_CONSUMER_40352", "1")
    kernel(owner, indexer, logits.clone(), lens, [31, 40], counts, empty)
    assert owner.forward_metadata.candidate_metadata is None
    published = owner.tail_forward_metadata.candidate_metadata
    assert published.request_masks is None
    for got, reference, rows in zip(published.request_block_ids, full, tail_lengths):
        torch.testing.assert_close(got, reference[-rows:])
