# SPDX-License-Identifier: Apache-2.0
"""HCU indexed V4.1 query preparation at the real indexer and dense prefill seams."""

from types import SimpleNamespace
from unittest.mock import Mock

import pytest
import torch


@pytest.fixture
def runtime(monkeypatch):
    from hcu_sglang.utils.detection import is_hcu
    if not is_hcu():
        pytest.skip("requires an HCU")
    from hcu_sglang.environ import register_envs
    from hcu_sglang.models.deepseek.current_v4.hcu_backend import dsv41_sparse
    register_envs()
    if dsv41_sparse._indexer_prep_op() is None:
        pytest.skip("LightOp indexed V4.1 query kernel unavailable")
    monkeypatch.setenv("SGLANG_HCU_OPT_INDEXER_PREP", "1")
    return dsv41_sparse


def inputs(rows, dtype):
    rng = torch.Generator(device="cuda").manual_seed(619)
    x = torch.randn((rows, 32, 128), device="cuda", dtype=torch.bfloat16, generator=rng)
    phase = torch.randn((137, 32), device="cuda", generator=rng)
    freqs = torch.polar(torch.ones_like(phase), phase)
    pos = torch.randint(0, 137, (rows,), device="cuda", dtype=dtype, generator=rng)
    return x, freqs, pos


def reference(mod, x, freqs, pos):
    return mod.fake_quant_fp4(mod.rope_tail(x, freqs[pos], 64))


@pytest.mark.parametrize("rows,dtype", [(0, torch.int64), (1, torch.int32), (37, torch.int64), (128, torch.int32)])
def test_queries_match_torch(runtime, monkeypatch, rows, dtype):
    mod = runtime
    x, freqs, pos = inputs(rows, dtype)
    expected = reference(mod, x, freqs, pos) if rows else x.clone()
    saved = [t.clone() for t in (x, freqs, pos)]
    native = Mock(wraps=mod._indexer_prep_op())
    monkeypatch.setattr(mod, "_indexer_prep_op", lambda: native)
    indexer = SimpleNamespace(wq_b=lambda t: (t, None), n_local_heads=32,
                              index_head_dim=128, rope_head_dim=64)
    result = mod.DeepseekV41Indexer.queries(indexer, x, freqs, positions=pos)
    native.assert_called_once()
    _, actual_freqs, actual_positions = native.call_args.args
    assert actual_freqs is freqs and actual_positions is pos
    torch.testing.assert_close(result, expected, atol=0, rtol=0)
    assert result.is_contiguous() and result.dtype == torch.bfloat16
    for src, original in zip((x, freqs, pos), saved):
        torch.testing.assert_close(src, original, atol=0, rtol=0)


def test_graph_replay_with_changed_input_and_positions(runtime):
    mod = runtime
    x, freqs, pos = inputs(33, torch.int64)
    stream = torch.cuda.Stream()
    stream.wait_stream(torch.cuda.current_stream())
    with torch.cuda.stream(stream):
        mod._rope_fq4(x, freqs, 64, positions=pos)
        graph = torch.cuda.CUDAGraph()
        with torch.cuda.graph(graph, stream=stream):
            result = mod._rope_fq4(x, freqs, 64, positions=pos)
        for _ in range(2):
            x.normal_()
            pos.random_(freqs.shape[0])
            graph.replay()
            torch.testing.assert_close(result, reference(mod, x, freqs, pos), atol=0, rtol=0)
    torch.cuda.current_stream().wait_stream(stream)


def test_quick_disable_and_unsupported_rows(runtime, monkeypatch):
    mod = runtime
    x, freqs, pos = inputs(7, torch.int64)
    native = Mock(wraps=mod._indexer_prep_op())
    monkeypatch.setattr(mod, "_indexer_prep_op", lambda: native)
    monkeypatch.setenv("SGLANG_HCU_OPT_INDEXER_PREP", "0")
    torch.testing.assert_close(mod._rope_fq4(x, freqs, 64, positions=pos),
                               reference(mod, x, freqs, pos), atol=0, rtol=0)
    monkeypatch.setenv("SGLANG_HCU_OPT_INDEXER_PREP", "1")
    for q in (x[:, 0], x[:, ::2]):
        got = mod._rope_fq4(q, freqs, 64, positions=pos)
        torch.testing.assert_close(got, reference(mod, q, freqs, pos), atol=0, rtol=0)
    got = mod._rope_fq4(x, freqs, 64, compressed_kv=True, positions=pos)
    compressed = mod.fake_quant_compressed_kv(mod.rope_tail(x, freqs[pos], 64))
    torch.testing.assert_close(got, compressed, atol=0, rtol=0)
    native.assert_not_called()
    monkeypatch.setattr(mod, "_indexer_prep_op", lambda: None)
    torch.testing.assert_close(mod._rope_fq4(x, freqs, 64, positions=pos),
                               reference(mod, x, freqs, pos), atol=0, rtol=0)


@pytest.mark.parametrize("rows,heads,index_dtype", [(0, 32, torch.int32), (1, 64, torch.int64), (32, 32, torch.int32)])
def test_decode_pack_and_head_weights_match_separate_ops(runtime, rows, heads, index_dtype):
    from hcu_sglang.models.deepseek.current_v4.hcu_backend import fp4_indexer

    q, freqs, pos = inputs(rows, index_dtype)
    if heads == 64:
        q = q.repeat(1, 2, 1)
    raw = torch.linspace(-3., 3., rows * heads, device="cuda", dtype=torch.bfloat16).reshape(rows, heads)
    factor = 0.017
    payload, scales, weights = fp4_indexer.index_q_rope_pack_weights_hcu(q, freqs, pos, raw, factor)
    assert (payload.shape, scales.shape, weights.shape) == ((rows * heads, 64), (rows * heads,), (rows, heads))
    assert (payload.dtype, scales.dtype, weights.dtype) == (torch.int8, torch.int32, torch.float32)
    if rows:
        grid = reference(runtime, q, freqs, pos)
        base_payload, base_scales = fp4_indexer.quantize_fp4_indexer_tensor(grid.flatten(0, 1), rne=True)
        torch.testing.assert_close(payload, base_payload, rtol=0, atol=0)
        torch.testing.assert_close(scales, base_scales.flatten(), rtol=0, atol=0)
        torch.testing.assert_close(weights, (raw * factor).float(), rtol=0, atol=0)


def test_dense_caller_preserves_full_table_and_real_query_positions(runtime, monkeypatch):
    from hcu_sglang.models.deepseek.current_v4.hcu_backend import backend

    q, freqs, pos = inputs(7, torch.int64)
    sent = []

    class BoundaryReached(Exception):
        pass

    def queries(projected, table, positions=None):
        sent.append((projected, table, positions))
        raise BoundaryReached()

    indexer = SimpleNamespace(uses_candidates=False, is_candidate_source=False, queries=queries)
    layer = SimpleNamespace(layer_id=0, compress_ratio=1, freqs_cis=freqs, indexer=indexer)
    sparse = torch.zeros(7, 32, dtype=torch.int32, device=q.device)
    core = SimpleNamespace(sparse_page_indices=lambda _: sparse, sparse_raw_indices=lambda _: None)
    gather = Mock(return_value=(None, None))
    pool = SimpleNamespace(get_low_ratio_index_k_fp4=gather)
    owner = SimpleNamespace(is_dsv41=True, forward_metadata=SimpleNamespace(core_metadata=core), token_to_kv_pool=pool,
                            req_to_token=torch.arange(137, device=q.device).view(1, -1))
    batch = SimpleNamespace(seq_lens_cpu=[6], req_pool_indices=torch.tensor([0], device=q.device))
    for enabled in (True, False):
        owner.fuse_indexer_q_prep = enabled
        monkeypatch.setattr(backend, "_hcu_mqa_prep_available", lambda: enabled)
        with pytest.raises(BoundaryReached):
            backend.DeepseekV4AttnBackend._low_ratio_index_topk_dense(
                owner, layer, q, q, pos, batch, torch.tensor([6], device=q.device), [6])
        assert gather.call_args.kwargs["use_hcu_fused_gather"] == enabled
        _, actual_table, actual_pos = sent[-1]
        if enabled:
            assert actual_table is freqs
            torch.testing.assert_close(actual_pos, pos[:6])
        else:
            assert actual_pos is None
            torch.testing.assert_close(actual_table, freqs[pos[:6]])


def test_decode_graph_keeps_live_q_positions_and_head_weights(runtime):
    from hcu_sglang.models.deepseek.current_v4.hcu_backend import fp4_indexer

    q, freqs, pos = inputs(32, torch.int64)
    raw = torch.empty((32, 32), dtype=torch.bfloat16, device=q.device).normal_()
    factor = 0.125
    stream = torch.cuda.Stream()
    stream.wait_stream(torch.cuda.current_stream())
    with torch.cuda.stream(stream):
        fp4_indexer.index_q_rope_pack_weights_hcu(q, freqs, pos, raw, factor)
        graph = torch.cuda.CUDAGraph()
        with torch.cuda.graph(graph, stream=stream):
            payload, scales, weights = fp4_indexer.index_q_rope_pack_weights_hcu(q, freqs, pos, raw, factor)
        for _ in range(2):
            q.normal_()
            raw.normal_()
            pos.random_(freqs.shape[0])
            graph.replay()
            grid = reference(runtime, q, freqs, pos)
            ref_payload, ref_scales = fp4_indexer.quantize_fp4_indexer_tensor(grid.flatten(0, 1), rne=True)
            torch.testing.assert_close(payload, ref_payload, rtol=0, atol=0)
            torch.testing.assert_close(scales, ref_scales, rtol=0, atol=0)
            torch.testing.assert_close(weights, (raw * factor).float(), rtol=0, atol=0)
    torch.cuda.current_stream().wait_stream(stream)


def test_decode_caller_selects_packer_before_python_fallback(runtime, monkeypatch):
    from hcu_sglang.models.deepseek.current_v4.hcu_backend import backend, fp4_indexer
    from sglang.srt.model_executor.runner_utils import capture_mode

    q, freqs, pos = inputs(32, torch.int64)
    raw = torch.ones((32, 32), device=q.device, dtype=q.dtype)

    class BoundaryReached(Exception):
        pass

    packed = Mock(side_effect=BoundaryReached)
    ordinary = Mock(side_effect=BoundaryReached)
    monkeypatch.setattr(fp4_indexer, "index_q_rope_pack_weights_hcu", packed)
    monkeypatch.setattr(capture_mode, "skip_low_ratio_indexer", lambda _: False)
    indexer = SimpleNamespace(
        n_heads=32, n_local_heads=32, index_head_dim=128, rope_head_dim=64,
        wq_b=lambda _: (q, None), head_weights_raw=lambda _: raw, head_weight_scale=.125,
        queries=ordinary,
    )
    owner = SimpleNamespace(token_to_kv_pool=object(), forward_metadata=SimpleNamespace(
        core_metadata=object(), c1_indexer_metadata=object()))
    layer = SimpleNamespace(compress_ratio=1, freqs_cis=freqs, indexer=indexer)
    for enabled, has_aot in [(True, False), (False, False), (True, True)]:
        monkeypatch.setenv("SGLANG_HCU_INDEX_Q_ROPE_PACK", "1" if enabled else "0")
        monkeypatch.setattr(backend, "_has_native_hcu_rope_fake_quant", lambda: has_aot)
        with pytest.raises(BoundaryReached):
            backend.DeepseekV4AttnBackend._low_ratio_index_topk_decode(owner, layer, q, q, pos)
    packed.assert_called_once()
    assert ordinary.call_count == 2


@pytest.mark.parametrize("device", ["cpu", "cuda"])
@pytest.mark.parametrize("ratio", [1, 2])
def test_partial_extend_slot_order_and_real_packed_gather(runtime, device, ratio):
    from hcu_sglang.mem_cache.hcu_v4_pool import DeepSeekV4IndexerPool
    from hcu_sglang.models.deepseek.current_v4.hcu_backend.backend import _extend_k_slots

    # Multiple request rows share non-contiguous pool pages and one has no
    # completed compressed token. The last extend can have fewer than eight
    # new tokens even when many compressed keys are already in the cache.
    lens = [35, 0, 12, 1]
    requests = [3, 0, 2, 1]
    width = 80
    req = torch.tensor(requests, dtype=torch.int64, device=device)
    mapping = (torch.arange(4 * width, device=device, dtype=torch.int32)
               .view(4, width).flip([1]) + 1)
    slots, starts = _extend_k_slots(mapping, ratio, lens, req)
    expected = torch.cat([mapping[r, :n * ratio:ratio].long() // ratio
                          for r, n in zip(requests, lens)])
    assert starts == [0, 35, 35, 47]
    torch.testing.assert_close(slots, expected, atol=0, rtol=0)
    empty, starts = _extend_k_slots(mapping, ratio, [0, 0], req[:2])
    assert not empty.numel() and starts == [0, 0]
    if device == "cuda":
        page = 64
        buf = (torch.arange(6 * page * 68, device=device) * 17 + 23).to(torch.uint8).view(6, -1)
        pool = SimpleNamespace(use_fp4_indexer=True, start_layer=2, page_size=page,
                               index_k_with_scale_buffer=[buf])
        actual = DeepSeekV4IndexerPool.get_index_k_fp4(pool, 2, slots, use_hcu_fused_gather=True)
        page_ids, offsets = expected // page, expected % page
        payload = buf[page_ids[:, None], offsets[:, None] * 64 + torch.arange(64, device=device)].view(torch.int8)
        scales = buf[page_ids[:, None], page * 64 + offsets[:, None] * 4 + torch.arange(4, device=device)]
        scales = scales.contiguous().view(torch.int32).flatten()
        torch.testing.assert_close(actual[0], payload, rtol=0, atol=0)
        torch.testing.assert_close(actual[1], scales, rtol=0, atol=0)
