# Copyright (c) 2026 Hygon Information Technology Co., Ltd.
# SPDX-License-Identifier: Apache-2.0

import sys
from types import ModuleType, SimpleNamespace
from unittest import mock

import pytest
import torch

from hcu_sglang import server_args_hooks as args_hooks
from hcu_sglang.layers.attention import flashattention_backend as subject
from hcu_sglang.layers.attention import flashattention_interface as interface
from sglang.srt.layers.attention.flashattention_backend import FlashAttentionBackend
from sglang.srt.plugins.hook_registry import HookRegistry, HookType


@pytest.mark.parametrize("dtype,expected", [(torch.bfloat16, True), (torch.float16, False), (None, False)])
def test_auto_dtype_uses_resolved_model_dtype(dtype, expected):
    config = SimpleNamespace(dtype=dtype)
    view = SimpleNamespace(dtype="auto", get_model_config=lambda: config)
    assert args_hooks._uses_bfloat16(view) is expected


@pytest.mark.parametrize("scale", [None, 0.75])
def test_registered_prepare_keeps_bf16_query_and_rejects_scales(monkeypatch, scale):
    monkeypatch.setattr(subject, "is_hcu", lambda: True)
    target = "sglang.srt.layers.attention.flashattention_backend.FlashAttentionBackend.prepare_paged_mha_query"
    registered = []
    monkeypatch.setattr(HookRegistry, "register", lambda *args: registered.append(args))
    subject.register()
    selected = [(kind, hook, None) for name, hook, kind in registered if name == target]
    assert selected == [(HookType.AROUND, subject.around_prepare_paged_mha_query, None)]
    original = FlashAttentionBackend.prepare_paged_mha_query
    monkeypatch.setattr(FlashAttentionBackend, "prepare_paged_mha_query", original)
    backend = SimpleNamespace(_hcu_bf16_fp8_kv_query=True, kv_cache_dtype_str="fp8_e5m2",
        kv_cache_dtype=torch.float8_e5m2, kv_cache_is_mxfp8=False, fa_impl_ver=3)
    value = torch.tensor(scale) if scale is not None else None
    layer = SimpleNamespace(head_dim=256, k_scale=value, v_scale=value)
    q = torch.randn(3, 12, 256, dtype=torch.bfloat16)
    old = original(backend, q, None, None, layer, logical_batch_size=2, kv_head_num=2, is_prefill=False)
    assert old[0].dtype == torch.float8_e5m2
    try:
        HookRegistry._apply_target(target, selected)
        if scale is not None:
            with pytest.raises(RuntimeError, match="requires unscaled KV cache"):
                FlashAttentionBackend.prepare_paged_mha_query(backend, q, None, None, layer,
                    logical_batch_size=2, kv_head_num=2, is_prefill=False)
        else:
            result = FlashAttentionBackend.prepare_paged_mha_query(backend, q, None, None, layer,
                logical_batch_size=2, kv_head_num=2, is_prefill=False)
            assert result[0] is q and result[0].dtype == torch.bfloat16
            assert result[1:] == (None, None, None, None)
    finally:
        FlashAttentionBackend.prepare_paged_mha_query = original


@pytest.mark.parametrize("case", ["non_hcu", "disabled", "float16", "large_head", "rope"])
def test_prepare_preserves_effective_fallback(monkeypatch, case):
    monkeypatch.setattr(subject, "is_hcu", lambda: case != "non_hcu")
    backend = SimpleNamespace(_hcu_bf16_fp8_kv_query=case != "disabled")
    q = torch.ones(1, 2, 256, dtype=torch.float16 if case == "float16" else torch.bfloat16)
    layer = SimpleNamespace(head_dim=384 if case == "large_head" else 256)
    rope = q if case == "rope" else None
    original = mock.Mock(return_value=object())
    result = subject.around_prepare_paged_mha_query(original, backend, q, rope, None, layer,
        logical_batch_size=1, kv_head_num=2, is_prefill=False)
    assert result is original.return_value
    original.assert_called_once_with(backend, q, rope, None, layer,
        logical_batch_size=1, kv_head_num=2, is_prefill=False)


@pytest.mark.parametrize("dtype", [torch.bfloat16, torch.float8_e5m2])
def test_ragged_prefill_reaches_varlen_without_rectangular_query_view(monkeypatch, dtype):
    q = torch.randn(7, 12, 256, dtype=torch.bfloat16)
    cu_q = torch.tensor([0, 2, 7], dtype=torch.int32)
    k = torch.zeros(4, 2, 64, 256, dtype=dtype)
    v = torch.zeros_like(k)
    expected = torch.ones_like(q)
    native = ModuleType("flash_attn")
    native.flash_attn_varlen_func = mock.Mock(return_value=expected)
    native.flash_attn_with_kvcache = mock.Mock(side_effect=AssertionError("unexpected fallback"))
    native.vllm_flash_attn_with_kvcache = mock.Mock(side_effect=AssertionError("unexpected decode"))
    monkeypatch.setitem(sys.modules, "flash_attn", native)
    monkeypatch.setattr(interface, "_layout_enabled", lambda: True)
    out = torch.empty_like(q)
    result = interface.flash_attn_with_kvcache(q, k.view(4, 64, 2, 256), v.view(4, 64, 2, 256),
        cache_seqlens=torch.tensor([20, 30], dtype=torch.int32), page_table=torch.tensor([[0], [1]], dtype=torch.int32),
        cu_seqlens_q=cu_q, max_seqlen_q=5, max_seqlen_k=30, causal=True, out=out)
    assert result is out
    torch.testing.assert_close(out, expected)
    kwargs = native.flash_attn_varlen_func.call_args.kwargs
    assert kwargs["q"] is q and kwargs["cu_seqlens_q"] is cu_q
    assert kwargs["k"].shape == kwargs["v"].shape == (4, 2, 64, 256)
    assert kwargs["layout"] == "bhsd" and kwargs["max_seqlen_q"] == 5
    native.flash_attn_varlen_func.assert_called_once()


@pytest.mark.parametrize("query_len", [1, 4])
def test_uniform_decode_and_verify_keep_native_query_contract(monkeypatch, query_len):
    q = torch.randn(2 * query_len, 12, 256, dtype=torch.bfloat16)
    k = torch.zeros(4, 2, 64, 256, dtype=torch.float8_e5m2)
    native = ModuleType("flash_attn")
    native.flash_attn_varlen_func = mock.Mock(return_value=q)
    native.vllm_flash_attn_with_kvcache = mock.Mock(return_value=q)
    native.flash_attn_with_kvcache = mock.Mock(side_effect=AssertionError("unexpected fallback"))
    monkeypatch.setitem(sys.modules, "flash_attn", native)
    monkeypatch.setattr(interface, "_layout_enabled", lambda: True)
    interface.flash_attn_with_kvcache(q, k.view(4, 64, 2, 256), k.view(4, 64, 2, 256),
        cache_seqlens=torch.tensor([20, 30], dtype=torch.int32), page_table=torch.tensor([[0], [1]], dtype=torch.int32),
        cu_seqlens_q=torch.tensor([0, query_len, 2 * query_len], dtype=torch.int32), max_seqlen_q=query_len, causal=True)
    if query_len == 1:
        native.vllm_flash_attn_with_kvcache.assert_called_once()
        assert native.vllm_flash_attn_with_kvcache.call_args.kwargs["q"].shape == (2, 1, 12, 256)
        native.flash_attn_varlen_func.assert_not_called()
    else:
        native.flash_attn_varlen_func.assert_called_once()
        assert native.flash_attn_varlen_func.call_args.kwargs["q"] is q
        native.vllm_flash_attn_with_kvcache.assert_not_called()
