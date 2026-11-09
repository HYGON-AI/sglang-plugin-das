# Copyright (c) 2026 Hygon Information Technology Co., Ltd.
# SPDX-License-Identifier: Apache-2.0

from types import SimpleNamespace
from unittest import mock

import pytest
import torch

from hcu_sglang import server_args_hooks as args_hooks
from hcu_sglang.layers.attention import flashattention_backend as subject
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
