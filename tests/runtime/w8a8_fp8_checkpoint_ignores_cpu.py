# Copyright (c) 2026 Hygon Information Technology Co., Ltd.
# SPDX-License-Identifier: Apache-2.0

import pytest
import torch

from hcu_sglang.layers.quantization import w8a8_fp8 as hooks
from hcu_sglang.utils import detection
from sglang.srt.layers.linear import LinearBase
from sglang.srt.layers.quantization import w8a8_fp8 as upstream
from sglang.srt.layers.quantization.unquant import UnquantizedLinearMethod
from sglang.srt.plugins.hook_registry import HookType, _wrap_fn


class LinearFixture(LinearBase):
    def __init__(self):
        torch.nn.Module.__init__(self)


@pytest.fixture
def configured(monkeypatch):
    cls = upstream.W8A8Fp8Config
    monkeypatch.setattr(detection, "is_hcu", lambda: True)
    monkeypatch.setattr(upstream, "cutlass_fp8_supported", lambda: False)
    build = _wrap_fn(cls.from_config.__func__, hooks.retain_checkpoint_ignores, HookType.AFTER)
    select = _wrap_fn(cls.get_quant_method, hooks.respect_checkpoint_ignores, HookType.AROUND)
    config = build(cls, config={
        "quant_method": "compressed-tensors",
        "ignore": ["linear_attn.in_proj_b", "linear_attn.in_proj_a", r"re:.*\.mtp\.fc$"],
    })
    return config, select


def test_serialized_checkpoint_preserves_ba_and_direct_ignores(configured):
    config, select = configured
    layer = LinearFixture()
    for prefix in ("model.layers.0.linear_attn.in_proj_ba", "model.mtp.fc"):
        assert isinstance(select(config, layer, prefix), UnquantizedLinearMethod)
    for prefix in ("model.layers.0.linear_attn.in_proj_qkvz", "model.mtp.fc_extra"):
        assert isinstance(select(config, layer, prefix), upstream.W8A8Fp8LinearMethod)


def test_unrelated_quant_config_and_non_hcu_keep_original(configured, monkeypatch):
    config, select = configured
    layer = LinearFixture()
    prefix = "model.layers.0.linear_attn.in_proj_ba"
    original_config = upstream.W8A8Fp8Config(is_checkpoint_fp8_serialized=True)
    assert isinstance(select(original_config, layer, prefix), upstream.W8A8Fp8LinearMethod)
    monkeypatch.setattr(detection, "is_hcu", lambda: False)
    assert isinstance(select(config, layer, prefix), upstream.W8A8Fp8LinearMethod)


def test_mixed_packed_shards_are_rejected(configured):
    config, select = configured
    config._hcu_checkpoint_ignore = ("linear_attn.in_proj_b",)
    with pytest.raises(ValueError, match="different quantization schemes"):
        select(config, LinearFixture(), "model.layers.0.linear_attn.in_proj_ba")
