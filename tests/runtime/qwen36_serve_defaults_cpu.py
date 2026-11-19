# Copyright (c) 2026 Hygon Information Technology Co., Ltd.
# SPDX-License-Identifier: Apache-2.0

import os
import sys
from unittest.mock import patch

import pytest

from hcu_sglang import environ


FLAGS = (
    "SGLANG_ENABLE_SPEC_V2",
    "SGLANG_USE_FUSED_TOPK_SOFTMAX",
    "SGLANG_USE_LIGHTOP",
    "SGLANG_USE_CAUSAL_CONV1D",
    "SGLANG_USE_AITER_LINEAR_ATTN",
    "SGLANG_ROCM_USE_AITER_MOE",
)
MODELSCOPE = "SGLANG_USE_MODELSCOPE"
BF16 = "/model/qwen3.6/Qwen3.6-35B-A3B"
FP8 = BF16 + "-Channel-fp8"


@pytest.fixture(autouse=True)
def clean_flags(monkeypatch):
    monkeypatch.setattr(environ, "is_hcu", lambda: True)
    with patch.dict(os.environ):
        for flag in (*FLAGS, MODELSCOPE):
            os.environ.pop(flag, None)
        yield


@pytest.mark.parametrize(
    ("argv", "modelscope"),
    [
        (["sglang", "serve", "--model-path", BF16], None),
        (["sglang", "serve", FP8 + "/"], "1"),
        (["sglang", "serve", f"--model={FP8}"], "1"),
        (["/opt/sglang/launch_server.py", "--model-path", BF16], None),
    ],
)
def test_qwen_defaults_keep_explicit_disable(argv, modelscope, monkeypatch):
    monkeypatch.setattr(sys, "argv", argv)
    monkeypatch.setenv("SGLANG_USE_LIGHTOP", "0")
    environ.apply_qwen36_serve_defaults()
    assert [os.getenv(flag) for flag in FLAGS] == ["1", "1", "0", "1", "1", "1"]
    assert os.getenv(MODELSCOPE) == modelscope


@pytest.mark.parametrize(
    ("argv", "is_hcu"),
    [
        (["sglang", "serve", "--model-path", BF16 + "-W8A8"], True),
        (["sglang", "serve", "--model-path", BF16], False),
        (["client.py", "--model-path", FP8], True),
        (["multiprocessing.spawn"], True),
    ],
)
def test_defaults_leave_other_launches_alone(argv, is_hcu, monkeypatch):
    monkeypatch.setattr(sys, "argv", argv)
    monkeypatch.setattr(environ, "is_hcu", lambda: is_hcu)
    environ.apply_qwen36_serve_defaults()
    assert all(flag not in os.environ for flag in (*FLAGS, MODELSCOPE))
