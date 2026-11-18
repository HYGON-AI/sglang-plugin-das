# Copyright (c) 2026 Hygon Information Technology Co., Ltd.
# SPDX-License-Identifier: Apache-2.0

"""The draft head must own its quant method without mutating the target head."""

from __future__ import annotations

import types
import unittest
from unittest import mock

import torch

from hcu_sglang.models import mtp_lmhead_quant as mtp_head
from hcu_sglang.models import qwen3_5_mtp_lmhead_quant as qwen_mtp


class _Method:
    def __init__(self, head, *_args):
        self.head = head
        self.apply = lambda *a, **k: None


class DraftHeadHostTest(unittest.TestCase):
    def test_install_keeps_the_target_head_untouched(self):
        base = torch.nn.Module()
        base.weight = torch.nn.Parameter(torch.zeros(4, 4))
        base.quant_method = object()
        model = types.SimpleNamespace(lm_head=base)
        with mock.patch.object(
            mtp_head, "DraftHeadQuantMethod", _Method
        ):
            qwen_mtp._install_draft_head(
                model=model, head=base, weight_q=torch.zeros(4, 4),
                weight_scale=torch.ones(4), gemm=object(), max_m=4,
            )
        self.assertIsNot(model.lm_head, base)
        self.assertIs(base.quant_method, base.quant_method)
        self.assertIsNot(model.lm_head.quant_method, base.quant_method)
        self.assertIs(model.lm_head.weight, base.weight)

    def test_restore_returns_the_target_head(self):
        base = torch.nn.Module()
        base.weight = torch.nn.Parameter(torch.zeros(4, 4))
        base.quant_method = object()
        model = types.SimpleNamespace(lm_head=base)
        with mock.patch.object(
            mtp_head, "DraftHeadQuantMethod", _Method
        ):
            qwen_mtp._install_draft_head(
                model=model, head=base, weight_q=torch.zeros(4, 4),
                weight_scale=torch.ones(4), gemm=object(), max_m=4,
            )
        qwen_mtp._restore_quant_method(model)
        self.assertIs(model.lm_head, base)
        self.assertFalse(model._mtp_fp8_installed)
