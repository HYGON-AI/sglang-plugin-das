# Copyright (c) 2026 Hygon Information Technology Co., Ltd.
# SPDX-License-Identifier: Apache-2.0

"""The FA around-init override must keep the registry's wrapper chain."""

from __future__ import annotations

import sys
import types
import unittest
from unittest import mock

from hcu_sglang.layers.attention import flashattention_backend as fb


class AroundInitHookChainTest(unittest.TestCase):
    def test_uses_the_registered_module_functions(self):
        def varlen(*args, **kwargs):
            return "varlen"

        def kvcache(*args, **kwargs):
            return "kvcache"

        module = types.ModuleType(
            "sglang.srt.layers.attention.flashattention_backend"
        )
        module.flash_attn_varlen_func = varlen
        module.flash_attn_with_kvcache = kvcache
        backend = types.SimpleNamespace()
        with mock.patch.object(fb, "is_hcu", return_value=True), \
                mock.patch.dict(
                    sys.modules,
                    {"sglang.srt.layers.attention.flashattention_backend": module},
                ):
            fb.around_init(lambda b: None, backend)
        self.assertIs(backend.flash_attn_varlen_func, varlen)
        self.assertIs(backend.flash_attn_with_kvcache, kvcache)
