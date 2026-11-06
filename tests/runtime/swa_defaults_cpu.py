# Copyright (c) 2026 Hygon Information Technology Co., Ltd.
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

import abc
import importlib.util
from pathlib import Path
import sys
from types import ModuleType
import unittest
from unittest.mock import patch

import torch
from hcu_sglang.mem_cache import swa
from sglang.srt.plugins.hook_registry import HookRegistry

ROOT = Path(__file__).resolve().parents[2]
MODULE = "sglang.srt.mem_cache.base_swa_memory_pool"


class SWADefaultsTests(unittest.TestCase):
    def setUp(self):
        fixture = ModuleType("sglang.srt.mem_cache.memory_pool")
        fixture.KVCache = abc.ABC
        source = ROOT.parent / "input-code/sglang/python/sglang/srt/mem_cache/base_swa_memory_pool.py"
        spec = importlib.util.spec_from_file_location(MODULE, source)
        self.native = importlib.util.module_from_spec(spec)
        with patch.dict(sys.modules, {fixture.__name__: fixture}):
            spec.loader.exec_module(self.native)
        self.modules = patch.dict(sys.modules, {MODULE: self.native})
        self.modules.start()
        HookRegistry.reset()
        base = self.native.BaseSWAKVPool

        class Pool(base):
            def register_mapping(self, mapping):
                self.mapping = mapping

            def translate_loc_from_full_to_swa(self, indices):
                return indices

            def get_state_buf_infos(self):
                return [], [], []

        self.pool_class = Pool

    def tearDown(self):
        HookRegistry.reset()
        self.modules.stop()

    def register(self, hcu):
        with patch.object(swa, "is_hcu", return_value=hcu):
            swa.register_hooks()

    def test_existing_instances_receive_default_and_keep_class_identity(self):
        pool = self.pool_class()
        base = self.native.BaseSWAKVPool
        method = base.translate_loc_from_full_to_swa
        with self.assertRaises(AttributeError):
            _ = pool.request_window
        self.register(True)
        self.assertIs(self.native.BaseSWAKVPool, base)
        self.assertIs(base.translate_loc_from_full_to_swa, method)
        self.assertIs(type(pool), self.pool_class)
        self.assertIsNone(pool.request_window)
        self.assertEqual(base.__abstractmethods__, frozenset({"register_mapping", "translate_loc_from_full_to_swa", "get_state_buf_infos"}))

    def test_non_hcu_preserves_baseline(self):
        self.register(False)
        self.assertFalse(hasattr(self.native.BaseSWAKVPool, "request_window"))
        with self.assertRaises(AttributeError):
            _ = self.pool_class().request_window

    def test_adapter_value_is_retained_on_repeated_registration(self):
        value = object()
        self.native.BaseSWAKVPool.request_window = value
        self.register(True)
        self.register(True)
        self.assertIs(self.pool_class().request_window, value)

    def test_subclass_and_instance_overrides_remain_private(self):
        marker = object()
        self.pool_class.request_window = marker
        self.register(True)
        self.assertIs(self.pool_class().request_window, marker)
        first, second = self.pool_class(), self.pool_class()
        first.request_window = object()
        self.assertIs(second.request_window, marker)
        self.assertIsNone(self.native.BaseSWAKVPool.request_window)

    def test_gpu_was_not_initialized(self):
        self.assertFalse(torch.cuda.is_initialized())


if __name__ == "__main__":
    unittest.main(verbosity=2)
