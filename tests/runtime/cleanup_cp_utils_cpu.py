# Copyright (c) 2026 Hygon Information Technology Co., Ltd.
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

from types import SimpleNamespace as S
import unittest
from unittest.mock import patch

import torch
from hcu_sglang.mem_cache import cleanup
from hcu_sglang.mem_cache.cp_cache_layer_split import utils
from hcu_sglang.mem_cache.cp_cache_layer_split.pool_base import CpCacheLayerSplitPoolBase


class CleanupCpuTests(unittest.TestCase):
    def request(self):
        return S(rid="cpu", external_kv_abort_requested=True, external_kv_cleanup_steps={})

    def test_completed_step_preserves_result_identity_and_frees_once(self):
        req, calls, result = self.request(), [], object()
        def action():
            calls.append("free")
            return result
        self.assertIs(cleanup.run_cleanup_step(req, "kv", action), result)
        self.assertIs(cleanup.run_cleanup_step(req, "kv", action), result)
        self.assertEqual(calls, ["free"])
        self.assertEqual(req.external_kv_cleanup_steps["kv"], ("done", result))

    def test_unknown_side_effect_is_quarantined_without_second_free(self):
        req, calls = self.request(), []
        def action():
            calls.append("free")
            raise RuntimeError("allocator outcome unknown")
        with self.assertRaisesRegex(RuntimeError, "allocator outcome unknown"):
            cleanup.run_cleanup_step(req, "kv", action)
        with self.assertRaisesRegex(cleanup.CleanupOutcomeUnknown, "cleanup step kv"):
            cleanup.run_cleanup_step(req, "kv", action)
        self.assertEqual(calls, ["free"])
        self.assertEqual(req.external_kv_cleanup_steps["kv"], ("unknown", None))

    def test_explicit_side_effect_free_failure_can_retry(self):
        req, calls = self.request(), []
        def action():
            calls.append("try")
            if len(calls) == 1:
                raise cleanup.RetryableCleanupError("no side effects")
            return 7
        with self.assertRaises(cleanup.RetryableCleanupError):
            cleanup.run_cleanup_step(req, "kv", action)
        self.assertNotIn("kv", req.external_kv_cleanup_steps)
        self.assertEqual(cleanup.run_cleanup_step(req, "kv", action), 7)
        self.assertEqual(len(calls), 2)

    def test_non_abort_runs_each_action_without_cleanup_state(self):
        req, calls = S(), []
        for _ in range(2):
            cleanup.run_cleanup_step(req, "kv", lambda: calls.append("free"))
        self.assertEqual(calls, ["free", "free"])
        self.assertFalse(hasattr(req, "external_kv_cleanup_steps"))

    def test_reentrant_free_is_blocked_and_independent_steps_remain_available(self):
        req, calls = self.request(), []
        def action():
            with self.assertRaises(cleanup.CleanupOutcomeUnknown):
                cleanup.run_cleanup_step(req, "kv", lambda: calls.append("double free"))
            return "done"
        self.assertEqual(cleanup.run_cleanup_step(req, "kv", action), "done")
        self.assertEqual(cleanup.run_cleanup_step(req, "metadata", lambda: 9), 9)
        self.assertEqual(calls, [])


class CpUtilsCpuTests(unittest.TestCase):
    def test_balanced_partitions_cover_every_layer_once(self):
        for total in range(21):
            for size in range(1, 9):
                ranges = [utils.get_layer_shard_range(rank, size, total) for rank in range(size)]
                self.assertEqual([x for start, end in ranges for x in range(start, end)], list(range(total)))
                widths = [end - start for start, end in ranges]
                self.assertLessEqual(max(widths) - min(widths), 1)
                self.assertEqual(widths, sorted(widths, reverse=True))
                for rank, (start, end) in enumerate(ranges):
                    for layer in range(start, end):
                        self.assertEqual(utils.get_layer_owner(layer, size, total), rank)

    def test_real_pool_base_consumes_global_ranges_and_owner_mapping(self):
        for size in range(2, 9):
            for total in range(1, 21):
                for rank in range(size):
                    pool = CpCacheLayerSplitPoolBase()
                    pool._init_cp_cache_layer_split(cp_rank=rank, cp_size=size, layer_shard_start_layer=11, layer_shard_layer_num=total)
                    expected = utils.get_global_layer_shard_range(rank, size, 11, total)
                    self.assertEqual(pool._owned_global_layer_range(), expected)
                    self.assertEqual(list(pool._build_owned_layer_local_index_map()), list(range(*expected)))
                    for layer in range(11, 11 + total):
                        self.assertEqual(pool._is_layer_owned(layer), expected[0] <= layer < expected[1])
                        self.assertEqual(pool._get_layer_owner_rank(layer), utils.get_layer_owner(layer - 11, size, total))
                    with self.assertRaises(ValueError):
                        pool._local_layer_idx(10)

    def test_invalid_stage_rank_and_missing_owner_raise(self):
        for args in [(-1, 2, 0, 4), (2, 2, 0, 4), (0, 0, 0, 4), (0, 2, -1, 4), (0, 2, 0, 0)]:
            with self.subTest(args=args), self.assertRaises(ValueError):
                utils.get_global_layer_shard_range(*args)
        for layer in (-1, 5):
            with self.assertRaises(ValueError):
                utils.get_layer_owner(layer, 2, 5)

    def test_config_gate_uses_explicit_parallel_state_fixture(self):
        for draft in (False, True):
            for enabled in (False, True):
                runner = S(is_draft_worker=draft, server_args=S(enable_cp_cache_layer_split=enabled))
                for size in (1, 4):
                    with patch.object(utils, "get_parallel", return_value=S(attn_cp_rank=2, attn_cp_size=size)) as getter:
                        expected = (2, 4) if not draft and enabled and size == 4 else (None, 1)
                        self.assertEqual(utils.get_cp_cache_layer_shard_info(runner), expected)
                        self.assertEqual(getter.call_count, int(not draft and enabled))


if __name__ == "__main__":
    suite = unittest.defaultTestLoader.loadTestsFromModule(__import__(__name__))
    result = unittest.TextTestRunner(verbosity=2).run(suite)
    if torch.cuda.is_initialized():
        raise AssertionError("CPU helper tests initialized CUDA")
    raise SystemExit(not result.wasSuccessful())
