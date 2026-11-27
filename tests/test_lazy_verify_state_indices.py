import sys
import unittest
from types import SimpleNamespace
from unittest.mock import patch

from hcu_sglang.layers.attention.linear.verify_state_indices import (
    defer_verify_valid_mask,
)


_UTILS_MODULE = "sglang.srt.layers.attention.linear.utils"


class VerifyMethods:
    def eager(self, cache_indices, query_start_loc, *, pool_size=20):
        return select_verify_intermediate_state_indices(
            self.default_indices,
            self.req_pool_indices,
            cache_indices[: query_start_loc.shape[0] - 1] >= 0,
            pool_size,
        )

    def already_lazy(self, cache_indices, query_start_loc, *, pool_size=20):
        return select_verify_intermediate_state_indices(
            self.default_indices,
            self.req_pool_indices,
            lambda: cache_indices[: query_start_loc.shape[0] - 1] >= 0,
            pool_size,
        )


class CacheIndices:
    def __init__(self, values, comparisons):
        self.values = values
        self.comparisons = comparisons

    def __getitem__(self, index):
        return CacheIndices(self.values[index], self.comparisons)

    def __ge__(self, value):
        self.comparisons.append(self.values)
        return [item >= value for item in self.values]


class LazyVerifyStateIndicesTests(unittest.TestCase):
    def setUp(self):
        self.pp_spec = False
        self.comparisons = []
        self.methods = VerifyMethods()
        self.methods.default_indices = [0, 1, 2]
        self.methods.req_pool_indices = [9, 8, 7, 6]
        self.cache_indices = CacheIndices([3, -1, 7, -1], self.comparisons)
        self.query_start_loc = SimpleNamespace(shape=(4,))

        utils = SimpleNamespace(
            pp_spec_stable_rows_enabled=lambda: self.pp_spec,
            select_verify_intermediate_state_indices=self.select,
        )
        patcher = patch.dict(sys.modules, {_UTILS_MODULE: utils})
        patcher.start()
        self.addCleanup(patcher.stop)
        patcher = patch.dict(
            VerifyMethods.eager.__globals__,
            {"select_verify_intermediate_state_indices": self.select},
        )
        patcher.start()
        self.addCleanup(patcher.stop)

    def select(self, default_indices, req_pool_indices, valid, pool_size):
        if not self.pp_spec:
            return default_indices
        return [
            index if is_valid else pool_size
            for index, is_valid in zip(req_pool_indices, valid)
        ]

    def test_pp_spec_off_skips_comparison(self):
        args = self.methods, self.cache_indices, self.query_start_loc
        original = VerifyMethods.eager(*args)
        self.assertIs(original, self.methods.default_indices)
        self.assertEqual(self.comparisons, [[3, -1, 7]])
        self.comparisons.clear()

        fixed = defer_verify_valid_mask(VerifyMethods.eager)
        self.assertIs(fixed(*args), self.methods.default_indices)
        self.assertEqual(self.comparisons, [])
        self.assertIs(defer_verify_valid_mask(VerifyMethods.eager), fixed)

    def test_pp_spec_on_keeps_mask_and_padding(self):
        self.pp_spec = True
        fixed = defer_verify_valid_mask(VerifyMethods.eager)
        args = self.methods, self.cache_indices, self.query_start_loc
        self.assertEqual(fixed(*args, pool_size=100), [9, 100, 7])
        self.assertEqual(self.comparisons, [[3, -1, 7]])
        self.assertEqual(fixed(*args), VerifyMethods.eager(*args))

    def test_already_lazy_method_is_unchanged(self):
        original = VerifyMethods.already_lazy
        self.assertIs(defer_verify_valid_mask(original), original)
