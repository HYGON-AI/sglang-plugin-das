# SPDX-License-Identifier: Apache-2.0
import os
import unittest
from unittest.mock import Mock, patch

import torch


class TestMiniMaxFlashMLAVerifySplits(unittest.TestCase):
    def setUp(self):
        from hcu_sglang.layers.attention import minimax_m31_flashmla as flashmla

        self.flashmla = flashmla
        self.addCleanup(flashmla._fp8_verify_split_override.cache_clear)

    def run_decode(self, batch, width, dtype):
        rows = batch * width
        q = torch.empty((rows, 64, 128), dtype=dtype)
        cache = torch.empty((128, 4, 128), dtype=dtype)
        topk = torch.zeros((4, rows, 16), dtype=torch.int32)
        page_table = torch.zeros((batch, 1), dtype=torch.int32)
        seq_lens = torch.ones(rows, dtype=torch.int32)
        decode = Mock(return_value=(torch.empty(q.shape, dtype=torch.bfloat16), None))
        expand = Mock(return_value=torch.zeros((batch, width, 4, 2048), dtype=torch.int32))
        wheel = (None, decode, None, expand, None, None)
        with (
            patch.object(self.flashmla, "flash_mla_gfx938_available", return_value=True),
            patch.object(self.flashmla, "_load_flash_mla", return_value=wheel),
        ):
            output = self.flashmla.flash_mla_sparse_decode_main(
                q, cache, cache, topk, page_table, seq_lens, 128, object(),
                verify_group_size=width,
            )
        self.assertEqual(output.shape, q.shape)
        return decode.call_args.kwargs

    def test_fp8_verify_default_is_fixed_across_batch_boundary(self):
        with patch.dict(os.environ):
            os.environ.pop("SGLANG_MINIMAX_FLASH_MLA_FP8_VERIFY_SPLITS", None)
            for batch in (8, 9, 10):
                with self.subTest(batch=batch):
                    kwargs = self.run_decode(batch, 8, torch.float8_e4m3fn)
                    splits = kwargs["num_splits"]
                    self.assertEqual(splits.device.type, "cpu")
                    self.assertEqual(splits.dtype, torch.int32)
                    self.assertEqual(splits.tolist(), [8])

    def test_baseline_dispatches_without_explicit_splits(self):
        with patch.dict(os.environ, {"SGLANG_MINIMAX_FLASH_MLA_FP8_VERIFY_SPLITS": "0"}):
            self.assertNotIn("num_splits", self.run_decode(9, 8, torch.float8_e4m3fn))
        self.assertNotIn("num_splits", self.run_decode(9, 1, torch.float8_e4m3fn))
        self.assertNotIn("num_splits", self.run_decode(9, 8, torch.bfloat16))

    def test_invalid_override_is_rejected(self):
        with patch.dict(os.environ, {"SGLANG_MINIMAX_FLASH_MLA_FP8_VERIFY_SPLITS": "3"}):
            with self.assertRaisesRegex(ValueError, "must be one of"):
                self.run_decode(9, 8, torch.float8_e4m3fn)


if __name__ == "__main__":
    unittest.main()
