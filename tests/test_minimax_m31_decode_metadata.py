# SPDX-License-Identifier: Apache-2.0
import unittest
from types import SimpleNamespace
from unittest.mock import Mock, patch

import torch

from hcu_sglang.layers.attention import minimax_m31_sparse as sparse


class TestMiniMaxDecodeMetadata(unittest.TestCase):
    def test_rebuild_after_each_forward_and_when_input_identity_changes(self):
        batch = SimpleNamespace(
            forward_mode=SimpleNamespace(is_decode=lambda: False, is_target_verify=lambda: True),
            can_run_tbo=False,
        )
        seq_lens = torch.arange(129, 145)
        slots = torch.tensor([1] * 8 + [3] * 8)
        params = {
            "req_to_token": torch.zeros(4, 256, dtype=torch.int32),
            "max_seqlen": 256,
            "packed_queries": 8,
            "seq_lens": seq_lens,
            "slot_ids": slots,
        }

        def forward(_model, _ids, _pos, _batch):
            a = sparse._decode_plan(params, reuse=True)
            self.assertIs(sparse._decode_plan(params, reuse=True), a)
            self.assertIsNot(sparse._decode_plan(params, reuse=False), a)
            self.assertIsNot(sparse._decode_plan({**params, "slot_ids": slots.clone()}, reuse=True), a)
            self.assertEqual(a.slots.tolist(), [1, 3])
            self.assertEqual(a.k_end.dtype, torch.int32)
            torch.testing.assert_close(a.k_end, seq_lens.int())
            return a

        with patch.object(sparse, "_page_table", side_effect=lambda *args: Mock()) as build:
            a = sparse._model_forward(forward, None, None, None, batch)
            b = sparse._model_forward(forward, None, None, None, batch)
        self.assertEqual(build.call_count, 6)
        self.assertIsNot(a.table, b.table)
        self.assertIsNone(sparse._decode_metadata.get())
