# SPDX-License-Identifier: Apache-2.0
import unittest
from types import SimpleNamespace
from unittest.mock import Mock, patch

import torch

from hcu_sglang.layers.attention import minimax_m31_fp8_prepare as prepare


class TestMiniMaxFP8Prepare(unittest.TestCase):
    def test_packed_projection_keeps_original_v_and_output_order(self):
        attn = SimpleNamespace(disable_index_value=True)
        packed = torch.arange(9856).view(1, -1)
        original = Mock()
        with patch.object(prepare, "_prepare", return_value=("q8", "k", "iq8", "ik")) as fuse:
            q, k, v, iq, ik, iv = prepare._prepare_packed(original, attn, packed, "pos", "batch")
        self.assertEqual((q, k, iq, ik, iv), ("q8", "k", "iq8", "ik", None))
        inputs = fuse.call_args.args[1]
        self.assertEqual([x.shape[1] for x in inputs], [8192, 512, 512, 512, 128])
        self.assertIs(v, inputs[2])
        torch.testing.assert_close(torch.cat(inputs, dim=1), packed)
        original.assert_not_called()

        with patch.object(prepare, "_prepare", return_value=None):
            self.assertIs(
                prepare._prepare_packed(original, attn, packed, "pos", "batch"),
                original.return_value,
            )
        original.assert_called_once_with(attn, packed, "pos", "batch")
