# SPDX-License-Identifier: Apache-2.0
import os
import unittest
from types import SimpleNamespace
from unittest.mock import Mock, patch

import torch

from hcu_sglang.layers.attention import minimax_m31_sparse as sparse


class TestMiniMaxCPPrefillMetadata(unittest.TestCase):
    def test_reuse_is_local_to_batch_and_segment(self):
        meta = SimpleNamespace()
        table = torch.tensor([[7, 8, -1], [10, 11, 12]], dtype=torch.int32)
        idx_q = torch.zeros(3, 4, 128, dtype=torch.bfloat16)
        idx_cache = torch.zeros(3, 1, 128, dtype=idx_q.dtype)
        cache = torch.zeros(3, 4, 128, dtype=idx_q.dtype)
        lengths = torch.tensor([132, 260], dtype=torch.int32)
        q_lengths = torch.tensor([2, 1], dtype=torch.int32)
        cu = torch.tensor([0, 2, 3], dtype=torch.int32)
        original = Mock()

        def prefill(batch_meta, segment, seq_lens=lengths):
            with sparse.cp_prefill_metadata(batch_meta, segment, seq_lens, q_lengths) as prefix:
                p = {
                    "q": idx_q, "k_cache": cache, "v_cache": cache,
                    "idx_q": idx_q, "idx_k_cache": idx_cache,
                    "slot_ids": torch.tensor([1, 2]),
                    "cu_seqlens": cu, "seq_lens": seq_lens, "prefix_lens": prefix,
                    "max_seqlen_k": 260, "seqlens_cpu": [2, 1],
                    "cached_topk_idx": None, "return_topk_idx": False,
                    "init_blocks": 0, "local_blocks": 1, "sm_scale": 0.125,
                    "q_scale": None, "k_scale": None, "v_scale": None,
                    "idx_q_scale": None, "idx_k_scale": None,
                }
                sparse._prefill(original, p)

        with (
            patch.dict(os.environ, {"SGLANG_MINIMAX_CACHE_PREFILL_INDEX_METADATA": "1"}),
            patch.object(sparse, "_args", side_effect=lambda original, args, kwargs: args[0]),
            patch.object(sparse, "_supported", return_value=True),
            patch.object(sparse, "_page_table", side_effect=lambda *args: table.clone()) as build_table,
            patch.object(sparse.native, "flash_mla_sparse_prefill_indexer") as indexer,
            patch.object(sparse.native, "flash_mla_sparse_prefill_main"),
        ):
            prefill(meta, 0)
            prefill(meta, 0)
            prefill(meta, 1, lengths - 5)
            prefill(SimpleNamespace(), 0)
            with patch.dict(os.environ, {"SGLANG_MINIMAX_CACHE_PREFILL_INDEX_METADATA": "0"}):
                prefill(meta, 0)
                prefill(meta, 0)

        calls = indexer.call_args_list
        first, next_layer, other_segment, next_batch, disabled, disabled_again = calls
        self.assertEqual(build_table.call_count, 5)
        torch.testing.assert_close(first.kwargs["k_end"], torch.tensor([131, 132, 260], dtype=torch.int32))
        torch.testing.assert_close(first.kwargs["safe_pages"], table.clamp_min(0).flatten())
        self.assertIs(first.args[2], next_layer.args[2])
        self.assertIs(first.kwargs["safe_pages"], next_layer.kwargs["safe_pages"])
        self.assertIs(first.kwargs["k_end"], next_layer.kwargs["k_end"])
        torch.testing.assert_close(other_segment.kwargs["k_end"], first.kwargs["k_end"] - 5)
        self.assertIsNot(first.args[2], next_batch.args[2])
        self.assertIsNot(disabled.args[2], disabled_again.args[2])
        self.assertIsNone(sparse._cp_prefill_metadata.get())
        original.assert_not_called()
