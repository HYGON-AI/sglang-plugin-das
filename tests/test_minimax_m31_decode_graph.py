# SPDX-License-Identifier: Apache-2.0
import os
import sys
import unittest
from types import SimpleNamespace
from unittest.mock import Mock, patch

import torch

from hcu_sglang.layers.attention import minimax_m31_decode_graph as stage1


class TestMiniMaxStage1GraphVariants(unittest.TestCase):
    def test_reserve_verify_width_and_keep_full_context_fallback(self):
        variants = stage1._Stage1GraphVariants(cap=148608, verify_width=8)
        compact, full = variants.capture_labels
        self.assertEqual(compact, "msa_stage1_cap_148608")
        for values, expected in (
            ([147456, 148600], compact),
            ([148601, 147456], full),
            ([150000, 150000], full),
            ([148480, 0], compact),
        ):
            batch = SimpleNamespace(batch_size=2, seq_lens_cpu=torch.tensor(values))
            self.assertEqual(variants.select(batch), expected)
        for unknown in (None, torch.tensor([148480]), Mock(device=SimpleNamespace(type="cuda"))):
            batch = SimpleNamespace(batch_size=2, seq_lens_cpu=unknown)
            self.assertEqual(variants.select(batch), full)
        batch = SimpleNamespace(batch_size=0, seq_lens_cpu=torch.tensor([1, 1]))
        self.assertEqual(variants.select(batch), full)

    def test_factory_respects_backend_and_service_scope(self):
        original = Mock()
        modes = SimpleNamespace(DECODE=1, TARGET_VERIFY=2)
        disagg = SimpleNamespace(disaggregation_mode="null")
        memory = SimpleNamespace(enable_hisparse=False)
        sparse = SimpleNamespace(block_size_k=128, page_size=128, max_context_len=1048576)
        class FakeHybrid:
            decode_backend = SimpleNamespace(sparse=sparse)

            def _select_backend(self, mode):
                if mode == modes.DECODE:
                    return self.decode_backend
                return SimpleNamespace(sparse=sparse)

        runner = SimpleNamespace(
            is_draft_worker=False,
            model_config=SimpleNamespace(quantization="slimquant_w4a8_marlin", hf_config=object()),
            attn_backend=FakeHybrid(),
        )
        modules = {
            "sglang.srt.layers.attention.hybrid_attn_backend": SimpleNamespace(HybridAttnBackend=FakeHybrid),
            "sglang.srt.layers.attention.minimax_sparse_backend": SimpleNamespace(MiniMaxSparseAttnBackend=SimpleNamespace),
            "sglang.srt.configs.model_config": SimpleNamespace(is_minimax_sparse=lambda _: True),
            "sglang.srt.model_executor.forward_batch_info": SimpleNamespace(ForwardMode=modes),
            "sglang.srt.runtime_context": SimpleNamespace(get_memory=lambda: memory, get_disagg=lambda: disagg),
        }
        create = stage1._stage1_graph_variants
        with (
            patch.dict(sys.modules, modules),
            patch.dict(os.environ, {"M31_STAGE1_GRAPH_MAX_KV_LEN": "148608"}),
            patch.object(torch.version, "hip", "test"),
            patch.object(stage1.native, "flash_mla_gfx938_available", return_value=True),
            patch.object(stage1.native, "flash_mla_stage1_graph_available", return_value=True),
        ):
            self.assertEqual(create(original, runner, modes.TARGET_VERIFY, 8).verify_width, 8)
            self.assertEqual(create(original, runner, modes.DECODE, 1).verify_width, 0)
            original.assert_not_called()
            with patch.dict(os.environ, {"M31_STAGE1_GRAPH_MAX_KV_LEN": ""}):
                self.assertIs(create(original, runner, modes.DECODE), original.return_value)
            memory.enable_hisparse = True
            self.assertIs(create(original, runner, modes.TARGET_VERIFY, 8), original.return_value)
            memory.enable_hisparse = False
            disagg.disaggregation_mode = "decode"
            runner.attn_backend.decode_backend = object()
            self.assertIs(create(original, runner, modes.DECODE), original.return_value)
            self.assertEqual(create(original, runner, modes.TARGET_VERIFY, 8).verify_width, 8)
            disagg.disaggregation_mode = "null"
            with patch.dict(os.environ, {"M31_STAGE1_GRAPH_MAX_KV_LEN": "148600"}):
                with self.assertRaisesRegex(ValueError, "positive multiple of 128"):
                    create(original, runner, modes.TARGET_VERIFY, 8)
