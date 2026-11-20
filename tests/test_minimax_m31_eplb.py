import sys
import unittest
from enum import Enum
from types import SimpleNamespace
from unittest.mock import patch

import torch

from hcu_sglang.eplb import minimax_m31


class TestMiniMaxEplb(unittest.TestCase):
    def test_routed_views_resolve_current_weights_and_skip_shared_aliases(self):
        experts = torch.nn.Module()
        experts.num_local_experts = 3
        experts.w13_weight = torch.nn.Parameter(torch.zeros(3, 4))
        experts.w13_weight_scale = torch.nn.Parameter(torch.ones(3, 1))
        experts.correction_bias = torch.nn.Parameter(torch.zeros(3))
        moe = SimpleNamespace(experts=experts, num_fused_shared_experts=1)
        routed = minimax_m31._RoutedWeights({4: moe})

        experts.w13_weight = torch.nn.Parameter(torch.full((3, 4), 2.0))
        experts.register_buffer("w13_weight_deepgemm", experts.w13_weight.detach())
        experts.register_buffer("w13_weight_deepgemm_masked", torch.full((3, 4), 3.0))
        experts.register_buffer("unrelated", torch.zeros(3, 4))
        utils = SimpleNamespace(
            filter_moe_weight_param_global_expert=lambda _, t, n: t.ndim > 0 and t.shape[0] == n,
        )
        with patch.dict(sys.modules, {"sglang.srt.layers.moe.utils": utils}):
            weights = routed[4]
        self.assertEqual(list(routed), [4])
        self.assertEqual([w.shape[0] for w in weights], [2, 2, 2])
        self.assertEqual([w[0, 0].item() for w in weights], [2.0, 1.0, 3.0])
        weights[0].fill_(7)
        self.assertEqual(experts.w13_weight[:, 0].tolist(), [7.0, 7.0, 2.0])

    def test_prefill_recording_excludes_verify_and_uses_global_counter(self):
        class Mode(Enum):
            EXTEND = 1
            SPLIT_PREFILL = 2
            TARGET_VERIFY = 3

        modes = SimpleNamespace(ForwardMode=Mode)
        capture = False
        recorder = SimpleNamespace(_is_model_capture_mode=lambda: capture)
        batch = SimpleNamespace(
            forward_mode=Mode.EXTEND, extend_num_tokens=8192, is_extend_in_batch=True,
        )
        with patch.dict(sys.modules, {
            "sglang.srt.eplb.expert_distribution": recorder,
            "sglang.srt.model_executor.forward_batch_info": modes,
        }):
            self.assertTrue(minimax_m31._record_prefill(None, batch))
            batch.forward_mode = Mode.TARGET_VERIFY
            self.assertFalse(minimax_m31._record_prefill(None, batch))
            self.assertTrue(minimax_m31._advance_prefill_counter(None, batch))
            capture = True
            self.assertFalse(minimax_m31._advance_prefill_counter(None, batch))

    def test_forward_metadata_preserves_original_rank_token_counts(self):
        gatherer = SimpleNamespace()
        batch = SimpleNamespace(
            original_global_num_tokens_cpu=[24, 32],
            global_num_tokens_cpu=[32, 32],
            forward_mode=SimpleNamespace(name="TARGET_VERIFY"),
            batch_size=3,
            input_ids=torch.empty(24),
            is_extend_in_batch=False,
        )
        minimax_m31._capture_forward_metadata(None, gatherer, batch)
        counts = torch.empty(4)
        result = minimax_m31._collect_forward_metadata({"global_physical_count": counts}, gatherer)
        self.assertIs(result["global_physical_count"], counts)
        self.assertEqual(result["forward_metadata"]["num_tokens"], 24)
        self.assertEqual(result["forward_metadata"]["global_num_tokens_cpu"], [24, 32])
