import unittest
from types import SimpleNamespace
from unittest.mock import Mock, patch

import torch

from hcu_sglang.layers.quantization import channelwise_fp8 as fp8
from hcu_sglang.layers.quantization.channelwise_int8 import (
    around_draft_lm_logits,
    around_project_bias,
    around_step_bias,
)


class TestDSparkFP8Heads(unittest.TestCase):
    def setUp(self):
        self.layer = torch.nn.Linear(16, 12, bias=False)
        fp8.install(self.layer)
        self.layer.weight.data.copy_(torch.ones(12, 16))
        self.layer.weight_scale_inv.data.copy_(torch.arange(1, 13).view(-1, 1))
        self.head = SimpleNamespace(
            markov_w2=self.layer, _tp_shard=SimpleNamespace(org_vocab_start=4, org_vocab_end=8),
            _opt_markov_w2_bf16=True, get_prev_embeddings=lambda ids: torch.ones(len(ids), 16, dtype=torch.bfloat16),
        )

    @staticmethod
    def projection(x, weight, scale, *, output_dtype):
        return (x.float() @ (weight.float() * scale).T).to(output_dtype)

    def test_tp_bias_matches_full_and_preserves_state(self):
        latent = self.head.get_prev_embeddings([0])
        with patch.object(fp8, 'linear', side_effect=self.projection):
            full = around_project_bias(None, self.head, latent)
            self.assertEqual(full.dtype, torch.float32)
            state = object()
            bias, returned = around_step_bias(None, self.head, [0], hidden_states=None, state=state)
            self.assertIs(state, returned)
            self.assertEqual(bias.dtype, torch.bfloat16)
            torch.testing.assert_close(bias.float(), full[:, 4:8])
            self.assertEqual(bias.argmax(dim=1).item(), 3)
            with self.assertRaises(ValueError):
                around_project_bias(None, self.head, latent, weight=self.layer.weight[4:8])

    def test_draft_logits_explicit_precision_and_fallback(self):
        model = SimpleNamespace(lm_head=self.layer, stages=[SimpleNamespace(norm=lambda x: x)],
                                _use_fp32_lm_head=True, _opt_markov_w2_tp_shard=True)
        x = torch.ones(2, 16, dtype=torch.bfloat16)
        with patch.object(fp8, 'linear', side_effect=self.projection):
            logits = around_draft_lm_logits(None, model, x)
            self.assertEqual(logits.dtype, torch.float32)
            torch.testing.assert_close(logits[:, 0], torch.tensor([16., 16.]))
        model._use_fp32_lm_head = False
        previous = Mock(return_value=x)
        self.assertIs(around_draft_lm_logits(previous, model, x), x)
        previous.assert_called_once_with(model, x)


if __name__ == '__main__':
    unittest.main()
