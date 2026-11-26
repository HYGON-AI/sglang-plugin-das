import unittest
from types import SimpleNamespace
from unittest.mock import Mock, patch

import torch

from hcu_sglang.layers.quantization import channelwise_fp8, channelwise_int8
from hcu_sglang.layers.quantization.channelwise_fp8_heads import compute_lm_head, init_lora_head


class TestChannelwiseHeads(unittest.TestCase):
    def test_fp32_logits_request_preserves_activation_dtype(self):
        x = torch.randn(2, 4, dtype=torch.bfloat16)
        head = SimpleNamespace(quant_method=channelwise_fp8.ChannelwiseFp8Method())
        owner = SimpleNamespace(use_fp32_lm_head=True)
        base = Mock()
        bias = torch.ones(3)
        with patch.object(head.quant_method, "apply_with_output_dtype") as apply:
            self.assertIs(compute_lm_head(base, owner, x, head, bias), apply.return_value)
            apply.assert_called_once_with(head, x, bias, output_dtype=torch.float32)
        base.assert_not_called()
        owner.use_fp32_lm_head = False
        self.assertIs(compute_lm_head(base, owner, x, head, bias), base.return_value)
        base.assert_called_once_with(owner, x, head, bias)

    def test_lora_rejects_channelwise_fp8_and_delegates_other_methods(self):
        base = Mock()
        owner, backend = object(), object()
        head = SimpleNamespace(quant_method=channelwise_fp8.ChannelwiseFp8Method())
        with self.assertRaisesRegex(ValueError, "lm_head LoRA"):
            init_lora_head(base, owner, head, backend)
        base.assert_not_called()
        head.quant_method = None
        self.assertIs(init_lora_head(base, owner, head, backend), base.return_value)
        base.assert_called_once_with(owner, head, backend)

    def test_normalized_float_config_has_no_int8_lm_head(self):
        config = Mock()
        config.get_name.return_value = "compressed-tensors"
        config.quant_format = "float-quantized"
        config.target_scheme_map = {"Linear": {"weights": SimpleNamespace(type="float")}}
        self.assertFalse(channelwise_int8.enabled(object(), config, "lm_head"))
        config.get_scheme_dict.assert_not_called()


if __name__ == "__main__":
    unittest.main()
