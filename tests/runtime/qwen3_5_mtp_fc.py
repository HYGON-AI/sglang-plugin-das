import unittest
from unittest.mock import patch

import torch

from hcu_sglang.models.qwen import qwen3_5_mtp


def quant_config(ignore=(), targets=("Linear",)):
    return qwen3_5_mtp.CompressedTensorsConfig.from_config({
        "format": "int-quantized",
        "ignore": list(ignore),
        "config_groups": {
            "group": {
                "targets": list(targets),
                "weights": {
                    "num_bits": 8, "type": "int", "symmetric": True,
                    "strategy": "channel", "dynamic": False,
                },
                "input_activations": {
                    "num_bits": 8, "type": "int", "symmetric": True,
                    "strategy": "token", "dynamic": True,
                },
            },
        },
    })


def init_draft(draft, config, quant_config=None, prefix=""):
    draft.quant_config = quant_config
    draft.fc = torch.nn.Linear(16, 8, bias=False, dtype=torch.bfloat16)


class TestMTPFC(unittest.TestCase):
    def build_fc(self, config, prefix="", is_hcu=True):
        draft = torch.nn.Module()
        with (
            patch.object(qwen3_5_mtp, "is_hcu", return_value=is_hcu),
            patch.object(
                qwen3_5_mtp.CompressedTensorsConfig,
                "_check_scheme_supported",
                return_value=True,
            ),
        ):
            qwen3_5_mtp.around_init(init_draft, draft, None, config, prefix)
        return draft.fc

    def test_explicit_scale_load_and_forward_contract(self):
        layer = self.build_fc(quant_config(targets=("draft.mtp.fc",)), prefix="draft")
        self.assertIsInstance(layer, qwen3_5_mtp.MTPReplicatedLinear)
        self.assertEqual(layer.weight.dtype, torch.int8)
        weight = torch.full((8, 16), 3, dtype=torch.int8)
        scale = torch.arange(1, 9, dtype=torch.bfloat16).reshape(8, 1) / 128
        layer.weight.weight_loader(layer.weight, weight)
        layer.weight_scale.weight_loader(layer.weight_scale, scale)
        torch.testing.assert_close(layer.weight, weight)
        torch.testing.assert_close(layer.weight_scale, scale.to(layer.weight_scale.dtype))
        expected = torch.ones(2, 8, dtype=torch.bfloat16)
        with patch.object(layer.quant_method, "apply", return_value=expected) as apply:
            self.assertIs(layer(torch.ones(2, 16, dtype=torch.bfloat16)), expected)
            self.assertEqual(layer(torch.empty(0, 16, dtype=torch.bfloat16)).shape, (0, 8))
            apply.assert_called_once()

    def test_keeps_bf16_for_excluded_and_unmatched_configs(self):
        for config in (
            None,
            quant_config(ignore=("mtp.fc",)),
            quant_config(ignore=("re:.*mtp.fc.*",)),
            quant_config(targets=("mtp.layers.0.mlp.gate_proj",)),
        ):
            with self.subTest(config=config):
                layer = self.build_fc(config)
                self.assertIs(type(layer), torch.nn.Linear)
                self.assertEqual(layer.weight.dtype, torch.bfloat16)

    def test_keeps_original_layer_off_hcu(self):
        layer = self.build_fc(quant_config(), is_hcu=False)
        self.assertIs(type(layer), torch.nn.Linear)


if __name__ == "__main__":
    unittest.main()
