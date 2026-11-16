from types import SimpleNamespace
import unittest
from unittest.mock import patch

import torch
from aiter.moe import MoeQuantType, MoeSolutionType
from hcu_sglang.layers.quantization import unquantized_aiter as route


class AiterW16A16SharedGateTest(unittest.TestCase):
    def test_respects_shared_expert_input_ownership(self):
        def vendor(**kwargs):
            x = kwargs["hidden_states"]
            return (x if kwargs["inplace"] else x.clone()).add_(10)

        config = SimpleNamespace(
            apply_router_weight_on_input=False, num_experts=2,
            activation="silu", routed_scaling_factor=None,
        )
        method = SimpleNamespace(moe_runner_config=config)
        layer = SimpleNamespace(
            w13_weight=torch.empty(2, 8, 4), w2_weight=torch.empty(2, 4, 4)
        )
        selected = SimpleNamespace(
            solution_type=MoeSolutionType.ASM, quant_type=MoeQuantType.W16A16,
        )
        with patch.object(route, "_enabled", return_value=True), patch(
            "hcu_sglang.compat.aiter.install_aiter_compat"
        ), patch("aiter.moe.get_aiter_moe_config", return_value=(True, selected)), patch(
            "aiter.moe.aiter_moe", side_effect=vendor
        ):
            for config.inplace in (False, True):
                hidden = torch.ones(3, 4)
                dispatch = SimpleNamespace(
                    hidden_states=hidden,
                    topk_output=(torch.ones(3, 1), torch.zeros(3, 1), None),
                )
                result = route.around_forward_cuda(None, method, layer, dispatch)
                torch.testing.assert_close(result.hidden_states, torch.full_like(hidden, 11))
                torch.testing.assert_close(hidden, torch.full_like(hidden, 11 if config.inplace else 1))


if __name__ == "__main__":
    unittest.main()
