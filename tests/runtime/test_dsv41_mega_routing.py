from types import SimpleNamespace
from unittest.mock import Mock, patch

import pytest
import torch

from hcu_sglang.layers.moe import mega_moe
from hcu_sglang.environ import register_envs


def moe_and_batch():
    topk = Mock()
    topk.topk_config.apply_routed_scaling_factor_on_output = True
    gate = Mock(e_score_correction_bias_vl=torch.ones(1))
    experts = SimpleNamespace(moe_runner_config=SimpleNamespace(swiglu_limit=None))
    config = SimpleNamespace(hidden_size=4, moe_intermediate_size=8, num_experts_per_tok=1)
    moe = SimpleNamespace(gate=gate, experts=experts, config=config, topk=topk,
                          layer_id=7, num_fused_shared_experts=0, is_hash=False)
    batch = Mock(input_embeds=None, mm_input_embeds=None)
    batch.forward_mode.is_extend_without_speculative.return_value = True
    batch.contains_mm_inputs.return_value = False
    return moe, batch


@pytest.mark.parametrize("capture,dp,vision,use_vl", [
    (False, 1, False, False), (True, 1, False, True),
    (False, 2, False, True), (False, 1, True, True),
])
def test_capture_and_modality_routing(capture, dp, vision, use_vl):
    register_envs()
    from hcu_sglang.multimodal.dsv41 import vl_routing

    moe, batch = moe_and_batch()
    batch.contains_mm_inputs.return_value = vision
    x = torch.empty(1, 4)
    ids = torch.zeros(1, dtype=torch.long)
    with patch.object(mega_moe, "_is_hcu", True), \
         patch.object(mega_moe, "get_is_capture_mode", return_value=capture), \
         patch.object(mega_moe, "get_parallel", return_value=SimpleNamespace(attn_dp_size=dp)), \
         patch.object(mega_moe.ExpertLocationDispatchInfo, "init_new"), \
         patch.object(vl_routing, "vision_topk") as vl, \
         patch.object(mega_moe, "run_mega_routed_experts") as run:
        mega_moe._run_mega_routed(moe, x, batch, ids, 1)
    selected = vl if use_vl else moe.topk
    selected.assert_called_once()
    (moe.topk if use_vl else vl).assert_not_called()
    assert run.call_args.args[2] is selected.return_value.topk_ids


@pytest.mark.parametrize("algorithm,skip,expected", [
    ("lp", False, 1), ("lp", True, 0), ("static", False, 0),
])
def test_empty_rank_lp_participation(algorithm, skip, expected):
    moe, batch = moe_and_batch()
    moe.experts._skip_expert_location = skip
    ctx = SimpleNamespace(moe=SimpleNamespace(ep_dispatch_algorithm=algorithm))
    x = torch.empty(0, 4)
    with patch.object(mega_moe, "_is_hcu", True), \
         patch.object(mega_moe, "get_exec", return_value=ctx), \
         patch.object(mega_moe, "run_mega_routed_experts") as run:
        mega_moe._run_mega_routed(moe, x, batch, None, 0)
    assert moe.topk.empty_topk_output.call_count == expected
    if expected:
        moe.topk.empty_topk_output.assert_called_once_with(x.device, layer_id=7)
    assert run.call_args.args[2:4] == (None, None)
