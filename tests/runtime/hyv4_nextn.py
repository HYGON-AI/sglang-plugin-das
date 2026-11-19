import copy
import unittest
from types import SimpleNamespace
from unittest.mock import Mock, patch

import torch

from hcu_sglang.models import hyv4_nextn as n
from hcu_sglang.models.hyv4_mtp_metadata import normalize_hyv4_mtp_weight_name, validate_hyv4_mtp_metadata


class NextNContract(unittest.TestCase):
    def test_mtp_profile_rejects_unsupported_tree_and_incomplete_chain(self):
        cfg=SimpleNamespace(speculative_algorithm="EAGLE",
            speculative_num_steps=3,speculative_eagle_topk=1,
            speculative_num_draft_tokens=4)
        self.assertTrue(n.supports_mtp_profile(cfg))
        for field,value in (("speculative_algorithm","EAGLE3"),
                            ("speculative_eagle_topk",2),
                            ("speculative_num_draft_tokens",3),
                            ("speculative_num_steps",0)):
            changed=copy.copy(cfg)
            setattr(changed,field,value)
            self.assertFalse(n.supports_mtp_profile(changed))

    def test_quantization_remap_keeps_input_and_special_layers(self):
        q = SimpleNamespace(
            ignore=["model.mtp.layers.0.enorm.weight",
                    "model.mtp_layers.0.self_attn.g_proj.weight"],
            ignored_layers=["mtp_layers.0.hnorm.weight"],
            linear_targets=["model.mtp_layers.0.self_attn.q_b_proj"],
            linear_fp8_config=object(),
        )
        old = copy.deepcopy(q)
        result = n.normalize_mtp_quant_config(q)
        self.assertEqual(result.ignore, ["model.enorm.weight", "model.decoder.self_attn.linear_gate.weight"])
        self.assertEqual(result.ignored_layers, ["model.hnorm.weight"])
        self.assertEqual(result.linear_targets, ["model.decoder.self_attn.q_b_proj"])
        self.assertIsNone(result.linear_fp8_config)
        self.assertEqual(q.ignore, old.ignore)
        self.assertIsNotNone(q.linear_fp8_config)
        self.assertEqual(n.normalize_mtp_quant_config(result).ignore, result.ignore)

    def test_non_hcu_retains_original(self):
        original = Mock(return_value="original")
        q = object()
        with patch.object(n, "is_hcu", return_value=False):
            self.assertEqual(n.around_mtp_quant_config(original, q), "original")
        original.assert_called_once_with(q)

    def test_registered_public_quantization_consumer_uses_mapping(self):
        from sglang.srt.plugins.hook_registry import HookRegistry
        import sglang.srt.models.hunyuan_v4_nextn as native
        n.register_hooks()
        HookRegistry.apply_hooks()
        targets={
            "sglang.srt.models.hunyuan_v4_nextn."+name for name in (
                "_mtp_quant_config", "HYV4ForCausalLMNextN.__init__",
                "HYV4ForCausalLMNextN.load_weights", "HYV4ModelNextN.__init__",
                "HYV4ModelNextN.forward")}
        self.assertTrue(targets.issubset(HookRegistry._patched))
        with patch.object(n,"is_hcu",return_value=True):
            q=native._mtp_quant_config(SimpleNamespace(
                ignore=["model.mtp.layers.0.self_attn.g_proj.weight"]))
        self.assertEqual(q.ignore,["model.decoder.self_attn.linear_gate.weight"])

    def test_checkpoint_aliases(self):
        self.assertEqual(normalize_hyv4_mtp_weight_name(
            "model.mtp.layers.0.self_attn.g_proj.weight"), "self_attn.linear_gate.weight")
        self.assertEqual(normalize_hyv4_mtp_weight_name(
            "model.mtp_layers.0.final_layernorm.weight"), "shared_head.norm.weight")
        self.assertIsNone(normalize_hyv4_mtp_weight_name("model.layers.0.weight"))

    def test_missing_checkpoint_weight_rejected(self):
        config = dict(hidden_size=4, num_attention_heads=2, v_head_dim=2,
                      q_lora_rank=2, kv_lora_rank=2, qk_nope_head_dim=2,
                      qk_rope_head_dim=2, index_n_heads=1, index_head_dim=2,
                      n_routed_experts=1, moe_intermediate_size=4)
        with self.assertRaisesRegex(ValueError, "Missing HYV4 MTP weight"):
            validate_hyv4_mtp_metadata(config, {}, packed_experts=True, mixed_int8=True)

    def test_load_converts_packed_and_preserves_channel_scale(self):
        quant = SimpleNamespace(checkpoint_format="hy4_compressed_tensors_w4a8")
        model = SimpleNamespace(
            quant_config=quant, config=SimpleNamespace(),
            named_parameters=lambda: iter([("model.decoder.weight_scale", torch.ones(1))]),
            _initialize_nextn_conf=lambda _: SimpleNamespace(nextn_layer_prefix="model.layers.78"),
        )
        model.do_load_weights = lambda stream, **kw: list(stream)
        weights = [
            ("model.layers.0.weight", torch.ones(1)),
            ("model.mtp_layers.0.mlp.experts.0.gate_proj.weight_packed", torch.zeros(2, 1, dtype=torch.int32)),
            ("model.mtp_layers.0.mlp.experts.0.gate_proj.weight_shape", torch.tensor([2, 8])),
            ("model.mtp.layers.0.self_attn.g_proj.weight", torch.ones(2, 4)),
            ("model.mtp_layers.0.self_attn.q_b_proj.weight_scale", torch.ones(2, 1)),
        ]
        with patch.object(n, "is_hcu", return_value=True), patch.object(
            n, "validate_hyv4_mtp_metadata"
        ) as validate, patch(
            "sglang.srt.models.hunyuan_v4.permute_hyv4_indexer_weight",
            side_effect=lambda name, weight, config: weight
        ):
            result = n.around_nextn_load_weights(Mock(), model, weights)
        self.assertEqual(len(result), 3)
        self.assertEqual(result[0][0], "model.layers.78.mlp.experts.0.gate_proj.weight")
        self.assertEqual(result[0][1].dtype, torch.int8)
        self.assertEqual(tuple(result[0][1].shape), (2, 4))
        self.assertEqual(result[1][0], "model.layers.78.self_attn.linear_gate.weight")
        self.assertTrue(result[2][0].endswith(".weight_scale"))
        self.assertFalse(result[2][0].endswith("_inv"))
        self.assertTrue(validate.call_args.kwargs["mixed_int8"])

    def test_deepep_ct_loader_keeps_real_packed_parameter_contract(self):
        from hcu_sglang.layers.quantization.compressed_w4a8 import HCUW4A8Int8DeepEPMoEMethod
        quant = SimpleNamespace(checkpoint_format="hy4_compressed_tensors_w4a8",
                                uses_hyv4_deepep=lambda: True)
        layer = torch.nn.Module()
        HCUW4A8Int8DeepEPMoEMethod(quant).create_weights(
            layer, 1, 8, 4, torch.bfloat16)
        self.assertIn("w13_weight_packed", dict(layer.named_parameters()))
        self.assertNotIn("w13_weight", dict(layer.named_parameters()))
        model = SimpleNamespace(
            quant_config=quant, config=SimpleNamespace(),
            named_parameters=lambda: iter([]),
            _initialize_nextn_conf=lambda _: SimpleNamespace(nextn_layer_prefix="model.layers.78"),
            do_load_weights=lambda stream, **kw: list(stream),
        )
        packed=torch.zeros(4,1,dtype=torch.int32)
        shape=torch.tensor([4,8],dtype=torch.int32)
        weights=[
            ("model.mtp_layers.0.mlp.experts.0.gate_proj.weight_packed", packed),
            ("model.mtp_layers.0.mlp.experts.0.gate_proj.weight_shape", shape),
        ]
        with patch.object(n,"is_hcu",return_value=True), patch.object(
            n,"validate_hyv4_mtp_metadata"
        ) as validate, patch(
            "sglang.srt.models.hunyuan_v4.permute_hyv4_indexer_weight",
            side_effect=lambda name, weight, config: weight
        ):
            result=n.around_nextn_load_weights(Mock(),model,weights)
        self.assertEqual(len(result),2)
        self.assertTrue(result[0][0].endswith(".weight_packed"))
        self.assertIs(result[0][1],packed)
        self.assertIs(result[1][1],shape)
        metadata=validate.call_args.args[1]
        self.assertEqual(metadata["mlp.experts.0.gate_proj.weight"],
                         {"shape":[4,4],"dtype":"torch.int8"})

    def test_duplicate_weight_rejected(self):
        quant = SimpleNamespace(checkpoint_format="hy4_compressed_tensors_w4a8")
        model = SimpleNamespace(
            quant_config=quant, config=SimpleNamespace(),
            named_parameters=lambda: iter([]),
            _initialize_nextn_conf=lambda _: SimpleNamespace(nextn_layer_prefix="model.layers.78"),
            do_load_weights=lambda stream, **kw: list(stream),
        )
        weights=[("model.mtp_layers.0.enorm.weight", torch.ones(4))]*2
        with patch.object(n, "is_hcu", return_value=True), patch(
            "sglang.srt.models.hunyuan_v4.permute_hyv4_indexer_weight",
            side_effect=lambda name, weight, config: weight
        ), self.assertRaisesRegex(ValueError, "Duplicate HYV4 MTP weight"):
            n.around_nextn_load_weights(Mock(), model, weights)

    def test_draft_model_waits_for_alt_stream(self):
        model=SimpleNamespace(alt_stream=object())
        original=Mock(return_value="output")
        current=Mock()
        with patch.object(n,"is_hcu",return_value=True), patch(
            "torch.cuda.current_stream",return_value=current
        ):
            result=n.around_nextn_model_forward(original,model,[],[],SimpleNamespace())
        self.assertEqual(result,"output")
        current.wait_stream.assert_called_once_with(model.alt_stream)

    def test_generic_cp_shards_both_mtp_streams_and_restores(self):
        from sglang.srt.layers.cp import utils
        from sglang.srt.layers.cp.interleave import InterleaveCPStrategy
        parallel=SimpleNamespace(attn_cp_size=8,attn_cp_rank=2)
        with patch("sglang.srt.layers.cp.base.get_parallel",return_value=parallel):
            strategy=InterleaveCPStrategy(8)
            metadata=strategy.build_metadata(17,[17],[17])
            batch=SimpleNamespace(
                input_ids=torch.arange(17),
                forward_mode=SimpleNamespace(is_context_parallel_extend=lambda: True),
                extend_seq_lens_cpu=[17],attn_cp_metadata=metadata,
                spec_info=SimpleNamespace(hidden_states=torch.arange(34).reshape(17,2)),
            )
            previous=batch.spec_info.hidden_states
            embeds=torch.arange(34).reshape(17,2)+100
            with patch.object(utils,"get_cp_strategy",return_value=strategy), patch.object(
                utils,"get_parallel",return_value=parallel
            ), patch.object(utils,"get_moe_a2a_backend",return_value=SimpleNamespace(is_none=lambda: False)):
                with utils.cp_shard_model_inputs(embeds,torch.arange(17),batch,batch.input_ids) as (x,p,ids):
                    torch.testing.assert_close(x[:2],batch.spec_info.hidden_states[:2]+100)
                    self.assertTrue(torch.all(x[2:]==0))
                    torch.testing.assert_close(ids[:2],torch.tensor([2,10]))
                self.assertIs(batch.spec_info.hidden_states,previous)
                self.assertFalse(hasattr(batch,"input_ids_global"))


if __name__ == "__main__":
    unittest.main()
