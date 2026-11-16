from contextlib import nullcontext
from types import SimpleNamespace
import os
import unittest
from unittest.mock import patch

import torch

import sgl_kernel
from hcu_sglang.multimodal_gen import register_legacy, wan_sla
from sglang.multimodal_gen.runtime.layers.attention.backends import sparse_linear_attn as sla
from sglang.multimodal_gen.runtime.platforms import plugins
from sglang.srt.plugins.hook_registry import HookRegistry as SrtRegistry


class WanSlaCommunityTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        plugins.load_plugins()
        plugins.apply_plugin_hooks()

    def _construct_wan_attention(self, skip=True, device="cpu", hcu=True):
        config = SimpleNamespace(attention_type="original", sla_topk=0.1)
        server_args = SimpleNamespace(
            attention_backend="sla_attn",
            attention_backend_config={"sla_topk": 0.25, "sla_skip_linear": skip},
        )
        holder = SimpleNamespace()

        def construct(model, actual_config, hf_config, quant_config):
            self.assertEqual(actual_config.attention_type, "sla" if hcu else "original")
            self.assertEqual(actual_config.sla_topk, 0.25 if hcu else 0.1)
            model.attention = sla.SparseLinearAttentionImpl(
                num_heads=2, head_size=64, topk_ratio=actual_config.sla_topk
            )

        with patch.object(wan_sla, "is_hcu", return_value=hcu), patch(
            "sglang.multimodal_gen.runtime.server_args.server_args.get_global_server_args",
            return_value=server_args,
        ), torch.device(device):
            wan_sla.around_wan_model_init(construct, holder, config, None)
        self.assertEqual(config.attention_type, "original")
        self.assertEqual(config.sla_topk, 0.1)
        self.assertFalse(wan_sla._wan_init.get())
        return holder.attention

    def test_diffusion_registry_and_legacy_ownership(self):
        targets = {
            "sglang.multimodal_gen.runtime.models.dits.wanvideo.WanTransformer3DModel.__init__",
            "sglang.multimodal_gen.runtime.layers.attention.backends.sparse_linear_attn._initialize_linear_attention",
            "sglang.multimodal_gen.runtime.layers.attention.backends.sparse_linear_attn.SparseLinearAttentionImpl.forward",
        }
        self.assertTrue(targets <= plugins.HookRegistry._patched)
        self.assertFalse(targets & set(SrtRegistry._hooks))
        before = {target: len(hooks) for target, hooks in plugins.HookRegistry._hooks.items()}
        register_legacy()
        plugins.load_plugins()
        plugins.apply_plugin_hooks()
        self.assertEqual(before, {t: len(h) for t, h in plugins.HookRegistry._hooks.items()})

    def test_meta_skip_has_no_projection_weights(self):
        attention = self._construct_wan_attention(device="meta")
        self.assertTrue(attention._hcu_wan_skip_linear)
        self.assertFalse(hasattr(attention, "proj_l"))
        self.assertFalse(hasattr(attention, "feature_map_q"))
        self.assertEqual(list(attention.named_parameters()), [])
        self.assertEqual(dict(attention.state_dict()), {})

    def test_skip_uses_only_official_sparse_path(self):
        attention = self._construct_wan_attention()
        query = torch.randn(1, 17, 2, 64, dtype=torch.bfloat16)
        key, value = torch.randn_like(query), torch.randn_like(query)
        sparse_result = torch.randn(1, 2, 17, 64, dtype=torch.bfloat16)
        with patch.object(sla, "get_block_map", return_value=(None, None, 3)) as block_map, patch.object(
            sla._attention, "apply", return_value=sparse_result
        ) as sparse, patch.object(torch, "matmul", side_effect=AssertionError("linear branch executed")):
            result = attention(query, key, value)
        self.assertTrue(torch.equal(result, sparse_result.transpose(1, 2)))
        self.assertEqual(block_map.call_args.kwargs["topk_ratio"], 0.25)
        for got, original in zip(sparse.call_args.args[:3], (query, key, value)):
            self.assertTrue(torch.equal(got, original.transpose(1, 2)))

    def test_disabled_skip_keeps_trainable_projection_and_linear_computation(self):
        attention = self._construct_wan_attention(skip=False)
        self.assertFalse(attention._hcu_wan_skip_linear)
        self.assertEqual(attention.proj_l.weight.missing_param_init, "zeros")
        self.assertEqual(attention.proj_l.bias.missing_param_init, "zeros")
        attention.dtype = torch.float32
        with torch.no_grad():
            attention.proj_l.weight.copy_(torch.eye(64))
            attention.proj_l.bias.fill_(0.25)
        query = torch.randn(1, 17, 2, 64)
        key, value = torch.randn_like(query), torch.randn_like(query)
        sparse_result = torch.zeros(1, 2, 17, 64)
        with patch.object(sla, "get_block_map", return_value=(None, None, 3)), patch.object(
            sla._attention, "apply", return_value=sparse_result
        ), patch.object(torch.amp, "autocast", return_value=nullcontext()), patch.object(
            attention, "_calc_linear_attention_with_torch",
            wraps=attention._calc_linear_attention_with_torch,
        ) as linear:
            result = attention(query, key, value)
        self.assertEqual(linear.call_count, 1)
        self.assertTrue(torch.isfinite(result).all())
        self.assertGreater(result.abs().max().item(), 0)

    def test_non_wan_and_non_hcu_keep_official_initialization(self):
        standalone = sla.SparseLinearAttentionImpl(num_heads=2, head_size=64)
        non_hcu = self._construct_wan_attention(hcu=False)
        for attention in (standalone, non_hcu):
            self.assertTrue(hasattr(attention, "proj_l"))
            self.assertFalse(hasattr(attention, "_hcu_wan_skip_linear"))
            self.assertFalse(hasattr(attention.proj_l.weight, "missing_param_init"))

    def test_wan_context_resets_on_constructor_failure(self):
        config = SimpleNamespace(attention_type="original", sla_topk=0.1)
        server_args = SimpleNamespace(attention_backend="sla_attn", attention_backend_config={})
        with patch.object(wan_sla, "is_hcu", return_value=True), patch(
            "sglang.multimodal_gen.runtime.server_args.server_args.get_global_server_args",
            return_value=server_args,
        ), self.assertRaisesRegex(RuntimeError, "constructor failed"):
            wan_sla.around_wan_model_init(
                lambda *args: (_ for _ in ()).throw(RuntimeError("constructor failed")),
                None, config, None,
            )
        self.assertFalse(wan_sla._wan_init.get())
        self.assertEqual(config.attention_type, "original")
        self.assertEqual(config.sla_topk, 0.1)

    @unittest.skipUnless(
        os.environ.get("HCU_PLUGIN_RUN_GPU_TESTS") == "1" and torch.cuda.is_available(),
        "requires explicit GPU test opt-in",
    )
    def test_real_sparse_gpu_and_changed_input_graph(self):
        sparse = self._construct_wan_attention(device="cuda")
        with torch.device("cuda"):
            full = sla.SparseLinearAttentionImpl(num_heads=2, head_size=64, topk_ratio=0.25)
        q = torch.randn(1, 513, 2, 64, device="cuda", dtype=torch.bfloat16)
        k, v = torch.randn_like(q), torch.randn_like(q)
        with torch.no_grad():
            for _ in range(3):
                expected = full(q, k, v)
                actual = sparse(q, k, v)
            torch.cuda.synchronize()
            self.assertTrue(torch.isfinite(actual).all())
            torch.testing.assert_close(actual, expected, atol=0, rtol=0)
            graph = torch.cuda.CUDAGraph()
            with torch.cuda.graph(graph):
                captured = sparse(q, k, v)
            for seed in [41, 42, 43]:
                torch.manual_seed(seed)
                q.normal_(); k.normal_(); v.normal_()
                expected = full(q, k, v)
                graph.replay()
                torch.testing.assert_close(captured, expected, atol=0, rtol=0)


if __name__ == "__main__":
    unittest.main()
