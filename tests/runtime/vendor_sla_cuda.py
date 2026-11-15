from types import SimpleNamespace
import os
import unittest
from unittest.mock import patch

import torch

import sgl_kernel
from hcu_sglang.multimodal_gen import attention, wan_sla
from hcu_sglang.multimodal_gen.rocm_sparse_linear_attn import RocmSparseLinearAttentionImpl
from sglang.multimodal_gen.runtime.platforms import AttentionBackendEnum, current_platform
from sglang.multimodal_gen.runtime.platforms.cuda import CudaPlatform
from sglang.multimodal_gen.runtime.platforms.plugins import load_plugins, apply_plugin_hooks


class VendorSlaCudaTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        load_plugins()
        apply_plugin_hooks()

    def test_cuda_sla_uses_vendor(self):
        self.assertTrue(current_platform.is_cuda())
        for dtype in [torch.float16, torch.bfloat16]:
            for hidden in [64, 128]:
                selected = CudaPlatform.get_attn_backend_cls_str(AttentionBackendEnum.SLA_ATTN, hidden, dtype)
                self.assertEqual(selected, "hcu_sglang.multimodal_gen.rocm_sparse_linear_attn.RocmSparseLinearAttentionBackend")

    def _construct(self, skip=True, device="cpu", local_blocks=0, topk=0.25):
        from sglang.multimodal_gen.runtime.layers.attention import turbo_layer
        from sglang.multimodal_gen.runtime.layers.attention import selector

        config = SimpleNamespace(attention_type="original", sla_topk=0.1)
        args = SimpleNamespace(attention_backend="sla_attn", attention_backend_config={
            "sla_topk": topk, "sla_skip_linear": skip, "sla_local_blocks": local_blocks,
        })
        holder = SimpleNamespace()

        def construct(model, config, hf_config, quant_config):
            model.attention = turbo_layer.MinimalA2AAttnOp(
                num_heads=2, head_size=128, attention_type=config.attention_type,
                topk=config.sla_topk, supported_attention_backends={AttentionBackendEnum.SLA_ATTN},
            )

        with patch.object(wan_sla, "is_hcu", return_value=True), patch.object(
            attention, "is_hcu", return_value=True
        ), patch("sglang.multimodal_gen.runtime.server_args.server_args.get_global_server_args", return_value=args), patch.object(
            turbo_layer, "get_global_server_args", return_value=args
        ), patch.object(turbo_layer, "get_compute_dtype", return_value=torch.bfloat16), patch.object(
            selector, "get_global_server_args", return_value=args
        ), torch.device(device):
            wan_sla.around_wan_model_init(construct, holder, config, None)
        self.assertEqual(config.attention_type, "original")
        self.assertFalse(wan_sla._wan_init.get())
        return holder.attention.local_attn

    def test_real_construction_skip_has_no_linear_weights(self):
        impl = self._construct(device="meta", local_blocks=2)
        self.assertIsInstance(impl, RocmSparseLinearAttentionImpl)
        self.assertTrue(impl.skip_linear_branch)
        self.assertEqual(impl.topk_ratio, 0.25)
        self.assertEqual(impl.local_blocks, 2)
        self.assertFalse(hasattr(impl, "proj_l"))
        self.assertEqual(list(impl.named_parameters()), [])

    def test_disable_skip_creates_zero_initialized_projection(self):
        impl = self._construct(skip=False)
        self.assertFalse(impl.skip_linear_branch)
        self.assertEqual(impl.proj_l.weight.missing_param_init, "zeros")
        self.assertTrue(torch.equal(impl.proj_l.weight, torch.zeros_like(impl.proj_l.weight)))

    def test_real_other_cuda_backend_preserves_binding(self):
        for platform in [CudaPlatform, current_platform]:
            selected = platform.get_attn_backend_cls_str(AttentionBackendEnum.TORCH_SDPA, 128, torch.bfloat16)
            self.assertTrue(selected.endswith(".sdpa.SDPABackend"))

    def test_other_cuda_requests_preserve_original(self):
        marker = object()
        calls = []

        def original(*args):
            calls.append(args)
            return marker

        with patch.object(attention, "is_hcu", return_value=True):
            for backend in [None, AttentionBackendEnum.TORCH_SDPA, AttentionBackendEnum.SAGE_SLA_ATTN]:
                self.assertIs(attention.around_cuda_backend_selection(original, CudaPlatform, backend, 128, torch.bfloat16), marker)
        with patch.object(attention, "is_hcu", return_value=False):
            self.assertIs(attention.around_cuda_backend_selection(original, CudaPlatform, AttentionBackendEnum.SLA_ATTN, 128, torch.bfloat16), marker)
        self.assertEqual(len(calls), 4)

    def test_invalid_vendor_shape_or_dtype_rejected(self):
        with patch.object(attention, "is_hcu", return_value=True):
            for head, dtype in [(96, torch.bfloat16), (128, torch.float32)]:
                with self.assertRaises(ValueError):
                    attention.around_cuda_backend_selection(lambda *a: self.fail("unexpected fallback"), CudaPlatform, AttentionBackendEnum.SLA_ATTN, head, dtype)

    @unittest.skipUnless(os.environ.get("HCU_PLUGIN_RUN_GPU_TESTS") == "1" and torch.cuda.is_available(), "requires GPU opt-in")
    def test_vendor_d128_numerics_and_changed_input_graph(self):
        import flash_attn

        self.assertNotIn("adapter:", str(flash_attn.__file__))
        for tokens in [513, 2304]:
            impl = self._construct(device="cuda", topk=1.0)
            q = torch.randn(1, tokens, 2, 128, device="cuda", dtype=torch.bfloat16)
            k, v = torch.randn_like(q), torch.randn_like(q)

            originals = [t.clone() for t in (q, k, v)]

            def reference():
                qc, kc, vc = (t.cpu().double().transpose(1, 2) for t in (q, k, v))
                p = torch.softmax((qc @ kc.transpose(-1, -2)) * (128 ** -0.5), -1)
                return (p @ vc).transpose(1, 2)

            with torch.no_grad(), patch.object(impl, "_linear_attention", side_effect=AssertionError("linear branch executed")):
                for _ in range(3): actual = impl(q, k, v)
                self.assertTrue(torch.isfinite(actual).all())
                torch.testing.assert_close(actual.cpu().double(), reference(), rtol=0.02, atol=0.003)
                for original, tensor in zip(originals, (q, k, v)):
                    self.assertTrue(torch.equal(original, tensor))
                graph = torch.cuda.CUDAGraph()
                with torch.cuda.graph(graph): captured = impl(q, k, v)
                for seed in [41, 42, 43]:
                    torch.manual_seed(seed)
                    q.normal_(); k.normal_(); v.normal_()
                    eager = impl(q, k, v)
                    graph.replay()
                    self.assertTrue(torch.equal(captured, eager))
            impl = self._construct(device="cuda", topk=0.25)
            with torch.no_grad():
                partial = impl(q, k, v)
            self.assertTrue(torch.isfinite(partial).all())
            self.assertEqual(partial.shape, q.shape)


if __name__ == "__main__":
    torch.set_num_threads(4)
    unittest.main()
