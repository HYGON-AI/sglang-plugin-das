import unittest
from types import SimpleNamespace
from unittest.mock import patch


class SharedFp8Tests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        from hcu_sglang.compat.aiter import bootstrap_aiter_compat
        bootstrap_aiter_compat()
        from hcu_sglang.environ import register_envs
        register_envs()
        from hcu_sglang.layers import row_fusion
        cls.row = row_fusion

    def test_local_opt_in_fallbacks(self):
        r = self.row

        def mlp(eligible=True):
            down = SimpleNamespace(scheme=SimpleNamespace(supports_fp8_prequantized_input=eligible),
                                   params_dtype=r.N.torch.bfloat16, input_is_parallel=True)
            return SimpleNamespace(shared_experts=SimpleNamespace(swiglu_limit=10., down_proj=down),
                                   _shared_expert_tp1=True)

        flag = r.N.envs.SGLANG_USE_FUSED_SILU_MUL_FP8_QUANT
        with patch.object(r, "_use_fused_silu_mul_fp8_quant", False):
            with flag.override(True), patch.object(r, "_shared_fp8_silu_provider", return_value=None):
                layer = mlp()
                r.configure_dsv41_shared_fp8(layer)
                self.assertFalse(r.supports_fused_silu_mul_fp8_quant_input(layer.shared_experts.down_proj))
            with flag.override(False), patch.object(r, "_shared_fp8_silu_provider", side_effect=AssertionError):
                layer=mlp()
                r.configure_dsv41_shared_fp8(layer)
                self.assertFalse(r.supports_fused_silu_mul_fp8_quant_input(layer.shared_experts.down_proj))
            with flag.override(True), patch.object(r, "_shared_fp8_silu_provider", return_value=self.fail):
                layer=mlp()
                other=mlp()
                r.configure_dsv41_shared_fp8(layer)
                self.assertTrue(r.supports_fused_silu_mul_fp8_quant_input(layer.shared_experts.down_proj))
                self.assertFalse(r.supports_fused_silu_mul_fp8_quant_input(other.shared_experts.down_proj))
                sharded = mlp()
                sharded._shared_expert_tp1 = False
                unclamped = mlp()
                unclamped.shared_experts.swiglu_limit = None
                for item in (mlp(False), sharded, unclamped):
                    r.configure_dsv41_shared_fp8(item)
                    self.assertFalse(r.supports_fused_silu_mul_fp8_quant_input(item.shared_experts.down_proj))
                r.configure_dsv41_shared_fp8(SimpleNamespace(shared_experts=None))

    def test_clamped_eager_and_graph(self):
        import torch
        from hcu_sglang.utils.detection import is_hcu_native_fp8_supported
        if not is_hcu_native_fp8_supported():
            self.skipTest("gfx938 required")
        from lightop.activation import fuse_silu_mul_fp8_quant
        from sglang.srt.models.deepseek_v2 import silu_and_mul_clamp

        r = self.row
        provider = r._shared_fp8_silu_provider()
        self.assertIsNotNone(provider)
        rng = torch.Generator(device="cuda").manual_seed(108)
        gu=torch.randn((20, 4608), device="cuda",dtype=torch.bfloat16,generator=rng)
        gu[0].zero_()
        gu[1,:2304].fill_(13)
        gu[1,2304:].fill_(-13)
        bf16=torch.empty((20,2304),device="cuda",dtype=torch.bfloat16)
        silu_and_mul_clamp(gu,bf16,10.)
        def fused():
            return provider(gu,fp8type=0,limit=10.)
        def check():
            q,s=fused()
            q_ref,s_ref=fuse_silu_mul_fp8_quant(gu,fp8type=0,limit=10.)
            torch.testing.assert_close(q.view(torch.uint8),q_ref.view(torch.uint8),rtol=0,atol=0)
            torch.testing.assert_close(s,s_ref,rtol=0,atol=0)
            reconstructed=q.float()*s
            torch.testing.assert_close(reconstructed,bf16.float(),rtol=.07,atol=.001)
            return q,s
        check()
        graph=torch.cuda.CUDAGraph()
        with torch.cuda.graph(graph):
            q,s=fused()
        previous=None
        for gain in (.25,2.):
            gu.normal_(generator=rng).mul_(gain)
            silu_and_mul_clamp(gu,bf16,10.)
            eager,scales=check()
            graph.replay()
            torch.cuda.synchronize()
            torch.testing.assert_close(q.view(torch.uint8),eager.view(torch.uint8),rtol=0,atol=0)
            torch.testing.assert_close(s,scales,rtol=0,atol=0)
            if previous is not None:
                self.assertFalse(torch.equal(s,previous))
            previous=s.clone()


if __name__=="__main__":
    unittest.main()
