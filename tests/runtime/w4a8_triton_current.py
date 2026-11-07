# Copyright (c) 2026 Hygon Information Technology Co., Ltd.
# SPDX-License-Identifier: Apache-2.0

from types import SimpleNamespace
from unittest.mock import patch
import os,unittest
import torch
import torch.nn.functional as F
from hcu_sglang.compat.aiter import bootstrap_aiter_compat
bootstrap_aiter_compat()
from hcu_sglang.environ import register_envs
register_envs()
from hcu_sglang.layers.quantization import slimquant_w4a8_marlin as Q

class W4A8CurrentTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        from hcu_sglang.layers.moe import alignment
        from sglang.srt.plugins.hook_registry import HookRegistry
        HookRegistry.reset();alignment.register_hooks();HookRegistry.apply_hooks()

    def test_explicit_triton_selects_raw_weight_method(self):
        from sglang.srt.layers.moe.fused_moe_triton import FusedMoE
        layer=object.__new__(FusedMoE);torch.nn.Module.__init__(layer)
        with patch.dict(os.environ,{'SGLANG_W4A8_TPMOE_BACKEND':'triton'}),patch.object(Q,'get_moe_a2a_backend',return_value=SimpleNamespace(is_megamoe=lambda:False,is_deepep=lambda:False)):
            method=Q.SlimQuantW4A8Int8MarlinConfig().get_quant_method(layer,'model.layers.0.mlp.experts')
        self.assertEqual(type(method).__name__,'SlimQuantW4A8Int8TritonMoEMethod')
        layer.w13_weight=torch.nn.Parameter(torch.tensor([[[18,52]]],dtype=torch.int8),requires_grad=False)
        layer.w2_weight=torch.nn.Parameter(torch.tensor([[[86,120]]],dtype=torch.int8),requires_grad=False)
        layer.w13_weight_scale=torch.nn.Parameter(torch.ones(1,1,1),requires_grad=False);layer.w2_weight_scale=torch.nn.Parameter(torch.ones(1,1,1),requires_grad=False)
        old=layer.w13_weight.clone();ptr=layer.w13_weight.data_ptr();method.process_weights_after_loading(layer)
        self.assertTrue(torch.equal(layer.w13_weight,old));self.assertEqual(layer.w13_weight.data_ptr(),ptr)

    @unittest.skipUnless(torch.cuda.is_available(),'GPU unavailable')
    def test_real_triton_moe_matches_int4_routing_reference(self):
        Q._ensure_lightop_w4a8_triton_available()
        generator=torch.Generator().manual_seed(13);tokens=3;experts=2;hidden=128;inter=64;topk=2
        w1=torch.randint(-2,3,(experts,2*inter,hidden),generator=generator,dtype=torch.int8).cuda();w2=torch.randint(-2,3,(experts,hidden,inter),generator=generator,dtype=torch.int8).cuda()
        def pack(w):return (((w[...,0::2].to(torch.int16)&15)<<4)|(w[...,1::2].to(torch.int16)&15)).to(torch.int8).contiguous()
        x=(torch.randn(tokens,hidden,generator=generator)*.1).bfloat16().cuda();ids=torch.tensor([[0,1],[1,0],[0,1]],device='cuda',dtype=torch.int32);weights=torch.tensor([[.25,.75],[.5,.5],[.75,.25]],device='cuda')
        s1=torch.full((experts,2*inter,1),1/16,device='cuda');s2=torch.full((experts,hidden,1),1/16,device='cuda');scratch=torch.empty(tokens*topk*max(2*inter,hidden),device='cuda',dtype=x.dtype)
        out=Q.fused_experts_impl_w4a8_triton(x,pack(w1),pack(w2),weights,ids,scratch,activation='silu',apply_router_weight_on_input=False,global_num_experts=experts,expert_map=None,w1_scale=s1,w2_scale=s2,routed_scaling_factor=1.5)
        qx,xs=Q.per_token_quant_int8(x);ref=torch.empty_like(x)
        for token,choices in enumerate([[0,1],[1,0],[0,1]]):
            values=[]
            for slot,expert in enumerate(choices):
                first=((qx[token].float()@w1[expert].float().T)*xs[token]).bfloat16();gate,up=first.chunk(2)
                activated=(F.silu(gate)*up).view(1,-1);qa,sa=Q.per_token_quant_int8(activated)
                second=((qa.float()@w2[expert].float().T)*sa*weights[token,slot]).bfloat16();values.append(second)
            ref[token]=(torch.cat(values).sum(0).bfloat16()*1.5).bfloat16()
        torch.testing.assert_close(out,ref,rtol=.03,atol=.03)
        print('W4A8 Triton GPU max abs error:',(out.float()-ref.float()).abs().max().item())

class W4A8GlmShapeGraphTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        from hcu_sglang.layers.moe import alignment
        from sglang.srt.plugins.hook_registry import HookRegistry
        HookRegistry.reset();alignment.register_hooks();HookRegistry.apply_hooks()

    @unittest.skipUnless(torch.cuda.is_available(),'GPU unavailable')
    def test_glm_tp4_shape_runs_and_replays_graph(self):
        Q._ensure_lightop_w4a8_triton_available()
        x=torch.zeros(1,6144,device='cuda',dtype=torch.bfloat16)
        w1=torch.zeros(256,1024,3072,device='cuda',dtype=torch.int8)
        w2=torch.zeros(256,6144,256,device='cuda',dtype=torch.int8)
        s1=torch.full((256,1024,1),1/16,device='cuda');s2=torch.full((256,6144,1),1/16,device='cuda')
        ids=torch.arange(8,device='cuda',dtype=torch.int32).reshape(1,8);weights=torch.full((1,8),.125,device='cuda')
        scratch=torch.empty(8*6144,device='cuda',dtype=x.dtype)
        def run():
            return Q.fused_experts_impl_w4a8_triton(x,w1,w2,weights,ids,scratch,activation='silu',apply_router_weight_on_input=False,global_num_experts=256,expert_map=None,w1_scale=s1,w2_scale=s2,routed_scaling_factor=1.)
        warm=run();torch.cuda.synchronize();self.assertEqual(warm.abs().max().item(),0.)
        graph=torch.cuda.CUDAGraph()
        with torch.cuda.graph(graph):out=run()
        graph.replay();torch.cuda.synchronize()
        self.assertEqual(out.shape,x.shape);self.assertEqual(out.abs().max().item(),0.)

class W4A8KimiSituTests(unittest.TestCase):
    """Kimi K3.1 channel-wise W4A8 on the Triton backend (SituGLU experts).

    KimiK3MoE drives the runner with activation="situ" and passes
    activation_situ_beta/linear_beta as gemm1_alpha/gemm1_clamp_limit
    (dims per the kimi_k31 unit-tree: latent experts in moe_hidden space).
    """

    @classmethod
    def setUpClass(cls):
        from hcu_sglang.layers.moe import alignment
        from sglang.srt.plugins.hook_registry import HookRegistry
        HookRegistry.reset();alignment.register_hooks();HookRegistry.apply_hooks()

    def test_apply_routes_situ_betas_from_runner_config(self):
        method=object.__new__(Q.SlimQuantW4A8Int8TritonMoEMethod)
        layer=SimpleNamespace(
            w13_weight=torch.zeros(8,256,32,dtype=torch.int8),
            w2_weight=torch.zeros(8,64,64,dtype=torch.int8),
            w13_weight_scale=torch.ones(8,256,1),w2_weight_scale=torch.ones(8,64,1),
            expert_map=None)
        x=torch.zeros(5,64);ids=torch.zeros(5,2,dtype=torch.int32);w=torch.full((5,2),.5)
        captured={}
        def fake_impl(*a,**k):
            captured.update(k);return a[0]
        # full kimi runner config (situ + both betas), then a silu-shaped
        # config carrying no gemm1_* at all: betas must default, not crash
        for rc,want_beta,want_lb,want_rsf in (
            (dict(routed_scaling_factor=1.5,apply_router_weight_on_input=False,num_experts=8,activation='situ',gemm1_alpha=1.25,gemm1_clamp_limit=8.0),1.25,8.0,1.5),
            (dict(routed_scaling_factor=None,apply_router_weight_on_input=False,num_experts=8,activation='situ'),1.0,None,1.0),
        ):
            method.moe_runner_config=SimpleNamespace(**rc)
            with patch.object(Q,'get_moe_cache',lambda *a,**k:None),patch.object(Q,'fused_experts_impl_w4a8_triton',fake_impl):
                out=method._apply_triton(layer,x,w,ids,rc['activation'])
            self.assertIs(out,x)
            self.assertEqual(captured['activation'],'situ')
            self.assertEqual(captured['situ_beta'],want_beta)
            self.assertEqual(captured['situ_linear_beta'],want_lb)
            self.assertEqual(captured['routed_scaling_factor'],want_rsf)

    @unittest.skipUnless(torch.cuda.is_available(),'GPU unavailable')
    def test_real_triton_moe_situ_matches_kimi_reference(self):
        Q._ensure_lightop_w4a8_triton_available()
        generator=torch.Generator().manual_seed(31)
        # kimi-k31-shaped latent MoE: experts run in moe_hidden space
        # (HIDDEN=256, LATENT=64, EXPERTS=8, moe_intermediate=128)
        tokens=5;experts=8;latent=64;inter=128;topk=2
        w1=torch.randint(-2,3,(experts,2*inter,latent),generator=generator,dtype=torch.int8).cuda()
        w2=torch.randint(-2,3,(experts,latent,inter),generator=generator,dtype=torch.int8).cuda()
        def pack(w):return (((w[...,0::2].to(torch.int16)&15)<<4)|(w[...,1::2].to(torch.int16)&15)).to(torch.int8).contiguous()
        x=(torch.randn(tokens,latent,generator=generator)*.1).bfloat16().cuda()
        ids=torch.tensor([[0,1],[1,2],[2,3],[3,4],[4,0]],device='cuda',dtype=torch.int32)
        weights=torch.tensor([[.25,.75],[.5,.5],[.75,.25],[.6,.4],[.1,.9]],device='cuda')
        s1=torch.full((experts,2*inter,1),1/16,device='cuda');s2=torch.full((experts,latent,1),1/16,device='cuda')
        scratch=torch.empty(tokens*topk*max(2*inter,latent),device='cuda',dtype=x.dtype)
        beta,linear_beta=1.25,8.0
        out=Q.fused_experts_impl_w4a8_triton(x,pack(w1),pack(w2),weights,ids,scratch,activation='situ',apply_router_weight_on_input=False,global_num_experts=experts,expert_map=None,w1_scale=s1,w2_scale=s2,routed_scaling_factor=1.5,situ_beta=beta,situ_linear_beta=linear_beta)
        qx,xs=Q.per_token_quant_int8(x);ref=torch.empty_like(x)
        def situ(gate,up):
            g=beta*torch.tanh(gate/beta)*torch.sigmoid(gate)
            u=linear_beta*torch.tanh(up/linear_beta)
            return g*u
        choices=[[0,1],[1,2],[2,3],[3,4],[4,0]]
        for token,expert_choices in enumerate(choices):
            values=[]
            for slot,expert in enumerate(expert_choices):
                first=((qx[token].float()@w1[expert].float().T)*xs[token]).bfloat16()
                gate,up=first.float().chunk(2)
                activated=situ(gate,up).bfloat16().view(1,-1);qa,sa=Q.per_token_quant_int8(activated)
                second=((qa.float()@w2[expert].float().T)*sa*weights[token,slot]).bfloat16();values.append(second)
            ref[token]=(torch.cat(values).sum(0).bfloat16()*1.5).bfloat16()
        torch.testing.assert_close(out,ref,rtol=.03,atol=.03)
        print('W4A8 Triton GPU (kimi situ) max abs error:',(out.float()-ref.float()).abs().max().item())

if __name__=='__main__':unittest.main()
