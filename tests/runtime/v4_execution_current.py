# Copyright (c) 2026 Hygon Information Technology Co., Ltd.
# SPDX-License-Identifier: Apache-2.0

from contextlib import ExitStack
from types import SimpleNamespace
from unittest.mock import patch
import inspect,unittest
import torch
from hcu_sglang.compat.aiter import bootstrap_aiter_compat
bootstrap_aiter_compat()
from hcu_sglang.environ import register_envs
register_envs()
from hcu_sglang.models.deepseek.current_v4.execution import core as C

class ConstructionStopped(Exception):pass

class V4ExecutionTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        from sglang.srt.models import deepseek_v4 as native, deepseek_v4_nextn as nextn
        from sglang.srt.plugins.hook_registry import HookRegistry
        from hcu_sglang.models.deepseek.deepseek_v4.hooks import decoder, model, causal_lm
        HookRegistry.reset();decoder.register_hooks();model.register_hooks();causal_lm.register_hooks();HookRegistry.apply_hooks()
        cls.public_model_init=staticmethod(native.DeepseekV4Model.__init__)
        cls.public_causal_init=staticmethod(native.DeepseekV4ForCausalLM.__init__)
        cls.decoder_alias=nextn.DeepseekV4DecoderLayer

    def test_nextn_constructor_alias_uses_current_platform_layer(self):
        self.assertIs(self.decoder_alias,C.DeepseekV4PlatformDecoderLayer)

    def test_hcu_model_uses_live_pp_group_and_five_streams(self):
        self.check_hcu_model_constructor(mock_platform=True)

    @unittest.skipUnless(C._is_hcu, 'requires HCU hardware')
    def test_hcu_model_constructor_with_actual_platform_dispatch(self):
        self.check_hcu_model_constructor(mock_platform=False)

    def check_hcu_model_constructor(self, *, mock_platform):
        config=SimpleNamespace(hidden_size=8,vocab_size=16,rms_norm_eps=1e-6,num_hidden_layers=1,hc_mult=4,hc_eps=1e-5,hc_pre_from_prev_sublayer=False,model_type='deepseek_v4',vision_n_layers=0)
        pp=SimpleNamespace(is_first_rank=True,is_last_rank=True);streams=[];calls=[];factory_signature=inspect.signature(C.make_pp_layers)
        def stream():
            obj=object();streams.append(obj);return obj
        def make_layers(*args,**kwargs):
            factory_signature.bind(*args,**kwargs)
            calls.append((args,kwargs));raise ConstructionStopped()
        def embedding(vocab,size,**kwargs):return torch.nn.Embedding(vocab,size)
        obj=object.__new__(C.NativeDeepseekV4Model)
        with ExitStack() as stack:
            stack.enter_context(patch.object(C.NativeDeepseekV4Model,'__init__',side_effect=AssertionError('old parent constructor was invoked')))
            if mock_platform:
                stack.enter_context(patch.object(C,'_is_cuda',False));stack.enter_context(patch.object(C,'_is_hcu',True))
            stack.enter_context(patch.object(C,'get_parallel',return_value=SimpleNamespace(pp_group=pp)))
            stack.enter_context(patch.object(C,'is_dp_attention_enabled',return_value=False))
            stack.enter_context(patch.object(C,'VocabParallelEmbedding',side_effect=embedding))
            stack.enter_context(patch.object(C.EngramLayout,'from_config',return_value=None))
            stack.enter_context(patch.object(torch,'get_device_module',return_value=SimpleNamespace(Stream=stream)))
            stack.enter_context(patch.object(C,'make_pp_layers',side_effect=make_layers))
            with self.assertRaises(ConstructionStopped):self.public_model_init(obj,config)
        self.assertIs(obj.pp_group,pp);self.assertEqual(len(streams),5)
        self.assertEqual(obj.alt_streams,streams);self.assertEqual(len(calls),1)
        self.assertIsNone(calls[0][1]['final_read'])
        self.assertIsNone(calls[0][1]['stage_facts'])
        self.assertIs(C.DeepseekV4DecoderLayer,C.DeepseekV4PlatformDecoderLayer)

    def test_platform_mixin_initializes_state_after_base(self):
        calls=[]
        def base(self,*args,**kwargs):
            torch.nn.Module.__init__(self);calls.append((args,kwargs));self.hc_pre_from_prev_sublayer=False
        with patch.object(C.DeepseekV4DecoderLayerBase,'__init__',base):
            layer=C.DeepseekV4PlatformDecoderLayer('config',layer_id=3)
        self.assertEqual(calls,[(('config',),{'layer_id':3})])
        self.assertIsNone(layer._next_layer);self.assertIsNone(layer.local_boundary);self.assertIsNone(layer.next_boundary)

    def test_native_causal_instance_reaches_current_model_constructor(self):
        config=SimpleNamespace(model_type='deepseek_v4',hidden_size=8,vocab_size=16,vision_n_layers=0)
        pp=SimpleNamespace(is_first_rank=True,is_last_rank=True);calls=[]
        def model(*args,**kwargs):calls.append((args,kwargs));raise ConstructionStopped()
        obj=object.__new__(C.NativeDeepseekV4ForCausalLM)
        with patch.object(C.NativeDeepseekV4ForCausalLM,'__init__',side_effect=AssertionError('old parent constructor was invoked')),patch.object(C,'get_parallel',return_value=SimpleNamespace(pp_group=pp,tp_size=4)),patch.object(C.NativeDeepseekV4ForCausalLM,'determine_num_fused_shared_experts',lambda owner:setattr(owner,'num_fused_shared_experts',0)),patch.object(C,'DeepseekV4Model',side_effect=model):
            with self.assertRaises(ConstructionStopped):self.public_causal_init(obj,config)
        self.assertEqual(len(calls),1)

    def test_native_ipc_rebinds_mapped_parameters_and_validates_aliases(self):
        from hcu_sglang.weight_cache.client import ChannelwiseIpcLoader
        from hcu_sglang.weight_cache.extra_state import export_extra_state

        def model_on(device):
            m = object.__new__(C.NativeDeepseekV4ForCausalLM)
            torch.nn.Module.__init__(m)
            m.model = torch.nn.Module()
            m.model.layers = torch.nn.ModuleList()
            for index in range(3):
                layer = object.__new__(C.DeepseekV4PlatformDecoderLayer)
                torch.nn.Module.__init__(layer)
                layer.hc_cfg = None
                layer.hc_pre_from_prev_sublayer = True
                layer.engram = torch.nn.Identity() if index == 2 else None
                layer._next_layer = None
                for stage in ("attn", "ffn"):
                    for key in ("fn", "scale", "base"):
                        value = torch.full((2,), index + 1, dtype=torch.float32, device=device)
                        setattr(layer, f"hc_{stage}_{key}", torch.nn.Parameter(value))
                    setattr(layer, f"_hc_{stage}_tf32_parts", None)
                    setattr(layer, f"_hc_{stage}_bf16_parts", None)
                layer.input_layernorm = torch.nn.LayerNorm(2, device=device)
                layer.post_attention_layernorm = torch.nn.LayerNorm(2, device=device)
                layer.register_buffer("row_offsets", torch.arange(2, device=device))
                layer._init_hyper_connections()
                m.model.layers.append(layer)
            return m

        daemon, engine = model_on("cpu"), model_on("meta")
        first, second, _ = daemon.model.layers
        first.bind_next(second)
        second.bind_next(daemon.model.layers[2])
        second._hc_attn_bf16_parts = (torch.zeros(3), torch.ones(2))
        second._init_hyper_connections()
        payload = export_extra_state(daemon, lambda t: t)
        loader = object.__new__(ChannelwiseIpcLoader)
        loader._transport_backend = SimpleNamespace(import_extra_tensor=lambda t: t)
        loader._deferred_alias_entries = {}
        all_tensors = (
            [(n, t, True) for n, t in daemon.named_parameters(remove_duplicate=False)]
            + [(n, t, False) for n, t in daemon.named_buffers(remove_duplicate=False)]
        )
        for name, t, is_param in all_tensors:
            if "._next_layer." in name:
                loader._deferred_alias_entries[name] = {
                    "shape": list(t.shape), "dtype": str(t.dtype).removeprefix("torch."),
                    "is_param": is_param,
                }
            else:
                loader._set_module_tensor(engine, name, t.detach().clone(), is_param)
        self.assertTrue(loader._deferred_alias_entries)
        self.assertEqual(engine.model.layers[0].attn_hc.fn.device.type, "meta")
        with patch.object(engine, "post_load_weights", side_effect=AssertionError("disk rewrite")):
            loader._apply_extra_state(engine, payload)
        first, second, third = engine.model.layers
        self.assertIs(first._next_layer, second)
        self.assertIsNone(second._next_layer)  # An Engram boundary stops the chain.
        self.assertIsNone(third._next_layer)
        self.assertIs(first._next_layer.row_offsets, second.row_offsets)
        self.assertIs(first._next_layer.attn_hc.fn, second.hc_attn_fn)
        self.assertIs(second.attn_hc.bf16_parts, second._hc_attn_bf16_parts)
        self.assertFalse(loader._deferred_alias_entries)
        for field, wrong in (("dtype", "int32"), ("is_param", False)):
            entry = {"shape": [2], "dtype": "float32", "is_param": True, field: wrong}
            loader._deferred_alias_entries = {"model.layers.0._next_layer.hc_attn_fn": entry}
            with self.assertRaisesRegex(RuntimeError, "1 of 1"):
                loader._check_deferred_aliases(engine)

class MhcSharedTypesTests(unittest.TestCase):
    def test_target_attention_and_execution_share_state_identity(self):
        from sglang.srt.models import deepseek_v4_mhc as native
        from hcu_sglang.models.deepseek.current_v4 import attention
        for name in ['HcConfig','HcSubLayer','HcNormed','HcQuantized','AttnOutput','HcPending','HcState','HcNextBoundary']:
            self.assertIs(getattr(C.mhc,name),getattr(native,name))
        self.assertIs(attention.mhc.AttnOutput,C.mhc.AttnOutput)

if __name__=='__main__':unittest.main()
