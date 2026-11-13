import importlib.util
import logging
import os
import sys
import types
import unittest
from enum import Enum
from pathlib import Path
from unittest.mock import patch
import torch
from hcu_sglang.multimodal_gen import attention

ROOT=Path(__file__).resolve().parents[1]
PREFIX="sglang.multimodal_gen.runtime.layers.attention.backends."
PLUGIN="hcu_sglang.multimodal_gen.flash_attn_2"

class H3FlashAttention2Tests(unittest.TestCase):
    def setUp(self):
        official=Path(os.environ.get("OFFICIAL",str(ROOT.parent/"sglang-official-0.5.22")))
        self.base=official/"python"
        if not (self.base/"sglang/multimodal_gen/runtime/layers/attention/backends/flash_attn_2.py").is_file():
            self.skipTest("fixed official FA2 source required")
        fake={}
        names=(PREFIX+"attention_backend",PREFIX+"flash_attn_2","sglang.kernels.kernel_api_logging","sglang.multimodal_gen.runtime.platforms","sglang.multimodal_gen.runtime.utils.logging_utils","flash_attn")
        for name in names:
            parts=name.split(".")
            for i in range(1,len(parts)+1):
                key=".".join(parts[:i])
                if key not in fake:
                    fake[key]=types.ModuleType(key);fake[key].__path__=[]
                if i>1:setattr(fake[".".join(parts[:i-1])],parts[i-1],fake[key])
        fake["sglang.kernels.kernel_api_logging"].wrap_method_with_debug_kernel_once=lambda obj,*a,**k:obj
        fake["sglang.multimodal_gen.runtime.platforms"].AttentionBackendEnum=Enum("Backend",["FA2"])
        fake["sglang.multimodal_gen.runtime.utils.logging_utils"].init_logger=logging.getLogger
        self.calls=[]
        def provider(*args,**kwargs):
            self.calls.append((args,kwargs));return (args[0],"metadata")
        fake["flash_attn"].flash_attn_varlen_func=provider
        fake["flash_attn"].flash_attn_func=lambda **kwargs:kwargs["q"]
        self.stack=patch.dict(sys.modules,fake);self.stack.start();self.addCleanup(self.stack.stop)
        self.load(PREFIX+"attention_backend",self.base/"sglang/multimodal_gen/runtime/layers/attention/backends/attention_backend.py")
        self.official=self.load(PREFIX+"flash_attn_2",self.base/"sglang/multimodal_gen/runtime/layers/attention/backends/flash_attn_2.py")
        self.mod=self.load(PLUGIN,ROOT/"hcu_sglang/multimodal_gen/flash_attn_2.py")

    def load(self,name,path):
        spec=importlib.util.spec_from_file_location(name,path);mod=importlib.util.module_from_spec(spec)
        sys.modules[name]=mod;spec.loader.exec_module(mod);return mod

    def test_official_requirement_admits_only_real_varlen_implementation(self):
        self.assertFalse(self.official.FlashAttention2Backend.supports_packed_varlen())
        self.assertTrue(self.mod.HcuFlashAttention2Backend.supports_packed_varlen())
        base=sys.modules[PREFIX+"attention_backend"]
        self.assertEqual(self.mod.HcuFlashAttention2Backend.unsupported_requirements(base.AttentionRequirements(packed_varlen=True)),())

    def test_legacy_provider_receives_packed_bounds_and_math_flags(self):
        impl=self.mod.HcuFlashAttention2Impl(2,4,True,0.5,dropout_p=0.125)
        x=torch.ones(5,2,8)[:,:,::2];bounds=torch.tensor([0,2,5],dtype=torch.int64)
        out=impl.forward_varlen(x,x,x,cu_seqlens=bounds,max_seqlen=3,cu_seqlens_host=(0,2,5))
        args,kwargs=self.calls[0]
        self.assertTrue(all(t.is_contiguous() for t in args[:3]))
        self.assertEqual(args[3].dtype,torch.int32)
        self.assertTrue(torch.equal(args[3],bounds.to(torch.int32)))
        self.assertIs(args[3],args[4]);self.assertEqual(args[5:],(3,3))
        self.assertEqual(kwargs,dict(dropout_p=0.125,softmax_scale=0.5,causal=True))
        self.assertTrue(torch.equal(out,x))

    def test_dense_forward_reuses_official_implementation(self):
        self.assertIs(self.mod.HcuFlashAttention2Impl.forward,self.official.FlashAttention2Impl.forward)

    def test_hcu_routing_selects_owned_backend_and_other_platform_uses_original(self):
        with patch.object(attention,"is_hcu",return_value=True),patch.object(attention.importlib.util,"find_spec",return_value=object()):
            self.assertEqual(attention.around_rocm_backend_selection(lambda *a:"base",None,types.SimpleNamespace(name="FA"),128,torch.bfloat16),PLUGIN+".HcuFlashAttention2Backend")
        with patch.object(attention,"is_hcu",return_value=False):
            self.assertEqual(attention.around_rocm_backend_selection(lambda *a:"base",None,None,128,torch.bfloat16),"base")
