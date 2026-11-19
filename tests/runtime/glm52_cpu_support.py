# Copyright (c) 2026 Hygon Information Technology Co., Ltd.
# SPDX-License-Identifier: Apache-2.0
import importlib.util
import sys
import types
from contextlib import contextmanager
from pathlib import Path

@contextmanager
def isolated_runtime():
    import torch
    modules = {}
    def module(name):
        if name not in modules:
            value = types.ModuleType(name)
            value.__path__ = []
            modules[name] = value
            if '.' in name:
                parent, child = name.rsplit('.', 1)
                setattr(module(parent), child, value)
        return modules[name]
    def load(name, path):
        parent, child = name.rsplit('.', 1)
        spec = importlib.util.spec_from_file_location(name, path)
        value = importlib.util.module_from_spec(spec)
        if name not in previous:
            previous[name] = sys.modules.get(name)
        sys.modules[name] = value
        setattr(module(parent), child, value)
        spec.loader.exec_module(value)
        return value
    root = Path(__file__).resolve().parents[2]
    native = module('sglang.srt.models.deepseek_v2')
    module('sglang.srt.models.deepseek_nextn')
    module('sglang.srt.layers.attention.dsa.utils').dsa_use_prefill_cp = lambda _: False
    module('sglang.srt.layers.cp.utils')
    utils = module('sglang.srt.utils')
    utils.get_bool_env_var = lambda *_: True
    utils.is_musa = lambda: False
    utils.is_npu = lambda: False
    module('sglang.srt.layers.moe.utils')
    module('sglang.srt.layers.moe.token_dispatcher.standard')
    config_module = module('sglang.srt.configs.model_config')
    config_module.is_glm_moe_dsa = lambda c: getattr(c, 'model_type', '') == 'glm_moe_dsa'
    module('hcu_sglang.utils.detection').is_hcu = lambda: True
    compat = module('hcu_sglang.models.deepseek.current_v4.hcu_backend.compat')
    compat.is_hcu = lambda: True
    cp = module('hcu_sglang.models.deepseek.current_v4.hcu_backend.hcu_cp_utils')
    cp.mla_use_prefill_cp = lambda _: False
    module('hcu_sglang.models.deepseek.deepseek_nextn.methods')
    module('hcu_sglang.layers.moe')
    class Gate(torch.nn.Module):
        pass
    native.MoEGate = Gate
    native.F = torch.nn.functional
    native.is_glm_moe_dsa = config_module.is_glm_moe_dsa
    flag = types.SimpleNamespace(value=True)
    native.envs = types.SimpleNamespace(
        SGLANG_HCU_GLM_CT_BF16_ROUTER=types.SimpleNamespace(get=lambda: flag.value),
        SGLANG_MOE_ROUTER_USE_CONFIG_DTYPE=types.SimpleNamespace(get=lambda: False),
    )
    native._is_cpu = False
    native._is_hip = False
    native._is_cuda = True
    native._use_aiter = False
    native.tiny_router_gemm_max_tokens = lambda **_: 16
    native.use_intel_amx_backend = lambda _: False
    native.get_exec = lambda: types.SimpleNamespace(deterministic=types.SimpleNamespace(enable_deterministic_inference=False))
    native.tiny_gemm_bf16 = lambda x, w, **_: torch.nn.functional.linear(x.float(), w.float())
    previous = {name: sys.modules.get(name) for name in modules}
    sys.modules.update(modules)
    try:
        nextn = load('hcu_sglang.models.deepseek.deepseek_nextn.methods.model', root / 'hcu_sglang/models/deepseek/deepseek_nextn/methods/model.py')
        gate = load('hcu_sglang.layers.moe.deepseek_gate', root / 'hcu_sglang/layers/moe/deepseek_gate.py')
        Gate.__init__ = gate.__init__
        Gate.forward = gate.forward
        yield types.SimpleNamespace(torch=torch, native=native, gate=gate, nextn=nextn, flag=flag, compat=compat, root=root, load=load, module=module)
    finally:
        for name, original in previous.items():
            if original is None:
                sys.modules.pop(name, None)
            else:
                sys.modules[name] = original
