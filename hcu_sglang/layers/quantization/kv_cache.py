# Copyright (c) 2026 Hygon Information Technology Co., Ltd.
# Modified by Hygon Information Technology Co., Ltd., 2026.

# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
# Adapted from https://github.com/vllm-project/vllm/blob/main/vllm/model_executor/layers/quantization/kv_cache.py

from __future__ import annotations

import sglang.srt.layers.quantization.kv_cache as N
from hcu_sglang.utils.detection import is_hcu
from sglang.srt.environ import envs

globals().update({name: value for name, value in vars(N).items() if not name.startswith("__")})


def process_weights_after_loading(self, layer) -> None:
    if layer.k_scale > 0.0 and layer.v_scale > 0.0:
        # We prefer to use separate k_scale and v_scale if present
        k_scale = layer.k_scale.to("cpu").tolist()
        v_scale = layer.v_scale.to("cpu").tolist()
        if is_fp8_fnuz():
            k_scale *= 2
            v_scale *= 2
    elif layer.k_scale <= 0.0 and layer.v_scale <= 0.0:
        kv_cache_scheme = getattr(self.quant_config, "kv_cache_scheme", None)
        require_checkpoint_scales = (
            bool(
                isinstance(kv_cache_scheme, dict)
                and kv_cache_scheme.get("require_checkpoint_scales", False)
            )
            or envs.SGLANG_REQUIRE_KV_CACHE_SCALES.get()
        )
        if require_checkpoint_scales:
            raise RuntimeError(
                "FP8 KV cache requires calibrated k_scale/v_scale tensors, "
                "but neither scale was loaded for attention layer "
                f"{getattr(layer, 'layer_id', '<unknown>')}. Refusing to "
                "fall back to the implicit scale=1.0 path."
            )
        # If no scales were loaded (both scales are invalid non-positive
        # values), use the default value of 1.0
        k_scale = 1.0
        v_scale = 1.0
    else:
        # If we find a single kv_scale in the checkpoint, we remap
        # kv_scale to k_scale during weight loading, and duplicate
        # k_scale to v_scale here
        assert layer.k_scale > 0.0
        scale_to_duplicate = max(layer.k_scale, layer.v_scale)
        k_scale = scale_to_duplicate.to("cpu").tolist()
        v_scale = scale_to_duplicate.to("cpu").tolist()
        if is_fp8_fnuz():
            k_scale *= 2
            v_scale *= 2

    if not isinstance(k_scale, float) or not isinstance(v_scale, float):
        raise ValueError("Only support per-tensor scaling factor for fp8 KV cache")

    # These are used in the final Attention.forward()
    layer.k_scale.copy_(k_scale)
    layer.v_scale.copy_(v_scale)
    layer.k_scale_float = k_scale
    layer.v_scale_float = v_scale


def around_process_weights_after_loading(original, *args, **kwargs):
    if not is_hcu():
        return original(*args, **kwargs)
    return process_weights_after_loading(*args, **kwargs)


def register_hooks() -> None:
    from sglang.srt.plugins.hook_registry import HookRegistry, HookType

    HookRegistry.register(
        "sglang.srt.layers.quantization.kv_cache.BaseKVCacheMethod.process_weights_after_loading",
        around_process_weights_after_loading,
        HookType.AROUND,
    )
