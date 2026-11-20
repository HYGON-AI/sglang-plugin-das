# Copyright (c) 2026 Hygon Information Technology Co., Ltd.
# Licensed under the Apache License, Version 2.0; see LICENSE.
"""HCU rotary fallback consumer: typed cache view and bounded launches.

Head splitting derives from pinned HCU 5f67ce01c3 rotary_embedding/base.py.
"""
from __future__ import annotations
import hashlib
import sys
from functools import lru_cache
from pathlib import Path
from typing import Optional
import torch
from sglang.srt.environ import envs
from hcu_sglang.utils.detection import is_hcu

_HCU_ROPE_MAX_THREADS = 256
_GLM53_ROPE_AOT_SHA256 = "07752f007a0ab720538514eb084e82297136c1d9ec9ac87353b6471766804e57"


def _glm53_fp8_model(config):
    if (
        tuple(getattr(config, "architectures", None) or ())
        not in (("GlmMoeDsaForCausalLM",), ("GlmMoeDsaForCausalLMNextN",))
        or getattr(config, "model_type", None) != "glm_moe_dsa"
        or getattr(config, "num_hidden_layers", None) != 78
        or getattr(config, "hidden_size", None) != 6144
        or getattr(config, "n_routed_experts", None) != 256
        or getattr(config, "num_experts_per_tok", None) != 8
        or getattr(config, "moe_intermediate_size", None) != 2048
        or getattr(config, "num_attention_heads", None) != 64
        or getattr(config, "q_lora_rank", None) != 2048
        or getattr(config, "kv_lora_rank", None) != 512
        or getattr(config, "qk_rope_head_dim", None) != 64
        or getattr(config, "rope_interleave", None) is not True
    ):
        return False
    quant = getattr(config, "quantization_config", None)
    if quant is not None and not isinstance(quant, dict):
        to_dict = getattr(quant, "to_dict", None)
        quant = to_dict() if callable(to_dict) else None
    if not isinstance(quant, dict):
        return False
    groups = quant.get("config_groups")
    if (
        quant.get("quant_method") != "compressed-tensors"
        or quant.get("format") != "float-quantized"
        or not isinstance(groups, dict)
        or len(groups) != 1
    ):
        return False
    group = next(iter(groups.values()))
    if not isinstance(group, dict) or group.get("targets") != ["Linear"]:
        return False
    weight = group.get("weights") or {}
    activation = group.get("input_activations") or {}
    if not isinstance(weight, dict) or not isinstance(activation, dict):
        return False
    return (
        weight.get("type") == "float"
        and weight.get("num_bits") == 8
        and weight.get("strategy") == "channel"
        and weight.get("dynamic") is False
        and weight.get("block_structure") is None
        and activation.get("type") == "float"
        and activation.get("num_bits") == 8
        and activation.get("strategy") == "token"
        and activation.get("dynamic") is True
    )


@lru_cache(maxsize=1)
def _glm53_aot_rotary():
    module = sys.modules.get("sgl_kernel.common_ops")
    path = Path(getattr(module, "__file__", ""))
    if (
        sys.version_info[:2] != (3, 10)
        or torch.__version__ != "2.11.0"
        or torch.version.hip != "6.3.26113"
        or not torch.compiled_with_cxx11_abi()
        or module is None
        or not path.is_file()
        or hashlib.sha256(path.read_bytes()).hexdigest() != _GLM53_ROPE_AOT_SHA256
        or not torch._C._dispatch_has_kernel_for_dispatch_key(
            "sgl_kernel::rotary_embedding", "CUDA"
        )
    ):
        raise RuntimeError("GLM-5.3 FP8 RoPE AOT provider identity or ABI mismatch")
    return torch.ops.sgl_kernel.rotary_embedding.default


def _rope_head_layout(tensor, tokens):
    if tensor.ndim == 2 and tensor.shape[0] == tokens and tensor.shape[1] % 64 == 0:
        return tensor.shape[1] // 64, 64
    if tensor.ndim == 3 and tensor.shape[0] == tokens and tensor.shape[2] == 64:
        return tensor.shape[1], tensor.stride(1)
    return 0, 0


def _use_glm53_fp8_aot_rope(layer, positions, query, key, offsets, fused):
    if (
        getattr(layer, "_hcu_glm53_fp8_aot_fallback", None) is None
        or offsets is not None
        or fused is not None
        or key is None
        or positions.ndim != 1
    ):
        return False
    cache = layer.cos_sin_cache
    q_heads, q_head_stride = _rope_head_layout(query, positions.numel())
    k_heads, k_head_stride = _rope_head_layout(key, positions.numel())
    return (
        positions.dtype == torch.int64
        and positions.is_contiguous()
        and layer.head_size == layer.rotary_dim == 64
        and q_heads > 0
        and k_heads > 0
        and q_heads % k_heads == 0
        and (k_heads == 1 or q_head_stride == k_head_stride)
        and query.stride(-1) == key.stride(-1) == 1
        and query.dtype == key.dtype == cache.dtype == torch.bfloat16
        and cache.ndim == 2
        and cache.shape[1] == 64
        and cache.is_contiguous()
        and positions.device == query.device == key.device == cache.device
    )

def _max_q_heads_within_launch_bound(rot_dim: int) -> int:
    if rot_dim <= 0:
        return 1
    heads = max(1, (_HCU_ROPE_MAX_THREADS * 2) // rot_dim)
    while heads > 1 and heads * rot_dim // 2 > _HCU_ROPE_MAX_THREADS:
        heads -= 1
    return heads


def _launch_rotary_embedding(
    fn,
    positions: torch.Tensor,
    query: torch.Tensor,
    key: Optional[torch.Tensor],
    head_size: int,
    cos_sin_cache: torch.Tensor,
    is_neox: bool,
) -> None:
    """Launch sgl_kernel.rotary_embedding without exceeding the HCU block limit.

    The kernel strides with blockDim.x, so splitting query heads across several
    launches is numerically the same as one launch. Key is applied once.
    """
    num_tokens = int(positions.numel())
    if num_tokens == 0:
        return
    rot_dim = int(cos_sin_cache.shape[-1])
    nq = query.numel() // num_tokens // head_size
    nk = 0 if key is None else key.numel() // num_tokens // head_size
    if nq * rot_dim // 2 <= _HCU_ROPE_MAX_THREADS:
        fn(positions, query, key, head_size, cos_sin_cache, is_neox)
        return

    max_q_heads = _max_q_heads_within_launch_bound(rot_dim)
    if max_q_heads * rot_dim // 2 > _HCU_ROPE_MAX_THREADS:
        fn(positions, query, key, head_size, cos_sin_cache, is_neox)
        return

    pos = positions.reshape(-1)
    q = query.reshape(num_tokens, nq, head_size)
    k = None if key is None else key.reshape(num_tokens, nk, head_size)

    def _launch(q_slice, k_slice):
        fn(pos, q_slice, k_slice, head_size, cos_sin_cache, is_neox)

    if k is None or nk == nq:
        for start in range(0, nq, max_q_heads):
            end = min(start + max_q_heads, nq)
            _launch(q[:, start:end], None if k is None else k[:, start:end])
        return

    if nk == 1:
        _launch(q[:, :max_q_heads], k)
        for start in range(max_q_heads, nq, max_q_heads):
            end = min(start + max_q_heads, nq)
            _launch(q[:, start:end], None)
        return

    group = nq // nk
    q_step = (max_q_heads // group) * group
    if group == 0 or q_step < group or nq % nk != 0:
        fn(positions, query, key, head_size, cos_sin_cache, is_neox)
        return
    for q_start in range(0, nq, q_step):
        q_end = min(q_start + q_step, nq)
        _launch(q[:, q_start:q_end], k[:, q_start // group : q_end // group])


class HcuRotaryFallback:
    def __init__(self, original, *, split_heads=True):
        self.original = original
        self.split_heads = split_heads
        self._cache_memo = None

    def __call__(self, positions, query, key, head_size, cos_sin_cache, is_neox):
        cache = cos_sin_cache
        if cache.device != query.device or cache.dtype != query.dtype:
            memo = self._cache_memo
            if (memo is None or memo[0] is not cache or
                    memo[1].device != query.device or memo[1].dtype != query.dtype):
                memo = (cache, cache.to(device=query.device, dtype=query.dtype))
                self._cache_memo = memo
            cache = memo[1]
        if self.split_heads:
            _launch_rotary_embedding(self.original, positions, query, key,
                                     head_size, cache, is_neox)
        else:
            self.original(positions, query, key, head_size, cache, is_neox)


def configure_glm_rope_cache(layer, config):
    layer.cos_sin_cache = layer.cos_sin_cache.to(dtype=layer.dtype)
    if (
        _glm53_fp8_model(config)
        and layer.dtype == torch.bfloat16
        and getattr(layer, "_hcu_lightop_rope", None) is None
    ):
        layer._hcu_glm53_fp8_aot_fallback = HcuRotaryFallback(_glm53_aot_rotary())


def after_rotary_init(result, layer, *args, **kwargs):
    if not is_hcu():
        return result
    if envs.SGLANG_HCU_USE_LIGHTOP_ROPE.get() and layer.dtype == torch.bfloat16:
        from lightop import rotary_embedding_fuse

        layer._hcu_lightop_rope = HcuRotaryFallback(
            rotary_embedding_fuse, split_heads=False
        )
    if getattr(layer, 'use_fallback_kernel', False):
        current = layer.fallback_rotary_embedding
        if not isinstance(current, HcuRotaryFallback):
            layer.fallback_rotary_embedding = HcuRotaryFallback(current)
    return result


def forward_cuda(original_fn, layer, positions, query, key, offsets=None,
                 fused_set_kv_buffer_arg=None):
    native = getattr(layer, '_hcu_lightop_rope', None)
    if native is None and _use_glm53_fp8_aot_rope(
        layer, positions, query, key, offsets, fused_set_kv_buffer_arg
    ):
        native = layer._hcu_glm53_fp8_aot_fallback
    if native is None or fused_set_kv_buffer_arg is not None:
        return original_fn(
            layer, positions, query, key, offsets=offsets,
            fused_set_kv_buffer_arg=fused_set_kv_buffer_arg,
        )
    if offsets is not None:
        positions = positions + offsets
    native(positions, query, key, layer.head_size, layer.cos_sin_cache,
           layer.is_neox_style)
    return query, key


def register_hooks():
    from sglang.srt.plugins.hook_registry import HookRegistry, HookType
    HookRegistry.register(
        'sglang.srt.layers.rotary_embedding.base.RotaryEmbedding.__init__',
        after_rotary_init, HookType.AFTER,
    )
    HookRegistry.register(
        'sglang.srt.layers.rotary_embedding.base.RotaryEmbedding.forward_cuda',
        forward_cuda, HookType.AROUND,
    )
