"""Qwen3.5 model hooks for HCU fused RMSNorm + RoPE KV-store."""

import logging

import torch

from sglang.srt.plugins.hook_registry import HookRegistry, HookType
from hcu_sglang.models.fused_rotary import (
    fused_rms_rotary_kv_store,
    use_fused_rms_rotary,
)
from hcu_sglang.layers import qwen_rms_int8
from hcu_sglang.utils.detection import is_hcu
from hcu_sglang.utils.module_patch import get_original_attr

logger = logging.getLogger(__name__)

_TARGET = "sglang.srt.models.qwen3_5.Qwen3_5AttentionDecoderLayer.self_attention"
_MODULE = "sglang.srt.models.qwen3_5"
_GDN_INIT_TARGET = _MODULE + ".Qwen3_5GatedDeltaNet.__init__"


def register() -> None:
    logger.info("[hcu_sglang] register AROUND hook: %s", _TARGET)
    HookRegistry.register(_TARGET, replacement_self_attention, HookType.AROUND)
    HookRegistry.register(_GDN_INIT_TARGET, around_gdn_init, HookType.AROUND)
    HookRegistry.register(
        _MODULE + ".Qwen3_5AttentionDecoderLayer.forward_prepare_cuda_fused",
        around_prepare_cuda_fused,
        HookType.AROUND,
    )
    if qwen_rms_int8.enabled:
        for target in (
            "sglang.srt.models.qwen3_5.Qwen3_5AttentionDecoderLayer.forward",
            "sglang.srt.models.qwen3_5.Qwen3_5LinearDecoderLayer.forward",
        ):
            HookRegistry.register(
                target,
                qwen_rms_int8.around_decoder_forward,
                HookType.AROUND,
            )


def replacement_self_attention(original, self, positions, hidden_states, forward_batch):
    return original(self, positions, hidden_states, forward_batch)


def around_gdn_init(original, self, *args, **kwargs):
    result = original(self, *args, **kwargs)
    if is_hcu():
        self.alt_stream = None
    return result


def around_prepare_cuda_fused(original, self, positions, hidden_states):
    if not (
        isinstance(hidden_states, torch.Tensor)
        and hidden_states.shape[0] == 0
        and is_hcu()
    ):
        return original(self, positions, hidden_states)
    from sglang.srt.models import qwen3_5 as native

    qkv, _ = self.qkv_proj(hidden_states)
    if self.attn_output_gate:
        q_gate, k, v = qkv.split(
            [self.q_size * 2, self.kv_size, self.kv_size], dim=-1
        )
    else:
        q_gate, k, v = qkv.split([self.q_size, self.kv_size, self.kv_size], dim=-1)
    q_out, k_out, gate_out = native.fused_qk_gemma_rmsnorm_rope_gate(
        q_gate,
        k,
        self.q_norm.weight.data,
        self.k_norm.weight.data,
        self.rotary_emb.cos_sin_cache,
        positions,
        self.q_norm.variance_epsilon,
        self.num_heads,
        self.num_kv_heads,
        self.head_dim,
        self.rotary_emb.rotary_dim,
        has_gate=self.attn_output_gate,
        mrope_axis_map=(self.rotary_emb.axis_map if positions.dim() == 2 else None),
    )
    return (
        q_out.new_empty((0, self.q_size)),
        k_out.new_empty((0, self.kv_size)),
        v,
        gate_out.new_empty((0, self.q_size)) if gate_out is not None else None,
    )
