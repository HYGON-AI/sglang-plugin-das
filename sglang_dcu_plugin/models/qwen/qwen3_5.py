"""Qwen3.5 model hooks for DCU fused RMSNorm + RoPE KV-store."""

import logging

import torch

from sglang.srt.plugins.hook_registry import HookRegistry, HookType
from sglang_dcu_plugin.utils.module_patch import get_original_attr

logger = logging.getLogger(__name__)

_TARGET = "sglang.srt.models.qwen3_5.Qwen3_5AttentionDecoderLayer.self_attention"
_MODULE = "sglang.srt.models.qwen3_5"


def register() -> None:
    logger.info("[sglang_dcu_plugin] register REPLACE hook: %s", _TARGET)
    HookRegistry.register(_TARGET, replacement_self_attention, HookType.REPLACE)


def _fused_rms_rotary_kv_store(
    fused_op, self, positions, q, k, v, forward_batch
):
    cos_sin_cache = self.rotary_emb.cos_sin_cache
    if cos_sin_cache.device != q.device or cos_sin_cache.dtype != q.dtype:
        cos_sin_cache = cos_sin_cache.to(q.device, dtype=q.dtype, non_blocking=True)
        self.rotary_emb.cos_sin_cache = cos_sin_cache
    k_buffer, v_buffer = forward_batch.token_to_kv_pool.get_kv_buffer(self.layer_id)

    return fused_op(
        positions,
        q,
        k,
        v,
        cos_sin_cache,
        self.head_dim,
        self.page_size,
        k_buffer,
        v_buffer,
        forward_batch.out_cache_loc,
        is_neox=True,
        weight_q=self.q_norm.weight,
        weight_k=self.k_norm.weight,
        output_dtype=self.kv_cache_dtype,
        residual_q=None,
        residual_k=None,
        k_scale=None,
        v_scale=None,
        epsilon=self.q_norm.variance_epsilon,
    )


def replacement_self_attention(self, positions, hidden_states, forward_batch):
    try:
        from lightop import gammarms_rotary_embedding_fuse_with_kv_store
    except Exception as exc:
        logger.warning(
            "[sglang_dcu_plugin] qwen3_5 fused RMS rotary is unavailable, fallback: %s",
            exc,
        )
        original = get_original_attr(
            _MODULE, "Qwen3_5AttentionDecoderLayer.self_attention"
        )
        return original(self, positions, hidden_states, forward_batch)

    qkv, _ = self.qkv_proj(hidden_states)

    if self.attn_output_gate:
        q_gate, k, v = qkv.split(
            [self.q_size * 2, self.kv_size, self.kv_size], dim=-1
        )
        orig_shape = q_gate.shape[:-1]
        q_gate = q_gate.view(*orig_shape, self.num_heads, -1)
        q, gate = torch.chunk(q_gate, 2, dim=-1)
        q = q.reshape(*orig_shape, -1)
        gate = gate.reshape(*orig_shape, -1)
    else:
        q, k, v = qkv.split([self.q_size, self.kv_size, self.kv_size], dim=-1)
        gate = None

    q, k, v = _fused_rms_rotary_kv_store(
        gammarms_rotary_embedding_fuse_with_kv_store,
        self,
        positions,
        q,
        k,
        v,
        forward_batch,
    )
    attn_output = self.attn(q, k, v, forward_batch)

    if gate is not None:
        attn_output = attn_output * torch.sigmoid(gate)

    output, _ = self.o_proj(attn_output)
    return output
