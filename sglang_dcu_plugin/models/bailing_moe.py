"""Bailing-MoE model hooks for DCU fused RMSNorm + RoPE KV-store."""

import logging

from sglang.srt.plugins.hook_registry import HookRegistry, HookType
from sglang_dcu_plugin.utils.module_patch import get_original_attr

logger = logging.getLogger(__name__)

_TARGET = "sglang.srt.models.bailing_moe.BailingMoEAttention.forward"
_MODULE = "sglang.srt.models.bailing_moe"


def register() -> None:
    logger.info("[sglang_dcu_plugin] register REPLACE hook: %s", _TARGET)
    HookRegistry.register(_TARGET, replacement_forward, HookType.REPLACE)


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
        weight_q=self.query_layernorm.weight,
        weight_k=self.key_layernorm.weight,
        output_dtype=self.kv_cache_dtype,
        residual_q=None,
        residual_k=None,
        k_scale=None,
        v_scale=None,
        epsilon=self.query_layernorm.variance_epsilon,
    )


def replacement_forward(self, positions, hidden_states, forward_batch):
    try:
        from lightop import rms_rotary_embedding_fuse_with_kv_store
    except Exception as exc:
        logger.warning(
            "[sglang_dcu_plugin] bailing_moe fused RMS rotary is unavailable, fallback: %s",
            exc,
        )
        original = get_original_attr(_MODULE, "BailingMoEAttention.forward")
        return original(self, positions, hidden_states, forward_batch)

    from sglang.srt.models.utils import (
        create_fused_set_kv_buffer_arg,
        enable_fused_set_kv_buffer,
    )

    if isinstance(hidden_states, tuple):
        if hidden_states[0].shape[0] == 0:
            return hidden_states
    else:
        if hidden_states.shape[0] == 0:
            return hidden_states

    qkv, _ = self.query_key_value(hidden_states)
    q, k, v = qkv.split([self.q_size, self.kv_size, self.kv_size], dim=-1)
    can_fuse_set_kv = (
        self.head_dim == self.rotary_emb.rotary_dim
        and enable_fused_set_kv_buffer(forward_batch)
    )

    if self.use_qk_norm:
        q, k, v = _fused_rms_rotary_kv_store(
            rms_rotary_embedding_fuse_with_kv_store,
            self,
            positions,
            q,
            k,
            v,
            forward_batch,
        )
    else:
        q, k = self.rotary_emb(
            positions,
            q,
            k,
            fused_set_kv_buffer_arg=(
                create_fused_set_kv_buffer_arg(
                    value=v,
                    layer=self.attn,
                    forward_batch=forward_batch,
                )
                if can_fuse_set_kv
                else None
            ),
        )

    context_layer = self.attn(
        q,
        k,
        v,
        forward_batch,
        save_kv_cache=not can_fuse_set_kv,
    )
    attn_output, _ = self.dense(context_layer)
    return attn_output
