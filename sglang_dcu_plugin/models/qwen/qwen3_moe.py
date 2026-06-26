"""Qwen3-MoE model hooks for DCU fused RMSNorm + RoPE KV-store."""

import logging

from sglang.srt.plugins.hook_registry import HookRegistry, HookType
from sglang_dcu_plugin.utils.module_patch import get_original_attr

logger = logging.getLogger(__name__)

_TARGET = "sglang.srt.models.qwen3_moe.Qwen3MoeAttention.apply_qk_norm_rope"
_MODULE = "sglang.srt.models.qwen3_moe"


def register() -> None:
    logger.info("[sglang_dcu_plugin] register REPLACE hook: %s", _TARGET)
    HookRegistry.register(_TARGET, replacement_apply_qk_norm_rope, HookType.REPLACE)


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


def replacement_apply_qk_norm_rope(self, qkv, positions, forward_batch):
    try:
        from lightop import rms_rotary_embedding_fuse_with_kv_store
    except Exception as exc:
        logger.warning(
            "[sglang_dcu_plugin] qwen3_moe fused RMS rotary is unavailable, fallback: %s",
            exc,
        )
        original = get_original_attr(_MODULE, "Qwen3MoeAttention.apply_qk_norm_rope")
        return original(self, qkv, positions, forward_batch)

    q, k, v = qkv.split([self.q_size, self.kv_size, self.kv_size], dim=-1)
    q, k, v = _fused_rms_rotary_kv_store(
        rms_rotary_embedding_fuse_with_kv_store,
        self,
        positions,
        q,
        k,
        v,
        forward_batch,
    )
    self._used_fused_qk_norm_rope_last_call = False
    return q, k, v
