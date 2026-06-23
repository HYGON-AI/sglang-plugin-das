"""TODO: migrate DCU code into sglang_dcu_plugin/models/qwen/qwen3_5.py."""

from __future__ import annotations

import torch
from lightop import gammarms_rotary_embedding_fuse_with_kv_store

import logging
logger = logging.getLogger(__name__)

def replacement_self_attention(self, positions, hidden_states, forward_batch):
    """Replacement for Qwen3_5AttentionDecoderLayer.self_attention()."""
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

    # Put custom self_attention logic here. The code below mirrors the current
    # in-tree implementation so the plugin can be enabled before customization.
    cos_sin_cache = self.rotary_emb.cos_sin_cache
    if (cos_sin_cache.device != q.device
            or cos_sin_cache.dtype != q.dtype):
        cos_sin_cache = cos_sin_cache.to(q.device,
                                        dtype=q.dtype,
                                        non_blocking=True)
        # Persist the converted cache so we don't re-copy/re-allocate
        # on every forward when the original buffer starts on CPU.
        self.rotary_emb.cos_sin_cache = cos_sin_cache
    k_buffer, v_buffer = forward_batch.token_to_kv_pool.get_kv_buffer(self.layer_id)

    q, k, v = gammarms_rotary_embedding_fuse_with_kv_store(
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

    attn_output = self.attn(q, k, v, forward_batch)

    if gate is not None:
        attn_output = attn_output * torch.sigmoid(gate)

    output, _ = self.o_proj(attn_output)
    return output