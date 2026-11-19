import os

import torch

from . import minimax_m31_flashmla as native

_FP8 = torch.float8_e4m3fn
_HEAD_COUNTS = (64, 4, 4, 128)


def _eligible_attention(attn, mode):
    return not (
        os.getenv("SGLANG_M3_OPTIMIZE_SPARSE_ATTN_PREP", "1") != "1"
        or os.getenv("SGLANG_MINIMAX_INDEX_FP8") != "dot"
        or not torch.version.hip
        or not (mode.is_decode() or mode.is_target_verify())
        or not attn.is_sparse_attention_layer
        or not attn.disable_index_value
        or not attn.use_gemma_norm
        or attn.qk_norm_type != "per_head"
        or attn.index_rotary_emb is not attn.rotary_emb
        or (attn.num_heads, attn.num_kv_heads, attn.num_idx_heads, attn.head_dim) != _HEAD_COUNTS
        or attn.idx_head_dim != 128
        or not 0 < attn.rotary_dim <= 128
        or attn.rotary_dim % 2
        or os.getenv("SGLANG_MINIMAX_KV_CALIBRATION_DIR")
        or os.getenv("SGLANG_USE_LIGHTOP_MINIMAX_QK_ROPE") == "1"
    )


def _eligible_sparse_backend(attn, mode):
    from sglang.srt.model_executor.forward_context import get_forward_context, has_forward_context
    from sglang.srt.runtime_context import get_memory, get_model, get_parallel

    if (
        not has_forward_context()
        or get_model().quantization != "slimquant_w4a8_marlin"
        or get_memory().enable_hisparse
        or get_parallel().dcp_size != 1
    ):
        return None
    sparse = native.sparse_backend_for_mode(get_forward_context().attn_backend, mode)
    layer = attn.attn
    if (
        sparse is None
        or sparse.is_npu
        or sparse._loc_mapping is not None
        or sparse.indexer_cp is not None
        or sparse.hisparse_coordinator is not None
        or sparse.score_type != "max"
        or sparse.block_size_k != 128
        or sparse.page_size != 128
        or sparse.topk_blocks != 16
        or sparse.fp8_attn_gemm
        or layer.layer_id not in sparse.disable_value_layer_ids
        or any(s not in (None, 1.0) for s in (
            layer.q_scale_float, layer.k_scale_float, layer.v_scale_float,
            layer.idx_q_scale_float, layer.idx_k_scale_float,
        ))
        or not native.flash_mla_gfx938_available()
        or not native.flash_mla_stage1_fp8_available()
        or not native.flash_mla_stage1_graph_available()
    ):
        return None
    return sparse


def _prepare_buffers(attn, inputs, positions, batch):
    mode = batch.forward_mode
    if not _eligible_attention(attn, mode):
        return None

    q = inputs[0]
    rows = q.shape[0]
    loc = batch.out_cache_loc
    cos_sin = attn.rotary_emb.cos_sin_cache
    norms = (attn.q_norm, attn.k_norm, attn.index_q_norm, attn.index_k_norm)
    if (
        not rows
        or not q.is_cuda
        or any(x.shape != (rows, width) or x.dtype != torch.bfloat16 or x.device != q.device
               for x, width in zip(inputs, (8192, 512, 512, 512, 128)))
        or not isinstance(loc, torch.Tensor)
        or any(
            x.shape != (rows,)
            or not x.is_contiguous()
            or x.device != q.device
            or x.dtype not in (torch.int32, torch.int64)
            for x in (positions, loc)
        )
        or cos_sin.device != q.device
        or not cos_sin.is_contiguous()
        or any(
            n.variance_epsilon != norms[0].variance_epsilon
            or n.weight.shape != (128,)
            or n.weight.dtype != norms[0].weight.dtype
            or n.weight.device != q.device
            or not n.weight.is_contiguous()
            for n in norms
        )
    ):
        return None

    sparse = _eligible_sparse_backend(attn, mode)
    if sparse is None:
        return None
    layer = attn.attn
    pool = sparse.kv_pool
    caches = (*pool.get_kv_buffer(layer.layer_id), pool.get_index_k_buffer(layer.layer_id))
    if any(
        x.dtype != _FP8
        or x.device != q.device
        or x.shape != (caches[0].shape[0], heads, 128)
        or not x.is_contiguous()
        for x, heads in zip(caches, (4, 4, 1))
    ):
        return None
    return caches


def _prepare(attn, inputs, positions, batch):
    caches = _prepare_buffers(attn, inputs, positions, batch)
    if caches is None:
        return None

    from ...kernels.minimax_sparse_fp8_prepare import sparse_qk_index_gemma_rmsnorm_rope_cache

    result = sparse_qk_index_gemma_rmsnorm_rope_cache(
        *inputs, *caches, batch.out_cache_loc,
        attn.q_norm.weight, attn.k_norm.weight,
        attn.index_q_norm.weight, attn.index_k_norm.weight,
        positions, attn.rotary_emb.cos_sin_cache, attn.q_norm.variance_epsilon,
        attn.head_dim, attn.rotary_dim, attn.rotary_emb.is_neox_style,
        fp8_cache=True,
    )
    attn._mark_sparse_kv_cached_by_fusion(batch, attn.attn.layer_id)
    return result


def _prepare_packed(original, attn, packed_qkv, positions, forward_batch):
    if packed_qkv.ndim == 2 and packed_qkv.shape[1] == 9856 and attn.disable_index_value:
        q, k, v, idx_q, idx_k = packed_qkv.split((8192, 512, 512, 512, 128), dim=1)
        result = _prepare(attn, (q, k, v, idx_q, idx_k), positions, forward_batch)
        if result is not None:
            q_out, k_out, idx_q_out, idx_k_out = result
            return q_out, k_out, v, idx_q_out, idx_k_out, None
    return original(attn, packed_qkv, positions, forward_batch)


def _prepare_split(original, attn, positions, q, k, v, idx_q, idx_k, idx_v, forward_batch):
    if idx_v is None:
        result = _prepare(attn, (q, k, v, idx_q, idx_k), positions, forward_batch)
        if result is not None:
            return result
    return original(attn, positions, q, k, v, idx_q, idx_k, idx_v, forward_batch)


def _prepare_unpacked(original, attn, positions, hidden_states, forward_batch):
    mode = forward_batch.forward_mode
    if (
        not attn._use_fused_qknorm_rope
        or attn._fused_qkv_index is not None
        or not _eligible_attention(attn, mode)
        or _eligible_sparse_backend(attn, mode) is None
    ):
        return original(attn, positions, hidden_states, forward_batch)

    attn._use_fused_qknorm_rope = False
    try:
        return original(attn, positions, hidden_states, forward_batch)
    finally:
        attn._use_fused_qknorm_rope = True


def register_hooks():
    from sglang.srt.plugins.hook_registry import HookRegistry, HookType

    for name, hook in (
        ("forward_prepare", _prepare_unpacked),
        ("_aiter_sparse_qknorm_cache", _prepare_packed),
        ("_sparse_qk_index_norm_rope_cache", _prepare_split),
    ):
        HookRegistry.register(
            "sglang.srt.models.minimax_m3.MiniMaxM3Attention." + name,
            hook,
            HookType.AROUND,
        )
