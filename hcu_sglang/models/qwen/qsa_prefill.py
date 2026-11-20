from __future__ import annotations

import logging
import msgspec
import torch
import torch.nn.functional as F
import triton

from sglang.srt.layers.attention import qwen_sparse_attn_backend as N
from sglang.srt.layers.attention.qsa import sparse_attn as S
from sglang.srt.runtime_context import get_parallel
from sglang.srt.environ import envs

logger = logging.getLogger(__name__)


class SparsePrefillPlan(msgspec.Struct, frozen=True):
    cu_q: object
    cu_k: object
    lengths: tuple
    extend_lengths: tuple
    impl: str = 'triton'
    page_table: object = None
    unit_scale: object = None


class SparseMetadata(N.QwenSparseAttnMetadata):
    hcu_sparse_prefill: object = None


def prepare_metadata(result, backend, batch):
    if (not batch.forward_mode.is_extend_without_speculative()
        or not batch.positions.is_cuda
        or get_parallel().attn_cp_size != 1
        or batch.seq_lens_cpu is None
        or batch.extend_seq_lens_cpu is None
        ):
        return result
    lengths = tuple(int(n) for n in batch.seq_lens_cpu)
    extend = tuple(int(n) for n in batch.extend_seq_lens_cpu)
    if not lengths:
        return result
    impl = envs.SGLANG_QWEN4_PREFILL_QSA_IMPL.get()
    if impl not in ('triton', 'native-packed', 'native-paged'):
        raise ValueError(f'Unsupported Qwen4 prefill QSA implementation: {impl}')
    page_table = unit_scale = None
    pool = backend.token_to_kv_pool
    storage = getattr(pool, 'full_kv_pool', pool)
    page = getattr(storage, 'page_size', 1)
    if (impl == 'native-paged' and page > 1
        and not getattr(storage, 'is_quantized_kv_cache', True)
        and not getattr(storage, 'use_hnd', True)):
        page_table = (result.token_slot_table[:, ::page] // page).to(torch.int32).contiguous()
        unit_scale = torch.ones(1, dtype=torch.float32, device=batch.positions.device)
    plan = SparsePrefillPlan(
        F.pad(batch.extend_seq_lens.to(torch.int32).cumsum(0, dtype=torch.int32), (1, 0)),
        F.pad(result.sequence_lengths.cumsum(0, dtype=torch.int32), (1, 0)),
        lengths, extend, impl, page_table, unit_scale,
    )
    fields = msgspec.structs.asdict(result)
    fields['hcu_sparse_prefill'] = plan
    return SparseMetadata(**fields)


def native_paged_prefill(backend, layer, q, indices, meta, plan):
    if (plan.page_table is None or indices.ndim != 2
        or any(length - extend < indices.shape[-1]
               for length, extend in zip(plan.lengths, plan.extend_lengths))
        or layer.scaling != layer.head_dim ** -.5):
        return None
    pool = backend.token_to_kv_pool
    storage = getattr(pool, 'full_kv_pool', pool)
    k = pool.get_key_buffer(layer.layer_id)
    v = pool.get_value_buffer(layer.layer_id)
    page = storage.page_size
    if (v.shape != k.shape or k.shape[-1] != 256
        or k.dtype not in (torch.float8_e5m2, torch.float8_e4m3fn)
        or v.dtype != k.dtype):
        return None
    if k.ndim == 4:
        from hcu_sglang.mem_cache.hcu_mha_pool import HCUMHATokenToKVPool
        if not isinstance(storage, HCUMHATokenToKVPool) or k.shape[2] != page:
            return None
        paged_k, paged_v = k.permute(0, 2, 1, 3), v.permute(0, 2, 1, 3)
    elif k.ndim == 3 and k.shape[0] % page == 0:
        paged_k, paged_v = k.view(-1, page, *k.shape[1:]), v.view(-1, page, *v.shape[1:])
    else:
        return None
    if q.shape[1] != paged_k.shape[2] * 12:
        return None
    from flash_attn.flash_attn_interface import sparse_gqa_paged_attn_fp8_func
    output = sparse_gqa_paged_attn_fp8_func(
        q, paged_k, paged_v,
        indices, plan.page_table, meta.token_to_batch_idx.to(torch.int32).contiguous(),
        k_scale=plan.unit_scale, v_scale=plan.unit_scale,
    )
    block_n = S._get_best_config(q.shape[0])[0]
    empty = (indices[:, :block_n] < 0).all(dim=-1)
    output.masked_fill_(empty[:, None, None], float('nan'))
    if not getattr(backend, '_hcu_native_paged_logged', False):
        logger.info('Qwen4 QSA native paged FP8 prefill active: Q=%s, KV=%s, page=%s',
                    tuple(q.shape), tuple(k.shape), page)
        backend._hcu_native_paged_logged = True
    return output


def sparse_chunk_prefill(q, k, v, indices, cu_q, cu_k, kv_lens, scale, max_q):
    k, v = k.contiguous(), v.contiguous()
    total_q, num_q_heads, head_dim = q.shape
    num_kv_heads = k.shape[1]
    group_size = num_q_heads // num_kv_heads
    block_n, warps, stages = S._get_best_config(total_q)
    out = torch.empty_like(q)
    S._sparse_gqa_chunk_prefill[(max_q, (cu_q.shape[0] - 1) * num_kv_heads)](
        q, k, v, out, indices, cu_q, cu_k, kv_lens, scale, indices.shape[-1],
        *q.stride(), *k.stride(), *v.stride(), *out.stride(),
        indices.stride(0), indices.stride(1) if indices.ndim == 3 else 0,
        indices.stride(2) if indices.ndim == 3 else indices.stride(1),
        NUM_KV_HEADS=num_kv_heads, GROUP_SIZE=group_size,
        BLOCK_M=max(16, triton.next_power_of_2(group_size)), BLOCK_N=block_n,
        HEAD_DIM=head_dim, num_warps=warps, num_stages=stages,
    )
    return out


def around_forward_extend(original, backend, q, k, v, layer, forward_batch,
                          save_kv_cache=True, topk_indices=None, **kwargs):
    meta = backend.forward_metadata
    plan = getattr(meta, 'hcu_sparse_prefill', None)
    if (plan is None or topk_indices is None or not q.is_cuda
        or getattr(forward_batch, 'attn_cp_metadata', None) is not None):
        return original(backend, q, k, v, layer, forward_batch,
                        save_kv_cache=save_kv_cache, topk_indices=topk_indices, **kwargs)
    if save_kv_cache:
        fused = backend._try_fused_kv_attention(q, k, v, layer, forward_batch, topk_indices)
        if fused is not None:
            return fused
        backend.token_to_kv_pool.set_kv_buffer(layer, forward_batch.out_cache_loc, k, v)
    q = q.reshape(-1, layer.tp_q_head_num, layer.head_dim)
    num_rows, valid_rows = q.shape[0], topk_indices.shape[0]
    if valid_rows > num_rows:
        raise ValueError('QSA top-k rows exceed query rows')
    q = q[:valid_rows].contiguous()
    indices = topk_indices.to(torch.int32).contiguous()
    native = (plan.impl != 'triton' and q.dtype == torch.bfloat16
              and layer.head_dim == 256 and k is not None and k.dtype == q.dtype
              and 1 <= layer.tp_q_head_num // k.shape[1] <= 16)
    if native and valid_rows == 0:
        return backend._pad_extend_output(torch.empty_like(q), num_rows)
    if native and plan.impl == 'native-paged':
        output = native_paged_prefill(backend, layer, q, indices, meta, plan)
        if output is not None:
            return backend._pad_extend_output(output, num_rows)
    if native:
        from flash_attn.flash_attn_interface import sparse_gqa_attn_func
        if all(length == extend for length, extend in zip(plan.lengths, plan.extend_lengths)):
            packed_k, packed_v = k[:valid_rows], v[:valid_rows]
        else:
            pool = backend.token_to_kv_pool
            packed_k, packed_v = N.pack_qsa_prefill_kv(
                k=pool.get_key_buffer(layer.layer_id), v=pool.get_value_buffer(layer.layer_id),
                req_to_token=backend.req_to_token_pool.req_to_token,
                req_indices=forward_batch.req_pool_indices, cu_k=plan.cu_k,
                total_k=sum(plan.lengths), max_k=max(plan.lengths, default=1), output_dtype=q.dtype,
            )
        output = sparse_gqa_attn_func(
            q, packed_k, packed_v, indices, plan.cu_q, plan.cu_k,
            meta.sequence_lengths, scale=layer.scaling,
            max_seqlen_q=max(plan.extend_lengths, default=1),
        )
        block_n = S._get_best_config(q.shape[0])[0]
        empty = (indices[..., :block_n] < 0).all(dim=-1)
        if empty.ndim == 2:
            empty = empty.repeat_interleave(q.shape[1] // empty.shape[1], dim=1)
            output.masked_fill_(empty[:, :, None], float('nan'))
        else:
            output.masked_fill_(empty[:, None, None], float('nan'))
        if not getattr(backend, '_hcu_native_packed_logged', False):
            logger.info('Qwen4 QSA native packed BF16 prefill active: Q=%s, KV=%s',
                        tuple(q.shape), tuple(packed_k.shape))
            backend._hcu_native_packed_logged = True
        return backend._pad_extend_output(output, num_rows)
    if all(length == extend for length, extend in zip(plan.lengths, plan.extend_lengths)):
        output = N.sparse_gqa_fwd_interface_triton(
            q, k[:valid_rows].contiguous(), v[:valid_rows].contiguous(),
            max(plan.lengths, default=1), indices, plan.cu_q, layer.scaling,
        )
    else:
        pool = backend.token_to_kv_pool
        packed_k, packed_v = N.pack_qsa_prefill_kv(
            k=pool.get_key_buffer(layer.layer_id), v=pool.get_value_buffer(layer.layer_id),
            req_to_token=backend.req_to_token_pool.req_to_token,
            req_indices=forward_batch.req_pool_indices, cu_k=plan.cu_k,
            total_k=sum(plan.lengths), max_k=max(plan.lengths, default=1), output_dtype=q.dtype,
        )
        output = sparse_chunk_prefill(q, packed_k, packed_v, indices, plan.cu_q,
                                     plan.cu_k, meta.sequence_lengths, layer.scaling,
                                     max(plan.extend_lengths, default=1))
    return backend._pad_extend_output(output, num_rows)


def around_mrope_extend(original, batch, model_runner, schedule_batch):
    config = model_runner.model_config.hf_config
    positions = batch.positions
    if (getattr(config, 'model_type', '').startswith('qwen4_exp')
        and positions is not None and positions.is_cuda and positions.ndim == 1
        and all(mm is None for mm in schedule_batch.multimodal_inputs)):
        batch.mrope_positions = positions.to(torch.int64).unsqueeze(0).expand(3, -1).contiguous()
        return
    return original(batch, model_runner, schedule_batch)
