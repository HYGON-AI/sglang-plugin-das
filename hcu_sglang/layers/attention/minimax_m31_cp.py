import logging
from functools import cache

import torch

LOG = logging.getLogger(__name__)


@cache
def _announce():
    LOG.info("MiniMax M3.1 query-sharded Zigzag sparse prefill active")


def _gather_kv(forward_batch, strategy, k, v, idx_k, idx_v):
    tensors = (k, v, idx_k) if idx_v is None else (k, v, idx_k, idx_v)
    if any(t.dtype != k.dtype or t.shape[0] != k.shape[0] for t in tensors):
        raise ValueError("MiniMax CP needs equal row counts and dtypes in the KV payload")
    flat = [t.flatten(1) for t in tensors]
    gathered = strategy.gather_kv_cache(torch.cat(flat, dim=1), forward_batch)
    restored = [
        part.reshape(gathered.shape[0], *t.shape[1:])
        for part, t in zip(gathered.split([x.shape[1] for x in flat], dim=1), tensors)
    ]
    if idx_v is None:
        restored.append(None)
    return restored


def _prefill_cp(original, self, q, k, v, layer, forward_batch, save_kv_cache=True,
                *, idx_q, idx_k, idx_v):
    from sglang.srt.layers.cp.utils import get_cp_strategy, is_cp_active

    if not is_cp_active(forward_batch):
        return original(
            self, q, k, v, layer, forward_batch, save_kv_cache,
            idx_q=idx_q, idx_k=idx_k, idx_v=idx_v,
        )
    strategy = get_cp_strategy()
    if strategy.name != "zigzag" or self.is_npu or self._loc_mapping is not None:
        raise ValueError("MiniMax HCU sparse prefill CP requires Zigzag and a local KV pool")
    _announce()
    from sglang.srt.layers.attention.minimax_sparse_ops.minimax_sparse import minimax_sparse_prefill

    disable_value = layer.layer_id in self.disable_value_layer_ids
    k, v, idx_k, idx_v = _gather_kv(
        forward_batch, strategy, k, v, idx_k, None if disable_value else idx_v
    )
    loc = forward_batch.out_cache_loc
    if loc.numel() != k.shape[0]:
        raise ValueError(f"MiniMax CP gathered {k.shape[0]} KV rows for {loc.numel()} slots")
    self.kv_pool.set_fused_kv_index_buffer(
        layer, loc, k, v, idx_k, idx_v,
        layer.k_scale_float, layer.v_scale_float, layer.idx_k_scale_float, layer.idx_v_scale_float,
    )
    k_cache, v_cache = self.kv_pool.get_kv_buffer(layer.layer_id)
    if disable_value:
        index_k, index_v = self.kv_pool.get_index_k_buffer(layer.layer_id), None
    else:
        index_k, index_v = self.kv_pool.get_index_kv_buffer(layer.layer_id)
    meta = forward_batch.attn_cp_metadata
    pieces = (
        (0, meta.total_q_prev_tokens, meta.cu_seqlens_q_prev_tensor,
         meta.kv_len_prev_tensor, meta.actual_seq_q_prev_tensor,
         meta.actual_seq_q_prev_list, meta.kv_len_prev_list),
        (meta.total_q_prev_tokens, meta.total_q_prev_tokens + meta.total_q_next_tokens,
         meta.cu_seqlens_q_next_tensor, meta.kv_len_next_tensor,
         meta.actual_seq_q_next_tensor, meta.actual_seq_q_next_list, meta.kv_len_next_list),
    )
    outputs = []
    index_outputs = []
    for start, end, cu, lengths, q_lengths, q_lengths_cpu, kv_lengths_cpu in pieces:
        if start == end:
            continue
        idx_o, o = minimax_sparse_prefill(
            q=q[start:end],
            k_cache=k_cache,
            v_cache=v_cache,
            sink=None,
            idx_q=idx_q[start:end],
            idx_k_cache=index_k,
            idx_v_cache=index_v,
            idx_sink=None,
            req_to_token=self.req_to_token,
            slot_ids=forward_batch.req_pool_indices,
            cu_seqlens=cu,
            seq_lens=lengths,
            prefix_lens=lengths - q_lengths,
            max_seqlen_q=max(q_lengths_cpu),
            max_seqlen_k=max(kv_lengths_cpu),
            block_size_q=min(self.block_size_q, max(1, 128 // (q.shape[1] // k_cache.shape[-2]))),
            block_size_k=self.block_size_k,
            topk=self.topk_blocks,
            init_blocks=self.init_blocks,
            local_blocks=self.local_blocks,
            score_type=self.score_type,
            disable_index_value=disable_value,
            seqlens_cpu=q_lengths_cpu,
            q_scale=layer.q_scale_float,
            k_scale=layer.k_scale_float,
            v_scale=layer.v_scale_float,
            idx_q_scale=layer.idx_q_scale_float,
            idx_k_scale=layer.idx_k_scale_float,
            idx_v_scale=layer.idx_v_scale_float,
            page_size=self.page_size,
        )
        outputs.append(o.flatten(1))
        if idx_o is not None:
            index_outputs.append(idx_o.flatten(1))
    padding = q.shape[0] - pieces[-1][1]
    if padding:
        outputs.append(q.new_zeros((padding, q.shape[1] * v_cache.shape[-1])))
        if not disable_value:
            index_outputs.append(idx_q.new_zeros((padding, idx_q.shape[1] * index_v.shape[-1])))
    return (
        None if disable_value else torch.cat(index_outputs, dim=0),
        torch.cat(outputs, dim=0) if outputs else q.flatten(1),
    )


def register_hooks():
    from sglang.srt.plugins.hook_registry import HookRegistry, HookType

    HookRegistry.register(
        "sglang.srt.layers.attention.minimax_sparse_backend.MiniMaxSparseAttnBackend.forward_extend",
        _prefill_cp,
        HookType.AROUND,
    )
