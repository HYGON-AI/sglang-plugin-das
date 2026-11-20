from __future__ import annotations

import logging

_route_logged = False


def select_rows(original, indexer, q, keys, starts, ends, positions, lengths, *, threshold, parallel):
    if threshold < 0:
        raise ValueError('QSA prefill TP row threshold must be nonnegative')
    if (threshold == 0 or not q.is_cuda or q.shape[0] < threshold
            or parallel.attn_tp_size == 1 or parallel.attn_cp_size != 1):
        return original(indexer, q, keys, starts, ends, positions, lengths)
    import torch.nn.functional as F
    global _route_logged
    world = parallel.attn_tp_size
    chunk = (q.shape[0]+world-1)//world
    begin = min(parallel.attn_tp_rank*chunk,q.shape[0])
    end = min(begin+chunk,q.shape[0])
    local = original(indexer,q[begin:end],keys,starts[begin:end],ends[begin:end],
                     positions[begin:end],lengths[begin:end])
    local = F.pad(local,(0,0,0,chunk-local.shape[0]),value=-1).contiguous()
    output = parallel.attn_tp_group.all_gather(local,dim=0)[:q.shape[0]]
    if not _route_logged:
        logging.getLogger(__name__).info(
            'Qwen4 QSA TP rows active: rows=%s, local_rows=%s, attn_tp=%s',
            q.shape[0],end-begin,world)
        _route_logged=True
    return output


def around_select_prefill_tokens(original, indexer, q, compressed_keys, row_starts,
                                 row_ends, query_positions, sequence_lengths_for_rows):
    from sglang.srt.environ import envs
    threshold = envs.SGLANG_HCU_QSA_PREFILL_TP_ROWS.get()
    if threshold == 0:
        return original(indexer,q,compressed_keys,row_starts,row_ends,
                        query_positions,sequence_lengths_for_rows)
    from sglang.srt.runtime_context import get_parallel
    return select_rows(original,indexer,q,compressed_keys,row_starts,row_ends,
                       query_positions,sequence_lengths_for_rows,
                       threshold=threshold,parallel=get_parallel())
