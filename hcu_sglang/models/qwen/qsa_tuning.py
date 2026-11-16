from __future__ import annotations


def row_chunk_size(budget_mib, rows, keys, heads, original):
    if budget_mib == 0:
        return original(rows, keys, heads)
    if not 1 <= budget_mib <= 2048:
        raise ValueError('QSA prefill logits budget must be 0 or 1..2048 MiB')
    if rows <= 0 or keys <= 0:
        return original(rows, keys, heads)
    if heads <= 0:
        raise ValueError('QSA query head count must be positive')
    block_q = max(1, 128 // heads)
    padded_rows = max(block_q, budget_mib * 1024 * 1024 // (keys * 4))
    padded_rows = max(block_q, padded_rows // block_q * block_q)
    return min(rows, padded_rows)


def around_row_chunk_size(original, rows, keys, heads):
    from sglang.srt.environ import envs
    return row_chunk_size(envs.SGLANG_HCU_QSA_PREFILL_LOGITS_MIB.get(),
                          rows, keys, heads, original)


def around_mqa_prefill(original, q, k, row_starts, row_ends, score_scale=None):
    from sglang.srt.environ import envs
    key_span = envs.SGLANG_HCU_QSA_PREFILL_KEY_SPAN.get()
    if key_span == 0:
        return original(q, k, row_starts, row_ends, score_scale)
    if key_span not in (1024, 2048, 4096, 8192):
        raise ValueError('QSA prefill key span must be 0, 1024, 2048, 4096 or 8192')
    import torch
    if (q.is_cuda and k.device == q.device
            and q.dtype == k.dtype == torch.bfloat16
            and q.ndim == k.ndim == 3
            and q.shape[1:] == (4, 128) and k.shape[1:] == (1, 128)
            and k.shape[0] >= 8192
            and row_starts.device == row_ends.device == q.device
            and row_starts.ndim == row_ends.ndim == 1
            and row_starts.numel() == row_ends.numel() == q.shape[0]):
        from hcu_sglang.models.qwen.qsa_mqa_split_key import prefill
        return prefill(q, k, row_starts, row_ends, score_scale, key_span=key_span)
    return original(q, k, row_starts, row_ends, score_scale)
