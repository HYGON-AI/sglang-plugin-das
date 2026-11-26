from sglang.kernels.ops.attention.dsv4.kv_layout import KVLayout


def dequantize_sparse_prefill_rows(buffer, token_ids, out, page_size, layout, *, lightop, fallback):
    if lightop and layout is KVLayout.V4 and page_size in (2, 64, 256):
        import torch
        from lightop.kvcache import dsv4_dequantize_k_cache_paged_out

        dsv4_dequantize_k_cache_paged_out(buffer.view(torch.uint8), token_ids, out, page_size)
    else:
        fallback(buffer, token_ids, page_size=page_size, out=out, layout=layout)
