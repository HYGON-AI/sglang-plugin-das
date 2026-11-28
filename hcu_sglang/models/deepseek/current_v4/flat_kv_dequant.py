from sglang.kernels.ops.attention.dsv4.kv_layout import KVLayout


def dequantize_sparse_prefill_rows(buffer, token_ids, out, page_size, layout, *, lightop, fallback, valid_count=None):
    if valid_count is not None:
        # Packed capacity may end in -1 IDs. The counted kernel never reads that suffix.
        from hcu_sglang.models.deepseek.current_v4.hcu_backend.dequant_k_cache import dequantize_k_cache_paged
        return dequantize_k_cache_paged(
            buffer, token_ids, page_size=page_size, out=out, layout=layout, valid_count=valid_count
        )
    if lightop and layout is KVLayout.V4 and page_size in (2, 64, 256):
        import torch
        from lightop.kvcache import dsv4_dequantize_k_cache_paged_out

        dsv4_dequantize_k_cache_paged_out(buffer.view(torch.uint8), token_ids, out, page_size)
    else:
        fallback(buffer, token_ids, page_size=page_size, out=out, layout=layout)


def compact_sparse_prefill_enabled(forward_batch) -> bool:
    """Only HCU eager P-worker C1/C2; keep Decode, Graph and TBO unchanged."""
    import torch
    from hcu_sglang.utils.detection import is_hcu
    from sglang.srt.model_executor.forward_batch_info import ForwardMode

    if not is_hcu() or forward_batch.forward_mode != ForwardMode.EXTEND:
        return False
    from hcu_sglang.layers.attention.current_dsa.forward_batch_utils import (
        effective_forward_mode,
    )
    from sglang.srt.model_executor.runner_utils.capture_mode import get_is_capture_mode
    from sglang.srt.runtime_context import get_disagg

    return (
        effective_forward_mode(forward_batch) == ForwardMode.EXTEND
        and get_disagg().disaggregation_mode != "decode"
        and not get_is_capture_mode()
        and not torch.cuda.is_current_stream_capturing()
        and getattr(forward_batch, "tbo_parent_token_range", None) is None
        and not getattr(forward_batch, "tbo_children", None)
    )


def plan_topk_rows(cache, token_ids, indices, lengths, topk, layout):
    """Select directly for a full-size V4 workspace, otherwise compact the union."""
    import torch
    from lightop import kvcache

    count = None
    use_selected = (
        layout is KVLayout.V4
        and token_ids.numel() <= indices.shape[0] * topk
        and hasattr(kvcache, "dsv4_dequantize_sparse_topk_out")
        and torch.cuda.get_device_properties(token_ids.device).gcnArchName.startswith(("gfx936", "gfx938"))
    )
    if not use_selected and hasattr(kvcache, "dsv4_pack_sparse_topk_workspace"):
        token_ids, indices, count = cache.pack_compressed_topk(token_ids, indices, lengths, topk)
    return token_ids, indices, count, use_selected


def dequantize_selected_rows(cache, token_ids, indices, lengths, dest, topk, page_size):
    import torch
    from lightop.kvcache import dsv4_dequantize_sparse_topk_out
    from hcu_sglang.models.deepseek.current_v4.hcu_backend.dequant_k_cache import fp8_dtype

    dsv4_dequantize_sparse_topk_out(
        cache.view(torch.uint8), token_ids, indices, lengths, dest, topk, page_size,
        fp8_fnuz=fp8_dtype == torch.float8_e4m3fnuz,
    )


def _fp8_routed_experts(config):
    if not isinstance(config, dict):
        return False
    if config.get("quant_method") == "fp8":
        return config.get("routed_experts_quant_method") in (None, "fp8")
    if config.get("quant_method") != "compressed-tensors":
        return False
    groups = config.get("config_groups")
    if not isinstance(groups, dict) or config.get("routed_experts_quant_method") == "mxfp4":
        return False
    routed = [g for g in groups.values() if isinstance(g, dict) and any(
        isinstance(t, str) and "experts" in t and "shared_experts" not in t
        for t in (g.get("targets") or [])
    )]
    if not routed and len(groups) == 1 and config.get("format") == "float-quantized":
        group = next(iter(groups.values()))
        if isinstance(group, dict) and group.get("targets") == ["Linear"]:
            ignore = config.get("ignore", [])
            if not any("expert" in str(rule) or rule == "Linear" for rule in ignore):
                routed = [group]
    return bool(routed) and all(
        isinstance(g.get(part), dict) and g[part].get("type") == "float" and g[part].get("num_bits") == 8
        for g in routed for part in ("weights", "input_activations")
    )


def configure_bf16_decode_gather(backend, is_hcu):
    import logging
    from sglang.srt.environ import envs

    logger = logging.getLogger(__name__)
    pool = backend.token_to_kv_pool
    config = backend.model_runner.model_config
    backend._dsv41_fp8_decode_gather = (
        is_hcu and getattr(config.hf_text_config, "model_type", None) in ("deepseek_v41", "deepseek_v41_text")
        and _fp8_routed_experts(config.hf_quant_config)
        and not pool.is_bf16_attention_kv_cache
        and getattr(pool, "kv_layout", None) is KVLayout.V4
    )
    flash = envs.SGLANG_DSV4_HCU_USE_BF16_FLASH_MLA
    lightop = envs.SGLANG_DSV4_HCU_USE_LIGHTOP_BF16_GATHER
    enabled = flash.get() and (flash.is_set() or backend._dsv41_fp8_decode_gather)
    native = lightop.get() and (lightop.is_set() or (backend._dsv41_fp8_decode_gather and enabled))
    backend._dsv4_bf16_flashmla_decode = enabled
    backend._dsv4_lightop_bf16_gather = native
    backend._dsv4_lightop_kvcache_op = None
    backend._dsv4_bf16_flashmla_workspaces = {}
    if native and not enabled:
        raise RuntimeError("SGLANG_DSV4_HCU_USE_LIGHTOP_BF16_GATHER requires SGLANG_DSV4_HCU_USE_BF16_FLASH_MLA=1")
    if not enabled:
        return
    if not is_hcu:
        raise RuntimeError("SGLANG_DSV4_HCU_USE_BF16_FLASH_MLA is only supported on HCU")
    if pool.is_bf16_attention_kv_cache:
        logger.warning("DSV4 attention KV cache is already BF16")
    if native:
        try:
            from lightop import op
        except (ImportError, OSError):
            op = None
        required = ("dsv4_gather_upconvert_k_cache_paged", "dsv4_gather_upconvert_dual_k_cache_paged")
        missing = [name for name in required if not callable(getattr(op, name, None))]
        if missing and lightop.is_set():
            raise RuntimeError(f"Requested LightOp gather is missing {', '.join(missing)}")
        if missing:
            logger.warning("LightOp lacks %s, using Triton for DSV4 gather", ", ".join(missing))
            backend._dsv4_lightop_bf16_gather = False
        else:
            backend._dsv4_lightop_kvcache_op = op
    provider = "LightOp" if backend._dsv4_lightop_bf16_gather else "Triton"
    logger.info("Enabled DSV4 %s FP8 KV gather/upconvert for BF16 FlashMLA", provider)


def use_bf16_decode_gather(backend, mode):
    if not backend._dsv4_bf16_flashmla_decode:
        return False
    if backend._dsv41_fp8_decode_gather:
        return mode.is_decode() or mode.is_target_verify() or mode.is_draft_extend_v2()
    return not mode.is_idle()


def get_bf16_gather_workspace(backend, slot, num_queries, topk):
    import torch

    if backend._dsv41_fp8_decode_gather:
        return (
            torch.empty((num_queries, topk, 1, 512), device=backend.device, dtype=torch.bfloat16),
            torch.empty((num_queries, 1, topk), device=backend.device, dtype=torch.int32),
        )
    backend._allocate_dsv4_bf16_flashmla_workspace(slot, num_queries, topk)
    workspace = backend._dsv4_bf16_flashmla_workspaces.get(slot)
    if workspace is None:
        raise RuntimeError("DSV4 BF16 FlashMLA gather workspace is unavailable")
    gathered, compact = workspace
    return gathered[:num_queries], compact[:num_queries]
