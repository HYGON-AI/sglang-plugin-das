"""Current CP cache lifetime and read indices at official backend boundaries."""
from __future__ import annotations
from hcu_sglang.models.deepseek.current_v4.flat_kv_dequant import configure_bf16_decode_gather, get_bf16_gather_workspace, use_bf16_decode_gather
from hcu_sglang.models.deepseek.current_v4.flat_kv_dequant import dequantize_sparse_prefill_rows, compact_sparse_prefill_enabled, plan_topk_rows, dequantize_selected_rows
import sglang.srt.layers.attention.deepseek_v4_backend as upstream
globals().update({name:value for name,value in vars(upstream).items() if not name.startswith('__')})
from hcu_sglang.utils.detection import is_hcu
from hcu_sglang.mem_cache.cp_cache_layer_split.deepseek_v4_helpers import (
    maybe_prepare_cp_cache_layer_split_forward, is_cp_cache_layer_split_deepseek_v4_pool,
)
from sglang.srt.layers.attention.hip_flash_mla import flash_mla_with_kvcache_entrypoint
from hcu_sglang.mem_cache.hcu_v4_pool import DeepSeekV4TokenToKVPool
from .dequant import gather_upconvert_k_cache_paged
_is_hcu = is_hcu()
_is_sm120 = bool(get_platform().is_sm120)

def _forward_attention(
    self,
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    layer: RadixAttention,
    forward_batch: ForwardBatch,
    compress_ratio: Literal[0, 1, 2, 4, 128],
    save_kv_cache: bool = True,
    attn_sink: Optional[torch.Tensor] = None,
    cos_sin_cache: Optional[torch.Tensor] = None,
    **_,
) -> Union[torch.Tensor, Tuple[torch.Tensor, torch.Tensor]]:
    if self.mtp_enabled and forward_batch.forward_mode.is_idle():
        return q.new_empty(
            (q.shape[0], q.shape[1], layer.v_head_dim),
            dtype=torch.bfloat16 if self.trtllm_attn else q.dtype,
        )

    assert k is v, "DeepseekV4 shares k and v"
    swa_k = k

    layer_id = layer.layer_id
    metadata = self.forward_metadata
    core_attn_metadata = metadata.core_attn_metadata
    token_to_kv_pool = self.token_to_kv_pool
    assert isinstance(token_to_kv_pool, DeepSeekV4TokenToKVPool)

    if isinstance(core_attn_metadata, DSV4AttnMetadata):
        if save_kv_cache:
            self.store_cache(layer_id, swa_k, forward_batch)
        swa_k_cache = token_to_kv_pool.get_swa_key_buffer_radix(layer_id)

        extra_k_cache, extra_indices, extra_topk_lengths = None, None, None
        if compress_ratio != 0:
            extra_k_cache = token_to_kv_pool.get_extra_key_buffer(layer_id)
            extra_indices = core_attn_metadata.sparse_page_indices(compress_ratio)
            extra_topk_lengths = core_attn_metadata.sparse_topk_lengths(
                compress_ratio
            )

        swa_kv_page_size = token_to_kv_pool.get_swa_key_page_size()
        assert swa_k_cache.ndim == 2
        # The kernel detects each cache's format from the last dim of this
        # view: 584 (V4), 528 (V4.1 fp8) or 288 (V4.1 fp4, extra cache only).
        k_cache_total_dim = token_to_kv_pool.get_swa_key_bytes_per_token()
        swa_k_cache = swa_k_cache[:, : swa_kv_page_size * k_cache_total_dim].view(
            swa_k_cache.shape[0], swa_kv_page_size, 1, k_cache_total_dim
        )

        if extra_k_cache is not None:
            extra_page_size = token_to_kv_pool.get_extra_key_page_size(layer_id)
            extra_total_dim = token_to_kv_pool.get_extra_key_bytes_per_token(
                layer_id
            )
            extra_k_cache = extra_k_cache[
                :, : extra_page_size * extra_total_dim
            ].view(
                extra_k_cache.shape[0],
                extra_page_size,
                1,
                extra_total_dim,
            )
        swa_page_indices = core_attn_metadata.swa_page_indices
        swa_topk_lengths = core_attn_metadata.swa_topk_lengths
        if is_cp_cache_layer_split_deepseek_v4_pool(token_to_kv_pool):
            swa_page_indices = token_to_kv_pool.remap_swa_indices_for_read(
                layer_id, swa_page_indices
            )
            if extra_indices is not None:
                extra_indices = token_to_kv_pool.remap_extra_indices_for_read(
                    layer_id, extra_indices
                )

        def match_num_queries(x, value):
            if x is None or x.shape[0] == q.shape[0]:
                return x
            if x.shape[0] > q.shape[0]:
                return x[: q.shape[0]]
            return _pad_tensor_to_size(x, q.shape[0], value=value)

        swa_page_indices = match_num_queries(swa_page_indices, value=0)
        swa_topk_lengths = match_num_queries(swa_topk_lengths, value=1)
        extra_indices = match_num_queries(extra_indices, value=-1)
        extra_topk_lengths = match_num_queries(extra_topk_lengths, value=1)

        if self.trtllm_attn:
            # The uniform-FP8 pool is readable only by trtllm-gen.
            return self._forward_trtllm(
                q=q,
                layer=layer,
                compress_ratio=compress_ratio,
                core_attn_metadata=core_attn_metadata,
                forward_batch=forward_batch,
                attn_sink=attn_sink,
                swa_page_indices=swa_page_indices,
                extra_indices=extra_indices,
                extra_topk_lengths=extra_topk_lengths,
            )

        if not _is_hcu:
            if q.ndim == 3:
                q = q.unsqueeze(1)
            if swa_page_indices.ndim == 2:
                swa_page_indices = swa_page_indices.unsqueeze(1)
            if extra_indices is not None and extra_indices.ndim == 2:
                extra_indices = extra_indices.unsqueeze(1)

        assert attn_sink is not None

        flashmla_metadata = core_attn_metadata.get_flashmla_metadata(compress_ratio)

        assert swa_page_indices.shape[-1] % 64 == 0, (
            f"{swa_page_indices.shape=}'s last dimension is not aligned to 64"
        )
        if extra_indices is not None:
            assert extra_indices.shape[-1] % 64 == 0, (
                f"{extra_indices.shape=}'s last dimension is not aligned to 64"
            )

        if not _is_hcu:
            use_mega_paged = cos_sin_cache is not None and self._mega_paged_prefill(
                forward_batch,
                num_qo_tokens=q.shape[0],
                layer_id=layer_id,
                compress_ratio=compress_ratio,
                num_heads=q.shape[-2],
            )

            # The tail stays dense: its window floor lives in swa_page_indices,
            # which the sparse chunk cache ignores.
            if (
                self._use_sparse_prefill(forward_batch, num_qo_tokens=q.shape[0])
                and not use_mega_paged
                and self.forward_metadata.late_layer_tail is None
                and token_to_kv_pool.request_window is None
            ):
                if use_dsv4_q8kv8_sparse_prefill(self.dsv4_prefill_backend):
                    return self._forward_prefill_sparse_q8kv8(
                        q=q,
                        layer_id=layer_id,
                        compress_ratio=compress_ratio,
                        forward_batch=forward_batch,
                        token_to_kv_pool=token_to_kv_pool,
                        core_attn_metadata=core_attn_metadata,
                        attn_sink=attn_sink,
                    )
                return self._forward_prefill_sparse(
                    q=q,
                    layer_id=layer_id,
                    compress_ratio=compress_ratio,
                    forward_batch=forward_batch,
                    token_to_kv_pool=token_to_kv_pool,
                    core_attn_metadata=core_attn_metadata,
                    attn_sink=attn_sink,
                    cos_sin_cache=cos_sin_cache,
                )

            if cos_sin_cache is not None:
                return self._forward_mega_decode(
                    q=q.squeeze(1),
                    swa_k_cache=swa_k_cache,
                    swa_page_indices=swa_page_indices.squeeze(1),
                    positions=match_num_queries(
                        core_attn_metadata.positions, value=0
                    ),
                    cos_sin_cache=cos_sin_cache,
                    attn_sink=attn_sink,
                    swa_topk_lengths=swa_topk_lengths,
                    extra_k_cache=extra_k_cache,
                    extra_indices=None
                    if extra_indices is None
                    else extra_indices.squeeze(1),
                    extra_topk_lengths=extra_topk_lengths,
                )

            if (
                self.is_dsv41
                and get_platform().is_sm100
                and can_use_swapab_attention(
                    q,
                    swa_k_cache,
                    extra_k_cache,
                    layer.tp_q_head_num,
                    self.head_dim_v,
                    self.softmax_scale,
                )
                and (
                    forward_batch.forward_mode.is_decode()
                    or forward_batch.forward_mode.is_target_verify()
                    or forward_batch.forward_mode.is_draft_extend_v2()
                )
            ):
                from sglang.kernels.ops.attention.dsv4.decode_attention_sm100 import (
                    swapab_attention,
                )

                return swapab_attention(
                    q[..., :16, :],
                    swa_k_cache,
                    swa_page_indices,
                    swa_topk_lengths,
                    attn_sink,
                    extra_k_cache,
                    extra_indices,
                    extra_topk_lengths,
                )

            if get_platform().is_sm120:
                from sglang.kernels.ops.attention.flash_mla_sm120 import (
                    SM120_DECODE_MAX_TOKENS,
                    flash_mla_with_kvcache_sm120,
                )

                # The 64-head pad only serves the decode kernel's h_q
                # specialization; the prefill kernel takes any h_q, so drop it.
                real_heads = layer.tp_q_head_num
                if q.shape[0] > SM120_DECODE_MAX_TOKENS:
                    if q.shape[-2] > real_heads:
                        q = q[..., :real_heads, :].contiguous()
                    if attn_sink is not None and attn_sink.shape[0] > real_heads:
                        attn_sink = attn_sink[:real_heads]

                o = flash_mla_with_kvcache_sm120(
                    q=q,
                    k_cache=swa_k_cache,
                    head_dim_v=self.head_dim_v,
                    softmax_scale=self.softmax_scale,
                    indices=swa_page_indices,
                    topk_length=swa_topk_lengths,
                    attn_sink=attn_sink,
                    extra_k_cache=extra_k_cache,
                    extra_indices_in_kvcache=extra_indices,
                    extra_topk_length=extra_topk_lengths,
                )[0]
            else:
                if _is_xpu:
                    from sgl_kernel import flash_mla_with_kvcache
                else:
                    from sgl_kernel.flash_mla import flash_mla_with_kvcache

                if self.is_dsv41:
                    _maybe_precompute_flashmla_sched_meta(
                        flashmla_metadata,
                        q=q,
                        indices=swa_page_indices,
                        topk_length=swa_topk_lengths,
                        extra_indices=extra_indices,
                        extra_topk_length=extra_topk_lengths,
                    )
                o = flash_mla_with_kvcache(
                    q=q,
                    k_cache=swa_k_cache,
                    head_dim_v=self.head_dim_v,
                    block_table=None,
                    cache_seqlens=None,
                    tile_scheduler_metadata=flashmla_metadata,
                    softmax_scale=self.softmax_scale,
                    is_fp8_kvcache=True,
                    indices=swa_page_indices,
                    topk_length=swa_topk_lengths,
                    attn_sink=attn_sink,
                    extra_k_cache=extra_k_cache,
                    extra_indices_in_kvcache=extra_indices,
                    extra_topk_length=extra_topk_lengths,
                )[0]

            o = o.squeeze(1)
            return o

        if not envs.SGLANG_DSV4_SPLIT_PREFILL_DECODE_MLA.get():
            return self._forward_flash_mla_decode(
                q=q,
                swa_k_cache=swa_k_cache,
                swa_page_indices=swa_page_indices,
                swa_topk_lengths=swa_topk_lengths,
                flashmla_metadata=flashmla_metadata,
                attn_sink=attn_sink,
                extra_k_cache=extra_k_cache,
                extra_indices=extra_indices,
                extra_topk_lengths=extra_topk_lengths,
                compress_ratio=compress_ratio,
                layer_id=layer_id,
                forward_batch=forward_batch,
            )

        if forward_batch.forward_mode.is_decode_or_idle() or (
            forward_batch.forward_mode.is_target_verify()
        ):
            return self._forward_flash_mla_decode(
                q=q,
                swa_k_cache=swa_k_cache,
                swa_page_indices=swa_page_indices,
                swa_topk_lengths=swa_topk_lengths,
                flashmla_metadata=flashmla_metadata,
                attn_sink=attn_sink,
                extra_k_cache=extra_k_cache,
                extra_indices=extra_indices,
                extra_topk_lengths=extra_topk_lengths,
                compress_ratio=compress_ratio,
                layer_id=layer_id,
                forward_batch=forward_batch,
            )

        if forward_batch.forward_mode.is_prefill(include_draft_extend_v2=True):
            if (
                self.forward_metadata.late_layer_tail is None
                and _should_use_sparse_prefill(
                    q, forward_batch, allow_cp=self.is_dsv41
                )
            ):
                return self._forward_prefill_sparse(
                    q=q,
                    layer_id=layer_id,
                    compress_ratio=compress_ratio,
                    forward_batch=forward_batch,
                    token_to_kv_pool=token_to_kv_pool,
                    core_attn_metadata=core_attn_metadata,
                    attn_sink=attn_sink,
                )
            return self._forward_flash_mla_prefill(
                q=q,
                swa_k_cache=swa_k_cache,
                swa_page_indices=swa_page_indices,
                swa_topk_lengths=swa_topk_lengths,
                flashmla_metadata=flashmla_metadata,
                attn_sink=attn_sink,
                extra_k_cache=extra_k_cache,
                extra_indices=extra_indices,
                extra_topk_lengths=extra_topk_lengths,
                forward_batch=forward_batch,
                compress_ratio=compress_ratio,
                layer_id=layer_id,
            )

        raise NotImplementedError(
            f"unsupported mode {forward_batch.forward_mode=} for DeepSeekV4 FlashMLA"
        )

    raise NotImplementedError("ragged attention")



def _forward_prefill_sparse(
    self,
    q: torch.Tensor,
    layer_id: int,
    compress_ratio: Literal[0, 1, 2, 4, 128],
    forward_batch: ForwardBatch,
    token_to_kv_pool: DeepSeekV4TokenToKVPool,
    core_attn_metadata: DSV4AttnMetadata,
    attn_sink: torch.Tensor,
    cos_sin_cache: Optional[torch.Tensor] = None,
) -> Union[torch.Tensor, Tuple[torch.Tensor, torch.Tensor]]:
    """Gathers each request's SWA window and c4/c128 cache into a flat bf16
    workspace that flash_mla_sparse_fwd reads through per-query rebased indices."""
    if _is_hcu:
        from hcu_sglang.layers.attention.debug_flash_mla_adapter import (
            flash_mla_sparse_fwd_entrypoint as flash_mla_sparse_fwd,
        )
    elif _is_xpu:
        from sgl_kernel import flash_mla_sparse_fwd
    else:
        from sgl_kernel.flash_mla import flash_mla_sparse_fwd

    # q is (b, 1, h_q, d_qk); flash_mla_sparse_fwd takes (s_q, h_q, d_qk).
    q_flat = q.squeeze(1)

    compact = self.is_dsv41 and core_attn_metadata.request_window_layout is None and compact_sparse_prefill_enabled(forward_batch)
    cache = self.forward_metadata.sparse_prefill_cache
    if cache is not None and (getattr(cache, "compact_seq_lens_cpu", None) is not None) != compact:
        cache = None
    if cache is None:
        cache = self._build_sparse_prefill_chunk_cache(
            forward_batch, core_attn_metadata, num_qo_tokens=q_flat.shape[0]
        )
        self.forward_metadata.sparse_prefill_cache = cache

    # Resolve the workspace + indices for this ratio, then dequant
    # SWA + compressed regions directly into the workspace (no torch.cat).
    compressed_slice = None
    extra_k_cache = None
    extra_k_layout = None
    extra_page_size = None
    flat_token_ids = None
    compact = getattr(cache, "compact_seq_lens_cpu", None) is not None
    workspace_kwargs = {"release_before_grow": True} if compact else {}
    count = None
    selected = False
    if compress_ratio == 0:
        workspace = self.sparse_prefill_workspace.get(cache.swa_token_ids.shape[0], **workspace_kwargs)
        combined_indices = cache.c0_combined_indices
        combined_lens = cache.c0_combined_lens
        swa_slice = workspace
    else:
        extra_page_size = token_to_kv_pool.get_extra_key_page_size(layer_id)
        extra_k_cache = token_to_kv_pool.get_extra_key_buffer(layer_id)
        extra_k_layout = token_to_kv_pool.get_extra_key_layout(layer_id)
        flat_token_ids, combined_indices, combined_lens = cache.layer_inputs(
            compress_ratio, core_attn_metadata, extra_page_size
        )
        if compact and compress_ratio in (1, 2) and q_flat.shape[0] and envs.SGLANG_DSV41_PREFILL_TOPK_KV_WORKSPACE.get():
            topk = core_attn_metadata.sparse_raw_indices(compress_ratio).shape[-1]
            flat_token_ids, combined_indices, count, selected = plan_topk_rows(
                cache, flat_token_ids, combined_indices, combined_lens, topk, extra_k_layout,
            )
        n_compressed = flat_token_ids.shape[0]
        workspace = self.sparse_prefill_workspace.get(
            n_compressed + cache.swa_token_ids.shape[0], **workspace_kwargs
        )
        compressed_slice = workspace[:n_compressed]
        swa_slice = workspace[n_compressed:]

    swa_token_ids = cache.swa_token_ids
    if is_cp_cache_layer_split_deepseek_v4_pool(token_to_kv_pool):
        swa_token_ids = token_to_kv_pool.remap_flat_token_ids_for_read(
            layer_id, swa_token_ids, family="swa"
        )
        if flat_token_ids is not None:
            flat_token_ids = token_to_kv_pool.remap_flat_token_ids_for_read(
                layer_id, flat_token_ids, family="extra"
            )

    swa_k_layout = token_to_kv_pool.get_swa_key_layout()
    use_lightop = _is_hcu and envs.SGLANG_LIGHTOP_DEQUANTIZE_K_CACHE_PAGED.get()
    if selected:
        dequantize_selected_rows(extra_k_cache, flat_token_ids, combined_indices, combined_lens,
                                 compressed_slice, topk, extra_page_size)
    elif compressed_slice is not None:
        dequantize_sparse_prefill_rows(
            extra_k_cache, flat_token_ids, compressed_slice, extra_page_size,
            extra_k_layout, lightop=use_lightop, fallback=dequantize_k_cache_paged, valid_count=count,
        )
    dequantize_sparse_prefill_rows(
        token_to_kv_pool.get_swa_key_buffer_radix(layer_id), swa_token_ids,
        swa_slice, cache.swa_page_size, swa_k_layout,
        lightop=use_lightop, fallback=dequantize_k_cache_paged,
    )
    kv = workspace

    if cos_sin_cache is not None:
        return self._forward_mega_prefill(
            q=q_flat,
            kv=kv,
            indices=combined_indices.unsqueeze(1),
            positions=core_attn_metadata.positions[: q_flat.shape[0]],
            cos_sin_cache=cos_sin_cache,
            attn_sink=attn_sink,
            topk_length=combined_lens,
        )

    o, _, _ = flash_mla_sparse_fwd(
        q=q_flat,
        kv=kv,
        indices=combined_indices.unsqueeze(1),
        sm_scale=self.softmax_scale,
        d_v=self.head_dim_v,
        attn_sink=attn_sink,
        topk_length=combined_lens,
    )
    return o


def around_init_forward_metadata(original, backend, forward_batch):
    if not (backend.mtp_enabled and _get_logical_forward_mode(forward_batch).is_idle()):
        maybe_prepare_cp_cache_layer_split_forward(backend.token_to_kv_pool, forward_batch)
    return original(backend, forward_batch)


def around_enter_late_layer_tail(original, backend, forward_batch):
    saved = original(backend, forward_batch)
    if is_cp_cache_layer_split_deepseek_v4_pool(backend.token_to_kv_pool):
        backend.token_to_kv_pool.reset_batch_active_pages()
    return saved


def register_hooks():
    from sglang.srt.plugins.hook_registry import HookRegistry, HookType
    base = 'sglang.srt.layers.attention.deepseek_v4_backend.DeepseekV4AttnBackend.'
    for name in ['_get_dsv4_bf16_flashmla_workspace','_allocate_dsv4_bf16_flashmla_workspace','_prepare_dsv4_bf16_flashmla_inputs',
                 '_build_flash_mla_input_dict','_call_flash_mla_with_kvcache',
                 '_forward_flash_mla_decode','_forward_flash_mla_prefill']:
        setattr(upstream.DeepseekV4AttnBackend, name, globals()[name])
    HookRegistry.register(base+'__init__',around_backend_init,HookType.AROUND)
    HookRegistry.register('sglang.srt.layers.attention.deepseek_v4_backend._create_flashmla_metadata',
        _create_flashmla_metadata,HookType.REPLACE)
    HookRegistry.register(base+'init_forward_metadata',around_init_forward_metadata,HookType.AROUND)
    HookRegistry.register(base+'enter_late_layer_tail',around_enter_late_layer_tail,HookType.AROUND)
    HookRegistry.register(base+'_forward_attention',_forward_attention,HookType.REPLACE)
    HookRegistry.register(base+'_forward_prefill_sparse',_forward_prefill_sparse,HookType.REPLACE)


def _should_use_sparse_prefill(q: torch.Tensor, forward_batch: ForwardBatch, *, allow_cp: bool = False) -> bool:
    sparse_prefill_enabled = envs.SGLANG_OPT_FLASHMLA_SPARSE_PREFILL.get()
    explicit_cp_override = (
        envs.SGLANG_OPT_FLASHMLA_SPARSE_PREFILL.is_set() and sparse_prefill_enabled
    )
    return (
        not _is_sm120
        and (allow_cp or not dsa_use_prefill_cp(forward_batch) or explicit_cp_override)
        and (q.shape[0] > _LARGE_INDEXER_QUERY_THRESHOLD or sparse_prefill_enabled)
    )



def _create_flashmla_metadata():
    if get_platform().is_sm120 or _is_xpu:
        return None
    if _is_hcu:
        import flash_mla
    else:
        import sgl_kernel.flash_mla as flash_mla

    return flash_mla.get_mla_metadata()[0]



def _allocate_dsv4_bf16_flashmla_workspace(
    self,
    slot: str,
    capacity: int,
    topk: int,
) -> None:
    if not self._dsv4_bf16_flashmla_decode or capacity <= 0 or topk <= 0:
        return
    current = self._dsv4_bf16_flashmla_workspaces.get(slot)
    if current is not None:
        current_capacity = current[0].shape[0]
        current_topk = current[0].shape[1]
        if current_capacity >= capacity and current_topk == topk:
            return
        # Drop the previous buffer before reallocating. c128 topk grows
        # every prefill chunk; keeping one workspace per topk OOMs on
        # long context with small chunked_prefill_size.
        del self._dsv4_bf16_flashmla_workspaces[slot]
        del current
    if torch.cuda.is_current_stream_capturing():
        raise RuntimeError(
            "DSV4 BF16 FlashMLA gather workspace must be allocated before "
            "CUDA graph capture"
        )
    gathered_kv = torch.empty(
        (capacity, topk, 1, 512),
        dtype=torch.bfloat16,
        device=self.device,
    )
    compact_indices = torch.empty(
        (capacity, 1, topk),
        dtype=torch.int32,
        device=self.device,
    )
    self._dsv4_bf16_flashmla_workspaces[slot] = (
        gathered_kv,
        compact_indices,
    )



def _prepare_dsv4_bf16_flashmla_inputs(
    self,
    *,
    swa_k_cache: torch.Tensor,
    swa_indices: torch.Tensor,
    swa_topk_lengths: torch.Tensor,
    extra_k_cache: Optional[torch.Tensor],
    extra_indices: Optional[torch.Tensor],
    extra_topk_lengths: Optional[torch.Tensor],
) -> Tuple[
    torch.Tensor,
    torch.Tensor,
    torch.Tensor,
    Optional[torch.Tensor],
    Optional[torch.Tensor],
    Optional[torch.Tensor],
]:
    assert swa_indices.ndim == 3 and swa_indices.shape[1] == 1
    num_queries, _, swa_topk = swa_indices.shape
    extra_topk = 0 if extra_indices is None else extra_indices.shape[-1]

    if self._dsv4_lightop_bf16_gather:
        op = self._dsv4_lightop_kvcache_op
        if op is None:
            raise RuntimeError("DSV4 LightOp gather backend was not initialized")

        swa_gathered, swa_compact = self._get_dsv4_bf16_flashmla_workspace(
            "swa", num_queries, swa_topk
        )
        swa_indices_2d = swa_indices[:, 0, :].contiguous()
        swa_lengths = swa_topk_lengths.reshape(-1).contiguous()
        swa_cache_u8 = swa_k_cache.view(torch.uint8)

        if extra_k_cache is None:
            op.dsv4_gather_upconvert_k_cache_paged(
                swa_cache_u8,
                swa_indices_2d,
                swa_lengths,
                swa_gathered,
                swa_compact,
                swa_k_cache.shape[1],
            )
            return (
                swa_gathered,
                swa_compact,
                swa_lengths,
                None,
                None,
                None,
            )

        assert extra_indices is not None and extra_topk_lengths is not None
        assert extra_indices.ndim == 3 and extra_indices.shape[1] == 1
        assert extra_indices.shape[0] == num_queries
        extra_gathered, extra_compact = self._get_dsv4_bf16_flashmla_workspace(
            "extra", num_queries, extra_topk
        )
        extra_indices_2d = extra_indices[:, 0, :].contiguous()
        extra_lengths = extra_topk_lengths.reshape(-1).contiguous()
        extra_cache_u8 = extra_k_cache.view(torch.uint8)
        op.dsv4_gather_upconvert_dual_k_cache_paged(
            swa_cache_u8,
            swa_indices_2d,
            swa_lengths,
            swa_gathered,
            swa_compact,
            swa_k_cache.shape[1],
            extra_cache_u8,
            extra_indices_2d,
            extra_lengths,
            extra_gathered,
            extra_compact,
            extra_k_cache.shape[1],
        )
        return (
            swa_gathered,
            swa_compact,
            swa_lengths,
            extra_gathered,
            extra_compact,
            extra_lengths,
        )

    # Preserve the original Triton fallback and its combined-cache ABI.
    combined_topk = swa_topk + extra_topk
    gathered_kv, compact_indices = self._get_dsv4_bf16_flashmla_workspace(
        "combined", num_queries, combined_topk
    )
    gather_upconvert_k_cache_paged(
        quant_k_cache=swa_k_cache,
        token_indices=swa_indices,
        topk_lengths=swa_topk_lengths,
        page_size=swa_k_cache.shape[1],
        out=gathered_kv,
        compact_indices=compact_indices,
    )
    combined_topk_lengths = swa_topk_lengths
    if extra_k_cache is not None:
        assert extra_indices is not None and extra_topk_lengths is not None
        assert extra_indices.ndim == 3 and extra_indices.shape[1] == 1
        assert extra_indices.shape[0] == num_queries
        gather_upconvert_k_cache_paged(
            quant_k_cache=extra_k_cache,
            token_indices=extra_indices,
            topk_lengths=extra_topk_lengths,
            page_size=extra_k_cache.shape[1],
            out=gathered_kv,
            compact_indices=compact_indices,
            output_offsets=swa_topk_lengths,
        )
        combined_topk_lengths = swa_topk_lengths + extra_topk_lengths
    return (
        gathered_kv,
        compact_indices,
        combined_topk_lengths,
        None,
        None,
        None,
    )



def _build_flash_mla_input_dict(
    self,
    *,
    q: torch.Tensor,
    swa_k_cache: torch.Tensor,
    swa_page_indices: torch.Tensor,
    swa_topk_lengths: torch.Tensor,
    flashmla_metadata,
    attn_sink: torch.Tensor,
    extra_k_cache: Optional[torch.Tensor],
    extra_indices: Optional[torch.Tensor],
    extra_topk_lengths: Optional[torch.Tensor],
) -> Dict[str, object]:
    # HCU FlashMLA accepts raw packed bytes, while torch's Fnuz dtype is not
    # an accepted tensor carrier in its matcher. Preserve bytes/stride/pointer.
    if _is_hcu:
        if swa_k_cache.dtype in (torch.float8_e4m3fn, torch.float8_e4m3fnuz):
            swa_k_cache = swa_k_cache.view(torch.uint8)
        if extra_k_cache is not None and extra_k_cache.dtype in (torch.float8_e4m3fn, torch.float8_e4m3fnuz):
            extra_k_cache = extra_k_cache.view(torch.uint8)
    return dict(
        q=q,
        k_cache=swa_k_cache,
        head_dim_v=self.head_dim_v,
        block_table=None,
        cache_seqlens=None,
        tile_scheduler_metadata=flashmla_metadata,
        softmax_scale=self.softmax_scale,
        is_fp8_kvcache=swa_k_cache.dtype != torch.bfloat16,
        indices=swa_page_indices,
        topk_length=swa_topk_lengths,
        attn_sink=attn_sink,
        extra_k_cache=extra_k_cache,
        extra_indices_in_kvcache=extra_indices,
        extra_topk_length=extra_topk_lengths,
    )



def _call_flash_mla_with_kvcache(
    self,
    *,
    input_dict: Dict[str, object],
    compress_ratio: Literal[0, 4, 128],
    layer_id: int,
) -> torch.Tensor:
    backend = envs.SGLANG_HACK_FLASHMLA_BACKEND.get()
    if _is_hcu and backend == "auto":
        # HCU uses the external FlashMLA kernel rather than HIP's auto
        # selection between AITER and TileLang decode implementations.
        backend = "kernel"
    return flash_mla_with_kvcache_entrypoint(**input_dict, backend=backend)[0]



def _forward_flash_mla_decode(
    self,
    *,
    q: torch.Tensor,
    swa_k_cache: torch.Tensor,
    swa_page_indices: torch.Tensor,
    swa_topk_lengths: torch.Tensor,
    flashmla_metadata,
    attn_sink: torch.Tensor,
    extra_k_cache: Optional[torch.Tensor],
    extra_indices: Optional[torch.Tensor],
    extra_topk_lengths: Optional[torch.Tensor],
    compress_ratio: Literal[0, 4, 128],
    layer_id: int,
    forward_batch: ForwardBatch,
) -> torch.Tensor:
    if q.ndim == 3:
        q = q.unsqueeze(1)
    if swa_page_indices.ndim == 2:
        swa_page_indices = swa_page_indices.unsqueeze(1)
    if extra_indices is not None and extra_indices.ndim == 2:
        extra_indices = extra_indices.unsqueeze(1)

    logical_forward_mode = _get_logical_forward_mode(forward_batch)
    use_bf16_gather = use_bf16_decode_gather(self, logical_forward_mode)
    if use_bf16_gather and swa_k_cache.dtype != torch.bfloat16:
        (
            swa_k_cache,
            swa_page_indices,
            swa_topk_lengths,
            extra_k_cache,
            extra_indices,
            extra_topk_lengths,
        ) = self._prepare_dsv4_bf16_flashmla_inputs(
            swa_k_cache=swa_k_cache,
            swa_indices=swa_page_indices,
            swa_topk_lengths=swa_topk_lengths,
            extra_k_cache=extra_k_cache,
            extra_indices=extra_indices,
            extra_topk_lengths=extra_topk_lengths,
        )

    input_dict = self._build_flash_mla_input_dict(
        q=q,
        swa_k_cache=swa_k_cache,
        swa_page_indices=swa_page_indices,
        swa_topk_lengths=swa_topk_lengths,
        flashmla_metadata=flashmla_metadata,
        attn_sink=attn_sink,
        extra_k_cache=extra_k_cache,
        extra_indices=extra_indices,
        extra_topk_lengths=extra_topk_lengths,
    )
    o = self._call_flash_mla_with_kvcache(
        input_dict=input_dict,
        compress_ratio=compress_ratio,
        layer_id=layer_id,
    )
    return o.squeeze(1)



def _forward_flash_mla_prefill(
    self,
    *,
    q: torch.Tensor,
    swa_k_cache: torch.Tensor,
    swa_page_indices: torch.Tensor,
    swa_topk_lengths: torch.Tensor,
    flashmla_metadata,
    attn_sink: torch.Tensor,
    extra_k_cache: Optional[torch.Tensor],
    extra_indices: Optional[torch.Tensor],
    extra_topk_lengths: Optional[torch.Tensor],
    forward_batch: ForwardBatch,
    compress_ratio: Literal[0, 4, 128],
    layer_id: int,
) -> torch.Tensor:
    # On the current Hygon FlashMLA package, flash_mla_with_kvcache is backed
    # by the sparse decode kernel. It treats every query token as a batch
    # item and requires topk_length to have shape (b). Reshaping prefill into
    # [batch, seqlen, ...] makes topk_length [batch, seqlen] and fails with
    # "topk_length must have shape (b)". Keep the old mixed-batch path here;
    # true separated prefill is handled by _forward_prefill_sparse via
    # flash_mla_sparse_fwd.
    return self._forward_flash_mla_decode(
        q=q,
        swa_k_cache=swa_k_cache,
        swa_page_indices=swa_page_indices,
        swa_topk_lengths=swa_topk_lengths,
        flashmla_metadata=flashmla_metadata,
        attn_sink=attn_sink,
        extra_k_cache=extra_k_cache,
        extra_indices=extra_indices,
        extra_topk_lengths=extra_topk_lengths,
        compress_ratio=compress_ratio,
        layer_id=layer_id,
        forward_batch=forward_batch,
    )



def around_backend_init(original, backend, *args, **kwargs):
    result = original(backend, *args, **kwargs)
    configure_bf16_decode_gather(backend, _is_hcu)
    return result


_get_dsv4_bf16_flashmla_workspace = get_bf16_gather_workspace

