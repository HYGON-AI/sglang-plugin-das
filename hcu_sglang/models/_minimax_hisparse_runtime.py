from __future__ import annotations


def coordinator_init(
    self,
    req_to_token_pool: ReqToTokenPool,
    token_to_kv_pool_allocator: Union[
        HiSparseTokenToKVPoolAllocator,
        DeepSeekV4HiSparseTokenToKVPoolAllocator,
    ],
    top_k: int,
    device_buffer_size: int,
    device: str,
    tp_group,
    host_to_device_ratio: int = 2,
    swap_in_block_size: int = 960,
    shared_index_layers: Optional[List[bool]] = None,
    num_draft_tokens: int = 0,
):
    self.req_to_token_pool = req_to_token_pool
    self.token_to_kv_pool_allocator = token_to_kv_pool_allocator
    self.top_k = top_k
    self.device_buffer_size = device_buffer_size
    self.device = device
    self.swap_in_block_size = swap_in_block_size
    # Timing probe: skip the host->device KV bytes to measure the "IO is
    # free" floor. Produces garbage output; benchmarking only.
    self.skip_io = envs.SGLANG_DEBUG_HISPARSE_SKIP_IO.get()
    if _is_xpu and self.skip_io:
        raise ValueError(
            "SGLANG_DEBUG_HISPARSE_SKIP_IO is unsupported on XPU: the AOT swap-in "
            "ops do not accept skip_io, so timings would silently include the KV "
            "copy."
        )
    self.compress_ratio = self.token_to_kv_pool_allocator.compress_ratio

    kvcache = self.token_to_kv_pool_allocator.get_kvcache()
    self.is_dsv4_hisparse = isinstance(
        self.token_to_kv_pool_allocator, DeepSeekV4HiSparseTokenToKVPoolAllocator
    )
    self.is_m3_hisparse = isinstance(kvcache, MiniMaxSparseKVPool)
    self.m3_flashmla_verify = None
    use_m3_flashmla = self.is_m3_hisparse
    if self.is_dsv4_hisparse:
        self.mem_pool_device = self.token_to_kv_pool_allocator.hisparse_kvcache
        page_size = self.mem_pool_device.page_size
        num_host_pages = (
            self.token_to_kv_pool_allocator.size_full // self.compress_ratio
            + page_size
            - 1
        ) // page_size
        self.mem_pool_host = DeepSeekV4PagedHostPool(
            pool_name="dsv4_hisparse_c4",
            device_buffers=self.mem_pool_device.kv_buffer,
            item_bytes=self.mem_pool_device.bytes_per_page_padded,
            num_host_pages=num_host_pages,
            slot_page_size=page_size,
            layout="layer_first",
        )
        self.item_size_bytes = (
            self.mem_pool_device.kv_cache_total_dim
            * self.mem_pool_device.store_dtype.itemsize
        )
    else:
        assert isinstance(
            self.token_to_kv_pool_allocator, HiSparseTokenToKVPoolAllocator
        )
        if self.is_m3_hisparse:
            self.mem_pool_device = kvcache.main_pool
            expected_heads = 4 if use_m3_flashmla else 1
            if self.mem_pool_device.head_num != expected_heads:
                raise ValueError(
                    f"MiniMax HiSparse requires {expected_heads} local KV heads "
                    f"for this backend, got {self.mem_pool_device.head_num}."
                )
            if use_m3_flashmla and (
                num_draft_tokens != 8
                or self.mem_pool_device.page_size != 128
                or self.mem_pool_device.head_dim != 128
                or self.mem_pool_device.v_head_dim != 128
                or kvcache.dense_pool is not None
                or kvcache.index_kv_pool is not None
            ):
                raise ValueError(
                    "MiniMax FlashMLA HiSparse requires MSA128 verify8 "
                    "with K-only indexers on all layers"
                )
            self.item_size_bytes = self.mem_pool_device.bytes_per_token_k
        else:
            self.mem_pool_device: HiSparseDSATokenToKVPool = kvcache
            self.mem_pool_host = MLATokenToKVPoolHost(
                buffer_info=self.mem_pool_device.get_device_pool_infos()[
                    0
                ].buffer_info,
                device_capacity_tokens=self.mem_pool_device.size
                if self.mem_pool_device.host_capacity_tokens is None
                else self.mem_pool_device.host_capacity_tokens,
                start_layer=self.mem_pool_device.start_layer,
                end_layer=self.mem_pool_device.end_layer,
                target_layer_num=self.mem_pool_device.layer_num,
                host_to_device_ratio=host_to_device_ratio,
                host_size=0,
                page_size=self.mem_pool_device.page_size,
                layout="layer_first",
                override_kv_cache_dim=self.mem_pool_device.kv_cache_dim,
            )
            self.item_size_bytes = self.mem_pool_host.token_stride_size
    self.page_size = self.mem_pool_device.page_size

    max_num_req_slots = req_to_token_pool.req_to_token.shape[0]
    max_context_len = req_to_token_pool.max_context_len
    max_compressed_context_len = (
        max_context_len + self.compress_ratio - 1
    ) // self.compress_ratio

    # to have an extra page for new tokens
    self.padded_buffer_size = (
        self.device_buffer_size + self.mem_pool_device.page_size
    )

    self.req_to_device_buffer = torch.zeros(
        (max_num_req_slots, self.padded_buffer_size),
        dtype=torch.int64,
        device=device,
    )
    self.req_device_buffer_size = torch.zeros(
        max_num_req_slots, dtype=torch.int64, device="cpu"
    )
    self._whole_page_device_buffers: set[int] = set()
    self.req_to_host_pool = torch.full(
        (max_num_req_slots, max_compressed_context_len + self.page_size),
        -1,
        dtype=torch.int64,
        device=device,
    )
    self.req_to_host_pool_allocated_len = torch.zeros(
        max_num_req_slots, dtype=torch.int64, device="cpu"
    )

    self.write_staging_stream = device_module.Stream()
    self.decode_backup_stream = device_module.Stream()
    self.ack_staging_queue: List[HiSparseAct] = []
    self.decode_producer_stream = None
    self._backup_done_event = device_module.Event()
    self._has_pending_backup = False

    self.tp_group = tp_group
    self.tp_world_size = torch.distributed.get_world_size(group=self.tp_group)

    # initialize data structures for swap-in kernel
    layer_num = self.mem_pool_device.layer_num
    token_metadata_width = 0 if use_m3_flashmla else self.padded_buffer_size
    self.req_device_buffer_tokens = torch.full(
        (layer_num, max_num_req_slots, token_metadata_width),
        -1,
        dtype=torch.int32,
        device=device,
    )
    self.req_device_buffer_token_locs = torch.full(
        (layer_num, max_num_req_slots, token_metadata_width),
        -1,
        dtype=torch.int32,
        device=device,
    )
    # The MSA adapter keeps page labels and does not use token-level LRU.
    self._lru_init = torch.arange(
        0 if use_m3_flashmla else self.device_buffer_size,
        dtype=torch.int16,
        device=device,
    )
    self.lru_slots = (
        self._lru_init.view(1, 1, -1)
        .repeat(layer_num, max_num_req_slots, 1)
        .contiguous()
    )
    self.spec_swap = HiSparseSpecSwapManager(
        self,
        num_draft_tokens=0 if use_m3_flashmla else num_draft_tokens,
    )
    if use_m3_flashmla:
        from hcu_sglang.layers.attention.minimax_hisparse import (
            MiniMaxHiSparseVerify,
        )

        self.m3_flashmla_verify = MiniMaxHiSparseVerify(
            self, max_num_req_slots, layer_num
        )
    if isinstance(self.token_to_kv_pool_allocator, HiSparseTokenToKVPoolAllocator):
        self.token_to_kv_pool_allocator.configure_spec_scratch(
            self.spec_swap.scratch_capacity,
            device_buffer_size=self.padded_buffer_size,
        )
    self._device_buffer_arange_i32 = torch.arange(
        0 if use_m3_flashmla else self.device_buffer_size,
        dtype=torch.int32,
        device=device,
    )

    # Pre-allocated output buffer for swap-in (CUDA-graph safe). MiniMax
    # selects blocks, so its flattened token-slot output can occupy any
    # prefix up to the full device working-set size.
    swap_output_width = (
        0
        if use_m3_flashmla
        else (self.device_buffer_size if self.is_m3_hisparse else self.top_k)
    )
    self.top_k_device_locs_buffer = torch.full(
        (max_num_req_slots, swap_output_width),
        -1,
        dtype=torch.int32,
        device=device,
    )
    self.raw_indices_buffer = torch.full(
        (max_num_req_slots, self.top_k), -1, dtype=torch.int32, device=device
    )
    # Scalar tensor: number of real (non-padded) requests in the batch.
    # Updated before each graph replay so padded blocks early-return.
    self.num_real_reqs = torch.zeros(1, dtype=torch.int32, device=device)

    # CPU flag: True means "skip backup on the next decode step" because
    # staging already backed up all prefill tokens.  Cleared after one step.
    self._skip_first_backup = [False] * max_num_req_slots

    self._init_shared_index_prefetch(
        shared_index_layers=shared_index_layers,
        layer_num=layer_num,
        max_num_req_slots=max_num_req_slots,
    )

    if self.is_m3_hisparse:
        device_module.synchronize()
        init_minimax_host_pool(self, host_to_device_ratio)
    if self.m3_flashmla_verify is not None:
        self.m3_flashmla_verify.bind_host_commit_buffers()


def init_minimax_host_pool(self, host_to_device_ratio: float) -> None:
    # Reserve GPU metadata before registering the large host pool.
    self.mem_pool_host = HiSparseMHATokenToKVPoolHost(
        buffer_info=self.mem_pool_device.get_device_pool_infos()[0].buffer_info,
        device_capacity_tokens=self.mem_pool_device.size
        if self.mem_pool_device.host_capacity_tokens is None
        else self.mem_pool_device.host_capacity_tokens,
        start_layer=self.mem_pool_device.start_layer,
        end_layer=self.mem_pool_device.end_layer,
        host_to_device_ratio=host_to_device_ratio,
        page_size=self.mem_pool_device.page_size,
    )
    self.mem_pool_host.pin_slot_indices = (
        os.environ.get("M31_HISPARSE_PIN_SLOT_INDICES") == "1"
    )
    # Fake PD never initializes the handoff KV. Seed a known,
    # finite host cache for the isolated capacity validation.
    if os.environ.get("M31_HISPARSE_FAKE_VALIDATE") == "1":
        self.mem_pool_host.kv_buffer.zero_()


def alloc_paged_token_slots(
    self,
    req_to_host_pool: torch.Tensor,
    req_to_host_pool_allocated_len: torch.Tensor,
    req_pool_idx: int,
    start_pos: int,
    num_tokens: int,
) -> torch.Tensor:
    """Allocate request host slots by page and return token-granular slots."""
    device = req_to_host_pool.device
    if num_tokens <= 0:
        return torch.empty((0,), dtype=torch.int64, device=device)

    allocated_len = int(req_to_host_pool_allocated_len[req_pool_idx])
    end_pos = start_pos + num_tokens
    page_end = self._round_up_to_page_size(end_pos)
    assert start_pos <= allocated_len

    if page_end > allocated_len:
        num_new_pages = (page_end - allocated_len) // self.page_size
        host_locs = self.alloc_page(num_new_pages)
        if host_locs is None:
            logger.error(
                "HiSparse: host mem pool alloc failed for %d host pages "
                "(req_pool_idx=%d, start_pos=%d, num_tokens=%d)",
                num_new_pages,
                req_pool_idx,
                start_pos,
                num_tokens,
            )
            raise RuntimeError(
                f"HiSparse host mem pool alloc failed for {num_new_pages} pages"
            )

        destination = req_to_host_pool[req_pool_idx, allocated_len:page_end]
        if device.type == "cuda":
            # Pool allocation returns ordinary CPU indices. Use pinned
            # storage for the async copy directly into the destination row.
            destination.copy_(host_locs.pin_memory(), non_blocking=True)
        else:
            destination[...] = host_locs.to(device=device, non_blocking=True)
        req_to_host_pool_allocated_len[req_pool_idx] = page_end

    return req_to_host_pool[req_pool_idx, start_pos:end_pos]


def admit_request_direct(self, req: Req) -> None:
    """Direct-to-host path: KV data already resides in host pool via RDMA.

    Skips staging DMA entirely. Only allocates a small device buffer
    (4KB) for decode-time swap-in, then marks the request as ready.
    Host indices were already written to req_to_host_pool.

    Metadata fixups after alloc_device_buffer():
    - alloc_device_buffer() sets device_buffer_tokens = [0, 1, ..., buf_size-1],
      which tells the swap-in kernel that those tokens are cached in the device
      buffer.  In the staging path this is correct (prefill filled the buffer),
      but here the buffer is empty.
    """
    self.alloc_device_buffer(req)

    host_len = self.host_token_len(req.kv.kv_allocated_len)
    if self.m3_flashmla_verify is not None:
        self.m3_flashmla_verify.reset(req.kv.req_pool_idx)
    elif host_len <= self.device_buffer_size:
        # Short sequences (seq_len <= device_buffer_size): the kernel fast path
        # returns device_buffer_locs directly without any host loading, so we
        # must preload all tokens from host pool into the device buffer
        # TODO(hzh0425): Optimize this.
        self._preload_to_device_buffer(req)
    else:
        # Long sequence: reset device_buffer_tokens to -1 so the kernel
        # sees all slots as empty -> every top-k lookup is a miss -> host load.
        self.req_device_buffer_tokens[
            :, req.kv.req_pool_idx, : self.device_buffer_size
        ] = -1
        self.spec_swap.invalidate_cache(req.kv.req_pool_idx)

    req.hisparse_staging = False
    self._skip_first_backup[req.kv.req_pool_idx] = True
    logger.debug("HiSparse: admitting request %s directly", req.rid)


def alloc_device_buffer_with_layout(self, allocated_indices, need_size: int) -> tuple[torch.Tensor, bool]:
    import torch

    assert need_size % self.page_size == 0
    # clear original reference and isolate the buffer from outside addressing, allocate new buffer if needed
    hisparse_indices = self.full_to_hisparse_device_index_mapping[allocated_indices]
    self.full_to_hisparse_device_index_mapping[allocated_indices] = 0
    # Zero means unmapped; after alloc_logical_only the mapping is all zeros.
    hisparse_indices = hisparse_indices[hisparse_indices > 0]
    owns_whole_pages = hisparse_indices.numel() == 0
    if len(hisparse_indices) >= need_size:
        buffer_indices = hisparse_indices[:need_size]
        self.free_hisparse_indices(hisparse_indices[need_size:])
    else:
        # page alignment, claiming the residual space for an incomplete page
        page_residual_length = len(hisparse_indices) % self.page_size
        if page_residual_length != 0:
            hisparse_indices = torch.cat(
                [
                    hisparse_indices,
                    torch.arange(
                        hisparse_indices[-1] + 1,
                        hisparse_indices[-1]
                        + self.page_size
                        - page_residual_length
                        + 1,
                        device=self.device,
                    ),
                ]
            )
        extra_indices = self.hisparse_attn_allocator.alloc(
            need_size - len(hisparse_indices)
        )
        assert extra_indices is not None, (
            "Hisparse allocation failed in alloc_device_buffer"
        )
        buffer_indices = torch.cat([hisparse_indices, extra_indices])
    return buffer_indices, owns_whole_pages


def alloc_device_buffer(self, req: Req) -> None:
    if self.is_dsv4_hisparse:
        allocated_len = req.extend_end
        alloc_size = self.padded_buffer_size
    else:
        allocated_len = req.kv.kv_allocated_len
        page_size = self.mem_pool_device.page_size
        # Allocate only enough for current tokens (page-aligned).
        # When prefill already fills device_buffer_size, include the reserved page.
        alloc_size = min(
            ((allocated_len + page_size - 1) // page_size) * page_size,
            self.device_buffer_size,
        )
        # Spec verify needs the complete, graph-stable buffer even for short prompts.
        if (
            self.spec_swap.enabled
            or self.m3_flashmla_verify is not None
            or alloc_size == self.device_buffer_size
        ):
            alloc_size = self.padded_buffer_size

    compressed_logical_indices = (
        self.mem_pool_device.translate_loc_from_full_to_compressed(
            self.req_to_token_pool.req_to_token[req.kv.req_pool_idx, :allocated_len]
        )
    )
    compressed_len = len(compressed_logical_indices)

    allocator = self.token_to_kv_pool_allocator
    if self.m3_flashmla_verify is not None:
        buffer_indices, owns_whole_pages = alloc_device_buffer_with_layout(
            allocator, compressed_logical_indices, alloc_size
        )
        if owns_whole_pages:
            self._whole_page_device_buffers.add(req.kv.req_pool_idx)
    else:
        buffer_indices = allocator.alloc_device_buffer(
            compressed_logical_indices, alloc_size
        )
    if buffer_indices is None:
        logger.error(
            "HiSparse: alloc_device_buffer failed for req %s "
            "(compressed_len=%d, alloc_size=%d)",
            req.rid,
            compressed_len,
            alloc_size,
        )
        raise RuntimeError("HiSparse alloc_device_buffer returned None")

    buffer_indices = buffer_indices.to(torch.int32)
    self.req_to_device_buffer[req.kv.req_pool_idx, :alloc_size] = buffer_indices
    self.req_device_buffer_size[req.kv.req_pool_idx] = alloc_size

    if self.m3_flashmla_verify is None:
        self.req_device_buffer_tokens[
            :, req.kv.req_pool_idx, : self.device_buffer_size
        ] = self._device_buffer_arange_i32
        self.req_device_buffer_token_locs[:, req.kv.req_pool_idx, :alloc_size] = (
            buffer_indices[:alloc_size]
        )
    self.spec_swap.allocate_scratch(req.kv.req_pool_idx)
    self.spec_swap.reset(req.kv.req_pool_idx)


def request_finished(self, req: Req):
    # release resources only after the execution of a potential overlapped batch
    if self.decode_producer_stream is not None:
        device_module.current_stream().wait_stream(self.decode_producer_stream)
    self.wait_for_pending_backup()

    # Use kv_allocated_len (not seqlen): under speculative decoding the
    # allocator can over-allocate beyond the committed seqlen, and those
    # extra slots may carry stale mapping entries pointing at buffer slots
    # we just freed via free_hisparse_indices(all_hi). If left set, the
    # subsequent release_kv_cache -> allocator.free -> free_hisparse path
    # re-frees them (double-free into the page allocator's free list).
    allocated_len = req.kv.kv_allocated_len
    req_pool_idx = req.kv.req_pool_idx
    allocated_locs = self.req_to_token_pool.req_to_token[
        req_pool_idx, :allocated_len
    ]

    # The spec commit kernel rotates physical locations between each
    # layer's persistent cache and scratch.  Their current union is the
    # request's allocation ownership; the original side-buffer row is no
    # longer authoritative after the first rotation.
    current_cap = int(self.req_device_buffer_size[req_pool_idx])
    if current_cap > 0:
        if self.m3_flashmla_verify is not None:
            owned_locs = self.req_to_device_buffer[req_pool_idx, :current_cap]
        else:
            owned_locs = self.req_device_buffer_token_locs[
                :, req_pool_idx, :current_cap
            ].reshape(-1)
            owned_locs = self.spec_swap.extend_owned_locs(req_pool_idx, owned_locs)
        allocator = self.token_to_kv_pool_allocator
        if req_pool_idx in self._whole_page_device_buffers:
            # MiniMax keeps ownership in req_to_device_buffer. Paging only
            # changes the per-head slot mapping, not the reserved page IDs.
            page_ids = owned_locs[:: allocator.page_size] // allocator.page_size
            allocator.hisparse_attn_allocator.free_page_ids(
                page_ids.to(dtype=torch.int64).sort().values
            )
            self._whole_page_device_buffers.remove(req_pool_idx)
        else:
            all_hi = torch.unique(owned_locs[owned_locs > 0])
            if all_hi.numel() > 0:
                allocator.free_hisparse_indices(all_hi)

        compressed_locs = (
            self.mem_pool_device.translate_loc_from_full_to_compressed(
                allocated_locs
            )
        )
        self.mem_pool_device.full_to_hisparse_device_index_mapping[
            compressed_locs
        ] = 0
    else:
        # The request can finish or be aborted during prefill, before a
        # device side buffer exists.  Its ordinary prefill mapping is still
        # owned by release_kv_cache; only release our separate workspace.
        self.spec_swap.free_unrotated_scratch(req_pool_idx)

    host_indices = self.mem_pool_host.allocated_host_indices(
        self.req_to_host_pool,
        req.kv.req_pool_idx,
        self.req_to_host_pool_allocated_len[req.kv.req_pool_idx],
    )
    if host_indices.numel() > 0:
        self.mem_pool_host.free(host_indices)

    # clear req info
    self.req_device_buffer_tokens[:, req.kv.req_pool_idx, :] = -1
    self.req_device_buffer_token_locs[:, req.kv.req_pool_idx, :] = -1
    self.req_to_device_buffer[req.kv.req_pool_idx, :] = 0
    self.req_device_buffer_size[req.kv.req_pool_idx] = 0
    self.req_to_host_pool[req.kv.req_pool_idx, :] = -1
    self.req_to_host_pool_allocated_len[req.kv.req_pool_idx] = 0
    self.lru_slots[:, req.kv.req_pool_idx, :].copy_(self._lru_init)
    self.spec_swap.clear_scratch(req_pool_idx)
    if self.m3_flashmla_verify is not None:
        self.m3_flashmla_verify.reset(req_pool_idx)
    self._skip_first_backup[req.kv.req_pool_idx] = False


def host_transfer_views(self, buffer_info, write_back_staging) -> None:
    self.element_dim = self.head_num * self.head_dim
    self._init_device_row_stride((*self.device_k_buffers, *self.device_v_buffers))
    # The JIT HiCache kernels also build with hipcc (ROCm): the PTX-only
    # helpers in hicache.cuh are guarded by USE_ROCM and the staged
    # write-back kernel has a ROCm path, so enable them on HIP too. This
    # keeps the ROCm write-back path consistent with CUDA.
    self.can_use_jit = (_is_cuda or _is_hip) and can_use_hicache_jit_kernel(
        page_size=self.page_size,
        element_size=self.element_dim * self.dtype.itemsize,
    )

    if self.layout == "page_unified":
        # No per-component per-layer view exists: a layer's K and V are
        # interleaved inside each head group's block. The page_unified
        # transfer kernels address the fused buffer directly instead.
        self.k_data_refs = []
        self.v_data_refs = []
    elif self.layout in ("page_first", "page_first_direct"):
        # Transpose [page, layer, ...] -> [layer, page, ...] to get per-layer views
        # This swaps strides without copying data
        k_transposed = self.k_buffer.transpose(0, 1)
        v_transposed = self.v_buffer.transpose(0, 1)
        self.k_data_refs = [k_transposed[i] for i in range(self.layer_num)]
        self.v_data_refs = [v_transposed[i] for i in range(self.layer_num)]
    else:
        self.k_data_refs = [self.k_buffer[i] for i in range(self.layer_num)]
        self.v_data_refs = [self.v_buffer[i] for i in range(self.layer_num)]
    self.k_data_ptrs = None
    self.v_data_ptrs = None
    self.host_kv_data_refs = self.k_data_refs + self.v_data_refs
    self._init_write_back_staging_buffers(buffer_info, write_back_staging)


def host_init_kv_buffer(self):
    if self.layout == "layer_first":
        dims = (2, self.layer_num, self.size, self.head_num, self.head_dim)
    elif self.layout == "page_first":
        dims = (2, self.size, self.layer_num, self.head_num, self.head_dim)
    elif self.layout == "page_first_direct":
        dims = (
            2,
            self.page_num,
            self.layer_num,
            self.page_size,
            self.head_num,
            self.head_dim,
        )
    elif self.layout == "page_head":
        dims = (
            2,
            self.page_num,
            self.head_num,
            self.page_size,
            self.layer_num,
            self.head_dim,
        )
    elif self.layout == "page_unified":
        self.page_unified_layout = PageUnifiedLayout(
            page_size=self.page_size,
            layer_num=self.layer_num,
            head_num=self.head_num,
            head_group_num=self.head_group_num,
            head_dim=self.head_dim,
            itemsize=self.dtype.itemsize,
        )
        dims = self.page_unified_layout.page_dims(self.page_num)
    else:
        raise ValueError(f"Unsupported layout: {self.layout}")
    self.token_stride_size = self.head_num * self.head_dim * self.dtype.itemsize
    self.layout_dim = self.token_stride_size * self.layer_num

    allocator = self.allocator
    if (
        is_hcu()
        and self.pin_memory
        and os.getenv("SGLANG_HISPARSE_HCU_PIN_HOST_KV", "0") == "1"
    ):
        from hcu_sglang.models._minimax_rocm_pinned_kv import RocmPinnedKVAllocator

        allocator = RocmPinnedKVAllocator()

    alloc_func = ALLOC_MEMORY_FUNCS[self._transfer_device.type]
    buffer = alloc_func(
        dims,
        dtype=self.dtype,
        device=self.device,
        pin_memory=self.pin_memory,
        allocator=allocator,
        registration_granularity_bytes=(
            # page_unified's block carries both components, so its page is
            # twice page_size * layout_dim.
            self.page_unified_layout.bytes_per_page
            if self.layout == "page_unified"
            else self.page_size * self.layout_dim
            if self.layout in ("page_first", "page_first_direct")
            else self.size * self.token_stride_size
            if self.layout == "layer_first" and True
            else None
        ),
    )
    return buffer


def commit_layer_ranges(self) -> list[tuple[int, int]]:
    # Each K and V slice passed to a fused commit must use one HIP mapping.
    from sglang.srt.mem_cache.pool_host.common import _CUDA_HOST_REGISTERED_RANGES_ATTR

    ranges = getattr(self.kv_buffer, _CUDA_HOST_REGISTERED_RANGES_ATTR, ())
    layer_bytes = self.k_data_refs[0].nbytes
    base = self.kv_buffer.data_ptr()
    cuts = {0, self.layer_num}
    for ptr, _ in ranges[1:]:
        layer, remainder = divmod(ptr - base, layer_bytes)
        if remainder:
            raise RuntimeError("HiSparse host registration splits a K/V layer")
        cuts.add(layer % self.layer_num)
    boundaries = sorted(cuts)
    return list(zip(boundaries, boundaries[1:]))


def allocator_init(
    self,
    size: int,
    page_size: int,
    dtype: torch.dtype,
    device: torch.device,
    kvcache: HiSparseDSATokenToKVPool | MiniMaxSparseKVPool,
    need_sort: bool,
    host_to_device_ratio: float = 2,
):
    self._kvcache = kvcache
    self._size_full = int(size * host_to_device_ratio) // page_size * page_size
    self._size_hisparse = size
    self.compress_ratio = 1
    self.dtype = dtype
    self.device = device
    self.page_size = page_size
    self.need_sort = need_sort
    self._spec_scratch_capacity = 0
    self._spec_device_buffer_size = 0

    paged_allocator_cls = (
        current_platform.get_paged_allocator_cls() or PagedTokenToKVPoolAllocator
    )
    self.logical_attn_allocator = paged_allocator_cls(
        self._size_full,
        self.page_size,
        self.dtype,
        self.device,
        kvcache,
        need_sort,
    )
    self.hisparse_attn_allocator = paged_allocator_cls(
        self._size_hisparse,
        self.page_size,
        self.dtype,
        self.device,
        kvcache,
        need_sort,
    )
    self.full_to_hisparse_device_index_mapping = torch.cat(
        [
            torch.zeros(
                self._size_full + self.page_size,
                dtype=torch.int64,
                device=self.device,
            ),
            torch.tensor([-1], dtype=torch.int64, device=self.device),
        ]
    )

    self.free_pages = None
    self.release_pages = None
    self.free_group = None
    self.clear()
    self._kvcache.register_mapping(
        weakref.proxy(self.full_to_hisparse_device_index_mapping)
    )


def minimax_pool_init(
    self,
    size: int,
    page_size: int,
    dtype: torch.dtype,
    head_num: int,
    head_dim: int,
    idx_head_dim: int,
    dense_layer_ids: List[int],
    sparse_layer_ids: List[int],
    device: str,
    disable_value_sparse_layer_ids: Optional[List[int]] = None,
    enable_memory_saver: bool = False,
    index_dtype: Optional[torch.dtype] = None,
    start_layer: Optional[int] = None,
    end_layer: Optional[int] = None,
    main_pool_cls=None,
    index_kv_pool_cls=None,
    index_k_pool_cls=None,
    enable_hisparse: bool = False,
    host_to_device_ratio: int = 2,
):
    # Do not call super().__init__() — delegate to sub-pools instead.
    self.size = size
    self.page_size = page_size
    self.dtype = dtype
    self.device = device
    self.use_minimax_fused_kv_index_store = (
        envs.SGLANG_OPT_USE_MINIMAX_FUSED_KV_INDEX_STORE.get()
    )

    local_dense_layer_ids = [
        lid for lid in dense_layer_ids if start_layer <= lid < end_layer
    ]
    local_sparse_layer_ids = [
        lid for lid in sparse_layer_ids if start_layer <= lid < end_layer
    ]

    index_dtype = index_dtype if index_dtype is not None else dtype
    index_pool_size = int(size * host_to_device_ratio) // page_size * page_size if enable_hisparse else size

    # Split sparse layers by V policy: kv_sparse (index_kv_pool holds K+V) vs
    # k_only_sparse (index_k_pool holds only K; V is never read).
    disable_set = set(disable_value_sparse_layer_ids or [])
    local_kv_sparse_layer_ids = [
        g for g in local_sparse_layer_ids if g not in disable_set
    ]
    local_k_only_sparse_layer_ids = [
        g for g in local_sparse_layer_ids if g in disable_set
    ]

    # Membership check across all sparse layers, regardless of split.
    self.sparse_layer_id_mapping: dict[int, int] = {
        gid: i for i, gid in enumerate(local_sparse_layer_ids)
    }
    # Per-sub-pool local indices.
    self.index_kv_layer_id_mapping: dict[int, int] = {
        gid: i for i, gid in enumerate(local_kv_sparse_layer_ids)
    }
    self.index_k_layer_id_mapping: dict[int, int] = {
        gid: i for i, gid in enumerate(local_k_only_sparse_layer_ids)
    }

    self._dense_layer_ids = set(local_dense_layer_ids)
    main_layer_num = len(local_dense_layer_ids) + len(local_sparse_layer_ids)
    if enable_hisparse:
        from sglang.srt.mem_cache.hisparse_memory_pool import (
            HiSparseMHAMainPool,
        )

        self.dense_pool = (
            main_pool_cls(
                size=index_pool_size,
                page_size=page_size,
                dtype=dtype,
                head_num=head_num,
                head_dim=head_dim,
                layer_num=len(local_dense_layer_ids),
                device=device,
                enable_memory_saver=enable_memory_saver,
                start_layer=start_layer,
                end_layer=start_layer + len(local_dense_layer_ids),
            )
            if local_dense_layer_ids
            else None
        )
        self.main_pool = HiSparseMHAMainPool(
            size=size,
            page_size=page_size,
            dtype=dtype,
            head_num=head_num,
            head_dim=head_dim,
            layer_num=len(local_sparse_layer_ids),
            device=device,
            enable_memory_saver=enable_memory_saver,
            start_layer=local_sparse_layer_ids[0],
            end_layer=end_layer,
        )
    else:
        self.dense_pool = None
        self.main_pool = main_pool_cls(
            size=size,
            page_size=page_size,
            dtype=dtype,
            head_num=head_num,
            head_dim=head_dim,
            layer_num=main_layer_num,
            device=device,
            enable_memory_saver=enable_memory_saver,
            start_layer=start_layer,
            end_layer=end_layer,
        )

    self.index_kv_pool: Optional[MHATokenToKVPool] = (
        index_kv_pool_cls(
            size=index_pool_size,
            page_size=page_size,
            dtype=index_dtype,
            head_num=1,
            head_dim=idx_head_dim,
            layer_num=len(local_kv_sparse_layer_ids),
            device=device,
            enable_memory_saver=enable_memory_saver,
        )
        if local_kv_sparse_layer_ids
        else None
    )

    self.index_k_pool: Optional[MHATokenToKOnlyPool] = (
        index_k_pool_cls(
            size=index_pool_size,
            page_size=page_size,
            dtype=index_dtype,
            head_num=1,
            head_dim=idx_head_dim,
            layer_num=len(local_k_only_sparse_layer_ids),
            device=device,
            enable_memory_saver=enable_memory_saver,
        )
        if local_k_only_sparse_layer_ids
        else None
    )

    if enable_hisparse:
        self.main_pool.model_layer_ids = tuple(local_sparse_layer_ids)
        if self.dense_pool is not None:
            self.dense_pool.model_layer_ids = tuple(local_dense_layer_ids)
    if self.index_kv_pool is not None:
        self.index_kv_pool.model_layer_ids = tuple(local_kv_sparse_layer_ids)

    self.mem_usage = self.main_pool.mem_usage
    if self.dense_pool is not None:
        self.mem_usage += self.dense_pool.mem_usage
    if self.index_kv_pool is not None:
        self.mem_usage += self.index_kv_pool.mem_usage
    if self.index_k_pool is not None:
        self.mem_usage += self.index_k_pool.mem_usage

    # HiCacheController reads these from the top-level KV pool wrapper.
    self.layer_num = main_layer_num
    self.start_layer = start_layer
    self.end_layer = end_layer
    # PD disaggregation reads these directly (no fallback) off the wrapper.
    self.head_num = self.main_pool.head_num
    self.head_dim = self.main_pool.head_dim
    self.v_head_dim = self.main_pool.v_head_dim
    self.store_dtype = self.main_pool.store_dtype
    self.layer_transfer_counter = None
    self._enable_hisparse = enable_hisparse


def dspark_decode(
    self,
    batch: ScheduleBatch,
    on_publish,
    grammar_barrier=None,
    coordination_plan=None,
) -> GenerationBatchResult:
    if batch.spec_info is None:
        batch.spec_info = DFlashDraftInputV2.create_idle_input(device=self.device)
    draft_input = batch.spec_info
    if not isinstance(draft_input, DFlashDraftInputV2):
        raise RuntimeError(
            "DSpark spec-v2 expected DFlashDraftInputV2 state on the running batch."
        )

    global_num_reqs = (
        max(batch.global_num_tokens)
        if self._draft_is_moe
        and get_parallel().attn_dp_enabled
        and batch.global_num_tokens is not None
        else None
    )
    if coordination_plan is not None:
        rank = get_parallel().attn_dp_rank
        target_local_only = len(batch.global_num_tokens) == 1
        coordination_plan.apply(
            batch,
            "draft",
            rank,
            local_only=target_local_only or self.draft_parallel_layout.local_dp,
        )

    if batch.forward_mode.is_idle():
        self._observers.note_idle_decode_step()
        if get_parallel().attn_dp_enabled:
            if self._draft_is_moe:
                self._proposer.run_idle_participation(batch)
            if coordination_plan is not None:
                coordination_plan.apply(
                    batch, "target", rank, local_only=target_local_only
                )
            self._verify_executor.run_idle_participation(
                batch=batch, idle_layout=self._idle_verify_ragged_layout(batch)
            )
        return self._decode_idle_result(on_publish=on_publish)

    batch.seq_lens.record_stream(
        torch.get_device_module(self.device).current_stream()
    )
    bs = len(batch.seq_lens)
    device = self.device
    prefix_lens = batch.seq_lens

    self._observers.begin_step()

    target_model = self.target_worker.model_runner.model
    anchor_ids = (
        self._proposer.draft_block_ids_view(bs=bs, device=prefix_lens.device)
        if self._prepare_inputs_once
        else None
    )
    from hcu_sglang.models.minimax_m31_seq_lens import consume_published_lengths

    consume_published_lengths(batch)
    window_seq_lens_cpu = (
        batch.seq_lens_cpu
        if batch.seq_lens_cpu is not None
        else draft_input.nxt_kv_lens_cpu
    )
    verify_window = alloc_verify_window(
        batch=batch,
        bs=bs,
        device=device,
        verify_num_draft_tokens=self.verify_num_draft_tokens,
        block_pos_offsets=self._block_pos_offsets,
        model_runner=self.model_runner,
        seq_lens_cpu=window_seq_lens_cpu,
        bonus=draft_input.bonus_tokens.view(-1) if anchor_ids is not None else None,
        anchor_ids=anchor_ids,
    )
    coordinator = self.model_runner.hisparse_coordinator
    hisparse_verify = None if coordinator is None else coordinator.m3_flashmla_verify
    if hisparse_verify is not None:
        hisparse_verify.prepare(batch, verify_window.verify_cache_loc)
    if self._verify_inputs_ready is not None:
        # The verify's inputs are final here; the early metadata waits for this.
        self._verify_inputs_ready.record()

    sampling_info = batch.sampling_info
    # Simulated acceptance overrides correct_len, so only the bonus token needs the
    # request's temperature: draft + accept run greedy, the bonus is still sampled.
    simulate_bonus_sampling_info = None
    if (
        self._simulate_acc_greedy
        and sampling_info is not None
        and not sampling_info.is_all_greedy
        and not batch.has_grammar
        and verify_logits_adjustments_are_noop(sampling_info)
        and not sampling_info.need_top_p_sampling
        and not sampling_info.need_top_k_sampling
        and not sampling_info.need_min_p_sampling
    ):
        simulate_bonus_sampling_info = sampling_info
        sampling_info = None
    if self._sim_draw_ahead:
        self._begin_simulated_step_ahead(
            bs=bs, simulate_bonus_sampling_info=simulate_bonus_sampling_info
        )
    with self._draft_context(), self._observers.segment(InfoSegment.DRAFT):
        proposal = self._proposer.propose(
            batch=batch,
            draft_input=draft_input,
            verify_window=verify_window,
            bs=bs,
            device=device,
            target_model=target_model,
            sampling_info=sampling_info,
            anchor_prepared=anchor_ids is not None,
        )
    if self._verify_inputs_ready is not None:
        self._stage_target_verify_metadata(
            batch=batch,
            verify_window=verify_window,
            seq_lens_cpu=window_seq_lens_cpu,
        )
    draft_block_ids = proposal.draft_block_ids
    draft_block = proposal.draft_block
    draft_tokens = draft_block.draft_tokens
    if coordination_plan is not None:
        coordination_plan.apply(batch, "target", rank, local_only=target_local_only)

    confidence = proposal.confidence

    verify_token_budget = self._verify_planner.resolve_verify_token_budget(
        draft_input=draft_input,
        confidence=confidence,
        prefix_lens=prefix_lens,
        req_pool_indices=batch.req_pool_indices,
    )

    layout = self._verify_planner.schedule_layout(
        req_pool_indices=batch.req_pool_indices,
        prefix_lens=prefix_lens,
        device=device,
        confidence=confidence,
        budget=verify_token_budget,
        global_num_reqs=global_num_reqs,
        dp_tier_num_tokens=self._dp_verify_tier_num_tokens(batch),
    )
    run_compact = self._verify_planner.should_run_compact(layout=layout)

    verify_ids_2d = proposal.verify_ids
    if verify_ids_2d is None:
        verify_ids_2d = torch.cat(
            [draft_block_ids[:, :1], draft_tokens], dim=1
        ).contiguous()
    else:
        self.num_folded_verify_ids_steps += 1
        if self.num_folded_verify_ids_steps == 1:
            logger.info("Verify ids read from the draft graph output (first step).")

    # Must stay ahead of the target verify launch below.
    grammar_tree = (
        GrammarTree.from_linear_chain(verify_ids_2d) if batch.has_grammar else None
    )

    # A live grammar forces the eager path: the folded epilogue accepts inside
    # the cuda graph off its own buffers, where the mask below never lands.
    fold_eligible = (
        self._verify_executor.verify_epilogue is not None
        and proposal.folded
        # The epilogue's in-graph accept is greedy (AcceptGreedy);
        # sampling batches must take the eager accept path even when the
        # draft proposal itself folded.
        and (sampling_info is None or sampling_info.is_all_greedy)
        and verify_logits_adjustments_are_noop(sampling_info)
        # The graph accept is greedy: a sampled simulated bonus stays in the graph
        # only when the epilogue draws it there (SGLANG_DSPARK_SIMULATE_FOLD).
        and (
            simulate_bonus_sampling_info is None
            or self._verify_executor.verify_epilogue.simulate_fold
        )
        and not batch.has_grammar
    )
    if not self._sim_draw_ahead:
        self._verify_executor.begin_simulated_step()
        if fold_eligible and self._verify_executor.verify_epilogue.simulate_fold:
            self._verify_executor.verify_epilogue.begin_sim_bonus(
                bs,
                (
                    None
                    if simulate_bonus_sampling_info is None
                    else simulate_bonus_sampling_info.temperatures[:bs]
                ),
            )
    prepare_mamba_track_for_verify(batch)
    with self._observers.segment(InfoSegment.TARGET_VERIFY):
        if run_compact:
            target_verify, hidden_strided = self._verify_executor.run_compact(
                batch=batch,
                layout=layout,
                draft_block_ids=draft_block_ids,
                draft_tokens=draft_tokens,
                bs=bs,
                device=device,
                sampling_info=sampling_info,
                verify_window=verify_window,
                inject_gate=fold_eligible,
            )
        else:
            if (
                self._verify_epilogue is not None
                and self._verify_planner.mode_value == "static"
            ):
                self._verify_epilogue.begin_static_step(bs, fold_eligible)
            target_verify = self._verify_executor.run_non_compact(
                batch=batch,
                draft_input=draft_input,
                verify_ids_2d=verify_ids_2d,
                verify_window=verify_window,
                sampling_info=sampling_info,
            )
            hidden_strided = None
    logits_output = target_verify.logits_output
    can_run_cuda_graph = target_verify.can_run_cuda_graph
    if batch.has_grammar:
        # run_compact scatters its rows back to (bs * chain_len), so the mask
        # lines up with the logits on both verify paths.
        grammar_mask = build_grammar_vocab_mask(
            reqs=batch.reqs,
            tree=grammar_tree,
            sampling_info=sampling_info,
            device=logits_output.next_token_logits.device,
            barrier=grammar_barrier,
        )
        if grammar_mask is not None:
            grammar_mask.apply(logits_output.next_token_logits)

    epilogue = self._verify_executor.verify_epilogue
    folded_accept = (
        fold_eligible
        and can_run_cuda_graph
        and (run_compact or self._verify_planner.mode_value == "static")
    )
    region_accept = None
    if (
        not folded_accept
        and can_run_cuda_graph
        and not run_compact
        and self._sampling_graph_eligible(
            batch=batch,
            proposal_folded=proposal.folded,
            sampling_info=sampling_info,
            simulate_bonus_sampling_info=simulate_bonus_sampling_info,
        )
    ):
        region_accept = self._sampling_accept_graph.run(
            bs=bs, target_logits=logits_output.next_token_logits
        )
    # The accept and its commits ran in a graph: the verify graph's fold or the
    # sampling-accept region replayed right after it.
    graph_accept = folded_accept or region_accept is not None
    if region_accept is not None:
        epilogue.note_static_verify(bs)
        accept = region_accept
    else:
        accept = self._verify_executor.accept_and_finalize(
            folded_accept=folded_accept,
            bs=bs,
            verify_ids_2d=verify_ids_2d,
            target_logits=logits_output.next_token_logits,
            draft_block=draft_block,
            sampling_info=sampling_info,
            draft_input=draft_input,
            layout=layout,
            prefix_lens=prefix_lens,
            draft_tokens=draft_tokens,
            simulate_bonus_sampling_info=simulate_bonus_sampling_info,
        )
    engram_in_graph = (
        folded_accept and not run_compact and epilogue.folds_engram_commit
    ) or (
        region_accept is not None and self._sampling_accept_graph.committed_engram
    )
    # The static verify graph committed the engram history when it folded the accept.
    if not engram_in_graph:
        self.model_runner.ngram_embedding_manager.update_after_verify(
            verify_ids_2d=verify_ids_2d,
            req_pool_indices=batch.req_pool_indices,
            commit_lens=accept.commit_lens,
        )
    if hisparse_verify is not None:
        hisparse_verify.commit(
            batch.req_pool_indices, prefix_lens, verify_window.verify_cache_loc, accept.commit_lens
        )
    sampling_mask_capture = SpeculativeSamplingMaskCapture.from_logits(
        sampling_info,
        next_token_logits=logits_output.next_token_logits,
        draft_input=draft_input,
        draft_token_num=self.verify_num_draft_tokens,
        bs=bs,
        greedy_mask=draft_block.greedy_mask,
    )
    if sampling_mask_capture is not None:
        logits_output.sampling_mask_output = sampling_mask_capture.build_output(
            out_tokens=accept.out_tokens,
            commit_lens=accept.commit_lens,
        )
    if batch.return_logprob:
        compute_spec_logprobs(
            batch,
            logits_output,
            accept.out_tokens.reshape(-1),
            chain_stride=self.verify_num_draft_tokens,
        )

    relay = self._future_relay_in_graph(
        run_compact=run_compact, folded_accept=graph_accept
    )
    if relay is not None:
        relay.published_seq_lens = accept.new_seq_lens
    if on_publish is not None:
        if confidence is not None:
            on_publish(accept.new_seq_lens, confidence=confidence)
        else:
            on_publish(accept.new_seq_lens)

    self._commit_target_mamba_states_after_verify(
        batch=batch,
        seq_lens_pre_verify=prefix_lens,
        seq_lens_post_verify=accept.new_seq_lens,
        commit_lens=accept.commit_lens,
    )

    folded_commit = graph_accept and epilogue.folds_commit
    # Consume in this step: every decode graph size shares one aux output,
    # which the next target forward overwrites (resolve_aux_hidden_states_width).
    if not folded_commit:
        # The draft FC was loaded in the draft TP group. Other attention DP
        # ranks may be idle, so its projection must use that same group.
        with self._draft_context():
            self._verify_executor.commit_hidden(
                batch=batch,
                layout=layout,
                hidden_strided=hidden_strided,
                verify_window=verify_window,
                logits_output=logits_output,
                commit_lens=accept.commit_lens,
                bs=bs,
                run_compact=run_compact,
            )
    logits_output.hidden_states = None

    self._observers.observe_verify_step(
        forward_ct=int(batch.forward_iter),
        reqs=batch.reqs,
        bs=bs,
        proposal_folded=proposal.folded,
        verify_ids_2d=verify_ids_2d,
        target_logits=logits_output.next_token_logits,
        layout=layout,
        confidence=confidence,
        confidence_raw=proposal.confidence_raw,
        prefix_lens=prefix_lens,
        draft_tokens=draft_tokens,
        draft_block=draft_block,
        sampling_info=sampling_info,
        correct_len=accept.correct_len,
        cap_trim_lens=accept.cap_trim_lens,
        bonus=accept.bonus,
        commit_lens=accept.commit_lens,
        verify_token_budget=verify_token_budget,
        req_pool_indices=batch.req_pool_indices,
        verify_tier_num_tokens=int(batch.spec_verify_tier_num_tokens),
        dp_tier_num_tokens=self._dp_verify_tier_num_tokens(batch),
    )

    next_draft_input = make_next_draft_input(
        bonus_tokens=accept.bonus,
        new_seq_lens=accept.new_seq_lens,
    )
    if relay is not None:
        relay.published_bonus = next_draft_input.bonus_tokens
    return GenerationBatchResult(
        logits_output=logits_output,
        next_token_ids=accept.out_tokens.reshape(-1),
        accept_lens=accept.commit_lens,
        block_accept_lens=self._block_accept_lens(
            accept=accept, folded_accept=graph_accept, bs=bs
        ),
        cap_lens=(
            layout.verify_lens.to(torch.int32) if layout is not None else None
        ),
        can_run_cuda_graph=can_run_cuda_graph,
        next_draft_input=next_draft_input,
        speculative_num_draft_tokens=int(self.verify_num_draft_tokens),
        new_seq_lens=accept.new_seq_lens,
    )


def _cuda_host_register(
    buffer: torch.Tensor, registration_granularity_bytes: int | None = None
) -> None:
    # Avoid oversized cudaHostRegister calls on large host pools.
    cudart = torch.cuda.cudart()
    base = buffer.data_ptr()
    total = buffer.numel() * buffer.element_size()
    chunk_limit_bytes = (
        max(envs.SGLANG_HICACHE_HOST_REGISTER_CHUNK_GB.get(), 1) * 1024**3
    )
    # Preserve the legacy single-call behavior unless the caller provides a
    # copy granularity. Splitting an unknown page-first layout at an arbitrary
    # byte offset can make one cudaMemcpyBatchAsync span two registrations.
    chunk_bytes = total
    if registration_granularity_bytes is not None:
        if registration_granularity_bytes <= 0:
            raise ValueError(
                "registration_granularity_bytes must be positive, got "
                f"{registration_granularity_bytes}"
            )
        if registration_granularity_bytes > chunk_limit_bytes:
            raise ValueError(
                "Host registration granularity exceeds the configured chunk limit: "
                f"granularity={registration_granularity_bytes}, "
                f"chunk_limit={chunk_limit_bytes}"
            )
        chunk_bytes = (
            chunk_limit_bytes // registration_granularity_bytes
        ) * registration_granularity_bytes
    registered_ranges: list[tuple[int, int]] = []
    try:
        offset = 0
        while offset < total:
            size = min(chunk_bytes, total - offset)
            ptr = base + offset
            status = cudart.cudaHostRegister(ptr, size, 2 if is_hcu() else 0)
            rc = int(status)
            if rc != 0:
                raise RuntimeError(
                    f"cudaHostRegister failed (rc={rc}, "
                    f"{cudart.cudaGetErrorString(status)}) at offset={offset} size={size} "
                    f"(total={total}, chunk_limit={chunk_bytes}); host buffer is not "
                    f"pinned and device transfers may silently return stale data."
                )
            registered_ranges.append((ptr, size))
            offset += size

        # Keep the exact registration bases alive with the tensor. CUDA requires
        # cudaHostUnregister to receive each base pointer, not just the tensor's
        # original base once after several independent registrations.
        setattr(buffer, _CUDA_HOST_REGISTERED_RANGES_ATTR, registered_ranges)
    except Exception:
        remaining_ranges = _cuda_host_unregister_ranges(
            cudart, registered_ranges, operation="registration rollback"
        )
        if remaining_ranges:
            setattr(buffer, _CUDA_HOST_REGISTERED_RANGES_ATTR, remaining_ranges)
        raise


def _cuda_host_unregister_ranges(
    cudart, registered_ranges: list[tuple[int, int]], *, operation: str
) -> list[tuple[int, int]]:
    failed_ranges = []
    for ptr, size in reversed(registered_ranges):
        status = cudart.cudaHostUnregister(ptr)
        rc = int(status)
        if rc != 0:
            failed_ranges.append((ptr, size))
            logger.warning(
                "cudaHostUnregister failed during %s (rc=%d, %s) for ptr=%#x size=%d",
                operation,
                rc,
                cudart.cudaGetErrorString(status),
                ptr,
                size,
            )
    failed_ranges.reverse()
    return failed_ranges


def _hip_runtime() -> ctypes.CDLL:
    from sglang.srt.distributed.device_communicators.cuda_wrapper import (
        find_loaded_library,
    )

    # Bind the copy torch loaded; dlopen by soname could load a second HIP runtime.
    path = find_loaded_library("libamdhip64") or find_loaded_library("libgalaxyhip")
    if path is None:
        raise RuntimeError("No loaded HIP runtime (libamdhip64 or libgalaxyhip)")
    lib = ctypes.CDLL(path)
    lib.hipHostMalloc.argtypes = [
        ctypes.POINTER(ctypes.c_void_p),
        ctypes.c_size_t,
        ctypes.c_uint,
    ]
    lib.hipHostMalloc.restype = ctypes.c_int
    lib.hipHostFree.argtypes = [ctypes.c_void_p]
    lib.hipHostFree.restype = ctypes.c_int
    lib.hipGetErrorString.argtypes = [ctypes.c_int]
    lib.hipGetErrorString.restype = ctypes.c_char_p
    return lib
