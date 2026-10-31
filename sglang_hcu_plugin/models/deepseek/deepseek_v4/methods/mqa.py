"""HCU DeepSeek V4 fork methods (auto-extracted for plugin hooks)."""

from __future__ import annotations

from sglang_hcu_plugin.models.deepseek.deepseek_v4 import constants as C
from sglang_hcu_plugin.models.deepseek.deepseek_v4.methods._runtime import *  # noqa: F403

def __init__(
        self,
        config: DeepSeekV4Config,
        layer_id: int,
        quant_config: Optional[QuantizationConfig] = None,
        prefix: str = "",
        alt_streams: Optional[List[torch.cuda.Stream]] = None,
        compress_ratio_override: Optional[int] = None,
    ) -> None:
        super().__init__()
        self.tp_rank = attn_tp_rank = get_attention_tp_rank()
        self.tp_size = attn_tp_size = get_attention_tp_size()
        self.nsa_enable_prefill_cp = is_nsa_enable_prefill_cp()
        if self.nsa_enable_prefill_cp:
            self.cp_size = get_attention_cp_size()
            self.tp_rank = attn_tp_rank = 0
            self.tp_size = attn_tp_size = 1
        self.layer_id = layer_id
        self.dim = config.hidden_size
        self.qk_rope_head_dim = config.qk_rope_head_dim
        self.qk_nope_head_dim = config.head_dim - config.qk_rope_head_dim
        self.head_dim = self.qk_rope_head_dim + self.qk_nope_head_dim
        self.n_heads = config.num_attention_heads
        self.n_local_heads = self.n_heads // attn_tp_size
        self.n_groups = config.o_groups
        self.n_local_groups = self.n_groups // attn_tp_size
        self.rope_head_dim = config.qk_rope_head_dim
        self.softmax_scale = self.head_dim**-0.5
        self.hidden_size = config.hidden_size
        self.q_lora_rank = config.q_lora_rank
        self.o_lora_rank = config.o_lora_rank
        self.eps = config.rms_norm_eps
        compress_ratio = (
            compress_ratio_override
            if compress_ratio_override is not None
            else config.compress_ratios[layer_id]
        )
        assert compress_ratio in [0, 4, 128]
        self.compress_ratio: Literal[0, 4, 128] = compress_ratio

        assert self.head_dim == config.head_dim
        assert config.num_key_value_heads == 1

        rope_theta, rope_scaling = get_rope_config(config)
        if rope_scaling:
            rope_scaling["rope_type"] = "deepseek_yarn"

        rope_base = config.compress_rope_theta if self.compress_ratio else rope_theta

        from sglang.srt.layers.deepseek_v4_rope import precompute_freqs_cis

        assert self.compress_ratio in {0, 4, 128}
        if self.compress_ratio:
            original_seq_len = rope_scaling["original_max_position_embeddings"]
        else:
            original_seq_len = 0

        freqs_cis = precompute_freqs_cis(
            dim=self.qk_rope_head_dim,
            seqlen=config.max_position_embeddings,
            original_seq_len=original_seq_len,
            base=rope_base,
            factor=rope_scaling["factor"],
            beta_fast=rope_scaling["beta_fast"],
            beta_slow=rope_scaling["beta_slow"],
        )
        self.register_buffer("freqs_cis", freqs_cis, persistent=False)
        self.freqs_cis: torch.Tensor

        self.register_buffer("cos_sin_cache_fused", None, persistent=False)
        self.cos_sin_cache_fused: Optional[torch.Tensor]
        if C._is_hcu and C._use_fused_qnorm_rope_kv_rope_quant:
            freqs_real = torch.view_as_real(self.freqs_cis)  # [max_pos, 32, 2]
            cos_sin_cache = torch.cat(
                [freqs_real[..., 0], freqs_real[..., 1]], dim=-1
            ).contiguous()  # [max_pos, 64], first 32 cos, last 32 sin
            self.cos_sin_cache_fused = cos_sin_cache

        if envs.SGLANG_OPT_USE_MULTI_STREAM_OVERLAP.get() and alt_streams is not None:
            self.alt_streams = alt_streams[:3]
            self.alt_streams_indexer = alt_streams[-2:]
        else:
            self.alt_streams = None
            self.alt_streams_indexer = None

        from sglang.srt.utils import is_blackwell_supported

        self._multi_stream_bs_limit = 128 if is_blackwell_supported() else 64

        self.compressor = None
        self.indexer = None
        if self.compress_ratio:
            self.compressor = Compressor(
                config,
                layer_id=self.layer_id,
                is_in_indexer=False,
                freqs_cis=freqs_cis,
                compress_ratio=self.compress_ratio,
                head_dim=self.head_dim,
                rotate=False,
                prefix=add_prefix("compressor", prefix),
            )
            if self.compress_ratio == 4:
                self.indexer = C4Indexer(
                    config,
                    freqs_cis=freqs_cis,
                    layer_id=layer_id,
                    quant_config=quant_config,
                    prefix=add_prefix("indexer", prefix),
                    alt_streams=self.alt_streams_indexer,
                )

        self.attn_sink = nn.Parameter(torch.empty(self.n_heads, dtype=torch.float32))
        self.fuse_wqa_wkv = envs.SGLANG_OPT_FUSE_WQA_WKV.get()
        if self.fuse_wqa_wkv:
            self.wqkv_a = ReplicatedLinear(
                self.hidden_size,
                self.q_lora_rank + self.head_dim,
                bias=False,
                quant_config=quant_config,
                prefix=add_prefix("wqkv_a", prefix),
            )
        else:
            self.wq_a = ReplicatedLinear(
                self.hidden_size,
                self.q_lora_rank,
                bias=False,
                quant_config=quant_config,
                prefix=add_prefix("wq_a", prefix),
            )
            self.wkv = ReplicatedLinear(
                self.hidden_size,
                self.head_dim,
                bias=False,
                quant_config=quant_config,
                prefix=add_prefix("wkv", prefix),
            )
        self.q_norm = RMSNorm(self.q_lora_rank, eps=self.eps)
        self.use_jit_norm = False
        self.wq_b = ColumnParallelLinear(
            self.q_lora_rank,
            self.n_heads * self.head_dim,
            bias=False,
            quant_config=quant_config,
            prefix=add_prefix("wq_b", prefix),
            tp_rank=attn_tp_rank,
            tp_size=attn_tp_size,
        )
        self.kv_norm = RMSNorm(self.head_dim, eps=self.eps)
        if _FP8_WO_A_GEMM and quant_config is not None:
            quant_config.ignore = [i for i in quant_config.ignore if 'wo_a' not in i]
        self.wo_a = ColumnParallelLinear(
            self.n_heads * self.head_dim // self.n_groups,
            self.n_groups * self.o_lora_rank,
            bias=False,
            quant_config=quant_config if _FP8_WO_A_GEMM else None,
            prefix=add_prefix("wo_a", prefix),
            tp_rank=attn_tp_rank,
            tp_size=attn_tp_size,
            **({} if _FP8_WO_A_GEMM else {"params_dtype": torch.bfloat16}),
        )
        if _FP8_WO_A_GEMM:
            assert hasattr(
                self.wo_a, "weight_scale"
            ), "FP8 quant_config must create weight_scale"
            self.wo_a.weight_scale_inv = self.wo_a.weight_scale
            self.wo_a.weight_scale_inv.format_ue8m0 = True
        self.wo_b = RowParallelLinear(
            self.n_groups * self.o_lora_rank,
            self.hidden_size,
            bias=False,
            quant_config=quant_config,
            reduce_results=attn_tp_size > 1,
            prefix=add_prefix("wo_b", prefix),
            tp_rank=attn_tp_rank,
            tp_size=attn_tp_size,
        )

        self.attn_mqa = RadixAttention(
            self.n_local_heads,
            self.head_dim,
            self.softmax_scale,
            num_kv_heads=1,
            layer_id=layer_id,
            quant_config=quant_config,
            prefix=add_prefix("attn_mqa", prefix),
        )

def _compute_q_b(
        self,
        q: torch.Tensor,
        positions: torch.Tensor,
        q_out: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        q, _ = self.wq_b(q)
        q = q.view(-1, self.n_local_heads, self.head_dim)
        # if q_out is None:
        #     q_out = torch.empty_like(q)
        # # Fused warp-per-(token, head) rmsnorm-self + RoPE + write to q_out.
        # fused_q_norm_rope(q, q_out, self.eps, self.freqs_cis, positions)
        # return q_out

        if C._is_hcu and C._use_dpskv4_lightop_rmsnorm:
            C.op.rms_norm_no_weight(None, q, None, self.eps)
        else:
            q = rms_normalize_triton(q, self.eps)
        if positions is not None:
            fused_rope(
                q[..., -self.qk_rope_head_dim :],
                None,
                self.freqs_cis,
                positions=positions,
            )
        else:
            apply_rotary_emb_triton(q[..., -self.qk_rope_head_dim :], self.freqs_cis)
        if q_out is not None:
            q_out.copy_(q)
        return q

def _forward_prepare(
        self,
        x: torch.Tensor,
        positions: torch.Tensor,
        forward_batch: ForwardBatch,
        attn_backend: DeepseekV4AttnBackend,
        q_out: Optional[torch.Tensor] = None,
    ) -> Tuple[torch.Tensor, Optional[torch.Tensor]]:
        if self.fuse_wqa_wkv:
            qkv_a, _ = self.wqkv_a(x)
            q_lora = qkv_a[..., : self.q_lora_rank]
            kv = qkv_a[..., self.q_lora_rank :]
        else:
            q_lora, _ = self.wq_a(x)
            kv, _ = self.wkv(x)
        q_lora = self.q_norm(q_lora)
        q, _ = self.wq_b(q_lora)
        q = q.view(-1, self.n_local_heads, self.head_dim)

        _cp_enabled = self.nsa_enable_prefill_cp and nsa_use_prefill_cp(forward_batch)
        _bf16_kv_cache = forward_batch.token_to_kv_pool.is_bf16_attention_kv_cache
        _use_lightop_qnorm_rope = (
            C._use_fused_qnorm_rope_kv_rope_quant and C._is_hcu
        )
        if _use_lightop_qnorm_rope:
            cos_sin_cache_fused = self.cos_sin_cache_fused
            assert cos_sin_cache_fused is not None
        else:
            cos_sin_cache_fused = None

        if _use_lightop_qnorm_rope and not _cp_enabled and not _bf16_kv_cache:
            # Fused kernel: Q no-weight RMSNorm + Q RoPE + KV RMSNorm +
            # K RoPE + K FP8 quant + cache insert.
            pool = forward_batch.token_to_kv_pool
            swa_pool = pool.swa_kv_pool
            layer_idx = self.layer_id - pool._stage_start
            k_cache = swa_pool.kv_buffer[layer_idx]
            cache_block_size = swa_pool.page_size
            raw_loc = forward_batch.out_cache_loc
            slot_mapping = pool.translate_loc_from_full_to_swa(raw_loc)

            C.op.fused_deepseek_v4_qnorm_rope_kvnorm_rope_quant_insert_int32(
                q, kv, self.kv_norm.weight, k_cache, slot_mapping,
                positions, cos_sin_cache_fused,
                self.eps, cache_block_size,
            )
        else:
            if _use_lightop_qnorm_rope:
                # CP must all-gather/rerange KV before cache insert, so only fuse
                # the local Q norm/RoPE and local KV norm/RoPE here.
                C.op.fused_deepseek_v4_qnorm_rope_kvnorm_rope(
                    q,
                    kv,
                    self.kv_norm.weight,
                    positions,
                    cos_sin_cache_fused,
                    self.eps,
                )
            else:
                kv = self.kv_norm(kv)
                if self.use_jit_norm:
                    q = rmsnorm_self(q, self.eps)
                else:
                    if C._is_hcu and C._use_dpskv4_lightop_rmsnorm:
                        C.op.rms_norm_no_weight(None, q, None, self.eps)
                    else:
                        q = rms_normalize_triton(q, self.eps)

                fused_rope(
                    q[..., -self.qk_rope_head_dim :],
                    kv[..., -self.qk_rope_head_dim :].unsqueeze(1),
                    self.freqs_cis,
                    positions=positions,
                )

            if _cp_enabled:
                kv = cp_all_gather_rerange_output(
                    kv.contiguous(),
                    self.cp_size,
                    forward_batch,
                    torch.cuda.current_stream(),
                )
                # if envs.SGLANG_DEBUG_HACK_CP_CHECK_RANK_CONSISTENCY.get():
                #     assert_tensor_identical_across_cp_ranks(
                #         kv,
                #         tag=f"kv_after_allgather layer_id={self.layer_id}",
                #         forward_batch=forward_batch,
                #     )

            if getattr(self, "overlap_store_cache", False):
                attn_backend.store_cache(
                    layer_id=self.layer_id,
                    swa_k=kv,
                    forward_batch=forward_batch,
                )

        if self.indexer is not None:
            self.indexer(x=x, q_lora=q_lora, forward_batch=forward_batch)
        if self.compressor is not None:
            attn_backend.forward_core_compressor(
                x,
                forward_batch,
                self.layer_id,
                self.compressor,
            )

        if q_out is not None:
            q_out.copy_(q)
        return q, kv

def forward(
        self,
        x: torch.Tensor,
        positions: torch.Tensor,
        forward_batch: ForwardBatch,
    ) -> torch.Tensor:
        if not get_attn_tp_context().input_scattered and x.shape[0] == 0:
            assert (
                not self.wo_b.reduce_results
            ), "short-circuiting allreduce will lead to hangs"
            return x

        attn_backend = forward_batch.attn_backend
        if TYPE_CHECKING:
            assert isinstance(attn_backend, DeepseekV4AttnBackend)

        enable_multi_stream = (
            envs.SGLANG_OPT_USE_MULTI_STREAM_OVERLAP.get()
            and self.alt_streams is not None
            and get_is_capture_mode()
            and x.shape[0] <= self._multi_stream_bs_limit
            and not (self.nsa_enable_prefill_cp and nsa_use_prefill_cp(forward_batch))
        )

        tp_slice, q_padded, q_out = slice(None), None, None
        if self.tp_size > 1:
            q_padded = x.new_empty(x.shape[0], self.n_heads, self.head_dim)
            rank = self.tp_rank
            tp_slice = slice(rank * self.n_local_heads, (rank + 1) * self.n_local_heads)
            q_out = q_padded[:, tp_slice, :]

        if enable_multi_stream:
            # Multi-stream path always fuses cache write into the K kernel,
            # so the bf16 KV intermediate is gone.
            q = self._forward_prepare_multi_stream(
                x, positions, forward_batch, attn_backend, q_out
            )
            kv = None
        else:
            q, kv = self._forward_prepare(
                x, positions, forward_batch, attn_backend, q_out
            )

        # Skip backend cache write only when the prepare path already wrote it.
        attn_k = kv if kv is not None else q
        _cp_enabled = self.nsa_enable_prefill_cp and nsa_use_prefill_cp(forward_batch)
        _fused_cache_insert_path = (
            C._use_fused_qnorm_rope_kv_rope_quant
            and not _cp_enabled
            and C._is_hcu
            and not enable_multi_stream
            and not forward_batch.token_to_kv_pool.is_bf16_attention_kv_cache
        )
        _cache_already_written = (
            enable_multi_stream
            or _fused_cache_insert_path
            or getattr(self, "overlap_store_cache", False)
        )
        o = attn_backend.forward(
            q=q_padded if q_padded is not None else q,
            k=attn_k,
            v=attn_k,
            layer=self.attn_mqa,
            forward_batch=forward_batch,
            compress_ratio=self.compress_ratio,
            attn_sink=self.attn_sink,
            save_kv_cache=not _cache_already_written,
        )
        o = o[:, tp_slice, :]
        fused_rope_inplace(
            o[..., -self.qk_rope_head_dim :],
            None,
            self.freqs_cis,
            positions=positions,
            inverse=True,
        )

        o = o.view(o.shape[0], self.n_local_groups, -1)

        if _FP8_WO_A_GEMM:
            import deepgemm as deep_gemm

            T, G, D = o.shape
            R = self.o_lora_rank

            o_fp8, o_s = per_token_quant_fp8(
                o.reshape(T * G, D).contiguous(),
            )

            lhs_fp8 = o_fp8.view(T, G, D)
            lhs_scale = o_s.view(T, G)

            w = self.wo_a.weight
            if tuple(w.shape) == (D, G * R):
                rhs_fp8 = w.t().contiguous().view(G, R, D)
            elif tuple(w.shape) == (G * R, D):
                rhs_fp8 = w.contiguous().view(G, R, D)
            else:
                raise RuntimeError(
                    f"unexpected wo_a.weight shape={tuple(w.shape)}, expected {(D, G * R)} or {(G * R, D)}"
                )

            weight_scale = getattr(self.wo_a, "weight_scale", None)
            if weight_scale is None:
                weight_scale = self.wo_a.weight_scale_inv
            rhs_scale = weight_scale.data.reshape(G, R).contiguous()

            output = torch.empty(T, G, R, device=o.device, dtype=torch.bfloat16)

            deep_gemm.fp8_einsum(
                "bhr,hdr->bhd",
                (lhs_fp8, lhs_scale),
                (rhs_fp8, rhs_scale),
                output,
                recipe=(1, 1, 128),
            )
            o = output
        else:
            wo_a = self.wo_a.weight.view(self.n_local_groups, self.o_lora_rank, -1)
            o = torch.einsum("tgd,grd->tgr", o, wo_a)

        o, _ = self.wo_b(o.flatten(1))

        return o

