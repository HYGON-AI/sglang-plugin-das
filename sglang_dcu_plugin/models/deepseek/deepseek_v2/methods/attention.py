"""DCU DeepSeek V2 fork methods (auto-extracted for plugin hooks)."""

from __future__ import annotations

from sglang_dcu_plugin.models.deepseek.deepseek_v2 import constants as C
from sglang_dcu_plugin.models.deepseek.deepseek_v2.methods._runtime import *  # noqa: F403

def __init__(
        self,
        config: PretrainedConfig,
        hidden_size: int,
        num_heads: int,
        qk_nope_head_dim: int,
        qk_rope_head_dim: int,
        v_head_dim: int,
        q_lora_rank: int,
        kv_lora_rank: int,
        rope_theta: float = 10000,
        rope_scaling: Optional[Dict[str, Any]] = None,
        max_position_embeddings: int = 8192,
        quant_config: Optional[QuantizationConfig] = None,
        reduce_results: bool = True,
        layer_id: int = None,
        prefix: str = "",
        alt_stream: Optional[torch.cuda.Stream] = None,
        skip_rope: bool = False,
        is_nextn: bool = False,
        input_layernorm: Optional[torch.nn.Module] = None,
    ) -> None:
        super().__init__()
        self.layer_id = layer_id
        self.hidden_size = hidden_size
        self.qk_nope_head_dim = qk_nope_head_dim
        self.qk_rope_head_dim = qk_rope_head_dim
        self.qk_head_dim = qk_nope_head_dim + qk_rope_head_dim
        self.v_head_dim = v_head_dim
        self.q_lora_rank = q_lora_rank
        self.kv_lora_rank = kv_lora_rank
        self.quant_config = quant_config
        self.is_nextn = is_nextn
        self.input_layernorm = input_layernorm  # this is from other place
        # self.input_layernorm_weight = input_layernorm_weight
        attn_tp_rank = get_attention_tp_rank()
        attn_tp_size = get_attention_tp_size()
        self.use_nsa = is_deepseek_nsa(config)
        self.nsa_enable_prefill_cp = is_nsa_enable_prefill_cp()
        if self.nsa_enable_prefill_cp:
            assert self.use_nsa, "CP currently only supports deepseek v3.2 model"
        # cp reuse the attn_tp comm group but need to duplicate the weights
        if self.nsa_enable_prefill_cp and self.use_nsa:
            self.cp_size = get_attention_cp_size()
        self.num_heads = num_heads
        assert num_heads % attn_tp_size == 0
        self.num_local_heads = num_heads // attn_tp_size
        self.scaling = self.qk_head_dim**-0.5
        self.rope_theta = rope_theta
        self.max_position_embeddings = max_position_embeddings
        self.kv_cache_dtype = get_global_server_args().kv_cache_dtype

        # NOTE modification to rope_scaling must be done early enough, b/c e.g. Indexer needs it
        if rope_scaling:
            rope_scaling["rope_type"] = "deepseek_yarn"

        # For tensor parallel attention
        if self.q_lora_rank is not None:
            self.fused_qkv_a_proj_with_mqa = ReplicatedLinear(
                self.hidden_size,
                self.q_lora_rank + self.kv_lora_rank + self.qk_rope_head_dim,
                bias=False,
                quant_config=quant_config,
                prefix=add_prefix("fused_qkv_a_proj_with_mqa", prefix),
            )
            self.q_a_layernorm = RMSNorm(self.q_lora_rank, eps=config.rms_norm_eps)
            self.q_b_proj = ColumnParallelLinear(
                q_lora_rank,
                self.num_heads * self.qk_head_dim,
                bias=False,
                quant_config=self._get_q_b_proj_quant_config(quant_config),
                eps=config.rms_norm_eps,
                prefix=add_prefix("q_b_proj", prefix),
                tp_rank=attn_tp_rank,
                tp_size=attn_tp_size,
            )
        else:
            self.q_proj = ColumnParallelLinear(
                self.hidden_size,
                self.num_heads * self.qk_head_dim,
                bias=False,
                quant_config=quant_config,
                prefix=add_prefix("q_proj", prefix),
                tp_rank=attn_tp_rank,
                tp_size=attn_tp_size,
            )
            self.kv_a_proj_with_mqa = ReplicatedLinear(
                self.hidden_size,
                self.kv_lora_rank + self.qk_rope_head_dim,
                bias=False,
                quant_config=quant_config,
                prefix=add_prefix("kv_a_proj_with_mqa", prefix),
            )

        self.skip_topk = None
        self.next_skip_topk = None
        if self.use_nsa:
            is_neox_style = not getattr(config, "indexer_rope_interleave", False)
            self.indexer = Indexer(
                hidden_size=hidden_size,
                index_n_heads=get_nsa_index_n_heads(config),
                index_head_dim=get_nsa_index_head_dim(config),
                rope_head_dim=qk_rope_head_dim,
                index_topk=get_nsa_index_topk(config),
                q_lora_rank=q_lora_rank,
                max_position_embeddings=max_position_embeddings,
                rope_theta=rope_theta,
                scale_fmt="ue8m0",
                block_size=128,
                rope_scaling=rope_scaling,
                is_neox_style=is_neox_style,
                prefix=add_prefix("indexer", prefix),
                quant_config=quant_config,
                layer_id=layer_id,
                alt_stream=alt_stream,
            )
            # Refer: https://arxiv.org/abs/2603.12201 for more details.
            # skip_topk: when True, this layer will skip computation and reuse previous layer's topk indices.
            # next_skip_topk: when True, the next layer will skip computation and reuse this layer's topk indices.
            if is_nextn:
                self.skip_topk = True
                self.next_skip_topk = True
            else:
                self.index_topk_freq = getattr(config, "index_topk_freq", 1)
                self.index_topk_pattern = getattr(config, "index_topk_pattern", None)
                self.index_skip_topk_offset = getattr(
                    config, "index_skip_topk_offset", None
                )
                if (
                    self.index_topk_pattern is None
                    and self.index_skip_topk_offset is not None
                ):
                    assert self.index_skip_topk_offset > 0, (
                        "index_skip_topk_offset must be positive; offset <= 0 "
                        "marks layer 0 as skip_topk with no prior topk to reuse"
                    )
                    self.skip_topk = (
                        max(layer_id - self.index_skip_topk_offset + 1, 0)
                        % self.index_topk_freq
                        != 0
                    )
                    self.next_skip_topk = (
                        max(layer_id - self.index_skip_topk_offset + 2, 0)
                        % self.index_topk_freq
                        != 0
                    )
                elif self.index_topk_pattern is None:
                    self.skip_topk = max(layer_id - 1, 0) % self.index_topk_freq != 0
                    self.next_skip_topk = layer_id % self.index_topk_freq != 0
                else:
                    self.skip_topk = self.index_topk_pattern[layer_id] == "S"
                    if layer_id < len(self.index_topk_pattern) - 1:
                        self.next_skip_topk = (
                            self.index_topk_pattern[layer_id + 1] == "S"
                        )
                    else:
                        self.next_skip_topk = False

        self.kv_b_proj = ColumnParallelLinear(
            self.kv_lora_rank,
            self.num_heads * (self.qk_nope_head_dim + self.v_head_dim),
            bias=False,
            quant_config=quant_config,
            prefix=add_prefix("kv_b_proj", prefix),
            tp_rank=attn_tp_rank,
            tp_size=attn_tp_size,
        )
        # O projection.
        self.o_proj = RowParallelLinear(
            self.num_heads * self.v_head_dim,
            self.hidden_size,
            bias=False,
            quant_config=quant_config,
            reduce_results=reduce_results,
            prefix=add_prefix("o_proj", prefix),
            tp_rank=attn_tp_rank,
            tp_size=attn_tp_size,
        )
        self.kv_a_layernorm = RMSNorm(self.kv_lora_rank, eps=config.rms_norm_eps)

        if not skip_rope:
            is_neox_style = not getattr(config, "rope_interleave", True)
            self.rotary_emb = get_rope_wrapper(
                qk_rope_head_dim,
                rotary_dim=qk_rope_head_dim,
                max_position=max_position_embeddings,
                base=rope_theta,
                rope_scaling=rope_scaling,
                is_neox_style=is_neox_style,
                device=get_global_server_args().device,
            )

            if rope_scaling and rope_scaling.get("apply_yarn_scaling", True):
                self.scaling = compute_mla_mscale_scaling(rope_scaling, self.scaling)
        else:
            self.rotary_emb = None
        self.use_deepseek_yarn_rope = rope_scaling is not None

        self.attn_mqa = RadixAttention(
            self.num_local_heads,
            self.kv_lora_rank + self.qk_rope_head_dim,
            self.scaling,
            num_kv_heads=1,
            layer_id=layer_id,
            v_head_dim=self.kv_lora_rank,
            quant_config=quant_config,
            prefix=add_prefix("attn_mqa", prefix),
        )

        self.attn_mha = RadixAttention(
            self.num_local_heads,
            self.qk_nope_head_dim + self.qk_rope_head_dim,
            self.scaling,
            num_kv_heads=self.num_local_heads,
            layer_id=layer_id,
            v_head_dim=self.v_head_dim,
            quant_config=quant_config,
            prefix=add_prefix("attn_mha", prefix),
        )

        self.alt_stream = alt_stream
        self.attn_mha.kv_b_proj = None

        self.w_kc = None
        self.w_vc = None
        self.w_scale = 1.0

        self.w_scale_k = None
        self.w_scale_v = None
        self.use_deep_gemm_bmm = False

        self.current_attention_backend = (
            None  # Attention backend used by current forward batch
        )

        self.has_fused_proj = hasattr(self, "fused_qkv_a_proj_with_mqa")
        self.is_packed_weight = (
            self.has_fused_proj
            and hasattr(self.fused_qkv_a_proj_with_mqa.quant_method, "quant_config")
            and self.fused_qkv_a_proj_with_mqa.quant_method.quant_config.get_name()
            in {"awq", "awq_marlin", "moe_wna16"}
        )
        self.use_min_latency_fused_a_gemm = (
            self.has_fused_proj
            and not self.is_packed_weight
            and self.fused_qkv_a_proj_with_mqa.weight.dtype == torch.bfloat16
            and self.fused_qkv_a_proj_with_mqa.weight.shape[0] == 2112
            and self.fused_qkv_a_proj_with_mqa.weight.shape[1] == 7168
            and C._is_cuda
            and 90 <= C._device_sm < 120
        )

        self.init_mha_forward()
        self.init_mla_forward()
        self.init_mla_fused_rope_rocm_forward()
        self.init_mla_fused_rope_cpu_forward()

def prepare_qkv_latent(
        self, hidden_states: torch.Tensor, forward_batch: ForwardBatch
    ):
        assert self.q_lora_rank is not None
        # When the module is wrapped with LoRA, the fused GEMM fast-path would
        # bypass the adapter because it reads weight.T directly.
        lora_active = getattr(self.fused_qkv_a_proj_with_mqa, "set_lora", False)
        if (
            (not isinstance(hidden_states, tuple))
            and hidden_states.shape[0] >= 1
            and hidden_states.shape[0] <= 16
            and self.use_min_latency_fused_a_gemm
            and not lora_active
        ):
            qkv_latent = dsv3_fused_a_gemm(
                hidden_states, self.fused_qkv_a_proj_with_mqa.weight.T
            )
        else:
            if C._use_fused_rms_quant and self.input_layernorm is not None:
                # NOTE: suspected
                if C._rms_quant_path == 1:
                # path 1
                    if forward_batch.residual_rms_per_quant_int8 is None:
                        qkv_latent, _ = self.fused_qkv_a_proj_with_mqa(
                            hidden_states, 
                            rms_weight=self.input_layernorm.weight.data,
                            residual=None,
                            update_hd=False)
                        forward_batch.residual_rms_per_quant_int8 = hidden_states
                    else:
                        qkv_latent, _ = self.fused_qkv_a_proj_with_mqa(
                            hidden_states, 
                            rms_weight=self.input_layernorm.weight.data,
                            residual=forward_batch.residual_rms_per_quant_int8,
                            update_hd=False)
                else:
                # path 2
                    if forward_batch.residual_rms_per_quant_int8 is None: 
                        _normed = self.input_layernorm(hidden_states)
                        forward_batch.residual_rms_per_quant_int8 = hidden_states
                    else:
                        _normed, _ = self.input_layernorm(hidden_states, forward_batch.residual_rms_per_quant_int8)
                    qkv_latent = self.fused_qkv_a_proj_with_mqa(_normed)[0]
                
            else:
                qkv_latent = self.fused_qkv_a_proj_with_mqa(hidden_states)[0]
        return qkv_latent

def forward_absorb_prepare(
        self,
        positions: torch.Tensor,
        hidden_states: torch.Tensor,
        forward_batch: ForwardBatch,
        zero_allocator: BumpAllocator,
        llama_4_scaling: Optional[torch.Tensor] = None,
        prev_topk_indices: Optional[torch.Tensor] = None,
    ):
        from sglang.srt.model_executor.cuda_graph_runner import get_is_capture_mode
        q_lora = None
        topk_indices = None
        if self.q_lora_rank is not None:
            q, latent_cache = (
                get_attn_tp_context()
                .fetch_qkv_latent()
                .split(
                    [self.q_lora_rank, self.kv_lora_rank + self.qk_rope_head_dim],
                    dim=-1,
                )
            )
            k_nope = latent_cache[..., : self.kv_lora_rank]
            # overlap qk norm
            if self.alt_stream is not None and get_is_capture_mode() and not C._use_fused_rmsnorm_rope and not C._use_fused_rms_quant:
                current_stream = torch.cuda.current_stream()
                self.alt_stream.wait_stream(current_stream)
                q = self.q_a_layernorm(q)
                with torch.cuda.stream(self.alt_stream):
                    k_nope = self.kv_a_layernorm(k_nope)
                current_stream.wait_stream(self.alt_stream)
            else:
                if C._use_aiter_gfx95 and self.q_b_proj.weight.dtype == torch.uint8:
                    q, _, k_nope, *_ = fused_rms_mxfp4_quant(
                        q,
                        self.q_a_layernorm.weight,
                        self.q_a_layernorm.variance_epsilon,
                        k_nope,
                        self.kv_a_layernorm.weight,
                        self.kv_a_layernorm.variance_epsilon,
                    )
                else:
                    q_lora = None
                    if (
                        C._use_aiter_gfx95
                        and self.q_b_proj.weight.dtype == torch.float8_e4m3fn
                    ):
                        if self.use_nsa:
                            q_quanted, q_lora, k_nope, _ = fused_rms_fp8_group_quant(
                                q,
                                self.q_a_layernorm.weight,
                                self.q_a_layernorm.variance_epsilon,
                                k_nope,
                                self.kv_a_layernorm.weight,
                                self.kv_a_layernorm.variance_epsilon,
                                group_size=128,
                                dtype_quant=torch.float8_e4m3fn,
                                res1=None,
                                output_unquantized_inp1=True,
                            )
                            q = q_quanted
                        else:
                            q, _, k_nope, _ = fused_rms_fp8_group_quant(
                                q,
                                self.q_a_layernorm.weight,
                                self.q_a_layernorm.variance_epsilon,
                                k_nope,
                                self.kv_a_layernorm.weight,
                                self.kv_a_layernorm.variance_epsilon,
                                group_size=128,
                                dtype_quant=torch.float8_e4m3fn,
                                res1=None,
                                output_unquantized_inp1=False,
                            )

                    else:
                        if not C._use_fused_rms_quant:
                            q = self.q_a_layernorm(q)
                        if not C._use_fused_rmsnorm_rope:
                            k_nope = self.kv_a_layernorm(k_nope)
            # q_lora needed by indexer
            if self.use_nsa:
                if q_lora is None:
                    q_lora = q
            # overlap q_b_proj and indexer during decode
            if (
                self.alt_stream is not None
                and get_is_capture_mode()
                and forward_batch.forward_mode.is_decode_or_idle()
                and q_lora is not None
            ):
                current_stream = torch.cuda.current_stream()
                self.alt_stream.wait_stream(current_stream)
                with torch.cuda.stream(self.alt_stream):
                    k_nope = k_nope.unsqueeze(1)
                    q = self.q_b_proj(q)[0].view(
                        -1, self.num_local_heads, self.qk_head_dim
                    )
                if not self.skip_topk or (self.is_nextn and prev_topk_indices is None):
                    topk_indices = self.indexer(
                        x=hidden_states,
                        q_lora=q_lora,
                        positions=positions,
                        forward_batch=forward_batch,
                        layer_id=self.layer_id,
                    )
                else:
                    topk_indices = maybe_capture_indexer_topk(
                        self.layer_id, prev_topk_indices
                    )
                current_stream.wait_stream(self.alt_stream)
            else:
                if not C._use_fused_rmsnorm_rope:
                    k_nope = k_nope.unsqueeze(1)
                if not C._use_fused_rms_quant:
                    q = self.q_b_proj(q)[0].view(-1, self.num_local_heads, self.qk_head_dim)
                else:
                    q = self.q_b_proj(q, rms_weight=self.q_a_layernorm.weight.data, residual=None,
                                    update_hd=False)[0].view(-1, self.num_local_heads, self.qk_head_dim)
                if q_lora is not None:
                    if not self.skip_topk or (
                        self.is_nextn and prev_topk_indices is None
                    ):
                        topk_indices = self.indexer(
                            x=hidden_states,
                            q_lora=q_lora,
                            positions=positions,
                            forward_batch=forward_batch,
                            layer_id=self.layer_id,
                        )
                    else:
                        topk_indices = maybe_capture_indexer_topk(
                            self.layer_id, prev_topk_indices
                        )
        else:
            q = self.q_proj(hidden_states)[0].view(
                -1, self.num_local_heads, self.qk_head_dim
            )
            latent_cache = self.kv_a_proj_with_mqa(hidden_states)[0]
            k_nope = latent_cache[..., : self.kv_lora_rank]
            if not C._use_fused_rmsnorm_rope:
                k_nope = self.kv_a_layernorm(k_nope).unsqueeze(1)

        q_nope, q_pe = q.split([self.qk_nope_head_dim, self.qk_rope_head_dim], dim=-1)
        k_pe = latent_cache[..., self.kv_lora_rank :].unsqueeze(1)
        if C._use_fused_rmsnorm_rope:
            k_nope_normed = torch.empty(k_nope.shape, dtype=k_nope.dtype, device=k_nope.device)
            weight = self.kv_a_layernorm.weight
            variance_epsilon = self.kv_a_layernorm.variance_epsilon
            cos_sin_cache = self.rotary_emb.cos_sin_cache
            kv_pool = forward_batch.token_to_kv_pool
            # 暂不注册自定义算子，多很多kernel影响性能
            C.fused_rms_norm_rope_contiguous(positions, q_pe, k_pe.squeeze(1), k_nope,
                                          k_nope_normed, weight, cos_sin_cache,
                                          forward_batch.out_cache_loc,
                                          kv_pool.kv_buffer[self.layer_id - kv_pool.start_layer],
                                          C.dtype_mapping.get(kv_pool.dtype),1.0,False,variance_epsilon)
            k_nope = k_nope.unsqueeze(1)
        if self.use_deep_gemm_bmm:
            q_nope_val, q_nope_scale, masked_m, expected_m, aligned_m = (
                per_token_group_quant_mla_deep_gemm_masked_fp8(q_nope.transpose(0, 1))
            )
            q_nope_out = q_nope.new_empty(
                (self.num_local_heads, aligned_m, self.kv_lora_rank)
            )
            deep_gemm_wrapper.grouped_gemm_nt_f8f8bf16_masked(
                (q_nope_val, q_nope_scale),
                (self.w_kc, self.w_scale_k),
                q_nope_out,
                masked_m,
                expected_m,
            )
            q_nope_out = q_nope_out[:, :expected_m, :]
        elif C._is_hip:
            # TODO(haishaw): add bmm_fp8 to ROCm
            if C._use_aiter_gfx95 and self.w_kc.dtype == torch.uint8:
                x = q_nope.transpose(0, 1)
                q_nope_out = torch.empty(
                    x.shape[0],
                    x.shape[1],
                    self.w_kc.shape[2],
                    device=x.device,
                    dtype=torch.bfloat16,
                )
                batched_gemm_afp4wfp4_pre_quant(
                    x,
                    self.w_kc.transpose(-2, -1),
                    self.w_scale_k.transpose(-2, -1),
                    torch.bfloat16,
                    q_nope_out,
                )
            else:
                # q_nope_out = ds_bmm_wrapper(q_nope, self.w_kc, self.w_scale, torch.bfloat16)
                 q_nope_out = torch.bmm(
                    q_nope.to(torch.bfloat16).transpose(0, 1),
                    self.w_kc.to(torch.bfloat16) * self.w_scale,
                )
        elif self.w_kc.dtype == torch.float8_e4m3fn and not C._is_dcu:
            # fix bmm_fp8 error under cublas12.9 caused by bumpallocator, detail in pr#11612
            q_nope_val, q_nope_scale = per_tensor_quant_mla_fp8(
                q_nope.transpose(0, 1),
                (
                    torch.zeros((1,), dtype=torch.float32, device=q_nope.device)
                    if _is_cublas_ge_129
                    else zero_allocator.allocate(1)
                ),
            )
            q_nope_out = bmm_fp8(
                q_nope_val, self.w_kc, q_nope_scale, self.w_scale, torch.bfloat16
            )
        else:
            # q_t =q_nope.transpose(0, 1).to(torch.bfloat16)
            #     w_vc_t = self.w_vc.contiguous().to(torch.float32).to(torch.bfloat16)
            #     torch.bmm(
            #         q_t,
            #         w_vc_t
            #     )
            q_nope_out = torch.bmm(q_nope.transpose(0, 1).to(torch.bfloat16), self.w_kc.to(torch.bfloat16) * self.w_scale)
        q_nope_out = q_nope_out.transpose(0, 1)
        
        if (
            self.rotary_emb is not None
            and (not self._fuse_rope_for_trtllm_mla(forward_batch))
            and (not C._use_aiter or not C._is_gfx95_supported or self.use_nsa)
            and not C._use_fused_rmsnorm_rope
        ):
            q_pe, k_pe = self.rotary_emb(positions, q_pe, k_pe)

        if nsa_use_prefill_cp(forward_batch):
            # support allgather+rerrange
            k_nope, k_pe = self.rebuild_cp_kv_cache(
                latent_cache, forward_batch, k_nope, k_pe
            )
        return (
            q_pe,
            k_pe,
            q_nope_out,
            k_nope,
            forward_batch,
            zero_allocator,
            positions,
            topk_indices,
            llama_4_scaling,
            # k if k is not None else None,
        )

def forward_absorb_core(
        self,
        q_pe,
        k_pe,
        q_nope_out,
        k_nope,
        forward_batch,
        zero_allocator,
        positions,
        topk_indices,
        llama_4_scaling,
        k=None,
    ):
        save_kv_cache = True
        if self.current_attention_backend in C.FORWARD_ABSORB_CORE_ATTENTION_BACKENDS:
            extra_args = {}
            if self._fuse_rope_for_trtllm_mla(forward_batch):
                extra_args = {
                    "cos_sin_cache": self.rotary_emb.cos_sin_cache,
                    "is_neox": self.rotary_emb.is_neox_style,
                    "llama_4_scaling": llama_4_scaling,
                }

            attn_output = self.attn_mqa(
                q_nope_out,
                k_nope,
                k_nope,
                forward_batch,
                q_rope=q_pe,
                k_rope=k_pe,
                **extra_args,
                **(dict(topk_indices=topk_indices) if topk_indices is not None else {}),
            )
        else:
            if C._use_aiter_gfx95:
                cos = self.rotary_emb.cos_cache
                sin = self.rotary_emb.sin_cache

                kv_cache_dtype = (
                    fp8_dtype if self.kv_cache_dtype == "fp8_e4m3" else q_nope_out.dtype
                )

                q, _, _, k = fused_qk_rope_cat_and_cache_mla(
                    q_nope_out,
                    q_pe,
                    k_nope,
                    k_pe,
                    forward_batch.token_to_kv_pool.get_key_buffer(
                        self.attn_mqa.layer_id
                    ),
                    forward_batch.out_cache_loc,
                    positions,
                    cos,
                    sin,
                    self.attn_mqa.k_scale,
                    self.rotary_emb.is_neox_style,
                    q_out_dtype=kv_cache_dtype,
                )

                save_kv_cache = False
                attn_output = self.attn_mqa( 
                    q,
                    k,
                    k_nope,
                    forward_batch,
                    **(dict(topk_indices=topk_indices) if topk_indices is not None else {}),
                )
            elif C._use_fused_mla_cat:
                save_kv_cache = False
                attn_output = self.attn_mqa( 
                    q_nope_out,
                    k_nope,
                    k_nope,
                    forward_batch,
                    q_rope=q_pe,
                    k_rope=k_pe,
                    **(dict(topk_indices=topk_indices) if topk_indices is not None else {}),
                )
            else:
                if C._use_opt_cat_decode and q_nope_out.shape[0] < 1024:
                    q = concat_decode_opt(q_nope_out, q_pe, dim=2)
                    # k = torch.cat([k_nope, k_pe], dim=-1)  # TODO: remove when use fused rms rope set kv

                    attn_output = self.attn_mqa(
                        q,
                        k_nope,
                        k_nope,
                        forward_batch,
                        k_rope=k_pe,
                        **(dict(topk_indices=topk_indices) if topk_indices is not None else {}),
                    )

                else:
                    q = torch.cat([q_nope_out, q_pe], dim=-1)
                    k = torch.cat([k_nope, k_pe], dim=-1)

                    attn_output = self.attn_mqa(
                        q,
                        k,
                        k_nope,
                        forward_batch,
                        **(dict(topk_indices=topk_indices) if topk_indices is not None else {}),
                    )

            # Apply llama 4 scaling if provided
            if llama_4_scaling is not None:
                q *= llama_4_scaling

                attn_output = self.attn_mqa(
                    q,
                    k,
                    k_nope,
                    forward_batch,
                    save_kv_cache=save_kv_cache,
                    **(dict(topk_indices=topk_indices) if topk_indices is not None else {}),
                )
        attn_output = attn_output.view(-1, self.num_local_heads, self.kv_lora_rank)
        if self.use_deep_gemm_bmm:
            attn_output_val, attn_output_scale, masked_m, expected_m, aligned_m = (
                per_token_group_quant_mla_deep_gemm_masked_fp8(
                    attn_output.transpose(0, 1)
                )
            )
            attn_bmm_output = attn_output.new_empty(
                (self.num_local_heads, aligned_m, self.v_head_dim)
            )
            deep_gemm_wrapper.grouped_gemm_nt_f8f8bf16_masked(
                (attn_output_val, attn_output_scale),
                (self.w_vc, self.w_scale_v),
                attn_bmm_output,
                masked_m,
                expected_m,
            )
            attn_bmm_output = (
                attn_bmm_output[:, :expected_m, :].transpose(0, 1).flatten(1, 2)
            )
        elif C._is_hip:
            # TODO(haishaw): add bmm_fp8 to ROCm
            if C._use_aiter_gfx95 and self.w_vc.dtype == torch.uint8:
                x = attn_output.transpose(0, 1)
                attn_bmm_output = torch.empty(
                    x.shape[0],
                    x.shape[1],
                    self.w_vc.shape[2],
                    device=x.device,
                    dtype=torch.bfloat16,
                )
                batched_gemm_afp4wfp4_pre_quant(
                    x,
                    self.w_vc.transpose(-2, -1),
                    self.w_scale_v.transpose(-2, -1),
                    torch.bfloat16,
                    attn_bmm_output,
                )
            else:
                # attn_bmm_output = ds_bmm_wrapper(attn_output, self.w_vc, self.w_scale, torch.bfloat16)
                attn_bmm_output = torch.bmm(
                    attn_output.to(torch.bfloat16).transpose(0, 1),
                    self.w_vc.to(torch.bfloat16) * self.w_scale,
                )

            if self.o_proj.weight.dtype == torch.uint8:
                attn_bmm_output = attn_bmm_output.transpose(0, 1)
                attn_bmm_output = fused_flatten_mxfp4_quant(attn_bmm_output)
            # elif self.o_proj.weight.dtype == torch.float8_e4m3fn:
            #     attn_bmm_output = attn_bmm_output.transpose(0, 1)
            #     attn_bmm_output = fused_flatten_fp8_group_quant(
            #         attn_bmm_output, group_size=128, dtype_quant=torch.float8_e4m3fn
            #     )
            else:
                attn_bmm_output = attn_bmm_output.transpose(0, 1).flatten(1, 2)

        # elif self.w_vc.dtype == torch.float8_e4m3fn:
        #     attn_output_val, attn_output_scale = per_tensor_quant_mla_fp8(
        #         attn_output.transpose(0, 1),
        #         (
        #             torch.zeros((1,), dtype=torch.float32, device=attn_output.device)
        #             if _is_cublas_ge_129
        #             else zero_allocator.allocate(1)
        #         ),
        #     )
        #     attn_bmm_output = bmm_fp8(
        #         attn_output_val,
        #         self.w_vc,
        #         attn_output_scale,
        #         self.w_scale,
        #         torch.bfloat16,
        #     )
        #     attn_bmm_output = attn_bmm_output.transpose(0, 1).flatten(1, 2)
        else:
            if is_in_piecewise_cuda_graph():
                # torch dynamo requires out= op was called where output tensor was non-contiguous
                attn_bmm_output = (
                    torch.bmm(attn_output.transpose(0, 1), self.w_vc)
                    .transpose(0, 1)
                    .flatten(1, 2)
                )
            else:
                attn_bmm_output = torch.empty(
                    (attn_output.shape[0], self.num_local_heads * self.v_head_dim),
                    dtype=attn_output.dtype,
                    device=attn_output.device,
                )
                q_t = attn_output.transpose(0, 1).to(torch.bfloat16)
                w_vc_t = self.w_vc.contiguous().to(torch.float32).to(torch.bfloat16)
                torch.bmm(
                    q_t,
                    w_vc_t,
                    out=attn_bmm_output.view(
                        -1, self.num_local_heads, self.v_head_dim
                    ).transpose(0, 1),
                )
                # torch.bmm(
                #     attn_output.transpose(0, 1).to(torch.bfloat16),
                #     self.w_vc.to(torch.bfloat16),
                #     out=attn_bmm_output.view(
                #         -1, self.num_local_heads, self.v_head_dim
                #     ).transpose(0, 1),
                # )
        output, _ = self.o_proj(attn_bmm_output)
        # 如果第一维是1才 squeeze
        if self.next_skip_topk is None:
            return output
        if not self.next_skip_topk:
            return output, None
        return output, topk_indices

def _set_mla_kv_buffer(
        self,
        latent_cache: torch.Tensor,
        kv_a: torch.Tensor,
        k_pe: torch.Tensor,
        forward_batch: ForwardBatch,
    ):
        if C._is_cuda or C._use_aiter_gfx95:
            # Save latent cache
            forward_batch.token_to_kv_pool.set_mla_kv_buffer(
                self.attn_mha, forward_batch.out_cache_loc, kv_a.unsqueeze(1), k_pe
            )
        elif C._is_npu:
            # To reduce a time-costing split operation
            forward_batch.token_to_kv_pool.set_kv_buffer(
                self.attn_mha, forward_batch.out_cache_loc, kv_a.unsqueeze(1), k_pe
            )
        else:
            latent_cache[:, :, : self.kv_lora_rank] = kv_a.unsqueeze(1)
            latent_cache[:, :, self.kv_lora_rank :] = k_pe

            # Save latent cache
            forward_batch.token_to_kv_pool.set_kv_buffer(
                self.attn_mha, forward_batch.out_cache_loc, latent_cache, None
            )

def _get_mla_kv_buffer(
        self,
        kv_indices: torch.Tensor,
        dst_dtype: torch.dtype,
        forward_batch: ForwardBatch,
    ):
        if C._is_cuda or C._use_aiter_gfx95:
            kv_a, k_pe = forward_batch.token_to_kv_pool.get_mla_kv_buffer(
                self.attn_mha, kv_indices, dst_dtype
            )
            kv_a = kv_a.squeeze(1)
        else:
            latent_cache_buf = forward_batch.token_to_kv_pool.get_key_buffer(
                self.attn_mha.layer_id
            )
            latent_cache = latent_cache_buf[kv_indices].contiguous().to(dst_dtype)

            kv_a, k_pe = latent_cache.split(
                [self.kv_lora_rank, self.qk_rope_head_dim], dim=-1
            )
            kv_a = kv_a.squeeze(1).contiguous()
        return kv_a, k_pe

def _get_mla_kv_buffer_from_fp8(
        self,
        forward_batch: ForwardBatch,
    ):
        """
        Dequantize FP8 KV cache to BF16 for MLA attention (NSA-specific format).

        Returns: (kv_a, k_pe) both in BF16
        """
        backend = forward_batch.attn_backend
        if isinstance(backend, TboAttnBackend):  # if enable tbo, get primary backend
            backend = backend.primary
        kv_indices = backend.forward_metadata.page_table_1_flattened
        assert (
            kv_indices is not None
        ), "page_table_1_flattened should have been generated for FP8 MHA path"

        kv_cache_fp8 = forward_batch.token_to_kv_pool.get_key_buffer(
            self.attn_mha.layer_id
        )

        kv_latent_bf16 = dequantize_k_cache_paged(kv_cache_fp8, kv_indices)

        kv_a = kv_latent_bf16[:, :, : self.kv_lora_rank].squeeze(1).contiguous()
        k_pe = kv_latent_bf16[:, :, self.kv_lora_rank :]

        return kv_a, k_pe

def _concat_and_cast_mha_k(self, k_nope, k_pe, forward_batch):
        # Temporary for DeepSeek V3/R1 only, but can generalize if needed
        k_shape = (k_nope.shape[0], self.num_local_heads, self.qk_head_dim)
        if (
            C._is_cuda
            and (self.num_local_heads == 128)
            and (self.qk_nope_head_dim == 128)
            and (self.qk_rope_head_dim == 64)
        ):
            k = k_nope.new_empty(*k_shape)
            concat_mla_k(k=k, k_nope=k_nope, k_rope=k_pe)
        elif C._is_cuda:
            # fa3 mha support fp8 inputs
            if (
                self.current_attention_backend == "fa3"
                and self.kv_cache_dtype != "auto"
            ):
                attn_dtype = forward_batch.token_to_kv_pool.dtype
            else:
                attn_dtype = k_nope.dtype
            k = k_nope.new_empty(*k_shape, dtype=attn_dtype)
            concat_and_cast_mha_k_triton(k, k_nope, k_pe)
        elif C._is_hip and self.current_attention_backend == "aiter":
            k = k_nope.new_empty(*k_shape)
            concat_and_cast_mha_k_triton(k, k_nope, k_pe)
        else:
            k = k_nope.new_empty(*k_shape)
            k[..., : self.qk_nope_head_dim] = k_nope
            k[..., self.qk_nope_head_dim :] = k_pe
        return k

def _chunked_prefix_attn_mha(
        self,
        q: torch.Tensor,
        accum_output: torch.Tensor,
        accum_lse: torch.Tensor,
        forward_batch: ForwardBatch,
    ) -> torch.Tensor:

        assert forward_batch.num_prefix_chunks is not None
        for i in range(forward_batch.num_prefix_chunks):
            forward_batch.set_prefix_chunk_idx(i)

            kv_indices = forward_batch.prefix_chunk_kv_indices[i]
            # Fetch latent cache from memory pool with precomputed chunked kv indices
            latent_cache_buf, dtype = forward_batch.token_to_kv_pool.get_key_buffer_DeepSeekV2(
                self.attn_mha.layer_id
            )
            latent_cache = (
                latent_cache_buf[forward_batch.prefix_chunk_kv_indices[i]]
                .contiguous()
                .view(dtype)
                .to(q.dtype)
            )

            kv_a_normed, k_pe = latent_cache.split(
                [self.kv_lora_rank, self.qk_rope_head_dim], dim=-1
            )
            kv_a_normed = kv_a_normed.squeeze(1).contiguous()
            kv = self.kv_b_proj(kv_a_normed)[0]
            kv = kv.view(
                -1, self.num_local_heads, self.qk_nope_head_dim + self.v_head_dim
            )
            v = kv[..., self.qk_nope_head_dim :]
            k_nope = kv[..., : self.qk_nope_head_dim]

            k = torch.empty(
                (
                    k_nope.shape[0],
                    self.num_local_heads,
                    self.qk_nope_head_dim + self.qk_rope_head_dim,
                ),
                dtype=v.dtype,
                device=v.device,
            )
            k[..., : self.qk_nope_head_dim] = k_nope
            k[..., self.qk_nope_head_dim :] = k_pe

            output, lse = self.attn_mha(q, k, v, forward_batch, save_kv_cache=False)
            tmp_output = torch.empty_like(accum_output)
            tmp_lse = torch.empty_like(accum_lse)
            merge_state_v2(output, lse, accum_output, accum_lse, tmp_output, tmp_lse)
            accum_output, accum_lse = tmp_output, tmp_lse
            del kv, k, v, output, lse, tmp_output, tmp_lse

        return accum_output

def forward_normal_chunked_kv_prepare(
        self,
        positions: torch.Tensor,
        hidden_states: torch.Tensor,
        forward_batch: ForwardBatch,
        zero_allocator: BumpAllocator,
    ):
        # In normal mha, the k and v tensors will become overly large when the prefix length is long.
        # To avoid this, we split the kv cache into chunks and process them one after another.
        # Since mha is compute friendly, the for loop induced here will not introduce significant overhead.
        # The top comments in https://github.com/vllm-project/vllm/blob/main/vllm/v1/attention/backends/mla/common.py
        # will be helpful for understanding the purpose of this function.

        # First do normal mha forward to get output for extended part
        return self.forward_normal_prepare(
            positions, hidden_states, forward_batch, zero_allocator
        )

def forward_normal_chunked_kv_core(self, q, k, v, forward_batch):
        has_extend_prefix = forward_batch.extend_prefix_lens_cpu is not None and any(
            forward_batch.extend_prefix_lens_cpu
        )
        # Only initialize the info once
        if has_extend_prefix and forward_batch.num_prefix_chunks is None:
            forward_batch.prepare_chunked_prefix_cache_info(q.device)
            if hasattr(forward_batch.attn_backend, "init_mha_chunk_metadata"):
                forward_batch.attn_backend.init_mha_chunk_metadata(forward_batch)

        forward_batch.mha_return_lse = has_extend_prefix
        # Do mha for extended part without prefix
        forward_batch.set_attn_attend_prefix_cache(False)
        attn_output = self.attn_mha(q, k, v, forward_batch, save_kv_cache=False)
        
        # Do mha attention with chunked prefix cache if there are any sequence with prefix
        if has_extend_prefix:
            attn_output, lse = attn_output
            forward_batch.set_attn_attend_prefix_cache(True)
            attn_output = self._chunked_prefix_attn_mha(
                q=q,
                accum_output=attn_output,
                accum_lse=lse,
                forward_batch=forward_batch,
            )

        attn_output = attn_output.reshape(-1, self.num_local_heads * self.v_head_dim)
        output, _ = self.o_proj(attn_output)
        return output

def forward_normal_one_shot_prepare(
        self,
        positions: torch.Tensor,
        hidden_states: torch.Tensor,
        forward_batch: ForwardBatch,
        zero_allocator: BumpAllocator,
    ):
        forward_batch.mha_one_shot = True
        return self.forward_normal_prepare(
            positions, hidden_states, forward_batch, zero_allocator
        )

def forward_normal_one_shot_core(self, q, k, v, forward_batch):
        has_extend_prefix = any(forward_batch.extend_prefix_lens_cpu)
        # Only initialize the info once
        if has_extend_prefix and forward_batch.num_prefix_chunks is None:
            forward_batch.num_prefix_chunks = 0
            if hasattr(forward_batch.attn_backend, "init_mha_chunk_metadata"):
                forward_batch.attn_backend.init_mha_chunk_metadata(forward_batch)
        forward_batch.mha_return_lse = False
        # Do mha for extended part without prefix
        forward_batch.set_attn_attend_prefix_cache(False)
        return self.forward_normal_core(q, k, v, forward_batch)

def forward_absorb_fused_mla_rope_prepare(
        self,
        positions: torch.Tensor,
        hidden_states: torch.Tensor,
        forward_batch: ForwardBatch,
        zero_allocator: BumpAllocator,
    ):
        enable_rope_fusion = (
            os.getenv("SGLANG_FUSED_MLA_ENABLE_ROPE_FUSION", "1") == "1"
        )
        # NOTE: hidden_states can be a tuple for some quantization paths.
        # For shape/device/dtype, use the first tensor; still pass the original
        # hidden_states through linear ops which may accept tuple inputs.
        hidden_states_tensor = (
            hidden_states[0] if isinstance(hidden_states, tuple) else hidden_states
        )

        q_len = hidden_states_tensor.shape[0]
        q_input = hidden_states_tensor.new_empty(
            q_len, self.num_local_heads, self.kv_lora_rank + self.qk_rope_head_dim
        )
        if self.q_lora_rank is not None:
            q, latent_cache = self.fused_qkv_a_proj_with_mqa(hidden_states)[0].split(
                [self.q_lora_rank, self.kv_lora_rank + self.qk_rope_head_dim], dim=-1
            )
            q = self.q_a_layernorm(q)
            q = self.q_b_proj(q)[0].view(-1, self.num_local_heads, self.qk_head_dim)
        else:
            q = self.q_proj(hidden_states)[0].view(
                -1, self.num_local_heads, self.qk_head_dim
            )
            latent_cache = self.kv_a_proj_with_mqa(hidden_states)[0]
        q_nope, q_pe = q.split([self.qk_nope_head_dim, self.qk_rope_head_dim], dim=-1)

        if C._is_hip:
            # TODO(haishaw): add bmm_fp8 to ROCm
            q_nope_out = torch.bmm(
                q_nope.to(torch.bfloat16).transpose(0, 1),
                self.w_kc.to(torch.bfloat16) * self.w_scale,
            )
        elif self.w_kc.dtype == torch.float8_e4m3fn:
            q_nope_val, q_nope_scale = per_tensor_quant_mla_fp8(
                q_nope.transpose(0, 1),
                zero_allocator.allocate(1),
                dtype=torch.float8_e4m3fn,
            )
            q_nope_out = bmm_fp8(
                q_nope_val, self.w_kc, q_nope_scale, self.w_scale, torch.bfloat16
            )
        else:
            q_nope_out = torch.bmm(q_nope.transpose(0, 1), self.w_kc)
        q_input[..., : self.kv_lora_rank] = q_nope_out.transpose(0, 1)
        v_input = latent_cache[..., : self.kv_lora_rank]
        v_input = self.kv_a_layernorm(v_input.contiguous()).unsqueeze(1)
        k_input = latent_cache.unsqueeze(1)
        k_input[..., : self.kv_lora_rank] = v_input

        if not enable_rope_fusion:
            k_pe = k_input[..., self.kv_lora_rank :]
            q_pe, k_pe = self.rotary_emb(positions, q_pe, k_pe)
            q_input[..., self.kv_lora_rank :] = q_pe
            k_input[..., self.kv_lora_rank :] = k_pe
            k_pe_output = None
        else:
            k_pe_output = torch.empty_like(k_input[..., self.kv_lora_rank :])

        q_input[..., self.kv_lora_rank :] = q_pe

        # attn_output = self.attn_mqa(q_input, k_input, v_input, forward_batch)
        # Use Fused ROPE with use_rope=OFF.
        attn_output = torch.empty(
            (q_len, self.num_local_heads, self.kv_lora_rank),
            dtype=q.dtype,
            device=q.device,
        )
        attn_logits, _, kv_indptr, kv_indices, _, _, _ = (
            forward_batch.attn_backend.forward_metadata
        )
        cos_sin_cache = self.rotary_emb.cos_sin_cache
        num_kv_split = forward_batch.attn_backend.num_kv_splits
        sm_scale = self.attn_mqa.scaling
        if attn_logits is None:
            attn_logits = torch.empty(
                (
                    forward_batch.batch_size,
                    self.num_local_heads,
                    num_kv_split,
                    self.kv_lora_rank + 1,
                ),
                dtype=torch.float32,
                device=q.device,
            )

        # save current latent cache.
        forward_batch.token_to_kv_pool.set_kv_buffer(
            self.attn_mqa, forward_batch.out_cache_loc, k_input, None
        )
        key_cache_buf = forward_batch.token_to_kv_pool.get_key_buffer(
            self.attn_mqa.layer_id
        )
        val_cache_buf = key_cache_buf[..., : self.kv_lora_rank]

        return (
            q_input,
            key_cache_buf,
            val_cache_buf,
            attn_output,
            kv_indptr,
            kv_indices,
            k_pe_output,
            cos_sin_cache,
            positions,
            attn_logits,
            num_kv_split,
            sm_scale,
            enable_rope_fusion,
            k_input,
            forward_batch,
            zero_allocator,
        )

def forward_absorb_fused_mla_rope_core(
        self,
        q_input,
        key_cache_buf,
        val_cache_buf,
        attn_output,
        kv_indptr,
        kv_indices,
        k_pe_output,
        cos_sin_cache,
        positions,
        attn_logits,
        num_kv_split,
        sm_scale,
        enable_rope_fusion,
        k_input,
        forward_batch,
        zero_allocator,
    ):
        decode_attention_fwd_grouped_rope(
            q_input,
            key_cache_buf,
            val_cache_buf,
            attn_output,
            kv_indptr,
            kv_indices,
            k_pe_output,
            self.kv_lora_rank,
            self.rotary_emb.rotary_dim,
            cos_sin_cache,
            positions,
            attn_logits,
            num_kv_split,
            sm_scale,
            logit_cap=self.attn_mqa.logit_cap,
            use_rope=enable_rope_fusion,
            is_neox_style=self.rotary_emb.is_neox_style,
        )

        if enable_rope_fusion:
            k_input[..., self.kv_lora_rank :] = k_pe_output
            forward_batch.token_to_kv_pool.set_kv_buffer(
                self.attn_mqa, forward_batch.out_cache_loc, k_input, None
            )

        attn_output = attn_output.view(-1, self.num_local_heads, self.kv_lora_rank)

        if C._is_hip:
            # TODO(haishaw): add bmm_fp8 to ROCm
            attn_bmm_output = torch.bmm(
                attn_output.to(torch.bfloat16).transpose(0, 1),
                self.w_vc.to(torch.bfloat16) * self.w_scale,
            )
        elif self.w_vc.dtype == torch.float8_e4m3fn:
            attn_output_val, attn_output_scale = per_tensor_quant_mla_fp8(
                attn_output.transpose(0, 1),
                zero_allocator.allocate(1),
                dtype=torch.float8_e4m3fn,
            )
            attn_bmm_output = bmm_fp8(
                attn_output_val,
                self.w_vc,
                attn_output_scale,
                self.w_scale,
                torch.bfloat16,
            )
        else:
            attn_bmm_output = torch.bmm(attn_output.transpose(0, 1), self.w_vc)
        attn_output = attn_bmm_output.transpose(0, 1).flatten(1, 2)
        output, _ = self.o_proj(attn_output)

        return output

def forward_absorb_fused_mla_rope_cpu_prepare(
        self,
        positions: torch.Tensor,
        hidden_states: torch.Tensor,
        forward_batch: ForwardBatch,
        zero_allocator: BumpAllocator,
    ):
        assert self.q_lora_rank is not None and use_intel_amx_backend(
            self
        ), "forward_absorb_fused_mla_rope_cpu_prepare requires q_lora_rank is not None and use_intel_amx_backend"

        q_input, k_input, v_input = (
            torch.ops.sgl_kernel.qkv_proj_with_rope_fused_weight(
                hidden_states,
                self.fused_qkv_a_proj_with_mqa.weight,
                self.q_b_proj.weight,
                self.w_kc,
                self.q_a_layernorm.weight,
                self.kv_a_layernorm.weight,
                positions,
                self.rotary_emb.cos_sin_cache,
                self.kv_a_layernorm.variance_epsilon,
                self.qkv_proj_with_rope_is_int8,
                self.qkv_proj_with_rope_is_fp8,
                (
                    self.fused_qkv_a_proj_with_mqa.weight_scale
                    if self.qkv_proj_with_rope_is_int8
                    else (
                        self.fused_qkv_a_proj_with_mqa.weight_scale_inv
                        if self.qkv_proj_with_rope_is_fp8
                        else None
                    )
                ),
                (
                    self.q_b_proj.weight_scale
                    if self.qkv_proj_with_rope_is_int8
                    else (
                        self.q_b_proj.weight_scale_inv
                        if self.qkv_proj_with_rope_is_fp8
                        else None
                    )
                ),
                True,  # is_vnni
                self.weight_block_size,
                self.q_lora_rank,
                self.kv_lora_rank,
                self.qk_rope_head_dim,
            )
        )
        return (q_input, k_input, v_input, forward_batch, zero_allocator)

def forward_absorb_fused_mla_rope_cpu_core(
        self, q_input, k_input, v_input, forward_batch, zero_allocator
    ):
        assert self.q_lora_rank is not None and use_intel_amx_backend(
            self
        ), "forward_absorb_fused_mla_rope_cpu_core requires q_lora_rank is not None and use_intel_amx_backend"

        attn_output = self.attn_mqa(q_input, k_input, v_input, forward_batch)
        attn_output = attn_output.view(-1, self.num_local_heads, self.kv_lora_rank)

        # [Note] Align shapes of bmm inputs.
        # Shapes of inputs:
        #   q_nope: [M, B, K]
        #   original self.w_kc: [B, K, N]
        #   current self.w_kc (which has been converted in PackWeightMethod): [B, N, K]

        # Shapes of inputs to sgl_kernel.cpu.bmm:
        #   out: [B, M, N]
        #   mat1: [B, M, K]
        #   mat2: [B, N, K]
        B = self.w_vc.size(0)
        N = self.w_vc.size(1)
        M = attn_output.size(0)
        output = torch.empty([M, int(B * N)], dtype=attn_output.dtype)
        attn_bmm_output = output.view([M, B, N]).transpose_(0, 1)
        torch.ops.sgl_kernel.bmm_cpu(
            attn_bmm_output,
            attn_output.transpose(0, 1),
            self.w_vc,
            True,  # is_vnni
            None,  # scale
        )
        attn_output = output
        output, _ = self.o_proj(attn_output)

        return output

