"""DCU DeepSeek V2 fork methods (auto-extracted for plugin hooks)."""

from __future__ import annotations

from sglang_dcu_plugin.models.deepseek.deepseek_v2 import constants as C
from sglang_dcu_plugin.models.deepseek.deepseek_v2.methods._runtime import *  # noqa: F403

def __init__(
        self,
        config: PretrainedConfig,
        layer_id: int,
        quant_config: Optional[QuantizationConfig] = None,
        prefix: str = "",
        alt_stream: Optional[torch.cuda.Stream] = None,
        is_nextn: bool = False,
        is_deepseek_v4: bool = False,
    ):
        super().__init__()
        self.tp_size = get_tensor_model_parallel_world_size()
        self.moe_ep_size = get_moe_expert_parallel_world_size()
        self.routed_scaling_factor = config.routed_scaling_factor
        self.n_shared_experts = config.n_shared_experts

        n_shared_experts = (
            0 if config.n_shared_experts is None else int(config.n_shared_experts)
        )
        _fusion_disabled = get_global_server_args().disable_shared_experts_fusion

        # num_fused_shared_experts drives weight remapping in deepseek_weight_loader:
        # mlp.shared_experts → mlp.experts.256 when > 0.
        self.num_fused_shared_experts = 0 if _fusion_disabled else n_shared_experts

        # DeepEP shared expert fusion: shared expert is fused into the same MoE kernel
        # as a local expert at the home EP rank. Expert layout is expanded from 256
        # routed to 256+EP_size (e.g. 272 for EP=16). TopK handles interleaving.
        _is_deepep_fusion = (
            is_deepep_class_backend() and self.num_fused_shared_experts > 0
        )

        if _is_deepep_fusion:
            # 256 routed + EP_size shared slots = 272 experts total (for EP=16)
            num_experts_for_moe = config.n_routed_experts + self.moe_ep_size
            top_k_for_moe = config.num_experts_per_tok + 1  # 8 routed + 1 shared
            # Interleaving for DeepEP dispatch is handled by TopK internally.
        else:
            num_experts_for_moe = (
                config.n_routed_experts + self.num_fused_shared_experts
            )
            top_k_for_moe = config.num_experts_per_tok + self.num_fused_shared_experts

        self.config = config
        self.layer_id = layer_id
        self.alt_stream = alt_stream
        self.is_nextn = is_nextn

        n_hash_layers = getattr(config, "num_hash_layers", 0)
        self.is_hash = layer_id < n_hash_layers and not (is_deepseek_v4 and is_nextn)

        if self.tp_size > config.n_routed_experts:
            raise ValueError(
                f"Tensor parallel size {self.tp_size} is greater than "
                f"the number of experts {config.n_routed_experts}."
            )

        if config.hidden_act != "silu":
            raise ValueError(
                f"Unsupported activation: {config.hidden_act}. "
                "Only silu is supported for now."
            )

        self.gate = MoEGate(
            config=config,
            quant_config=quant_config,
            prefix=add_prefix("gate", prefix),
            is_nextn=is_nextn,
            is_hash_moe=self.is_hash,
            is_deepseek_v4=is_deepseek_v4,
        )

        # scaling factor for fused shared experts on AMD-platform.
        # DeepEP doesn't need this: shared expert is only computed on home rank
        # (not all-reduced), so no 1/ep_size correction is needed.
        fused_shared_experts_scaling_factor = None
        if (
            self.moe_ep_size > 1
            and self.num_fused_shared_experts > 0
            and not _is_deepep_fusion
        ):
            # if enable_ep_moe tp_szie == ep_size, every gpu get shared experts gemm output
            # so we scale with 1 / self.moe_ep_size in ep mode which will make it equalation as in tp mode
            # with fused_shared_experts
            fused_shared_experts_scaling_factor = 1.0 / float(self.moe_ep_size)

        self.experts = get_moe_impl_class(quant_config)(
            num_experts=num_experts_for_moe
            + get_global_server_args().ep_num_redundant_experts,
            num_fused_shared_experts=self.num_fused_shared_experts,
            top_k=top_k_for_moe,
            hidden_size=config.hidden_size,
            intermediate_size=config.moe_intermediate_size,
            layer_id=self.layer_id,
            quant_config=quant_config,
            routed_scaling_factor=self.routed_scaling_factor,
            routing_method_type=getattr(
                config, "routing_method_type", RoutingMethodType.DeepSeekV3
            ),
            swiglu_limit=getattr(config, "swiglu_limit", None),
            prefix=add_prefix("experts", prefix),
        )

        self.num_expert_group = getattr(config, "n_group", None)
        self.topk_group = getattr(config, "topk_group", None)
        self.use_grouped_topk = (
            self.num_expert_group is not None
            and self.topk_group is not None
            and self.num_expert_group > self.topk_group
        )

        if self.is_hash and not (is_nextn and is_deepseek_v4):
            self.topk = HashTopK(
                topk=config.num_experts_per_tok + self.num_fused_shared_experts,
                num_experts=config.n_routed_experts,
                num_fused_shared_experts=self.num_fused_shared_experts,
                vocab_size=config.vocab_size,
                scoring_func=config.scoring_func,
                routed_scaling_factor=self.routed_scaling_factor,
                apply_routed_scaling_factor_on_output=self.experts.should_fuse_routed_scaling_factor_in_topk,
            )
        else:
            # Default: grouped noaux_tc top-k. Covers V3/V3.2/GLM-5/Glm4MoeLite.
            topk_kwargs = dict(
                top_k=config.num_experts_per_tok + self.num_fused_shared_experts,
                layer_id=self.layer_id,
                renormalize=config.norm_topk_prob,
                use_grouped_topk=True,
                num_expert_group=config.n_group,
                num_fused_shared_experts=self.num_fused_shared_experts,
                topk_group=config.topk_group,
                correction_bias=self.gate.e_score_correction_bias,
                quant_config=quant_config,
                routed_scaling_factor=self.routed_scaling_factor,
                apply_routed_scaling_factor_on_output=self.experts.should_fuse_routed_scaling_factor_in_topk,
                fused_shared_experts_scaling_factor=fused_shared_experts_scaling_factor,
                # Some Fp4 MoE backends require the output format to be bypassed but the MTP layers are unquantized
                # and requires the output format to be standard (except trtllm). We use quant_config to determine the output format.
                output_format=(
                    TopKOutputFormat.STANDARD
                    if (quant_config is None)
                    and (not get_moe_runner_backend().is_flashinfer_trtllm())
                    else None
                ),
            )
            # DSV4 override: ungrouped sqrtsoftplus + fp4 expert layout flag.
            if is_deepseek_v4:
                topk_kwargs.update(
                    use_grouped_topk=False,
                    scoring_func=config.scoring_func,
                    is_fp4_experts=getattr(quant_config, "is_fp4_experts", False),
                )
            self.topk = TopK(**topk_kwargs)

        self.shared_experts_is_int8 = False
        self.shared_experts_is_fp8 = False
        self.shared_experts_weight_block_size = None
        self._shared_expert_tp1 = False
        # Shared experts: skip when fused into MoE kernel (self.num_fused_shared_experts > 0)
        # or when DeepEP fusion is enabled (shared expert is local slot 16 in FusedMoE, no separate MLP).
        if (
            config.n_shared_experts is not None
            and config.n_shared_experts > 0
            and self.num_fused_shared_experts == 0
            and not _is_deepep_fusion
        ):
            intermediate_size = config.moe_intermediate_size * config.n_shared_experts
            # Disable TP for shared experts for A2A/FP4 allgather paths, or when
            # explicitly requested for DSV4 checkpoints whose shared scales are
            # not divisible by the global TP size.
            _shared_expert_use_tp1 = (
                get_moe_a2a_backend().is_deepep()
                or get_moe_a2a_backend().is_mooncake()
                or get_moe_a2a_backend().is_nixl()
                or get_moe_a2a_backend().is_mori()
                or get_moe_a2a_backend().is_ascend_fuseep()
                or get_moe_a2a_backend().is_flashinfer()
                or get_moe_a2a_backend().is_megamoe()
                or should_use_flashinfer_cutlass_moe_fp4_allgather()
                or envs.SGLANG_SHARED_EXPERT_TP1.get()
            )
            self.shared_experts = DeepseekV2MLP(
                hidden_size=config.hidden_size,
                intermediate_size=intermediate_size,
                hidden_act=config.hidden_act,
                quant_config=quant_config,
                reduce_results=False,
                swiglu_limit=getattr(config, "swiglu_limit", None),
                prefix=add_prefix("shared_experts", prefix),
                **(dict(tp_rank=0, tp_size=1) if _shared_expert_use_tp1 else {}),
            )
            self._shared_expert_tp1 = _shared_expert_use_tp1
            is_packed_weight = hasattr(
                self.shared_experts.gate_up_proj.quant_method, "quant_config"
            ) and self.shared_experts.gate_up_proj.quant_method.quant_config.get_name() in {
                "awq",
                "awq_marlin",
                "moe_wna16",
            }
            self.shared_experts_is_int8 = (
                not is_packed_weight
                and self.shared_experts.gate_up_proj.weight.dtype == torch.int8
            )
            self.shared_experts_is_fp8 = (
                not is_packed_weight
                and self.shared_experts.gate_up_proj.weight.dtype == torch.float8_e4m3fn
            )
            # if self.shared_experts_is_fp8:
            #     if (
            #         C._use_aiter
            #         and config.quantization_config.get("quant_method")
            #         == "compressed-tensors"
            #     ):
            #         # For compressed-tensors ptpc model, don't need to check the weight_block_size
            #         pass
            #     else:
            #         assert (
            #             self.shared_experts.gate_up_proj.quant_method.quant_config.weight_block_size
            #             == self.shared_experts.down_proj.quant_method.quant_config.weight_block_size
            #         )
            #         self.shared_experts_weight_block_size = (
            #             self.shared_experts.gate_up_proj.quant_method.quant_config.weight_block_size
            #         )

        self.top_k = config.num_experts_per_tok

        if (
            get_moe_a2a_backend().is_deepep()
            or get_moe_a2a_backend().is_mooncake()
            or get_moe_a2a_backend().is_nixl()
            or get_moe_a2a_backend().is_mori()
            or get_moe_a2a_backend().is_ascend_fuseep()
        ):
            # TODO: we will support tp < ep in the future
            self.ep_size = get_moe_expert_parallel_world_size()
            self.num_experts = (
                config.n_routed_experts
                + get_global_server_args().ep_num_redundant_experts
            )
            self.renormalize = config.norm_topk_prob
            self.correction_bias = (
                self.gate.e_score_correction_bias.data
                if self.gate.e_score_correction_bias is not None
                else None
            )

        self._enable_a2a_moe = (
            get_moe_a2a_backend().is_deepep()
            or get_moe_a2a_backend().is_mooncake()
            or get_moe_a2a_backend().is_nixl()
            or get_moe_a2a_backend().is_mori()
            or get_moe_a2a_backend().is_ascend_fuseep()
            or get_moe_a2a_backend().is_flashinfer()
        )
        self._fuse_shared_experts_inside_sbo = SboFlags.fuse_shared_experts_inside_sbo()

def forward(
        self,
        hidden_states: torch.Tensor,
        forward_batch: Optional[ForwardBatch] = None,
        should_allreduce_fusion: bool = False,
        use_reduce_scatter: bool = False,
        gemm_output_zero_allocator: BumpAllocator = None,
        input_ids: Optional[torch.Tensor] = None,
        input_ids_global: Optional[torch.Tensor] = None,
        rms_weight: Optional[torch.Tensor] = None,
        residual: Optional[torch.Tensor] = None,
        update_hd: Optional[bool] = False,
    ) -> torch.Tensor:
        from sglang.srt.layers.moe.mega_moe import forward_mega_moe, should_use_mega_moe

        if should_use_mega_moe(self, hidden_states):
            return forward_mega_moe(
                self,
                hidden_states,
                forward_batch,
                input_ids_global=input_ids_global,
            )

        if not self._enable_a2a_moe:
            if (
                self.alt_stream is not None
                and self.num_fused_shared_experts == 0
                and hidden_states.shape[0] > 0
                and get_is_capture_mode()
                and not (
                    get_global_server_args().enable_torch_compile
                    and hidden_states.shape[0]
                    <= get_global_server_args().torch_compile_max_bs
                    * (get_global_server_args().speculative_num_draft_tokens or 1)
                )
            ):
                return self.forward_normal_dual_stream(
                    hidden_states,
                    should_allreduce_fusion,
                    use_reduce_scatter,
                    gemm_output_zero_allocator,
                    input_ids,
                    input_ids_global=input_ids_global,
                    rms_weight = rms_weight,
                    residual = residual,
                )
            else:
                return self.forward_normal(
                    hidden_states,
                    should_allreduce_fusion,
                    use_reduce_scatter,
                    gemm_output_zero_allocator,
                    input_ids,
                    input_ids_global=input_ids_global,
                    rms_weight = rms_weight,
                    residual = residual,
                )
        else:
            return self.forward_deepep(hidden_states, forward_batch, input_ids_global=input_ids_global, rms_weight=rms_weight, residual=residual,)

def forward_normal(
        self,
        hidden_states: torch.Tensor,
        should_allreduce_fusion: bool = False,
        use_reduce_scatter: bool = False,
        gemm_output_zero_allocator: BumpAllocator = None,
        input_ids: Optional[torch.Tensor] = None,
        input_ids_global: Optional[torch.Tensor] = None,
        rms_weight: Optional[torch.Tensor] = None,
        residual: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        if hasattr(self, "shared_experts") and use_intel_amx_backend(
            self.shared_experts.gate_up_proj
        ):
            return self.forward_cpu(hidden_states, should_allreduce_fusion)
        server_args = get_global_server_args()
        dispatch_info = (
            ExpertLocationDispatchInfo.init_new(layer_id=self.layer_id)
            if server_args.enable_eplb
            else None
        )
        defer_shared = not self.experts.moe_runner_config.inplace
        if hidden_states.shape[0] > 0:
            if not defer_shared and not self._fuse_shared_experts_inside_sbo:
                shared_output = self._forward_shared_experts(
                    hidden_states, gemm_output_zero_allocator,
                )
            # router_logits: (num_tokens, n_experts)
            router_logits = self.gate(hidden_states, gemm_output_zero_allocator)
            topk_kwargs = (
                {"input_ids": input_ids_global}
                if getattr(self, "is_hash", False)
                else {}
            )
            topk_output = self.topk(
                hidden_states,
                router_logits,
                expert_location_dispatch_info=dispatch_info,
                **topk_kwargs,
            )
        else:
            shared_output = None
            topk_output = self.topk.empty_topk_output(hidden_states.device)

        if C._use_lightop_moe_sum_mul_add:
            if C._use_fused_rms_quant and i_q is not None and i_s is not None:  # maybe needn't
                final_hidden_states = self.experts(hidden_states, topk_output, shared_output=shared_output, i_q=i_q, i_s=i_s)
            else:
                final_hidden_states = self.experts(hidden_states, topk_output, shared_output=shared_output)
        else:
            if self._fuse_shared_experts_inside_sbo:
                shared_output = None

                def _pre_combine_hook(
                    dispatcher: BaseDispatcher, combine_input: CombineInput
                ):

                    nonlocal shared_output
                    self.alt_stream.wait_stream(torch.cuda.current_stream())
                    with torch.cuda.stream(self.alt_stream):
                        shared_output = self._forward_shared_experts(
                            hidden_states, gemm_output_zero_allocator
                        )

                    pre_combine_hook_handle.remove()

                def _post_combine_hook(
                    dispatcher: BaseDispatcher, hidden_states: torch.Tensor
                ):
                    nonlocal shared_output
                    torch.cuda.current_stream().wait_stream(self.alt_stream)
                    post_combine_hook_handle.remove()

                pre_combine_hook_handle = self.experts.dispatcher.register_pre_combine_hook(
                    _pre_combine_hook
                )
                post_combine_hook_handle = (
                    self.experts.dispatcher.register_post_combine_hook(_post_combine_hook)
                )
            final_hidden_states = self.experts(
            hidden_states,
            topk_output,
        )
        if (
            not C._is_cuda
            and not _is_musa
            and not C._is_dcu
            and not _is_xpu
            and not C._use_aiter
            or isinstance(self.experts.quant_method, KTEPWrapperMethod)
        ):
            # fused in biased_grouped_topk so we can skip here
            final_hidden_states *= self.routed_scaling_factor

        if (
            defer_shared
            and hidden_states.shape[0] > 0
            and not self._fuse_shared_experts_inside_sbo
        ):
            shared_output = self._forward_shared_experts(
                hidden_states, gemm_output_zero_allocator
            )

        final_hidden_states = maybe_fuse_routed_scale_and_shared_add(
            self.experts,
            final_hidden_states,
            None if self._shared_expert_tp1 else shared_output,
            self.routed_scaling_factor,
        )

        if self.tp_size > 1 and not should_skip_post_experts_all_reduce(
            is_tp_path=True,
            use_reduce_scatter=use_reduce_scatter,
            should_allreduce_fusion=should_allreduce_fusion,
        ):
            final_hidden_states = tensor_model_parallel_all_reduce(final_hidden_states)
        # TP1 shared experts are replicated, so add them after all-reduce to
        # avoid summing the same shared output once per TP rank.
        if shared_output is not None and self._shared_expert_tp1:
            final_hidden_states += shared_output
        return final_hidden_states

def forward_deepep(
        self,
        hidden_states: torch.Tensor,
        forward_batch: ForwardBatch,
        input_ids_global: Optional[torch.Tensor] = None,
        rms_weight: Optional[torch.Tensor] = None,
        residual: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        shared_output = None
        sbo_enabled_flag = self._fuse_shared_experts_inside_sbo and not self.is_nextn
        sbo_overlap_dispatch_flag = (
            sbo_enabled_flag and SboFlags.enable_dispatch_shared_one_stream_overlap()
        )
        sbo_overlap_combine_flag = (
            sbo_enabled_flag and SboFlags.enable_combine_shared_two_stream_overlap()
        )

        if hidden_states.shape[0] > 0:
            i_q = None
            i_s = None
            # router_logits: (num_tokens, n_experts)
            router_logits = self.gate(hidden_states, forward_batch=forward_batch)
            if not sbo_enabled_flag and self.num_fused_shared_experts == 0:
                if self.alt_stream is not None:
                    self.alt_stream.wait_stream(torch.cuda.current_stream())
                    with torch.cuda.stream(self.alt_stream):
                        if C._use_fused_rms_quant:
                            shared_output, new_resi, i_q, i_s = self._forward_shared_experts(hidden_states, rms_weight=rms_weight, residual=residual)
                        else:
                            shared_output = self._forward_shared_experts(hidden_states)
                        shared_output.record_stream(self.alt_stream)
                        shared_event = self.alt_stream.record_event()
                else:
                    if C._use_fused_rms_quant:
                        shared_output, new_resi, i_q, i_s = self._forward_shared_experts(hidden_states, rms_weight=rms_weight, residual=residual)
                    else:
                        shared_output = self._forward_shared_experts(hidden_states)
            topk_kwargs = (
                {"input_ids": input_ids_global}
                if getattr(self, "is_hash", False)
                else {}
            )
            topk_output = self.topk(
                hidden_states,
                router_logits,
                num_token_non_padded=forward_batch.num_token_non_padded,
                expert_location_dispatch_info=ExpertLocationDispatchInfo.init_new(
                    layer_id=self.layer_id,
                ),
                **topk_kwargs,
            )
        else:
            topk_output = self.topk.empty_topk_output(hidden_states.device)

        if sbo_overlap_dispatch_flag:
            shared_output = None

            def _deepep_dispatch_hook(dispatcher: BaseDispatcher):
                nonlocal shared_output
                shared_output = self._forward_shared_experts(hidden_states)
                for handle in deepep_dispatch_hook_handle:
                    handle.remove()

            def _post_dispatch_hook(
                dispatcher: BaseDispatcher, dispatch_output: DispatchOutput
            ):
                combine_overlap_args, down_gemm_overlap_args, meta_overlap_args = (
                    compute_overlap_args(dispatch_output, self.alt_stream)
                )
                dispatcher.set_overlap_args(
                    combine_overlap_args=combine_overlap_args,
                    meta_overlap_args=meta_overlap_args,
                )
                self.experts.set_overlap_args(
                    down_gemm_overlap_args=down_gemm_overlap_args,
                    meta_overlap_args=meta_overlap_args,
                )
                post_dispatch_hook_handle.remove()

            def _post_combine_hook(
                dispatcher: BaseDispatcher, hidden_states: torch.Tensor
            ):
                dispatcher.clear_overlap_args()
                self.experts.clear_overlap_args()
                post_combine_hook_handle.remove()

            assert isinstance(self.experts.dispatcher, MaybeTboDeepEPDispatcher)
            deepep_dispatch_hook_handle = (
                self.experts.dispatcher.register_deepep_dispatch_hook(
                    _deepep_dispatch_hook
                )
            )
            post_dispatch_hook_handle = (
                self.experts.dispatcher.register_post_dispatch_hook(_post_dispatch_hook)
            )
            post_combine_hook_handle = (
                self.experts.dispatcher.register_post_combine_hook(_post_combine_hook)
            )

        elif sbo_overlap_combine_flag:
            shared_output = None

            def _post_dispatch_hook(
                dispatcher: BaseDispatcher, dispatch_output: DispatchOutput
            ):

                combine_overlap_args, down_gemm_overlap_args, meta_overlap_args = (
                    compute_overlap_args(dispatch_output, self.alt_stream)
                )
                dispatcher.set_overlap_args(
                    combine_overlap_args=combine_overlap_args,
                    meta_overlap_args=meta_overlap_args,
                )
                self.experts.set_overlap_args(
                    down_gemm_overlap_args=down_gemm_overlap_args,
                    meta_overlap_args=meta_overlap_args,
                )

                post_dispatch_hook_handle.remove()

            # NOTE: sbo not compatable with rms_quant
            def _pre_combine_hook(
                dispatcher: BaseDispatcher, combine_input: CombineInput
            ):

                nonlocal shared_output

                if (
                    e := dispatcher.meta_overlap_args.get("record_event_after_down")
                ) is not None:
                    e.record()

                # TODO reduce sm for non-deepgemm
                # with deep_gemm_wrapper.configure_deep_gemm_num_sms(
                #     dispatcher.meta_overlap_args["compute_num_sms"]
                # ):
                shared_output = self._forward_shared_experts(hidden_states)

                pre_combine_hook_handle.remove()

            def _post_combine_hook(
                dispatcher: BaseDispatcher, hidden_states: torch.Tensor
            ):
                dispatcher.clear_overlap_args()
                self.experts.clear_overlap_args()
                post_combine_hook_handle.remove()

            post_dispatch_hook_handle = (
                self.experts.dispatcher.register_post_dispatch_hook(_post_dispatch_hook)
            )
            pre_combine_hook_handle = self.experts.dispatcher.register_pre_combine_hook(
                _pre_combine_hook
            )
            post_combine_hook_handle = (
                self.experts.dispatcher.register_post_combine_hook(_post_combine_hook)
            )
        elif envs.SGLANG_BLACKWELL_OVERLAP_SHARED_EXPERTS_OUTSIDE_SBO.get():
            # On GB200: Shared experts overlapped on alt_stream, down gemm overlapped with DeepEP Combine

            def _post_dispatch_hook(
                dispatcher: BaseDispatcher, dispatch_output: DispatchOutput
            ):

                combine_overlap_args, down_gemm_overlap_args, meta_overlap_args = (
                    compute_overlap_args(dispatch_output, self.alt_stream)
                )
                dispatcher.set_overlap_args(
                    combine_overlap_args=combine_overlap_args,
                    meta_overlap_args=meta_overlap_args,
                )
                self.experts.set_overlap_args(
                    down_gemm_overlap_args=down_gemm_overlap_args,
                    meta_overlap_args=meta_overlap_args,
                )

                post_dispatch_hook_handle.remove()

            def _pre_combine_hook(
                dispatcher: BaseDispatcher, combine_input: CombineInput
            ):
                if (
                    e := dispatcher.meta_overlap_args.get("record_event_after_down")
                ) is not None:
                    e.record()
                pre_combine_hook_handle.remove()

            def _post_combine_hook(
                dispatcher: BaseDispatcher, hidden_states: torch.Tensor
            ):
                dispatcher.clear_overlap_args()
                self.experts.clear_overlap_args()
                post_combine_hook_handle.remove()

            post_dispatch_hook_handle = (
                self.experts.dispatcher.register_post_dispatch_hook(_post_dispatch_hook)
            )
            pre_combine_hook_handle = self.experts.dispatcher.register_pre_combine_hook(
                _pre_combine_hook
            )
            post_combine_hook_handle = (
                self.experts.dispatcher.register_post_combine_hook(_post_combine_hook)
            )

        final_hidden_states = self.experts(
            hidden_states=hidden_states,
            topk_output=topk_output,
        )

        if (
            hidden_states.shape[0] > 0
            and not sbo_enabled_flag
            and self.num_fused_shared_experts == 0
            and self.alt_stream is not None
        ):
            torch.cuda.current_stream().wait_event(shared_event)

        if shared_output is not None:
            x = shared_output
            # aiter moe call will handle routed_scaling_factor in the function
            # so add C._use_aiter condition to eliminate to use self.routed_scaling_factor in add_ call
            if self.experts.should_fuse_routed_scaling_factor_in_topk or C._use_aiter:
                x.add_(final_hidden_states)
            else:
                x.add_(final_hidden_states, alpha=self.routed_scaling_factor)
            final_hidden_states = x
        else:
            if not (
                self.experts.should_fuse_routed_scaling_factor_in_topk or C._use_aiter
            ):
                final_hidden_states *= self.routed_scaling_factor

        return final_hidden_states

def forward_normal_dual_stream(
        self,
        hidden_states: torch.Tensor,
        should_allreduce_fusion: bool = False,
        use_reduce_scatter: bool = False,
        gemm_output_zero_allocator: BumpAllocator = None,
        input_ids: Optional[torch.Tensor] = None,
        input_ids_global: Optional[torch.Tensor] = None,
        rms_weight: Optional[torch.Tensor] = None,
        residual: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        current_stream = torch.cuda.current_stream()
        self.alt_stream.wait_stream(current_stream)
        # shared_output = self._forward_shared_experts(
        #     hidden_states, gemm_output_zero_allocator
        # )
        i_q = None
        i_s = None
        if C._use_fused_rms_quant:
            shared_output, new_resi, i_q, i_s = self._forward_shared_experts(
                hidden_states, gemm_output_zero_allocator,
                rms_weight = rms_weight,
                residual = residual,
            )
        else:
            shared_output = self._forward_shared_experts(
                hidden_states, gemm_output_zero_allocator,
            )
        server_args = get_global_server_args()
        dispatch_info = (
            ExpertLocationDispatchInfo.init_new(layer_id=self.layer_id)
            if server_args.enable_eplb
            else None
        )
        with torch.cuda.stream(self.alt_stream):
            # router_logits: (num_tokens, n_experts)
            router_logits = self.gate(hidden_states, gemm_output_zero_allocator)
            topk_kwargs = (
                {"input_ids": input_ids_global}
                if getattr(self, "is_hash", False)
                else {}
            )
            topk_output = self.topk(
                hidden_states,
                router_logits,
                expert_location_dispatch_info=dispatch_info,
                **topk_kwargs,
            )
            final_hidden_states = self.experts(hidden_states, topk_output)
            if not (C._is_cuda or _is_musa) or isinstance(
                self.experts.quant_method, KTEPWrapperMethod
            ):
                final_hidden_states *= self.routed_scaling_factor

        current_stream.wait_stream(self.alt_stream)

        final_hidden_states = maybe_fuse_routed_scale_and_shared_add(
            self.experts,
            final_hidden_states,
            None if self._shared_expert_tp1 else shared_output,
            self.routed_scaling_factor,
        )

        if self.tp_size > 1 and not should_skip_post_experts_all_reduce(
            is_tp_path=True,
            use_reduce_scatter=use_reduce_scatter,
            should_allreduce_fusion=should_allreduce_fusion,
        ):
            final_hidden_states = tensor_model_parallel_all_reduce(final_hidden_states)
        # TP1 shared experts are replicated, so add them after all-reduce to
        # avoid summing the same shared output once per TP rank.
        if self._shared_expert_tp1:
            final_hidden_states += shared_output
        return final_hidden_states

def _forward_shared_experts(
        self,
        hidden_states,
        gemm_output_zero_allocator: BumpAllocator = None,
        rms_weight: Optional[torch.Tensor] = None,
        residual: Optional[torch.Tensor] = None,
        update_hd: Optional[bool] = True,
    ):
        if (hidden_states.shape[0] > 0) and (self.num_fused_shared_experts == 0):
            return self.shared_experts(
                hidden_states,
                gemm_output_zero_allocator=gemm_output_zero_allocator,
                rms_weight = rms_weight,
                residual = residual,
                update_hd = update_hd,
            )
        else:
            if C._use_fused_rms_quant and rms_weight is not None and residual is not None:
                return None, None, None, None
            return None
