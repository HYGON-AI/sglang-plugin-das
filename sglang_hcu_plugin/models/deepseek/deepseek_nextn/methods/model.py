"""HCU deepseek_nextn fork methods (auto-extracted for plugin hooks)."""

from __future__ import annotations

from sglang_hcu_plugin.models.deepseek.deepseek_nextn.methods._runtime import *  # noqa: F403

def __init__(
        self,
        config: PretrainedConfig,
        quant_config: Optional[QuantizationConfig] = None,
        prefix: str = "",
    ) -> None:
        super().__init__()
        if enable_nextn_moe_bf16_cast_to_fp8(quant_config):
            # refer to real DeepSeek V3 quant config
            moe_quant_config_override = Fp8Config(
                is_checkpoint_fp8_serialized=True,
                weight_block_size=[128, 128],
            )
        else:
            moe_quant_config_override = None

        if quant_config is not None and quant_config.get_name() == "modelopt_fp4":
            logger.warning(
                "Overriding DeepseekV3ForCausalLMNextN quant config for modelopt_fp4 Deepseek model."
            )
            quant_config = None

        self.vocab_size = config.vocab_size

        self.embed_tokens = VocabParallelEmbedding(
            config.vocab_size,
            config.hidden_size,
            use_attn_tp_group=is_dp_attention_enabled(),
            prefix=add_prefix("embed_tokens", prefix),
        )

        self.enorm = RMSNorm(config.hidden_size, eps=config.rms_norm_eps)
        self.hnorm = RMSNorm(config.hidden_size, eps=config.rms_norm_eps)

        if quant_config is not None and quant_config.get_name() == "quark":
            self.eh_proj = ReplicatedLinear(
                2 * config.hidden_size,
                config.hidden_size,
                bias=False,
                quant_config=quant_config,
                prefix=add_prefix("eh_proj", prefix),
            )
        else:
            self.eh_proj = nn.Linear(
                2 * config.hidden_size, config.hidden_size, bias=False
            )

        self.rot_weight = None
        if _is_npu:
            rot_weight_path = get_global_server_args().model_path + "/rot.safetensors"
            if os.path.isfile(rot_weight_path):
                self.rot_weight = load_file(rot_weight_path)
                self.rot_weight = self.rot_weight["rot.weight"].npu()

        self.alt_stream = (
            torch.cuda.Stream()
            if _is_cuda or envs.SGLANG_NPU_USE_MULTI_STREAM.get() or is_sbo_enabled() 
            else None
        )

        layer_name = "decoder"
        if _is_npu and (
            get_global_server_args().speculative_draft_model_path
            == get_global_server_args().model_path
        ):
            layer_name = "layers." + str(config.num_hidden_layers)

        self.quant_config = quant_config
        self.decoder = DeepseekV2DecoderLayer(
            config,
            0,
            quant_config=quant_config,
            moe_quant_config_override=moe_quant_config_override,
            is_nextn=True,
            prefix=add_prefix(layer_name, prefix),
            alt_stream=self.alt_stream,
        )

        self.shared_head = nn.Module()
        self.shared_head.norm = RMSNorm(config.hidden_size, eps=config.rms_norm_eps)
        self.nsa_enable_prefill_cp = is_nsa_enable_prefill_cp()
        if self.nsa_enable_prefill_cp:
            self.cp_size = get_attention_cp_size()
        else:
            self.cp_size = None

def forward(
        self,
        input_ids: torch.Tensor,
        positions: torch.Tensor,
        forward_batch: ForwardBatch,
        input_embeds: torch.Tensor = None,
    ) -> torch.Tensor:
        if _is_npu and self.quant_config is None:
            os.environ["SGLANG_DEEPEP_BF16_DISPATCH"] = "1"
            os.environ["DEEP_NORMAL_MODE_USE_INT8_QUANT"] = "0"
        zero_allocator = BumpAllocator(
            buffer_size=2,
            dtype=torch.float32,
            device=(
                input_embeds.device if input_embeds is not None else input_ids.device
            ),
        )

        if input_embeds is None:
            hidden_states = self.embed_tokens(input_ids)
        else:
            hidden_states = input_embeds

        if hidden_states.shape[0] > 0:
            eh_input = torch.cat(
                (
                    self.enorm(hidden_states),
                    self.hnorm(
                        forward_batch.spec_info.hidden_states
                        if self.rot_weight is None
                        else torch.matmul(
                            forward_batch.spec_info.hidden_states, self.rot_weight
                        )
                    ),
                ),
                dim=-1,
            )
            if isinstance(self.eh_proj, ReplicatedLinear):
                hidden_states, _ = self.eh_proj(eh_input)
            else:
                hidden_states = self.eh_proj(eh_input)

        if nsa_use_prefill_cp(forward_batch, self.nsa_enable_prefill_cp):
            hidden_states = cp_split_and_rebuild_data(forward_batch, hidden_states)
            positions = cp_split_and_rebuild_position(forward_batch, positions)
        residual = None
        with get_global_expert_distribution_recorder().disable_this_region():
            hidden_states, residual, topk_indices = self.decoder(
                positions,
                hidden_states,
                forward_batch,
                residual,
                zero_allocator,
                prev_topk_indices=(
                    forward_batch.topk_indices
                    if forward_batch.reuse_mtp_topk_indices
                    else None
                ),
            )
            if forward_batch.reuse_mtp_topk_indices:
                forward_batch.topk_indices = topk_indices

        if not forward_batch.forward_mode.is_idle():
            if residual is not None:
                hidden_states, _ = self.shared_head.norm(hidden_states, residual)
            else:
                hidden_states = self.shared_head.norm(hidden_states)

            if nsa_use_prefill_cp(forward_batch, self.nsa_enable_prefill_cp):
                # allgather + rerrange
                hidden_states = cp_all_gather_rerange_output(
                    hidden_states,
                    self.cp_size,
                    forward_batch,
                    torch.cuda.current_stream(),
                )

        if _is_npu and self.quant_config is None:
            os.environ["SGLANG_DEEPEP_BF16_DISPATCH"] = "0"
            os.environ["DEEP_NORMAL_MODE_USE_INT8_QUANT"] = "1"
        return hidden_states

