from __future__ import annotations

import logging
from functools import lru_cache
from typing import NamedTuple

import torch

from hcu_sglang.layers.glm53_decode import absorb_bmm_scope, phase_enabled
from sglang.srt.environ import envs
from sglang.srt.models.deepseek_common.attention_forward_methods import (
    forward_mla_rocm as N,
)

logger = logging.getLogger(__name__)


class DecodeQWithStoredKV(NamedTuple):
    query: torch.Tensor


@lru_cache(maxsize=1)
def mla_provider():
    if not envs.SGLANG_LIGHTOP_MLA_KV_FUSED.get():
        return None, (192,)
    try:
        from lightop import rmsnorm

        getattr(rmsnorm.op, "fused_rms_norm_rope_contiguous_q_cat")
        return rmsnorm.fused_rms_norm_rope_contiguous_q_cat, getattr(
            rmsnorm.op, "MLA_ROPE_SUPPORTED_Q_HEAD_STRIDES", (192,)
        )
    except (ImportError, AttributeError, OSError) as exc:
        logger.warning("LightOp MLA fusion unavailable; using current path: %s", exc)
        return None, (192,)


def report(attn, state, **details):
    # Emit once per state on the first layer, including during graph capture.
    # A selected path still needs a GPU trace to confirm runtime execution.
    if attn.layer_id != 0:
        return
    reported = getattr(attn, "_lightop_mla_reported_paths", None)
    if reported is None:
        reported = attn._lightop_mla_reported_paths = set()
    if state not in reported:
        reported.add(state)
        logger.info("LightOp MLA fusion %s: %s", state, details)


def prepare_lightop_mla(attn, batch, latent, positions):
    """Return a plan before skipping KV RMSNorm; no tensors are mutated here."""
    _lightop_mla_qcat, _lightop_mla_q_head_strides = mla_provider()
    if _lightop_mla_qcat is None or not phase_enabled(attn, batch):
        return None
    from hcu_sglang.layers.attention.current_dsa.forward_batch_utils import (
        effective_forward_mode,
    )
    from sglang.srt.layers.attention.dsa.utils import dsa_use_prefill_cp
    from sglang.srt.layers.rotary_embedding.base import RotaryEmbedding
    from sglang.srt.layers.rotary_embedding.rope_variant import (
        DeepseekScalingRotaryEmbedding,
    )
    from hcu_sglang.cp_runtime import mla_use_prefill_cp
    from sglang.srt.mem_cache.memory_pool import DSATokenToKVPool, MLATokenToKVPool
    from sglang.srt.model_executor.forward_context import (
        get_attn_backend,
        get_token_to_kv_pool,
    )
    from sglang.srt.runtime_context import get_parallel

    mode = effective_forward_mode(batch)
    backend = get_attn_backend()
    pool = get_token_to_kv_pool()
    rope = attn.rotary_emb
    norm = attn.kv_a_layernorm
    if (
        not (mode.is_decode() or mode.is_target_verify() or mode.is_draft_extend_v2())
        or not attn.use_dsa
        or get_parallel().dcp_enabled
        or dsa_use_prefill_cp(batch)
        or mla_use_prefill_cp(batch)
        or attn.attn_mqa.is_cross_attention
        or getattr(backend, "dsa_decode_impl", None) != "flashmla_kv"
        or getattr(backend, "use_mha", True)
        or getattr(backend, "hisparse_coordinator", None) is not None
        # DSA uses this subclass with the same MLA KV buffer layout.
        # Keep an exact whitelist: HiSparse/layer-split pools need their
        # own cache-write handling and must not bypass it.
        or type(pool) not in (MLATokenToKVPool, DSATokenToKVPool)
        or not pool.dsa_kv_cache_store_fp8
        or pool.layer_transfer_counter is not None
        or (attn.kv_lora_rank, attn.qk_nope_head_dim, attn.qk_rope_head_dim)
        not in ((512, 128, 64), (512, 192, 64))
        or attn.qk_nope_head_dim + attn.qk_rope_head_dim
        not in _lightop_mla_q_head_strides
        or type(rope) not in (RotaryEmbedding, DeepseekScalingRotaryEmbedding)
        or rope.rotary_dim != 64
        or norm.cast_x_before_out_mul
        or norm.variance_size_override is not None
        or attn._fuse_rope_for_trtllm_mla(batch)
        or attn._skip_rope_for_dsa_tilelang_fused()
        or attn._skip_rope_for_aiter_fused_mla()
    ):
        report(
            attn,
            "fallback_config",
            pool=type(pool).__name__,
            backend=type(backend).__name__,
            decode_impl=getattr(backend, "dsa_decode_impl", None),
            rope=type(rope).__name__,
            q_head_width=attn.qk_nope_head_dim + attn.qk_rope_head_dim,
            supported_q_head_strides=_lightop_mla_q_head_strides,
        )
        return None
    cache = pool.kv_buffer[attn.layer_id - pool.start_layer]
    cos_sin = rope.cos_sin_cache
    slots = batch.out_cache_loc
    if (
        latent.ndim != 2
        or latent.shape[0] == 0
        or latent.shape[1] != 576
        or latent.stride(1) != 1
        or latent.stride(0) < 576
        or latent.stride(0) % 8
        or cache.dtype != torch.uint8
        or not cache.is_contiguous()
        or cache.ndim != 3
        or cache.shape[1:] != (1, 656)
        or cos_sin.ndim != 2
        or cos_sin.shape[1] != 64
        or not cos_sin.is_contiguous()
        or norm.weight.shape != (512,)
        or not norm.weight.is_contiguous()
        or any(
            t.dtype != torch.bfloat16 or t.device != latent.device or t.data_ptr() % 16
            for t in (latent, norm.weight, cos_sin)
        )
        or not latent.is_cuda
        or cache.device != latent.device
        or any(
            t.dtype != torch.int64
            or t.device != latent.device
            or t.shape != (latent.shape[0],)
            or not t.is_contiguous()
            for t in (positions, slots)
        )
    ):
        report(
            attn,
            "fallback_layout",
            latent=(tuple(latent.shape), latent.stride(), str(latent.dtype)),
            cache=(tuple(cache.shape), str(cache.dtype)),
            cos_sin=(tuple(cos_sin.shape), str(cos_sin.dtype)),
            positions=(tuple(positions.shape), str(positions.dtype)),
            slots=(tuple(slots.shape), str(slots.dtype)),
        )
        return None
    return _lightop_mla_qcat, cache, cos_sin


def run_lightop_mla(attn, plan, batch, positions, q_pe, k_pe, k_raw, q_nope):
    """Validate the late Q layout before performing any in-place writes."""
    _, _lightop_mla_q_head_strides = mla_provider()
    tokens = k_raw.shape[0]
    heads = attn.num_local_heads
    q_head_stride = attn.qk_nope_head_dim + attn.qk_rope_head_dim
    if (
        q_pe.shape != (tokens, heads, 64)
        or q_head_stride not in _lightop_mla_q_head_strides
        or q_pe.stride() != (heads * q_head_stride, q_head_stride, 1)
        or q_nope.shape != (tokens, heads, 512)
        or q_nope.stride(2) != 1
        or k_pe.shape != (tokens, 1, 64)
        or k_pe.stride(0) != k_raw.stride(0)
        or k_pe.stride(2) != 1
        or any(
            t.dtype != torch.bfloat16 or t.device != k_raw.device or t.data_ptr() % 16
            for t in (q_pe, q_nope, k_pe)
        )
    ):
        report(
            attn,
            "fallback_q_layout",
            q_pe=(tuple(q_pe.shape), q_pe.stride()),
            q_nope=(tuple(q_nope.shape), q_nope.stride()),
            k_pe=(tuple(k_pe.shape), k_pe.stride()),
        )
        return None
    report(attn, "selected", tokens=tokens, heads=heads)
    fn, cache, cos_sin = plan
    k_normed = torch.empty((tokens, 512), device=k_raw.device, dtype=k_raw.dtype)
    q_cat = torch.empty((tokens, heads, 576), device=q_pe.device, dtype=q_pe.dtype)
    # Runtime failures are not swallowed: the op mutates RoPE and KV buffers,
    # so retrying the unfused path after a failed launch could rotate twice.
    fn(
        positions,
        q_pe,
        k_pe.squeeze(1),
        k_raw,
        k_normed,
        attn.kv_a_layernorm.weight,
        cos_sin,
        batch.out_cache_loc,
        cache,
        q_nope,
        q_cat,
        "fp8_e4m3",
        1.0,
        is_neox_style=attn.rotary_emb.is_neox_style,
        epsilon=attn.kv_a_layernorm.variance_epsilon,
        flashmla_fp8_layout=True,
        kv_quant_epsilon=1e-6,
    )
    return q_cat, k_normed.unsqueeze(1)


def forward_absorb_rocm_prepare(
    self: N.DeepseekV2AttentionMLA,
    positions: N.torch.Tensor,
    hidden_states: N.torch.Tensor,
    forward_batch: N.ForwardBatch,
    zero_allocator: N.BumpAllocator,
    llama_4_scaling: N.Optional[N.torch.Tensor] = None,
    prev_topk_indices: N.Optional[N.torch.Tensor] = None,
):
    from sglang.srt.model_executor.runner import get_is_capture_mode

    q_replicate_active = (
        self.is_dcp_q_replicate_phase(forward_batch)
        and not self.use_deep_gemm_bmm
        and self.w_kc_qrep is not None
        and self.q_b_proj_qrep_weight is not None
    )
    lightop_plan = None
    lightop_q_cat = None
    q_lora = None
    topk_indices = None
    if self.q_lora_rank is not None:
        q, latent_cache = (
            N.get_attn_tp_context()
            .fetch_qkv_latent()
            .split(
                [self.q_lora_rank, self.kv_lora_rank + self.qk_rope_head_dim],
                dim=-1,
            )
        )
        k_nope = latent_cache[..., : self.kv_lora_rank]

        if (
            not N._use_aiter
            and not q_replicate_active
            and not N.is_kv_b_lora_active(self)
            and not N._SGLANG_EXPERIMENTAL_LORA_OPTI
        ):
            lightop_plan = prepare_lightop_mla(
                self, forward_batch, latent_cache, positions
            )

        if lightop_plan is not None:
            q = self.q_a_layernorm(q)
        # overlap qk norm
        elif self.alt_stream is not None and get_is_capture_mode():
            current_stream = N.torch.cuda.current_stream()
            self.alt_stream.wait_stream(current_stream)
            q = self.q_a_layernorm(q)
            with N.torch.cuda.stream(self.alt_stream):
                k_nope = self.kv_a_layernorm(k_nope)
            current_stream.wait_stream(self.alt_stream)
        elif N._use_aiter_gfx95 and self.q_b_proj.weight.dtype == N.torch.uint8:
            q, _, k_nope, *_ = N.fused_rms_mxfp4_quant(
                q,
                self.q_a_layernorm.weight,
                self.q_a_layernorm.variance_epsilon,
                k_nope,
                self.kv_a_layernorm.weight,
                self.kv_a_layernorm.variance_epsilon,
            )
        elif N._use_aiter_gfx95 and N._is_block_scale_fp8(self.q_b_proj):
            emit_transposed_scale = N.emit_transposed_bpreshuffle_scale(
                q.shape[0],
                on_bpreshuffle_gfx95=N._use_aiter_bpreshuffle_gfx95,
            )
            if self.use_dsa:
                q_quanted, q_lora, k_nope, _ = N.fused_rms_fp8_group_quant(
                    q,
                    self.q_a_layernorm.weight,
                    self.q_a_layernorm.variance_epsilon,
                    k_nope,
                    self.kv_a_layernorm.weight,
                    self.kv_a_layernorm.variance_epsilon,
                    group_size=128,
                    dtype_quant=N.torch.float8_e4m3fn,
                    res1=None,
                    output_unquantized_inp1=True,
                    transpose_scale=emit_transposed_scale,
                )
                if emit_transposed_scale:
                    q_quanted = N.view_aiter_fused_rms_transposed_fp8_scale_tuple(
                        q_quanted
                    )
                elif N._use_aiter_bpreshuffle_gfx95:
                    q_quanted = N.materialize_bpreshuffle_fp8_scale_tuple(q_quanted)
                q = q_quanted
            else:
                q, _, k_nope, _ = N.fused_rms_fp8_group_quant(
                    q,
                    self.q_a_layernorm.weight,
                    self.q_a_layernorm.variance_epsilon,
                    k_nope,
                    self.kv_a_layernorm.weight,
                    self.kv_a_layernorm.variance_epsilon,
                    group_size=128,
                    dtype_quant=N.torch.float8_e4m3fn,
                    res1=None,
                    output_unquantized_inp1=False,
                    transpose_scale=emit_transposed_scale,
                )
                if emit_transposed_scale:
                    q = N.view_aiter_fused_rms_transposed_fp8_scale_tuple(q)
                elif N._use_aiter_bpreshuffle_gfx95:
                    q = N.materialize_bpreshuffle_fp8_scale_tuple(q)
        elif N._use_aiter:
            q, k_nope = N.fused_qk_rmsnorm_bf16(
                q,
                self.q_a_layernorm.weight,
                self.q_a_layernorm.variance_epsilon,
                k_nope,
                self.kv_a_layernorm.weight,
                self.kv_a_layernorm.variance_epsilon,
            )
        else:
            q = self.q_a_layernorm(q)
            k_nope = self.kv_a_layernorm(k_nope)

        # q_lora needed by indexer
        if self.use_dsa:
            if q_lora is None:
                q_lora = q

        # overlap q_b_proj and indexer during decode
        if (
            self.alt_stream is not None
            and get_is_capture_mode()
            and forward_batch.forward_mode.is_decode_or_idle()
            and q_lora is not None
            and not q_replicate_active
        ):
            current_stream = N.torch.cuda.current_stream()
            self.alt_stream.wait_stream(current_stream)
            with N.torch.cuda.stream(self.alt_stream):
                k_nope = k_nope.unsqueeze(1)
                q = self.q_b_proj_forward(q)
            if self.should_run_indexer(prev_topk_indices):
                topk_indices = self.indexer(
                    x=hidden_states,
                    q_lora=q_lora,
                    positions=positions,
                    forward_batch=forward_batch,
                    layer_id=self.layer_id,
                )
            else:
                # skip_topk reuses prev layer's indices; mirror into this
                # layer's slot so the captured buffer matches what's used.
                topk_indices = N.maybe_capture_indexer_topk(
                    self.layer_id, prev_topk_indices
                )
            current_stream.wait_stream(self.alt_stream)
        else:
            k_nope = k_nope.unsqueeze(1)
            if q_replicate_active:
                q = N.torch.nn.functional.linear(q, self.q_b_proj_qrep_weight).view(
                    -1,
                    self.num_local_heads * self.dcp_size,
                    self.qk_head_dim,
                )
            else:
                q = self.q_b_proj_forward(q)

            if q_lora is not None:
                if self.should_run_indexer(prev_topk_indices):
                    topk_indices = self.indexer(
                        x=hidden_states,
                        q_lora=q_lora,
                        positions=positions,
                        forward_batch=forward_batch,
                        layer_id=self.layer_id,
                    )
                else:
                    topk_indices = N.maybe_capture_indexer_topk(
                        self.layer_id, prev_topk_indices
                    )
    else:
        if q_replicate_active:
            q = N.torch.nn.functional.linear(
                hidden_states, self.q_b_proj_qrep_weight
            ).view(
                -1,
                self.num_local_heads * self.dcp_size,
                self.qk_head_dim,
            )
        else:
            q = self.q_proj(hidden_states)[0].view(
                -1, self.num_local_heads, self.qk_head_dim
            )
        latent_cache = self.kv_a_proj_with_mqa(hidden_states)[0]
        k_nope = latent_cache[..., : self.kv_lora_rank]
        k_nope = self.kv_a_layernorm(k_nope).unsqueeze(1)

    q_nope, q_pe, k_pe = self._split_q_nope_pe(q, latent_cache)

    # The fused kernel wins at decode-sized batches (decode, target verify,
    # draft extend); prefill shapes run faster as the two separate launches.
    fuse_bmm_rope_cache = (
        not q_replicate_active
        and not forward_batch.forward_mode.is_extend_or_draft_extend_or_mixed()
        and N._can_fuse_bmm_rope_cat_and_cache(self)
    )

    if q_replicate_active:
        q_nope_out = (
            N.torch.bmm(q_nope.transpose(0, 1), self.w_kc_qrep)
            .transpose(0, 1)
            .contiguous()
        )
    else:
        _kvb_q = None
        if N._SGLANG_EXPERIMENTAL_LORA_OPTI:
            # Fork the kv_b q-correction A-step onto the LoRA side stream to overlap the bmm.
            from sglang.srt.lora.trtllm_lora_temp.deepseek_mla_correction import (
                kv_b_lora_q_prepare,
            )

            _kvb_q = kv_b_lora_q_prepare(self, q_nope)

        if self.use_deep_gemm_bmm:
            (
                q_nope_val,
                q_nope_scale,
                masked_m,
                expected_m,
                aligned_m,
            ) = N.per_token_group_quant_mla_deep_gemm_masked_fp8(q_nope.transpose(0, 1))
            q_nope_out = q_nope.new_empty(
                (self.num_local_heads, aligned_m, self.kv_lora_rank)
            )
            N.deep_gemm_wrapper.grouped_gemm_nt_f8f8bf16_masked(
                (q_nope_val, q_nope_scale),
                (self.w_kc, self.w_scale_k),
                q_nope_out,
                masked_m,
                expected_m,
            )
            q_nope_out = q_nope_out[:, :expected_m, :]
        elif fuse_bmm_rope_cache:
            # The absorb is the first half of the kernel core will run, and
            # that kernel wants q_nope un-absorbed. Nothing to do here.
            q_nope_out = None
        else:
            q_nope_out = N.rocm_absorb_q_bmm(
                self, q_nope, is_capture_mode=get_is_capture_mode()
            )

        if q_nope_out is not None:
            q_nope_out = q_nope_out.transpose(0, 1)
            if N._SGLANG_EXPERIMENTAL_LORA_OPTI:
                from sglang.srt.lora.trtllm_lora_temp.deepseek_mla_correction import (
                    kv_b_lora_q_apply,
                )

                q_nope_out = kv_b_lora_q_apply(self, q_nope, q_nope_out, _kvb_q)
            elif N.is_kv_b_lora_active(self):
                q_nope_out = N.apply_kv_b_lora_q_correction(self, q_nope, q_nope_out)

    if lightop_plan is not None:
        fused = run_lightop_mla(
            self,
            lightop_plan,
            forward_batch,
            positions,
            q_pe,
            k_pe,
            latent_cache[..., : self.kv_lora_rank],
            q_nope_out,
        )
        if fused is None:
            k_nope = self.kv_a_layernorm(
                latent_cache[..., : self.kv_lora_rank]
            ).unsqueeze(1)
        else:
            lightop_q_cat, k_nope = fused

    fuse_rope_for_trtllm_mla = self._fuse_rope_for_trtllm_mla(forward_batch)

    force_rope_for_aiter_dcp_decode = (
        self.dcp_size > 1
        and (
            forward_batch.forward_mode.is_decode()
            or forward_batch.forward_mode.is_target_verify()
            or forward_batch.forward_mode.is_draft_extend_v2()
        )
        and N._use_aiter_gfx95
        and self.current_attention_backend
        not in N.FORWARD_ABSORB_CORE_ATTENTION_BACKENDS
    )
    if (
        lightop_q_cat is None
        and self.rotary_emb is not None
        and (
            force_rope_for_aiter_dcp_decode
            or (
                (not fuse_rope_for_trtllm_mla)
                and (not self._skip_rope_for_dsa_tilelang_fused())
                and (not self._skip_rope_for_aiter_fused_mla())
                and (
                    not N._use_aiter
                    or not N._is_gfx95_supported
                    or self.use_dsa
                    or (
                        self.current_attention_backend
                        not in N.FORWARD_ABSORB_CORE_ATTENTION_BACKENDS
                        and self.current_attention_backend != "aiter"
                    )
                )
            )
        )
    ):
        q_pe, k_pe = self.rotary_emb(positions, q_pe, k_pe)

    # all_gather q_pe, q_nope_out,take tp8 as an example， q_pe [B, H, ROPE_DIM], q_nope_out [B, H, NOPE_DIM] gathered to [B, H * dcp_world_size, ROPE_DIM] [B, H * dcp_world_size, NOPE_DIM] for decode batch, and all gather k_pe, k_nope for extend batch.
    if self.dcp_size > 1:
        if self.is_dcp_mla_decode_phase(forward_batch):
            if not q_replicate_active:
                q_nope_out, q_pe = N.all_gather_q_for_mla_decode(
                    q_nope_out=q_nope_out,
                    q_pe=q_pe,
                )
        elif forward_batch.forward_mode.is_extend():
            # Assemble the full sequence into dcp_kv_buffer, which the
            # backend attends over instead of the sharded local cache.
            if (
                forward_batch.attn_dcp_metadata is not None
                and forward_batch.attn_dcp_metadata.dcp_kv_buffer is not None
            ):
                N.all_gather_kv_cache_for_mla_extend(
                    N.get_token_to_kv_pool(),
                    self.attn_mqa,
                    forward_batch.extend_prefix_lens_cpu,
                    forward_batch.attn_dcp_metadata.dcp_local_prefix_kv_indices,
                    forward_batch.attn_dcp_metadata.dcp_extend_prefix_lens_sum,
                    forward_batch.attn_dcp_metadata.dcp_kv_buffer,
                    self.kv_lora_rank,
                    k_nope,
                    k_pe,
                )
        else:
            N.logger.warning(f"not supported forward_mode {forward_batch.forward_mode}")

    return (
        q_pe,
        k_pe,
        DecodeQWithStoredKV(lightop_q_cat) if lightop_q_cat is not None else q_nope_out,
        k_nope,
        forward_batch,
        zero_allocator,
        positions,
        topk_indices,
        llama_4_scaling,
        q_nope if fuse_bmm_rope_cache else None,
    )


def around_rocm_prepare(original, attn, *args, **kwargs):
    batch = kwargs.get("forward_batch", args[2] if len(args) > 2 else None)
    with absorb_bmm_scope(batch):
        if not phase_enabled(attn, batch):
            return original(attn, *args, **kwargs)
        return forward_absorb_rocm_prepare(attn, *args, **kwargs)
