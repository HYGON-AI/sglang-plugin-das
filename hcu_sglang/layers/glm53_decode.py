# Copyright (c) 2026 Hygon Information Technology Co., Ltd.
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

import logging
from contextlib import contextmanager
from contextvars import ContextVar
from copy import copy
from functools import lru_cache

import torch

from hcu_sglang.utils.detection import is_hcu
from sglang.srt.environ import envs

logger = logging.getLogger(__name__)
_o_proj_phase = ContextVar("hcu_glm53_o_proj_phase", default=None)
_fused_query = ContextVar("hcu_glm53_fused_query", default=None)
_absorb_bmm_batch = ContextVar("hcu_glm53_absorb_bmm_batch", default=None)
_GLM_ARCHS = {"GlmMoeDsaForCausalLM", "GlmMoeDsaForCausalLMNextN"}


def model_matches(config):
    return bool(_GLM_ARCHS & set(getattr(config, "architectures", None) or ()))


@lru_cache(maxsize=1)
def hcu_gemm():
    if not is_hcu() or not envs.SGLANG_GLM_HCU_GEMM.get():
        return None
    import hcu_ops

    return hcu_ops


@lru_cache(maxsize=1)
def fusion_phases():
    phases = frozenset(
        p.strip() for p in envs.SGLANG_LIGHTOP_FUSION_PHASES.get().split(",")
    )
    if not phases <= {"verify", "draft_decode", "draft_extend"}:
        raise ValueError("Unknown SGLANG_LIGHTOP_FUSION_PHASES value")
    return phases


def phase_enabled(attn, batch):
    if not (
        getattr(attn, "_lightop_glm_fusions", False)
        and (
            envs.SGLANG_LIGHTOP_NORM_QUANT.get()
            or envs.SGLANG_LIGHTOP_MLA_KV_FUSED.get()
        )
    ):
        return False
    if batch is None or torch.compiler.is_compiling():
        return False
    from hcu_sglang.layers.attention.current_dsa.forward_batch_utils import (
        effective_forward_mode,
    )
    from sglang.srt.batch_invariant_ops import is_batch_invariant_mode_enabled
    from sglang.srt.runtime_context import get_exec

    if (
        is_batch_invariant_mode_enabled()
        or get_exec().deterministic.enable_deterministic_inference
    ):
        return False
    mode = effective_forward_mode(batch)
    if attn.is_nextn:
        if mode.is_decode():
            return "draft_decode" in fusion_phases()
        if mode.is_extend() or mode.is_draft_extend_v2():
            return "draft_extend" in fusion_phases()
        return False
    return mode.is_target_verify() and "verify" in fusion_phases()


def supports_input_quant(layer):
    if layer is None:
        return False
    from sglang.srt.layers.quantization.compressed_tensors.schemes import (
        compressed_tensors_w8a8_fp8 as Q,
    )

    scheme = getattr(layer, "scheme", None)
    return (
        not Q._use_aiter
        and isinstance(scheme, Q.CompressedTensorsW8A8Fp8)
        and scheme.strategy == Q.QuantizationStrategy.CHANNEL
        and scheme.weight_block_size is None
        and not scheme.is_static_input_scheme
        and getattr(layer, "input_scale", None) is None
        and layer.weight.dtype == torch.float8_e4m3fn
        and layer.weight_scale.dtype == torch.float32
    )


@contextmanager
def o_proj_scope(attn, batch):
    phase = None
    if envs.SGLANG_GLM_OPT_DECODE.get() and getattr(
        attn, "_lightop_glm_fusions", False
    ):
        from hcu_sglang.layers.attention.current_dsa.forward_batch_utils import (
            effective_forward_mode,
        )

        if supports_input_quant(attn.o_proj):
            phase = effective_forward_mode(batch)
    token = _o_proj_phase.set(phase)
    try:
        yield
    finally:
        _o_proj_phase.reset(token)


def try_o_proj_quant(x, output, scale):
    if not envs.SGLANG_GLM_OPT_DECODE.get() or torch.compiler.is_compiling():
        return False
    phase = _o_proj_phase.get()
    if phase is None:
        return False
    if phase.is_decode():
        rows = (1, 2, 3, 4, 5)
    elif phase.is_target_verify() or phase.is_draft_extend_v2():
        rows = (6, 12, 18, 24, 30)
    else:
        return False
    if (
        x.ndim != 2
        or x.shape[0] not in rows
        or x.shape[1] != 16384
        or x.dtype != torch.bfloat16
        or output.dtype != torch.float8_e4m3fn
        or scale.dtype != torch.float32
        or output.shape != x.shape
        or scale.shape != (x.shape[0], 1)
        or x.stride() != (16384, 1)
        or output.stride() != (16384, 1)
        or scale.stride() != (1, 1)
        or not all(
            t.is_cuda and t.device == x.device and t.is_contiguous()
            for t in (x, output, scale)
        )
        or x.data_ptr() % 64
        or output.data_ptr() % 64
        or scale.data_ptr() % 4
        or torch.version.hip is None
        or torch.cuda.get_device_properties(x.device).gcnArchName.split(":", 1)[0]
        != "gfx938"
    ):
        return False
    ranges = []
    for tensor in (x, output, scale):
        storage = tensor.untyped_storage()
        begin, size = storage.data_ptr(), storage.nbytes()
        offset, width = tensor.storage_offset(), tensor.element_size()
        if (
            begin == 0
            or size <= 0
            or offset < 0
            or offset * width + tensor.numel() * width > size
            or tensor.data_ptr() != begin + offset * width
        ):
            return False
        ranges.append((begin, begin + size))
    if any(
        a < d and c < b for i, (a, b) in enumerate(ranges) for c, d in ranges[i + 1 :]
    ):
        return False
    from lightop import op

    op.per_token_quant_fp8(output, x, scale)
    return True


def fp8_gemm(qinput, x_scale, weight, weight_scale, output):
    from sglang.srt.model_executor.forward_context import (
        get_attn_backend,
        has_forward_context,
    )

    backend = get_attn_backend() if has_forward_context() else None
    gemm = hcu_gemm()
    if gemm is not None and getattr(backend, "_glm53_model", False):
        if gemm.try_skinny_fp8(qinput, x_scale, weight, weight_scale, output):
            return
    import deepgemm

    deepgemm.fp8_gemm((qinput, x_scale), (weight, weight_scale), output)


@contextmanager
def absorb_bmm_scope(batch):
    if torch.compiler.is_compiling():
        yield
        return
    token = _absorb_bmm_batch.set(batch)
    try:
        yield
    finally:
        _absorb_bmm_batch.reset(token)


def _absorb_bmm_fusion_enabled(attn):
    if (
        not getattr(attn, "_lightop_glm_fusions", False)
        or not envs.SGLANG_GLM_OPT_DECODE.get()
        or torch.compiler.is_compiling()
    ):
        return False
    batch = _absorb_bmm_batch.get()
    if batch is None:
        return False

    from hcu_sglang.layers.attention.current_dsa.forward_batch_utils import (
        effective_forward_mode,
    )
    from sglang.srt.batch_invariant_ops import is_batch_invariant_mode_enabled
    from sglang.srt.runtime_context import get_exec

    if (
        is_batch_invariant_mode_enabled()
        or get_exec().deterministic.enable_deterministic_inference
    ):
        return False

    mode = effective_forward_mode(batch)
    return (
        mode.is_decode()
        or mode.is_target_verify()
        or mode.is_draft_extend_v2()
        or (attn.is_nextn and mode.is_extend())
    )


def absorb_bmm(attn, x, weight, out=None):
    if _absorb_bmm_fusion_enabled(attn):
        from hcu_sglang.kernels.ops.glm_absorb_bmm import try_glm_absorb_bmm

        result = try_glm_absorb_bmm(x, weight, out)
        if result is not None:
            return result
    gemm = hcu_gemm()
    if gemm is not None and getattr(attn, "_hcu_glm_gemm", False):
        result = gemm.try_bmm_bf16(x, weight, out)
        if result is not None:
            return result
    return torch.bmm(x, weight) if out is None else torch.bmm(x, weight, out=out)


def _channel_fp8_absorb():
    from hcu_sglang.layers.attention.hisparse import _matches_channel_fp8
    from hcu_sglang.utils.detection import is_hcu_native_fp8_supported
    from sglang.srt.arg_groups.model_override_base import model_config_of, resolved_view
    from sglang.srt.runtime_context import get_context

    config = model_config_of(resolved_view(get_context().server_args)).hf_config
    return is_hcu_native_fp8_supported() and model_matches(config) and _matches_channel_fp8(config)


def around_q_bmm(original, attn, q_nope, *, is_capture_mode):
    if not getattr(attn, "_hcu_glm_gemm", False) or not (
        attn.w_kc.dtype == torch.bfloat16
        or (attn.w_kc.dtype == torch.float32 and _channel_fp8_absorb())
    ):
        return original(attn, q_nope, is_capture_mode=is_capture_mode)
    from sglang.srt.models.deepseek_common.attention_forward_methods.forward_mla_rocm import (
        _absorb_weight_bf16,
    )

    return absorb_bmm(
        attn,
        q_nope.to(torch.bfloat16).transpose(0, 1),
        _absorb_weight_bf16(attn.w_kc, attn.w_scale),
    )


def around_v_bmm(original, attn, attn_output):
    if not getattr(attn, "_hcu_glm_gemm", False) or not (
        attn.w_vc.dtype == torch.bfloat16
        or (attn.w_vc.dtype == torch.float32 and _channel_fp8_absorb())
    ):
        return original(attn, attn_output)
    from sglang.srt.models.deepseek_common.attention_forward_methods.forward_mla_rocm import (
        _absorb_weight_bf16,
    )
    from sglang.srt.models.deepseek_common.utils import _is_block_scale_fp8

    weight = _absorb_weight_bf16(attn.w_vc, attn.w_scale)
    if (
        attn_output.dtype == torch.bfloat16
        and attn.w_vc.dtype == torch.bfloat16
        and attn.o_proj.weight.dtype != torch.uint8
        and not _is_block_scale_fp8(attn.o_proj)
    ):
        output = torch.empty(
            attn_output.shape[0],
            attn_output.shape[1],
            attn.w_vc.shape[-1],
            dtype=torch.bfloat16,
            device=attn_output.device,
        )
        absorb_bmm(attn, attn_output.transpose(0, 1), weight, output.transpose(0, 1))
        return output.flatten(1, 2)
    if (
        attn.w_vc.dtype == torch.float32
        and attn.o_proj.weight.dtype != torch.uint8
        and not _is_block_scale_fp8(attn.o_proj)
    ):
        return (
            absorb_bmm(attn, attn_output.to(torch.bfloat16).transpose(0, 1), weight)
            .transpose(0, 1)
            .flatten(1, 2)
        )
    return original(attn, attn_output)


@lru_cache(maxsize=1)
def norm_quant_provider():
    try:
        from lightop import rmsnorm

        getattr(rmsnorm.op, "rms_norm_fp8_quant")
        return rmsnorm.rms_norm_per_token_fp8_quant, getattr(
            rmsnorm.op, "RMS_NORM_FP8_FP32_RESIDUAL_WIDTHS", ()
        )
    except (ImportError, AttributeError, OSError) as exc:
        logger.warning("LightOp norm fusion unavailable; using current path: %s", exc)
        return None, ()


def norm_quant(norm, x, residual=None, post_residual_addition=None):
    from sglang.srt.batch_invariant_ops import is_batch_invariant_mode_enabled
    from sglang.srt.runtime_context import get_exec

    function, widths = norm_quant_provider()
    compatible = (
        function is not None
        and getattr(norm._forward_method, "__name__", "")
        in ("forward_hip", "forward_cuda")
        and not torch.compiler.is_compiling()
        and not is_batch_invariant_mode_enabled()
        and not get_exec().deterministic.enable_deterministic_inference
        and post_residual_addition is None
        and not getattr(x, "_sglang_needs_allreduce_fusion", False)
        and not norm.cast_x_before_out_mul
        and norm.variance_size_override is None
        and not norm.fp32_residual
        and norm.override_orig_dtype is None
        and not getattr(norm, "x_pad_to_multiple", 0)
        and x.is_cuda
        and x.dtype == torch.bfloat16
        and x.ndim == 2
        and x.shape[0] > 0
        and x.shape[1] in widths
        and x.is_contiguous()
        and x.data_ptr() % 16 == 0
        and norm.weight.dtype == x.dtype
        and norm.weight.device == x.device
        and norm.weight.is_contiguous()
        and norm.weight.shape == (x.shape[1],)
        and norm.weight.data_ptr() % 16 == 0
    )
    if residual is not None:
        compatible = compatible and (
            residual.dtype == x.dtype
            and residual.device == x.device
            and residual.shape == x.shape
            and residual.is_contiguous()
            and residual.data_ptr() % 16 == 0
            and residual.data_ptr() != x.data_ptr()
        )
    if not compatible:
        return (
            (norm(x), x)
            if residual is None
            else norm(x, residual, post_residual_addition)
        )
    out = x.clone() if residual is None else x
    quant, scale = function(
        out,
        norm.weight,
        epsilon=norm.variance_epsilon,
        fp8type=0,
        residual=residual,
        update_input=True,
    )
    return (quant, scale, out), (x if residual is None else residual)


class LightopNorm:
    def __init__(self, norm):
        self.norm = norm

    def __getattr__(self, name):
        return getattr(self.norm, name)

    def __call__(self, x, residual=None, post_residual_addition=None, **kwargs):
        if kwargs:
            return self.norm(x, residual, post_residual_addition, **kwargs)
        output, residual_out = norm_quant(
            self.norm, x, residual, post_residual_addition
        )
        return output if residual is None else (output, residual_out)


def prepare_norm_stage(stage, batch):
    attn = getattr(stage, "_hcu_decode_attn", None)
    linear = getattr(stage, "_hcu_decode_consumer", None)
    if (
        attn is None
        or not envs.SGLANG_LIGHTOP_NORM_QUANT.get()
        or not phase_enabled(attn, batch)
    ):
        return stage
    from sglang.srt.layers.attention.dsa.utils import dsa_use_prefill_cp
    from sglang.srt.layers.layer_boundary.adapters.attention import get_attn_tp_context
    from hcu_sglang.cp_runtime import mla_use_prefill_cp
    from sglang.srt.models.deepseek_common.attention_forward_methods import (
        AttnForwardMethod,
    )

    if (
        not attn.use_dsa
        or get_attn_tp_context().input_scattered
        or attn.dispatch_attn_forward_method(batch) != AttnForwardMethod.MLA_ROCM
        or dsa_use_prefill_cp(batch)
        or mla_use_prefill_cp(batch)
        or not supports_input_quant(linear)
    ):
        return stage
    invocation = copy(stage)
    invocation.plan = copy(stage.plan)
    invocation.plan.norm = LightopNorm(stage.norm)
    return invocation


def mark_decoder(layer):
    attn = layer.self_attn
    if not getattr(attn, "_lightop_glm_fusions", False):
        return
    if hasattr(layer, "attn_boundary"):
        layer.attn_boundary._hcu_decode_attn = attn
        layer.attn_boundary._hcu_decode_consumer = getattr(
            attn, "fused_qkv_a_proj_with_mqa", None
        )
    if hasattr(layer, "ffn_boundary"):
        layer.ffn_boundary._hcu_decode_attn = attn
        layer.ffn_boundary._hcu_decode_consumer = getattr(
            layer.mlp, "gate_up_proj", None
        )
    from sglang.srt.models.deepseek_v2 import DeepseekV2MLP

    if isinstance(layer.mlp, DeepseekV2MLP):
        layer.mlp._lightop_norm_input = True


def around_forward_prepare(original, attn, positions, hidden_states, *args, **kwargs):
    if (
        getattr(attn, "_lightop_glm_fusions", False)
        and envs.SGLANG_LIGHTOP_NORM_QUANT.get()
        and isinstance(hidden_states, tuple)
        and len(hidden_states) == 3
        and isinstance(hidden_states[2], torch.Tensor)
    ):
        hidden_states = hidden_states[2]
    return original(attn, positions, hidden_states, *args, **kwargs)


def around_attn_dispatch(original, attn, batch):
    method = original(attn, batch)
    if not (
        getattr(attn, "_lightop_glm_fusions", False)
        or getattr(attn, "_hcu_glm_gemm", False)
    ):
        return method
    from hcu_sglang.layers.attention.current_dsa.forward_batch_utils import (
        effective_forward_mode,
    )
    from sglang.srt.models.deepseek_common.attention_forward_methods import (
        AttnForwardMethod,
    )
    from sglang.srt.runtime_context import get_parallel

    mode = effective_forward_mode(batch)
    if (
        attn.use_dsa
        and attn.current_attention_backend in ("dsa", "nsa")
        and method == AttnForwardMethod.MLA
        and supports_input_quant(attn.o_proj)
        and (
            mode.is_decode_or_idle()
            or mode.is_target_verify()
            or mode.is_draft_extend_v2()
            or (
                mode.is_extend()
                and get_parallel().attn_dcp_size > 1
                and _channel_fp8_absorb()
            )
        )
    ):
        return AttnForwardMethod.MLA_ROCM
    return method


def around_rocm_core(
    original,
    attn,
    q_pe,
    k_pe,
    q_nope_out,
    k_nope,
    batch,
    zero_allocator,
    positions,
    topk_indices,
    llama_4_scaling,
    q_nope_unabsorbed=None,
):
    from hcu_sglang.layers.attention.current_dsa.glm53_decode_mla import (
        DecodeQWithStoredKV,
    )

    query = q_nope_out.query if isinstance(q_nope_out, DecodeQWithStoredKV) else None
    if query is not None:
        if llama_4_scaling is not None:
            query[..., : attn.kv_lora_rank] *= llama_4_scaling
            llama_4_scaling = None
        q_nope_out, q_pe, k_nope = query, None, None
    token = _fused_query.set(query)
    try:
        with absorb_bmm_scope(batch), o_proj_scope(attn, batch):
            return original(
                attn,
                q_pe,
                k_pe,
                q_nope_out,
                k_nope,
                batch,
                zero_allocator,
                positions,
                topk_indices,
                llama_4_scaling,
                q_nope_unabsorbed,
            )
    finally:
        _fused_query.reset(token)


def around_radix_forward(
    original, layer, q, k, v, batch, save_kv_cache=True, *args, **kwargs
):
    query = _fused_query.get()
    if query is not None and q is query:
        save_kv_cache = False
        kwargs["lightop_fused_q"] = True
    return original(layer, q, k, v, batch, save_kv_cache, *args, **kwargs)


def around_lm_head(original, processor, hidden_states, lm_head, embedding_bias=None):
    from sglang.srt.layers.logits_processor import (
        should_apply_lm_head_quant_method,
        use_intel_amx_backend,
    )

    gemm = hcu_gemm()
    quant_method = getattr(lm_head, "quant_method", None)
    if (
        gemm is None
        or not model_matches(processor.config)
        or (hasattr(lm_head, "set_lora") and hasattr(lm_head, "apply_lora"))
        or should_apply_lm_head_quant_method(lm_head, quant_method)
        or not hasattr(lm_head, "weight")
        or processor.use_fp32_lm_head
        or use_intel_amx_backend(lm_head)
        or processor.rl_on_policy_target is not None
    ):
        return original(processor, hidden_states, lm_head, embedding_bias)
    if not getattr(lm_head, "_hcu_nn_checked", False):
        lm_head._hcu_nn_checked = True
        weight = lm_head.weight
        if (
            not torch.cuda.is_current_stream_capturing()
            and weight.ndim == 2
            and weight.dtype == torch.bfloat16
            and weight.is_cuda
            and weight.is_contiguous()
        ):
            weight.data = weight.data.t().contiguous().t()
            torch.cuda.empty_cache()
    if hidden_states.dtype == lm_head.weight.dtype:
        logits = gemm.try_skinny_nn_bf16(hidden_states, lm_head.weight.T)
        if logits is not None:
            return logits
    return original(processor, hidden_states, lm_head, embedding_bias)


def around_decoder_forward(original, layer, *args, **kwargs):
    if getattr(layer.self_attn, "_hcu_glm_gemm", False) and not getattr(
        layer, "_hcu_layouts_checked", False
    ):
        layer._hcu_layouts_checked = True
        if not torch.cuda.is_current_stream_capturing():
            for module in layer.modules():
                method = getattr(module, "quant_method", None)
                weight = getattr(module, "weight", None)
                if (
                    type(method).__name__ == "UnquantizedLinearMethod"
                    and isinstance(weight, torch.Tensor)
                    and type(weight.data) is torch.Tensor
                    and weight.is_cuda
                    and weight.dtype == torch.bfloat16
                    and weight.ndim == 2
                    and min(weight.shape) >= 1024
                    and weight.is_contiguous()
                ):
                    weight.data = weight.data.t().contiguous().t()
    return original(layer, *args, **kwargs)


def register_hooks():
    if not is_hcu():
        return
    from sglang.srt.plugins.hook_registry import HookRegistry, HookType
    from hcu_sglang.layers.attention.current_dsa.glm53_decode_mla import (
        around_rocm_prepare,
    )
    from hcu_sglang.layers.attention.current_dsa.dcp import around_mla_dcp_phase

    if envs.SGLANG_GLM_OPT_DECODE.get() or envs.SGLANG_GLM_LIGHTOP_FUSIONS.get():
        fusion_phases()
    if envs.SGLANG_GLM_HCU_GEMM.get():
        hcu_gemm()
    base = (
        "sglang.srt.models.deepseek_common.attention_forward_methods.forward_mla_rocm"
    )
    for target, hook in (
        (
            "sglang.srt.models.deepseek_common.attention_forward_methods.forward_mla.DeepseekMLAForwardMixin.is_dcp_mla_decode_phase",
            around_mla_dcp_phase,
        ),
        (
            "sglang.srt.models.deepseek_v2.DeepseekV2AttentionMLA.dispatch_attn_forward_method",
            around_attn_dispatch,
        ),
        (
            base + ".DeepseekMLARocmForwardMixin.forward_absorb_rocm_prepare",
            around_rocm_prepare,
        ),
        (base + ".rocm_absorb_q_bmm", around_q_bmm),
        (base + ".rocm_absorb_v_bmm", around_v_bmm),
        (
            base + ".DeepseekMLARocmForwardMixin.forward_absorb_rocm_core",
            around_rocm_core,
        ),
        (
            "sglang.srt.models.deepseek_v2.DeepseekV2AttentionMLA.forward_prepare",
            around_forward_prepare,
        ),
        (
            "sglang.srt.models.deepseek_v2.DeepseekV2DecoderLayer.forward",
            around_decoder_forward,
        ),
        (
            "sglang.srt.layers.radix_attention.RadixAttention.forward",
            around_radix_forward,
        ),
        (
            "sglang.srt.layers.logits_processor.LogitsProcessor._compute_lm_head",
            around_lm_head,
        ),
    ):
        HookRegistry.register(target, hook, HookType.AROUND)
