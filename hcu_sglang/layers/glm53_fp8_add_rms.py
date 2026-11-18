# Copyright (c) 2026 Hygon Information Technology Co., Ltd.
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

import importlib.util
import logging
import math
import sys
from functools import lru_cache
from pathlib import Path

import torch
from compressed_tensors.quantization import QuantizationStrategy
from sglang.srt.environ import envs


_LOGGER = logging.getLogger(__name__)
_REFERENCE_SHA256 = "11e6a0dff8d96913fb6a22ff12adee7179c004a3dab97c2443fa2bbb0ecab295"


def _channel_fp8(scheme):
    from sglang.srt.layers.quantization.compressed_tensors.schemes.compressed_tensors_w8a8_fp8 import (
        CompressedTensorsW8A8Fp8,
    )

    return (
        isinstance(scheme, CompressedTensorsW8A8Fp8)
        and scheme.strategy == QuantizationStrategy.CHANNEL
        and scheme.weight_block_size is None
        and not scheme.is_static_input_scheme
    )


def mark_decoder_norms(decoder):
    config = decoder.config
    if (
        decoder.is_nextn
        or tuple(getattr(config, "architectures", None) or ()) != ("GlmMoeDsaForCausalLM",)
        or getattr(config, "model_type", None) != "glm_moe_dsa"
        or getattr(config, "hidden_size", None) != 6144
        or not getattr(decoder.self_attn, "use_dsa", False)
    ):
        return
    quant = getattr(config, "quantization_config", None)
    if not isinstance(quant, dict) or quant.get("quant_method") != "compressed-tensors":
        return
    groups = quant.get("config_groups")
    if not isinstance(groups, dict) or set(groups) != {"FP8_DYNAMIC"}:
        return
    group = groups["FP8_DYNAMIC"]
    if not isinstance(group, dict):
        return
    weight, activation = group.get("weights") or {}, group.get("input_activations") or {}
    if not isinstance(weight, dict) or not isinstance(activation, dict):
        return
    if (
        weight.get("num_bits") != 8
        or weight.get("type") != "float"
        or weight.get("strategy") != "channel"
        or weight.get("block_structure") is not None
        or activation.get("num_bits") != 8
        or activation.get("type") != "float"
        or activation.get("strategy") != "token"
        or activation.get("dynamic") is not True
        or activation.get("block_structure") is not None
    ):
        return
    schemes = tuple(
        getattr(getattr(decoder.self_attn, name, None), "scheme", None)
        for name in ("fused_qkv_a_proj_with_mqa", "q_b_proj")
    )
    if not all(_channel_fp8(scheme) for scheme in schemes):
        return
    for role in ("input_layernorm", "post_attention_layernorm"):
        norm = getattr(decoder, role)
        norm._glm53_fp8_add_rms_owner = (decoder.layer_id, role)
        norm._glm53_fp8_add_rms_schemes = schemes


@lru_cache(maxsize=1)
def _warn_missing_extension():
    _LOGGER.warning("GLM-5.3 FP8 add-RMS requested without an extension path; using the existing route")


@lru_cache(maxsize=1)
def _load_extension(path):
    from hcu_sglang.layers.moe.glm53_fp8_prefill_ops import _runtime_abi, _sha256

    path = Path(path)
    if not path.is_absolute():
        raise ValueError("GLM-5.3 FP8 add-RMS extension path must be absolute")
    path = path.resolve(strict=True)
    _runtime_abi()
    if _sha256(path) != _REFERENCE_SHA256:
        raise RuntimeError("GLM-5.3 FP8 add-RMS requires the fixed reference binary")
    name = "add_rms_align16_diag"
    existing = sys.modules.get(name)
    if existing is not None:
        if Path(existing.__file__).resolve() != path:
            raise RuntimeError("GLM-5.3 FP8 add-RMS was loaded from another path")
        return existing
    spec = importlib.util.spec_from_file_location(name, path)
    if spec is None or spec.loader is None:
        raise RuntimeError("Cannot load GLM-5.3 FP8 add-RMS extension")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    if not callable(getattr(module, "run", None)):
        raise RuntimeError("GLM-5.3 FP8 add-RMS extension has no run entry")
    sys.modules[name] = module
    return module


def try_add_rms(norm, x, residual, post_residual_addition, quant_linear):
    if torch.compiler.is_compiling() or not envs.SGLANG_HCU_GLM53_FP8_ADD_RMS_ALIGN16.get():
        return False
    if residual is None or post_residual_addition is not None or quant_linear is not None:
        return False
    from sglang.srt.model_executor.forward_context import get_forward_context, has_forward_context
    from sglang.srt.model_executor.runner import get_is_capture_mode
    from sglang.srt.model_executor.runner_backend_utils.breakable_cuda_graph.context import is_in_breakable_cuda_graph
    from sglang.srt.model_executor.runner_backend_utils.tc_piecewise_cuda_graph import is_in_tc_piecewise_cuda_graph

    if not has_forward_context():
        return False
    backend = get_forward_context().attn_backend
    if (
        not getattr(backend, "_physical_kv_compact_eligible", False)
        or backend.forward_metadata is not backend._physical_kv_compact_scope_metadata
        or not getattr(norm, "_glm53_fp8_add_rms_owner", None)
        or getattr(norm, "_quant_linear", None) is not None
        or get_is_capture_mode()
        or is_in_breakable_cuda_graph()
        or is_in_tc_piecewise_cuda_graph()
    ):
        return False
    schemes = getattr(norm, "_glm53_fp8_add_rms_schemes", ())
    if len(schemes) != 2 or not all(_channel_fp8(scheme) for scheme in schemes):
        return False
    if (
        norm.variance_size_override is not None
        or norm.cast_x_before_out_mul
        or norm.fp32_residual
        or not norm.has_weight
        or norm.override_orig_dtype is not None
        or getattr(norm, "x_pad_to_multiple", 0)
        or not math.isfinite(norm.variance_epsilon)
        or norm.variance_epsilon <= 0
    ):
        return False
    weight = norm.weight.data
    tensors = (x, residual, weight)
    if (
        x.ndim != 2
        or x.shape[0] not in (704, 832, 1408, 3520, 4096)
        or x.shape[1] != 6144
        or residual.shape != x.shape
        or weight.shape != (6144,)
        or any(t.dtype != torch.bfloat16 for t in tensors)
        or not all(t.is_cuda and t.device == x.device for t in tensors)
        or not all(t.is_contiguous() and t.data_ptr() % 16 == 0 for t in tensors)
        or len({t.untyped_storage().data_ptr() for t in tensors}) != 3
    ):
        return False
    if torch.cuda.is_current_stream_capturing():
        return False
    if torch.cuda.get_device_properties(x.device).gcnArchName.split(":", 1)[0] != "gfx938":
        return False
    path = envs.SGLANG_HCU_GLM53_FP8_ADD_RMS_ALIGN16_SO.get()
    if not path:
        _warn_missing_extension()
        return False
    _load_extension(path).run(x, residual, weight, norm.variance_epsilon, 1)
    return True
