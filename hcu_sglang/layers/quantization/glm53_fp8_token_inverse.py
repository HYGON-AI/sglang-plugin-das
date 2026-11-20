# Copyright (c) 2026 Hygon Information Technology Co., Ltd.
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

import hashlib
import importlib.util
import json
import logging
import os
import sys
import sysconfig
from contextlib import contextmanager
from contextvars import ContextVar
from functools import lru_cache
from pathlib import Path

import torch
from compressed_tensors.quantization import QuantizationStrategy
from sglang.srt.environ import envs

from hcu_sglang.layers.attention.current_dsa.forward_batch_utils import effective_forward_mode
from hcu_sglang.utils.detection import is_hcu


_ATTENTION = ContextVar("hcu_glm53_fp8_token_attention", default=None)
_INPUT = ContextVar("hcu_glm53_fp8_token_input", default=None)
_LOGGER = logging.getLogger(__name__)
_CONFIG_SHA256 = "bc21bd68d92085465810c3da72c4b8a43b16c65c2c8f55f52547045f36c1abbf"
_SOURCE_SHA256 = "27ae5b80c407ccdb21640948f505185ef6c558529e3c3938aa305bea2e2fbdfc"
_REFERENCE_SHA256 = "e7870a15db717c3761e80725e238b8709ee1adb4ff92239121b7e4b8211f0361"
_MODULE = "token_inverse_diag"
_FLAGS = [
    "-O3",
    "-std=c++17",
    "--offload-arch=gfx938",
    "-DTORCH_VERSION_OVER_290",
    "--gpu-max-threads-per-block=1024",
    "-mllvm",
    "-enable-num-vgprs-512=true",
]


def _sha256(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


@lru_cache(maxsize=8)
def _config_matches(path, size, mtime_ns):
    del size, mtime_ns
    try:
        return _sha256(path) == _CONFIG_SHA256
    except OSError:
        return False


def _model_matches(owner, server_args):
    from sglang.srt.runtime_context import process_model_config

    try:
        config = process_model_config().hf_config
        path = Path(server_args.model_path) / "config.json"
        stat = path.stat()
    except (AttributeError, OSError, TypeError, ValueError):
        return False
    return (
        tuple(getattr(config, "architectures", None) or ()) == ("GlmMoeDsaForCausalLM",)
        and getattr(config, "model_type", None) == "glm_moe_dsa"
        and getattr(config, "num_hidden_layers", None) == 78
        and getattr(config, "hidden_size", None) == 6144
        and getattr(config, "n_routed_experts", None) == 256
        and getattr(config, "num_experts_per_tok", None) == 8
        and getattr(config, "moe_intermediate_size", None) == 2048
        and getattr(config, "first_k_dense_replace", None) == 3
        and not getattr(owner, "is_nextn", True)
        and getattr(owner, "use_dsa", False)
        and type(getattr(owner, "layer_id", None)) is int
        and 0 <= owner.layer_id < 78
        and getattr(owner, "hidden_size", None) == 6144
        and getattr(owner, "num_heads", None) == 64
        and getattr(owner, "num_local_heads", None) == 64
        and getattr(owner, "v_head_dim", None) == 256
        and getattr(owner, "kv_lora_rank", None) == 512
        and _config_matches(str(path), stat.st_size, stat.st_mtime_ns)
    )


def _eligible_attention(owner, batch, method):
    if not is_hcu() or batch is None or torch.version.hip is None:
        return False
    from sglang.srt.batch_invariant_ops import is_batch_invariant_mode_enabled
    from sglang.srt.layers.attention.dsa.utils import dsa_use_prefill_cp
    from sglang.srt.layers.cp.base import get_cp_strategy
    from sglang.srt.layers.cp.interleave import InterleaveCPStrategy
    from sglang.srt.mem_cache.memory_pool import DSATokenToKVPool
    from sglang.srt.model_executor.forward_batch_info import ForwardMode
    from sglang.srt.model_executor.forward_context import get_forward_context, has_forward_context
    from sglang.srt.model_executor.runner import get_is_capture_mode
    from sglang.srt.model_executor.runner_backend_utils.breakable_cuda_graph.context import is_in_breakable_cuda_graph
    from sglang.srt.model_executor.runner_backend_utils.tc_piecewise_cuda_graph import is_in_tc_piecewise_cuda_graph
    from sglang.srt.models.deepseek_common.attention_forward_methods.forward_methods import AttnForwardMethod
    from sglang.srt.runtime_context import get_exec, get_parallel, get_server_args
    from hcu_sglang.layers.attention.current_dsa.dsa_topk_backend import TopkTransformMethod
    from hcu_sglang.models.deepseek.current_v4.execution.cp_helpers import is_cp_v2_active

    if not has_forward_context():
        return False
    parallel = get_parallel()
    server_args = get_server_args()
    if (
        not _model_matches(owner, server_args)
        or method not in (AttnForwardMethod.MLA, AttnForwardMethod.MLA_ROCM)
        or getattr(server_args, "enable_lora", False)
        or getattr(server_args, "lora_paths", None)
        or envs.SGLANG_EXPERIMENTAL_LORA_OPTI.get()
        or (parallel.tp_size, parallel.attn_cp_size, parallel.moe_ep_size, parallel.pp_size) != (8, 8, 8, 1)
        or parallel.attn_tp_size != 1
        or parallel.enable_dsa_cache_layer_split
        or parallel.dcp_enabled
        or parallel.attn_dcp_size != 1
        or batch.forward_mode != ForwardMode.EXTEND
        or effective_forward_mode(batch) != ForwardMode.EXTEND
        or getattr(batch, "can_run_tbo", False)
        or getattr(batch, "tbo_parent_token_range", None) is not None
        or getattr(batch, "tbo_children", None) is not None
        or batch.attn_cp_metadata is None
        or not batch.extend_prefix_lens_cpu
        or not any(batch.extend_prefix_lens_cpu)
        or not dsa_use_prefill_cp(batch)
        or not is_cp_v2_active(batch)
        or not isinstance(get_cp_strategy(), InterleaveCPStrategy)
        or get_exec().deterministic.enable_deterministic_inference
        or is_batch_invariant_mode_enabled()
        or torch.compiler.is_compiling()
        or get_is_capture_mode()
        or is_in_breakable_cuda_graph()
        or is_in_tc_piecewise_cuda_graph()
        or torch.cuda.is_current_stream_capturing()
    ):
        return False
    backend = get_forward_context().attn_backend
    pool = getattr(backend, "token_to_kv_pool", None)
    topk_method = getattr(backend, "get_topk_transform_method", None)
    return (
        type(pool) is DSATokenToKVPool
        and hasattr(pool, "layer_transfer_counter")
        and pool.layer_transfer_counter is None
        and getattr(pool, "dsa_kv_cache_store_fp8", False)
        and getattr(pool, "kv_cache_dim", None) == 656
        and getattr(backend, "dcp_size", None) == 1
        and getattr(backend, "hisparse_coordinator", None) is None
        and not getattr(backend, "use_mha", True)
        and getattr(backend, "dsa_prefill_impl", None) == "flashmla_sparse"
        and callable(topk_method)
        and topk_method(ForwardMode.EXTEND) == TopkTransformMethod.RAGGED
    )


@contextmanager
def attention_scope(owner, batch, method):
    active = owner if _eligible_attention(owner, batch, method) else None
    token = _ATTENTION.set(active)
    try:
        yield
    finally:
        _ATTENTION.reset(token)


def _eligible_projection(scheme, layer, value):
    from sglang.srt.layers.quantization.compressed_tensors.schemes import compressed_tensors_w8a8_fp8 as Q

    owner = _ATTENTION.get()
    if owner is None or getattr(owner, "o_proj", None) is not layer or not isinstance(value, torch.Tensor):
        return False
    weight = getattr(layer, "weight", None)
    weight_scale = getattr(layer, "weight_scale", None)
    return (
        type(scheme) is Q.CompressedTensorsW8A8Fp8
        and getattr(layer, "scheme", None) is scheme
        and scheme.strategy == QuantizationStrategy.CHANNEL
        and not scheme.is_static_input_scheme
        and scheme.weight_block_size is None
        and getattr(layer, "input_scale", None) is None
        and getattr(layer, "input_is_parallel", False)
        and getattr(layer, "bias", None) is None
        and isinstance(weight, torch.Tensor)
        and weight.shape == (16384, 6144)
        and weight.stride() == (1, 16384)
        and weight.dtype == torch.float8_e4m3fn
        and isinstance(weight_scale, torch.Tensor)
        and weight_scale.shape == (6144, 1)
        and weight_scale.dtype == torch.float32
        and weight_scale.is_contiguous()
        and value.shape == (4096, 16384)
        and value.stride() == (16384, 1)
        and value.dtype == torch.bfloat16
        and value.is_cuda
        and weight.device == weight_scale.device == value.device
        and value.storage_offset() == 0
        and value.data_ptr() % 16 == 0
    )


@contextmanager
def projection_scope(scheme, layer, value):
    active = value if _eligible_projection(scheme, layer, value) else None
    token = _INPUT.set(active)
    try:
        yield
    finally:
        _INPUT.reset(token)


@lru_cache(maxsize=1)
def _warn_missing_extension():
    _LOGGER.warning("GLM-5.3 FP8 token inverse requested without an extension path; using the existing route")


def _runtime_abi():
    if (
        sysconfig.get_config_var("SOABI") != "cpython-310-x86_64-linux-gnu"
        or torch.__version__ != "2.11.0"
        or torch.version.hip != "6.3.26113"
        or not torch.compiled_with_cxx11_abi()
    ):
        raise RuntimeError("GLM-5.3 FP8 token inverse requires the validated Python/Torch/HIP ABI")
    root = Path(os.environ.get("ROCM_HOME") or os.environ.get("ROCM_PATH") or "/opt/dtk")
    dtk = (root / ".info" / "rocm_version").read_text(encoding="utf-8").strip()
    if dtk.split("-", 1)[0].replace(".", "").replace(" ", "").replace("V", "") != "26041":
        raise RuntimeError(f"GLM-5.3 FP8 token inverse requires DTK 26.04.1, found {dtk!r}")


@lru_cache(maxsize=4)
def _load_extension(path):
    path = Path(path)
    if not path.is_absolute():
        raise ValueError("GLM-5.3 FP8 token inverse extension path must be absolute")
    path = path.resolve(strict=True)
    if path.name != _MODULE + ".so":
        raise ValueError(f"GLM-5.3 FP8 token inverse extension must be named {_MODULE}.so")
    _runtime_abi()
    binary_sha = _sha256(path)
    if binary_sha != _REFERENCE_SHA256:
        manifest = json.loads((path.parent / "build-manifest.json").read_text(encoding="utf-8"))
        source = manifest.get("source", {})
        provenance = Path(__file__).with_name("csrc") / "glm53_fp8_token_inverse.source.json"
        builder = Path(__file__).with_name("build_glm53_fp8_token_inverse.py")
        if (
            manifest.get("status") != "BUILT_AND_IMPORTED_GPU_UNVERIFIED"
            or manifest.get("module") != _MODULE
            or manifest.get("binary") != path.name
            or manifest.get("binary_sha256") != binary_sha
            or source.get("source_sha256") != _SOURCE_SHA256
            or source.get("reference_binary_sha256") != _REFERENCE_SHA256
            or source.get("module") != _MODULE
            or source.get("headers") != {}
            or manifest.get("source_manifest_sha256") != _sha256(provenance)
            or manifest.get("build_tool_sha256") != _sha256(builder)
            or manifest.get("hip_flags") != _FLAGS
            or manifest.get("python_abi") != sysconfig.get_config_var("SOABI")
            or manifest.get("torch") != torch.__version__
            or manifest.get("hip") != torch.version.hip
            or manifest.get("dtk") != "26041"
            or manifest.get("cxx11_abi") is not True
        ):
            raise RuntimeError("GLM-5.3 FP8 token inverse extension manifest or runtime ABI mismatch")
    existing = sys.modules.get(_MODULE)
    if existing is not None:
        if Path(existing.__file__).resolve() != path:
            raise RuntimeError("GLM-5.3 FP8 token inverse extension was loaded from another path")
        return existing
    spec = importlib.util.spec_from_file_location(_MODULE, path)
    if spec is None or spec.loader is None:
        raise RuntimeError("Cannot load GLM-5.3 FP8 token inverse extension")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    if not callable(getattr(module, "run", None)):
        raise RuntimeError("GLM-5.3 FP8 token inverse extension has no run entry")
    sys.modules[_MODULE] = module
    return module


def try_token_inverse(value, output, scale):
    bound = _INPUT.get()
    if bound is None or not envs.SGLANG_HCU_GLM53_FP8_TOKEN_INVERSE.get():
        return False
    if (
        value.device != bound.device
        or value.dtype != bound.dtype
        or value.shape != bound.shape
        or value.stride() != bound.stride()
        or value.storage_offset() != bound.storage_offset()
        or value.data_ptr() != bound.data_ptr()
        or value.untyped_storage().data_ptr() != bound.untyped_storage().data_ptr()
        or value.shape != (4096, 16384)
        or value.dtype != torch.bfloat16
        or not value.is_cuda
        or not value.is_contiguous()
        or value.storage_offset() != 0
        or output.shape != value.shape
        or output.dtype != torch.float8_e4m3fn
        or scale.shape != (4096, 1)
        or scale.dtype != torch.float32
        or torch.compiler.is_compiling()
        or torch.cuda.is_current_stream_capturing()
        or torch.cuda.get_device_properties(value.device).gcnArchName.split(":", 1)[0] != "gfx938"
    ):
        return False
    tensors = (value, output, scale)
    if (
        not all(t.is_cuda and t.device == value.device for t in tensors)
        or not all(t.is_contiguous() and t.storage_offset() == 0 and t.data_ptr() % 16 == 0 for t in tensors)
        or len({t.untyped_storage().data_ptr() for t in tensors}) != len(tensors)
    ):
        return False
    path = envs.SGLANG_HCU_GLM53_FP8_TOKEN_INVERSE_SO.get()
    if not path:
        _warn_missing_extension()
        return False
    _load_extension(path).run(value, output, scale, 1)
    return True
