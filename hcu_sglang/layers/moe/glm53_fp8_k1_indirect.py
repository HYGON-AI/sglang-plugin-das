from __future__ import annotations

import hashlib
import importlib.util
import json
import logging
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


_MODULE = "hcu_sglang_glm53_fp8_k1_indirect"
_CONFIG_SHA256 = "bc21bd68d92085465810c3da72c4b8a43b16c65c2c8f55f52547045f36c1abbf"
_SOURCE_COMMIT = "ca0c569f424dd1b10a3f65e89c1f5e2b4bc502b2"
_HEADER_SHA256 = "7fd1262d9afa93dad3c5493b4432f15635431e893c1f2aa012e3bc3ae2a927ca"
_owner = ContextVar("hcu_glm53_fp8_k1_owner", default=None)
_logger = logging.getLogger(__name__)


@lru_cache(maxsize=8)
def _config_matches(path, size, mtime_ns):
    del size, mtime_ns
    try:
        return hashlib.sha256(Path(path).read_bytes()).hexdigest() == _CONFIG_SHA256
    except OSError:
        return False


def _model_matches(owner, server_args):
    config = owner.config
    if (
        tuple(getattr(config, "architectures", None) or ()) != ("GlmMoeDsaForCausalLM",)
        or getattr(config, "model_type", None) != "glm_moe_dsa"
        or getattr(config, "num_hidden_layers", None) != 78
        or getattr(config, "hidden_size", None) != 6144
        or getattr(config, "n_routed_experts", None) != 256
        or getattr(config, "num_experts_per_tok", None) != 8
        or getattr(config, "moe_intermediate_size", None) != 2048
        or getattr(config, "first_k_dense_replace", None) != 3
        or getattr(owner, "is_nextn", True)
    ):
        return False
    try:
        path = Path(server_args.model_path) / "config.json"
        stat = path.stat()
    except (OSError, TypeError):
        return False
    return _config_matches(str(path), stat.st_size, stat.st_mtime_ns)


def _eligible_scope(owner, batch, hidden_states):
    if not is_hcu() or batch is None or not hidden_states.is_cuda or torch.version.hip is None:
        return False
    from sglang.srt.batch_invariant_ops import is_batch_invariant_mode_enabled
    from sglang.srt.layers.attention.dsa.utils import dsa_use_prefill_cp
    from sglang.srt.layers.cp.base import is_interleave
    from sglang.srt.layers.moe import get_moe_a2a_backend
    from sglang.srt.mem_cache.memory_pool import DSATokenToKVPool
    from sglang.srt.model_executor.forward_batch_info import ForwardMode
    from sglang.srt.model_executor.forward_context import get_forward_context, has_forward_context
    from sglang.srt.model_executor.runner import get_is_capture_mode
    from sglang.srt.model_executor.runner_backend_utils.breakable_cuda_graph.context import is_in_breakable_cuda_graph
    from sglang.srt.model_executor.runner_backend_utils.tc_piecewise_cuda_graph import is_in_tc_piecewise_cuda_graph
    from sglang.srt.runtime_context import get_exec, get_parallel, get_server_args
    from hcu_sglang.layers.attention.current_dsa.dsa_topk_backend import TopkTransformMethod
    from hcu_sglang.models.deepseek.current_v4.execution.cp_helpers import is_cp_v2_active

    if not has_forward_context():
        return False
    parallel = get_parallel()
    server_args = get_server_args()
    if (
        not _model_matches(owner, server_args)
        or not get_moe_a2a_backend().is_deepep()
        or getattr(owner, "num_fused_shared_experts", None) != 0
        or getattr(owner, "is_hash", True)
        or getattr(server_args, "enable_lora", False)
        or getattr(server_args, "lora_paths", None)
        or (parallel.tp_size, parallel.attn_cp_size, parallel.moe_ep_size, parallel.pp_size) != (8, 8, 8, 1)
        or parallel.enable_dsa_cache_layer_split
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
        or not is_interleave()
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
        and backend.get_topk_transform_method(ForwardMode.EXTEND) == TopkTransformMethod.RAGGED
    )


@contextmanager
def forward_scope(owner, batch, hidden_states):
    active = owner.experts if _eligible_scope(owner, batch, hidden_states) else None
    token = _owner.set(active)
    try:
        yield
    finally:
        _owner.reset(token)


def _sha256(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


@lru_cache(maxsize=1)
def _warn_missing_extension():
    _logger.warning("GLM-5.3 FP8 K1 indirect requested without an extension path; using the existing route")


@lru_cache(maxsize=4)
def _load_extension(path):
    path = Path(path)
    if not path.is_absolute():
        raise ValueError("GLM-5.3 FP8 K1 extension path must be absolute")
    path = path.resolve(strict=True)
    manifest = json.loads((path.parent / "build-manifest.json").read_text(encoding="utf-8"))
    source = manifest.get("source", {})
    csrc = Path(__file__).with_name("csrc")
    provenance = csrc / "glm53_fp8_k1_indirect.source.json"
    binding = csrc / "glm53_fp8_k1_indirect.cu"
    build_tool = Path(__file__).with_name("build_glm53_fp8_k1_indirect.py")
    if (
        manifest.get("status") != "BUILT_AND_IMPORTED_GPU_UNVERIFIED"
        or manifest.get("binary") != path.name
        or path.name != _MODULE + ".so"
        or manifest.get("binary_sha256") != _sha256(path)
        or source.get("commit") != _SOURCE_COMMIT
        or source.get("header_sha256") != _HEADER_SHA256
        or manifest.get("source_manifest_sha256") != _sha256(provenance)
        or manifest.get("binding_sha256") != _sha256(binding)
        or manifest.get("build_tool_sha256") != _sha256(build_tool)
        or manifest.get("python_abi") != sysconfig.get_config_var("SOABI")
        or manifest.get("torch") != torch.__version__
        or manifest.get("hip") != torch.version.hip
        or manifest.get("dtk") != "26041"
        or manifest.get("cxx11_abi") is not True
        or not torch.compiled_with_cxx11_abi()
    ):
        raise RuntimeError("GLM-5.3 FP8 K1 extension manifest or runtime ABI mismatch")
    existing = sys.modules.get(_MODULE)
    if existing is not None:
        if Path(existing.__file__).resolve() != path:
            raise RuntimeError("GLM-5.3 FP8 K1 extension already loaded from another path")
        return existing
    spec = importlib.util.spec_from_file_location(_MODULE, path)
    if spec is None or spec.loader is None:
        raise RuntimeError("Cannot load GLM-5.3 FP8 K1 extension")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    if not callable(getattr(module, "metadata", None)) or not callable(getattr(module, "indirect", None)):
        raise RuntimeError("GLM-5.3 FP8 K1 extension entry points are missing")
    sys.modules[_MODULE] = module
    return module


def try_indirect(layer, x, scale, topk, weights, counts, padded_counts, weight, n):
    if _owner.get() is not layer:
        return None
    scheme = getattr(layer, "scheme", None)
    config = layer.moe_runner_config
    if (
        type(scheme).__name__ != "CompressedTensorsW8A8Fp8MoE"
        or scheme.weight_quant.strategy != QuantizationStrategy.CHANNEL
        or scheme.input_quant.strategy != QuantizationStrategy.TOKEN
        or scheme.weight_block_size is not None
        or scheme.static_input_scales
        or config.activation != "silu"
        or len(counts) != 32
        or any(type(count) is not int or count < 0 or count % 256 for count in counts)
    ):
        return None
    m = sum(counts)
    if (
        not 0 < m < 2**31
        or not 0 < m * n * 2 < 2**31
        or x.ndim != 2
        or x.shape[1] != 6144
        or not 0 < x.shape[0] * 6144 < 2**31
        or n != 4096
        or scale is None
        or scale.shape != (x.shape[0], 1)
        or topk.shape != (x.shape[0], 8)
        or weights.shape != topk.shape
        or padded_counts.shape != (32,)
        or weight[0].shape != (32, 96, 128, 2, 4, 16, 16)
        or weight[1].shape != (32, 4096, 1)
        or x.dtype != torch.float8_e4m3fn
        or weight[0].dtype != torch.float8_e4m3fn
        or scale.dtype != torch.float32
        or weight[1].dtype != torch.float32
        or weights.dtype != torch.float32
        or topk.dtype != torch.int64
        or padded_counts.dtype != torch.int32
    ):
        return None
    tensors = (x, scale, topk, weights, padded_counts, *weight)
    if (
        not all(t.is_cuda and t.device == x.device for t in tensors)
        or not all(t.is_contiguous() and t.data_ptr() % 16 == 0 for t in tensors)
        or len({t.untyped_storage().data_ptr() for t in tensors}) != len(tensors)
        or torch.cuda.get_device_properties(x.device).gcnArchName.split(":", 1)[0] != "gfx938"
    ):
        return None
    path = envs.SGLANG_HCU_GLM53_FP8_K1_INDIRECT_SO.get()
    if not path:
        _warn_missing_extension()
        return None
    module = _load_extension(path)
    row_src = torch.empty((m,), dtype=torch.int32, device=x.device)
    m_indices = torch.empty_like(row_src)
    output_index = torch.empty_like(topk, dtype=torch.int32)
    counters = torch.empty_like(padded_counts)
    ordinary_quant = config.swiglu_limit is None and not envs.SGLANG_HCU_GLM53_FP8_EXPERT_INVERSE.get()
    output_factory = torch.empty if ordinary_quant else torch.zeros
    gateup = output_factory((m, n), dtype=torch.bfloat16, device=x.device)
    module.metadata(topk, padded_counts, counters, row_src, m_indices, output_index)
    module.indirect(x, weight[0], gateup, scale, weight[1], m_indices, row_src)
    return gateup, m_indices, output_index
