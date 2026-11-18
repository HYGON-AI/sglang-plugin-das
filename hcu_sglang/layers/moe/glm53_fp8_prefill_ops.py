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

from hcu_sglang.layers.moe.glm53_fp8_k1_indirect import _eligible_scope


_OWNER = ContextVar("hcu_glm53_fp8_prefill_ops_owner", default=None)
_LOGGER = logging.getLogger(__name__)
_SOURCE_SHA256 = {
    "activation": "41d63cdbf951c1303192734a3538fe51c142121688091811109afc133b558ac1",
    "gather": "603b6928c7b4671eeab890b3de8158bbb9e0c2be6b174584961b04c52dc114b7",
}
_REFERENCE_SHA256 = {
    "activation": "a13d3751d5cde8e9a4b66ea1cca769431dbb6051a1ef113bab6ef7b129d4bf9a",
    "gather": "e225959f6b88a57af897471241ae994ed24d1e1de48b6a0ef07c0383df6f638d",
}
_HEADERS = {
    "reduce.h": "071eb3de2cb5688c5b37c322ad45d09cd2ff374f12d435a326f031ff1187d7aa",
    "reduce_hip.h": "db6e51131423c0f0f2e401b7ff2d991d70f788bbbe39e9ba8c1db07eaee1d5da",
}
_MODULE = {"activation": "activation_inverse_diag", "gather": "epgather_k8_diag"}
_ACTIVATION_ROWS = frozenset((14336, 84224, 84992))


@contextmanager
def prefill_scope(owner, batch, hidden_states):
    active = owner.experts if _eligible_scope(owner, batch, hidden_states) else None
    token = _OWNER.set(active)
    try:
        yield
    finally:
        _OWNER.reset(token)


def _sha256(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


@lru_cache(maxsize=2)
def _warn_missing_extension(kind):
    _LOGGER.warning("GLM-5.3 FP8 %s requested without an extension path; using the existing route", kind)


def _runtime_abi():
    if (
        sysconfig.get_config_var("SOABI") != "cpython-310-x86_64-linux-gnu"
        or torch.__version__ != "2.11.0"
        or torch.version.hip != "6.3.26113"
        or not torch.compiled_with_cxx11_abi()
    ):
        raise RuntimeError("GLM-5.3 FP8 prefill extension requires the validated Python/Torch/HIP ABI")
    root = Path(os.environ.get("ROCM_HOME") or os.environ.get("ROCM_PATH") or "/opt/dtk")
    dtk = (root / ".info" / "rocm_version").read_text(encoding="utf-8").strip()
    if dtk.split("-", 1)[0].replace(".", "").replace(" ", "").replace("V", "") != "26041":
        raise RuntimeError(f"GLM-5.3 FP8 prefill extension requires DTK 26.04.1, found {dtk!r}")


@lru_cache(maxsize=4)
def _load_extension(kind, path):
    path = Path(path)
    if not path.is_absolute():
        raise ValueError("GLM-5.3 FP8 prefill extension path must be absolute")
    path = path.resolve(strict=True)
    name = _MODULE[kind]
    if path.name != name + ".so":
        raise ValueError(f"GLM-5.3 FP8 prefill extension must be named {name}.so")
    _runtime_abi()
    binary_sha = _sha256(path)
    if binary_sha != _REFERENCE_SHA256[kind]:
        manifest = json.loads((path.parent / "build-manifest.json").read_text(encoding="utf-8"))
        source = manifest.get("source", {})
        provenance = Path(__file__).with_name("csrc") / "glm53_fp8_prefill_ops.source.json"
        builder = Path(__file__).with_name("build_glm53_fp8_prefill_ops.py")
        flags = ["-O3", "-std=c++17", "--offload-arch=gfx938"]
        if kind == "activation":
            flags.append("-DTORCH_VERSION_OVER_290")
        flags.extend(("--gpu-max-threads-per-block=512", "-mllvm", "-enable-num-vgprs-512=true"))
        if (
            manifest.get("status") != "BUILT_AND_IMPORTED_GPU_UNVERIFIED"
            or manifest.get("kind") != kind
            or manifest.get("module") != name
            or manifest.get("binary") != path.name
            or manifest.get("binary_sha256") != binary_sha
            or source.get("source_sha256") != _SOURCE_SHA256[kind]
            or source.get("module") != name
            or source.get("reference_binary_sha256") != _REFERENCE_SHA256[kind]
            or source.get("headers") != (_HEADERS if kind == "activation" else {})
            or manifest.get("source_manifest_sha256") != _sha256(provenance)
            or manifest.get("build_tool_sha256") != _sha256(builder)
            or manifest.get("hip_flags") != flags
            or manifest.get("python_abi") != sysconfig.get_config_var("SOABI")
            or manifest.get("torch") != torch.__version__
            or manifest.get("hip") != torch.version.hip
            or manifest.get("dtk") != "26041"
            or manifest.get("cxx11_abi") is not True
        ):
            raise RuntimeError("GLM-5.3 FP8 prefill extension manifest or runtime ABI mismatch")
    existing = sys.modules.get(name)
    if existing is not None:
        if Path(existing.__file__).resolve() != path:
            raise RuntimeError(f"GLM-5.3 FP8 prefill extension {name} was loaded from another path")
        return existing
    spec = importlib.util.spec_from_file_location(name, path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"Cannot load GLM-5.3 FP8 prefill extension {name}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    if not callable(getattr(module, "run", None)):
        raise RuntimeError(f"GLM-5.3 FP8 prefill extension {name} has no run entry")
    sys.modules[name] = module
    return module


def _layer_eligible(layer):
    if _OWNER.get() is not layer:
        return False
    scheme = getattr(layer, "scheme", None)
    config = getattr(layer, "moe_runner_config", None)
    return (
        type(scheme).__name__ == "CompressedTensorsW8A8Fp8MoE"
        and getattr(getattr(scheme, "weight_quant", None), "strategy", None)
        == QuantizationStrategy.CHANNEL
        and getattr(getattr(scheme, "input_quant", None), "strategy", None)
        == QuantizationStrategy.TOKEN
        and getattr(scheme, "weight_block_size", None) is None
        and not getattr(scheme, "static_input_scales", True)
        and getattr(layer, "use_fp8_w8a8", False)
        and not getattr(layer, "use_block_quant", True)
        and not getattr(layer, "use_w4afp8", True)
        and not getattr(layer, "use_w4a8_marlin", True)
        and not getattr(layer, "use_w8a8_marlin", True)
        and not getattr(layer, "use_w4a16_marlin", False)
        and config is not None
        and config.activation == "silu"
        and not torch.compiler.is_compiling()
        and not torch.cuda.is_current_stream_capturing()
    )


def _device_eligible(value):
    return (
        value.is_cuda
        and torch.version.hip is not None
        and torch.cuda.get_device_properties(value.device).gcnArchName.split(":", 1)[0] == "gfx938"
    )


def try_expert_activation_inverse(layer, value):
    if not envs.SGLANG_HCU_GLM53_FP8_EXPERT_INVERSE.get() or not _layer_eligible(layer):
        return None
    if (
        layer.moe_runner_config.swiglu_limit is not None
        or value.ndim != 2
        or value.shape[0] not in _ACTIVATION_ROWS
        or value.shape[1] != 4096
        or value.dtype != torch.bfloat16
        or not value.is_contiguous()
        or value.storage_offset() != 0
        or value.data_ptr() % 16
        or not _device_eligible(value)
    ):
        return None
    path = envs.SGLANG_HCU_GLM53_FP8_EXPERT_INVERSE_SO.get()
    if not path:
        _warn_missing_extension("expert inverse")
        return None
    module = _load_extension("activation", path)
    quantized = torch.empty((value.shape[0], 2048), dtype=torch.float8_e4m3fn, device=value.device)
    scale = torch.empty((value.shape[0], 1), dtype=torch.float32, device=value.device)
    module.run(value, quantized.view(torch.uint8), scale, 1)
    return quantized, scale


def try_gather_k8(layer, input_tensor, ids, weights, input_index, output):
    if not envs.SGLANG_HCU_GLM53_FP8_EP_GATHER_K8.get() or not _layer_eligible(layer):
        return False
    if (
        input_tensor.ndim != 2
        or output.ndim != 2
        or ids.ndim != 2
        or ids.shape[1] != 8
        or weights.shape != ids.shape
        or input_index.shape != ids.shape
        or input_tensor.shape[1] != 6144
        or output.shape != (ids.shape[0], 6144)
        or input_tensor.shape[0] == 0
        or output.shape[0] == 0
        or input_tensor.dtype != torch.bfloat16
        or output.dtype != torch.bfloat16
        or ids.dtype != torch.int64
        or weights.dtype != torch.float32
        or input_index.dtype != torch.int32
        or not _device_eligible(input_tensor)
    ):
        return False
    tensors = (input_tensor, ids, weights, input_index, output)
    if (
        not all(t.is_cuda and t.device == input_tensor.device for t in tensors)
        or not all(t.is_contiguous() and t.data_ptr() % 16 == 0 for t in tensors)
        or len({t.untyped_storage().data_ptr() for t in tensors}) != len(tensors)
    ):
        return False
    path = envs.SGLANG_HCU_GLM53_FP8_EP_GATHER_K8_SO.get()
    if not path:
        _warn_missing_extension("K8 gather")
        return False
    module = _load_extension("gather", path)
    module.run(input_tensor, ids, weights, input_index, output, 1)
    return True
