# Copyright (c) 2026 Hygon Information Technology Co., Ltd.
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

import hashlib
import importlib.util
import json
import os
import sys
import sysconfig
import threading
from functools import lru_cache
from pathlib import Path

import torch


_MODULE = "hcu_sglang_glm53_fp8_decode_skinny"
_REFERENCE_PACKAGE = "_hcu_sglang_glm53_fp8_skinny_reference"
_REFERENCE_FILES = {
    "__init__.py": "ed2807c5589a074a4f137a788a8fc6f8a5ad571a633d7fc37d4813530abb0284",
    "skinny_cfg.json": "be50e87f4d0d6664bbb0d65865406d8dabb5c3f95a8494fcd8eb163b52860339",
    "_C.cpython-310-x86_64-linux-gnu.so": "8aa3f2e39bf93dfca597714d80511f355cb6b600a5b09d0c0fdd86c83fac5e34",
}
_SOURCE_SHA256 = "6b3f733cb8025fbfaa7871aa609164b19866b99561f37fb6aa0d5ad472dc760e"
_FLAGS = ["-O3", "--offload-arch=gfx938"]
_LOCK = threading.RLock()


def _sha256(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _runtime_abi():
    if (
        sysconfig.get_config_var("SOABI") != "cpython-310-x86_64-linux-gnu"
        or torch.__version__ != "2.11.0"
        or torch.version.hip != "6.3.26113"
        or not torch.compiled_with_cxx11_abi()
    ):
        raise RuntimeError("GLM-5.3 decode skinny requires the validated Python/Torch/HIP ABI")
    root = Path(os.environ.get("ROCM_HOME") or os.environ.get("ROCM_PATH") or "/opt/dtk")
    dtk = (root / ".info" / "rocm_version").read_text(encoding="utf-8").strip()
    if dtk.split("-", 1)[0].replace(".", "").replace(" ", "").replace("V", "") != "26041":
        raise RuntimeError(f"GLM-5.3 decode skinny requires DTK 26.04.1, found {dtk!r}")


def _absolute(path):
    path = Path(path)
    if not path.is_absolute():
        raise ValueError("GLM-5.3 decode skinny path must be absolute")
    return path.resolve(strict=True)


def _check_entries(module):
    if not all(callable(getattr(module, name, None)) for name in ("skinny_gemm", "bmm_skinny")):
        raise RuntimeError("GLM-5.3 decode skinny extension has missing entries")
    return module


@lru_cache(maxsize=1)
def _load_reference(path):
    directory = _absolute(path)
    if not directory.is_dir():
        raise ValueError("GLM-5.3 decode skinny reference must be a package directory")
    _runtime_abi()
    for name, expected in _REFERENCE_FILES.items():
        if _sha256(directory / name) != expected:
            raise RuntimeError(f"GLM-5.3 decode skinny reference hash mismatch: {name}")
    with _LOCK:
        existing = sys.modules.get(_REFERENCE_PACKAGE)
        if existing is not None:
            if (
                Path(existing.__file__).resolve() != directory / "__init__.py"
                or Path(existing._C.__file__).resolve() != directory / "_C.cpython-310-x86_64-linux-gnu.so"
            ):
                raise RuntimeError("GLM-5.3 decode skinny reference already loaded from another path")
            return _check_entries(existing._C)
        spec = importlib.util.spec_from_file_location(
            _REFERENCE_PACKAGE,
            directory / "__init__.py",
            submodule_search_locations=[str(directory)],
        )
        if spec is None or spec.loader is None:
            raise RuntimeError("Cannot load the GLM-5.3 decode skinny reference")
        package = importlib.util.module_from_spec(spec)
        sys.modules[_REFERENCE_PACKAGE] = package
        try:
            spec.loader.exec_module(package)
            if Path(package._C.__file__).resolve() != directory / "_C.cpython-310-x86_64-linux-gnu.so":
                raise RuntimeError("GLM-5.3 decode skinny reference resolved to a different binary")
            return _check_entries(package._C)
        except Exception:
            sys.modules.pop(_REFERENCE_PACKAGE + "._C", None)
            sys.modules.pop(_REFERENCE_PACKAGE, None)
            raise


@lru_cache(maxsize=1)
def _load_built(path):
    path = _absolute(path)
    if not path.is_file() or path.name != _MODULE + ".so":
        raise ValueError(f"GLM-5.3 decode skinny extension must be named {_MODULE}.so")
    _runtime_abi()
    binary_sha = _sha256(path)
    manifest = json.loads((path.parent / "build-manifest.json").read_text(encoding="utf-8"))
    provenance = Path(__file__).with_name("csrc") / "glm53_fp8_decode_skinny.source.json"
    builder = Path(__file__).with_name("build_glm53_fp8_decode_skinny.py")
    from hcu_sglang.layers.quantization.build_glm53_fp8_decode_skinny import _BINDING

    source = manifest.get("source", {})
    if (
        manifest.get("status") != "BUILT_AND_IMPORTED_GPU_UNVERIFIED"
        or manifest.get("module") != _MODULE
        or manifest.get("binary") != path.name
        or manifest.get("binary_sha256") != binary_sha
        or source.get("source_sha256") != _SOURCE_SHA256
        or source.get("reference_binary_sha256") != _REFERENCE_FILES["_C.cpython-310-x86_64-linux-gnu.so"]
        or source.get("module") != _MODULE
        or source != json.loads(provenance.read_text(encoding="utf-8"))
        or manifest.get("source_manifest_sha256") != _sha256(provenance)
        or manifest.get("build_tool_sha256") != _sha256(builder)
        or manifest.get("binding_sha256") != hashlib.sha256(_BINDING.encode("utf-8")).hexdigest()
        or manifest.get("hip_flags") != _FLAGS
        or manifest.get("python_abi") != sysconfig.get_config_var("SOABI")
        or manifest.get("torch") != torch.__version__
        or manifest.get("hip") != torch.version.hip
        or manifest.get("dtk") != "26041"
        or manifest.get("cxx11_abi") is not True
    ):
        raise RuntimeError("GLM-5.3 decode skinny build manifest or runtime ABI mismatch")
    with _LOCK:
        existing = sys.modules.get(_MODULE)
        if existing is not None:
            if Path(existing.__file__).resolve() != path:
                raise RuntimeError("GLM-5.3 decode skinny extension already loaded from another path")
            return _check_entries(existing)
        spec = importlib.util.spec_from_file_location(_MODULE, path)
        if spec is None or spec.loader is None:
            raise RuntimeError("Cannot load the GLM-5.3 decode skinny extension")
        module = importlib.util.module_from_spec(spec)
        sys.modules[_MODULE] = module
        try:
            spec.loader.exec_module(module)
            if Path(module.__file__).resolve() != path:
                raise RuntimeError("GLM-5.3 decode skinny extension resolved to a different binary")
            return _check_entries(module)
        except Exception:
            sys.modules.pop(_MODULE, None)
            raise
