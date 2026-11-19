# Copyright (c) 2026 Hygon Information Technology Co., Ltd.
# SPDX-License-Identifier: Apache-2.0

import argparse
import hashlib
import json
import logging
import os
import platform
import shutil
import subprocess
import sys
import sysconfig
import tempfile
from pathlib import Path


_MODULE = "hcu_sglang_glm53_fp8_k1_indirect"
_SOURCES = Path(__file__).resolve().with_name("csrc")
_BINDING = _SOURCES / "glm53_fp8_k1_indirect.cu"
_HEADER = _SOURCES / "glm53_fp8_k1_wasp.cuh"
_ARCH = "gfx938"
_HIP_FLAGS = (
    "-O3",
    "-std=c++20",
    f"--offload-arch={_ARCH}",
    "--gpu-max-threads-per-block=1024",
    "-mcode-object-version=5",
    "-mllvm",
    "-support-768-vgprs=true",
    "-DHIP_ENABLE_WARP_SYNC_BUILTINS",
)
_logger = logging.getLogger(__name__)


def _rocm_home():
    return Path(os.environ.get("ROCM_HOME") or os.environ.get("ROCM_PATH") or "/opt/dtk").resolve()


def _build_metadata(torch, rocm_home):
    if sys.platform != "linux" or platform.machine() != "x86_64" or sys.implementation.name != "cpython":
        raise RuntimeError("The K1 extension requires Linux x86-64 and CPython")
    if torch.version.hip is None:
        raise RuntimeError("The K1 extension requires a HIP build of PyTorch")
    return {
        "python": platform.python_version(),
        "python_abi": sysconfig.get_config_var("SOABI"),
        "torch": torch.__version__,
        "torch_commit": torch.version.git_version,
        "hip": torch.version.hip,
        "dtk_version": (rocm_home / ".info" / "rocm_version").read_text(encoding="utf-8").strip(),
        "rocm_home": str(rocm_home),
        "cxx11_abi": torch.compiled_with_cxx11_abi(),
        "libc": platform.libc_ver(),
        "arch": _ARCH,
        "hip_flags": _HIP_FLAGS,
    }


def cached_extension_path(torch):
    import fcntl

    key = hashlib.sha256(json.dumps(_build_metadata(torch, _rocm_home()), sort_keys=True).encode())
    builder = Path(__file__).resolve()
    for source in (builder, _BINDING, _HEADER):
        key.update(hashlib.sha256(source.read_bytes()).digest())
    cache_home = Path(os.environ.get("XDG_CACHE_HOME") or Path.home() / ".cache").expanduser()
    root = (cache_home / "hcu_sglang" / "glm53-fp8-k1").resolve()
    root.mkdir(mode=0o700, parents=True, exist_ok=True)
    output = root / key.hexdigest()
    binary = output / f"{_MODULE}.so"
    with root.joinpath(output.name + ".lock").open("a") as lock:
        fcntl.flock(lock, fcntl.LOCK_EX)
        if not binary.is_file() or not (output / "build-manifest.json").is_file():
            if output.exists():
                shutil.rmtree(output)
            _logger.info("Building GLM-5.3 FP8 K1 extension in %s", output)
            with tempfile.TemporaryDirectory(prefix="build-", dir=root) as directory:
                staged = Path(directory) / "extension"
                subprocess.run(
                    [sys.executable, str(builder), "--output", str(staged)], check=True
                )
                staged.rename(output)
    return binary


def _build(output):
    rocm_home = _rocm_home()
    compiler = rocm_home / "bin" / "aicc"
    if not compiler.is_file():
        raise RuntimeError(f"AICC not found at {compiler}")
    os.environ.update(
        ROCM_HOME=str(rocm_home),
        PYTORCH_ROCM_ARCH=_ARCH,
        PYTORCH_NVCC=str(compiler),
        CUDA_VISIBLE_DEVICES="",
        HIP_VISIBLE_DEVICES="",
        ROCR_VISIBLE_DEVICES="",
    )
    import torch
    from torch.utils.cpp_extension import load

    metadata = _build_metadata(torch, rocm_home)
    if output.exists():
        raise FileExistsError(output)
    with tempfile.TemporaryDirectory(prefix="glm53-fp8-k1-") as directory:
        build_dir = Path(directory)
        shutil.copyfile(_HEADER, build_dir / _HEADER.name)
        binding = build_dir / _BINDING.name
        shutil.copyfile(_BINDING, binding)
        module = load(
            name=_MODULE,
            sources=[str(binding)],
            build_directory=directory,
            extra_cflags=["-O3"],
            extra_cuda_cflags=list(_HIP_FLAGS),
            with_cuda=True,
        )
        if not callable(module.metadata) or not callable(module.indirect):
            raise RuntimeError("The built extension does not export both K1 entry points")
        output.mkdir(parents=True, exist_ok=False)
        binary = output / f"{_MODULE}.so"
        shutil.copyfile(module.__file__, binary)
        manifest = {
            **metadata,
            "status": "BUILT_AND_IMPORTED_GPU_UNVERIFIED",
            "binary": binary.name,
        }
        (output / "build-manifest.json").write_text(
            json.dumps(manifest, indent=2) + "\n", encoding="utf-8"
        )
    print(binary)


def main():
    parser = argparse.ArgumentParser(description="Build the plugin GLM-5.3 FP8 K1 extension")
    parser.add_argument("--output", required=True, type=Path)
    args = parser.parse_args()
    _build(args.output.resolve())


if __name__ == "__main__":
    main()
