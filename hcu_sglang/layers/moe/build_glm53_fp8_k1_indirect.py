# Copyright (c) 2026 Hygon Information Technology Co., Ltd.
# SPDX-License-Identifier: Apache-2.0

import argparse
import json
import os
import shutil
import sys
import sysconfig
import tempfile
from pathlib import Path


_MODULE = "hcu_sglang_glm53_fp8_k1_indirect"
_SOURCES = Path(__file__).resolve().with_name("csrc")
_HIP_FLAGS = (
    "-O3",
    "-std=c++20",
    "--offload-arch=gfx938",
    "--gpu-max-threads-per-block=1024",
    "-mcode-object-version=5",
    "-mllvm",
    "-support-768-vgprs=true",
    "-DHIP_ENABLE_WARP_SYNC_BUILTINS",
)


def _build(output):
    if sys.version_info[:2] != (3, 10) or sys.platform != "linux" or os.uname().machine != "x86_64":
        raise RuntimeError("The extension requires Linux x86-64 and CPython 3.10")
    rocm_home = Path(os.environ.get("ROCM_HOME") or os.environ.get("ROCM_PATH") or "/opt/dtk")
    compiler = rocm_home / "bin" / "aicc"
    if not compiler.is_file():
        raise RuntimeError(f"AICC not found at {compiler}")
    dtk = (rocm_home / ".info" / "rocm_version").read_text(encoding="utf-8").strip()
    dtk_abi = dtk.split("-", 1)[0].replace(".", "").replace(" ", "").replace("V", "")
    if dtk_abi != "26041":
        raise RuntimeError(f"The extension requires DTK 26.04.1, found {dtk!r}")
    os.environ.update(
        ROCM_HOME=str(rocm_home),
        PYTORCH_ROCM_ARCH="gfx938",
        PYTORCH_NVCC=str(compiler),
        CUDA_VISIBLE_DEVICES="",
        HIP_VISIBLE_DEVICES="",
        ROCR_VISIBLE_DEVICES="",
    )
    import torch
    from torch.utils.cpp_extension import load

    if torch.__version__ != "2.11.0" or torch.version.hip != "6.3.26113" or not torch.compiled_with_cxx11_abi():
        raise RuntimeError("The extension requires Torch 2.11.0, HIP 6.3.26113 and CXX11 ABI 1")

    if output.exists():
        raise FileExistsError(output)
    with tempfile.TemporaryDirectory(prefix="glm53-fp8-k1-") as directory:
        build_dir = Path(directory)
        header = build_dir / "glm53_fp8_k1_wasp.cuh"
        shutil.copyfile(_SOURCES / header.name, header)
        binding = build_dir / "glm53_fp8_k1_indirect.cu"
        shutil.copyfile(_SOURCES / binding.name, binding)
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
            "status": "BUILT_AND_IMPORTED_GPU_UNVERIFIED",
            "binary": binary.name,
            "hip_flags": _HIP_FLAGS,
            "python_abi": sysconfig.get_config_var("SOABI"),
            "torch": torch.__version__,
            "hip": torch.version.hip,
            "dtk": dtk_abi,
            "dtk_version": dtk,
            "cxx11_abi": torch.compiled_with_cxx11_abi(),
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
