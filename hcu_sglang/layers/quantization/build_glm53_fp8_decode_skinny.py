# Copyright (c) 2026 Hygon Information Technology Co., Ltd.
# SPDX-License-Identifier: Apache-2.0

import argparse
import hashlib
import json
import os
import shutil
import sysconfig
import tempfile
from pathlib import Path


_SOURCE_SHA256 = "6b3f733cb8025fbfaa7871aa609164b19866b99561f37fb6aa0d5ad472dc760e"
_REFERENCE_SHA256 = "8aa3f2e39bf93dfca597714d80511f355cb6b600a5b09d0c0fdd86c83fac5e34"
_MODULE = "hcu_sglang_glm53_fp8_decode_skinny"
_FLAGS = ("-O3", "--offload-arch=gfx938")
_BINDING = """#include <torch/extension.h>
#include <cstdint>
void skinny_gemm(const torch::Tensor&, const torch::Tensor&, const torch::Tensor&, const torch::Tensor&, torch::Tensor&, int64_t, int64_t, bool, int64_t);
void bmm_skinny(const torch::Tensor&, const torch::Tensor&, torch::Tensor&, int64_t, int64_t, bool);
PYBIND11_MODULE(TORCH_EXTENSION_NAME, module) {
  module.def("skinny_gemm", &skinny_gemm);
  module.def("bmm_skinny", &bmm_skinny);
}
"""


def _sha256(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _build(args):
    provenance_path = Path(__file__).with_name("csrc") / "glm53_fp8_decode_skinny.source.json"
    provenance = json.loads(provenance_path.read_text(encoding="utf-8"))
    if (
        provenance.get("schema") != 1
        or provenance.get("module") != _MODULE
        or provenance.get("source_file") != "csrc/skinny_gemm.cu"
        or provenance.get("source_commit") != "e4f43b5daa1c19eb8c39597c1404000d4938a89e"
        or provenance.get("source_sha256") != _SOURCE_SHA256
        or provenance.get("reference_binary_sha256") != _REFERENCE_SHA256
        or provenance.get("hip_flags") != list(_FLAGS)
    ):
        raise ValueError("Unexpected GLM-5.3 decode skinny provenance")
    source = args.source.resolve(strict=True)
    if not source.is_file() or _sha256(source) != _SOURCE_SHA256:
        raise ValueError(f"Unexpected GLM-5.3 decode skinny source: {source}")
    if sysconfig.get_config_var("SOABI") != "cpython-310-x86_64-linux-gnu":
        raise RuntimeError("The extension requires Linux x86-64 and CPython 3.10")
    root = Path(os.environ.get("ROCM_HOME") or os.environ.get("ROCM_PATH") or "/opt/dtk").resolve(strict=True)
    compiler = args.compiler.resolve(strict=True)
    if not compiler.is_file():
        raise ValueError(f"Compiler is not a file: {compiler}")
    dtk = (root / ".info" / "rocm_version").read_text(encoding="utf-8").strip()
    if dtk.split("-", 1)[0].replace(".", "").replace(" ", "").replace("V", "") != "26041":
        raise RuntimeError(f"The extension requires DTK 26.04.1, found {dtk!r}")
    os.environ.update(
        ROCM_HOME=str(root),
        PYTORCH_ROCM_ARCH="gfx938",
        PYTORCH_NVCC=str(compiler),
        CUDA_VISIBLE_DEVICES="",
        HIP_VISIBLE_DEVICES="",
        ROCR_VISIBLE_DEVICES="",
        MAX_JOBS="1",
    )
    import torch
    from torch.utils.cpp_extension import load

    if torch.__version__ != "2.11.0" or torch.version.hip != "6.3.26113" or not torch.compiled_with_cxx11_abi():
        raise RuntimeError("The extension requires Torch 2.11.0, HIP 6.3.26113 and CXX11 ABI 1")
    if torch.cuda.is_initialized():
        raise RuntimeError("The extension builder requires an uninitialized GPU runtime")
    output = args.output.resolve()
    output.mkdir(parents=True, exist_ok=False)
    try:
        with tempfile.TemporaryDirectory(prefix="glm53-fp8-decode-skinny-") as directory:
            build_dir = Path(directory)
            copied_source = build_dir / "skinny_gemm.cu"
            shutil.copyfile(source, copied_source)
            if _sha256(copied_source) != _SOURCE_SHA256:
                raise ValueError("Copied decode skinny source hash mismatch")
            binding = build_dir / "binding.cpp"
            binding.write_text(_BINDING, encoding="utf-8")
            module = load(
                name=_MODULE,
                sources=[str(copied_source), str(binding)],
                build_directory=str(build_dir),
                extra_cflags=["-O3"],
                extra_cuda_cflags=list(_FLAGS),
                with_cuda=True,
            )
            if not all(callable(getattr(module, name, None)) for name in ("skinny_gemm", "bmm_skinny")):
                raise RuntimeError("The built decode skinny extension has missing entries")
            if torch.cuda.is_initialized():
                raise RuntimeError("The extension build unexpectedly initialized the GPU runtime")
            binary = output / f"{_MODULE}.so"
            shutil.copyfile(module.__file__, binary)
            manifest = {
                "status": "BUILT_AND_IMPORTED_GPU_UNVERIFIED",
                "module": _MODULE,
                "source": provenance,
                "source_manifest_sha256": _sha256(provenance_path),
                "build_tool_sha256": _sha256(Path(__file__)),
                "binding_sha256": hashlib.sha256(_BINDING.encode("utf-8")).hexdigest(),
                "binary": binary.name,
                "binary_sha256": _sha256(binary),
                "compiler_sha256": _sha256(compiler),
                "hip_flags": list(_FLAGS),
                "python_abi": sysconfig.get_config_var("SOABI"),
                "torch": torch.__version__,
                "hip": torch.version.hip,
                "dtk": "26041",
                "dtk_version": dtk,
                "cxx11_abi": torch.compiled_with_cxx11_abi(),
                "gpu_initialized_after_import": False,
            }
            (output / "build-manifest.json").write_text(
                json.dumps(manifest, indent=2) + "\n", encoding="utf-8"
            )
    except Exception as exc:
        (output / "build-failure.json").write_text(
            json.dumps({"status": "BUILD_FAILED", "module": _MODULE, "error": str(exc)}, indent=2) + "\n",
            encoding="utf-8",
        )
        raise
    print(binary, flush=True)


def main():
    parser = argparse.ArgumentParser(description="Build the private GLM-5.3 FP8 decode skinny extension")
    parser.add_argument("--source", required=True, type=Path)
    parser.add_argument("--compiler", type=Path, default=Path("/opt/dtk/dcc/bin/dcc"))
    parser.add_argument("--output", required=True, type=Path)
    _build(parser.parse_args())


if __name__ == "__main__":
    main()
