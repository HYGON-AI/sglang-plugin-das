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


_SOURCE_SHA256 = "27ae5b80c407ccdb21640948f505185ef6c558529e3c3938aa305bea2e2fbdfc"
_REFERENCE_SHA256 = "e7870a15db717c3761e80725e238b8709ee1adb4ff92239121b7e4b8211f0361"
_MODULE = "token_inverse_diag"
_FLAGS = (
    "-O3",
    "-std=c++17",
    "--offload-arch=gfx938",
    "-DTORCH_VERSION_OVER_290",
    "--gpu-max-threads-per-block=1024",
    "-mllvm",
    "-enable-num-vgprs-512=true",
)


def _sha256(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _build(args):
    provenance_path = Path(__file__).with_name("csrc") / "glm53_fp8_token_inverse.source.json"
    provenance = json.loads(provenance_path.read_text(encoding="utf-8"))
    if (
        provenance.get("schema") != 1
        or provenance.get("module") != _MODULE
        or provenance.get("source") != "token_inverse.cu"
        or provenance.get("source_sha256") != _SOURCE_SHA256
        or provenance.get("reference_binary_sha256") != _REFERENCE_SHA256
        or provenance.get("headers") != {}
        or provenance.get("hip_flags") != list(_FLAGS)
    ):
        raise ValueError("Unexpected GLM-5.3 FP8 token inverse provenance")
    source = args.source.resolve(strict=True)
    if not source.is_file() or _sha256(source) != _SOURCE_SHA256:
        raise ValueError(f"Unexpected token inverse source: {source}")
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
        with tempfile.TemporaryDirectory(prefix="glm53-fp8-token-inverse-") as directory:
            build_dir = Path(directory)
            copied_source = build_dir / provenance["source"]
            shutil.copyfile(source, copied_source)
            if _sha256(copied_source) != _SOURCE_SHA256:
                raise ValueError("Copied token inverse source hash mismatch")
            module = load(
                name=_MODULE,
                sources=[str(copied_source)],
                build_directory=str(build_dir),
                extra_cflags=["-O3", "-std=c++17"],
                extra_cuda_cflags=list(_FLAGS),
                with_cuda=True,
            )
            if not callable(getattr(module, "run", None)):
                raise RuntimeError("The built token inverse extension has no run entry")
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
    parser = argparse.ArgumentParser(description="Build the plugin GLM-5.3 FP8 token inverse extension")
    parser.add_argument("--source", required=True, type=Path)
    parser.add_argument("--compiler", type=Path, default=Path("/opt/dtk/dcc/bin/dcc"))
    parser.add_argument("--output", required=True, type=Path)
    _build(parser.parse_args())


if __name__ == "__main__":
    main()
