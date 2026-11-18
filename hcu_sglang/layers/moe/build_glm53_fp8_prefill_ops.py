# Copyright (c) 2026 Hygon Information Technology Co., Ltd.
# SPDX-License-Identifier: Apache-2.0

import argparse
import hashlib
import json
import os
import shutil
import sys
import sysconfig
import tempfile
from pathlib import Path


_EXPECTED = {
    "activation": "41d63cdbf951c1303192734a3538fe51c142121688091811109afc133b558ac1",
    "gather": "603b6928c7b4671eeab890b3de8158bbb9e0c2be6b174584961b04c52dc114b7",
}
_HEADERS = {
    "reduce.h": "071eb3de2cb5688c5b37c322ad45d09cd2ff374f12d435a326f031ff1187d7aa",
    "reduce_hip.h": "db6e51131423c0f0f2e401b7ff2d991d70f788bbbe39e9ba8c1db07eaee1d5da",
}
_MODULES = {"activation": "activation_inverse_diag", "gather": "epgather_k8_diag"}
_FLAGS = (
    "-O3",
    "-std=c++17",
    "--offload-arch=gfx938",
    "--gpu-max-threads-per-block=512",
    "-mllvm",
    "-enable-num-vgprs-512=true",
)


def _sha256(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _input(path, expected):
    path = path.resolve(strict=True)
    if not path.is_file() or _sha256(path) != expected:
        raise ValueError(f"Unexpected build input: {path}")
    return path


def _build(args):
    provenance_path = Path(__file__).with_name("csrc") / "glm53_fp8_prefill_ops.source.json"
    provenance = json.loads(provenance_path.read_text(encoding="utf-8"))
    sources = {
        "activation": _input(args.activation_source, _EXPECTED["activation"]),
        "gather": _input(args.gather_source, _EXPECTED["gather"]),
    }
    headers = {name: _input(args.header_dir / name, digest) for name, digest in _HEADERS.items()}
    for kind in _EXPECTED:
        if (
            provenance[kind]["source_sha256"] != _EXPECTED[kind]
            or provenance[kind]["module"] != _MODULES[kind]
            or provenance[kind]["headers"] != (_HEADERS if kind == "activation" else {})
        ):
            raise ValueError(f"Unexpected provenance for {kind}")
    if sysconfig.get_config_var("SOABI") != "cpython-310-x86_64-linux-gnu":
        raise RuntimeError("The extensions require Linux x86-64 and CPython 3.10")
    root = Path(os.environ.get("ROCM_HOME") or os.environ.get("ROCM_PATH") or "/opt/dtk").resolve(strict=True)
    compiler = args.compiler.resolve(strict=True)
    if not compiler.is_file():
        raise ValueError(f"Compiler is not a file: {compiler}")
    dtk = (root / ".info" / "rocm_version").read_text(encoding="utf-8").strip()
    if dtk.split("-", 1)[0].replace(".", "").replace(" ", "").replace("V", "") != "26041":
        raise RuntimeError(f"The extensions require DTK 26.04.1, found {dtk!r}")
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
        raise RuntimeError("The extensions require Torch 2.11.0, HIP 6.3.26113 and CXX11 ABI 1")
    output = args.output.resolve()
    output.mkdir(parents=True, exist_ok=False)
    for kind in ("activation", "gather"):
        destination = output / kind
        destination.mkdir()
        try:
            with tempfile.TemporaryDirectory(prefix=f"glm53-fp8-{kind}-") as directory:
                build_dir = Path(directory)
                source = build_dir / provenance[kind]["source"]
                shutil.copyfile(sources[kind], source)
                include = build_dir / "headers"
                include.mkdir()
                for name, header in headers.items():
                    shutil.copyfile(header, include / name)
                flags = list(_FLAGS)
                if kind == "activation":
                    flags.insert(3, "-DTORCH_VERSION_OVER_290")
                module = load(
                    name=_MODULES[kind],
                    sources=[str(source)],
                    build_directory=str(build_dir),
                    extra_cflags=["-O3", "-std=c++17"],
                    extra_cuda_cflags=flags,
                    extra_include_paths=[str(include)],
                    with_cuda=True,
                )
                if not callable(getattr(module, "run", None)):
                    raise RuntimeError(f"The built {kind} extension has no run entry")
                binary = destination / f"{_MODULES[kind]}.so"
                shutil.copyfile(module.__file__, binary)
                manifest = {
                    "status": "BUILT_AND_IMPORTED_GPU_UNVERIFIED",
                    "kind": kind,
                    "module": _MODULES[kind],
                    "source": provenance[kind],
                    "source_manifest_sha256": _sha256(provenance_path),
                    "build_tool_sha256": _sha256(Path(__file__)),
                    "binary": binary.name,
                    "binary_sha256": _sha256(binary),
                    "compiler_sha256": _sha256(compiler),
                    "hip_flags": flags,
                    "python_abi": sysconfig.get_config_var("SOABI"),
                    "torch": torch.__version__,
                    "hip": torch.version.hip,
                    "dtk": "26041",
                    "dtk_version": dtk,
                    "cxx11_abi": torch.compiled_with_cxx11_abi(),
                }
                (destination / "build-manifest.json").write_text(
                    json.dumps(manifest, indent=2) + "\n", encoding="utf-8"
                )
        except Exception as exc:
            (destination / "build-failure.json").write_text(
                json.dumps({"status": "BUILD_FAILED", "kind": kind, "error": str(exc)}, indent=2) + "\n",
                encoding="utf-8",
            )
            raise
        print(binary, flush=True)


def main():
    parser = argparse.ArgumentParser(description="Build the plugin GLM-5.3 FP8 prefill extensions")
    parser.add_argument("--activation-source", required=True, type=Path)
    parser.add_argument("--gather-source", required=True, type=Path)
    parser.add_argument("--header-dir", required=True, type=Path)
    parser.add_argument("--compiler", type=Path, default=Path("/opt/dtk/dcc/bin/dcc"))
    parser.add_argument("--output", required=True, type=Path)
    _build(parser.parse_args())


if __name__ == "__main__":
    main()
