import argparse
import hashlib
import json
import os
import shutil
import subprocess
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


def _sha256(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _extract_kernel(source, provenance):
    commit = provenance["commit"]
    git = ["git", "-C", str(source)]
    head = subprocess.check_output([*git, "rev-parse", "HEAD"], text=True).strip()
    if head != commit:
        raise ValueError(f"DeepGEMM checkout must be at {commit}, found {head}")
    content = subprocess.check_output([*git, "show", f"{commit}:{provenance['header']}"])
    if hashlib.sha256(content).hexdigest() != provenance["header_sha256"]:
        raise ValueError("DeepGEMM WASP header SHA256 mismatch")

    kernel, delimiter, _ = content.decode().partition("\ntorch::Tensor run(\n")
    if not delimiter:
        raise ValueError("DeepGEMM WASP generic host entry was not found")
    for old, new in (
        (
            "namespace m_grouped_fp8_gemm_nt_contiguous_wasp_pair32_bm256_bn256_nloop4{",
            "namespace hcu_sglang_glm53_fp8_k1_wasp {",
        ),
        (
            "indirect_offsets[i] = src < 0 ? 0x80000000u",
            "indirect_offsets[i] = (src < 0 || src >= params.recv_rows) ? 0x80000000u",
        ),
        (
            "ga = scale_src < 0 ? 0x80000000u",
            "ga = (scale_src < 0 || scale_src >= params.recv_rows) ? 0x80000000u",
        ),
        (
            "    if (params.m_indices[blockIdx.y * GemmTraitsPC::kBlockM] < 0) {",
            "    const int expert = params.m_indices[blockIdx.y * GemmTraitsPC::kBlockM];\n"
            "    if (expert < 0 || expert >= params.experts) {",
        ),
    ):
        if kernel.count(old) != 1:
            raise ValueError(f"Unexpected DeepGEMM WASP source at {old!r}")
        kernel = kernel.replace(old, new, 1)
    return kernel + "\n}\n"


def _build(source, output):
    provenance_path = _SOURCES / "glm53_fp8_k1_indirect.source.json"
    provenance = json.loads(provenance_path.read_text(encoding="utf-8"))
    kernel = _extract_kernel(source, provenance)
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
        header.write_text(kernel, encoding="utf-8")
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
            "source": provenance,
            "source_manifest_sha256": _sha256(provenance_path),
            "build_tool_sha256": _sha256(Path(__file__)),
            "generated_header_sha256": _sha256(header),
            "binding_sha256": _sha256(binding),
            "binary": binary.name,
            "binary_sha256": _sha256(binary),
            "compiler_sha256": _sha256(compiler),
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
    parser.add_argument("--deepgemm-source", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    args = parser.parse_args()
    _build(args.deepgemm_source.resolve(strict=True), args.output.resolve())


if __name__ == "__main__":
    main()
