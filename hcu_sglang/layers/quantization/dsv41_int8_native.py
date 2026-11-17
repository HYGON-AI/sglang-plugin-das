import ctypes
import functools
import hashlib
import json
import os
from pathlib import Path
import shutil
import subprocess
import weakref

import torch


@functools.lru_cache(maxsize=8)
def _architecture(device):
    return getattr(torch.cuda.get_device_properties(device), "gcnArchName", "").split(":", 1)[0]


def supported(x, groups, rank, output_dtype):
    return (
        x.is_cuda and x.ndim == 3 and x.shape[1] == groups
        and x.dtype == output_dtype == torch.bfloat16
        and groups == 8 and rank == 1024 and x.shape[-1] == 4096
        and _architecture(x.device.index) == "gfx936"
    )


@functools.lru_cache(maxsize=1)
def library():
    import fcntl

    compiler = shutil.which("hipcc")
    if compiler is None:
        raise RuntimeError("HCU INT8 projection requires hipcc for its initial build")
    sdk = Path(compiler).parent.parent
    if not (sdk / "include/hipblaslt/hipblaslt.h").is_file():
        sdk = Path(compiler).resolve().parent.parent
    source = Path(__file__).resolve().parents[2] / "jit_overlay/csrc/dsv41_int8_projection.cu"
    header = sdk / "include/hipblaslt/hipblaslt.h"
    blas = (sdk / "lib/libhipblaslt.so").resolve()
    identity = source.read_bytes() + header.read_bytes() + repr([(str(p.resolve()), p.stat().st_size, p.stat().st_mtime_ns) for p in (blas, Path(compiler))]).encode()
    key = hashlib.sha256(identity).hexdigest()[:24]
    folder = Path.home() / ".cache/hcu_sglang/dsv41_int8" / key
    folder.mkdir(parents=True, exist_ok=True)
    binary = folder / "projection.so"
    with (folder / "build.lock").open("a") as lock:
        fcntl.flock(lock.fileno(), fcntl.LOCK_EX)
        if not binary.exists():
            temporary = folder / "projection.pending.so"
            command = [compiler, "-O3", "-std=c++17", "-shared", "-fPIC", "--offload-arch=gfx936", str(source), "-I" + str(sdk / "include"), "-L" + str(sdk / "lib"), "-Wl,-rpath," + str(sdk / "lib"), "-lhipblaslt", "-o", str(temporary)]
            with (folder / "build.stdout").open("wb") as stdout, (folder / "build.stderr").open("wb") as stderr:
                result = subprocess.run(command, stdout=stdout, stderr=stderr, timeout=180)
            if result.returncode:
                raise RuntimeError(f"HCU INT8 projection build failed; see {folder / 'build.stderr'}")
            os.replace(temporary, binary)
            (folder / "build.json").write_text(json.dumps({"command": command, "source_sha256": hashlib.sha256(source.read_bytes()).hexdigest(), "library_sha256": hashlib.sha256(binary.read_bytes()).hexdigest()}))
    lib = ctypes.CDLL(str(binary))
    lib.woa_plan.argtypes = [ctypes.c_int] + [ctypes.c_int64] * 4 + [ctypes.c_void_p, ctypes.c_size_t]
    lib.woa_plan.restype = ctypes.c_void_p
    lib.woa_workspace.argtypes = [ctypes.c_void_p]
    lib.woa_workspace.restype = ctypes.c_uint64
    lib.woa_destroy.argtypes = [ctypes.c_void_p]
    lib.woa_destroy.restype = None
    lib.woa_run.argtypes = [ctypes.c_void_p] * 11 + [ctypes.c_size_t]
    lib.woa_run.restype = ctypes.c_int
    lib._file_path = str(binary)
    return lib


class _Plan:
    def __init__(self, device, m, groups, rank, k):
        if torch.cuda.is_current_stream_capturing():
            raise RuntimeError("Warm up the HCU INT8 projection shape before graph capture")
        self.lib = library()
        error = ctypes.create_string_buffer(1024)
        self.pointer = self.lib.woa_plan(device, m, groups, rank, k, error, len(error))
        if not self.pointer:
            raise RuntimeError(error.value.decode())
        self.workspace = self.lib.woa_workspace(self.pointer)
        self._finalizer = weakref.finalize(self, self.lib.woa_destroy, self.pointer)


@functools.lru_cache(maxsize=128)
def _plan(device, m, groups, rank, k):
    return _Plan(device, m, groups, rank, k)


def project(x, weight, weight_scale, groups, rank):
    if (
        not supported(x, groups, rank, x.dtype)
        or weight.dtype != torch.int8 or weight.shape != (groups * rank, x.shape[-1])
        or weight_scale.dtype != torch.float32 or weight_scale.shape != (groups * rank, 1)
        or not weight.is_contiguous() or not weight_scale.is_contiguous()
        or weight.device != x.device or weight_scale.device != x.device
    ):
        raise ValueError("Unsupported native HCU grouped INT8 projection tensors")
    m, _, k = x.shape
    x = x.contiguous()
    with torch.cuda.device(x.device):
        plan = _plan(x.device.index, m, groups, rank, k)
        q = torch.empty_like(x, dtype=torch.int8)
        scales = torch.empty((m, groups, 1), dtype=torch.float32, device=x.device)
        raw = torch.empty((m, groups, rank), dtype=torch.int32, device=x.device)
        output = torch.empty((m, groups, rank), dtype=x.dtype, device=x.device)
        workspace = torch.empty(plan.workspace, dtype=torch.uint8, device=x.device)
        error = ctypes.create_string_buffer(1024)
        stream = torch.cuda.current_stream(x.device).cuda_stream
        result = plan.lib.woa_run(plan.pointer, x.data_ptr(), q.data_ptr(), scales.data_ptr(), weight.data_ptr(), weight_scale.data_ptr(), raw.data_ptr(), output.data_ptr(), workspace.data_ptr(), stream, error, len(error))
        if result:
            raise RuntimeError(error.value.decode())
    return output
