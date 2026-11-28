import ctypes
from functools import cache

from sglang.srt.distributed.device_communicators.cuda_wrapper import find_loaded_library

from hcu_sglang.utils.detection import is_hcu


@cache
def _host_pointer_runtime():
    path = find_loaded_library("libgalaxyhip") or find_loaded_library("libamdhip64")
    if path is None:
        raise RuntimeError("HIP runtime is not loaded in the current process")
    lib = ctypes.CDLL(path)
    lib.hipHostGetDevicePointer.restype = ctypes.c_int
    lib.hipHostGetDevicePointer.argtypes = [
        ctypes.POINTER(ctypes.c_void_p),
        ctypes.c_void_p,
        ctypes.c_uint,
    ]
    return lib


def around_registered_device_ptr(original_fn, host_ptr):
    if not is_hcu():
        return original_fn(host_ptr)
    device_ptr = ctypes.c_void_p()
    err = _host_pointer_runtime().hipHostGetDevicePointer(
        ctypes.byref(device_ptr), ctypes.c_void_p(host_ptr), 0
    )
    if err != 0 or not device_ptr.value:
        raise RuntimeError(f"hipHostGetDevicePointer failed: {err}")
    return device_ptr.value


def register_hooks():
    from sglang.srt.plugins.hook_registry import HookRegistry, HookType

    HookRegistry.register(
        "sglang.srt.layers.engram._registered_device_ptr",
        around_registered_device_ptr,
        HookType.AROUND,
    )

    if is_hcu():
        from .engram_hash import around_gate, around_hash_ids, around_torch_hash_ids
        from .engram_host import HcuEngramEmbedding

        HookRegistry.register(
            "sglang.srt.layers.engram.EngramEmbedding", HcuEngramEmbedding, HookType.REPLACE
        )
        HookRegistry.register(
            "sglang.srt.layers.engram.engram_hash_ids",
            around_hash_ids,
            HookType.AROUND,
        )
        HookRegistry.register(
            "sglang.srt.layers.engram.EngramHasher._torch_hash_ids",
            around_torch_hash_ids,
            HookType.AROUND,
        )
        HookRegistry.register(
            "sglang.srt.layers.engram.engram_gate",
            around_gate,
            HookType.AROUND,
        )
