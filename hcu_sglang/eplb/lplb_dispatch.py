# Copyright (c) 2026 Hygon Information Technology Co., Ltd.
# SPDX-License-Identifier: Apache-2.0

from functools import lru_cache
from pathlib import Path

from hcu_sglang.utils.detection import is_hcu


@lru_cache(maxsize=None)
def _hcu_dispatch_module(max_copies, block_dim):
    from sglang.kernels.jit.utils import load_jit, make_cpp_args

    source = (
        Path(__file__).resolve().parents[1]
        / "jit_overlay/csrc/lplb/dispatch_probability.cuh"
    )
    args = make_cpp_args(max_copies, block_dim)
    return load_jit(
        "hcu_lplb_dispatch_probability",
        *args,
        cuda_files=[str(source)],
        cuda_wrappers=[("dispatch_probability", f"dispatch_probability<{args}>")],
        extra_include_paths=[str(source.parent)],
    )


def around_dispatch_module(original, max_copies, block_dim):
    if not is_hcu():
        return original(max_copies, block_dim)
    return _hcu_dispatch_module(max_copies, block_dim)
