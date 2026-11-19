# Copyright (c) 2026 Hygon Information Technology Co., Ltd.
# SPDX-License-Identifier: Apache-2.0

from sglang.kernels.ops.attention.utils import (
    concat_mla_absorb_q_general as _baseline_concat,
)
from sglang.srt.environ import envs
from hcu_sglang.utils.detection import is_hcu


def concat_mla_absorb_q_general(q_nope, q_rope):
    field = envs.SGLANG_ENABLE_HCU_CONCAT_MLA_ABSORB_Q
    if not is_hcu() or not field.is_set():
        return _baseline_concat(q_nope, q_rope)
    if field.get() and q_nope.shape[-1] == 512 and q_rope.shape[-1] == 64:
        from sgl_kernel import concat_mla_absorb_q

        return concat_mla_absorb_q(q_nope, q_rope)
    import torch

    return torch.cat([q_nope, q_rope], dim=-1)
