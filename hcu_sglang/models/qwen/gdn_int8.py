# Copyright 2025 Qwen Team
# Copyright 2025 SGLang Team
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ==============================================================================

import logging
from functools import lru_cache
from typing import NamedTuple

import torch
import triton
from sglang.srt.layers import linear
from sglang.srt.layers.quantization.compressed_tensors.schemes.compressed_tensors_w8a8_int8 import (
    CompressedTensorsW8A8Int8,
)
from sglang.srt.utils import get_bool_env_var

from hcu_sglang.utils.detection import is_hcu

logger = logging.getLogger(__name__)
enabled = get_bool_env_var("SGLANG_HCU_QWEN_GDN_RMS_INT8_FUSION") and is_hcu()


class GdnGatedInt8Input(NamedTuple):
    activation: torch.Tensor
    scale: torch.Tensor
    dtype: torch.dtype


def _supports_fusion(layer):
    scheme = getattr(layer.out_proj, "scheme", None)
    return (
        type(scheme) is CompressedTensorsW8A8Int8
        and not scheme.is_static_input_scheme
        and scheme.input_symmetric
        and layer.out_proj.input_is_parallel
        and layer.head_v_dim == 128
        and layer.norm.norm_before_gate
        and layer.norm.group_size is None
        and layer.norm.activation in ("swish", "silu")
        and layer.norm.bias is None
    )


@lru_cache(maxsize=1)
def _kernel():
    try:
        from lightop.norm import gated_rms_norm_per_token_int8_quant
    except ImportError:
        from hcu_sglang.layers.attention.fla.gated_norm_int8 import (
            gated_rmsnorm_per_token_int8,
        )

        def gated_rms_norm_per_token_int8_quant(x, z, weight, eps, *, num_heads):
            return gated_rmsnorm_per_token_int8(x, z, weight, eps, heads=num_heads)
    return gated_rms_norm_per_token_int8_quant


@lru_cache(maxsize=None)
def _log_dispatch(mode):
    logger.info("Qwen GDN gated RMSNorm + dynamic INT8 quant fused before out_proj (%s)", mode)


def _fused_output(layer, x, z, heads, forward_batch):
    weight = layer.norm.weight
    if (
        not x.shape[0]
        or not x.is_cuda
        or x.dtype != torch.bfloat16
        or z.dtype != x.dtype
        or weight.dtype != x.dtype
        or z.device != x.device
        or weight.device != x.device
        or x.stride(-1) != 1
        or z.stride(-1) != 1
        or not weight.is_contiguous()
        or not 1 <= heads <= 64
        or x.shape[0] % heads
        or (z.ndim == 3 and z.size(1) != heads)
    ):
        return None
    q, scale = _kernel()(x, z, weight, layer.norm.eps, num_heads=heads)
    _log_dispatch(forward_batch.forward_mode.name)
    return layer.out_proj(GdnGatedInt8Input(q, scale, x.dtype))[0]


def around_forward(original, self, hidden_states, forward_batch):
    if torch.compiler.is_compiling() or not _supports_fusion(self) or linear.get_forward().sp_active:
        return original(self, hidden_states, forward_batch)

    from sglang.srt.models import qwen3_5 as N

    projected_states_qkvz, projected_states_ba = self._forward_input_proj(
        hidden_states
    )

    if N._is_xpu and N.get_exec().mamba.linear_attn_backend == "intel_xpu":
        from sglang.srt.model_executor.forward_context import get_attn_backend

        backend = get_attn_backend()
        backend = getattr(backend, "linear_attn_backend", backend)
        if backend.supports_fused_gdn(self.attn, forward_batch):
            return self._forward_xpu(
                backend, projected_states_qkvz, projected_states_ba, forward_batch
            )

    use_strided_prefill_z = False
    packed_verify = False
    if forward_batch.forward_mode.is_target_verify():
        from sglang.srt.model_executor.forward_context import get_attn_backend

        backend = get_attn_backend()
        backend = getattr(backend, "linear_attn_backend", backend)
        packed_verify = getattr(backend, "supports_packed_verify", False)
    use_fused_proj_conv = (
        N._gdn_decode_fused_proj_conv
        and (forward_batch.forward_mode.is_decode() or packed_verify)
        and isinstance(projected_states_qkvz, torch.Tensor)
        and isinstance(projected_states_ba, torch.Tensor)
    )
    use_fused_contiguous_unpack = (
        self.num_v_heads // self.num_k_heads in N._GDN_FUSED_QKVZBA_RATIOS
    )
    if use_fused_proj_conv:
        # GDN owns indexed Conv1D state and the safe unpack/Conv boundary;
        # it replaces these temporary B/A placeholders before recurrence.
        mixed_qkv = (projected_states_qkvz, projected_states_ba)
        z = None
        b = projected_states_ba
        a = projected_states_ba
    elif use_fused_contiguous_unpack:
        if N._is_cpu:
            num_k_heads_tp = self.num_k_heads // self.attn_tp_size
            num_v_heads_tp = self.num_v_heads // self.attn_tp_size
        else:
            num_k_heads_tp = triton.cdiv(self.num_k_heads, self.attn_tp_size)
            num_v_heads_tp = triton.cdiv(self.num_v_heads, self.attn_tp_size)
        use_strided_prefill_z = (
            N._is_cuda and forward_batch.forward_mode.is_extend_without_speculative()
        )
        split_fn = (
            N.qwen3_5_gdn_prefill_projection_views
            if use_strided_prefill_z
            else N.fused_qkvzba_split_reshape_cat_contiguous
        )
        mixed_qkv, z, b, a = split_fn(
            projected_states_qkvz,
            projected_states_ba,
            num_k_heads_tp,
            num_v_heads_tp,
            self.head_k_dim,
            self.head_v_dim,
        )
    else:
        query, key, value, z, b, a = self.fix_query_key_value_ordering(
            projected_states_qkvz, projected_states_ba
        )
        b = b.contiguous()
        a = a.contiguous()

        query, key, value = map(
            lambda x: x.reshape(x.shape[0], -1), (query, key, value)
        )
        mixed_qkv = torch.cat((query, key, value), dim=-1)

    attn_result = self.attn(
        forward_batch,
        mixed_qkv=mixed_qkv,
        a=a,
        b=b,
    )
    if use_fused_proj_conv:
        if not isinstance(attn_result, tuple) or len(attn_result) != 2:
            raise RuntimeError(
                "Fused GDN projection/Conv1D backend must return (core_attn_out, z)"
            )
        core_attn_out, z = attn_result
    else:
        core_attn_out = attn_result
    assert z is not None

    z_shape_og = z.shape
    # reshape input data into 2D tensor
    core_attn_out = core_attn_out.reshape(-1, core_attn_out.shape[-1])
    if use_strided_prefill_z:
        z_flat_shape = (z.numel() // z.shape[-1], z.shape[-1])
    else:
        z = z.reshape(-1, z.shape[-1])
        z_flat_shape = z.shape

    # Add padding for DP-Attn
    if core_attn_out.shape != z_flat_shape:
        core_attn_out_pad = z.new_zeros(z_flat_shape)
        core_attn_out_pad[: core_attn_out.shape[0], :] = core_attn_out
        core_attn_out = core_attn_out_pad

    nv = self.num_v_heads // self.attn_tp_size
    if not use_strided_prefill_z and N._fp8_tuple_input(self.out_proj, len(z) // nv):
        fp8, _, _ = N._layer_norm_fwd(
            core_attn_out,
            self.norm.weight,
            None,
            self.norm.eps,
            z=z,
            is_rms_norm=True,
            activation=self.norm.activation,
            quant_heads=nv,
        )
        return self.out_proj(fp8)[0]

    fused = _fused_output(self, core_attn_out, z, nv, forward_batch)
    if fused is not None:
        return fused

    core_attn_out = self.norm(core_attn_out, z)
    core_attn_out = core_attn_out.reshape(z_shape_og)
    core_attn_out = core_attn_out.reshape(
        *core_attn_out.shape[:-2],
        core_attn_out.shape[-2] * core_attn_out.shape[-1],
    )

    output, _ = self.out_proj(core_attn_out)
    return output
