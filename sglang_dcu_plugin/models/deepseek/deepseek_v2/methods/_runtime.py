"""Shared runtime imports for extracted DeepSeek V2 fork methods."""

from __future__ import annotations

import logging
from contextlib import nullcontext
from typing import Any, Dict, Iterable, List, Optional, Tuple, Union

import torch
import torch.nn.functional as F
from torch import nn
from transformers import PretrainedConfig

from sglang.jit_kernel.deepseek_v4 import (
    silu_and_mul_clamp,
    silu_and_mul_contig_post_quant,
)
from sglang.srt.batch_overlap.single_batch_overlap import SboFlags, compute_overlap_args
from sglang.srt.batch_overlap.two_batch_overlap import (
    MaybeTboDeepEPDispatcher,
    model_forward_maybe_tbo,
)
from sglang.srt.compilation.piecewise_context_manager import is_in_piecewise_cuda_graph
from sglang.srt.configs.model_config import (
    compute_mla_mscale_scaling,
    get_nsa_index_head_dim,
    get_nsa_index_n_heads,
    get_nsa_index_topk,
    is_deepseek_nsa,
)
from sglang.srt.distributed import (
    divide,
    get_moe_expert_parallel_world_size,
    get_pp_group,
    get_tensor_model_parallel_world_size,
    parallel_state,
    tensor_model_parallel_all_reduce,
)
from sglang.srt.distributed.device_communicators.pynccl_allocator import (
    use_symmetric_memory,
)
from sglang.srt.environ import envs
from sglang.srt.eplb.expert_distribution import get_global_expert_distribution_recorder
from sglang.srt.eplb.expert_location import ModelConfigForExpertLocation
from sglang.srt.eplb.expert_location_dispatch import ExpertLocationDispatchInfo
from sglang.srt.layers import deep_gemm_wrapper
from sglang.srt.layers.activation import SiluAndMul
from sglang.srt.layers.amx_utils import PackWeightMethod
from sglang.srt.layers.attention.lightop_concat import concat_decode_opt
from sglang.srt.layers.attention.nsa.nsa_indexer import Indexer
from sglang.srt.layers.attention.nsa.utils import (
    can_nsa_cp_split,
    is_nsa_enable_prefill_cp,
    nsa_use_prefill_cp,
)
from sglang.srt.layers.attention.tbo_backend import TboAttnBackend
from sglang.srt.layers.communicator import (
    LayerCommunicator,
    LayerScatterModes,
    enable_moe_dense_fully_dp,
    get_attn_tp_context,
)
from sglang.srt.layers.communicator_nsa_cp import NSACPLayerCommunicator
from sglang.srt.layers.dp_attention import (
    get_attention_cp_rank,
    get_attention_cp_size,
    get_attention_tp_group,
    get_attention_tp_rank,
    get_attention_tp_size,
    is_dp_attention_enabled,
)
from sglang.srt.layers.layernorm import RMSNorm
from sglang.srt.layers.linear import (
    ColumnParallelLinear,
    MergedColumnParallelLinear,
    ReplicatedLinear,
    RowParallelLinear,
)
from sglang.srt.layers.logits_processor import LogitsProcessor
from sglang.srt.layers.moe import (
    get_moe_a2a_backend,
    get_moe_runner_backend,
    should_skip_post_experts_all_reduce,
    should_use_flashinfer_cutlass_moe_fp4_allgather,
)
from sglang.srt.layers.moe.ep_moe.layer import get_moe_impl_class
from sglang.srt.layers.moe.fused_moe_triton.layer import FusedMoE
from sglang.srt.layers.moe.hash_topk import HashTopK
from sglang.srt.layers.moe.kt_ep_wrapper import KTEPWrapperMethod
from sglang.srt.layers.moe.token_dispatcher.base import (
    BaseDispatcher,
    CombineInput,
    DispatchOutput,
)
from sglang.srt.layers.moe.topk import TopK, TopKOutputFormat
from sglang.srt.layers.moe.utils import (
    RoutingMethodType,
    filter_moe_weight_param_global_expert,
    is_deepep_class_backend,
    is_sbo_enabled,
    is_tbo_enabled,
)
from sglang.srt.layers.quantization.base_config import QuantizationConfig
from sglang.srt.layers.quantization.fp8 import Fp8Config
from sglang.srt.layers.quantization.fp8_kernel import (
    create_per_token_group_quant_fp8_output_scale,
    per_tensor_quant_mla_fp8,
    per_token_group_quant_mla_deep_gemm_masked_fp8,
)
from sglang.srt.layers.quantization.mxfp4_flashinfer_trtllm_moe import (
    maybe_fuse_routed_scale_and_shared_add,
)
from sglang.srt.layers.radix_attention import RadixAttention
from sglang.srt.layers.rotary_embedding import get_rope_wrapper
from sglang.srt.layers.utils import PPMissingLayer
from sglang.srt.layers.utils.cp_utils import (
    cp_all_gather_rerange_output,
    cp_split_and_rebuild_data,
    cp_split_and_rebuild_position,
    prepare_context_parallel_metadata,
)
from sglang.srt.layers.vocab_parallel_embedding import (
    ParallelLMHead,
    VocabParallelEmbedding,
)
from sglang.srt.model_executor.cuda_graph_runner import get_is_capture_mode
from sglang.srt.model_executor.forward_batch_info import ForwardBatch, PPProxyTensors
from sglang.srt.models.deepseek_common.attention_backend_handler import (
    AttentionBackendRegistry,
)
from sglang.srt.models.deepseek_common.attention_forward_methods import (
    AttnForwardMethod,
    DeepseekMHAForwardMixin,
    DeepseekMLACpuForwardMixin,
    DeepseekMLAForwardMixin,
    DeepseekMLARocmForwardMixin,
)
from sglang.srt.models.deepseek_common.deepseek_weight_loader import (
    DeepseekV2WeightLoaderMixin,
)
from sglang.srt.models.deepseek_common.utils import (
    _get_llama_4_scaling,
    _is_cublas_ge_129,
    _is_musa,
    _is_xpu,
)
from sglang.srt.models.deepseek_v2 import DeepseekV2MLP, DeepseekV2MoE
from sglang.srt.server_args import get_global_server_args
from sglang.srt.speculative.spec_info import SpeculativeAlgorithm
from sglang.srt.state_capturer.indexer_topk import maybe_capture_indexer_topk
from sglang.srt.utils import (
    BumpAllocator,
    LazyValue,
    add_prefix,
    is_non_idle_and_non_empty,
    log_info_on_rank0,
    make_layers,
    use_intel_amx_backend,
)

from sglang_dcu_plugin.models.deepseek.deepseek_v2 import constants as C

logger = logging.getLogger(__name__)

# Re-export fork flags for ``from _runtime import *`` in method modules.
_is_dcu = C._is_dcu
_is_hip = C._is_hip
_is_cuda = C._is_cuda
_is_npu = C._is_npu
_is_cpu = C._is_cpu
_is_cpu_amx_available = C._is_cpu_amx_available
_device_sm = C._device_sm
_is_gfx95_supported = C._is_gfx95_supported
_is_sbo_enabled = C._is_sbo_enabled
_use_aiter = C._use_aiter
_use_aiter_gfx95 = C._use_aiter_gfx95
_use_lightop_moe_sum_mul_add = C._use_lightop_moe_sum_mul_add
_use_fused_silu_mul_quant = C._use_fused_silu_mul_quant
_use_opt_cat_decode = C._use_opt_cat_decode
_use_fused_mla_cat = C._use_fused_mla_cat
_use_fused_rmsnorm_rope = C._use_fused_rmsnorm_rope
_use_fused_rms_quant = C._use_fused_rms_quant
_rms_quant_path = C._rms_quant_path
FORWARD_ABSORB_CORE_ATTENTION_BACKENDS = C.FORWARD_ABSORB_CORE_ATTENTION_BACKENDS
dtype_mapping = C.dtype_mapping
fused_rms_norm_rope_contiguous = C.fused_rms_norm_rope_contiguous

if C._use_aiter:
    from sglang.srt.layers.rocm_linear_utils import aiter_dsv3_router_gemm

if C._use_aiter_gfx95:
    from sglang.srt.layers.rocm_linear_utils import (
        get_dsv3_gemm_output_zero_allocator_size,
    )

if C._is_cuda:
    from flashinfer.gemm import mm_M1_16_K7168_N256 as _raw_dsv3_router_gemm
    from sgl_kernel import dsv3_fused_a_gemm, dsv3_router_gemm
elif C._is_hip:
    from sglang.srt.layers.attention.triton_ops.rocm_mla_decode_rope import (
        decode_attention_fwd_grouped_rope,
    )
    from sglang.srt.layers.quantization.awq.awq_triton import (
        awq_dequantize_triton as awq_dequantize,
    )
    from sgl_kernel import merge_state_v2
elif C._is_npu:
    from sglang.srt.hardware_backend.npu.modules.deepseek_v2_attention_mla_npu import (
        forward_dsa_core_npu,
        forward_dsa_prepare_npu,
        forward_mha_core_npu,
        forward_mha_prepare_npu,
        forward_mla_core_npu,
        forward_mla_prepare_npu,
    )
elif _is_musa:
    from sgl_kernel import dsv3_fused_a_gemm, dsv3_router_gemm

# Helpers defined in upstream forward_mla (used by inlined attention methods).
_bmm_fp8 = getattr(DeepseekMLAForwardMixin, "bmm_fp8", None)
if _bmm_fp8 is not None:
    bmm_fp8 = _bmm_fp8
else:
    from sglang.srt.models.deepseek_common.attention_forward_methods import (
        forward_mla as _forward_mla_mod,
    )

    bmm_fp8 = getattr(_forward_mla_mod, "bmm_fp8", None)

try:
    from sglang.srt.models.deepseek_common.attention_forward_methods.forward_mla import (
        fused_rms_fp8_group_quant,
        fused_rms_mxfp4_quant,
    )
except ImportError:
    fused_rms_fp8_group_quant = None  # type: ignore[misc, assignment]
    fused_rms_mxfp4_quant = None  # type: ignore[misc, assignment]

try:
    from sglang.srt.layers.rocm_linear_utils import batched_gemm_afp4wfp4_pre_quant
except ImportError:
    batched_gemm_afp4wfp4_pre_quant = None  # type: ignore[misc, assignment]

try:
    from sglang.srt.layers.attention.triton_ops.nsa_dequant_k_cache import (
        dequantize_k_cache_paged,
    )
except ImportError:
    try:
        from sglang.srt.layers.attention.nsa.dequant_k_cache import (
            dequantize_k_cache_paged,
        )
    except ImportError:
        dequantize_k_cache_paged = None  # type: ignore[misc, assignment]
