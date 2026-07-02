"""Shared runtime imports for extracted DeepSeek V4 fork methods."""

from __future__ import annotations

import concurrent.futures
import logging
import os
from pathlib import Path
from typing import Iterable, List, Literal, Optional, Set, Tuple, Union

import torch
import torch.nn as nn
import torch.nn.functional as F

import sglang.srt.models.deepseek_v2 as deepseek_v2
from sglang.jit_kernel.deepseek_v4 import (
    fused_norm_rope_inplace,
    fused_q_norm_rope,
    fused_rope,
    fused_rope_inplace,
    rmsnorm_self,
)
from sglang.srt.configs.deepseek_v4 import DeepSeekV4Config
from sglang.srt.distributed import (
    get_pp_group,
    get_tensor_model_parallel_world_size,
    get_tp_group,
)
from sglang.srt.environ import envs
from sglang.srt.eplb.expert_location import ModelConfigForExpertLocation
from sglang.srt.layers.attention.dsv4.compressor import Compressor
from sglang.srt.layers.attention.dsv4.indexer import C4Indexer
from sglang.srt.layers.attention.nsa.utils import (
    can_nsa_cp_split,
    is_nsa_enable_prefill_cp,
    is_nsa_prefill_cp_round_robin_split,
    nsa_use_prefill_cp,
)
from sglang.srt.layers.communicator import get_attn_tp_context
from sglang.srt.layers.dp_attention import (
    _DpGatheredBufferWrapper,
    attn_tp_all_gather,
    dp_gather_partial,
    dp_scatter,
    get_attention_cp_rank,
    get_attention_cp_size,
    get_attention_dp_size,
    get_attention_tp_rank,
    get_attention_tp_size,
    get_global_dp_buffer,
    get_local_dp_buffer,
    is_dp_attention_enabled,
)
from sglang.srt.layers.layernorm import RMSNorm
from sglang.srt.layers.linear import ColumnParallelLinear, RowParallelLinear
from sglang.srt.layers.logits_processor import LogitsProcessor
from sglang.srt.layers.moe import get_moe_a2a_backend
from sglang.srt.layers.moe.fused_moe_triton import FusedMoE
from sglang.srt.layers.quantization.fp8_kernel import (
    per_token_quant_fp8,
    sglang_per_token_group_quant_fp8,
)
from sglang.srt.layers.rotary_embedding import get_rope_wrapper
from sglang.srt.layers.utils import PPMissingLayer, get_layer_id
from sglang.srt.layers.utils.cp_utils import (
    cp_all_gather_rerange_output,
    cp_split_and_rebuild_data,
    cp_split_and_rebuild_position,
    prepare_context_parallel_metadata,
)
from sglang.srt.layers.vocab_parallel_embedding import VocabParallelEmbedding
from sglang.srt.mem_cache.memory_pool import RadixAttention
from sglang.srt.model_executor.cuda_graph_runner import (
    compile_in_capture_mode,
    get_is_capture_mode,
)
from sglang.srt.model_executor.forward_batch_info import ForwardBatch, PPProxyTensors
from sglang.srt.model_loader.utils import maybe_executor_submit, should_async_load
from sglang.srt.model_loader.weight_utils import default_weight_loader
from sglang.srt.models.dbrx import ReplicatedLinear
from sglang.srt.models.deepseek_v2 import ParallelLMHead, _is_cuda, _is_hip, _is_npu
from sglang.srt.models.deepseek_v4 import (
    DeepseekV4DecoderLayer,
    DeepseekV4ForCausalLM,
    DeepseekV4Model,
    MQALayer,
    _dequant_fp8_wo_a,
)
from sglang.srt.server_args import get_global_server_args
from sglang.srt.utils import LazyValue, add_prefix, log_info_on_rank0, make_layers
from sglang.srt.utils.hf_transformers_utils import get_rope_config

from sglang_dcu_plugin.models.deepseek.deepseek_v4 import constants as C

logger = logging.getLogger(__name__)

_FP8_WO_A_GEMM = envs.SGLANG_OPT_FP8_WO_A_GEMM.get()

_is_dcu = C._is_dcu
_use_dpskv4_lightop_rmsnorm = C._use_dpskv4_lightop_rmsnorm
_use_fused_qnorm_rope_kv_rope_quant = C._use_fused_qnorm_rope_kv_rope_quant
_use_aiter_tilelang_mhc = C._use_aiter_tilelang_mhc
op = C.op

try:
    from aiter.ops.tilelang import mhc_post_fwd, mhc_pre_big_fuse
except ImportError:
    mhc_post_fwd = None  # type: ignore[misc, assignment]
    mhc_pre_big_fuse = None  # type: ignore[misc, assignment]

try:
    from sglang.srt.layers.quantization import QuantizationConfig
except ImportError:
    QuantizationConfig = object  # type: ignore[misc, assignment]
