"""Shared runtime imports for extracted deepseek_nextn fork methods."""

from __future__ import annotations

import logging
import os
from typing import Iterable, Optional, Tuple

import torch
from safetensors.torch import load_file
from torch import nn
from transformers import PretrainedConfig

from sglang.srt.distributed import get_pp_group, get_tensor_model_parallel_world_size
from sglang.srt.environ import envs
from sglang.srt.eplb.expert_distribution import get_global_expert_distribution_recorder
from sglang.srt.layers.attention.nsa.utils import (
    can_nsa_cp_split,
    is_nsa_enable_prefill_cp,
    nsa_use_prefill_cp,
)
from sglang.srt.layers.dp_attention import (
    get_attention_cp_rank,
    get_attention_cp_size,
    is_dp_attention_enabled,
)
from sglang.srt.layers.layernorm import RMSNorm
from sglang.srt.layers.linear import ReplicatedLinear
from sglang.srt.layers.moe.utils import is_sbo_enabled
from sglang.srt.layers.quantization import Fp8Config
from sglang.srt.layers.quantization.base_config import QuantizationConfig
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
from sglang.srt.model_executor.forward_batch_info import ForwardBatch
from sglang.srt.models.deepseek_common.utils import enable_nextn_moe_bf16_cast_to_fp8
from sglang.srt.models.deepseek_v2 import DeepseekV2DecoderLayer, DeepseekV3ForCausalLM
from sglang.srt.models.utils import WeightsMapper
from sglang.srt.server_args import get_global_server_args
from sglang.srt.utils import BumpAllocator, add_prefix, is_cuda, is_npu

logger = logging.getLogger(__name__)

_is_cuda = is_cuda()
_is_npu = is_npu()
