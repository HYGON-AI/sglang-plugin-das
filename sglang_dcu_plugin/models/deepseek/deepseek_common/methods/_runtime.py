"""Shared runtime imports for extracted deepseek_common fork methods."""

from __future__ import annotations

import concurrent.futures
import logging
from typing import Dict, Iterable, List, Optional, Tuple

import torch
import torch.nn as nn
import tqdm
from transformers import PretrainedConfig

from sglang.srt.distributed.parallel_state import GroupCoordinator
from sglang.srt.environ import envs
from sglang.srt.layers import deep_gemm_wrapper
from sglang.srt.layers.moe.fused_moe_triton.layer import FusedMoE
from sglang.srt.layers.quantization.base_config import QuantizationConfig
from sglang.srt.layers.quantization.fp8_utils import (
    block_quant_dequant,
    block_quant_to_tensor_quant,
    channel_quant_to_tensor_quant,
    inverse_transform_scale_ue8m0,
    normalize_e4m3fn_to_e4m3fnuz,
    quant_weight_ue8m0,
)
from sglang.srt.layers.quantization.int8_utils import (
    block_dequant as int8_block_dequant,
)
from sglang.srt.layers.utils import get_layer_id
from sglang.srt.model_loader.utils import (
    maybe_executor_submit,
    should_async_load,
    should_deepgemm_weight_requant_ue8m0,
)
from sglang.srt.model_loader.weight_utils import (
    RUNAI_STREAMER_TENSOR_ATTR,
    default_weight_loader,
)
from sglang.srt.models.deepseek_common.deepseek_weight_loader import (
    _clone_if_runai_streamed_tensor,
)
from sglang.srt.models.deepseek_common.utils import (
    _is_cuda,
    _is_fp8_fnuz,
    _is_hip,
    _is_musa,
    _is_npu,
    _is_xpu,
    _use_aiter_gfx95,
    awq_dequantize_func,
    enable_nextn_moe_bf16_cast_to_fp8,
)
from sglang.srt.utils import bind_or_assign, get_bool_env_var, log_info_on_rank0

if _use_aiter_gfx95:
    from sglang.srt.layers.quantization.quark.utils import quark_post_load_weights

logger = logging.getLogger(__name__)

NVFP4_CKPT_FP8_ATTN_QUANT_MODULES = ["q_b_proj"]
