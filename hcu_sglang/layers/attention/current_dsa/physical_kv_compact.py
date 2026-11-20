# Copyright (c) 2026 Hygon Information Technology Co., Ltd.
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

import hashlib
from dataclasses import dataclass
from pathlib import Path

import torch
import triton
import triton.language as tl


def model_matches(model_runner):
    model_config = model_runner.model_config
    config = model_config.hf_config
    quant = getattr(config, "quantization_config", None)
    if not isinstance(quant, dict):
        return False
    groups = quant.get("config_groups")
    if not isinstance(groups, dict) or set(groups) != {"FP8_DYNAMIC"}:
        return False
    group = groups["FP8_DYNAMIC"]
    if not isinstance(group, dict):
        return False
    weights = group.get("weights") or {}
    activation = group.get("input_activations") or {}
    if not isinstance(weights, dict) or not isinstance(activation, dict):
        return False
    if (
        model_runner.is_draft_worker
        or getattr(model_runner.server_args, "enable_lora", False)
        or getattr(model_runner.server_args, "lora_paths", None)
        or tuple(getattr(config, "architectures", None) or ()) != ("GlmMoeDsaForCausalLM",)
        or getattr(config, "model_type", None) != "glm_moe_dsa"
        or getattr(config, "num_hidden_layers", None) != 78
        or getattr(config, "hidden_size", None) != 6144
        or getattr(config, "n_routed_experts", None) != 256
        or getattr(config, "num_experts_per_tok", None) != 8
        or getattr(config, "moe_intermediate_size", None) != 2048
        or getattr(config, "first_k_dense_replace", None) != 3
        or getattr(config, "num_attention_heads", None) != 64
        or getattr(config, "kv_lora_rank", None) != 512
        or getattr(config, "index_kpool", 1) != 1
        or getattr(config, "index_topk", None) != 2048
        or getattr(model_config, "quantization", None) not in ("compressed-tensors", "compressed_tensors")
        or quant.get("quant_method") != "compressed-tensors"
        or quant.get("format") != "float-quantized"
        or group.get("targets") != ["Linear"]
        or weights.get("type") != "float"
        or weights.get("num_bits") != 8
        or weights.get("strategy") != "channel"
        or weights.get("dynamic") is not False
        or weights.get("block_structure") is not None
        or activation.get("type") != "float"
        or activation.get("num_bits") != 8
        or activation.get("strategy") != "token"
        or activation.get("dynamic") is not True
    ):
        return False
    try:
        config_path = Path(model_config.model_path) / "config.json"
        return hashlib.sha256(config_path.read_bytes()).hexdigest() == (
            "bc21bd68d92085465810c3da72c4b8a43b16c65c2c8f55f52547045f36c1abbf"
        )
    except (AttributeError, OSError, TypeError, ValueError):
        return False


@dataclass(frozen=True)
class PhysicalKVPlan:
    logical_rows: torch.Tensor
    physical_rows: torch.Tensor
    inverse: torch.Tensor

    @property
    def has_duplicates(self):
        return self.physical_rows.numel() < self.logical_rows.numel()


def build_physical_kv_plan(logical_rows):
    assert logical_rows.is_cuda and logical_rows.ndim == 1
    assert logical_rows.is_contiguous()
    assert logical_rows.dtype in (torch.int32, torch.int64)
    assert 0 < logical_rows.numel() <= torch.iinfo(torch.int32).max
    physical_rows, inverse = torch.unique(logical_rows, sorted=True, return_inverse=True)
    return PhysicalKVPlan(logical_rows, physical_rows, inverse.to(torch.int32))


@triton.jit
def _remap_physical_kv_indices(indices, inverse, output, count, logical_count, physical_count, BLOCK: tl.constexpr):
    x = tl.program_id(0) * BLOCK + tl.arange(0, BLOCK)
    i = tl.load(indices + x, x < count, other=-1)
    valid = (i >= 0) & (i < logical_count) & (x < count)
    j = tl.load(inverse + i, valid, other=0)
    mapped = tl.where(valid, j, tl.where(i < 0, i, physical_count))
    tl.store(output + x, mapped, x < count)


def remap_physical_kv_indices(indices, plan):
    assert indices.is_cuda and indices.is_contiguous() and indices.dtype == torch.int32
    assert indices.device == plan.inverse.device
    result = torch.empty_like(indices)
    if indices.numel():
        _remap_physical_kv_indices[(triton.cdiv(indices.numel(), 512),)](
            indices,
            plan.inverse,
            result,
            indices.numel(),
            plan.logical_rows.numel(),
            plan.physical_rows.numel(),
            BLOCK=512,
        )
    return result
