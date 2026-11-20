"""HYV4 compressed-tensors W4A8 hooks for slimquant_w4a8_marlin on HCU."""

from __future__ import annotations

import logging

import torch

from sglang.srt.plugins.hook_registry import HookRegistry, HookType

logger = logging.getLogger(__name__)

_MODULE = "sglang.srt.models.hunyuan_v4"
_INIT_TARGET = f"{_MODULE}.HYV4ForCausalLM.__init__"
_LOAD_WEIGHTS_TARGET = f"{_MODULE}.HYV4ForCausalLM.load_weights"

_SLIMQUANT_W4A8 = "slimquant_w4a8_marlin"
_HY4_CT_W4A8_FORMAT = "hy4_compressed_tensors_w4a8"


def register_hooks() -> None:
    for target, hook in (
        (_INIT_TARGET, replacement_init),
        (_LOAD_WEIGHTS_TARGET, replacement_load_weights),
    ):
        logger.info("[hcu_sglang] register AROUND hook: %s", target)
        HookRegistry.register(target, hook, HookType.AROUND)


def adapt_hyv4_packed_weight(name, weight, quant_config):
    if getattr(quant_config, "checkpoint_format", None) != _HY4_CT_W4A8_FORMAT:
        return name, weight
    if name.endswith(".weight_shape"):
        return None, None
    if name.endswith(".weight_packed"):
        if ".mlp.experts." not in name or weight.dtype != torch.int32 or weight.ndim != 2:
            raise ValueError(f"Expected a packed INT32 HYV4 expert matrix: {name}")
        return name.removesuffix("_packed"), weight.contiguous().view(torch.int8)
    return name, weight


def replacement_init(original_fn, self, config, quant_config=None, prefix=""):
    if (
        quant_config is not None
        and quant_config.get_name() == _SLIMQUANT_W4A8
        and getattr(quant_config, "checkpoint_format", None) != _HY4_CT_W4A8_FORMAT
    ):
        raise ValueError(
            "HYV4 with slimquant_w4a8_marlin requires a compressed-tensors W4A8 "
            f"checkpoint (checkpoint_format={_HY4_CT_W4A8_FORMAT}), got "
            f"{getattr(quant_config, 'checkpoint_format', None)!r}"
        )
    return original_fn(self, config, quant_config=quant_config, prefix=prefix)


def replacement_load_weights(original_fn, self, weights):
    quant_config = getattr(self, "quant_config", None)
    if getattr(quant_config, "checkpoint_format", None) != _HY4_CT_W4A8_FORMAT:
        return original_fn(self, weights)

    def mapped_weights():
        for name, loaded_weight in weights:
            name, loaded_weight = adapt_hyv4_packed_weight(
                name, loaded_weight, quant_config
            )
            if name is None:
                continue
            yield name, loaded_weight

    return original_fn(self, mapped_weights())
