"""HYV4 compressed-tensors W4A8 hooks for slimquant_w4a8_marlin on HCU."""

from __future__ import annotations

from contextvars import ContextVar
import logging
from types import MethodType

import torch

from sglang.srt.plugins.hook_registry import HookRegistry, HookType

from hcu_sglang.models.hyv4_prefill_cp import prepare_cp_inputs
from hcu_sglang.utils.detection import is_hcu

logger = logging.getLogger(__name__)

_MODULE = "sglang.srt.models.hunyuan_v4"
_INIT_TARGET = f"{_MODULE}.HYV4ForCausalLM.__init__"
_LOAD_WEIGHTS_TARGET = f"{_MODULE}.HYV4ForCausalLM.load_weights"
_DENSE_CP_INIT = ContextVar("hyv4_dense_cp_init", default=False)

_SLIMQUANT_W4A8 = "slimquant_w4a8_marlin"
_HY4_CT_W4A8_FORMAT = "hy4_compressed_tensors_w4a8"


def register_hooks() -> None:
    from hcu_sglang.models import hyv4_env_hooks

    hyv4_env_hooks.register_hooks()
    from hcu_sglang.models import hyv4_nextn

    hyv4_nextn.register_hooks()
    for target, hook in (
        (_INIT_TARGET, replacement_init),
        (_LOAD_WEIGHTS_TARGET, replacement_load_weights),
        (f"{_MODULE}.HYV4DecoderLayer.__init__", around_decoder_init),
        ("sglang.srt.models.deepseek_v2.DeepseekV2MLP.__init__", around_mlp_init),
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
    result = original_fn(self, config, quant_config=quant_config, prefix=prefix)
    if _prefill_cp_enabled():
        self.prepare_cp_inputs = MethodType(prepare_cp_inputs, self)
    return result


def _prefill_cp_enabled():
    from sglang.srt.runtime_context import get_parallel

    return is_hcu() and get_parallel().enable_prefill_cp


def around_decoder_init(original_fn, self, config, layer_id, *args, **kwargs):
    if config.mlp_layer_types[layer_id] != "dense" or not _prefill_cp_enabled():
        return original_fn(self, config, layer_id, *args, **kwargs)
    token = _DENSE_CP_INIT.set(True)
    try:
        return original_fn(self, config, layer_id, *args, **kwargs)
    finally:
        _DENSE_CP_INIT.reset(token)


def around_mlp_init(original_fn, self, *args, **kwargs):
    if _DENSE_CP_INIT.get():
        if kwargs.get("parallel_group", "tp") != "tp":
            raise RuntimeError("Unexpected HYV4 CP dense FFN parallel group")
        kwargs["parallel_group"] = "replicated"
    return original_fn(self, *args, **kwargs)


def replacement_load_weights(original_fn, self, weights):
    quant_config = getattr(self, "quant_config", None)
    if getattr(quant_config, "checkpoint_format", None) != _HY4_CT_W4A8_FORMAT:
        return original_fn(self, weights)
    if quant_config.uses_hyv4_deepep():
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
