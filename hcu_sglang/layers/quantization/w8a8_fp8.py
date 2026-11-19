"""HCU W8A8 FP8 quantization adaptations."""

from __future__ import annotations


def retain_checkpoint_ignores(result, cls, config):
    if config.get("quant_method") == "compressed-tensors":
        result._hcu_checkpoint_ignore = tuple(config.get("ignore") or ())
        packed = config.get("packed_modules_mapping") or {}
        result._hcu_checkpoint_packed = {
            "in_proj_ba": ["in_proj_b", "in_proj_a"],
            **packed,
        }


def respect_checkpoint_ignores(original, config, layer, prefix):
    from hcu_sglang.utils.detection import is_hcu

    ignore = getattr(config, "_hcu_checkpoint_ignore", ())
    if not ignore or not is_hcu():
        return original(config, layer, prefix)

    from sglang.srt.layers.linear import LinearBase
    from sglang.srt.layers.quantization.compressed_tensors.utils import should_ignore_layer
    from sglang.srt.layers.quantization.unquant import UnquantizedLinearMethod

    if isinstance(layer, LinearBase) and should_ignore_layer(
        prefix, ignore, config._hcu_checkpoint_packed
    ):
        return UnquantizedLinearMethod()
    return original(config, layer, prefix)


def register_hooks() -> None:
    from sglang.srt.plugins.hook_registry import HookRegistry, HookType

    target = "sglang.srt.layers.quantization.w8a8_fp8.W8A8Fp8Config"
    HookRegistry.register(
        f"{target}.from_config", retain_checkpoint_ignores, HookType.AFTER
    )
    HookRegistry.register(
        f"{target}.get_quant_method", respect_checkpoint_ignores, HookType.AROUND
    )
    from . import fp8_linear, fp8_quant, fp8_block, fp8_mxscale, compressed_fp8, compressed_int8
    fp8_quant.register_hooks()
    fp8_linear.register_hooks()
    fp8_block.register_hooks()
    fp8_mxscale.register_hooks()
    compressed_fp8.register_hooks()
    compressed_int8.register_hooks()
