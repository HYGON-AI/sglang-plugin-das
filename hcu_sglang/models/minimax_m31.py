import os

from hcu_sglang.layers.quantization.registry import register_hooks

ENTRY_POINT = "hcu_minimax_m31"


def register() -> None:
    selected = {name.strip() for name in os.getenv("SGLANG_PLUGINS", "").split(",")}
    if ENTRY_POINT not in selected or "hcu" in selected:
        return
    if os.getenv("HCU_PLUGIN_MINIMAX_M31_QUANT", "1") == "1":
        from sglang.srt.arg_groups.choices import (
            QUANTIZATION_CHOICES,
            add_quantization_method_choices,
        )

        from hcu_sglang.layers.quantization.registry import SLIMQUANT_MARLIN_NAME

        if SLIMQUANT_MARLIN_NAME not in QUANTIZATION_CHOICES:
            add_quantization_method_choices([SLIMQUANT_MARLIN_NAME])
        register_hooks()

    from hcu_sglang.models import minimax_m31_dspark

    minimax_m31_dspark.register_hooks()

    if os.getenv("HCU_PLUGIN_MINIMAX_M31_FLASHMLA", "0") == "1":
        from hcu_sglang.layers.attention import minimax_m31_sparse

        minimax_m31_sparse.register_hooks()
